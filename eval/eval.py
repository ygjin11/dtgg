'''external package'''
import logging
import torch
import cv2
from tool.utils import sample
logger = logging.getLogger(__name__)
import atari_py
from collections import deque
import random
import yaml
import os
from rich.console import Console
console = Console()
import torch.cuda as cuda

current_file = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file)
parent_directory = os.path.dirname(current_directory)
config_path = parent_directory + '/config/config_main/main.yaml'
with open(config_path, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
os.environ["CUDA_VISIBLE_DEVICES"] = config['eval']['device']

#@ our package
from model.dt import GPTConfig, GPT
from model.dt_condition import GPTConfig_condition, GPT_condition 
from offline.trainer import Trainer, TrainerConfig
from model.load_guide import load_language
from model.load_guide import load_action
from model.load_guide import load_traj
from model.load_guide import load_traj_5

#! env
class Env():

    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

#! parameters of env
class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 5000
        self.game = game
        self.history_length = 4

#! whole evaluation process 
def interact(device, ret, game, seed, model, max_timestep, instruction_type, action_dim, condition=None):
    T_rewards = []
    done = True
    #@ evaluate 10 times each seed
    eval_num = 10
    for i in range(eval_num):
        if i % 2 == 0:
            seed = seed + 100
        args=Args(game.lower(), seed)
        env = Env(args)
        env.eval()
    
        state = env.reset()
        state = state.type(torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        rtgs = [ret]

        ## laod instruction
        if condition:
            pass

        
        ## first state is from env, first rtg is target return, and first timestep is 0
        if instruction_type == 'raw':
            sampled_action = sample(model, state, 1, temperature=1.0, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device),
                instruction=None)
        else:
            sampled_action = sample(model, state, 1, temperature=1.0, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device),
                instruction=instruction) 
            
        ## if beyond action dim, limit in the scope of action dim
        if sampled_action[0][0].item() >= action_dim:
            sampled_action[0][0] = 0
            #! sampled_action[0][0] = random.randint(0, action_dim-1)

        j = 0
        all_states = state
        actions = []
        while True:
            if done:
                state, reward_sum, done = env.reset(), 0, False
            action = sampled_action.cpu().numpy()[0,-1]
            actions += [sampled_action]
            state, reward, done = env.step(action)
            reward_sum += reward
            j += 1

            if done:
                T_rewards.append(reward_sum)
                break

            state = state.unsqueeze(0).unsqueeze(0).to(device)
            all_states = torch.cat([all_states, state], dim=0)
            rtgs += [rtgs[-1] - reward]
            ## all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
            ## timestep is just current timestep
            if instruction_type == 'raw':
                sampled_action = sample(model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)))
            else:
                sampled_action = sample(model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)),
                    instruction = instruction)
                
            ## if beyond action dim, limit in the scope of action dim
            if sampled_action[0][0].item() >= action_dim:
                sampled_action[0][0] = 0
                #! sampled_action[0][0] = random.randint(0, action_dim-1)
    
    env.close()
    eval_return = sum(T_rewards)/eval_num 
    console.print("target return: %d, eval return: %d" % (ret, eval_return))
    return eval_return

if __name__ == '__main__': 
    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    parent_directory = os.path.dirname(current_directory)
    config_path = parent_directory + '/config/config_main/main.yaml'
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    context_length = config['eval']['context_length'] 
    model_type = config['eval']['model_type']
    game_list = config['game_list']
    game_path = parent_directory + '/config/config_game/' + game_list + '.yaml'
    ckpt_eval = config['eval']['ckpt_eval']
    action_space = config['action_space']
    eval_rtg = config['eval']['eval_rtg']
    condition_dim = config['eval']['condition_dim']
    instruct_dir = config['instruct_dir']
    with open(game_path, 'r') as yaml_file:
        config_game = yaml.safe_load(yaml_file)
    games = config_game['eval']

    condition_type = config['eval']['condition_type']
    enable_retrieval = config['eval']['enable_retrieval']
    enable_instruct = config['eval']['enable_instruct']
    enable_language = config['eval']['enable_language']
    enable_trajectory = config['eval']['enable_trajectory']
    enable_action = config['eval']['enable_action']

    seed = config['seed'] 
    max_timestep = config['eval']['max_timestep']

    console.print(f'seed: {seed}')

    if condition_type == 'raw':
        mconf = GPTConfig(action_space, context_length*3,\
                n_layer=6, n_head=8, n_embd=128,\
                model_type=model_type, max_timestep=max_timestep)
        model = GPT(mconf)
        
    else:
        console.print(f'condition - language: {enable_language}')
        console.print(f'condition - trajectory: {enable_trajectory}')
        console.print(f'condition - action: {enable_action}')
        console.print(f'condition - instruct: {enable_instruct}')
        console.print(f'condition - retrieval: {enable_retrieval}')
        
        mconf = GPTConfig_condition(action_space, action_space,
        n_layer=6, n_head=8, n_embd=128, model_type=model_type, max_timestep=max_timestep,\
        condition_dim=condition_dim, condition_type=condition_type,\
        enable_retrieval=enable_retrieval, enable_language=enable_language,\
        enable_trajectory=enable_trajectory, enable_action=enable_action, \
        enable_instruct=enable_instruct) 
        model = GPT_condition(mconf)

    ## laod ckpt
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    console.log('load ckpt')
    model.train(False)
    state_dict = torch.load(ckpt_eval)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    ## interact
    for item in games:
        game = item['name']
        action_dim = item['action_dim']
        sum_retrun = 0
        console.log(f'eval {item} {condition_type}')

        if condition_type != 'raw':
            pass

        interact(device ,eval_rtg, game, seed, model, max_timestep, condition_type, action_dim, condition)
 
