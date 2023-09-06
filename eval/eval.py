import logging
import torch
from tool.utils import sample
logger = logging.getLogger(__name__)
import atari_py
from collections import deque
import random
import cv2
import torch
import yaml
import os
from tool.clip_extract_lang import get_language_clip
from model.dt import GPT, GPTConfig
from model.dt_condition import GPTConfig_condition, GPT_condition 
import pdb

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

class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4

def interact(ret, env, model, max_timestep, instruction_type, game):
    T_rewards, T_Qs = [], []
    done = True
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # load instruction
    game_embed = get_language_clip(game).to(device)

    # load trajectory
    # TODO

    # load instruction
    # TODO

    num = 5
    for i in range(num):
        state = env.reset()
        state = state.type(torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        rtgs = [ret]
        # first state is from env, first rtg is target return, and first timestep is 0
        if instruction_type == 'raw':
            sampled_action = sample(model, state, 1, temperature=1.0, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device),
                instruction=None)
        else:
            # laod instruction
            if instruction_type == 'language':
                instruction = game_embed
            
            sampled_action = sample(model, state, 1, temperature=1.0, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device),
                instruction=instruction)  
                     
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
            # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
            # timestep is just current timestep
            if instruction_type == 'raw':
                sampled_action = sample(model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)))
            else:
                # laod instruction
                if instruction_type == 'language':
                    instruction = game_embed

                sampled_action = sample(model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)),
                    instruction = instruction)

    env.close()
    eval_return = sum(T_rewards)/num
    print("target return: %d, eval return: %d" % (ret, eval_return))
    return eval_return

if __name__ == '__main__': 
    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    parent_directory = os.path.dirname(current_directory)
    config_path = parent_directory + '/config/config_para/para.yaml'
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    seed = config['seed'] 
    context_length = config['train']['context_length'] 
    model_type = config['train']['model_type']
    instruction_type = config['instruction_type']
    game_list = config['game_list']
    game_path = parent_directory + '/config/config_game/' + game_list + '.yaml'
    ckpt_eval = config['eval']['ckpt_eval']
    max_timestep = config['max_timestep']
    action_space = config['action_space']
    eval_rtg = config['eval']['eval_rtg']
    condition_dim = config['condition_dim']

    with open(game_path, 'r') as yaml_file:
        config_game = yaml.safe_load(yaml_file)
    games = config_game['eval']

    if instruction_type == 'raw':
        # model config
        mconf = GPTConfig(action_space, context_length*3,
                          n_layer=6, n_head=8, n_embd=128, model_type=model_type, max_timestep=max_timestep)
        model = GPT(mconf)
         
    elif instruction_type == 'language':
        mconf = GPTConfig_condition(action_space, context_length*3,
                n_layer=6, n_head=8, n_embd=128, model_type=model_type, max_timestep=max_timestep, embed_dim=condition_dim)
        model = GPT_condition(mconf)

    # laod ckpt
    print('@@@@@@@@@@@@@@@@@@@@@@@ load ckpt @@@@@@@@@@@@@@@@@@@@@@@')
    model.train(False)
    state_dict = torch.load(ckpt_eval)
    model.load_state_dict(state_dict)

    # interact
    print('@@@@@@@@@@@@@@@@@@@@@@@ eval @@@@@@@@@@@@@@@@@@@@@@@')
    for game in games:
        num = 20
        sum_retrun = 0
        for i in range(num):
            print(f'@@@@@@@@@@@@@@@@@@@@@@@ eval {i} @@@@@@@@@@@@@@@@@@@@@@@')
            seed = seed + random.randint(0, 100)
            args=Args(game.lower(), seed)
            env = Env(args)
            env.eval()
            eval_return = interact(eval_rtg, env, model, max_timestep, instruction_type, game)
            sum_retrun += eval_return 
            average_return = sum_retrun /num
        print(f"average return of {game}: {average_return}")  
