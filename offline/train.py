
#@ our package
from model.dt import GPTConfig, GPT
from model.dt_condition import GPTConfig_condition, GPT_condition 
from offline.trainer import Trainer, TrainerConfig
from tool.utils import set_seed
from create_dataset import create_dataset
from model.load_guide import load_language
from model.load_guide import load_action
from model.load_guide import load_traj
from model.load_guide import load_traj_5

#@ external package
import csv
import logging
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,  random_split
import os
import yaml
from collections import deque
import os
import yaml
from PIL import Image
import clip
from rich.console import Console
console = Console()
from multiprocessing import Pool
import sys
from tqdm import tqdm,trange

#@ our dataset(Dateset)
class StateActionReturnDataset(Dataset):
    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps,\
                game, all_games, condition_type, enable_retrieval, enable_language,\
                enable_trajectory, enable_action, enable_instruct):        
        ## data part
        self.block_size = block_size
        self.vocab_size = 18
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.game = game
        ## condition part
        self.all_games = all_games
        self.condition_type = condition_type
        self.enable_retrieval = enable_retrieval
        self.enable_language = enable_language
        self.enable_trajectory = enable_trajectory
        self.enable_action = enable_action
        self.enable_instruct = enable_instruct
        ## device only cpu?
        self.device = 'cpu'
        ## load clip model
        self.clip_model, self.clip_process = clip.load("ViT-B/32", device=self.device)
        
        if self.condition_type != 'raw':       
            if self.enable_language: 
                console.log("load condition - language.")
                self.game_dict_language = dict()
                for game in all_games:  
                    self.game_dict_language[game] = load_language(game, self.clip_model,\
                                                                   self.device).reshape(-1)  
            if self.enable_trajectory:
                if self.enable_retrieval:
                    console.log("load condition - 5 trajectories.")
                    self.game_dict_trajectory = dict()
                    for game in all_games:
                        self.game_dict_trajectory[game] = load_traj_5\
                            (game, self.clip_model, self.clip_process, self.device)
                else:
                    console.log("load condition - 1 trajectory.")
                    self.game_dict_trajectory = dict()
                    for game in all_games:
                        self.game_dict_trajectory[game] = load_traj\
                            (game, self.clip_model, self.clip_process, self.device)
            
            if self.enable_action:
                console.log("load condition - action.")
                self.game_dict_act_num = dict()
                self.game_dict_act = dict()
                for game in all_games:
                    self.game_dict_act_num[game], self.game_dict_act[game] = load_action\
                        (game, self.clip_model, self.device)
                    
            if self.enable_instruct:
                if self.enable_retrieval:
                    console.log("load condition - 5 instructs.")
                else:
                    console.log("load condition - 1 instruct.")
            
            if self.enable_retrieval:   
                def process_states_in_batches(data, clip_model, clip_process, batch_size, batch_epoch):
                    num_samples = len(data)
                    processed_states = None

                    for i in trange(batch_epoch):
                        actual_batch_size = min(batch_size, num_samples - i*batch_size)
                        # Process the batch of states
                        states_down = np.array(data[i:i + actual_batch_size], dtype=np.float32).transpose(0, 2, 3, 1)[:, :, :, 0]
                        preprocessed_states = torch.cat([clip_process(Image.fromarray(np.uint8(state))).unsqueeze(0) for state in states_down], dim=0)
                        # Encode the batch of preprocessed states
                        with torch.no_grad():
                            states_embed = clip_model.encode_image(preprocessed_states)
                        if processed_states is None:
                            processed_states = states_embed
                        else:
                            processed_states = torch.cat((processed_states, states_embed), dim=0)
                    return processed_states
    
                batch_size = 20000
                batch_epoch = len(self.data) //batch_size + (len(self.data) % batch_size != 0)
                print(f"process state using CLIP(batch_size: {batch_size}, batch_epoch: {batch_epoch})")
                self.processed_states = process_states_in_batches(self.data, self.clip_model, self.clip_process, batch_size, batch_epoch)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: ## first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), \
                            dtype=torch.float32).reshape(block_size, -1) 
                             # (block_size, 4*84*84)
        states = states / 255.
        # (block_size, 1)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) 
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)
         
        if self.condition_type == 'raw':
            return states, actions, rtgs, timesteps
        else:
            game = self.game[idx:done_idx][0]
            ## default
            language = torch.tensor([0])
            trajectory = torch.tensor([0])
            act_num = torch.tensor([0])
            act = torch.tensor([0])  
            instruct = torch.tensor([0])
            s_embed = torch.tensor([0])

            if self.enable_language:
                language = self.game_dict_language[game]
            if self.enable_trajectory:
                trajectory = self.game_dict_trajectory[game]
            if self.enable_action:
                act_num = self.game_dict_act_num[game]
                act = self.game_dict_act[game]
            if self.enable_instruct:
                pass
            if self.enable_retrieval:
                 s_embed = self.processed_states[idx:done_idx]

            return states, actions, rtgs, timesteps, language, trajectory, \
                   act_num, act, instruct, s_embed


if __name__ == '__main__': 
    #@ load condif
    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    parent_directory = os.path.dirname(current_directory)
    config_path = parent_directory + '/config/config_main/main.yaml'
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    context_length = config['train']['context_length'] 
    epochs = config['train']['epochs']
    model_type = config['train']['model_type'] 
    num_steps = config['train']['num_steps']
    num_buffers = config['train']['num_buffers']
    batch_size = config['train']['batch_size']
    trajectories_per_buffer = config['train']['trajectories_per_buffer']
    ckpt_path = config['train']['ckpt_path']

    data_dir_prefix = config['dataset_dir']
    instruct_dir = config['instruct_dir'] 
    condition_type = config['train']['condition_type']
    game_list = config['train']['game_list']
    game_path = parent_directory + '/config/config_game/' + game_list + '.yaml'
    condition_dim = config['train']['condition_dim']
    enable_retrieval = config['train']['enable_retrieval']
    enable_instruct = config['train']['enable_instruct']
    enable_language = config['train']['enable_language']
    enable_trajectory = config['train']['enable_trajectory']
    enable_action = config['train']['enable_action']

    with open(game_path, 'r') as yaml_file:
        config_game = yaml.safe_load(yaml_file)
    all_games = config_game['train']
    
    seed = config['seed'] 
    set_seed(seed)

    console.print(f'seed: {seed}')
    console.print(f"condtion_type: {condition_type}")
    #@ load datasdet
    if condition_type != 'raw':
        console.print(f'condition - language: {enable_language}')
        console.print(f'condition - trajectory: {enable_trajectory}')
        console.print(f'condition - action: {enable_action}')
        console.print(f'condition - instruct: {enable_instruct}')
        console.print(f'condition - retrieval: {enable_retrieval}')
    
    # obss, actions, returns, done_idxs, rtgs, timesteps, game\
    #       = create_dataset(num_buffers, num_steps, all_games,\
    #                        data_dir_prefix, trajectories_per_buffer)
    console.log(f'load data...')
    obss, actions, returns, done_idxs, rtgs, timesteps, game = [], [], [], [], [], [], []
    #@ mp to load dataset
    num_game_batch = config['train']['num_game_batch']

    def load_game_data(game):
        return create_dataset(num_buffers, num_steps, [game], data_dir_prefix, trajectories_per_buffer)
    
    list_game = []
    for i in range(int(len(all_games) / num_game_batch) + 1):
        if i*num_game_batch < len(all_games):
            list_game.append(all_games[i*num_game_batch:min(i*num_game_batch+num_game_batch,len(all_games))])

    console.print(f"game batch: {list_game}")
    # all games is seperate into ...
    for i in range(len(list_game)):
        all_games_batch = list_game[i]
        console.print(f'load game batch {all_games_batch}')
        pool = Pool(processes=len(all_games_batch))
        game_data = pool.map(load_game_data, all_games_batch)
        pool.close()
        pool.join()
        for data in game_data:
            obss.extend(data[0])
            actions.extend(data[1])
            returns.extend(data[2])
            done_idxs.extend(data[3])
            rtgs.extend(data[4])
            timesteps.extend(data[5])
            game.extend(data[6])
        
        console.print(f'len of dataset: {len(obss)}')

 
    console.log(f'generat dataset...')
    dataset = StateActionReturnDataset(obss, context_length*3, actions, done_idxs, rtgs, timesteps,\
                                      game, all_games, condition_type, enable_retrieval, enable_language,\
                                      enable_trajectory, enable_action, enable_instruct)  
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    console.log(f'train-initialize...')  
    ## DT
    if condition_type == 'raw':
        mconf = GPTConfig(dataset.vocab_size, dataset.block_size,\
                        n_layer=6, n_head=8, n_embd=128,\
                        model_type=model_type, max_timestep=max(timesteps))
        model = GPT(mconf)

    ## task conditioned DT
    else:
        mconf = GPTConfig_condition(dataset.vocab_size, dataset.block_size,
                n_layer=6, n_head=8, n_embd=128, model_type=model_type, max_timestep=max(timesteps),\
                condition_dim=condition_dim, condition_type=condition_type,\
                enable_retrieval=enable_retrieval, enable_language=enable_language,\
                enable_trajectory=enable_trajectory, enable_action=enable_action, \
                enable_instruct=enable_instruct) 
        model = GPT_condition(mconf) 

    tconf = TrainerConfig(max_epochs=epochs, batch_size=batch_size,\
                          learning_rate=6e-4, lr_decay=True, warmup_tokens=512*20,\
                          final_tokens=2*len(train_dataset)*context_length*3, num_workers=4,\
                          seed=seed, model_type=model_type, game=game,\
                          max_timestep=max(timesteps), ckpt_path=ckpt_path)
    trainer = Trainer(model, train_dataset, test_dataset, tconf, game_list,\
                      condition_type=condition_type, enable_retrieval=enable_retrieval,\
                      enable_language=enable_language, enable_trajectory=enable_trajectory,\
                      enable_action=enable_action, enable_instruct=enable_instruct) 

    console.log(f'train...') 
    trainer.train()


