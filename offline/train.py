
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
                # console.log(f"process state using clip - preprocess.")
                # states_down = np.array(self.data[0:len(self.data)]).\
                #     transpose(0, 2, 3, 1)[:, :, :, 0] # n 224 224
                # preprocessed_states = torch.cat([self.clip_process(Image.fromarray(np.uint8(states_down[i]))).unsqueeze(0)\
                #                          for i in range(len(self.data))], dim=0) # n 3 224 224
                # print(preprocessed_states.shape)

                # console.log(f"process state using clip - clip extract.")
                # states_embed = self.clip_model.encode_image(preprocessed_states) # n 512
                # del preprocessed_states
                # # print(states_embed.shape)

                def process_states_in_batches(data, clip_model, clip_process, batch_size):
                    num_samples = len(data)
                    processed_states_list = []

                    for i in range(0, batch_size):
                        print(i)
                        if i + batch_size > num_samples:
                            batch_size = num_samples - i
                        batch_data = data[i:i + batch_size]
                        print(f"Processing batch {i // batch_size + 1}")

                        # Process the batch of states
                        states_down = np.array(batch_data).transpose(0, 2, 3, 1)[:, :, :, 0]  # n 224 224
                        preprocessed_states = torch.cat([clip_process(Image.fromarray(np.uint8(states_down[j]))).unsqueeze(0) for j in range(len(batch_data))], dim=0)  # n 3 224 224

                        # Encode the batch of preprocessed states
                        states_embed = clip_model.encode_image(preprocessed_states)  # n 512
                        processed_states_list.append(states_embed)

                    processed_states = torch.cat(processed_states_list, dim=0)
                    return processed_states
                
                console.log(f"process state using clips.")
                batch_size = len(self.data) // 1000 + 1
                processed_states = process_states_in_batches(self.data, self.clip_model, self.clip_process, batch_size)
                print(processed_states.shape)

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
                states_2 = np.array(self.data[idx:done_idx]).transpose(0, 2, 3, 1)[:, :, :, 0]  
                s_embed = torch.cat([self.clip_process(Image.fromarray(np.uint8(states_2[i]))).unsqueeze(0)\
                                         for i in range(block_size)], dim=0)
                del states_2
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
    console.log(f'loading data...')
    obss, actions, returns, done_idxs, rtgs, timesteps, game = [], [], [], [], [], [], []
    #@ mp to load dataset
    num_batch = config['train']['num_batch']
    num_game_batch = config['train']['num_game_batch']

    def load_game_data(game):
        return create_dataset(num_buffers, num_steps/num_batch, [game], data_dir_prefix, trajectories_per_buffer)
    
    list_game = []
    for i in range(num_game_batch):
        list_game.append(all_games[i*num_game_batch:i*num_game_batch+num_game_batch])

    index = 0
    # all games is seperate into ...
    for i in range(num_batch*num_game_batch):
        all_games_batch = list_game[index]
        pool = Pool(processes=len(all_games_batch))

        index += 1
        if index == num_game_batch:
            index = 0

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
        if len(obss) > num_steps * len(all_games):
            break
 
    console.log(f'generating dataset...')
    dataset = StateActionReturnDataset(obss, context_length*3, actions, done_idxs, rtgs, timesteps,\
                                      game, all_games, condition_type, enable_retrieval, enable_language,\
                                      enable_trajectory, enable_action, enable_instruct)  
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    console.log(f'training-initialize...')  
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

    console.log(f'training...') 
    trainer.train()


