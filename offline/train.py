'''external package'''
import csv
import logging
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,  random_split
from collections import deque
import os
import yaml
from rich.console import Console
console = Console()
'''our package'''
from model.dt import GPTConfig, GPT
from model.dt_condition import GPTConfig_condition, GPT_condition 
from offline.trainer import Trainer, TrainerConfig
from tool.utils import set_seed
from create_dataset import create_dataset
from tool.clip_extract_lang import get_language_clip

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps, game, instruction_type, games, instruct_dir):        
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.game = game
        self.type = instruction_type

        if instruction_type != 'raw':
             ## load text embedding
            if instruction_type == 'language':
                self.game_dict = dict()
                for i in range(len(games)):  
                    dir = instruct_dir + games[i] + '/language.txt'
                    with open(dir, 'r') as file:
                        Text = file.read()
                    self.game_dict[games[i]] = get_language_clip(Text).to('cpu')
            ## load trajexctory embedding
            if instruction_type == 'trajectoery':
                pass
            ## load instruction embedding
            if instruction_type == 'instruction':
                pass

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
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)
         
        if self.type == 'raw':
            return states, actions, rtgs, timesteps
        else:
            instruction = self.game[idx:done_idx]
            if self.type == 'language':
                instruction =  self.game_dict[instruction[0]].reshape(-1)  
            if self.type == 'trajectory':
                pass        
            if self.type == 'instruction':
                pass
            return states, actions, rtgs, timesteps, instruction

if __name__ == '__main__': 

    console.log('Loading config and dataset...')

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
    instruction_type = config['instruction_type']
    game_list = config['game_list']
    game_path = parent_directory + '/config/config_game/' + game_list + '.yaml'
    condition_dim = config['condition_dim']

    with open(game_path, 'r') as yaml_file:
        config_game = yaml.safe_load(yaml_file)
    games = config_game['train']
    
    seed = config['seed'] 
    set_seed(seed)

    #@ load datasdet
    obss, actions, returns, done_idxs, rtgs, timesteps, game = create_dataset(num_buffers, num_steps, games, data_dir_prefix, trajectories_per_buffer)
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )
    dataset = StateActionReturnDataset(obss, context_length*3, actions, done_idxs, rtgs, timesteps, game, instruction_type, games, instruct_dir)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    config['action_space'] = int( max(actions)+1 )
    config['max_timestep'] = int( max(timesteps) )
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    console.log('train-init model...')  
    ## original DT
    if instruction_type == 'raw':
        mconf = GPTConfig(dataset.vocab_size, dataset.block_size,
                        n_layer=6, n_head=8, n_embd=128, model_type=model_type, max_timestep=max(timesteps))
        model = GPT(mconf)
    ## task conditioned DT
    elif instruction_type == 'language':
        mconf = GPTConfig_condition(dataset.vocab_size, dataset.block_size,
                        n_layer=6, n_head=8, n_embd=128, model_type=model_type, max_timestep=max(timesteps), embed_dim=condition_dim, instruction_type=instruction_type)
        model = GPT_condition(mconf)

    console.log('train-init trainer...') 
    tconf = TrainerConfig(max_epochs=epochs, batch_size=batch_size, learning_rate=6e-4,
                lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*context_length*3,
                num_workers=4, seed=seed, model_type=model_type, game=game, max_timestep=max(timesteps), ckpt_path=ckpt_path)
    trainer = Trainer(model, train_dataset, test_dataset, tconf, instruction_type, game_list)
    
    console.log('train-training...') 
    trainer.train()
