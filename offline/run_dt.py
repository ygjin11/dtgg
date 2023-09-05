import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset,  random_split
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from create_dataset import create_dataset
from tool.clip_extract_lang import get_language_clip
import os
import yaml

# dataset
class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps, game, instruction_type, game_list):        
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.game = game
        self.game_dict = dict()
        for i in range(len(game_list)):  
            self.game_dict[game_list[i]] = get_language_clip(game_list[i])
        self.type = instruction_type

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
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
                instruction =  [self.game_dict[item] for item in instruction] 
                instruction = torch.cat(instruction, dim=0) # (block_size, 512)
            if self.type == 'trajectory':
                pass        
            if self.type == 'instruction':
                pass
            return states, actions, rtgs, timesteps, instruction

if __name__ == '__main__': 

    # load config
    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    parent_directory = os.path.dirname(current_directory)
    config_path = parent_directory + '/config/config_para/para.yaml'
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
        
    seed = config['seed'] 
    context_length = config['context_length'] 
    epochs = config['epochs'] 
    model_type = config['model_type']
    num_steps = config['num_steps']
    num_buffers = config['num_buffers']
    batch_size = config['batch_size']
    trajectories_per_buffer = config['trajectories_per_buffer']
    data_dir_prefix = config['dataset_dir']
    instruction_type = config['instruction_type']
    game_list = config['game_list']
    ckpt_path = config['ckpt_path']
    game_path = parent_directory + '/config/config_game/' + game_list + '.yaml'

    with open(game_path, 'r') as yaml_file:
        config_game = yaml.safe_load(yaml_file)
    game_list = config_game['train']
        
    set_seed(seed)

    obss, actions, returns, done_idxs, rtgs, timesteps, game = create_dataset(num_buffers, num_steps, game_list, data_dir_prefix, trajectories_per_buffer)
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )
    dataset = StateActionReturnDataset(obss, context_length*3, actions, done_idxs, rtgs, timesteps, game, instruction_type, game_list)

    config['action_space'] = int( max(actions)+1 )
    config['max_timestep'] = int( max(timesteps) )

    print(max(timesteps))

    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    # original DT training
    if instruction_type == 'raw':
        mconf = GPTConfig(dataset.vocab_size, dataset.block_size,
                        n_layer=6, n_head=8, n_embd=128, model_type=model_type, max_timestep=max(timesteps))
        model = GPT(mconf)

        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        tconf = TrainerConfig(max_epochs=epochs, batch_size=batch_size, learning_rate=6e-4,
                            lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*context_length*3,
                            num_workers=4, seed=seed, model_type=model_type, game=game, max_timestep=max(timesteps), ckpt_path=ckpt_path)
    
        trainer = Trainer(model, train_dataset, test_dataset, tconf, 'raw')
        trainer.train()
    
    # task conditioned training
    else:
        pass