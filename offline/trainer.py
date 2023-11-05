
#@ our package
from tool.utils import sample
#@ external package
import math
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from collections import deque
import random
import cv2
import torch
from PIL import Image
import os
import yaml
from rich.console import Console
console = Console()
import clip
import time


class TrainerConfig:
    ## optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    ## learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    ## checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    max_timestep = 100 

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, \
                 game_list, condition_type, enable_retrieval,enable_language,\
                 enable_trajectory, enable_action, enable_instruct):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.condition_type = condition_type
        self.game_list = game_list
        self.enable_language = enable_language
        self.enable_trajectory = enable_trajectory
        self.enable_action = enable_action
        self.enable_instruct = enable_instruct
        self.enable_retrieval = enable_retrieval
        self.max_timestep = self.config.max_timestep

        ## take over whatever gpus are on the system
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, loss):
        ## DataParallel wrappers keep raw model object in .module attribute

        ## get time
        from datetime import datetime
        parsed_time = datetime.now().date()
        
        ## save checkpoint
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        console.print(f"saving {self.config.ckpt_path}")
        if not os.path.exists(self.config.ckpt_path + str(parsed_time)):
            os.makedirs(self.config.ckpt_path + str(parsed_time))
        if self.condition_type == 'raw':
            torch.save(raw_model.state_dict(), self.config.ckpt_path + str(parsed_time) \
                       + '/' + str(self.max_timestep) + '_' + str(round(loss,2)) + '_' +\
                          str(self.condition_type) + '_'\
                       + self.game_list + '.ckpt')
        else:
            torch.save(raw_model.state_dict(), self.config.ckpt_path + str(parsed_time) \
                       + '/' + str(self.max_timestep) + '_' + str(round(loss,2)) + '_' +\
                          str(self.condition_type) + '_'\
                       + str(self.enable_retrieval) + '_' + str(self.enable_language) + '_'\
                       + str(self.enable_trajectory)  + '_' + str(self.enable_action)  + '_' \
                       + str(self.enable_instruct) + '_' + self.game_list + '.ckpt')

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config, self.condition_type)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

            #@ raw trainer：not use condition information to train
            if self.condition_type == 'raw':
                for it, (x, y, r, t) in pbar:

                    # place data on the correct device
                    x = x.to(self.device)
                    y = y.to(self.device)
                    r = r.to(self.device)
                    t = t.to(self.device)

                    # forward the model
                    with torch.set_grad_enabled(is_train):
                        # logits, loss = model(x, y, r)
                        logits, loss = model(x, y, y, r, t)
                        loss = loss.mean() 
                        # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                    if is_train:

                        # backprop and update the parameters
                        model.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()

                        # decay the learning rate based on our progress
                        if config.lr_decay:
                            self.tokens += (y >= 0).sum() 
                            # number of tokens processed this step (i.e. label is not -100)
                            if self.tokens < config.warmup_tokens:
                                # linear warmup
                                lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                            else:
                                # cosine learning rate decay
                                progress = float(self.tokens - config.warmup_tokens) / \
                                           float(max(1, config.final_tokens - config.warmup_tokens))
                                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            lr = config.learning_rate * lr_mult
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr
                        else:
                            lr = config.learning_rate

                        # report progress
                        pbar.set_description(f"epoch {epoch+1} iter {it}: train loss\
                                              {loss.item():.5f}. lr {lr:e}")
            
            #@ condition trainer: use condition information to train 
            else:
                for it, (x, y, r, t, lang, traj, a_t, a, i, s_embed) in pbar:
                    # place data on the correct device

                    x = x.to(self.device)
                    y = y.to(self.device)
                    r = r.to(self.device)
                    t = t.to(self.device)
                    lang = lang.to(self.device)
                    traj = traj.to(self.device)
                    a_t = a_t.to(self.device)
                    a = a.to(self.device)
                    i = i.to(self.device)
                    s_embed = s_embed.to(self.device)
                    
                    # forward the model
                    with torch.set_grad_enabled(is_train):
                        logits, loss = model(x, y, y, r, t, lang, traj, a_t, a, i, s_embed) 
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                    if is_train:
                        # backprop and update the parameters
                        model.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()

                        # decay the learning rate based on our progress
                        if config.lr_decay:
                            self.tokens += (y >= 0).sum()
                             # number of tokens processed this step (i.e. label is not -100)
                            if self.tokens < config.warmup_tokens:
                                # linear warmup
                                lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                            else:
                                # cosine learning rate decay
                                progress = float(self.tokens - config.warmup_tokens) /\
                                      float(max(1, config.final_tokens - config.warmup_tokens))
                                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            lr = config.learning_rate * lr_mult
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr
                        else:
                            lr = config.learning_rate

                        # report progress
                        pbar.set_description(f"epoch {epoch+1} iter {it}: train loss\
                                              {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                return test_loss
    
        self.tokens = 0
        for epoch in range(config.max_epochs):
            run_epoch('train')
            test_loss = run_epoch('test')
            console.log(f"epoch test_loss: {test_loss}")
            self.save_checkpoint(test_loss)