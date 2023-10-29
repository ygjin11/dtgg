from tool.clip_extract_vision import get_vision_clip
from tool.clip_extract_lang import get_language_clip
from rich.console import Console
console = Console()
import torch
import clip
import torch.nn as nn
import numpy as np
import json
import os
import yaml



## load game description
def load_language(game, model, device):
    dir = '/home/Userlist/jinyg/dtmi/instruct/' +  game + '/language.txt'
    with open(dir, 'r') as file:
        Text = file.read()
    return get_language_clip(model, Text, device)

## load trajectory
def load_traj(game, model, process, device):
    #@ load traj
    instruct_dir = f'/home/Userlist/jinyg/dtmi/instruct/{game}/frame/'
    guide_loaded = []
    guide_dir = f'{instruct_dir}{0}/'
    for j in range(20):
        guide_dir_j = f'{guide_dir}{j}.png'
        guide_loaded.append(get_vision_clip(model, process, guide_dir_j, device))
    result = torch.cat(guide_loaded, dim=0).to(torch.float)
    return result


## load action description
def load_action(game, model, device):
    instruct_dir = f'/home/Userlist/jinyg/dtmi/instruct/{game}/action.json'
    with open(instruct_dir, 'r') as file:
        action = json.load(file)
    input_num = action['number']
    act_seq = action['seq']
    #@ action number
    act_num = torch.tensor(input_num).to(device)
    #@ action sequence
    act = []
    for key, value in act_seq.items():
        act_i = get_language_clip(model, value, device)
        act.append(act_i)
    act = torch.cat(act, dim=0).to(torch.float)

    return act_num, act

## load game example - trajectory & instruction
def load_traj_5(game, model, process, device):
    #@ load traj
    result = []
    instruct_dir = f'/home/Userlist/jinyg/dtmi/instruct/{game}/frame/'
    for i in range(5):
        guide_loaded = []
        guide_dir = f'{instruct_dir}{i}/'
        for j in range(20):
            guide_dir_j = f'{guide_dir}{j}.png'
            guide_loaded.append(get_vision_clip(model, process, guide_dir_j, device))
        result_i = torch.cat(guide_loaded, dim=0).to(torch.float).reshape(1,20,512)
        result.append(result_i)
    result = torch.cat(result, dim=0).to(torch.float)
    return result

## MLP: concat all info 
512, 256, 256, 512
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.weight = nn.Parameter(torch.randn(3))
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, language, act_num, act, traj):

        acc_all = torch.cat((act_num, act), dim=1)
        weight = self.softmax(self.weight)
        x = weight[0] * language + weight[1] * acc_all + weight[2] * traj
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

## encoder：encode seq to vectort
class Attention_Seqtovec(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers):
        super(Attention_Seqtovec, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, num_heads),
            num_layers
        )
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        x = self.transformer(x.permute(1, 0, 2))
        # x: (seq_length, batch_size, input_dim)
        output_vector = x[-1, :, :]
        output_vector = self.fc(output_vector)
        return output_vector

if __name__ == '__main__':
    game = 'Air_Raid'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, process = clip.load("ViT-B/32", device=device)
    
    console.log("load guide - trajectory")
    trajectory_1, trajectory_2, trajectory_3, trajectory_4, trajectory_5 \
        = load_traj_5(game, model, process, device)
    trajectory_1, trajectory_2, trajectory_3, trajectory_4, trajectory_5 \
        = trajectory_1.reshape(1,20,512), trajectory_2.reshape(1,20,512), \
        trajectory_3.reshape(1,20,512), trajectory_4.reshape(1,20,512), trajectory_5.reshape(1,20,512)
    console.print(f"trajectory shape : {trajectory_1.shape}")
    
    
    console.log("load guide - action")
    act_num, act = load_action(game, model, device)
    act_num, act = act_num.unsqueeze(0), act.unsqueeze(0)
    console.print(f"act_num, act shape : {act_num.shape},{act.shape}")
    
    console.log("load guide - lanauge")
    language = load_language(game, model, device)
    console.print(f"language shape : {language.shape}")

    #@ input model
    ## language 
    # batch, 512
    ## traj 
    # batch, ((512, 512) ,20) * 5
    ## act_num, act
    # batch, 18
    # batch, x, 512
   
    console.log("encode guide - trajectory")
    trajinst_seqtovec = Attention_Seqtovec(1024, 512, 2, 1).to(device)
    traj_seqtovec = Attention_Seqtovec(512, 512, 2, 1).to(device)
    trajectory_1, trajectory_2, trajectory_3, trajectory_4, trajectory_5 = \
         traj_seqtovec(trajectory_1), traj_seqtovec(trajectory_2), traj_seqtovec(trajectory_3) \
            ,traj_seqtovec(trajectory_4), traj_seqtovec(trajectory_5) 
    console.print(f"trajectory shape : {trajectory_1.shape}")

    console.log("encode guide - action")
    embedding_num = nn.Embedding(18, 256).to(device)
    act_num =  embedding_num(act_num)
    act_seqtovec = Attention_Seqtovec(512, 256, 2, 1).to(device)
    act = act_seqtovec(act)
    console.print(f"act_num & act shape : {act_num.shape} {act.shape}")
    
    console.log("fuse")
    Fusion = MLP(512, 1024, 512).to(device)
    cond_1 = Fusion(language, act_num, act, trajectory_1)
    cond_2 = Fusion(language, act_num, act, trajectory_2)
    cond_3 = Fusion(language, act_num, act, trajectory_3)
    cond_4 = Fusion(language, act_num, act, trajectory_4)
    cond_5 = Fusion(language, act_num, act, trajectory_5)
    print(f"condition shape : {cond_1.shape}")


