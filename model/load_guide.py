from tool.clip_extract_vision import get_vision_clip
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
console = Console()
import torch
import torch.nn as nn
import numpy as np

def load_traj(game):
    #@ load traj
    instruct_dir = f'/home/Userlist/jinyg/dtmi/instruct/{game}/frame/'
    guide_loaded = []
    guide_dir = f'{instruct_dir}{0}/'
    for j in range(20):
        guidedir_j = f'{guide_dir}{j}.png'
        guide_loaded.append(get_vision_clip(guidedir_j))
    result = torch.cat(guide_loaded, dim=0).to(torch.float)
    return result

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
    
    console.log("load guide")
    game = 'Air_Raid'
    guide_loaded = load_traj(game).reshape(1, 20, 512)
    print(guide_loaded.shape)

    console.log("atttention seq to vec")
    att_seqtovec = Attention_Seqtovec(512, 512, 2, 1).to(guide_loaded.device)
    res = att_seqtovec(guide_loaded)
    print(res.shape)
