import math

import torch
import torch.nn as nn
import numpy as np

def hyperfanin_init_weight(linear_layer, hypernet_in, mainnet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in * mainnet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)


def hyperfanin_init_bias(linear_layer, hypernet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)


class SimpleGenerator(nn.Module):
    def __init__(self, config, input_dim, hidden_size):
        super().__init__()
        adapter_dim = config.adapter_dim
        self.input_dim = input_dim
        # self.hidden_dim = config.hypernetwork_bottleneck
        self.linear1 = nn.Linear(self.input_dim, 128)
        self.activation_fn = nn.ReLU()
        self.linear2 = nn.Linear(128, 64)
        self.LayerNorm = nn.LayerNorm(64, eps=1e-6)
        # output weights
        self.weight_up = nn.Linear(64, hidden_size * adapter_dim)
        self.weight_down = nn.Linear(64, hidden_size * adapter_dim)
        self.bias_up = nn.Linear(64, hidden_size)
        self.bias_down = nn.Linear(64, adapter_dim)
        # init weights
        hyperfanin_init_weight(self.weight_up, 64, adapter_dim)
        hyperfanin_init_weight(self.weight_down, 64, hidden_size)
        hyperfanin_init_bias(self.bias_up, 64)
        hyperfanin_init_bias(self.bias_down, 64)

    def forward(self, x):  # x:batch * task_dim+layer_dim
        x = self.linear1(x)  # x:batch * (hidden_size * adapter_dim)
        x = self.activation_fn(x)
        x = self.linear2(x)  # x是投影后的task-embedding
        x = self.LayerNorm(x)
        return (
            self.weight_up(x),  # batch * hidden_dim * adapter_dim
            self.weight_down(x),
            self.bias_up(x),
            self.bias_down(x),
        )


class ParameterGenerator(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.config = config
        self.layer_embed = nn.Embedding(config.n_layer, config.l_embed_dim)
        self.decoder = SimpleGenerator(
            config, config.condition_dim + config.l_embed_dim, hidden_size
        )

    def forward(self, hidden_inputs, similarity):
        # hidden_inputs: bs * guide_num * guide_dim
        layers = []
        # hidden_inputs = torch.randn(128, 5, 256, device=hidden_inputs.device)  # bs * guide_num * guide_dim
        # similarity = torch.randn(128, 1, 5, device=hidden_inputs.device,)  # bs * 1 * guide_num
        # setup idxs we need
        layers_idxs = torch.arange(
            0,
            self.config.n_layer,
            dtype=torch.long,
            device=hidden_inputs.device,
        )
        layers_idxs = layers_idxs.repeat(hidden_inputs.size(0), 1)  #  bs * layer_num

        for i in range(self.config.n_layer):
            final_output = []
            layer_embed = self.layer_embed(layers_idxs[:, i]).unsqueeze(1).expand(-1, 5, -1)  # bs * guide_num * layer_emb_dim
            temp = torch.cat([hidden_inputs, layer_embed], dim=2)  # bs * guide_num * (guide_dim+layer_emb_dim)
            temp_output = list(self.decoder(temp))  # bs * guide_num * (hidden_dim * adapter_dim)
            for j in range(len(temp_output)):
                final_output.append((torch.bmm(similarity, temp_output[j])))
            final_output = tuple(final_output)
            layers.append(final_output)   #  12层，每层对应batch * weight
        return layers
