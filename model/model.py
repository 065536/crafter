import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

class NNBase(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        
        # 确定通道数（通常是从obs_space提取）
        if isinstance(obs_space, tuple) or isinstance(obs_space, list):
            channel = obs_space[1]
            height = obs_space[2]
            width = obs_space[3]
        else:
            raise ValueError("obs_space must be a tuple or list containing (batch_size, channels, height, width)")

        # 定义图像嵌入的卷积层
        self.image_conv = nn.Sequential(
            nn.Conv2d(channel, 16, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)), 
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)), 
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.ReLU()
        )
        
        # 计算卷积层输出的大小
        dummy_x = torch.zeros((1, channel, height, width))
        conv_output_size = np.prod(self.image_conv(dummy_x).view(1, -1).size(1))

        # 定义inventory的嵌入层
        inventory_keys = [
            'health', 'food', 'drink', 'energy', 'sapling', 'wood', 'stone', 
            'coal', 'iron', 'diamond', 'wood_pickaxe', 'stone_pickaxe', 
            'iron_pickaxe', 'wood_sword', 'stone_sword', 'iron_sword'
        ]
        self.inventory_size = len(inventory_keys)
        self.inventory_embedding = nn.Sequential(
            nn.Linear(self.inventory_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 最终的嵌入大小
        self.embedding_size = conv_output_size + 64
        self.action_space = action_space
    
    def forward(self, obs, masks=None, states=None):
        raise NotImplementedError


class MLPBase(NNBase):
    def __init__(self, obs_space, action_space):
        super().__init__(obs_space, action_space)
        
        # 定义actor的模型
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )
        # 定义critic的模型
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def init_states(self, device=None, num_trajs=1):
        return None

    def forward(self, obs, masks=None, states=None):
        input_dim = len(obs['semantic'].size())
        assert input_dim == 4, "Expected input of 4 dimensions (batch_size, channels, height, width), but got {}.".format(input_dim)
        
        # 处理语义观察的卷积操作
        x = self.image_conv(obs['semantic'])  
        x = x.view(x.size(0), -1)  

        # 直接处理inventory的输入，形状为 [batch_size, features]
        inventory_input = obs["inventory"]
        inventory_embedding = self.inventory_embedding(inventory_input)
        
        # 拼接嵌入
        full_embedding = torch.cat((x, inventory_embedding), dim=1)
        
        # actor-critic
        value = self.critic(full_embedding).squeeze(1)
        action_logits = self.actor(full_embedding)
        dist = Categorical(logits=action_logits)

        return dist, value, full_embedding


class Critic_network(NNBase):
    def __init__(self, obs_space, action_space):
        super().__init__(obs_space, action_space)
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, obs):
        input_dim = len(obs['semantic'].size())
        assert input_dim == 4, "Expected input of 4 dimensions (batch_size, channels, height, width), but got {}.".format(input_dim)
        
        # 处理语义观察的卷积操作
        x = self.image_conv(obs['semantic'])  
        x = x.view(x.size(0), -1)  

        # 直接处理inventory的输入，形状为 [batch_size, features]
        inventory_input = obs["inventory"]
        inventory_embedding = self.inventory_embedding(inventory_input)
        
        # 拼接嵌入
        full_embedding = torch.cat((x, inventory_embedding), dim=1)
        value = self.critic(full_embedding).squeeze(1)
        
        return value