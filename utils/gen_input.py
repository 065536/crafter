import torch
import numpy as np

def gen_input(info, device):
    map = info["semantic"]
    x, y = info["player_pos"]
    inv = info["inventory"]
    
    obs_size = 9
    
    radius = obs_size // 2
    
    partial_obs = np.zeros((obs_size, obs_size), dtype=map.dtype)
    
    map_size = map.shape[0]  
    x_min = max(0, x - radius)
    x_max = min(map_size, x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(map_size, y + radius + 1)

    x_start = radius - (x - x_min)
    y_start = radius - (y - y_min)

    partial_obs[x_start:x_start + (x_max - x_min), y_start:y_start + (y_max - y_min)] = map[x_min:x_max, y_min:y_max]
    
    partial_obs = np.expand_dims(partial_obs, axis=0) # add channel
    partial_obs = np.expand_dims(partial_obs, axis=0) # add batch_size
    partial_obs_tensor = torch.tensor(partial_obs, dtype=torch.float32).to(device)
    
    
    inventory_keys = [
            'health', 'food', 'drink', 'energy', 'sapling', 'wood', 'stone', 
            'coal', 'iron', 'diamond', 'wood_pickaxe', 'stone_pickaxe', 
            'iron_pickaxe', 'wood_sword', 'stone_sword', 'iron_sword'
        ]
    inventory_array = np.array([inv.get(key, 0) for key in inventory_keys])
    inventory_array = np.expand_dims(inventory_array, axis = 0)  # add batch_size
    inventory_tensor = torch.tensor(inventory_array, dtype=torch.float32).to(device)
    
    return partial_obs_tensor, inventory_tensor