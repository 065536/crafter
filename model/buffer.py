import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch

class Buffer:
    """
    A buffer for storing trajectory data and calculating returns for the policy
    and critic updates.
    """
    def __init__(self, gamma=0.99, lam=0.95, device='cpu'):
        self.gamma = gamma
        self.lam = lam  # Unused in this example
        self.device = device
        self.clear()

    def __len__(self):
        return self.ptr
    
    def clear(self):
        self.obs = []
        self.next_obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.controller_prob = []
        self.meta_controller_probs = []
        self.meta_controller_values = []
        self.meta_controller_tensor = []

        self.ptr = 0
        self.traj_idx = [0]
        self.returns = []
        self.ep_returns = []  # For logging
        self.ep_lens = []
        self.done = []

    def store(self, state, next_state, action, reward, value, log_probs, controller_prob, meta_controller_probs, meta_controller_values, meta_controller_tensor, done):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        if len(self.obs) == 0:
            # Initialize storage for dictionary keys
            self.obs = {key: [] for key in state.keys()}
            self.next_obs = {key: [] for key in next_state.keys()}

        for key in state.keys():
            self.obs[key].append(state[key].detach().cpu())  # Detach and move to CPU
            self.next_obs[key].append(next_state[key].detach().cpu())  # Same for next_obs

        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_probs)
        self.controller_prob.append(controller_prob)
        self.meta_controller_probs.append(meta_controller_probs)
        self.meta_controller_values.append(meta_controller_values)
        self.meta_controller_tensor.append(meta_controller_tensor)
        self.done.append(done)
        self.ptr += 1

    def finish_path(self, last_val=None):
        self.traj_idx.append(self.ptr)
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]

        returns = []
        R = last_val if last_val is not None else 0
        for reward in reversed(rewards):
            R = self.gamma * R + reward
            returns.insert(0, R)

        self.returns += returns
        self.ep_returns.append(np.sum(rewards))
        self.ep_lens.append(len(rewards))
    
    def get(self):
        # Convert lists of frames into arrays of shape [trajectory_len, C, H, W]
        obs_dict = {
            key: torch.cat(self.obs[key], dim=0).to(self.device) 
            for key in self.obs.keys()
        }
        next_obs_dict = {
            key: torch.cat(self.next_obs[key], dim=0).to(self.device) 
            for key in self.next_obs.keys()
        }
        
        def gen_input(input):

            if len(input) == 0:
                return torch.tensor([], dtype=torch.float32).view(-1, 1).to(self.device)
            
            if isinstance(input[0], (int, float)):
                input_tensor = torch.tensor(input, dtype=torch.float32).view(-1, 1)
            else:
                flat_input = np.concatenate(input)
                input_tensor = torch.tensor(flat_input, dtype=torch.float32).view(-1, 1)
            
            return input_tensor.to(self.device)
        
        def process_meta_controller_probs(meta_controller_probs):
            """
            Converts a list of arrays, each with length 17, into a tensor with dimensions (n, 17).
            
            Args:
                meta_controller_probs (list of arrays): A list where each element is an array of length 17.
            
            Returns:
                torch.Tensor: A tensor of shape (n, 17).
            """
            # Stack the list into a single tensor
            tensor = torch.stack([torch.tensor(prob, dtype=torch.float32) for prob in meta_controller_probs])
            
            return tensor
        
        # print(f"self.returns: {self.returns}")
        
        return (
            obs_dict,
            next_obs_dict,
            torch.tensor(self.actions).view(-1, 1).to(self.device),  # Use torch.tensor instead of torch.stack for actions
            gen_input(self.returns),  # Convert numpy arrays to torch tensors
            gen_input(self.values),
            gen_input(self.log_probs),
            process_meta_controller_probs(self.controller_prob),
            process_meta_controller_probs(self.meta_controller_probs),
            gen_input(self.meta_controller_values),
            process_meta_controller_probs(self.meta_controller_tensor),
            gen_input(self.done)
        )

    def sample(self, batch_size=64, recurrent=False):
        if recurrent:
            random_indices = np.random.permutation(len(self.ep_lens))
            last_index = random_indices[-1]
            sampler = []
            indices = []
            num_sample = 0
            for i in random_indices:
                indices.append(i)
                num_sample += self.ep_lens[i]
                if num_sample > batch_size or i == last_index:
                    sampler.append(indices)
                    indices = []
                    num_sample = 0
        else:
            random_indices = SubsetRandomSampler(range(self.ptr))
            sampler = BatchSampler(random_indices, batch_size, drop_last=True)

        obs_dict, next_obs_dict, actions, returns, values, log_probs, controller_prob, meta_controller_probs, meta_controller_values, meta_controller_tensor, done = self.get()

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for indices in sampler:
            if recurrent:
                obs_batch = {key: pad_sequence([obs_dict[key][self.traj_idx[i]:self.traj_idx[i+1]] for i in indices], batch_first=False) for key in obs_dict}
                next_obs_batch = {key: pad_sequence([next_obs_dict[key][self.traj_idx[i]:self.traj_idx[i+1]] for i in indices], batch_first=False) for key in next_obs_dict}
                action_batch = pad_sequence([actions[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices], batch_first=False).flatten(0, 1)
                return_batch = pad_sequence([returns[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices], batch_first=False).flatten(0, 1)
                advantage_batch = pad_sequence([advantages[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices], batch_first=False).flatten(0, 1)
                values_batch = pad_sequence([values[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices], batch_first=False).flatten(0, 1)
                mask = pad_sequence([torch.ones_like(r) for r in return_batch], batch_first=False).flatten(0, 1)
                log_prob_batch = pad_sequence([log_probs[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices], batch_first=False).flatten(0, 1)
                controller_prob_batch = pad_sequence([controller_prob[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices], batch_first=False).flatten(0, 1)
                meta_controller_prob_batch = pad_sequence([meta_controller_probs[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices], batch_first=False).flatten(0, 1)
                meta_controller_value_batch = pad_sequence([meta_controller_values[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices], batch_first=False).flatten(0, 1)
                meta_controller_tensor_batch = pad_sequence([meta_controller_tensor[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices], batch_first=False).flatten(0, 1)
                done_batch = pad_sequence([done[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices], batch_first=False).flatten(0, 1)
            else:
                obs_batch = {key: obs_dict[key][indices] for key in obs_dict}
                next_obs_batch = {key: next_obs_dict[key][indices] for key in next_obs_dict}
                action_batch = actions[indices]
                return_batch = returns[indices]
                advantage_batch = advantages[indices]
                values_batch = values[indices]
                mask = torch.FloatTensor([1])
                log_prob_batch = log_probs[indices]
                controller_prob_batch = controller_prob[indices]
                meta_controller_prob_batch = meta_controller_probs[indices]
                meta_controller_value_batch = meta_controller_values[indices]
                meta_controller_tensor_batch = meta_controller_tensor[indices]
                done_batch = done[indices]

            yield obs_batch, next_obs_batch, action_batch.to(self.device), return_batch.to(self.device), advantage_batch.to(self.device), values_batch.to(self.device), mask.to(self.device), log_prob_batch.to(self.device), controller_prob_batch.to(self.device), meta_controller_prob_batch.to(self.device), meta_controller_value_batch.to(self.device), meta_controller_tensor_batch.to(self.device), done_batch.to(self.device)
