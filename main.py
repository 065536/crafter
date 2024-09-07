import argparse
import numpy as np
import pygame
from PIL import Image
import crafter
from crafter.env import EnvWithDirection
from utils import ObservationToOption, text_obs, OptionToAction, gen_input, gen_prompt, check_task_done, count_tokens
from model import PPO, Buffer, Critic
import torch
import os
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque

ACTION_LIST = ['noop', 'move_left', 'move_right', 'move_up', 'move_down', 'do', 'sleep', 
            'place_stone', 'place_table', 'place_furnace', 'place_plant', 
            'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe', 
            'make_wood_sword', 'make_stone_sword', 'make_iron_sword']

def main():
    boolean = lambda x: bool(['False', 'True'].index(x))
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--area', nargs=2, type=int, default=(64, 64))
    parser.add_argument('--view', type=int, nargs=2, default=(9, 9))
    parser.add_argument('--length', type=int, default=None)
    parser.add_argument('--health', type=int, default=9)
    parser.add_argument('--window', type=int, nargs=2, default=(600, 600))
    parser.add_argument('--size', type=int, nargs=2, default=(0, 0))
    parser.add_argument('--record', type=str, default=None)
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--wait', type=boolean, default=False)
    parser.add_argument('--death', type=str, default='quit', choices=[
        'continue', 'reset', 'quit'])
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", type=str, default="log") # Where to log diagnostics to
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
    parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
    parser.add_argument("--max_step", type=int, default=200)
    parser.add_argument("--task", type=str, default="make_stone_pickaxe", help=["make_stone_pickaxe", "eat_cow"])
    parser.add_argument("--n_iter", type=int, default=1000)
    
    args = parser.parse_args()
    
    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)

    crafter.constants.items['health']['max'] = args.health
    crafter.constants.items['health']['initial'] = args.health

    size = list(args.size)
    size[0] = size[0] or args.window[0]
    size[1] = size[1] or args.window[1]

    # env = EnvWithDirection(
    #     area=args.area, view=args.view, length=args.length, seed=args.seed)
    # env = crafter.Recorder(env, args.record)
    # obs = env.reset()
    # info = {
    #     'inventory': env._player.inventory.copy(),
    #     'achievements': env._player.achievements.copy(),
    #     'discount': 1,
    #     'semantic': env._sem_view(),
    #     'player_pos': env._player.pos,
    #     'reward': 0,
    # }
    # partial_obs, inv = gen_input(info, args.device)
    # obs_space = partial_obs.shape
    
    device = args.device
    save_path = os.path.join(dir_path, args.logdir)
    batch_size = args.batch_size
    max_step = args.max_step
    task = args.task
    gamma = args.gamma
    lam = args.lam
    n_iter = args.n_iter
    n_games = 0
    total_steps = 0
    meta_controller_coef = 10
    total_token = 0
    if_render = False
    
    buffer = Buffer(gamma, lam, device)
    buffer.clear()
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=save_path)

    # Initialize a deque to store the results of the last N games
    N = 10  # Number of recent games to consider for success rate calculation
    recent_results = deque(maxlen=N)

    for i in range(n_iter):
        print("\n" * 5)
        print("=" * 50)
        print("start a new game")
        env = EnvWithDirection(
        area=args.area, view=args.view, length=args.length, seed=args.seed)
        env = crafter.Recorder(env, args.record)
        obs = env.reset()
        info = {
            'inventory': env._player.inventory.copy(),
            'achievements': env._player.achievements.copy(),
            'discount': 1,
            'semantic': env._sem_view(),
            'player_pos': env._player.pos,
            'reward': 0,
        }
        partial_obs, inv = gen_input(info, args.device)
        
        obs_space = partial_obs.shape

        agent = PPO(obs_space,
                    env.action_space.n,
                    device,
                    save_path,
                    batch_size=batch_size)
        meta_value_network = Critic(obs_space, env.action_space.n)
        meta_value_network.critic_network.to(device)
        
        achievements = set()
        duration = 0
        return_ = 0
        was_done = False
        done = False
        mem = {}
        
        if n_games <= n_iter / 10:
            is_IL = True
        else:
            is_IL = False

        print('Diamonds exist:', env._world.count('diamond'))
        if if_render:
            pygame.init()
            screen = pygame.display.set_mode(args.window)
            clock = pygame.time.Clock()
            

        running = True

        while running and not done and duration <= max_step:

            
            meta_action = []
            
            if is_IL:
                # call meta-controller
                facing_dir = env._direction
                res = text_obs(info)
                obs_to_option = ObservationToOption(res, info)
                option = obs_to_option.decide_option(mem)
                # print("option = ", option)
                prompt = gen_prompt(res, task)
                token = count_tokens(prompt)
                total_token += token
                option_to_action = OptionToAction(info, facing_dir, mem)
                meta_action = option_to_action.option_to_action(option)
                # print("action = ", meta_action)
            else:
                meta_action.append(0)
            
            # Environment step
            if isinstance(meta_action, int) : meta_action = [meta_action]
            for a in meta_action:
                
                if if_render:
                    image = env.render(size)
                    if size != args.window:
                        image = Image.fromarray(image)
                        image = image.resize(args.window, resample=Image.NEAREST)
                        image = np.array(image)
                    surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
                    screen.blit(surface, (0, 0))
                    pygame.display.flip()
                    clock.tick(args.fps)
                
                # call controller
                input_obs = dict()
                facing_dir = env._direction
                part_obs, inv = gen_input(info, device)
                input_obs["semantic"] = part_obs
                input_obs["inventory"] = inv
                
                dist, value, _ = agent(input_obs)
                controller_logits = dist.logits
                controller_prob = torch.softmax(controller_logits, dim=-1)
                controller_log_prob = torch.log(controller_prob)
                
                if is_IL:
                    meta_controller_probs = np.zeros(env.action_space.n)
                    meta_controller_probs[a] = 1
                    meta_controller_probs_tensor = torch.tensor(meta_controller_probs)
                    meta_controller_probs_tensor = meta_controller_probs_tensor.unsqueeze(0)
                    meta_controller_probs_tensor = meta_controller_probs_tensor.to(controller_prob.device)
                    # prob = controller_prob + meta_controller_coef * meta_controller_probs_tensor
                    prob = meta_controller_coef * meta_controller_probs_tensor
                    probs = F.softmax(prob.squeeze(), dim=-1)
                    action = torch.multinomial(probs, 1)
                    
                    # print(f"a: {a}")
                    # print(f"meta_controller_probs_tensor: {meta_controller_probs_tensor}")
                    # print(f"controller_prob: {controller_prob}")
                    # print(f"probs: {probs}")
                    # print(f"action: {action}")
                    
                    meta_controller_value = meta_value_network(input_obs)
                else:
                    action = dist.sample()
                
                log_probs = dist.log_prob(action)
                action = action.cpu().item()
                
                if not is_IL : a = action

                next_obs, reward, done, next_info = env.step(a)
                opt = OptionToAction(info, facing_dir, mem)
                opt.update_map_memory()
                mem = opt.map_memory
            
                if done is None:
                    done = check_task_done(env, task)
                
                next_input_obs = dict()
                next_input_obs["semantic"], next_input_obs["inventory"] = gen_input(next_info, device)
                
                info = next_info

                if done : done = torch.tensor([.1])
                else : done = torch.tensor([.0])
                reward = torch.tensor([reward])
                
                if is_IL:
                    buffer.store(input_obs, 
                                next_input_obs,
                                action, 
                                reward.detach().cpu().numpy(), 
                                value.detach().cpu().numpy(), 
                                log_probs.detach().cpu().numpy(), 
                                controller_log_prob.squeeze(0).detach().cpu().numpy(),
                                meta_controller_probs,
                                meta_controller_value.detach().cpu().numpy(),
                                meta_controller_probs_tensor.squeeze(0).detach().cpu().numpy(),
                                done.detach().cpu().numpy()
                            )
                else:
                    buffer.store(input_obs, 
                                next_input_obs,
                                action, 
                                reward.detach().cpu().numpy(), 
                                value.detach().cpu().numpy(), 
                                log_probs.detach().cpu().numpy(), 
                                controller_log_prob.detach().cpu().numpy(),
                                controller_log_prob.detach().cpu().numpy(),
                                value.detach().cpu().numpy(),
                                controller_log_prob.detach().cpu().numpy(),
                                done.detach().cpu().numpy()
                            )
            
                duration += 1
                total_steps += 1
                
                if done:
                    value = 0
                else:
                    value = agent(input_obs)[1].detach().cpu().numpy()

                buffer.finish_path(last_val=value)
                
                # Achievements
                unlocked = {
                    name for name, count in env._player.achievements.items()
                    if count > 0 and name not in achievements}
                for name in unlocked:
                    achievements |= unlocked
                    total = len(env._player.achievements.keys())
                    print(f'Achievement ({len(achievements)}/{total}): {name}')
                if env._step > 0 and env._step % 100 == 0:
                    print(f'Time step: {env._step}')
                if reward:
                    print(f'Reward: {reward}')
                    return_ += reward

                # Log reward to TensorBoard
                writer.add_scalar("Train/Reward", return_, total_steps)
                writer.add_scalar("Train/Steps_Per_Game", duration, total_steps)
                
                if total_steps % 50 == 0:
                    if is_IL:
                        loss1 = meta_value_network.update_policy(buffer)
                        loss2 = agent.update_network(buffer)
                    else:
                        agent.update_policy(buffer)

                # Episode end
                if done and not was_done:
                    was_done = True
                    print('Episode done!')
                    print('Duration:', duration)
                    print('Return:', return_)
                    
                    # Append the result of this game to the deque (1 if successful, 0 otherwise)
                    recent_results.append(1 if done else 0)
    
                    # Calculate the success rate over the last N games
                    success_rate = sum(recent_results) / len(recent_results)
    
                    # Log the success rate to TensorBoard
                    writer.add_scalar("Train/Success_Rate", success_rate, n_games)
                    
                    if args.death == 'quit':
                        running = False
                        break
                    if args.death == 'reset':
                        print('\nStarting a new episode.')
                        env.reset()
                        achievements = set()
                        was_done = False
                        duration = 0
                        return_ = 0
                    if args.death == 'continue':
                        pass
            
            
            if not running:
                break
        n_games += 1
        # pygame.quit()
        
        if n_games % 10 == 1:
            meta_controller_coef *= 0.99
            
            # evaluate the model
            done = False
            eval_successes = 0
            eval_steps = []
            eval_rewards = []
            while not done and duration <= max_step:
                env.reset()
                info = {
                    'inventory': env._player.inventory.copy(),
                    'achievements': env._player.achievements.copy(),
                    'discount': 1,
                    'semantic': env._sem_view(),
                    'player_pos': env._player.pos,
                    'reward': 0,
                }
                partial_obs, inv = gen_input(info, args.device)
                input_obs["semantic"] = part_obs
                input_obs["inventory"] = inv
                
                dist, value, _ = agent(input_obs)
                action = dist.sample()
                action = action.cpu().item()
                
                next_obs, reward, done, next_info = env.step(action)
                if done is None:
                    done = check_task_done(env, task)
                
                eval_steps.append(duration)
                eval_rewards.append(reward)
                eval_successes += 1 if done else 0
            
            # Ensure we only log if there were evaluation steps
            if eval_steps:
                writer.add_scalar("Evaluate/Success_Rate", eval_successes / len(eval_steps), n_games)
                writer.add_scalar("Evaluate/Average_Steps_Per_Game", np.mean(eval_steps), n_games)
                writer.add_scalar("Evaluate/Average_Reward_Per_Game", np.mean(eval_rewards), n_games)
            else:
                print("No evaluation steps recorded. Skipping logging.")
                
    # Log total token count
    writer.add_scalar("Train/Total_Token", total_token, n_iter)
    
    # Close the writer
    writer.close()
    
    
        
if __name__ == '__main__':
    main()
