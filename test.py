import argparse
import numpy as np
import pygame
from PIL import Image
import crafter
from crafter.env import EnvWithDirection
from utils import ObservationToOption, text_obs, OptionToAction


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
    parser.add_argument('--death', type=str, default='reset', choices=[
        'continue', 'reset', 'quit'])
    args = parser.parse_args()

    crafter.constants.items['health']['max'] = args.health
    crafter.constants.items['health']['initial'] = args.health

    size = list(args.size)
    size[0] = size[0] or args.window[0]
    size[1] = size[1] or args.window[1]

    env = EnvWithDirection(
        area=args.area, view=args.view, length=args.length, seed=args.seed)
    env = crafter.Recorder(env, args.record)
    obs = env.reset()
    info = None
    
    achievements = set()
    duration = 0
    return_ = 0
    was_done = False
    done = False

    print('Diamonds exist:', env._world.count('diamond'))

    pygame.init()
    screen = pygame.display.set_mode(args.window)
    clock = pygame.time.Clock()
    running = True
    mem = {}

    while running and not done:
        facing_dir = env._direction
        # Rendering
        image = env.render(size)
        if size != args.window:
            image = Image.fromarray(image)
            image = image.resize(args.window, resample=Image.NEAREST)
            image = np.array(image)
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(args.fps)
        if info is None:
            action = env.action_space.sample()
        else:
            # Observation to Option and Action decision
            res = text_obs(info)
            print(f"obs: {res}")
            obs_to_option = ObservationToOption(res, info)
            option = obs_to_option.decide_option(mem)
            print(f"option: {option}")
            option_to_action = OptionToAction(info, facing_dir, mem)
            action = option_to_action.option_to_action(option)
            option_to_action.update_map_memory()
            mem = option_to_action.map_memory

        # Environment step
        if isinstance(action, int) : action = [action]
        print(action)
        obs, reward, done, info = None, None, None, None
        for a in action :
            obs, reward, done, info = env.step(a)

        opt = OptionToAction(info, facing_dir, mem)
        opt.update_map_memory()
        mem = opt.map_memory
        
        duration += 1

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

        # Episode end
        if done and not was_done:
            was_done = True
            print('Episode done!')
            print('Duration:', duration)
            print('Return:', return_)
            if args.death == 'quit':
                running = False
            if args.death == 'reset':
                print('\nStarting a new episode.')
                env.reset()
                achievements = set()
                was_done = False
                duration = 0
                return_ = 0
            if args.death == 'continue':
                pass

    pygame.quit()

if __name__ == '__main__':
    main()
