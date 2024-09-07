from .textual_disc import OBJECT_TO_IDX, IDX_TO_OBJECT
import heapq
import random

def calc_dis(x, y, mx, my):
    return abs(x - mx) + abs(y - my)

class ObservationToOption:
    def __init__(self, observation, info):
        self.environment, self.inventory = self.parse_observation(observation)
        self.observation = observation
        self.info = info

    def parse_observation(self, observation):
        if "You have" in observation:
            environment_part, rest = observation.split("You have")
            inventory_part = rest.split("You can")[0].strip()
        else:
            environment_part = observation
            inventory_part = ""

        environment = [item.strip() for item in environment_part.replace("Agent sees", "").replace("You are", "").split(",")]
        inventory = {}
        if inventory_part:
            inventory_items = inventory_part.replace(".", "").strip().split(", ")
            for item in inventory_items:
                parts = item.split(" ", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    count = int(parts[0])
                    name = parts[1]
                    inventory[name] = count
                else:
                    inventory[item] = 1


        return environment, inventory

    def check_needs(self, mem):
        # print("env = ", self.environment)
        if "thirsty" in self.observation:
            if OBJECT_TO_IDX["water"] in mem.values():
                return "drink water"
            else:
                return "explore"
        if "hungry" in self.observation:
            if OBJECT_TO_IDX["cow"] in mem.values():
                return "eat cow"
            else:
                return "explore"
        if "sleepy" in self.observation:
            return "sleep"
        return None

    def detect_enemies(self, mem):
        enemies = {"zombie": "attack zombie", "skeleton": "attack skeleton"}
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        player_pos = self.info["player_pos"]
        
        for dx, dy in directions:
            for step in range(1, 2):  # 检查四格范围内
                check_pos = (player_pos[0] + dx * step, player_pos[1] + dy * step)
                if check_pos[0] in range(0, 64) and check_pos[1] in range(0, 64) :
                    item = IDX_TO_OBJECT[self.info["semantic"][check_pos[0]][check_pos[1]]]
                    if item in enemies:
                        return enemies[item]
        return None

    def resource_collection(self, mem):
        wood_count = self.inventory.get("wood", 0)
        stone_count = self.inventory.get("stone", 0)

        # print("We have", self.inventory)

        has_workbench = OBJECT_TO_IDX["table"] in mem.values()
        has_pickaxe = "wood_pickaxe" in self.inventory
        has_stone_pickaxe = "stone_pickaxe" in self.inventory
        
        assert not has_stone_pickaxe, "Successfully make stone pickaxe!"

        # 假设 environment 表示 8x8 视野范围
        if wood_count < 4 and not has_workbench:
            if OBJECT_TO_IDX["tree"] in mem.values():
                return "chop tree"
            else:
                return "explore"
        elif "stone" in self.environment and not has_workbench:  # 视野内有石头时制作table
            return "build table"
        elif has_workbench and not has_pickaxe:
            return "craft wood_pickaxe"
        elif has_pickaxe:
            if stone_count == 0:
                if OBJECT_TO_IDX["stone"] in mem.values():
                    return "get stone"
                else:
                    return "explore"
            elif stone_count > 0 and wood_count > 0:
                return "craft stone_pickaxe"
        return "explore"

    def decide_option(self, mem):
        option = self.detect_enemies(mem)
        if option:
            return option

        option = self.check_needs(mem)
        if option:
            return option
        option = self.resource_collection(mem)
        if option:
            return option
        
        # 如果没有其他选项，探索随机方向
        return "explore"

class OptionToAction:
    def __init__(self, info, env_direction, mem):
        self.grid = info["semantic"]
        self.info = info
        self.direction = env_direction
        self.action_names = ['noop', 'move_left', 'move_right', 'move_up', 'move_down', 'do', 'sleep', 
                             'place_stone', 'place_table', 'place_furnace', 'place_plant', 
                             'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe', 
                             'make_wood_sword', 'make_stone_sword', 'make_iron_sword']
        self.target_position = None
        self.map_memory = mem  # 记录已探索的地图

    def is_passable(self, x, y, debug = True) :
        if IDX_TO_OBJECT[self.grid[x][y]] == "grass" or IDX_TO_OBJECT[self.grid[x][y]] == "path" or IDX_TO_OBJECT[self.grid[x][y]] == "sand" :
            return True
        else :
            return False

    def update_map_memory(self):
        player_pos = self.info["player_pos"]
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                x, y = player_pos[0] + dx, player_pos[1] + dy
                if 0 <= x < len(self.grid) and 0 <= y < len(self.grid[0]):
                    self.map_memory[(x, y)] = self.grid[x][y]

    def explore(self) :
        epsilon = 0.1  # 探索的概率
        directions = {
            "move_up": (0, -1),
            "move_down": (0, 1),
            "move_left": (-1, 0),
            "move_right": (1, 0)
        }

        if random.random() < epsilon:
            # 随机选择一个方向
            random_direction = random.choice(list(directions.keys()))
            return [self.action_names.index(random_direction)]

        current_pos = self.info["player_pos"]
        max_new_explored = 0
        best_directions = []

        for action, (dx, dy) in directions.items():
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            if not self.is_passable(new_pos[0], new_pos[1], debug = False) : continue
            new_explored_count = 0

            for x in range(new_pos[0] - 4, new_pos[0] + 5):
                for y in range(new_pos[1] - 4, new_pos[1] + 5):
                    if (x, y) not in self.map_memory:
                        new_explored_count += 1

            if new_explored_count > max_new_explored:
                max_new_explored = new_explored_count
                best_directions = [action]
            elif new_explored_count == max_new_explored:
                best_directions.append(action)

        if best_directions:
            chosen_direction = random.choice(best_directions)
            return [self.action_names.index(chosen_direction)]
        else:
            random_direction = random.choice(list(directions.keys()))
            return [self.action_names.index(random_direction)]
        

    def find_unexplored_direction(self, start):
        directions = {
            "move_up": (0, -1),
            "move_down": (0, 1),
            "move_left": (-1, 0),
            "move_right": (1, 0)
        }
        for action, (dx, dy) in directions.items():
            neighbor = (start[0] + dx, start[1] + dy)
            if neighbor not in self.map_memory:
                return action
        return None

    def a_star_find_path_to_target(self, start, goal):
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        start = tuple(start)
        goal = tuple(goal)

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        directions = {
            "move_up": (0, -1),
            "move_down": (0, 1),
            "move_left": (-1, 0),
            "move_right": (1, 0)
        }

        def is_goal_reached(current, goal):
            # Check if current is adjacent to goal
            return heuristic(current, goal) == 1

        while open_set:
            _, current = heapq.heappop(open_set)

            if is_goal_reached(current, goal):
                path = self.reconstruct_path(came_from, current)
                # Add final move towards goal
                final_move = self.get_facing_direction(goal, current)
                path.append(final_move)
                return path

            for action, (dx, dy) in directions.items():
                neighbor = (current[0] + dx, current[1] + dy)
                tentative_g_score = g_score[current] + 1

                if (0 <= neighbor[0] < 64) and (0 <= neighbor[1] < 64):
                    if not self.is_passable(neighbor[0], neighbor[1]):
                        continue

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = (current, action)
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        ## IF cannot go to goal, explore.
        return self.explore()


    def get_facing_direction(self, current, previous):
        direction_map = {
            (0, -1): "move_up",
            (0, 1): "move_down",
            (-1, 0): "move_left",
            (1, 0): "move_right"
        }
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        return self.action_names.index(direction_map.get((dx, dy)))

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            current, action = came_from[current]
            path.append(self.action_names.index(action))
        path.reverse()
        return path

    def option_to_action(self, option, update = False):
        self.update_map_memory()  # 更新地图记忆
        current_pos = self.info["player_pos"]

        if option.startswith("go to"):
            target = option.split("go to ")[1]
            start_position = self.info["player_pos"]

            goal_position = self.find_goal_position(target)
            if goal_position is None : return self.explore()
            
            self.target_position = goal_position
            path = self.a_star_find_path_to_target(start_position, goal_position)
            if path:
                return path
            else:
                return [self.action_names.index("noop")]

        elif option == "explore":
            return self.explore()

        elif option in ["chop tree", "get stone", "get stone with stone_pickaxe", "attack skeleton", "attack zombie", "drink water", "eat cow"]:
            if self.target_position is None:
                target = option.split(" ")[-1]
                target_pos = self.find_goal_position(target)
                if target_pos is None : return self.explore()
                self.target_position = target_pos
                
            if self.is_facing_target(self.target_position, option):
                if option == "drink water" : return [self.action_names.index("do")] * 5
                return [self.action_names.index("do")]
            else:
                path = self.a_star_find_path_to_target(self.info["player_pos"], self.target_position)
                if path:
                    return path
                else:
                    return [self.action_names.index("noop")]

        elif option.startswith("craft"):
            workbench_position = self.find_goal_position("table")
            if workbench_position is None : return self.explore()
            if self.is_facing_target(workbench_position, option):
                item = option.split("craft ")[1]
                return [self.action_names.index("make_" + item)]
            else : 
                path = self.a_star_find_path_to_target(self.info["player_pos"], workbench_position)
                return path

        elif option.startswith("build"):
            structure = option.split("build ")[1]
            directions = {
                "move_left": (-1, 0),
                "move_right": (1, 0),
                "move_up": (0, -1),
                "move_down": (0, 1)
            }

            for direction, (dx, dy) in directions.items():
                new_pos = (self.info["player_pos"][0] + dx, self.info["player_pos"][1] + dy)

                if self.is_passable(new_pos[0], new_pos[1]):
                    return [self.action_names.index("do"), self.action_names.index("place_" + structure)]

            # 如果没有找到可放置的位置，随机移动
            random_direction = random.choice(list(directions.keys()))
            return [self.action_names.index("do"), self.action_names.index(random_direction)]

        else:
            return [self.action_names.index(option)]

    def is_facing_target(self, target_position, option):
        player_pos = self.info["player_pos"]

        dx = target_position[0] - player_pos[0]
        dy = target_position[1] - player_pos[1]

        if self.direction == 'north' and dx == 0 and dy == -1:
            return True
        elif self.direction == 'south' and dx == 0 and dy == 1:
            return True
        elif self.direction == 'west' and dy == 0 and dx == -1:
            return True
        elif self.direction == 'east' and dy == 0 and dx == 1:
            return True
        return False

    def find_goal_position(self, target):
        def calc_dis(x, y, mx, my):
            return abs(x - mx) + abs(y - my)

        player_pos = self.info["player_pos"]
        min_dis = float('inf')
        min_pos = None

        for pos, obj in self.map_memory.items():
            if IDX_TO_OBJECT[obj] == target:
                dist = calc_dis(player_pos[0], player_pos[1], pos[0], pos[1])
                if dist < min_dis:
                    min_dis = dist
                    min_pos = pos

        # assert min_pos is not None, f"Cannot find target {target}!"
        return min_pos
