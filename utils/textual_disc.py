IDX_TO_OBJECT = {
    1 : "water",
    2 : "grass",
    3 : "stone",
    4 : "path",
    5 : "sand",
    6 : "tree",
    7 : "lava",
    8 : "coal",
    9 : "iron",
    10 : "diamond",
    11 : "table", 
    12 : "furnace",
    13 : "player",
    14 : "cow",
    15 : "zombie",
    16 : "skeleton",
    17 : "arrow",
    18 : "plant"
}

OBJECT_TO_IDX = {
    "water": 1,
    "grass": 2,
    "stone": 3,
    "path": 4,
    "sand": 5,
    "tree": 6,
    "lava": 7,
    "coal": 8,
    "iron": 9,
    "diamond": 10,
    "table": 11,
    "furnace": 12,
    "player": 13,
    "cow": 14,
    "zombie": 15,
    "skeleton": 16,
    "arrow": 17,
    "plant": 18
}

PRE_REQ = { # values are [inv_items], [world_items]
        'eat plant': ([], ['plant']),
        'attack zombie': ([], ['zombie']),
        'attack skeleton': ([], ['skeleton']),
        'attack cow': ([], ['cow']),
        'eat cow': ([], ['cow']),
        'chop tree': ([], ['tree']),
        'mine stone': (['wood_pickaxe'], ['stone']),
        'mine coal': (['wood_pickaxe'], ['coal']),
        'mine iron': (['stone_pickaxe'], ['iron']),
        'mine diamond': (['iron_pickaxe'], ['diamond']),
        'drink water': ([], ['water']),
        'chop grass': ([], ['grass']),
        'sleep': ([], []),
        'place stone': (['stone'], []),
        'place crafting table': (['wood'], []),
        'make crafting table': (['wood'], []),
        'place furnace': (['stone'], []),
        'place plant': (['sapling'], []),
        'make wood pickaxe': (['wood'], ['table']),
        'make stone pickaxe': (['stone', 'wood'], ['table']),
        # 'make iron pickaxe': (['wood', 'coal', 'iron'], ['table', 'furnace']),
        'make wood sword': (['wood'], ['table']),
        'make stone sword': (['wood', 'stone'], ['table']),
        # 'make iron sword': (['wood', 'coal', 'iron'], ['table', 'furnace']),
    }


def text_obs(info) :
    
    text = "Agent sees "
    object = ""

    x, y = info["player_pos"]
    map = info["semantic"]
    inv = info["inventory"]
    view = dict()

    object_list = []

    # process obs
    for mx in range(max(0, x - 4), min(64, x + 5)) :
        for my in range(max(0, y - 3), min(64, y + 4)) :
            if not map[mx][my] in view.keys() and map[mx][my] != 13:
                view[map[mx][my]] = 1
                object_list.append(IDX_TO_OBJECT[map[mx][my]])

    
    for idx in range(len(object_list)) :
        if idx < len(object_list) - 1 : object += object_list[idx] + ", "
        else : object += object_list[idx] + ". "

    # process state
    state = ""
    # state = f"Your HP is {inv["health"]}, your food point is {inv["food"]}, your energy is {inv["energy"]}, and your drink point is {inv["drink"]}"
    if inv["food"] < 3 : state += "You are hungry. "
    if inv["drink"] < 3 : state += "You are thirsty. "
    if inv["energy"] < 3 : state += "You are tired. "

    # process inv
    items = []
    for name, num in inv.items() :
        if name != "health" and name != "drink" and name != "food" and name != "energy" :
            if num > 0 :
                items.append((name, num))
    
    hold = "You have "
    for idx in range(len(items)) :
        if idx < len(items) - 1 : hold += f"{items[idx][1]} {items[idx][0]}, "
        else : hold += f"{items[idx][1]} {items[idx][0]}."

    if len(items) <= 0 : hold = ""

    actions = []
    action_text = "You can "

    for action, pre in PRE_REQ.items() :
        flag = True
        for item in pre[0] :
            if "crafting table" in action :
                if inv["wood"] < 2 : flag = False
            if "furnace" in action :
                if inv["stone"] < 4 : flag = False

            if inv[item] < 1 :
                flag = False
                break
        
        for block in pre[1] :
            if block not in object_list :
                flag = False
                break

        if flag :
            actions.append(action)

    for idx in range(len(actions)) :
        if idx < len(actions) - 1 : action_text += f"{actions[idx]}, "
        else : action_text += f"{actions[idx]}. "

    result = text + object + state + hold + "\n" + action_text
    return result
