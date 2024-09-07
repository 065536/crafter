import tiktoken

GAME_DESCRIPTION = {
    "make_stone_pickaxe": {
        "description": "You are a game agent in the Crafter environment. Your goal is to make a stone pickaxe. First, you need to collect four woods. Next, you need to build a table to make a wood pickaxe. Then, you should use the wood pickaxe to get a stone. Finally, you should get back to the table and make the stone pickaxe.",
        "options": [
            "attack zombie",
            "attack skeleton",
            "drink water",
            "eat cow",
            "sleep",
            "chop tree",
            "get stone",
            "craft wood_pickaxe",
            "craft stone_pickaxe",
            "build table",
            "explore"
        ]
    },
    
    "eat_cow": {
        "description": "You are a game agent in the Crafter environment. You need to find a cow in the environment and eat it.",
        "options": [
            "attack zombie",
            "attack skeleton",
            "drink water",
            "eat cow",
            "sleep",
            "chop tree",
            "get stone",
            "craft wood_pickaxe",
            "craft stone_pickaxe",
            "build table",
            "explore"
        ]
    }
}

def gen_prompt(text_desc, task):
    task_dic = GAME_DESCRIPTION[task]
    prompt = task_dic["description"]
    prompt += text_desc
    options_str = ", ".join(task_dic["options"])
    prompt += f"Choose an rational option from [{options_str}]."
    return prompt

def check_task_done(env, task):
    achievements = env._player.achievements
    
    task_to_achievement = {
        "eat_cow": "eat_cow",
        "make_stone_pickaxe": "make_stone_pickaxe",
        "collect_wood": "collect_wood",
        "defeat_zombie": "defeat_zombie",
        "defeat_skeleton": "defeat_skeleton",
        "place_table": "place_table",
    }

    if task in task_to_achievement:
        achievement = task_to_achievement[task]
        if achievements.get(achievement, 0) > 0:
            return True
        else:
            return False
    else:
        raise ValueError(f"Task '{task}' is not recognized.")
    

def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)
