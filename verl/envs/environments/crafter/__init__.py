from .env import ACTIONS, CrafterLanguageWrapper
from .llm_agents_wrapper import CrafterLLMAgentsWrapper

ACTION_DICT = {
    "Noop": "do nothing",
    "Move West": "move west on flat ground",
    "Move East": "move east on flat ground",
    "Move North": "move north on flat ground",
    "Move South": "move south on flat ground",
    "Do": "Multiuse action to mine stone, gather materials, chop trees, drink from lakes, and attack any creature you face",
    "Sleep": "sleep when energy level is below maximum",
    "Place Stone": "place a stone in front, with 1 stone in inventory",
    "Place Table": "place a table in front, with 2 wood in inventory",
    "Place Furnace": "place a furnace in front, with 4 stone in inventory",
    "Place Plant": "place a plant in front, with 1 sampling in inventory",
    "Make Wood Pickaxe": "craft a wood pickaxe with a nearby table and 1 wood in inventory",
    "Make Stone Pickaxe": "craft a stone pickaxe with a nearby table, 1 wood, and 1 stone in inventory",
    "Make Iron Pickaxe": "craft an iron pickaxe with a nearby table and furnace, 1 wood, 1 coal, and 1 iron in inventory",
    "Make Wood Sword": "craft a wood sword with a nearby table and 1 wood in inventory",
    "Make Stone Sword": "craft a stone sword with a nearby table, 1 wood, and 1 stone in inventory",
    "Make Iron Sword": "craft an iron sword with a nearby table and furnace, 1 wood, 1 coal, and 1 iron in inventory",
}

ACHIEVEMENTS = [
    "Collect Wood",
    "Place Table",
    "Eat Cow",
    "Collect Sampling",
    "Collect Drink",
    "Make Wood Pickaxe",
    "Make Wood Sword",
    "Place Plant",
    "Defeat Zombie",
    "Collect Stone",
    "Place Stone",
    "Eat Plant",
    "Defeat Skeleton",
    "Make Stone Pickaxe",
    "Make Stone Sword",
    "Wake Up",
    "Place Furnace",
    "Collect Coal",
    "Collect Iron",
    "Make Iron Pickaxe",
    "Make Iron Sword",
    "Collect Diamond"
]

def get_instruction_prompt(task=None, info=None):
    
    if info is not None:
        achievements_list = info["achievements"]
        achievements_list = [key for key, value in achievements_list.items() if value == 0]
        achievements_list = [x.replace("_", " ").title() for x in achievements_list]
        achievements_list = [x for x in ACHIEVEMENTS if x in achievements_list]
    else:
        achievements_list = ACHIEVEMENTS
    achievements_text = ",\n".join(f"{i+1}. {achievement}" for i, achievement in enumerate(achievements_list))
    
    action_strings = ",\n".join(f"{action}: {ACTION_DICT[action]}" for action in ACTIONS)
    instruction_prompt = f"""
You are an agent playing Crafter. The following are the only valid actions you can take in the game, followed by a short description of each action:

{action_strings}.

These are the game achievements you can get:
{achievements_text}

In a moment I will present a history of actions and observations from the game.
Your goal is to get as far as possible by completing all the achievements.

These are the game rules:

1. Tree: Trees can be chopped down to collect wood, which is essential for crafting tools and weapons. While an axe makes chopping faster, you can also use your hands if necessary.
2. Cow: Killing a cow yields meat, which restores your health and hunger levels. Although you can attack it with your hands, using a sword or other weapon is more efficient.
3. Grass: Grass generally provides no resources, but occasionally it drops seeds for farming. You can interact with it freely without any tools.
4. Stone: Mining stone provides materials, i.e., stones, for crafting stronger tools and building furnaces. You need at least a wooden pickaxe to collect stone.
5. Path: Paths don’t provide any resources but can be walked on freely without requirements.
6. Coal: Coal can be mined and used to craft iron pickaxes and iron swords. You need at least a wooden pickaxe to collect coal.
7. Iron: Iron is crucial for crafting advanced tools and weapons. You need a stone pickaxe or better to mine iron.
8. Diamond: You need a iron pickaxe to mine diamond.
9. Crafting Table (Table): Allows you to craft various tools and weapons when standing next to it. You need 2 wood to build a crafting table if you don’t have one.
10. Furnace: Enables you to craft iron tools and weapons, when nearby.
11. Water: Drinking water restores your thirst level.
12. Zombie: Zombies are hostile and attack if you get too close. Using a sword is recommended, though you can fight them with your hands if needed.
13. Skeleton: Skeletons are hostile and attack from a distance with arrows. It’s best to use a sword to defeat them, but you can use your hands if necessary.
14. Plant: Plants can be placed to grow food, which restores your hunger level. You need seeds to plant them.

PLAY!
""".strip()

    return instruction_prompt
