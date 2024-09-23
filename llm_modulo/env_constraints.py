import numpy as np
OBJECT_TO_IDX = {
    "walkable_area": 0,
    "agent": 1,
    "wall": 2,
    "wood": 3,
    "workshop1": 4,
    "workshop2": 5,
    "workshop3": 6,
    "wood_process": 7,
}
ACTION_DICT = {
    0: 'up', 
    1: 'down', 
    2: 'left', 
    3: 'right'}
TEXT_ACTION_DICT = {
    'up': 0,
    'down': 1,
    'left': 2,
    'right': 3}
 
class EnvironmentConstraints:
    
    def __init__(self, env):
        self.env = env
       
    def get_agent_pos(self, observation):
        """
        Returns the agent position from the observation.
        Input: symbolic observation['image'] - (10x15)
        Output: agent position (x,y)
        """
        observation = observation[0:150].reshape(10,15)
        pos = np.where(observation == OBJECT_TO_IDX['agent']) # (x,y) based on the image of the state in the observation    
        agent_pos = (pos[0][0],pos[1][0])                   
        return agent_pos
        
        raise ValueError("Agent not found in the observation.")   
   
class MineCraft10x15(EnvironmentConstraints):
    
    def __init__(self, env):
        self.env = env
        
    def feasible_actions(self, agent_pos, action_history):
        """
        Returns the feasible actions in the current state.
        Input: agent_pos (x,y) from symbolic obs, agent_dir, valid actions taken by the agent
        Output: list of actions
        """       
        actions = [] # actions that are always valid
        message = {"up":"","down":"","left":"","right":""}
        
        if agent_pos==(1,1):
            actions.extend(["right","down"])
            message = {"up":"wall","down":"","left":"wall","right":""}
        elif agent_pos==(1,13):
            actions.extend(["down","left"])
            message = {"up":"wall","down":"","left":"","right":"wall"}
        elif agent_pos==(8,1):
            actions.extend(["up","right"])
            message = {"up":"","down":"wall","left":"wall","right":""}
        elif agent_pos==(8,13):
            actions.extend(["up","left"])
            message = {"up":"","down":"wall","left":"","right":"wall"}
        elif agent_pos[1] == 1:
            if agent_pos[0] in [2,3,4,5,6,7]:
                actions.extend(["up","right","down"])
                message = {"up":"","down":"","left":"wall","right":""}
        elif agent_pos[1] == 13:
            if agent_pos[0] in [2,3,4,5,6,7]:
                actions.extend(["up","left","down"])
                message = {"up":"","down":"","left":"","right":"wall"}
        else:
            actions.extend(["up","right","down","left"])
            if agent_pos == (7,1): # at workshop_1
                if self.env.is_stick_made:
                    message['extra'] = "You are at workshop-1. And you have already made sticks."
                else:
                    if self.env.n_processed_wood == 0:
                        message['extra'] = "You are at workshop-1. And you can not make sticks because you do not have processed sticks."
            elif agent_pos == (7,3): # at workshop_2
                if self.env.is_plank_made:
                    message['extra'] = "You are at workshop-2. And you have already made planks."
                else:
                    if self.env.n_processed_wood == 0:
                        message['extra'] = "You are at workshop-2. And you can not make planks because you do not have processed sticks."
            elif agent_pos == (7,5): # at workshop_3
                if not self.env.is_stick_made and not self.env.is_plank_made:
                    message['extra'] = "You are at workshop-3. And you can not make ladder because you do not have sticks and planks with you."
                elif self.env.is_stick_made and not self.env.is_plank_made:
                    message['extra'] = "You are at workshop-3 and you have sticks with you. But you can not make ladder because you do not have planks with you."
                elif not self.env.is_stick_made and self.env.is_plank_made:
                    message['extra'] = "You are at workshop-3 and you have planks with you. But you can not make ladder because you do not have sticks with you."
            elif agent_pos == (7,8): # at processing_unit
                if self.env.all_wood_collected:
                    message['extra'] = "You are at wood processing unit. And you have alreaady processed all the woods."
                else:
                    if self.carry_list['wood'] == 0:
                        message['extra'] = "You are at wood processing unit. But you can not process wood because you do not have any raw wood blocks."
            #raise ValueError("Agent position not recognized.")
    
        return actions,message