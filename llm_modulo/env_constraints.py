import numpy as np

OBJECT_TO_IDX = {
    "walkable_area": 0,
    "agent": 1,
    "wall": 2,
    "tube": 3,
    "ladder": 4,
    "worn_ladder": 5,
    "hidden_key": 6,
    "key": 7,
    "door": 8,
    "dead_agent": 9}
ACTION_DICT = {
    0: 'up', 
    1: 'down', 
    2: 'left', 
    3: 'riight'}
TEXT_ACTION_DICT = {
    'up': 0,
    'down': 1,
    'left': 2,
    'right': 3}
 ##################### 8 x 11 #########################
#     0   1   2   3   4   5   6   7   8   9   10  
#
# 0    2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
#------------------------------------------------------
# 1    2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 2 |
#------------------------------------------------------
# 2    2 | 2 | 2 | 2 | 2 | 4 | 2 | 3 | 2 | 2 | 2 |
#------------------------------------------------------
# 3    2 | 2 | 2 | 2 | 2 | 4 | 2 | 3 | 2 | 2 | 2 |
#------------------------------------------------------
# 4    2 | 2 | 2 | 2 | 2 | 4 | 2 | 3 | 2 | 2 | 2 |
#------------------------------------------------------
# 5    2 | 7 | 2 | 6 | 2 | 4 | 2 | 3 | 2 | 2 | 2 |
#------------------------------------------------------
# 6    2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
#------------------------------------------------------
# 7    2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
 ###################################################### 

 class EnvironmentConstraints:
    
    def __init__(self, env):
        self.env = env
       
    def get_agent_pos(self, observation):
        """
        Returns the agent position from the observation.
        Input: symbolic observation['image'] - (8x11)
        Output: agent position (x,y)
        """
        observation = observation.reshape(8,11)
        pos = np.where(observation == OBJECT_TO_IDX['agent']) # (x,y) based on the image of the state in the observation    
        agent_pos = (pos[0][0],pos[1][0])                   
        return agent_pos
        
        raise ValueError("Agent not found in the observation.")   
   
class Mario8x11(EnvironmentConstraints):
    
    def __init__(self, env):
        self.env = env
        
    def feasible_actions(self, agent_pos, action_history):
        """
        Returns the feasible actions in the current state.
        Input: agent_pos (x,y) from symbolic obs, agent_dir, valid actions taken by the agent
        Output: list of actions
        """       
        actions = [] # actions that are always valid
        
        if agent_pos[0]==1: # agent is in the second row
           if agent_pos[1] == 1:
               actions.append("right")
           elif agent_pos[1] in [2,3,4,6,8]:
               acction.extend(["right","left"])
           elif agent_pos[1] in [5,7]:
               action.extend(["right","left","down"])
        elif agent_pos[0] == 6:
            if agent_pos[1] in [2,4,6,7,8]:
                action.extend(["right","left"])
            elif agent_pos[1] == 9:
                action.append("left")
            elif agent_pos[1] == 3:
                action.extend(["right","left","up"])
            elif agent_pos[1] == 1:
                action.extend(["right","up"])
            elif agent_pos[1] == 5:
                action.extend(["right","left"])
                if env.grid[5, 5] == env.objects.ladder.id:
                    action.append("up")
        elif agent_pos[1] == 7:
            if agent_pos[0] in [2,3,4,5]:
                action.append("down")
        elif agent_pos[1] == 5:
            if agent_pos[0] == 2:
                if env.grid[3, 5] == env.objects.ladder.id:
                    action.append("down")
                else:
                    action.append("up")
            elif agent_pos[0] == 3:
                if env.grid[2, 5] == env.objects.ladder.id:
                    action.append("up")
                if env.grid[4, 5] == env.objects.ladder.id:
                    action.append("down")
            elif agent_pos[0] == 4:
                if env.grid[3, 5] == env.objects.ladder.id:
                    action.append("up")
                if env.grid[5, 5] == env.objects.ladder.id:
                    action.append("down")
            elif agent_pos[0] == 5:    
                if env.grid[4, 5] == env.objects.ladder.id:
                    action.append("up")
                else:
                    action.append("down")
        elif agent_pos[0]==5:
            if agent_pos[1] in [1,3]:
                action.append("down")
        else:
            raise ValueError("Agent position not recognized.")
    
        return actions