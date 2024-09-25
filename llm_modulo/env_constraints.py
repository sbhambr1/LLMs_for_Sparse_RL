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
    3: 'right'}
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
        message = {"up":"","down":"","left":"","right":"","door":""}
        
        if agent_pos[0]==1: # agent is in the second row
            if agent_pos[1] == 1:
                actions.append("right")
                message = {"up":"wall","down":"wall","left":"wall","right":"","door":""}    
            elif agent_pos[1] in [2,3,4,6,8]:
                actions.append("right")
                actions.append("left")
                message = {"up":"wall","down":"wall","left":"","right":"","door":""}
            elif agent_pos[1] == 5:
                actions.append("right")
                actions.append("left")
                message = {"up":"wall","down":"worn_ladder","left":"","right":"","door":""}
            elif agent_pos[1] == 7:
                actions.append("right")
                actions.append("left")
                actions.append("down")
                message = {"up":"wall","down":"","left":"","right":"","door":""}
            elif agent_pos[1] == 9:
                actions.append("left")
                if not self.env.picked_key and not self.env.picked_hidden_key:
                    message = {"up":"wall","down":"wall","left":"","right":"wall","door":"both_keys"}
                elif not self.env.picked_key and self.env.picked_hidden_key:
                    message = {"up":"wall","down":"wall","left":"","right":"wall","door":"key"}
                elif self.env.picked_key and not self.env.picked_hidden_key:
                    message = {"up":"wall","down":"wall","left":"","right":"wall","door":"hidden_key"}
            # message.append("wall")    
        elif agent_pos[0] == 6:
            if agent_pos[1] in [6,7,8]:
                actions.append("right")
                actions.append("left")
                if agent_pos[1]==7:
                    message = {"up":"tube","down":"wall","left":"","right":"","door":""}
                else:
                    message = {"up":"wall","down":"wall","left":"","right":"","door":""}
            elif agent_pos[1] == 9:
                actions.append("left")
                message = {"up":"wall","down":"wall","left":"","right":"wall","door":""}
            elif agent_pos[1] in [2,3,4]:
                actions.append("right")
                actions.append("left")
                actions.append("up")
                message = {"up":"","down":"wall","left":"","right":"","door":""}
            elif agent_pos[1] == 1:
                actions.append("right")
                actions.append("up")
                message = {"up":"","down":"wall","left":"wall","right":"","door":""}
            elif agent_pos[1] == 5:
                actions.append("right")
                actions.append("left")
                actions.append("up")
                message = {"up":"","down":"wall","left":"","right":"","door":""}
        elif agent_pos[1] == 7:
            if agent_pos[0] in [2,3,4,5]:
                actions.append("down")
            message = {"up":"tube","down":"","left":"wall","right":"wall","door":""}
        elif agent_pos[1] == 5:
            if agent_pos[0] in [2, 3, 4, 5]:
                actions.append("up")
                message = {"up":"","down":"worn_ladder","left":"wall","right":"wall","door":""}
                # if self.env.grid[3, 5] == self.env.objects.ladder.id:
                #     actions.append("down")
                #     message = {"up":"worn_ladder","down":"","left":"wall","right":"wall","door":""}
                # else:
                    
            # elif agent_pos[0] == 3:
            #     message = {"up":"","down":"","left":"wall","right":"wall","door":""}
            #     if self.env.grid[2, 5] == self.env.objects.ladder.id:
            #         actions.append("up")
            #     else:
            #         message["up"] = "worn_ladder"
            #     if self.env.grid[4, 5] == self.env.objects.ladder.id:
            #         actions.append("down")
            #     else:
            #         message["down"] = "worn_ladder"
            # elif agent_pos[0] == 4:
            #     message = {"up":"","down":"","left":"wall","right":"wall","door":""}
            #     if self.env.grid[3, 5] == self.env.objects.ladder.id:
            #         actions.append("up")
            #     else:
            #         message["up"] = "worn_ladder"
            #     if self.env.grid[5, 5] == self.env.objects.ladder.id:
            #         actions.append("down")
            #     else:
            #         message["down"] = "worn_ladder"
            # elif agent_pos[0] == 5:  
            #     message = {"up":"","down":"","left":"wall","right":"wall","door":""}  
            #     if self.env.grid[4, 5] == self.env.objects.ladder.id:
            #         actions.append("up")
            #         message["down"] = "worn_ladder"
            #     else:
            #         actions.append("down")
            #         message["up"] = "worn_ladder"
        elif agent_pos[0]==5:
            if agent_pos[1] == 1:
                actions.append("down")
                actions.append("right")
                message = {"up":"wall","down":"","left":"wall","right":"","door":""}
            elif agent_pos[1] in [2,3,4]:
                actions.append("down")
                actions.append("left")
                actions.append("right")
                message = {"up":"wall","down":"","left":"","right":"","door":""}
        else:
            raise ValueError("Agent position not recognized.")
    
        return actions,message