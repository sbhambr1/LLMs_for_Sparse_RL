OBJECT_TO_IDX = {
    "unseen": -1,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}
ACTION_DICT = {
    0: 'turn left', 
    1: 'turn right', 
    2: 'move forward', 
    3: 'pickup key', 
    4: 'drop', 
    5: 'open door', 
    6: 'done'}
TEXT_ACTION_DICT = {
    'turn left': 0,
    'turn right': 1,
    'move forward': 2,
    'pickup key': 3,
    'drop': 4,
    'open door': 5,
    'done': 6}
DIRECTION_DICT = {
    'right': 0,
    'down': 1,
    'left': 2,
    'up': 3}

class EnvironmentConstraints:
    
    def __init__(self, env, seed):
        self.env = env
        self.seed = seed
       
    def get_agent_pos(self, observation):
        """
        Returns the agent position from the observation.
        Input: symbolic observation['image']
        Output: agent position
        """
        agent_obs = observation['image']
        agent_dir = observation['direction']
        for i in range(1,6):
            for j in range(3):
                if agent_obs[i][j][2] == OBJECT_TO_IDX['agent']:
                    agent_pos = (j, i) # (x,y) based on the image of the state in the observation                       
                    return agent_pos, agent_dir
        
        raise ValueError("Agent not found in the observation.") 
        
class DoorKey5x5(EnvironmentConstraints):
    
    def __init__(self, env, seed):
        super().__init__(env, seed)
        
    def feasible_actions(self, agent_pos, agent_dir, action_history):
        """
        Returns the feasible actions in the current state.
        Input: agent_pos (x,y) from symbolic obs, agent_dir, valid actions taken by the agent
        Output: list of actions
        """       
        actions = ['turn left', 'turn right'] # actions that are always valid
        
        if self.seed == 0:
            if agent_pos[0]==1: # agent is in the second row
                if agent_pos[1]==1:
                    # agent has picked key already, can open door if facing it, and can move forward if door is opened
                    if agent_dir == DIRECTION_DICT['right']:
                        for j in range(len(action_history)):
                            if action_history[j] == 'open door':
                                actions.append('move forward')
                                break
                        actions.append('open door')
                    elif agent_dir == DIRECTION_DICT['down']: # agent is facing down (wrong direction)
                        actions.append('move forward')
                elif agent_pos[1]==2:
                    # agent has opened door and moved forward to door cell, can move forward if facing right (correct) or left (wrong) direction
                    if agent_dir == DIRECTION_DICT['right'] or agent_dir == DIRECTION_DICT['left']:
                        actions.append('move forward')
                elif agent_pos[1]==3:
                    # agent has moved forward, can move forward if facing down (correct) or left (wrong) direction
                    if agent_dir == DIRECTION_DICT['down'] or agent_dir == DIRECTION_DICT['left']:
                        actions.append('move forward')
                    pass
            elif agent_pos[0]==2: # agent is in the third row
                if agent_pos[1]==1 or agent_pos[1]==3:
                    # agent has moved forward, can move forward if facing up or down
                    if agent_dir == DIRECTION_DICT['down'] or agent_dir == DIRECTION_DICT['up']:
                        actions.append('move forward')
            elif agent_pos[0]==3: # agent is in the fourth row
                if agent_pos[1]==1:
                    # agent is in the initial state, can pickup key if facing up (correct) direction, and move forward facing up if key is picked
                    if agent_dir == DIRECTION_DICT['up']:
                        for j in range(len(action_history)):
                            if action_history[j] == 'pickup key':
                                actions.append('move forward')
                                break
                        actions.append('pickup key')
                    
            else:
                raise ValueError("Agent position not recognized.")
        
            return actions
        
        elif self.seed == 1:
            if agent_pos[0]==1:
                if agent_pos[1]==1 or agent_pos[1]==3:
                    if agent_dir == DIRECTION_DICT['down']:
                        actions.append('move forward')
            elif agent_pos[0]==2:
                if agent_pos[1]==1:
                    if agent_dir == DIRECTION_DICT['up']:
                        actions.append('move forward')
                    elif agent_dir == DIRECTION_DICT['right']:
                        for j in range(len(action_history)):
                            if action_history[j] == 'open door':
                                actions.append('move forward')
                                break
                        actions.append('open door')
                    elif agent_dir == DIRECTION_DICT['down']:
                        for j in range(len(action_history)):
                            if action_history[j] == 'pickup key':
                                actions.append('move forward')
                                break
                        actions.append('pickup key')
                elif agent_pos[1]==2:
                    if agent_dir == DIRECTION_DICT['right'] or agent_dir == DIRECTION_DICT['left']:
                        actions.append('move forward')
                elif agent_pos[1]==3:
                    if agent_dir == DIRECTION_DICT['left'] or agent_dir == DIRECTION_DICT['up'] or agent_dir == DIRECTION_DICT['down']:
                        actions.append('move forward')
            elif agent_pos[0]==3:
                if agent_pos[1]==1:
                    if agent_dir == DIRECTION_DICT['up']:
                        actions.append('move forward')
                        
            else:
                raise ValueError("Agent position not recognized.")
        
            return actions
        
        elif self.seed == 2:
            if agent_pos[0]==1:
                if agent_pos[1]==1:
                    if agent_dir == DIRECTION_DICT['right']:
                        for j in range(len(action_history)):
                            if action_history[j] == 'open door':
                                actions.append('move forward')
                                break
                        actions.append('open door')
                    elif agent_dir == DIRECTION_DICT['down']:
                        actions.append('move forward')
                elif agent_pos[1]==2:
                    if agent_dir == DIRECTION_DICT['right'] or agent_dir == DIRECTION_DICT['left']:
                        actions.append('move forward')
                elif agent_pos[1]==3:
                    if agent_dir == DIRECTION_DICT['down'] or agent_dir == DIRECTION_DICT['left']:
                        actions.append('move forward')
            elif agent_pos[0]==2:
                if agent_pos[1]==1:
                    if agent_dir == DIRECTION_DICT['up']:
                        actions.append('move forward')
                    elif agent_dir == DIRECTION_DICT['down']:
                        for j in range(len(action_history)):
                            if action_history[j] == 'pickup key':
                                actions.append('move forward')
                                break
                        actions.append('pickup key')
                elif agent_pos[1]==3:
                    if agent_dir == DIRECTION_DICT['up'] or agent_dir == DIRECTION_DICT['down']:
                        actions.append('move forward')
            elif agent_pos[0]==3:
                if agent_pos[1]==1:
                    if agent_dir == DIRECTION_DICT['up']:
                        actions.append('move forward')
            
            else:
                raise ValueError("Agent position not recognized.")
        
            return actions
        
        else:
            raise NotImplementedError("Seed not implemented.")
        
    
class EmptyRandom5x5(EnvironmentConstraints):
    
    def __init__(self, env, seed):
        super().__init__(env, seed)
        
    def feasible_actions(self, agent_pos, agent_dir, action_history):
        """
        Returns the feasible actions in the current state.
        Input: agent_pos (x,y) from symbolic obs, agent_dir, valid actions taken by the agent
        Output: list of actions
        (All seeds have the same constraints, as the agent's initial position is randomized but the goal location is fixed.)
        """
        actions = ['turn left', 'turn right'] # actions that are always valid
        
        if agent_pos[0]==1:
            if agent_pos[1]==1:
                if agent_dir == DIRECTION_DICT['right'] or agent_dir == DIRECTION_DICT['down']:
                    actions.append('move forward')
            elif agent_pos[1]==2:
                if agent_dir == DIRECTION_DICT['right'] or agent_dir == DIRECTION_DICT['left'] or agent_dir == DIRECTION_DICT['down']:
                    actions.append('move forward')
            elif agent_pos[1]==3:
                if agent_dir == DIRECTION_DICT['down'] or agent_dir == DIRECTION_DICT['left']:
                    actions.append('move forward')
        elif agent_pos[0]==2:
            if agent_pos[1]==1:
                if agent_dir == DIRECTION_DICT['up'] or agent_dir == DIRECTION_DICT['right'] or agent_dir == DIRECTION_DICT['down']:
                    actions.append('move forward')
            elif agent_pos[1]==2:
                actions.append('move forward')
            elif agent_pos[1]==3:
                if agent_dir == DIRECTION_DICT['up'] or agent_dir == DIRECTION_DICT['left'] or agent_dir == DIRECTION_DICT['down']:
                    actions.append('move forward')
        elif agent_pos[0]==3:
            if agent_pos[1]==1:
                if agent_dir == DIRECTION_DICT['up'] or agent_dir == DIRECTION_DICT['right']:
                    actions.append('move forward')
            elif agent_pos[1]==2:
                if agent_dir == DIRECTION_DICT['up'] or agent_dir == DIRECTION_DICT['right'] or agent_dir == DIRECTION_DICT['left']:
                    actions.append('move forward')
        
        else:
            raise ValueError("Agent position not recognized.")
        
        return actions
    

class LavaGapS5(EnvironmentConstraints):
    
    def __init__(self, env, seed):
        super().__init__(env, seed)
        
    def feasible_actions(self, agent_pos, agent_dir, action_history):
        """
        Returns the feasible actions in the current state.
        Input: agent_pos (x,y) from symbolic obs, agent_dir, valid actions taken by the agent
        Output: list of actions
        """
        actions = ['turn left', 'turn right'] # actions that are always valid
        
        if self.seed == 0:
            if agent_pos[0]==1:
                if agent_pos[1]==1 or agent_pos[1]==3:
                    if agent_dir == DIRECTION_DICT['down']:
                        actions.append('move forward')
            elif agent_pos[0]==2:
                if agent_pos[1]==1 or agent_pos[1]==3:
                    if agent_dir == DIRECTION_DICT['down'] or agent_dir == DIRECTION_DICT['up']:
                        actions.append('move forward')
            elif agent_pos[0]==3:
                if agent_pos[1]==1:
                    if agent_dir == DIRECTION_DICT['up'] or agent_dir == DIRECTION_DICT['right']:
                        actions.append('move forward')
                elif agent_pos[1]==2:
                    if agent_dir == DIRECTION_DICT['left'] or agent_dir == DIRECTION_DICT['right']:
                        actions.append('move forward')
            else:
                raise ValueError("Agent position not recognized.")
        
        elif self.seed == 1:
            if agent_pos[0]==1:
                if agent_pos[1]==1 or agent_pos[1]==3:
                    if agent_dir == DIRECTION_DICT['down']:
                        actions.append('move forward')
            elif agent_pos[0]==2:
                if agent_pos[1]==1 or agent_pos[1]==3:
                    if agent_dir == DIRECTION_DICT['down'] or agent_dir == DIRECTION_DICT['up']:
                        actions.append('move forward')
                elif agent_pos[1]==2:
                    if agent_dir == DIRECTION_DICT['left'] or agent_dir == DIRECTION_DICT['right']:
                        actions.append('move forward')
            elif agent_pos[0]==3:
                if agent_pos[1]==1:
                    if agent_dir == DIRECTION_DICT['up']:
                        actions.append('move forward')
            else:
                raise ValueError("Agent position not recognized.")
        
        else:
            raise NotImplementedError("Seed not implemented.")
        
        return actions
    
class KeyCorridorS3R1(EnvironmentConstraints):
    
    def __init__(self, env, seed):
        super().__init__(env, seed)
        
    def feasible_actions(self, agent_pos, agent_dir, action_history):
        """
        Returns the feasible actions in the current state.
        Input: agent_pos (x,y) from symbolic obs, agent_dir, valid actions taken by the agent
        Output: list of actions
        (All seeds have the same constraints, as the door and key colors are randomized but the goal location and configuration is fixed.)
        """
        actions = ['turn left', 'turn right'] # actions that are always valid
        if agent_pos[0]==1:
            if agent_pos[1]==1:
                if agent_dir == DIRECTION_DICT['right']:
                    actions.append('move forward')
            elif agent_pos[1]==2:
                if agent_dir == DIRECTION_DICT['right']:
                    actions.append('move forward')
                elif agent_dir == DIRECTION_DICT['left']:
                    for j in range(len(action_history)):
                        if action_history[j] == 'pickup key':
                            actions.append('move forward')
                            break
                    actions.append('pickup key')
            elif agent_pos[1]==3:
                if agent_dir == DIRECTION_DICT['left']:
                    for j in range(len(action_history)):
                        if action_history[j] == 'open door':
                            actions.append('move forward')
                            break
                    actions.append('open door')
                elif agent_dir == DIRECTION_DICT['right']:
                    for j in range(len(action_history)):
                        if action_history[j] == 'pickup key':
                            open_door_count = 0
                            for k in range(j, len(action_history)):
                                if action_history[k] == 'open door':
                                    open_door_count += 1
                            if open_door_count == 1:
                                actions.append('open door')
                            elif open_door_count == 2:
                                actions.append('move forward')
                            break
            elif agent_pos[1]==4:
                if agent_dir == DIRECTION_DICT['left'] or agent_dir == DIRECTION_DICT['right']:
                    actions.append('move forward')
        
        else:
            raise ValueError("Agent position not recognized.")
        
        return actions
        
    