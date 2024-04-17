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
    3: 'pickup', 
    4: 'drop', 
    5: 'toggle', 
    6: 'done'}
DIRECTION_DICT = {
    'right': 0,
    'down': 1,
    'left': 2,
    'up': 3}

class EnvironmentConstraints:
    
    def __init__(self, env, seed):
        self.env = env
        self.constraints = []
        self.seed = seed
        
class DoorKey5x5(EnvironmentConstraints):
    
    def __init__(self, env, seed):
        super().__init__(env, seed)
        self.constraints = []
        
    def get_agent_pos(self, observation):
        """
        Returns the agent position from the observation.
        Input: symbolic observation['image']
        Output: agent position
        """
        agent_obs = observation['image']
        agent_dir = observation['direction']
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    if agent_obs[i][j][k] == OBJECT_TO_IDX['agent']:
                        agent_pos = (i, j)                        
                        return agent_pos, agent_dir
        
        raise ValueError("Agent not found in the observation.")
        
    def feasible_actions(self, agent_pos, agent_dir, action_history, seed=0):
        """
        Returns the feasible actions in the current state.
        Input: agent_pos (x,y) from symbolic obs, agent_dir
        Output: list of actions
        """       
        actions = [] 
        if seed == 0:
            if agent_pos[0]==1: # agent is in the second row
                if agent_pos[1]==1:
                    # agent has picked key already, can open door if facing it, and can move forward if door is opened
                    if agent_dir == DIRECTION_DICT['right']:
                        if ACTION_DICT[action_history[-1]] == 'toggle': # agent has opened door already
                            actions.append(['move forward'])
                        else:
                            actions.append(['toggle']) # agent has not opened door yet
                    elif agent_dir == DIRECTION_DICT['down']: # agent is facing down (wrong direction)
                        actions.append('move forward')
                elif agent_pos[1]==2:
                    # agent has opened door and moved forward to door cell, can move forward if facing right (correct) or left (wrong) direction
                    if agent_dir == DIRECTION_DICT['right'] or agent_dir == DIRECTION_DICT['left']:
                        actions.append(['move forward'])
                    
                elif agent_pos[1]==3:
                    # agent has moved forward, can move forward if facing down (correct) or left (wrong) direction
                    pass
            elif agent_pos[0]==2: # agent is in the third row
                if agent_pos[1]==1:
                    # agent has picked key already, can move forward if facing up (correct) or down (wrong) direction
                    pass
                elif agent_pos[1]==3:
                    # agent has moved forward closer to goal, can move forward if facing down (correct) or up (wrong) direction
                    pass
            elif agent_pos[0]==3: # agent is in the fourth row
                if agent_pos[1]==1:
                    # agent is in the initial state, can pickup key if facing up (correct) direction, and move forward facing up if key is picked
                    pass
                elif agent_pos[1]==3:
                    # agent has reached goal cell, can move forward facing up (wrong) direction or execute action done
                    pass
            else:
                raise ValueError("Agent position not recognized.")
        
            # append actions that are always valid in any state
            actions.append(['turn left', 'turn right'])
            return actions
        
        else:
            raise NotImplementedError("Seed not implemented.")