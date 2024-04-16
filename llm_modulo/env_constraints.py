class EnvironmentConstraints:
    
    def __init__(self, env, seed):
        self.env = env
        self.constraints = []
        self.seed = seed
        
class DoorKey5x5(EnvironmentConstraints):
    
    def __init__(self, env, seed):
        super().__init__(env, seed)
        self.constraints = []
        
    def feasible_actions(self, agent_pos):
        """
        Returns the feasible actions in the current state.
        Input: agent_pos (symbolic observation), seed
        Output: list of actions
        """        
        raise NotImplementedError
    