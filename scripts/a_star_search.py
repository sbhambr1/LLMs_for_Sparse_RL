ORIENTATION_DICT = {
    'UP': (-1,0),
    'DOWN': (1,0),
    'LEFT': (0,-1),
    'RIGHT': (0,1)}

import argparse
import heapq

parser = argparse.ArgumentParser()

parser.add_argument('--env', type=str, default='MiniGrid-Empty-Random-5x5-v0', help='Environment name, choose from MiniGrid-Empty-Random-5x5-v0, MiniGrid-LavaGapS5-v0, MiniGrid-DoorKey-5x5-v0, MiniGrid-DoorKey-6x6-v0. Only seed 0 configurations are available.')

def grid_generator(env='MiniGrid-Empty-Random-5x5-v0'):
    if env == 'MiniGrid-Empty-Random-5x5-v0':
        grid = [
            ['empty', 'start', 'empty'],
            ['empty', 'empty', 'empty'],
            ['empty', 'empty', 'goal' ]
        ]
        actions = ['turn left', 'turn right', 'move forward']
        start = (0, 1)
        goal = (2, 2)
        orientation = 'DOWN'
    elif env == 'MiniGrid-LavaGapS5-v0':
        grid = [
            ['start', 'obstacle', 'empty'],
            ['empty', 'obstacle', 'empty'],
            ['empty', 'empty',    'goal' ]
        ]
        actions = ['turn left', 'turn right', 'move forward']
        start = (0, 0)
        goal = (2, 2)
        orientation = 'RIGHT'
    elif env == 'MiniGrid-DoorKey-5x5-v0':
        grid = [
            ['empty', 'door',     'empty'],
            ['key',   'obstacle', 'empty'],
            ['start', 'obstacle', 'goal' ]
        ]
        actions = ['turn left', 'turn right', 'move forward', 'pickup key', 'open door']
        start = (2, 0)
        goal = (2, 2)
        orientation = 'LEFT'
    elif env == 'MiniGrid-DoorKey-6x6-v0':
        grid = [
            ['empty', 'empty', 'door',     'empty'],
            ['empty', 'empty', 'obstacle', 'empty'],
            ['start', 'key',   'obstacle', 'empty'],
            ['empty', 'empty', 'obstacle', 'goal' ]
        ]
        actions = ['turn left', 'turn right', 'move forward', 'pickup key', 'open door']
        start = (2, 0)
        goal = (3, 3)
        orientation = 'DOWN'
    else:
        raise ValueError('Invalid environment')
    
    return grid, actions, start, goal, orientation

def heuristic(state, goal):
    # Manhattan distance heuristic
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

def is_valid_location(grid, location):
    # Check if the location is within the grid boundaries and not an obstacle
    x, y = location
    if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] != 'obstacle':
        # if grid[x][y] == 'key':
        #     if picked_key==False:
        #         return False
        # if grid[x][y] == 'door':
        #     if opened_door==False:
        #         return False
        return True

def a_star_search(grid, start, goal, orientation):
    # A* search algorithm
    visited = set()
    nodes_expanded = 0
    picked_key = False
    opened_door = False
    queue = [(heuristic(start, goal), 0, start, orientation, [])]  # (heuristic value, cost, position, orientation, path)
    
    while queue:
        _, cost, current_pos, current_orientation, path = heapq.heappop(queue)
        
        if current_pos == goal:
            return path + [current_pos], nodes_expanded
        
        if (current_pos, current_orientation) in visited:
            continue
        
        visited.add((current_pos, current_orientation))
        nodes_expanded += 1
        
        for action in ACTIONS:
            if action == 'turn left':
                if current_orientation == 'UP':
                    new_orientation = 'LEFT'
                elif current_orientation == 'DOWN':
                    new_orientation = 'RIGHT'
                elif current_orientation == 'LEFT':
                    new_orientation = 'DOWN'
                elif current_orientation == 'RIGHT':
                    new_orientation = 'UP'
                new_pos = current_pos
            elif action == 'turn right':
                if current_orientation == 'UP':
                    new_orientation = 'RIGHT'
                elif current_orientation == 'DOWN':
                    new_orientation = 'LEFT'
                elif current_orientation == 'LEFT':
                    new_orientation = 'UP'
                elif current_orientation == 'RIGHT':
                    new_orientation = 'DOWN'
                new_pos = current_pos
            elif action == 'move forward':
                new_x, new_y = current_pos[0] + ORIENTATION_DICT[current_orientation][0], current_pos[1] + ORIENTATION_DICT[current_orientation][1]
                new_pos = (new_x, new_y)
                new_orientation = current_orientation
            elif action == 'pickup key':
                adjacent_pos = (current_pos[0] + ORIENTATION_DICT[current_orientation][0], current_pos[1] + ORIENTATION_DICT[current_orientation][1])
                if is_valid_location(grid, adjacent_pos) and grid[adjacent_pos[0]][adjacent_pos[1]] == 'key' and picked_key==False:
                    picked_key = True
                    new_pos = current_pos
                    new_orientation = current_orientation
                else:
                    continue
            elif action == 'open door':
                adjacent_pos = (current_pos[0] + ORIENTATION_DICT[current_orientation][0], current_pos[1] + ORIENTATION_DICT[current_orientation][1])
                if is_valid_location(grid, adjacent_pos) and grid[adjacent_pos[0]][adjacent_pos[1]] == 'door' and picked_key==True and opened_door==False:
                    opened_door = True
                    new_pos = current_pos
                    new_orientation = current_orientation
                else:
                    continue
            
            if is_valid_location(grid, new_pos) and (new_pos, new_orientation) not in visited:
                if picked_key==False and grid[new_pos[0]][new_pos[1]] == 'key':
                    new_cost = -10
                    new_heuristic = new_cost + heuristic(new_pos, goal)
                    heapq.heappush(queue, (new_heuristic, new_cost, new_pos, new_orientation, path + [(action, new_pos)]))
                elif opened_door==False and grid[new_pos[0]][new_pos[1]] == 'door' and picked_key==True:
                    new_cost = -10
                    new_heuristic = new_cost + heuristic(new_pos, goal)
                    heapq.heappush(queue, (new_heuristic, new_cost, new_pos, new_orientation, path + [(action, new_pos)]))
                else:
                    new_cost = cost + 1
                    new_heuristic = new_cost + heuristic(new_pos, goal)
                    heapq.heappush(queue, (new_heuristic, new_cost, new_pos, new_orientation, path + [(action, new_pos)]))
    
    return None  # No path found

if __name__ == '__main__':
    
    args = parser.parse_args()
    grid, ACTIONS, start, goal, orientation = grid_generator(env=args.env)
    
    path, nodes_expanded = a_star_search(grid, start, goal, orientation)
    if path:
        print("Path found:")
        for action, position in path:
            print("Action:", action, "|| Position:", position)
        print("Number of nodes expanded:", nodes_expanded)
    else:
        print("No path found.")

