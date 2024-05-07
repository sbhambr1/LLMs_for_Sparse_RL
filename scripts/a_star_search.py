import heapq

# Define the possible actions (up, down, left, right)
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def heuristic(state, goal):
    # Manhattan distance heuristic
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

def is_valid_location(grid, location):
    # Check if the location is within the grid boundaries and not an obstacle
    x, y = location
    return 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] != 'obstacle'

def a_star_search(grid, start, goal):
    # A* search algorithm
    visited = set()
    queue = [(heuristic(start, goal), 0, start, [])]  # (heuristic value, cost, position, path)
    
    while queue:
        _, cost, current_pos, path = heapq.heappop(queue)
        
        if current_pos == goal:
            return path + [current_pos]
        
        if current_pos in visited:
            continue
        
        visited.add(current_pos)
        
        for action in ACTIONS:
            new_x, new_y = current_pos[0] + action[0], current_pos[1] + action[1]
            new_pos = (new_x, new_y)
            
            if is_valid_location(grid, new_pos) and new_pos not in visited:
                new_cost = cost + 1
                new_heuristic = new_cost + heuristic(new_pos, goal)
                heapq.heappush(queue, (new_heuristic, new_cost, new_pos, path + [current_pos]))
    
    return None  # No path found

# Example grid
grid = [
    ['empty', 'empty', 'empty', 'empty', 'empty'],
    ['obstacle', 'empty', 'obstacle', 'empty', 'empty'],
    ['empty', 'empty', 'empty', 'empty', 'empty'],
    ['empty', 'obstacle', 'empty', 'obstacle', 'empty'],
    ['empty', 'empty', 'empty', 'empty', 'empty']
]

start = (0, 0)
goal = (4, 4)

path = a_star_search(grid, start, goal)
if path:
    print("Path found:", path)
else:
    print("No path found.")
