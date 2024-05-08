import heapq

# Define grid, actions, start, goal, and orientation
grid = [
    ['empty', 'door', 'empty'],
    ['key', 'obstacle', 'empty'],
    ['start', 'obstacle', 'goal']
]
actions = ['turn left', 'turn right', 'move forward', 'pickup key', 'open door']
start = (2, 0)
goal = (2, 2)
orientation = 'LEFT'

# Define action costs
action_costs = {
    'turn left': 1,
    'turn right': 1,
    'move forward': 1,
    'pickup key': 1,
    'open door': 1
}

# Define heuristic function (Manhattan distance)
def heuristic(node):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def is_valid_position(grid, position):
    x, y = position
    return 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] != 'obstacle'

# Define function to generate successors
def get_successors(grid, node, orientation):
    x, y = node
    successors = []
    picked_key = False
    opened_door = False

    # Generate possible successor nodes
    for action in actions:
        if action == 'turn left':
            # Determine new orientation after turning left
            if orientation == 'UP':
                new_orientation = 'LEFT'
            elif orientation == 'LEFT':
                new_orientation = 'DOWN'
            elif orientation == 'DOWN':
                new_orientation = 'RIGHT'
            else:  # orientation == 'RIGHT'
                new_orientation = 'UP'
            successors.append(((x, y), new_orientation, action))

        elif action == 'turn right':
            # Determine new orientation after turning right
            if orientation == 'UP':
                new_orientation = 'RIGHT'
            elif orientation == 'RIGHT':
                new_orientation = 'DOWN'
            elif orientation == 'DOWN':
                new_orientation = 'LEFT'
            else:  # orientation == 'LEFT'
                new_orientation = 'UP'
            successors.append(((x, y), new_orientation, action))

        elif action == 'move forward':
            # Calculate new position after moving forward
            if orientation == 'UP':
                new_pos = (x - 1, y)
            elif orientation == 'DOWN':
                new_pos = (x + 1, y)
            elif orientation == 'LEFT':
                new_pos = (x, y - 1)
            else:  # orientation == 'RIGHT'
                new_pos = (x, y + 1)

            # Check if the new position is within the grid and not an obstacle
            if is_valid_position(grid, new_pos):
                if (grid[new_pos[0]][new_pos[1]] == 'key' and picked_key) or (grid[new_pos[0]][new_pos[1]] == 'door' and opened_door):
                    successors.append((new_pos, orientation, action))

        elif action == 'pickup key':
            # Check if there is a key in an adjacent cell and facing the key
            if is_valid_position(grid, (x, y+1)) and grid[x][y+1]=='key' and orientation == 'RIGHT':
                successors.append(((x, y), orientation, action))
                picked_key = True
            elif is_valid_position(grid, (x, y-1)) and grid[x][y-1]=='key' and orientation == 'LEFT':
                successors.append(((x, y), orientation, action))
                picked_key = True
            elif is_valid_position(grid, (x+1, y)) and grid[x+1][y]=='key' and orientation == 'DOWN':
                successors.append(((x, y), orientation, action))
                picked_key = True
            elif is_valid_position(grid, (x-1, y)) and grid[x-1][y]=='key' and orientation == 'UP':
                successors.append(((x, y), orientation, action))
                picked_key = True

        elif action == 'open door':
            # Check if there is a door in an adjacent cell and facing the door
            if is_valid_position(grid, (x, y+1)) and grid[x][y+1]=='door' and orientation == 'RIGHT':
                successors.append(((x, y), orientation, action))
                opened_door = True
            elif is_valid_position(grid, (x, y-1)) and grid[x][y-1]=='door' and orientation == 'LEFT':
                successors.append(((x, y), orientation, action))
                opened_door = True
            elif is_valid_position(grid, (x+1, y)) and grid[x+1][y]=='door' and orientation == 'DOWN':
                successors.append(((x, y), orientation, action))
                opened_door = True
            elif is_valid_position(grid, (x-1, y)) and grid[x-1][y]=='door' and orientation == 'UP':
                successors.append(((x, y), orientation, action))
                opened_door = True

    return successors

# Define function for A* search
def astar_search(grid, start, goal, orientation):
    open_list = [(0, start, orientation, [])]  # (f, node, orientation, actions)
    closed_set = set()

    while open_list:
        f, node, orientation, actions = heapq.heappop(open_list)

        if node == goal:
            return actions

        if node not in closed_set:
            closed_set.add(node)
            successors = get_successors(grid, node, orientation)

            for successor in successors:
                next_node, next_orientation, action = successor
                g = len(actions) + action_costs[action]
                h = heuristic(next_node)
                f = g + h
                new_actions = actions + [action]
                heapq.heappush(open_list, (f, next_node, next_orientation, new_actions))

    return None  # No path found

# Execute A* search
path = astar_search(grid, start, goal, orientation)
print("Path:", path)
