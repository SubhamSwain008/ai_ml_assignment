import streamlit as st
from pyamaze import maze
from queue import PriorityQueue
import matplotlib.pyplot as plt
from PIL import Image
import random
import math
from collections import deque

# Define the A* algorithm
def manhant_dis(cell1, cell2):
    x1, y1 = cell1
    x2, y2 = cell2
    return abs(x1 - x2) + abs(y1 - y2)

def A_star(m):
    start = (m.rows, m.cols)
    g_score = {cell: float('inf') for cell in m.grid}
    g_score[start] = 0
    f_score = {cell: float('inf') for cell in m.grid}
    f_score[start] = manhant_dis(start, (1, 1))

    open = PriorityQueue()
    open.put((manhant_dis(start, (1, 1)), manhant_dis(start, (1, 1)), start))

    apath = {}

    while not open.empty():
        currCell = open.get()[2]
        if currCell == (1, 1):
            break
        for d in 'ESNW':
            if m.maze_map[currCell][d]:
                if d == 'E':
                    childCell = (currCell[0], currCell[1] + 1)
                if d == 'W':
                    childCell = (currCell[0], currCell[1] - 1)
                if d == 'N':
                    childCell = (currCell[0] - 1, currCell[1])
                if d == 'S':
                    childCell = (currCell[0] + 1, currCell[1])
                
                temp_g_score = g_score[currCell] + 1
                temp_f_score = temp_g_score + manhant_dis(childCell, (1, 1))

                if temp_f_score < f_score[childCell]:
                    g_score[childCell] = temp_g_score
                    f_score[childCell] = temp_f_score
                    open.put((temp_f_score, manhant_dis(childCell, (1, 1)), childCell))
                    apath[childCell] = currCell

    fwdpath = {}
    cell = (1, 1)

    while cell != start:
        fwdpath[apath[cell]] = cell
        cell = apath[cell]

    return fwdpath

# Streamlit code
st.title("Initial Maze")

# Create a maze
m = maze(5, 5)

# Manually setup the maze grid and walls
m.maze_map = {
    (1, 1): {'E': True, 'W': False, 'N': False, 'S': True},
    (1, 2): {'E': True, 'W': True, 'N': False, 'S': False},
    (1, 3): {'E': True, 'W': True, 'N': False, 'S': False},
    (1, 4): {'E': True, 'W': True, 'N': False, 'S': True},
    (1, 5): {'E': False, 'W': True, 'N': False, 'S': True},
    (2, 1): {'E': True, 'W': False, 'N': True, 'S': False},
    (2, 2): {'E': True, 'W': True, 'N': False, 'S': True},
    (2, 3): {'E': False, 'W': True, 'N': False, 'S': True},
    (2, 4): {'E': True, 'W': False, 'N': True, 'S': True},
    (2, 5): {'E': False, 'W': True, 'N': True, 'S': True},
    (3, 1): {'E': True, 'W': False, 'N': False, 'S': True},
    (3, 2): {'E': False, 'W': True, 'N': True, 'S': True},
    (3, 3): {'E': True, 'W': False, 'N': True, 'S': False},
    (3, 4): {'E': True, 'W': True, 'N': True, 'S': True},
    (3, 5): {'E': False, 'W': True, 'N': True, 'S': False},
    (4, 1): {'E': True, 'W': False, 'N': True, 'S': False},
    (4, 2): {'E': True, 'W': True, 'N': True, 'S': True},
    (4, 3): {'E': False, 'W': True, 'N': False, 'S': True},
    (4, 4): {'E': True, 'W': False, 'N': True, 'S': False},
    (4, 5): {'E': False, 'W': True, 'N': False, 'S': True},
    (5, 1): {'E': True, 'W': False, 'N': True, 'S': False},
    (5, 2): {'E': False, 'W': True, 'N': True, 'S': False},
    (5, 3): {'E': True, 'W': False, 'N': True, 'S': False},
    (5, 4): {'E': True, 'W': True, 'N': False, 'S': False},
    (5, 5): {'E': False, 'W': True, 'N': False, 'S': False},
}
# Function to draw and save the initial maze as an image
def draw_initial_maze(m):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, m.cols)
    ax.set_ylim(0, m.rows)

    for cell in m.grid:
        x, y = cell[1] - 1, m.rows - cell[0]

        if not m.maze_map[cell]['E'] or cell[1] == m.cols:  # Right wall
            ax.plot([x, x], [y, y ], color='yellow', lw=2)
        if not m.maze_map[cell]['W']:  # Left wall
            ax.plot([x, x], [y, y + 1], color='black', lw=2)
        if not m.maze_map[cell]['N']:  # Upper wall
            ax.plot([x, x + 1], [y + 1, y + 1], color='black', lw=2)
        if not m.maze_map[cell]['S'] or cell[0] == m.rows:  # Bottom wall
            ax.plot([x, x + 1], [y, y], color='black', lw=2)

    ax = plt.gca()
    ax.set_xticklabels(['' for _ in ax.get_xticks()])
    ax.set_yticklabels(['' for _ in ax.get_yticks()])
    ax=ax.invert_xaxis
    return fig

# Draw and save the initial maze as an image
initial_fig = draw_initial_maze(m)
plt.savefig("initial_maze.png", bbox_inches='tight', pad_inches=0)

# Load the saved initial image and display it in Streamlit
initial_image = Image.open("initial_maze.png")
st.image(initial_image, caption="Initial Maze", use_column_width=True)

# Solve the maze
path = A_star(m)
def draw_maze(m, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, m.cols)
    ax.set_ylim(0, m.rows)

    # Draw the grid
    for cell in m.grid:
        x, y = cell[1] - 1, m.rows - cell[0]  # Adjusting the coordinates for plotting
        
        # Debugging output for the current cell
        print(f"Cell: {cell}, E: {m.maze_map[cell]['E']}, W: {m.maze_map[cell]['W']}, N: {m.maze_map[cell]['N']}, S: {m.maze_map[cell]['S']}")
        
        # Draw the right wall if it's missing or at the edge
        if not m.maze_map[cell]['E'] or cell[1] == m.cols:
            ax.plot([x + 1, x + 1], [y, y + 1], color='black', lw=2)
        # Draw the left wall
        if not m.maze_map[cell]['W']:
            ax.plot([x, x], [y, y + 1], color='black', lw=2)
        # Draw the upper wall
        if not m.maze_map[cell]['N']:
            ax.plot([x, x + 1], [y + 1, y + 1], color='black', lw=2)
        # Draw the bottom wall if it's missing or at the edge
        if not m.maze_map[cell]['S'] or cell[0] == m.rows:
            ax.plot([x, x + 1], [y, y], color='black', lw=2)
    
    # Draw the path
    for (cell1, cell2) in path.items():
        x1, y1 = cell1[1] - 1, m.rows - cell1[0]
        x2, y2 = cell2[1] - 1, m.rows - cell2[0]
        ax.plot([x1 + 0.5, x2 + 0.5], [y1 + 0.5, y2 + 0.5], color='red', lw=3)

    plt.axis('on')
    # plt.gca().invert_yaxis()
    return fig

# Draw and save the maze with the solution as an image
fig = draw_maze(m, path)
plt.savefig("maze_solution.png", bbox_inches='tight', pad_inches=0)

# Load the saved image and display it in Streamlit
image = Image.open("maze_solution.png")


# Define the Manhattan distance heuristic
def manhant_dis(cell1, cell2):
    x1, y1 = cell1
    x2, y2 = cell2
    return abs(x1 - x2) + abs(y1 - y2)

# Simulated Annealing algorithm
def simulated_annealing(m, initial_temp=100, cooling_rate=0.99):
    start = (m.rows, m.cols)
    current = start
    temperature = initial_temp
    apath = {}
    
    while current != (1, 1):
        next_cell = None
        best_heuristic = float('inf')
        possible_moves = []

        # Explore neighbors
        for d in 'ESNW':
            if m.maze_map[current][d]:
                if d == 'E':
                    childCell = (current[0], current[1] + 1)
                if d == 'W':
                    childCell = (current[0], current[1] - 1)
                if d == 'N':
                    childCell = (current[0] - 1, current[1])
                if d == 'S':
                    childCell = (current[0] + 1, current[1])
                
                heuristic = manhant_dis(childCell, (1, 1))
                possible_moves.append((childCell, heuristic))

        # Sort by best heuristic to worse
        possible_moves.sort(key=lambda x: x[1])

        # Choose the best move (lowest heuristic)
        for move in possible_moves:
            if move[1] < best_heuristic:
                best_heuristic = move[1]
                next_cell = move[0]

        # Simulated Annealing condition: accept a worse move with some probability
        if next_cell:
            if best_heuristic > manhant_dis(current, (1, 1)):
                prob = math.exp((manhant_dis(current, (1, 1)) - best_heuristic) / temperature)
                if random.uniform(0, 1) < prob:
                    apath[next_cell] = current
                    current = next_cell
            else:
                apath[next_cell] = current
                current = next_cell

        # Cool down temperature
        temperature *= cooling_rate

    # Reconstruct the path
    fwdpath = {}
    cell = (1, 1)

    while cell != start:
        fwdpath[apath[cell]] = cell
        cell = apath[cell]

    return fwdpath



# Function to draw and save the initial maze as an image
def draw_initial_maze(m):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, m.cols)
    ax.set_ylim(0, m.rows)

    for cell in m.grid:
        x, y = cell[1] - 1, m.rows - cell[0]

        if not m.maze_map[cell]['E'] or cell[1] == m.cols:  # Right wall
            ax.plot([x, x], [y, y], color='black', lw=2)
        if not m.maze_map[cell]['W']:  # Left wall
            ax.plot([x, x], [y, y + 1], color='black', lw=2)
        if not m.maze_map[cell]['N']:  # Upper wall
            ax.plot([x, x + 1], [y + 1, y + 1], color='black', lw=2)
        if not m.maze_map[cell]['S'] or cell[0] == m.rows:  # Bottom wall
            ax.plot([x, x + 1], [y, y], color='black', lw=2)

    plt.axis('on')
    # plt.gca().invert_yaxis()
    return fig

# # Draw and save the initial maze as an image
# initial_fig = draw_initial_maze(m)
# plt.savefig("initial_maze.png", bbox_inches='tight', pad_inches=0)

# # Load the saved initial image and display it in Streamlit
# initial_image = Image.open("initial_maze.png")
# st.image(initial_image, caption="Initial Maze", use_column_width=True)

# Solve the maze using Simulated Annealing
path = simulated_annealing(m)

# Function to draw and save the maze with the solution path
def draw_maze(m, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, m.cols)
    ax.set_ylim(0, m.rows)

    # Draw the grid
    for cell in m.grid:
        x, y = cell[1] - 1, m.rows - cell[0]  # Adjusting the coordinates for plotting
        
        # Draw the right wall if it's missing or at the edge
        if not m.maze_map[cell]['E'] or cell[1] == m.cols:
            ax.plot([x + 1, x + 1], [y, y + 1], color='black', lw=2)
        # Draw the left wall
        if not m.maze_map[cell]['W']:
            ax.plot([x, x], [y, y + 1], color='black', lw=2)
        # Draw the upper wall
        if not m.maze_map[cell]['N']:
            ax.plot([x, x + 1], [y + 1, y + 1], color='black', lw=2)
        # Draw the bottom wall if it's missing or at the edge
        if not m.maze_map[cell]['S'] or cell[0] == m.rows:
            ax.plot([x, x + 1], [y, y], color='black', lw=2)
    
    # Draw the path
    for (cell1, cell2) in path.items():
        x1, y1 = cell1[1] - 1, m.rows - cell1[0]
        x2, y2 = cell2[1] - 1, m.rows - cell2[0]
        ax.plot([x1 + 0.5, x2 + 0.5], [y1 + 0.5, y2 + 0.5], color='red', lw=3)

    plt.axis('on')
    # plt.gca().invert_yaxis()
    return fig

# Draw and save the maze with the solution as an image
fig = draw_maze(m, path)
plt.savefig("maze_solution.png", bbox_inches='tight', pad_inches=0)

# Load the saved image and display it in Streamlit
image = Image.open("maze_solution.png")

### csp


class CSP_Maze_Solver:
    def __init__(self, m):
        self.maze = m
        self.start = (m.rows, m.cols)
        self.goal = (1, 1)
        self.domain = {cell: [] for cell in m.grid}
        self.assignment = {}
        self.visited = set()

    def initialize_domain(self):
        # Define valid moves for each cell based on maze structure
        for cell in self.maze.grid:
            if self.maze.maze_map[cell]['E']:
                self.domain[cell].append((cell[0], cell[1] + 1))
            if self.maze.maze_map[cell]['W']:
                self.domain[cell].append((cell[0], cell[1] - 1))
            if self.maze.maze_map[cell]['N']:
                self.domain[cell].append((cell[0] - 1, cell[1]))
            if self.maze.maze_map[cell]['S']:
                self.domain[cell].append((cell[0] + 1, cell[1]))

    def is_consistent(self, cell, value):
        # Ensure no revisiting of cells
        return value not in self.visited

    def assign(self, cell, value):
        self.assignment[cell] = value
        self.visited.add(value)

    def unassign(self, cell):
        if cell in self.assignment:
            self.visited.remove(self.assignment[cell])
            del self.assignment[cell]

    def backtrack(self, cell):
        # If goal is reached, return solution
        if cell == self.goal:
            return True

        # Try valid moves in the domain
        for value in self.domain[cell]:
            if self.is_consistent(cell, value):
                self.assign(cell, value)
                if self.backtrack(value):
                    return True
                self.unassign(cell)  # Backtrack

        return False

    def solve(self):
                self.initialize_domain()
                self.assign(self.start, self.start)  # Start from the starting cell
                if self.backtrack(self.start):
                    return self.assignment
                else:
                    return None  # No solution found

        # Instantiate and solve using the CSP solver



solver = CSP_Maze_Solver(m)
path = solver.solve()

# Display the path
if path:
    # st.write("Maze solved using CSP!")
    print('')
else:
    st.write("No solution found.")
# Function to draw and save the initial maze as an image
def draw_initial_maze(m):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, m.cols)
    ax.set_ylim(0, m.rows)

    for cell in m.grid:
        x, y = cell[1] - 1, m.rows - cell[0]

        if not m.maze_map[cell]['E'] or cell[1] == m.cols:  # Right wall
            ax.plot([x, x], [y, y], color='black', lw=2)
        if not m.maze_map[cell]['W']:  # Left wall
            ax.plot([x, x], [y, y + 1], color='black', lw=2)
        if not m.maze_map[cell]['N']:  # Upper wall
            ax.plot([x, x + 1], [y + 1, y + 1], color='black', lw=2)
        if not m.maze_map[cell]['S'] or cell[0] == m.rows:  # Bottom wall
            ax.plot([x, x + 1], [y, y], color='black', lw=2)

    plt.axis('off')
    plt.gca().invert_yaxis()
    return fig

# Draw and save the initial maze as an image
initial_fig = draw_initial_maze(m)
plt.savefig("initial_maze.png", bbox_inches='tight', pad_inches=0)

# # Load the saved initial image and display it in Streamlit
# initial_image = Image.open("initial_maze.png")
# st.image(initial_image, caption="Initial Maze", use_column_width=True)

# Solve the maze using CSP
solver = CSP_Maze_Solver(m)
path = solver.solve()

# Function to draw and save the maze with the solution path
def draw_maze(m, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, m.cols)
    ax.set_ylim(0, m.rows)

    # Draw the grid
    for cell in m.grid:
        x, y = cell[1] - 1, m.rows - cell[0]  # Adjusting the coordinates for plotting
        
        # Draw the right wall if it's missing or at the edge
        if not m.maze_map[cell]['E'] or cell[1] == m.cols:
            ax.plot([x + 1, x + 1], [y, y + 1], color='black', lw=2)
        # Draw the left wall
        if not m.maze_map[cell]['W']:
            ax.plot([x, x], [y, y + 1], color='black', lw=2)
        # Draw the upper wall
        if not m.maze_map[cell]['N']:
            ax.plot([x, x + 1], [y + 1, y + 1], color='black', lw=2)
        # Draw the bottom wall if it's missing or at the edge
        if not m.maze_map[cell]['S'] or cell[0] == m.rows:
            ax.plot([x, x + 1], [y, y], color='black', lw=2)
    
    # Draw the path
    for (cell1, cell2) in path.items():
        x1, y1 = cell1[1] - 1, m.rows - cell1[0]
        x2, y2 = cell2[1] - 1, m.rows - cell2[0]
        ax.plot([x1 + 0.5, x2 + 0.5], [y1 + 0.5, y2 + 0.5], color='red', lw=3)

    ax = plt.gca()
    ax.set_xticklabels(['' for _ in ax.get_xticks()])
    ax.set_yticklabels(['' for _ in ax.get_yticks()])
    ax=ax.invert_xaxis
    return fig

# Draw and save the maze with the solution as an image
fig = draw_maze(m, path)
plt.savefig("maze_solution.png", bbox_inches='tight', pad_inches=0)

# Load the saved image and display it in Streamlit
image = Image.open("maze_solution.png")

## hill


# Define the Manhattan distance heuristic
def manhant_dis(cell1, cell2):
    x1, y1 = cell1
    x2, y2 = cell2
    return abs(x1 - x2) + abs(y1 - y2)

# Hill Climbing algorithm
def hill_climbing(m):
    start = (m.rows, m.cols)
    current = start
    apath = {}
    while current != (1, 1):
        next_cell = None
        best_heuristic = float('inf')
        
        for d in 'ESNW':
            if m.maze_map[current][d]:
                if d == 'E':
                    childCell = (current[0], current[1] + 1)
                if d == 'W':
                    childCell = (current[0], current[1] - 1)
                if d == 'N':
                    childCell = (current[0] - 1, current[1])
                if d == 'S':
                    childCell = (current[0] + 1, current[1])
                
                heuristic = manhant_dis(childCell, (1, 1))
                
                if heuristic < best_heuristic:
                    best_heuristic = heuristic
                    next_cell = childCell
        
        if next_cell is None:  # If stuck, break (local maxima)
            break

        apath[next_cell] = current
        current = next_cell

    fwdpath = {}
    cell = (1, 1)

    while cell != start:
        fwdpath[apath[cell]] = cell
        cell = apath[cell]

    return fwdpath


# Function to draw and save the initial maze as an image
def draw_initial_maze(m):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, m.cols)
    ax.set_ylim(0, m.rows)

    for cell in m.grid:
        x, y = cell[1] - 1, m.rows - cell[0]

        if not m.maze_map[cell]['E'] or cell[1] == m.cols:  # Right wall
            ax.plot([x, x], [y, y], color='black', lw=2)
        if not m.maze_map[cell]['W']:  # Left wall
            ax.plot([x, x], [y, y + 1], color='black', lw=2)
        if not m.maze_map[cell]['N']:  # Upper wall
            ax.plot([x, x + 1], [y + 1, y + 1], color='black', lw=2)
        if not m.maze_map[cell]['S'] or cell[0] == m.rows:  # Bottom wall
            ax.plot([x, x + 1], [y, y], color='black', lw=2)

    plt.axis('off')
    plt.gca().invert_yaxis()
    return fig

# Draw and save the initial maze as an image
initial_fig = draw_initial_maze(m)
plt.savefig("initial_maze.png", bbox_inches='tight', pad_inches=0)

# # Load the saved initial image and display it in Streamlit
# initial_image = Image.open("initial_maze.png")
# st.image(initial_image, caption="Initial Maze", use_column_width=True)

# Solve the maze using Hill Climbing
path = hill_climbing(m)

# Function to draw and save the maze with the solution path
def draw_maze(m, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, m.cols)
    ax.set_ylim(0, m.rows)

    # Draw the grid
    for cell in m.grid:
        x, y = cell[1] - 1, m.rows - cell[0]  # Adjusting the coordinates for plotting
        
        # Draw the right wall if it's missing or at the edge
        if not m.maze_map[cell]['E'] or cell[1] == m.cols:
            ax.plot([x + 1, x + 1], [y, y + 1], color='black', lw=2)
        # Draw the left wall
        if not m.maze_map[cell]['W']:
            ax.plot([x, x], [y, y + 1], color='black', lw=2)
        # Draw the upper wall
        if not m.maze_map[cell]['N']:
            ax.plot([x, x + 1], [y + 1, y + 1], color='black', lw=2)
        # Draw the bottom wall if it's missing or at the edge
        if not m.maze_map[cell]['S'] or cell[0] == m.rows:
            ax.plot([x, x + 1], [y, y], color='black', lw=2)
    
    # Draw the path
    for (cell1, cell2) in path.items():
        x1, y1 = cell1[1] - 1, m.rows - cell1[0]
        x2, y2 = cell2[1] - 1, m.rows - cell2[0]
        ax.plot([x1 + 0.5, x2 + 0.5], [y1 + 0.5, y2 + 0.5], color='red', lw=3)

    ax = plt.gca()
    ax.set_xticklabels(['' for _ in ax.get_xticks()])
    ax.set_yticklabels(['' for _ in ax.get_yticks()])
    ax=ax.invert_xaxis
    return fig

# Draw and save the maze with the solution as an image
fig = draw_maze(m, path)
plt.savefig("maze_solution.png", bbox_inches='tight', pad_inches=0)

# Load the saved image and display it in Streamlit
image = Image.open("maze_solution.png")




if st.button("Maze with Hill Climbing Path"):

   st.image(image, caption="Maze with Hill Climbing Path", use_column_width=True)

if st.button("Maze with CSP Path"):
  st.image(image, caption="Maze with CSP Path", use_column_width=True)

if st.button("Maze with Simulated Annealing Path"):
  st.image(image, caption="Maze with Simulated Annealing Path", use_column_width=True)

if st.button("Maze with A* Path"):
   st.image(image, caption="Maze with A* Path", use_column_width=True)
