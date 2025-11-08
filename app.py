"""
================================================================================
8-PUZZLE SOLVER - STREAMLIT APP
================================================================================
CS 6613 Project 1 - A* Search Algorithm
Author: Vikram Markali   vrm9190
        Raj Jain         

This program solves the 8-puzzle problem using A* search algorithm with two
different heuristic functions:
- h1: Manhattan Distance
- h2: Linear Conflict (Manhattan Distance + 2 * Linear Conflicts)
================================================================================
"""

import streamlit as st
import heapq
from copy import deepcopy
import time
import io

# ============================================================================
# PUZZLE STATE CLASS
# ============================================================================

class PuzzleState:
    """
    Represents a single state in the 8-puzzle problem.
    
    Attributes:
        board (list): 3x3 matrix representing current puzzle configuration
        goal (list): 3x3 matrix representing target configuration
        parent (PuzzleState): Reference to parent state in search tree
        action (str): Action taken to reach this state ('L', 'R', 'U', 'D')
        g (int): Cost from initial state to current state (depth/path cost)
        blank_pos (tuple): (row, col) position of blank tile (0)
    """
    
    def __init__(self, board, goal, parent=None, action=None, g=0):
        """
        Initialize a new puzzle state.
        
        Args:
            board: 3x3 list representing current puzzle state
            goal: 3x3 list representing goal state
            parent: Parent state (None for initial state)
            action: Action that led to this state (None for initial state)
            g: Path cost from initial state (0 for initial state)
        """
        self.board = board
        self.goal = goal
        self.parent = parent
        self.action = action
        self.g = g
        self.blank_pos = self.find_blank()
    
    def find_blank(self):
        """
        Find the position of the blank tile (represented by 0).
        
        Returns:
            tuple: (row, col) coordinates of blank tile
        """
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    return (i, j)
        return None
    
    def __lt__(self, other):
        """Less-than comparison for heap operations (not used, required by heapq)."""
        return False
    
    def get_board_tuple(self):
        """
        Convert board to immutable tuple for use as dictionary key.
        
        Returns:
            tuple: Immutable representation of board state
        """
        return tuple(tuple(row) for row in self.board)
    
    def find_tile_position(self, tile, board):
        """
        Find the position of a specific tile value in a board.
        
        Args:
            tile: The tile value to find (1-8)
            board: The board to search in
            
        Returns:
            tuple: (row, col) coordinates of the tile
        """
        for i in range(3):
            for j in range(3):
                if board[i][j] == tile:
                    return (i, j)
        return None
    
    # ========================================================================
    # HEURISTIC 1: MANHATTAN DISTANCE
    # ========================================================================
    
    def manhattan_distance(self):
        """
        Calculate Manhattan Distance heuristic (h1).
        
        Manhattan distance is the sum of horizontal and vertical distances
        of each tile from its goal position. Blank tile (0) is not counted.
        
        Formula: h1(n) = Σ |current_row - goal_row| + |current_col - goal_col|
                 for all tiles (excluding blank)
        
        Returns:
            int: Total Manhattan distance for this state
        """
        distance = 0
        
        # Check each position on the board
        for i in range(3):
            for j in range(3):
                if self.board[i][j] != 0:  # Skip blank tile
                    tile = self.board[i][j]
                    # Find where this tile should be in goal state
                    goal_i, goal_j = self.find_tile_position(tile, self.goal)
                    # Add Manhattan distance for this tile
                    distance += abs(i - goal_i) + abs(j - goal_j)
        
        return distance
    
    # ========================================================================
    # HEURISTIC 2: LINEAR CONFLICT
    # ========================================================================
    
    def linear_conflict(self):
        """
        Calculate Linear Conflict heuristic (h2).
        
        Linear conflict adds to Manhattan distance by detecting when two tiles
        are in their correct row or column but in reversed order relative to
        their goal positions. Each such conflict requires at least 2 additional
        moves to resolve.
        
        Formula: h2(n) = h1(n) + 2 × (number of linear conflicts)
        
        A linear conflict occurs when:
        1. Two tiles tj and tk are in the same row/column
        2. Their goal positions are also in that same row/column
        3. tj is to the right/below of tk
        4. But goal position of tj is to the left/above goal position of tk
        
        Returns:
            int: Manhattan distance + 2 * conflicts
        """
        # Start with Manhattan distance as base
        h1 = self.manhattan_distance()
        conflicts = 0
        
        # Check for ROW conflicts
        # A row conflict occurs when two tiles are in the same row,
        # both belong to that row in the goal, but are in wrong order
        for i in range(3):
            for j in range(3):
                if self.board[i][j] != 0:
                    tile1 = self.board[i][j]
                    goal_i1, goal_j1 = self.find_tile_position(tile1, self.goal)
                    
                    # If tile1 belongs to this row in the goal state
                    if i == goal_i1:
                        # Check tiles to the right of tile1
                        for k in range(j + 1, 3):
                            if self.board[i][k] != 0:
                                tile2 = self.board[i][k]
                                goal_i2, goal_j2 = self.find_tile_position(tile2, self.goal)
                                
                                # If tile2 also belongs to this row but should be LEFT of tile1
                                if i == goal_i2 and goal_j1 > goal_j2:
                                    conflicts += 1
        
        # Check for COLUMN conflicts
        # A column conflict occurs when two tiles are in the same column,
        # both belong to that column in the goal, but are in wrong order
        for j in range(3):
            for i in range(3):
                if self.board[i][j] != 0:
                    tile1 = self.board[i][j]
                    goal_i1, goal_j1 = self.find_tile_position(tile1, self.goal)
                    
                    # If tile1 belongs to this column in the goal state
                    if j == goal_j1:
                        # Check tiles below tile1
                        for k in range(i + 1, 3):
                            if self.board[k][j] != 0:
                                tile2 = self.board[k][j]
                                goal_i2, goal_j2 = self.find_tile_position(tile2, self.goal)
                                
                                # If tile2 also belongs to this column but should be ABOVE tile1
                                if j == goal_j2 and goal_i1 > goal_i2:
                                    conflicts += 1
        
        # Each conflict requires at least 2 additional moves to resolve
        return h1 + 2 * conflicts
    
    # ========================================================================
    # STATE OPERATIONS
    # ========================================================================
    
    def get_successors(self):
        """
        Generate all valid successor states from current state.
        
        Moves the blank tile in all possible directions (L, R, U, D).
        Move order is important for deterministic behavior and consistency
        with expected results.
        
        Returns:
            list: List of PuzzleState objects representing valid next states
        """
        successors = []
        i, j = self.blank_pos
        
        # Define moves: (row_delta, col_delta, action_name)
        # Order: L, R, U, D (Left, Right, Up, Down)
        # This represents moving the BLANK tile in that direction
        moves = [
            (0, -1, 'L'),   # Move blank LEFT (swap with left tile)
            (0, 1, 'R'),    # Move blank RIGHT (swap with right tile)
            (-1, 0, 'U'),   # Move blank UP (swap with upper tile)
            (1, 0, 'D')     # Move blank DOWN (swap with lower tile)
        ]
        
        # Try each possible move
        for di, dj, action in moves:
            ni, nj = i + di, j + dj
            
            # Check if new position is within board boundaries
            if 0 <= ni < 3 and 0 <= nj < 3:
                # Create new board with tiles swapped
                new_board = deepcopy(self.board)
                new_board[i][j], new_board[ni][nj] = new_board[ni][nj], new_board[i][j]
                
                # Create new state with incremented cost (g + 1)
                new_state = PuzzleState(new_board, self.goal, self, action, self.g + 1)
                successors.append(new_state)
        
        return successors
    
    def is_goal(self):
        """
        Check if current state matches the goal state.
        
        Returns:
            bool: True if current board matches goal, False otherwise
        """
        return self.board == self.goal


# ============================================================================
# A* SEARCH ALGORITHM
# ============================================================================

def a_star_search(initial_state, heuristic='manhattan'):
    """
    A* Search Algorithm with Graph Search (no repeated states).
    
    This implements the A* algorithm to find the optimal solution to the
    8-puzzle problem. The algorithm uses:
    - f(n) = g(n) + h(n) where:
      - g(n) = cost from initial state to current state (path length)
      - h(n) = estimated cost from current state to goal (heuristic)
      - f(n) = estimated total cost through current state
    
    Graph Search ensures no state is expanded more than once by tracking
    all reached states and their best known g-values.
    
    Args:
        initial_state (PuzzleState): Starting state of the puzzle
        heuristic (str): Which heuristic to use:
            - 'manhattan': h1 (Manhattan Distance)
            - 'linear_conflict': h2 (Linear Conflict)
    
    Returns:
        tuple: (solution_path, nodes_generated, f_values)
            - solution_path: List of actions ['L', 'R', 'U', 'D'] from start to goal
            - nodes_generated: Total number of nodes created (including root)
            - f_values: List of f(n) values along the solution path
    """
    
    # Select heuristic function based on parameter
    h_func = lambda state: state.manhattan_distance() if heuristic == 'manhattan' else state.linear_conflict()
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    # Counter for FIFO tie-breaking when f-values are equal
    counter = 0
    
    # Calculate f-value for initial state
    h_initial = h_func(initial_state)
    f_initial = initial_state.g + h_initial  # g=0 for initial state
    
    # Frontier (open list): Priority queue ordered by f-value
    # Each entry: (f, -g, counter, state)
    # -g for tie-breaking: prefer states with higher g (deeper in tree)
    # counter for FIFO tie-breaking when both f and g are equal
    frontier = [(f_initial, -initial_state.g, counter, initial_state)]
    counter += 1
    
    # Reached states (closed list): Dictionary mapping board state -> best g-value
    # This prevents re-expansion of states and keeps track of best path cost
    reached = {initial_state.get_board_tuple(): initial_state.g}
    
    # Count total nodes generated (including root node)
    nodes_generated = 1
    
    # ========================================================================
    # MAIN SEARCH LOOP
    # ========================================================================
    
    while frontier:
        # Get state with lowest f-value from frontier
        current_f, neg_g, _, current_state = heapq.heappop(frontier)
        current_tuple = current_state.get_board_tuple()
        
        # PRUNING: Skip if we already found a better path to this state
        # This handles stale entries in the priority queue
        if current_tuple in reached and current_state.g > reached[current_tuple]:
            continue
        
        # GOAL TEST: Check if we reached the goal
        if current_state.is_goal():
            # Reconstruct solution path by following parent pointers
            path = []
            f_values = []
            state = current_state
            
            # Trace back from goal to initial state
            while state is not None:
                if state.action is not None:
                    path.append(state.action)
                # Calculate f-value for this state
                h = h_func(state)
                f_values.append(state.g + h)
                state = state.parent
            
            # Reverse to get path from initial to goal
            return path[::-1], nodes_generated, f_values[::-1]
        
        # EXPANSION: Generate and process all successor states
        for successor in current_state.get_successors():
            successor_tuple = successor.get_board_tuple()
            
            # GRAPH SEARCH: Only add successor if:
            # 1. We haven't seen this state before, OR
            # 2. We found a better path (lower g-value) to this state
            if successor_tuple not in reached or successor.g < reached[successor_tuple]:
                # Update best known g-value for this state
                reached[successor_tuple] = successor.g
                
                # Calculate f-value for successor
                h = h_func(successor)
                f = successor.g + h
                
                # Add to frontier with tie-breaking values
                heapq.heappush(frontier, (f, -successor.g, counter, successor))
                counter += 1
                
                # Count this node as generated
                nodes_generated += 1
    
    # No solution found (should not happen for solvable puzzles)
    return None, nodes_generated, None


# ============================================================================
# INPUT/OUTPUT HELPER FUNCTIONS
# ============================================================================

def parse_input_content(content):
    """
    Parse input file content into initial and goal states.
    
    Input format (7 lines):
        Lines 1-3: Initial state (3 rows of 3 integers)
        Line 4: Blank line
        Lines 5-7: Goal state (3 rows of 3 integers)
    
    Args:
        content (str): Raw content from input file
    
    Returns:
        tuple: (initial_board, goal_board) as 3x3 lists
    """
    lines = content.strip().splitlines()
    
    # Parse initial state (lines 1-3)
    initial = []
    for i in range(3):
        row = list(map(int, lines[i].strip().split()))
        initial.append(row)
    
    # Parse goal state (lines 5-7, skipping blank line 4)
    goal = []
    for i in range(4, 7):
        row = list(map(int, lines[i].strip().split()))
        goal.append(row)
    
    return initial, goal


def format_output(initial, goal, depth, nodes, solution, f_values):
    """
    Format solution output according to project requirements.
    
    Output format (12 lines):
        Lines 1-3: Initial state
        Line 4: Blank
        Lines 5-7: Goal state
        Line 8: Blank
        Line 9: Depth (d)
        Line 10: Total nodes generated (N)
        Line 11: Solution actions (space-separated)
        Line 12: F-values along path (space-separated)
    
    Args:
        initial: Initial board state (3x3 list)
        goal: Goal board state (3x3 list)
        depth: Solution depth (number of moves)
        nodes: Total nodes generated
        solution: List of actions ['L', 'R', 'U', 'D']
        f_values: List of f(n) values along solution path
    
    Returns:
        str: Formatted output string
    """
    output = []
    
    # Lines 1-3: Initial state
    for row in initial:
        output.append(' '.join(map(str, row)))
    output.append('')  # Line 4: Blank
    
    # Lines 5-7: Goal state
    for row in goal:
        output.append(' '.join(map(str, row)))
    output.append('')  # Line 8: Blank
    
    # Line 9: Depth
    output.append(str(depth))
    
    # Line 10: Total nodes generated
    output.append(str(nodes))
    
    # Line 11: Solution actions
    output.append(' '.join(solution) if solution else '')
    
    # Line 12: F-values
    output.append(' '.join(map(str, f_values)) if f_values else '')
    
    return '\n'.join(output)


# ============================================================================
# STREAMLIT WEB USER INTERFACE
# ============================================================================
# 
# This section creates an interactive web interface for the 8-puzzle solver.
# Users can upload input files, visualize puzzle states, solve puzzles, and
# download results for both heuristics.
#
# ============================================================================

# Configure page settings
st.set_page_config(
    page_title="8-Puzzle Solver", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS STYLING (Dark Theme)
# ============================================================================
st.markdown("""
<style>
    .main {
        background-color: #1a1d24;
        max-width: 100% !important;
    }
    .stApp {
        background-color: #1a1d24;
    }
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: white !important;
    }
    h3 {
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }
    .puzzle-cell {
        width: 50px;
        height: 50px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        font-weight: bold;
        border: 2px solid #2d3748;
    }
    .puzzle-cell-filled {
        background: #3b82f6;
        color: white;
    }
    .puzzle-cell-empty {
        background: #2d3748;
        color: transparent;
    }
    .puzzle-cell-goal {
        background: #10b981;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.3rem;
        color: #3b82f6 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: white !important;
        font-size: 0.9rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: white;
        background-color: transparent;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #3b82f6 !important;
    }
    .stDownloadButton > button {
        background-color: #10b981 !important;
        color: white !important;
        border: none !important;
        width: 100%;
    }
    .stButton > button {
        background-color: #3b82f6 !important;
        color: white !important;
        border: none !important;
        width: 100%;
        font-weight: bold;
    }
    section[data-testid="stFileUploadDropzone"] {
        background-color: #2d3748;
        border: 2px dashed #4a5568;
        padding: 1rem;
        min-height: 80px;
    }
    section[data-testid="stFileUploadDropzone"] label {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PAGE HEADER
# ============================================================================

st.markdown("""
<div style='background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;'>
    <h1 style='color: white; margin: 0; font-size: 1.8rem;'>🧩 8-Puzzle Solver</h1>
    <p style='color: #bfdbfe; margin: 0.2rem 0 0 0; font-size: 0.85rem;'>A* Algorithm | Manhattan Distance & Linear Conflict Heuristics</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
# Streamlit's session state persists data across reruns

if 'initial' not in st.session_state:
    st.session_state.initial = None
if 'goal' not in st.session_state:
    st.session_state.goal = None
if 'result_h1' not in st.session_state:
    st.session_state.result_h1 = None
if 'result_h2' not in st.session_state:
    st.session_state.result_h2 = None

# ============================================================================
# MAIN LAYOUT - LEFT (35%) AND RIGHT (65%) SPLIT
# ============================================================================

col_left, col_right = st.columns([0.35, 0.65])

# ============================================================================
# LEFT COLUMN - FILE UPLOAD & PUZZLE STATES
# ============================================================================

with col_left:
    # --- File Upload ---
    st.markdown("### 📁 Input File")
    uploaded_file = st.file_uploader("Upload", type=["txt"], label_visibility="collapsed")
    
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        st.session_state.initial, st.session_state.goal = parse_input_content(content)
        st.success(f"✓ {uploaded_file.name}")
    
    # --- Puzzle States (Smaller) ---
    if st.session_state.initial and st.session_state.goal:
        st.markdown("### 🎯 Puzzle States")
        
        col_init, col_goal = st.columns(2)
        
        with col_init:
            st.markdown("<p style='font-size:0.8rem;margin-bottom:0.2rem;'><strong>Initial</strong></p>", unsafe_allow_html=True)
            grid_html = "<div style='margin: 0.2rem 0;'><table style='border-collapse: collapse; margin: 0 auto;'>"
            for row in st.session_state.initial:
                grid_html += "<tr>"
                for cell in row:
                    if cell == 0:
                        grid_html += f"<td class='puzzle-cell puzzle-cell-empty' style='width:22px;height:22px;font-size:0.7rem;padding:1px;'></td>"
                    else:
                        grid_html += f"<td class='puzzle-cell puzzle-cell-filled' style='width:22px;height:22px;font-size:0.7rem;padding:1px;'>{cell}</td>"
                grid_html += "</tr>"
            grid_html += "</table></div>"
            st.markdown(grid_html, unsafe_allow_html=True)
        
        with col_goal:
            st.markdown("<p style='font-size:0.8rem;margin-bottom:0.2rem;'><strong>Goal</strong></p>", unsafe_allow_html=True)
            grid_html = "<div style='margin: 0.2rem 0;'><table style='border-collapse: collapse; margin: 0 auto;'>"
            for row in st.session_state.goal:
                grid_html += "<tr>"
                for cell in row:
                    if cell == 0:
                        grid_html += f"<td class='puzzle-cell puzzle-cell-empty' style='width:22px;height:22px;font-size:0.7rem;padding:1px;'></td>"
                    else:
                        grid_html += f"<td class='puzzle-cell puzzle-cell-goal' style='width:22px;height:22px;font-size:0.7rem;padding:1px;'>{cell}</td>"
                grid_html += "</tr>"
            grid_html += "</table></div>"
            st.markdown(grid_html, unsafe_allow_html=True)
        
        # --- Solve Button ---
        st.markdown("")
        st.markdown('<div id="solve-button-anchor"></div>', unsafe_allow_html=True)
        if st.button("🚀 SOLVE", use_container_width=True):
            with st.spinner("Solving..."):
                # Solve with h1 (Manhattan Distance)
                state_h1 = PuzzleState(st.session_state.initial, st.session_state.goal)
                start_time = time.time()
                solution_h1, nodes_h1, f_values_h1 = a_star_search(state_h1, 'manhattan')
                time_h1 = time.time() - start_time
                
                # Solve with h2 (Linear Conflict)
                state_h2 = PuzzleState(st.session_state.initial, st.session_state.goal)
                start_time = time.time()
                solution_h2, nodes_h2, f_values_h2 = a_star_search(state_h2, 'linear_conflict')
                time_h2 = time.time() - start_time
                
                if solution_h1:
                    st.session_state.result_h1 = {
                        'solution': solution_h1,
                        'nodes': nodes_h1,
                        'depth': len(solution_h1),
                        'f_values': f_values_h1,
                        'time': time_h1
                    }
                
                if solution_h2:
                    st.session_state.result_h2 = {
                        'solution': solution_h2,
                        'nodes': nodes_h2,
                        'depth': len(solution_h2),
                        'f_values': f_values_h2,
                        'time': time_h2
                    }
                
                st.session_state.just_solved = True
                st.rerun()

# ============================================================================
# RIGHT COLUMN - RESULTS DISPLAY
# ============================================================================

with col_right:
    st.markdown("### 📊 Solution Results")
    
    # Auto-scroll after solve to show button at top
    if st.session_state.get('just_solved', False):
        st.components.v1.html("""
            <script>
                setTimeout(function() {
                    window.parent.document.getElementById('solve-button-anchor').scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }, 100);
            </script>
        """, height=0)
        st.session_state.just_solved = False
    
    if st.session_state.result_h1 or st.session_state.result_h2:
        # Create tabs for h1 and h2 results
        tab1, tab2 = st.tabs(["h₁ - Manhattan Distance", "h₂ - Linear Conflict"])
        
        # ------------------------------------------------------------------------
        # TAB 1: h₁ (Manhattan Distance) Results
        # ------------------------------------------------------------------------
        with tab1:
            if st.session_state.result_h1:
                result = st.session_state.result_h1
            # Display metrics (depth, nodes, time)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Depth", result['depth'])
            with col2:
                st.metric("Nodes", result['nodes'])
            with col3:
                st.metric("Time", f"{result['time']:.4f}s")
            
            st.markdown("**✓ Solution Found!**")
            
            # Compare efficiency with h2
            if st.session_state.result_h2:
                efficiency = st.session_state.result_h1['nodes'] - st.session_state.result_h2['nodes']
                if efficiency > 0:
                    st.markdown(f"**Efficiency:** {efficiency} more nodes than h₂")
                elif efficiency < 0:
                    st.markdown(f"**Efficiency:** {abs(efficiency)} fewer nodes than h₂")
            
            # Display solution path
            st.markdown("**Solution Path:**")
            st.code(' '.join(result['solution']), language=None)
            
            # Display f-values
            st.markdown("**F-Values:**")
            st.code(' '.join(map(str, result['f_values'])), language=None)
            
            # Download button for output file
            st.download_button(
                label="💾 Download Output (h1)",
                data=format_output(
                    st.session_state.initial, 
                    st.session_state.goal,
                    result['depth'],
                    result['nodes'],
                    result['solution'],
                    result['f_values']
                ),
                file_name="output_h1.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # ------------------------------------------------------------------------
        # TAB 2: h₂ (Linear Conflict) Results
        # ------------------------------------------------------------------------
        with tab2:
            if st.session_state.result_h2:
                result = st.session_state.result_h2
                
                # Display metrics (depth, nodes, time)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Depth", result['depth'])
                with col2:
                    st.metric("Nodes", result['nodes'])
                with col3:
                    st.metric("Time", f"{result['time']:.4f}s")
                
                st.markdown("**✓ Solution Found!**")
                
                # Compare efficiency with h1
                if st.session_state.result_h1:
                    efficiency = st.session_state.result_h2['nodes'] - st.session_state.result_h1['nodes']
                    if efficiency > 0:
                        st.markdown(f"**Efficiency:** {efficiency} more nodes than h₁")
                    elif efficiency < 0:
                        st.markdown(f"**Efficiency:** {abs(efficiency)} fewer nodes than h₁")
                
                # Display solution path
                st.markdown("**Solution Path:**")
                st.code(' '.join(result['solution']), language=None)
                
                # Display f-values
                st.markdown("**F-Values:**")
                st.code(' '.join(map(str, result['f_values'])), language=None)
                
                # Download button for output file
                st.download_button(
                    label="💾 Download Output (h2)",
                    data=format_output(
                        st.session_state.initial,
                        st.session_state.goal,
                        result['depth'],
                        result['nodes'],
                        result['solution'],
                        result['f_values']
                    ),
                    file_name="output_h2.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    else:
        st.info("👆 Upload an input file and click SOLVE PUZZLE to see results")
