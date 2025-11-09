# 🧩 8-Puzzle Solver

A web-based 8-puzzle solver using the A* algorithm with two heuristics: Manhattan Distance (h₁) and Linear Conflict (h₂).

## 🌟 Features

- **A* Search Algorithm**: Efficient pathfinding with graph search optimization
- **Dual Heuristics**:
  - **h₁ (Manhattan Distance)**: Sum of distances each tile is from its goal position
  - **h₂ (Linear Conflict)**: Manhattan Distance + 2×(linear conflicts)
- **Interactive Web UI**: Clean, modern interface built with Streamlit
- **Real-time Comparison**: Compare efficiency between both heuristics
- **Visual Puzzle Display**: See initial and goal states side-by-side
- **Downloadable Results**: Export solutions as text files

## 🚀 Live Demo

[View Live Demo](https://ai-project1.streamlit.app/) *(Coming soon)*

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## 💻 Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Vikram739/AI-Project1.git
cd AI-Project1
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📝 Input File Format

Create a text file with the following 7-line format:

```
START STATE
0 1 3
4 2 5
7 8 6

GOAL STATE
1 2 3
4 5 6
7 8 0
```

- Line 1: "START STATE"
- Lines 2-4: Initial puzzle configuration (0 represents empty space)
- Line 5: "GOAL STATE"
- Lines 6-8: Goal puzzle configuration

### Sample Input Files

Three example files are included:
- `Input1.txt` - Easy puzzle (depth 5)
- `Input2.txt` - Medium puzzle (depth 10)
- `Input3.txt` - Hard puzzle (depth 23)

## 🎯 How to Use

1. **Upload Input File**: Click "Browse files" and select your input file
2. **View Puzzle States**: Initial and goal states are displayed
3. **Solve**: Click "SOLVE PUZZLE" button
4. **Compare Results**: View both h₁ and h₂ solutions in separate tabs
5. **Download**: Export results using the download buttons

## 📊 Output Information

For each heuristic, the app displays:
- **Depth**: Solution path length
- **Nodes**: Number of nodes expanded
- **Time**: Execution time in seconds
- **Solution Path**: Sequence of moves (L, R, U, D)
- **F-Values**: f(n) values for each step
- **Efficiency Comparison**: Node count difference between h₁ and h₂

## 🛠️ Technology Stack

- **Python 3.14**
- **Streamlit 1.51.0** - Web framework
- **heapq** - Priority queue implementation
- **copy.deepcopy** - State management

## 📁 Project Structure

```
AI-Project1/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── .gitignore         # Git ignore rules
├── Input1.txt         # Sample input (easy)
├── Input2.txt         # Sample input (medium)
└── Input3.txt         # Sample input (hard)
```

## 🧮 Algorithm Details

### A* Search
- Uses priority queue with f(n) = g(n) + h(n)
- Graph search: tracks visited states to avoid repetition
- Tie-breaking: (f, -g, counter, state)
- Move order: L, R, U, D (for consistency)

### Manhattan Distance (h₁)
```
h(n) = Σ |current_x - goal_x| + |current_y - goal_y|
```

### Linear Conflict (h₂)
```
h(n) = h₁(n) + 2 × conflicts
```
A linear conflict occurs when two tiles are in their goal row/column but in reversed order.

## 📈 Performance

On a complex puzzle (Input3, depth 23):
- **h₁**: 970 nodes expanded
- **h₂**: 525 nodes expanded
- **Improvement**: 46% fewer nodes with h₂

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Authors

- **Vikram739** - [GitHub Profile](https://github.com/Vikram739)

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- NYU Tandon School of Engineering - AI Course Fall 2025
- A* algorithm based on classic AI pathfinding techniques
- Linear Conflict heuristic for improved efficiency
