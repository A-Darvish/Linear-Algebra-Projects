# Linear Algebra Projects

This repository contains two Python projects showcasing applications of linear algebra in computational problems. These projects leverage concepts such as matrix operations, null space computations, and iterative methods to solve practical problems.

## Overview

### Project 1: Null Space Basis and Normalization
This project focuses on calculating the null space basis of a stochastic matrix and normalizing it to derive a probabilistic vector. It demonstrates the following concepts:

- Transformation of matrices into echelon and reduced echelon forms.
- Computation of the null space basis for a given matrix.
- Normalization of basis vectors to derive a unique probabilistic solution.

#### Key Functions
- `echelon_form(matrix)`: Converts a matrix to its echelon form.
- `reduced_echelon_form(matrix)`: Converts an echelon matrix to reduced echelon form.
- `null_space_basis(matrix, dim)`: Computes the basis for the null space of the matrix.

#### How to Run
1. Execute the script:
   ```bash
   python p1.py
   ```
2. Enter the dimensions and rows of the stochastic matrix when prompted.

#### Example Output
- Transformation of the stochastic matrix to “M - I” form.
- Null space basis vectors.
- A normalized probabilistic vector derived from the null space.

---

### Project 2: Iterative PageRank Calculation
This project implements an iterative method to compute PageRank-like values using adjacency matrices and rank propagation.

#### Key Concepts
- Construction of a transition matrix from a graph's adjacency matrix.
- Iterative rank propagation to compute steady-state rank values.
- Application of matrix-vector multiplication in iterative processes.

#### Key Functions
- `mat_multiply(mat, vector, dim)`: Multiplies a matrix with a vector to compute the next rank state.

#### How to Run
1. Prepare an input graph as a text file (e.g., `graph.txt`) with edges listed in the format:
   ```
   (start_node) -> (end_node)
   ```
2. Update the file path in the script.
3. Execute the script:
   ```bash
   python p2.py
   ```

#### Example Output
- Transition matrix derived from the graph.
- Rank values at each iteration.

---

## Dependencies
- **Python 3.x**
- **NumPy**: Ensure that NumPy is installed before running the scripts.

Install dependencies using pip:
```bash
pip install numpy
```

## Project Structure
```
|-- p1  # Null Space Basis and Normalization
|-- p2  # Iterative PageRank Calculation
    |-- graph.txt  # Example input graph (used in Project 2)
```

