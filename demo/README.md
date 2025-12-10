# Demo Notebooks

This folder contains demonstration Jupyter notebooks for the Repeatable Statistical Inference Algorithms with synthetic distributions.

## Notebooks Overview

1. **`demo-distribution.ipynb`**
   - Demonstrates the various probability distributions implemented in the framework
   - Shows visualization and sampling capabilities

2. **`demo-stat-query.ipynb`**
   - Basic statistical query demonstrations
   - Shows non-repeatable statistical testing

3. **`demo-repeatable-stat-query.ipynb`**
   - Demonstrates repeatable statistical query with α-quantization
   - Shows exact repeatability across multiple trials
   - Includes both Monte Carlo and importance sampling examples

## Running the Notebooks

All notebooks have been configured to import modules from the parent directory. The first cell in each notebook adds the parent directory to the Python path:

```python
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
```

To run the notebooks:

1. Navigate to the demo directory:
   ```bash
   cd demo
   ```

2. Start Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

3. Open any notebook and run all cells

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Jupyter Notebook/Lab