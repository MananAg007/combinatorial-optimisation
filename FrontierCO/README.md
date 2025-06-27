---
license: apache-2.0
task_categories:
- text-generation
tags:
- code
---

# FrontierCO: Benchmark Dataset for Frontier Combinatorial Optimization

## Overview

**FrontierCO** is a curated benchmark suite for evaluating ML-based solvers on large-scale and real-world **Combinatorial Optimization (CO)** problems. The benchmark spans **8 classical CO problems** across **5 application domains**, providing both training and evaluation instances specifically designed to test the frontier of ML and LLM capabilities in solving NP-hard problems.

code for evaluating agent https://github.com/sunnweiwei/CO-Bench?tab=readme-ov-file#evaluation-on-frontierco

code for running classifical solver, generate training data, evaluating neural solver: https://github.com/sunnweiwei/FrontierCO

Evaluation Code: https://github.com/sunnweiwei/FrontierCO

---

## Dataset Structure

Each subdirectory corresponds to a specific CO task:

```
FrontierCO/
├── CFLP/
│   ├── easy_test_instances/
│   ├── hard_test_instances/
│   ├── valid_instances/
│   └── config.py
├── CPMP/
├── CVRP/
├── FJSP/
├── MIS/
├── MDS/
├── STP/
├── TSP/
└── ...
```

Each task folder contains:

* `easy_test_instances/`: Benchmark instances that are solvable by SOTA human-designed solvers.
* `hard_test_instances/`: Instances that remain computationally intensive or lack known optimal solutions.
* `valid_instances/` *(if applicable)*: Additional instances for validation or development.
* `config.py`: Metadata about instance format, solver settings, and reference solutions.

---

## Tasks Covered

The benchmark currently includes the following problems:

* **MIS** – Maximum Independent Set
* **MDS** – Minimum Dominating Set
* **TSP** – Traveling Salesman Problem
* **CVRP** – Capacitated Vehicle Routing Problem
* **CFLP** – Capacitated Facility Location Problem
* **CPMP** – Capacitated p-Median Problem
* **FJSP** – Flexible Job-shop Scheduling Problem
* **STP** – Steiner Tree Problem

Each task includes:

* Easy and hard test sets with varying difficulty and practical relevance
* Training and validation instances where applicable, generated using problem-specific generators
* Reference results for classical and ML-based solvers

---

## Data Sources

Instances are sourced from a mix of:

* Public repositories (e.g., [TSPLib](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/), [CVRPLib](http://vrp.galgos.inf.puc-rio.br/))
* DIMACS and PACE Challenges
* Synthetic instance generators used in prior ML and optimization research
* Manual curation from recent SOTA solver evaluation benchmarks

For tasks lacking open benchmarks, we include high-quality synthetic instances aligned with real-world difficulty distributions.

---

## Usage

To use this dataset, clone the repository and select the task of interest. Each `config.py` file documents the format and how to parse or evaluate the instances.

```bash
git clone https://huggingface.co/datasets/CO-Bench/FrontierCO
cd FrontierCO/CFLP
```

Load a data instance
```python
from config import load_data
instance = load_data('easy_test_instances/i1000_1.plc')
print(instance)
```

Generate a solution
```python
# Your solution generation code goes here.
# For example:
solution = my_solver_func(**instance)
```

### Evaluate a solution
```python
from config import eval_func
score = eval_func(**instance, **solution)
print("Evaluation score:", score)
```

---

## Citation

If you use **FrontierCO** in your research or applications, please cite the following paper:

```bibtex
@misc{feng2025comprehensive,
  title={A Comprehensive Evaluation of Contemporary ML-Based Solvers for Combinatorial Optimization},
  author={Shengyu Feng and Weiwei Sun and Shanda Li and Ameet Talwalkar and Yiming Yang},
  year={2025},
}
```

---

## License

This dataset is released under the MIT License. Refer to `LICENSE` file for details.

---
