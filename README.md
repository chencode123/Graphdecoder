# Introduction

Implementation for AI-driven method for exhaustive hazard scenario generation in chemical process systems

# install

```bash
conda env create -f environment.yml
conda activate graphdecoder
```

# Dataset

This dataset was generated based on rule-based logic. The input and output are built independently using different scripts.

## Input

- `data_forming.ipynb`: Generates node features for Process Flow Diagrams (PFDs).  
  These features serve as the structural input for downstream tasks.

## Output

- Hazard scenarios are created using **yEd Graph Editor** and saved in `total.graphml`.
- `json_generate.ipynb`: Reads `total.graphml` and converts the scenarios into structured JSON format, saved as `graph_relations_labeled_structured.json`.

## Integration

- `data_preprocess.ipynb`: Combines the input features and output scenarios into a unified binary file `graph_data.bin`.  
  This file serves as the dataset for model training, validation, and testing.


# Usage 

To run the training script:

```bash
python train.py --config configs/base.yaml --epochs 50

```

To run the generate.py:

```bash
python train.py --config configs/base.yaml --epochs 50

```

To run the generate_case_study.py:

```bash
python train.py --config configs/base.yaml --epochs 50

```
