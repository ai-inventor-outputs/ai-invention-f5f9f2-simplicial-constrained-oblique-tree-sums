# AI Invention Research Repository

This repository contains artifacts from an AI-generated research project.

## Research Paper

[![Download PDF](https://img.shields.io/badge/Download-PDF-red)](https://github.com/ai-inventor-outputs/ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums/blob/main/paper/paper.pdf) [![LaTeX Source](https://img.shields.io/badge/LaTeX-Source-orange)](https://github.com/ai-inventor-outputs/ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums/tree/main/paper) [![Figures](https://img.shields.io/badge/Figures-5-blue)](https://github.com/ai-inventor-outputs/ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums/tree/main/figures)

## Quick Start - Interactive Demos

Click the badges below to open notebooks directly in Google Colab:

### Jupyter Notebooks

| Folder | Description | Open in Colab |
|--------|-------------|---------------|
| `dataset_iter1_sc_ots_tabular` | SC-OTS Tabular Benchmark Suite: 18 Datasets with G... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums/blob/main/dataset_iter1_sc_ots_tabular/demo/data_code_demo.ipynb) |
| `experiment_iter2_simplicial_comp` | Simplicial Complex Construction Pipeline Validatio... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums/blob/main/experiment_iter2_simplicial_comp/demo/method_code_demo.ipynb) |
| `experiment_iter2_baseline_benchm` | Baseline Benchmarks: FIGS, XGBoost, XGBoost-Oracle... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums/blob/main/experiment_iter2_baseline_benchm/demo/method_code_demo.ipynb) |
| `experiment_iter2_sc_ots_simplici` | SC-OTS: Simplicial-Constrained Oblique Tree Sums E... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums/blob/main/experiment_iter2_sc_ots_simplici/demo/method_code_demo.ipynb) |
| `evaluation_iter3_comprehensive_s` | Comprehensive Statistical Evaluation of SC-OTS Exp... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums/blob/main/evaluation_iter3_comprehensive_s/demo/eval_code_demo.ipynb) |
| `experiment_iter3_sc_ots_ablation` | SC-OTS Ablation: Simplicial vs Random-Matched vs U... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums/blob/main/experiment_iter3_sc_ots_ablation/demo/method_code_demo.ipynb) |
| `experiment_iter3_sc_ots_v2_simpl` | SC-OTS v2: Simplicial-Constrained Oblique Tree Sum... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums/blob/main/experiment_iter3_sc_ots_v2_simpl/demo/method_code_demo.ipynb) |
| `evaluation_iter4_definitive_fina` | Definitive Final Synthesis: SC-OTS Hypothesis Verd... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums/blob/main/evaluation_iter4_definitive_fina/demo/eval_code_demo.ipynb) |
| `experiment_iter4_xgboost_simplic` | XGBoost Simplicial-Constraint Ablation: 5 Modes × ... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums/blob/main/experiment_iter4_xgboost_simplic/demo/method_code_demo.ipynb) |
| `experiment_iter4_tda_vs_baseline` | TDA vs. Baselines: Interaction Detection Benchmark... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums/blob/main/experiment_iter4_tda_vs_baseline/demo/method_code_demo.ipynb) |

### Research & Documentation

| Folder | Description | View Research |
|--------|-------------|---------------|
| `research_iter1_sc_ots_survey` | SC-OTS Survey... | [![View Research](https://img.shields.io/badge/View-Research-green)](https://github.com/ai-inventor-outputs/ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums/blob/main/research_iter1_sc_ots_survey/demo/research_demo.md) |
| `research_iter1_ph_foundations` | PH Foundations... | [![View Research](https://img.shields.io/badge/View-Research-green)](https://github.com/ai-inventor-outputs/ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums/blob/main/research_iter1_ph_foundations/demo/research_demo.md) |

## Repository Structure

Each artifact has its own folder with source code and demos:

```
.
├── <artifact_id>/
│   ├── src/                     # Full workspace from execution
│   │   ├── method.py            # Main implementation
│   │   ├── method_out.json      # Full output data
│   │   ├── mini_method_out.json # Mini version (3 examples)
│   │   └── ...                  # All execution artifacts
│   └── demo/                    # Self-contained demos
│       └── method_code_demo.ipynb # Colab-ready notebook (code + data inlined)
├── <another_artifact>/
│   ├── src/
│   └── demo/
├── paper/                       # LaTeX paper and PDF
├── figures/                     # Visualizations
└── README.md
```

## Running Notebooks

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badges above to run notebooks directly in your browser.
No installation required!

### Option 2: Local Jupyter

```bash
# Clone the repo
git clone https://github.com/ai-inventor-outputs/ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums.git
cd ai-invention-f5f9f2-simplicial-constrained-oblique-tree-sums

# Install dependencies
pip install jupyter

# Run any artifact's demo notebook
jupyter notebook exp_001/demo/
```

## Source Code

The original source files are in each artifact's `src/` folder.
These files may have external dependencies - use the demo notebooks for a self-contained experience.

---
*Generated by AI Inventor Pipeline - Automated Research Generation*
