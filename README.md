# LLaMA-ES

This repository contains the implementation of LLaMA-ES, an innovative approach for tuning the hyperparameters of Evolution Strategies (ES), specifically the Covariance Matrix Adaptation Evolution Strategy (CMA-ES). Our method optimizes numerical black-box functions by leveraging a Large Language Model (LLM) to iteratively suggest parameter improvements based on the history of previous runs.

We use the LLaMA3 model with 70 billion parameters to conduct experiments on various numerical benchmark optimization problems. This dynamic and intelligent parameter adjustment process significantly enhances the performance of CMA-ES, demonstrating its effectiveness in achieving competitive parameter tuning results.

The approach will be published at ESANN in Oktober 2024.

## Key Features

- **Hyperparameter Tuning**: Utilizes LLaMA3 to suggest optimal CMA-ES parameters.
- **Iterative Improvement**: Adjusts parameters based on historical run data.
- **Benchmark Testing**: Validated on several numerical optimization problems.

## Installation

To install the necessary dependencies, run:

```bash
git clone https://github.com/yourusername/LLaMA-ES.git
cd LLaMA-ES
pip install -r requirements.txt
