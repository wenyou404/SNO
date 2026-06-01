# SNO

This repository contains code for **Self-supervised Neural Operator (SNO)**, a framework for solving partial differential equations with solver-free training data generation.

SNO combines three components:

- a physics-informed sampler for generating training functions,
- a function encoder (FE) for compact function representations,
- an operator learning model for mapping PDE conditions and source terms to solutions.

At the moment, this repository only includes the code for the **1D nonlinear reaction-diffusion equation** experiment. Code for the remaining examples in the paper will be uploaded as soon as possible, depending on the publication timeline.

## Repository Structure

```text
1D_reaction_diffusion/
└── code/
    ├── dataset.py
    ├── FE.ipynb
    └── OL.ipynb
```

## 1D Reaction-Diffusion Code

The current release includes:

- `dataset.py`: shared configuration, model definitions, data-generation utilities, derivative helpers, checkpoint utilities, and parameter settings used by the notebooks.
- `FE.ipynb`: code for training and inspecting the Function Encoder used in the paper.
- `OL.ipynb`: code for the final operator learning stage.

The default paths in `dataset.py` are relative to the `1D_reaction_diffusion` directory. Generated figures, data, and checkpoints are expected under:

```text
1D_reaction_diffusion/figure/
1D_reaction_diffusion/data/
1D_reaction_diffusion/checkpoints/
```

Large generated datasets and trained checkpoints are not included in this initial code release.

## Requirements

The code is written in Python/JAX. The main dependencies used by the notebooks include:

- `jax`
- `jaxlib`
- `flax`
- `optax`
- `orbax-checkpoint`
- `jaxopt`
- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`

Please install versions compatible with your local CUDA/JAX environment.

## Citation

If you use this code, please cite the corresponding SNO paper once it is available.
