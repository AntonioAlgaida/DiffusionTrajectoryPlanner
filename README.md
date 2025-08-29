# Diffusion Trajectory Planner


<p align="center">
  A PyTorch implementation of a conditional Denoising Diffusion Probabilistic Model (DDPM) for multi-modal trajectory prediction. This project trains a U-Net on the Waymo Open Motion Dataset to generate plausible, human-like driving behaviors by denoising trajectories from pure Gaussian noise.
</p>

<p align="center">
  <!-- <a href="https://arxiv.org/abs/2508.18397" target="_blank"> -->
    <!-- <img src="https://img.shields.io/badge/ArXiv-2508.18397-b31b1b.svg?style=flat-square" alt="ArXiv Paper"> -->
  <!-- </a> -->
  <!-- <a href="http://arxiv.org/licenses/nonexclusive-distrib/1.0/" target="_blank"> -->
    <!-- <img src="https://img.shields.io/badge/Paper%20License-arXiv%20Perpetual-b31b1b.svg?style=flat-square" alt="ArXiv Paper License"> -->
  <!-- </a> -->
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/Code%20License-MIT-blue.svg?style=flat-square" alt="Code License">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Made%20with-PyTorch-EE4C2C.svg?style=flat-square&logo=pytorch" alt="Made with PyTorch">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat-square&logo=python" alt="Python 3.10">
</p>

<p align="center">
  A project by <strong>Antonio Guillen-Perez</strong> | 
  <a href="https://antonioalgaida.github.io/" target="_blank"><strong>Portfolio</strong></a> | 
  <a href="https://www.linkedin.com/in/antonioguillenperez/" target="_blank"><strong>LinkedIn</strong></a> | 
  <a href="https://scholar.google.com/citations?user=BFS6jXwAAAAJ" target="_blank"><strong>Google Scholar</strong></a>
</p>


- [Diffusion Trajectory Planner](#diffusion-trajectory-planner)
  - [1. Project Mission](#1-project-mission)
  - [2. Technical Approach](#2-technical-approach)
  - [3. Project Structure](#3-project-structure)
  - [4. Setup and Installation](#4-setup-and-installation)
  - [5. Data Preparation Pipeline](#5-data-preparation-pipeline)
      - [Step 0: Download the Waymo Open Motion Dataset](#step-0-download-the-waymo-open-motion-dataset)
      - [Step 1: Parse Raw Data to `.npz`](#step-1-parse-raw-data-to-npz)
      - [Step 2: Featurize Data for Diffusion](#step-2-featurize-data-for-diffusion)
      - [Step 3: Compute Normalization Statistics](#step-3-compute-normalization-statistics)
  - [6. Training](#6-training)
  - [7. Evaluation](#7-evaluation)
      - [Using the DDIM Sampler (Recommended for fast evaluation)](#using-the-ddim-sampler-recommended-for-fast-evaluation)
      - [Using the DDPM Sampler (High-fidelity, but much slower)](#using-the-ddpm-sampler-high-fidelity-but-much-slower)
  - [8. Results and Analysis](#8-results-and-analysis)
    - [Quantitative Results](#quantitative-results)
    - [Qualitative Examples](#qualitative-examples)
  - [9. Future Work \& Potential Extensions](#9-future-work--potential-extensions)
  - [Acknowledgements](#acknowledgements)

---


<!-- *Qualitative result showing a fan-out of 20 diverse trajectory predictions (purple) against the ground truth (red).* -->

## 1. Project Mission

The development of safe and intelligent autonomous vehicles hinges on their ability to reason about an uncertain and multi-modal future. Traditional deterministic approaches, which predict a single "best guess" future, often fail to capture the rich distribution of plausible behaviors a human driver might exhibit. This can lead to policies that are overly conservative or dangerously indecisive in complex, interactive scenarios.

This project directly confronts this challenge by fundamentally shifting the modeling paradigm from deterministic regression to conditional generative modeling. The mission is to develop a policy that learns to represent and sample from the entire, complex distribution of plausible expert behaviors, enabling the generation of driving behaviors that are not only safe but also contextually appropriate, diverse, and human-like.

## 2. Technical Approach

The core of this project is a **Conditional Denoising Diffusion Probabilistic Model (DDPM)**. The model is trained to reverse a gradual noising process on normalized trajectory data.

1.  **Data Pipeline:** The raw Waymo Open Motion Dataset is processed through a multi-stage pipeline (`src/data_processing/`). This includes parsing, intelligent filtering of static/unreliable scenarios, and feature extraction to produce `(Context, Target)` pairs.
2.  **Normalization:** To ensure a stable training target, all 8-second future trajectories (`Target`) are normalized to a `[-1, 1]` range using statistics computed across the entire training set.
3.  **Context Encoding:** The scene `Context` is encoded by a powerful **StateEncoder**. It uses dedicated sub-networks for each entity (ego history, agents, map, etc.) and fuses them using a **Transformer Encoder** to produce a single, holistic `scene_embedding`.
4.  **Denoising Model:** The primary model is a **Conditional 1D U-Net**. It takes a noisy trajectory `x_t` and learns to predict the original noise `ε` that was added, conditioned on the `scene_embedding` and the noise level `t`.
5.  **Sampling:** At inference, we start with pure Gaussian noise `x_T` and iteratively apply the trained denoiser to recover a clean, normalized trajectory `x_0`, which is then de-normalized back into meter-space. This repository implements both the slow, stochastic **DDPM** sampler and the fast, deterministic **DDIM** sampler.
6. **Training:** The model is trained using the AdamW optimizer with a cosine learning rate schedule and gradient clipping to ensure stable convergence. The training process is monitored using TensorBoard, tracking metrics such as training and validation loss.
7. **Evaluation:** The trained model is evaluated on the Waymo validation set using standard metrics like `minADE`, `minFDE`, and `MissRate@2m`. The evaluation script supports both DDPM and DDIM sampling methods.


To ensure stability, all trajectory data is **normalized** to a `[-1, 1]` range before being used in the diffusion process.

## 3. Project Structure

```
diffusion-trajectory-planner/
├── configs/
│   └── main_config.yaml
├── data/
│   ├── (gitignored) processed_npz/
│   └── (gitignored) featurized_v3_diffusion/
├── models/
│   ├── (gitignored) checkpoints/
│   └── (gitignored) normalization_stats.pt
├── notebooks/
│   ├── 1_analyze_source_data.ipynb
│   ├── 2_analyze_featurized_data.ipynb
│   └── 3_analyze_final_results.ipynb
├── src/
│   ├── data_processing/
│   │   ├── parser.py
│   │   ├── featurizer_diffusion.py
│   │   └── compute_normalization_stats.py
│   ├── diffusion_policy/
│   │   ├── dataset.py
│   │   ├── networks.py
│   │   └── train.py
│   └── evaluation/
│       └── evaluate_prediction.py
└── README.md
```

## 4. Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/diffusion-trajectory-planner.git
    cd diffusion-trajectory-planner
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create --name virtuoso_env python=3.10
    conda activate virtuoso_env
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 5. Data Preparation Pipeline

This is a multi-step process to convert the raw Waymo data into a format suitable for training.

#### Step 0: Download the Waymo Open Motion Dataset
Download the `.tfrecord` files for the motion prediction task from the [Waymo Open Dataset website](https://waymo.com/open/download/). Place the `scenario` folder containing the training and validation shards into a directory of your choice.

#### Step 1: Parse Raw Data to `.npz`
This initial step converts the raw `.tfrecord` files into a more accessible NumPy format.
> **Note:** This `parser.py` script is a prerequisite and is assumed to be adapted from a previous project.

Update `configs/main_config.yaml` with the correct path to your raw data, then run the parser.

#### Step 2: Featurize Data for Diffusion
This script processes the `.npz` files, performs intelligent data curation and filtering, and saves the final `(Context, Target)` pairs as `.pt` files.

```bash
python -m src.data_processing.featurizer_diffusion
```
When prompted, choose `[d]` to delete any old data and start fresh.

#### Step 3: Compute Normalization Statistics
This crucial step computes the min/max statistics needed to normalize the trajectory data. This must be run after featurizing the data.

```bash
python -m src.data_processing.compute_normalization_stats
```
This will create a `models/normalization_stats.pt` file, which is required for training and evaluation.

## 6. Training

Once the data preparation is complete, you can launch the main training script.

```bash
python -m src.diffusion_policy.train
```

The script will create a new, timestamped directory in `runs/DiffusionPolicy_Training/` for this run. All TensorBoard logs and model checkpoints will be saved there.

You can monitor the training progress live using TensorBoard:
```bash
tensorboard --logdir runs
```
Navigate to `http://localhost:6006/` in your browser. Look for a smooth, downward-trending validation loss curve that plateaus at a small, non-zero value.

## 7. Evaluation

After training, you can evaluate your best model checkpoint to get quantitative metrics. The script supports both the fast `ddim` sampler and the high-fidelity `ddpm` sampler.

#### Using the DDIM Sampler (Recommended for fast evaluation)
```bash
python -m src.evaluation.evaluate_prediction \
    --checkpoint runs/DiffusionPolicy_Training/YOUR_RUN_TIMESTAMP/checkpoints/best_model.pth \
    --sampler ddim \
    --steps 50
```

#### Using the DDPM Sampler (High-fidelity, but much slower)
```bash
python -m src.evaluation.evaluate_prediction \
    --checkpoint runs/DiffusionPolicy_Training/YOUR_RUN_TIMESTAMP/checkpoints/best_model.pth \
    --sampler ddpm
```

The script will print a summary of the final metrics (`minADE`, `minFDE`, `MissRate@2m`) and save a detailed `.json` report in the same directory as your checkpoint.

## 8. Results and Analysis

*TBC*

### Quantitative Results

The model was evaluated on the full Waymo Open Motion Dataset validation set. The following metrics were achieved using the DDIM sampler with 50 steps:

| Metric      | Value |
|-------------|-------|
| minADE      | TBD   |
| minFDE      | TBD   |
| MissRate@2m | TBD   |

### Qualitative Examples

The following visualizations, generated by the `notebooks/3_analyze_final_results.ipynb` notebook, demonstrate the model's ability to generate diverse and contextually appropriate trajectories.

<!-- **(Placeholder: Insert one or two of your best "fan-out" plots here after you generate them.)** -->
![Qualitative Result 1](placeholder.png)
<!-- *Caption: A challenging left-turn scenario where the model correctly generates a multi-modal distribution of plausible turning paths.* -->
*TBC*

## 9. Future Work & Potential Extensions

This project provides a strong foundation for several exciting research directions:

-   **PCA Latent Diffusion:** Implementing the diffusion process in a low-dimensional PCA latent space, as proposed in the MotionDiffuser paper, to improve speed, smoothness, and accuracy.
-   **Guided Sampling:** Implementing a guided sampling framework to enforce rules (e.g., collision avoidance) or achieve specific goals at inference time.
-   **Multi-Agent Prediction:** Extending the `StateEncoder` and denoiser to jointly predict trajectories for multiple interacting agents.
-   **Closed-Loop Planning:** Integrating the generative model as a proposal distribution within a Model Predictive Control (MPC) loop for closed-loop vehicle control.

## Acknowledgements

This work is heavily inspired by and builds upon the foundational concepts introduced in papers such as [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) and [MotionDiffuser](https://arxiv.org/abs/2306.03083).