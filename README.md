# Data-driven Rulesets for Soft Artificial Life (DRSALife) Model for Reaction-Diffusion Dynamics

This repository contains the codebase for the Data-driven Rulesets for Soft Artificial Life (DRSALife) model, focused on data-driven modeling of reaction-diffusion emergent dynamical systems (forming Turing patterns).

## Description

To run the experiments (17 in total: 6 for Gaussian noise, 6 for observability near equilibrium state, and 5 for temporal sparsity):

1. Run `CA_simdb.py` to simulate experimental data based on finite difference simulations with added noise, sparsity, and observability.
2. Train models (with varying hyperparameters) for each experiment by navigating to each experiment directory and running `start_parallel.bat`. Wait for the default model (`realization_0`) training to finish, then execute `run_parallel.bat` to train several models in parallel.
3. Process overall results by going to the parent/root directory, running `result_pics_pde_processing.py`, then `result_parallel_processing.py`, and finally `result_combine.py`.

## Configuration

The `config_params.json` file contains all the main configuration parameters for creating training data and processing results after training the models. You can modify this file to set experiment-specific parameters such as noise levels, sparsity, observability, and other settings. Make sure to review and update `config_params.json` before starting new experiments to ensure the correct setup.

**Configuration parameters:**

- **Experiment parameters:**
  - `snr_l`: Noise levels. Default: `[1, 10, 25, 30, 35, 100]`
  - `sparsity_l`: Sparsity levels. Default: `[10, 30, 40, 50, 80]`
  - `equil_order_l`: Observability levels. Default: `[-0.5, -1.0, -1.6, -1.8, -2.0, -2.2]`

- **Training data parameters:**
  - `nb_cases`: Number of finite difference simulations per experiment for training. Default: `10`
  - `t`: Finite difference simulation time duration. Default: `25`
  - `a`: Finite difference simulation PDE parameter. Default: `2.8e-4`
  - `b`: Finite difference simulation PDE parameter. Default: `5e-3`
  - `tau`: Finite difference simulation PDE parameter. Default: `0.1`
  - `k`: Finite difference simulation PDE parameter. Default: `-0.005`
  - `size`: Finite difference simulation grid size. Default: `60`
  - `dt`: Finite difference simulation time step duration. Default: `0.001`
  - `out_sample`: Finite difference simulation sampling duration. Default: `1.0`

- **Test simulation parameters:**
  - `tsteps`: Test simulation time duration. Default: `8`
  - `num_reruns`: Number of test simulations. Default: `100`
  - `num_processors`: Number of processors for parallel processing. Default: `8`
  - `num_trials`: Number of training trials for hyperparameter tuning. Used to pick the best model out of `num_trials` trained models. Default: `50`

## Results

- Results can be found in the `Visualizations` folder. In each experiment folder, snapshots of simulations using finite difference vs. learned CA model (best realization) can be found. Estimated PDE parameters are also available in a CSV file.
- Statistics for all results from all experiments can be found in `Combined_results.xlsx`. Refresh the Excel sheets to display recently processed results.

## Author

- Saumitra Dwivedi (saumitra.dwivedi@ntnu.no) https://orcid.org/0000-0001-7493-6950
