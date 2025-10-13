# Data-driven Rulesets for Soft Artifical Life (DRSALife) model for Reaction-Diffusion Dynamics

This repository contains the codebase for Data-driven Rulesets for Soft Artifical Life (DRSALife) model in context of Data-driven modeling of Reaction Diffusion emergent dynamical systems (forming Turing patterns).

## Description
In order to run the experiments (there are 17 experiments in total; 6 for gaussian noise, 6 for observability near equillibrium state, and 5 for temporal sparsity):

- Run CA_simdb.py to simulate experimental data based on finite difference simulations added with noise, sparsity, observability.
- Train models (with varying hyperparameters) for each experiment by going to each experiment directory and running start_parallel.bat, wait for default model (realization_0) training to finish and then executing run_parallel.bat to train several models in parallel.
- Process overall results by going to the parent/root directory, first running result_pics_pde_processing.py, then running result_parallel_processing.py and finally running result_combine.py.

## Configuration

- The `config_params.json` file contains all the main configuration parameters for creating training data and processing results after training the models. You can modify this file to set experiment-specific parameters such as noise levels, sparsity, observability, and other settings. Make sure to review and update `config_params.json` before starting new experiments to ensure the correct setup.

Following describes the config paramaters:

- Experiments parameters:
-- snr_l: Noise levels. Default values -> [1,10,25,30,35,100]
-- sparsity_l: Sparisty levels levels. Default values -> [10,30,40,50,80]
-- equil_order_l: Observability levels levels. Default values -> [-0.5,-1.0,-1.6,-1.8,-2.0,-2.2]

- Training data parameters
-- nb_cases: Number of finite difference simulation per experiment for training. Default value -> 10,
-- t: Finite difference simulation time duration. Default value -> 25
-- a: Finite difference simulation PDE parameter. Default value -> 2.8e-4
-- b: Finite difference simulation PDE parameter. Default value -> 5e-3
-- tau: Finite difference simulation PDE parameter. Default value -> 0.1
-- k: Finite difference simulation PDE parameter. Default value -> -0.005
-- size: Finite difference simulation grid size. Default value -> 60
-- dt: Finite difference simulation time step duration. Default value -> 0.001
-- out_sample: Finite difference simulation sampling duration. Default value -> 1.0

- Test simulations parameters
-- tsteps: Test simulation time duration. Default value -> 8
-- num_reruns: Number of Test simulations. Default value ->  100
-- num_processors: Number of processors to be used for parallel processing. Default value -> 8
-- num_trials: Number of training trials used for hyperparameter tuning. This parameter is used to pick the best model of of 'num_trials' trained models. Default value -> 50


## Results:

- Results can be found in the Visualizations folder. In each experiment folder, snapshots of simulations using finite difference vs learnt CA model (best realization) can be found. Also, estimated PDE paramaters can be found in a csv file.
- A statistics of all results from all experiments can be found in Combined_results.xlsx. You need to refresh the excel sheets to display recently processed results.

## Author of Codebase
- Saumitra Dwivedi (saumitra.dwivedi@ntnu.no) https://orcid.org/0000-0001-7493-6950
