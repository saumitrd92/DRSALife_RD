This repository contains the codebase for Data-driven Rulesets for Soft Artifical Life (DRSALife) model in context of Data-driven modeling of Reaction Diffusion emergent dynamical systems (forming Turing patterns).

In order to run the experiments (there are 17 experiments in total; 6 for gaussian noise, 6 for observability near equillibrium state, and 5 for temporal sparsity):\n


-- Run CA_simdb.py to simulate experimental data based on finite difference simulations added with noise, sparsity, observability.\n
-- Train models (with varying hyperparameters) for each experiment by going to each experiment directory and running start_parallel.bat, wait for default model (realization_0) training to finish and then executing run_parallel.bat to train several models in parallel.\n
-- Process overall results by going to the parent/root directory, first running result_pics_pde_processing.py, then running result_parallel_processing.py and finally running result_combine.py.\n

Results:\n
-- Results can be found in the Visualizations folder. In each experiment folder, snapshots of simulations using finite difference vs learnt CA model (best realization) can be found. Also, estimated PDE paramaters can be found in a csv file.\n
-- A statistics of all results from all experiments can be found in Combined_results.xlsx. You need to refresh the excel sheets to display recently processed results.\n

Author of Codebase\n
-- Saumitra Dwivedi (saumitra.dwivedi@ntnu.no) https://orcid.org/0000-0001-7493-6950
