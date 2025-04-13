# Dendrogram for SGMoE

## Description
Github for Hierarchical Clustering and Model Selection for SoftmaxMoE Using Dendrograms of Mixing Measures.

## Dependencies
- Python version: 3.12.
- Suggested IDE: Pycharm.
- Required Packages:
  - Standard Python libraries.
  - Numpy
  - Scipy
  - Scikit-learn
  - Matplotlib
## Usage
- Run `HD_NMoE/execute.py` to execute the experiments.
- Adjust `running_experiment = "experiment 5"` to change the experiment. For instance, you want to run experiment 3, then change it to `running_experiment = "experiment 3"`
- For a certain experiment, such as experiment 5, its details are stored in `HD_NMoE\experiment 5\description.json`. The details include:
  - "name": "experiment 5" - The name of the experiment.
  - "type": "voronoi_loss_dic" - The type of experiment, producing Voronoi Loss function and DIC results.
  - "n_tries": 1 - Order of trials, this mean this is the second running.
  - "n_features": 1 - Input X's number of feature. The ouput Y's number of feature is 1 by default in our experimental scope.
  - "K": 2 - Number of true experts.
  - "K_max": 4 - Number of component in overfitted setting.
  - "n_min": 100 - Minimum sample size.
  - "n_max": 100000 - Maximum sample size.
  - "n_iter": 200 - Total number of sample sizes.
  - "alphak": [[-8], [25]] - True gating coefficients, the last expert's coefficients are 0's by default.
  - "betak": [[15, -5], [-20, 20]] - True expert coefficients.
  - "sigmak": [0.3, 0.4] - True sigmas.
  - "favourable": true - Favourable setting.
  - "seed": 2035 - Current random seed.
  - "spacing_type": "log" - Spacing type of sample sizes. Linear spacing "linear" or logarithmic spacing "log".

## Plotting
- For plotting Voronoi Loss, run `plotting/plot_avarage_voronoi_loss_full.py`. Adjust the parameter for 
