# Dendrogram for SGMoE

## Description
GitHub repository for **Hierarchical Clustering and Model Selection** in SoftmaxMixture of Experts (SGMoE), using dendrograms of mixing measures.

---

## Dependencies
- **Python version**: 3.12  
- **Suggested IDE**: PyCharm  
- **Required packages**:
  - Standard Python libraries
  - `numpy`
  - `scipy`
  - `scikit-learn`
  - `matplotlib`

---

## Usage

- Run the main experiment file:
  ```bash
  python HD_NMoE/execute.py

- Adjust `running_experiment = "experiment 5"` to change the experiment. For instance, you want to run experiment 3, then change it to `running_experiment = "experiment 3"`
## Experiment Configuration
- For a certain experiment, such as experiment 5, its details are stored in `HD_NMoE\experiment 5\description.json`. The details include:
  - `'name'`: `'experiment 5'`  
  - `'type'`: `'voronoi_loss_dic'`  
  - `'n_tries'`: `1` – Trial index.  
  - `'n_features'`: `1` – Input X's number of features. The output Y's number of features is 1 by default in our experimental scope.  
  - `'K'`: `2` – Number of true experts.  
  - `'K_max'`: `4` – Number of experts in overfitted setting.  
  - `'n_min'`: `100` – Minimum sample size.  
  - `'n_max'`: `100000` – Maximum sample size.  
  - `'n_iter'`: `200` – Total number of sample sizes.  
  - `'alphak'`: `[[-8], [25]]` – True gating coefficients; the last expert's coefficients are 0's by default.  
  - `'betak'`: `[[15, -5], [-20, 20]]` – True expert coefficients.  
  - `'sigmak'`: `[0.3, 0.4]` – True sigmas.  
  - `'favourable'`: `true` – Favourable setting.  
  - `'seed'`: `2035` – Current random seed.  
  - `'spacing_type'`: `'log'` – Spacing type of sample sizes (`'linear'` or `'log'`).


## Plotting Voronoi Loss
- Run `python plotting/plot_avarage_voronoi_loss_full.py`

- Adjust the parameters for your setting:

```python i_plot = [2, 4]
# The i_plot list indicates which type of loss to plot:
# 0: Exact loss (D1)
# 1: Exact loss (D2)
# 2: Over loss (D2)
# 3: Merge loss (D1)
# 4: Merge loss (D2)

# Parameters for selecting the range of data points
n_begin = 0 # Start index (0-based)
n_end = 200 # End index (exclusive)

error_bar = 0 # Set to 1 to include error bars in the plot
dual = 0 # Set to 1 to include dual data, which means that you have two sets of data to plot
normal_regressor = 0 # Set to 1 if you want to use a normal linear regressor along with RANSAC

# The range of trials to be plotted
trials_1 = range(1, 2)
trials_2 = range(1000, 1014)

experiment_name = "experiment 5"
setting = "voronoi_loss_K4-2_100_100000_200"
```

 ## Plotting DIC
- Run `plotting/histogram_dic_full.py`

- Adjust the parameters for your setting:

```python # Path to data folder
directory = "../data/experiment 5"
setting = "voronoi_loss_K4-2_100_100000_80"
true_argmin_dic = 2 # True K_0

# Change this if you have 2 ranges of data:
# 0: only first range
# 1: only second range
# 2: both ranges
dual = 2
if dual == 0:
    indices = range(1000, 1024)
elif dual == 1:
    indices = range(1100, 1111)
elif dual == 2:
    indices = list(range(1000, 1024)) + list(range(1100, 1111))
else:
    raise ValueError("Invalid value for 'dual'. Use 0, 1, or 2.") ```
