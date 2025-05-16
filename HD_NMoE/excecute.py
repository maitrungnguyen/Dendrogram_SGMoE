from voronoi_experiment_w_nmin import voronoi_experiment_w_nmin
from voronoi_experiment_w_nmin import dsc_aic_bic_icl_test
import json
import numpy as np

running_experiment = "experiment 5"

with open(f"{running_experiment}/description.json", "r") as file:
    data = json.load(file)

while True:
#     try:
        print("Trial:",data["n_tries"])
        dsc_aic_bic_icl_test(data["n_min"], data["n_max"], data["n_iter"], data["n_features"], data["K"], data["K_max"],data["alphak"],
                                  data["betak"], data["sigmak"],
                                  data["n_tries"],
                                    data["name"], favourable=data["favourable"],
                                    seed=data["seed"],
                                  exact_enable= True,
                                    spacing_type=data["spacing_type"],
                                  )

        print("Done")
        data["n_tries"] += 1
        data["seed"] += 7
        with open(f"{running_experiment}/description.json", "w") as file:
            json.dump(data, file)
    # except:
    #     data["seed"] += 7
    #     with open(f"{running_experiment}/description.json", "w") as file:
    #         json.dump(data, file)
    #     continue

