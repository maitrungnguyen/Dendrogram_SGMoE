from voronoi_experiment_w_nmin import voronoi_experiment_w_nmin
import json
import numpy as np


with open("experiment 5/description.json", "r") as file:
    data = json.load(file)

while True:
    try:
        print("Trial:",data["n_tries"])
        voronoi_experiment_w_nmin(data["n_min"], data["n_max"], data["n_iter"], data["n_features"], data["K"], data["K_max"],data["alphak"],
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
        with open("experiment 5/description.json", "w") as file:
            json.dump(data, file)
    except:
        data["seed"] += 7
        with open("experiment 5/description.json", "w") as file:
            json.dump(data, file)
        continue

