from voronoi_experiment_w_nmin import voronoi_experiment_w_nmin
import json
import numpy as np

with open("experiment 5/description.json", "r") as file:
    data = json.load(file)

while True:
    # try:
        print("Check")
        voronoi_experiment_w_nmin(data["n_min"], data["n_max"], data["n_iter"], data["n_features"], data["K"], data["K_max"],data["alphak"],
                                  data["betak"], data["sigmak"],
                                  data["n_tries"],
                                    data["name"], favourable=data["favourable"]
                                  )

        print("Done")
        data["n_tries"] += 1
        with open("experiment 5/description.json", "w") as file:
            json.dump(data, file)
        print(data["n_tries"])
    # except:
    #     break

