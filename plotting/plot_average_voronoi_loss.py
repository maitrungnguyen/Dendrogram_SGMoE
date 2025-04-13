import json
import numpy
import matplotlib.pyplot as plt
import numpy as np

sample_list = []
exact_data = []
exact_data_d2 = []
over_data = []
merge_data = []
merge_data_d2 = []

for i in range (1138, 1195):
    print(i)
    temp = []
    with open(f"../data/experiment 3/voronoi_loss_K3_100_100000_80_{i}.json", "r") as file:
        temp = (json.load(file))

    temp_exact = np.array(temp["exact_d1"])
    exact_data.append(temp_exact[:, 1])

    temp_exact_d2 = np.array(temp["exact_d2"])
    exact_data_d2.append(temp_exact_d2[:, 1])

    temp_over = np.array(temp["over"])
    over_data.append(temp_over[:, 1])

    temp_merge = np.array(temp["merge_d2"])
    merge_data.append(temp_merge[:, 1])

    temp_merge_d2 = np.array(temp["merge_d2"])
    merge_data_d2.append(temp_merge_d2[:, 1])

    sample_list = (temp_merge[:, 0])

exact_data = np.array(exact_data)
over_data = np.array(over_data)
merge_data = np.array(merge_data)
exact_data_d2 = np.array(exact_data_d2)
merge_data_d2 = np.array(merge_data_d2)

average_exact = np.mean(exact_data, axis = 0)
average_over = np.mean(over_data, axis = 0)
average_merge = np.mean(merge_data, axis = 0)
average_exact_d2 = np.mean(exact_data_d2, axis = 0)
average_merge_d2 = np.mean(merge_data_d2, axis = 0)

#print(average_merge)

#linear regression
x = np.array(sample_list).reshape(-1, 1)
y = average_exact.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x, y)
print(reg.coef_)
print(reg.intercept_)
label = "$\mathcal{D}_1(\widehat G_n, G_{*})$"
plt.plot(sample_list, average_exact, color='red', label = label)

plt.plot(x, reg.coef_*x + reg.intercept_, color='black', linestyle='-.', label=str(np.round(
    reg.intercept_[0],1)) + \
                                                                        "$n^{" + str(np.round(reg.coef_[0][0],5)) + "}$")

# Labels & Legend
plt.xlabel("log(sample size)", fontsize=14)
plt.ylabel("log(loss)", fontsize=14)
plt.title("Exacted-Setting", fontsize=16)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


