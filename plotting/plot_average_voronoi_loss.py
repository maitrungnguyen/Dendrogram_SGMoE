import json
import numpy
import matplotlib.pyplot as plt
import numpy as np

sample_list = []
exact_data = []
over_data = []
merge_data = []

for i in range (1, 57):
    print(i)
    temp = []
    with open(f"../data/experiment 1/voronoi_loss_K3_1000_100000_40_{i}.json", "r") as file:
        temp = (json.load(file))

    temp_exact = np.array(temp["exact"])
    exact_data.append(temp_exact[:, 1])

    temp_over = np.array(temp["over"])
    over_data.append(temp_over[:, 1])

    temp_merge = np.array(temp["merge"])
    merge_data.append(temp_merge[:, 1])

    sample_list = (temp_merge[:, 0])

exact_data = np.array(exact_data)
over_data = np.array(over_data)
merge_data = np.array(merge_data)
print(merge_data)

average_exact = np.mean(exact_data, axis = 0)
average_over = np.mean(over_data, axis = 0)
average_merge = np.mean(merge_data, axis = 0)

print(average_merge)

#linear regression
x = np.array(sample_list).reshape(-1, 1)
y = average_merge.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x, y)
print(reg.coef_)
print(reg.intercept_)
label = "$\mathcal{D}_2(\widehat G_n, G_{*})$"
plt.plot(sample_list, average_merge, color='red', label = label)

plt.plot(x, reg.coef_*x + reg.intercept_, color='black', linestyle='-.', label=str(np.round(
    reg.intercept_[0],1)) + \
                                                                        "$n^{" + str(np.round(reg.coef_[0][0],5)) + "}$")

# Labels & Legend
plt.xlabel("log(sample size)", fontsize=14)
plt.ylabel("log(loss)", fontsize=14)
plt.title("Log-Log Scale Linear Regression", fontsize=16)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


