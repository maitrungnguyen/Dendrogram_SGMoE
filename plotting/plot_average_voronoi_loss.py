import json
import numpy
import matplotlib.pyplot as plt
import numpy as np

sample_list = []
exact_data = []
over_data = []
merge_data = []

for i in range (4, 7):
    print(i)
    temp = []
    with open(f"../data/experiment 1/voronoi_loss_K3_15000_100000_40_{i}.json", "r") as file:
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
plt.plot(sample_list, average_merge, label="Merge")

plt.plot(sample_list, reg.coef_*x + reg.intercept_, label="Linear Regression")
plt.legend()
plt.show()


