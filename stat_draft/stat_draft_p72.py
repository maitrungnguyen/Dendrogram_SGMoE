from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


ptsd = [10, 20, 25, 28, 31, 35, 37, 38, 38, 39, 39, 42, 46]
healthy = [23, 39, 40, 41, 43, 47, 51, 58, 63, 66, 67, 69, 72]

#boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=[ptsd, healthy], color="skyblue")
plt.title("Boxplot of Strength Data")
plt.xlabel("Strength")
plt.ylabel("Group")
plt.xticks([0, 1], ["PTSD", "Healthy"])
plt.show()

#histogram
plt.figure(figsize=(8, 6))
sns.histplot(ptsd, color="skyblue", kde=True)
sns.histplot(healthy, color="red", kde=True)
plt.title("Histogram of Strength Data")
plt.xlabel("Strength")
plt.ylabel("Density")
plt.legend(["PTSD", "Healthy"])
plt.show()




