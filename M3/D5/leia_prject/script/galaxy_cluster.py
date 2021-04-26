
''' Star wars galaxy info '''
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib.pyplot as plt
import seaborn as sns
print('Setup complete')

galaxy_df = pd.read_csv("galaxies.csv")
galaxy_df.head()

model = KMeans(n_clusters=3)
model.fit(galaxy_df)
labels = model.predict(galaxy_df)  # this target grup find from ata
labels

points = galaxy_df.values


def separate_labels(labels, points):
    galaxy_0 = []
    galaxy_1 = []
    galaxy_2 = []
    for i in range(labels.shape[0]):
        if labels[i] == 0:
            galaxy_0.append(points[i])
        elif labels[i] == 1:
            galaxy_1.append(points[i])
        else:
            galaxy_2.append(points[i])
    # YOUR CODE HERE
    return np.array(galaxy_0), np.array(galaxy_1), np.array(galaxy_2)


galaxy_0, galaxy_1, galaxy_2 = separate_labels(labels, points)

print(galaxy_0.shape)
print(galaxy_1.shape)
print(galaxy_2.shape)

colors = ['b', 'c', 'y']

galaxy_0_plot = plt.scatter(
    galaxy_0[:, 0],  galaxy_0[:, 1], marker='o', color=colors[0])
galaxy_1_plot = plt.scatter(
    galaxy_1[:, 0], galaxy_1[:, 1], marker='o', color=colors[1])
galaxy_2_plot = plt.scatter(
    galaxy_2[:, 0], galaxy_2[:, 1], marker='o', color=colors[2])

plt.legend((galaxy_0_plot, galaxy_1_plot, galaxy_2_plot),
           ('1-Galaxy', '2-Galaxy', '3-Galaxy'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.rcParams["figure.figsize"] = (20, 10)
plt.savefig("gaalxy_all.png")
# plt.annotate(label="All in one ")
plt.show()

mean_gal_0 = np.mean(galaxy_0[1])
mean_gal_1 = np.mean(galaxy_1[1])
mean_gal_2 = np.mean(galaxy_2[1])
x_max = galaxy_0[:, 0].max(axis=0)
x_max
finder = np.where(galaxy_0[:, 0] == x_max)
cord = galaxy_0[finder]

x_p = cord[0, 0]

y_p = cord[0, 1]

plt.scatter(galaxy_0[:, 0],  galaxy_0[:, 1], marker='o', color=colors[0])
plt.plot(x_p, y_p, 'ro')
plt.text(x_p, y_p, "Beby Yodo here ")
plt.savefig("beby.png")
plt.show()

''' Yoda planet PCAs '''
