# AI code to implement K-means clustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# load the iris dataset
iris = load_iris()

# extract the features
X = iris.data[:, :2]

# create a K-means model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# fit the model to the data
kmeans.fit(X)

# get the cluster labels for the data
y_pred = kmeans.labels_

# visualize the clusters
plt.figure(1, figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
plt.xticks(())
plt.yticks(())
plt.show()