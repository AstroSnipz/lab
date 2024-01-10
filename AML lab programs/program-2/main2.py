import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = pd.DataFrame(iris.data)
X.columns = ["Sepal_length", "Sepal_width", "Petal_length", "Petal_width"]
Y = pd.DataFrame(iris.target)
Y.columns = ["Labels"]

model1 = KMeans(n_clusters=3, n_init=10)
model1.fit(X)

plt.figure(figsize=(14,7))
colormap = np.array(["red", "lime", "black"])

plt.subplot(1,3,1)
plt.scatter(X.Petal_length, X.Petal_width, c=colormap[Y.Labels], s=40)
plt.title("original")
plt.xlabel("Petal_length")
plt.ylabel("petal_width")

plt.subplot(1,3,2)
plt.scatter(X.Petal_length, X.Petal_width, c=colormap[model1.labels_], s=40)
plt.title("KMeans")
plt.xlabel("Petal_length")
plt.ylabel("petal_width")

scalar = StandardScaler()
scalar.fit(X)
xsa = scalar.transform(X)
xs = pd.DataFrame(xsa, columns = X.columns)

gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
y_gmm = gmm.predict(xs)

plt.subplot(1,3,3)
plt.scatter(X.Petal_length, X.Petal_width, c=colormap[y_gmm], s=40)
plt.title("Guassian Mixture")
plt.xlabel("petal_length")
plt.ylabel("petal_width")

plt.show()