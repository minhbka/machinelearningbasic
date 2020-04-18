
from __future__ import print_function
import numpy as np
from k_mean_clustering.kmean_with_random_data.support import kmeans, kmeans_display
from sklearn.cluster import KMeans
np.random.seed(18)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis=0)

K = 3 # 3 clusters

original_label = np.asarray([0]*N + [1]*N + [2]*N).T

kmeans_display(X, original_label)

(centroids, labels, it) = kmeans(X, K)

print('Centers found by our algorithm:\n', centroids[-1])


kmeans_display(X, labels[-1], 'res.pdf')

model = KMeans(n_clusters=3, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(model.cluster_centers_)
pred_label = model.predict(X)
kmeans_display(X, pred_label, 'res_scikit.pdf')
