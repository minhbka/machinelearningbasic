import numpy as np
from k_mean_clustering.kmean_hand_writting_number.display_network import *
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.io import loadmat
mnist = loadmat('data/mnist-original.mat')
data = mnist['data'].T
# N = 200
# X = data[np.random.choice(data.shape[0], N)]/255.
# plt.axis('off')
# A = display_network(X.T, 10, N/10)
# f2 = plt.imshow(A, interpolation='nearest' )
# plt.gray()
# plt.savefig('mnist_ex.png', bbox_inches='tight', dpi = 600)
# plt.show()

K = 10          # number of clusters
N = 10000       # number of samples
X = data[np.random.choice(data.shape[0], N)]
kmeans = KMeans(n_clusters=K).fit(X)
pred_label = kmeans.predict(X)

plt.axis('off')
A = display_network(kmeans.cluster_centers_.T, 10, 1)
f2 = plt.imshow(A, interpolation='nearest',cmap=plt.cm.jet)

plt.savefig('mnist_centroids.png', bbox_inches='tight', dpi = 600)
plt.show()

N0 = 10
K = 10
X1 = np.zeros((N0 * K, 784))
X2 = np.zeros((N0 * K, 784))

for k in range(K):
    Xk = X[pred_label == k, :]

    # random points in each cluster
    X1[N0 * k: N0 * k + N0, :] = Xk[:N0, :]

    # N0 nearest points
    centroid_k = kmeans.cluster_centers_[k]
    neigh = NearestNeighbors(N0)  # get 5 nearest neighbors
    neigh.fit(Xk)

    X2[N0 * k: N0 * k + N0, :] = Xk[neigh.kneighbors([centroid_k], N0)[1][0], :]


# random points in cluster
plt.axis('off')
A = display_network(X1.T, K, N0)
f2 = plt.imshow(A, interpolation='nearest' )
plt.gray()
plt.savefig('mnist_cluster_random.png', bbox_inches='tight', dpi = 600)
plt.show()