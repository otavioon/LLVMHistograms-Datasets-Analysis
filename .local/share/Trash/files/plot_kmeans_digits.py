import sys
import numpy as np
import yaml as yl
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def load_dataset(filename, data_dir):
    fin = open(filename, 'r')
    data = yl.load(fin)
    fin.close()

    dataset = []
    
    for phase in ['training', 'validation', 'test']:
        for label, samples in data[phase].items():
            for sample in samples:
                histogram = np.load('{}/{}/{}.npz'.format(data_dir, label, sample))
                histogram = histogram['values']
                dataset.append(list(histogram))

    return dataset
    
n_clusters = int(sys.argv[1])
filename = sys.argv[2]
data_dir = sys.argv[3]
if len(sys.argv) > 4:
    plot_type = sys.argv[4]
else:
    plot_type = 'no_zone'

data = load_dataset(filename, data_dir)
print(len(data))
    
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

plt.figure(1)
plt.clf()

if plot_type == 'zone':
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.imshow(Z, interpolation="nearest",
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired, aspect="auto", origin="lower", )
    
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
        
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )

else:
    predicted = kmeans.predict(reduced_data)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=predicted, s=5, cmap=plt.cm.Paired)
    
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c='black', s=50, alpha=0.5,
        zorder=10,
    )

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
#plt.show()

filename = filename.replace('.yaml', '_clusters.pdf')
plt.savefig(filename, bbox_inches='tight',pad_inches = 0)
