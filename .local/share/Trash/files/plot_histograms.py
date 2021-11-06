import sys
import os
import numpy as np
import yaml as yl
import matplotlib.pyplot as plt
import argparse
import glob
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot values from LLVM histograms.')
    parser.add_argument('filename', type=str, help='CSV file to read')
    parser.add_argument('output', type=str, help='output file for image')
    args = parser.parse_args()

    df = pd.read_csv(args.filename)
    features = [str(i) for i in range(65)]
    x = df.loc[:, features].values
    y = df.loc[:, ['class']]

    # Standartize data
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    finalDf = pd.concat([principalDf, df[['class']]], axis = 1)

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    c_min, c_max = int(y.min()), int(y.max())+1
    print(f"Min: {c_min}, Max: {c_max}")
    targets = [i for i in range(c_max)]
    colors = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])[:c_max]
    print(f"Targets: {targets}, colors: {colors}")
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf["class"] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'PC1'], finalDf.loc[indicesToKeep, 'PC2'], c=color, s = 50)

    ax.legend(targets)
    ax.grid()

    plt.show()
    plt.savefig(args.output, bbox_inches='tight')
