import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib.gridspec as gridspec

def getImages(folder, imgSize=(32, 32)):
    images, labels = [], []
    for dirName in os.listdir(folder):
        dirPath = os.path.join(folder, dirName)
        if os.path.isdir(dirPath):
            for file in os.listdir(dirPath):
                with Image.open(os.path.join(dirPath, file)) as img:
                    resizedImg = img.convert('L').resize(imgSize)
                    images.append(np.array(resizedImg).flatten())
                    labels.append(int(dirName))
    return np.array(images), np.array(labels)

def applyPCA(images, numComps):
    U, S, Vt = np.linalg.svd(images - np.mean(images, axis=0), full_matrices=False)
    return np.dot(U, np.diag(S))[:, :numComps]

def doKMeans(data, numClusters, numInit=10, maxIters=300):
    bestInertia = np.inf
    bestCenters = None
    bestLabels = None

    for _ in range(numInit):
        centers = data[np.random.choice(data.shape[0], numClusters, replace=False)]
        for _ in range(maxIters):
            dists = np.sqrt(((data - centers[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(dists, axis=0)
            newCenters = np.array([data[labels == i].mean(axis=0) for i in range(numClusters)])
            if np.all(centers == newCenters):
                break
            centers = newCenters

        inertia = sum(((data - centers[labels])**2).sum(axis=1))
        if inertia < bestInertia:
            bestInertia = inertia
            bestCenters = centers
            bestLabels = labels

    return bestLabels, bestCenters

def displayClusters(images, clusters, numClusters, saveTo, title):
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(numClusters, 5)
    for clusterNum in range(numClusters):
        clusterImgs = images[clusters == clusterNum]
        for i in range(min(5, len(clusterImgs))):
            ax = plt.subplot(gs[clusterNum, i])
            ax.imshow(clusterImgs[i].reshape(32, 32), cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title(f'Group {clusterNum + 1}')
    plt.suptitle(title)
    plt.savefig(os.path.join(saveTo, f'{title}.png'))
    plt.show()

def plotClusters(data, clusters, saveTo, title):
    plt.figure(figsize=(8, 6))
    for i in np.unique(clusters):
        plt.scatter(data[clusters == i, 0], data[clusters == i, 1], label=f'Group {i+1}')
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(saveTo, f'{title}.png'))
    plt.show()

if __name__ == "__main__":
    imgFolder = "PIE"
    resultFolder = "result/GMM"

    images, _ = getImages(imgFolder)

    for dim in [None, 200, 80]:
        imagesToUse = images if dim is None else applyPCA(images, dim)
        clusterLabels, _ = doKMeans(imagesToUse, 3)
        title = f'Cool K-means Clustering with {dim} PCA' if dim else 'K-means Clusters on Regular Images'
        displayClusters(images, clusterLabels, 3, resultFolder, title)
        plotClusters(imagesToUse, clusterLabels, resultFolder, f'K-means Cluster Spread with {dim} PCA')
