import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

def loadImageData(folder, img_size=(32, 32)):
    imgList, labelList = [], []
    for dirName in os.listdir(folder):
        dirPath = os.path.join(folder, dirName)
        if os.path.isdir(dirPath):
            for imgFile in os.listdir(dirPath):
                with Image.open(os.path.join(dirPath, imgFile)) as img:
                    resizedImg = img.convert('L').resize(img_size)
                    imgList.append(np.array(resizedImg).flatten())
                    labelList.append(int(dirName))
    return np.array(imgList), np.array(labelList)

def doPCA(data, n=3):
    meanAdjusted = data - np.mean(data, axis=0)
    covMatrix = np.cov(meanAdjusted, rowvar=False)
    values, vectors = np.linalg.eigh(covMatrix)
    sortedIndices = np.argsort(values)[::-1]
    return vectors[:, sortedIndices[:n]]

def getEigenfaces(imgs, comps=3):
    vectors = doPCA(imgs, comps)
    faces = vectors.T.reshape((comps, 32, 32))
    return faces

def showFaces(faces, title, path):
    fig, ax = plt.subplots(1, len(faces), figsize=(10, 5))
    for i, face in enumerate(faces):
        ax[i].imshow(face, cmap='gray')
        ax[i].set_title(f'Face {i+1}')
        ax[i].axis('off')
    plt.suptitle(title)
    plt.savefig(os.path.join(path, 'Faces.png'))
    plt.show()

def splitData(data, labels, size=0.3, seed=42):
    np.random.seed(seed)
    shuffled = np.random.permutation(len(data))
    testSize = int(len(data) * size)
    testIndices = shuffled[:testSize]
    trainIndices = shuffled[testSize:]
    return data[trainIndices], data[testIndices], labels[trainIndices], labels[testIndices]

def predictKNN(test, trainData, trainLabels, k=1):
    dist = np.sqrt(((trainData - test) ** 2).sum(axis=1))
    nearest = np.argsort(dist)[:k]
    return np.bincount(trainLabels[nearest]).argmax()

def classifyPCA(train, trainL, test, testL, comps=3, neighbors=1):
    vectors = doPCA(train, comps)
    transformedTrain = (train - np.mean(train, axis=0)).dot(vectors)
    transformedTest = (test - np.mean(test, axis=0)).dot(vectors)
    predictions = np.array([predictKNN(t, transformedTrain, trainL, neighbors) for t in transformedTest])
    pieLabel = 26
    accuracyPie = np.mean(testL[testL != pieLabel] == predictions[testL != pieLabel])
    accuracyOwn = np.mean(testL[testL == pieLabel] == predictions[testL == pieLabel])
    return accuracyPie, accuracyOwn

def plotPCA(data, labels, components, title, folder):
    plt.figure()
    if components == 3:
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', marker='o')
    else:
        scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(os.path.join(folder, f'{title}.png'))
    plt.show()

def plotPCADistribution(data, labels, vectors, label, folder):
    transformed = (data - np.mean(data, axis=0)).dot(vectors)
    plt.figure()
    if vectors.shape[1] == 3:
        ax = plt.axes(projection='3d')
        ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], c=labels, cmap='viridis', alpha=0.5)
        ax.scatter(transformed[labels == label, 0], transformed[labels == label, 1], transformed[labels == label, 2], c='red', label='Highlight')
    else:
        plt.scatter(transformed[:, 0], transformed[:, 1], c=labels, cmap='viridis', alpha=0.5)
        plt.scatter(transformed[labels == label, 0], transformed[labels == label, 1], c='red', label='Highlight')
    plt.legend()
    plt.title('PCA Distribution')
    plt.savefig(os.path.join(folder, f'PCA_Distribution.png'))
    plt.show()

def errorAnalysis(data, vectors, path):
    transformed = (data - np.mean(data, axis=0)).dot(vectors)
    reconstructed = transformed.dot(vectors.T) + np.mean(data, axis=0)
    errors = np.mean(np.square(data - reconstructed), axis=1)
    plt.figure()
    plt.hist(errors, bins=50, color='blue', alpha=0.7)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Error Analysis')
    plt.savefig(os.path.join(path, 'Error_Analysis.png'))
    plt.show()

if __name__ == "__main__":
    imgFolder = "PIE"
    resultsFolder = "result/PCA"
    os.makedirs(resultsFolder, exist_ok=True)

    imgData, labels = loadImageData(imgFolder)
    trainData, testData, trainLabels, testLabels = splitData(imgData, labels, 0.3)

    sampledIndices = random.sample(range(len(trainData)), 500)
    sampledData = trainData[sampledIndices]
    sampledLabels = trainLabels[sampledIndices]

    faces = getEigenfaces(sampledData)
    showFaces(faces, "Faces", resultsFolder)

    vectors2D = doPCA(sampledData, 2)
    pca2D = (sampledData - np.mean(sampledData, axis=0)).dot(vectors2D)
    plotPCA(pca2D, sampledLabels, 2, "PCA 2D", resultsFolder)

    vectors3D = doPCA(sampledData, 3)
    pca3D = (sampledData - np.mean(sampledData, axis=0)).dot(vectors3D)
    plotPCA(pca3D, sampledLabels, 3, "PCA 3D", resultsFolder)

    plotPCADistribution(sampledData, sampledLabels, vectors2D, 26, resultsFolder)
    plotPCADistribution(sampledData, sampledLabels, vectors3D, 26, resultsFolder)

    resultsPath = os.path.join(resultsFolder, 'results.txt')
    with open(resultsPath, 'w') as file:
        for dim in [40, 80, 200]:
            vectors = doPCA(trainData, dim)
            accPie, accOwn = classifyPCA(trainData, trainLabels, testData, testLabels, dim, 1)
            file.write(f'PCA {dim} - Others: {accPie:.2f}\n')
            file.write(f'PCA {dim} - Own: {accOwn:.2f}\n')

    for dim in [40, 80, 200]:
        vectors = doPCA(imgData, dim)
        errorAnalysis(imgData, vectors, resultsFolder)

