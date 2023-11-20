import numpy as np
from PIL import Image
import os

def loadImages(folder, tag, imgSize=(32, 32)):
    images = []
    for fileName in os.listdir(folder):
        filePath = os.path.join(folder, fileName)
        if os.path.isfile(filePath) and fileName.lower().endswith(('.jpg', '.png')):
            with Image.open(filePath) as img:
                resized = img.convert('L').resize(imgSize)
                images.append((np.array(resized).flatten(), tag))
    return images
def splitData(data, size=0.3):
    np.random.shuffle(data)
    splitPoint = int(len(data) * (1 - size))
    return data[:splitPoint], data[splitPoint:]
def scaleData(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std
def doPCA(X, numComps):
    centered = X - np.mean(X, axis=0)
    covMatrix = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(covMatrix)
    ordered = np.argsort(values)[::-1]
    return centered.dot(vectors[:, ordered[:numComps]])
def linearSVM(X, y, C, epochs=1000, lr=0.01):
    samples, features = X.shape
    w = np.zeros(features)
    b = 0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            cond = yi * (np.dot(xi, w) - b) >= 1
            if cond:
                w -= lr * (2 * C * w)
            else:
                w -= lr * (2 * C * w - np.dot(xi, yi))
                b -= lr * yi
    return w, b
def predictSVM(X, w, b):
    return np.sign(np.dot(X, w) - b)
def calcAccuracy(actual, predicted):
    return np.sum(actual == predicted) / len(actual)
pieFolder = "PIE"
resultFolder = "result/SVM"
personFolders = sorted([folder for folder in os.listdir(pieFolder) if folder.isnumeric()])
otherPics = [img for id in personFolders[:25] for img in loadImages(os.path.join(pieFolder, id), 1)]
selfPics = loadImages(os.path.join(pieFolder, "26"), -1)
otherTrain, otherTest = splitData(otherPics)
selfTrain, selfTest = splitData(selfPics)
trainData = otherTrain + selfTrain
testData = otherTest + selfTest
X_train, y_train = zip(*trainData)
X_test, y_test = zip(*testData)
X_train = scaleData(np.array(X_train))
X_test = scaleData(np.array(X_test))
y_train = np.array(y_train)
y_test = np.array(y_test)
results = {}
C_values = [1e-2, 1e-1, 1]
for C in C_values:
    w, b = linearSVM(X_train, y_train, C)
    preds = predictSVM(X_test, w, b)
    results[f'Raw pics, C={C}'] = calcAccuracy(y_test, preds)
    for comps in [80, 200]:
        trainPCA = doPCA(X_train, comps)
        testPCA = doPCA(X_test, comps)
        w, b = linearSVM(trainPCA, y_train, C)
        preds = predictSVM(testPCA, w, b)
        acc = calcAccuracy(y_test, preds)
        results[f'PCA {comps} components, C={C}'] = acc
if not os.path.exists(resultFolder):
    os.makedirs(resultFolder)
resultsPath = os.path.join(resultFolder, 'SVM_resul.txt')
with open(resultsPath, 'w') as file:
    for setup, acc in results.items():
        file.write(f"Accuracy for {setup}: {acc:.10f}\n")
