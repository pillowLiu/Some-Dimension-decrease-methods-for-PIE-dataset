import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

def LoadAndPreprocessImages(directory, size=(32, 32)):
    Pics, Labels = [], []
    for Name in os.listdir(directory):
        Path = os.path.join(directory, Name)
        if os.path.isdir(Path):
            for File in os.listdir(Path):
                with Image.open(os.path.join(Path, File)) as Img:
                    Resized = Img.convert('L').resize(size)
                    Pics.append(np.array(Resized).flatten())
                    Labels.append(int(Name))
    Pics = np.array(Pics)
    Mean = np.mean(Pics, axis=0)
    Std = np.std(Pics, axis=0)
    StandardizedPics = (Pics - Mean) / Std
    return StandardizedPics, np.array(Labels)

def AccuracyScore(YTrue, YPred):
    return np.sum(YTrue == YPred) / len(YTrue)

def SplitData(X, Y, TestSize, RandomState=42):
    np.random.seed(RandomState)
    ShuffledIndices = np.random.permutation(len(X))
    TestSetSize = int(len(X) * TestSize)
    TestIndices = ShuffledIndices[:TestSetSize]
    TrainIndices = ShuffledIndices[TestSetSize:]
    return X[TrainIndices], X[TestIndices], Y[TrainIndices], Y[TestIndices]

def AccuracyScoreSpecific(YTrue, YPred, Label):
    Correct = np.sum((YTrue == YPred) & (YTrue == Label))
    Total = np.sum(YTrue == Label)
    return Correct / Total if Total > 0 else 0

def VisualizeLda(XLda, Y, NComponents, Title, SavePath, LabelOwnPhoto=26):
    plt.figure()
    if NComponents == 3:
        Ax = plt.axes(projection='3d')
        Ax.scatter(XLda[Y != LabelOwnPhoto, 0], XLda[Y != LabelOwnPhoto, 1], XLda[Y != LabelOwnPhoto, 2], c='blue', alpha=0.5)
        Ax.scatter(XLda[Y == LabelOwnPhoto, 0], XLda[Y == LabelOwnPhoto, 1], XLda[Y == LabelOwnPhoto, 2], c='red', label='My Photo')
    else:
        plt.scatter(XLda[Y != LabelOwnPhoto, 0], XLda[Y != LabelOwnPhoto, 1], c='blue', alpha=0.5)
        plt.scatter(XLda[Y == LabelOwnPhoto, 0], XLda[Y == LabelOwnPhoto, 1], c='red', label='My Photo')
    plt.legend()
    plt.title(Title)
    plt.savefig(os.path.join(SavePath, f'{Title}.png'))
    plt.show()

def LdaFit(X, Y):
    ClassMeans = []
    for I in np.unique(Y):
        ClassMeans.append((I, np.mean(X[Y == I], axis=0)))
    OverallMean = np.mean(X, axis=0)
    BetweenClassMatrix = np.zeros((X.shape[1], X.shape[1]))
    for I, MeanVec in ClassMeans:
        N = X[Y == I].shape[0]
        MeanVec = MeanVec.reshape(-1, 1)
        OverallMean = OverallMean.reshape(-1, 1)
        BetweenClassMatrix += N * (MeanVec - OverallMean).dot((MeanVec - OverallMean).T)
    WithinClassMatrix = np.zeros((X.shape[1], X.shape[1]))
    for I in np.unique(Y):
        ClassScatter = np.cov(X[Y == I], rowvar=False)
        WithinClassMatrix += ClassScatter
    EigVals, EigVecs = np.linalg.eigh(np.linalg.inv(WithinClassMatrix).dot(BetweenClassMatrix))
    SortedIndices = np.argsort(EigVals)[::-1]
    return EigVecs[:, SortedIndices], EigVals[SortedIndices]

def LdaTransform(X, Eigenvectors,NumComponents):
    if NumComponents == -1:
        return np.dot(X, Eigenvectors)
    else:
        return np.dot(X, Eigenvectors[:, :NumComponents])

def NnPredict(X, XTrain, YTrain):
    Predictions = []
    for I in range(X.shape[0]):
        Distances = np.sum((XTrain - X[I]) ** 2, axis=1)
        NearestNeighbor = np.argmin(Distances)
        Predictions.append(YTrain[NearestNeighbor])
    return Predictions

if __name__ == "__main__":
    PicsFolder = "PIE"
    ResultsFolder = "result/LDA"
    LabelOwnPhoto = 26

    Pics, Tags = LoadAndPreprocessImages(PicsFolder)
    XTrain, XTest, YTrain, YTest = SplitData(Pics, Tags, TestSize=0.3)

    Dimensions = [2, 3, 9]
    AccFile = os.path.join(ResultsFolder, 'lda_accuracy.txt')
    with open(AccFile, 'w') as File:
        for Dim in Dimensions:
            Eigenvectors, Eigenvalues = LdaFit(XTrain, YTrain)
            XTrainLda = LdaTransform(XTrain, Eigenvectors, Dim)
            XTestLda = LdaTransform(XTest, Eigenvectors, Dim)

            VisualizeLda(XTrainLda, YTrain, Dim, f'LDA {Dim}D Visualization', ResultsFolder)

            YPred = NnPredict(XTestLda, XTrainLda, YTrain)

            AccuracyAll = AccuracyScore(YTest, YPred)
            File.write(f'LDA {Dim}D - All: {AccuracyAll:.2f}\n')

            AccuracyOwn = AccuracyScoreSpecific(YTest, YPred, LabelOwnPhoto)
            File.write(f'LDA {Dim}D - My Photo: {AccuracyOwn:.2f}\n')

