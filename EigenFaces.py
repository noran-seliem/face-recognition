from ast import List
import numpy as np
import cv2
from glob import glob
from natsort import natsorted
from time import time 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import re
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
arrayOfImages=[]
arrayOfimagesNumer=[]


class FacesFeatures:
    def __init__(self):
        self.r=1
        self.DatasetPath = './ATT_Faces_Dataset'
        self.SubjectsNumber = 40    # Fixed for the given Dataset
        self.ClassesNumber = 10     # Fixed for the given Dataset
        self.Images = List
        self.ImagesNumber = self.SubjectsNumber * self.ClassesNumber
        self.ImageResolution = (56, 46)    # Fixed for the given Dataset
        self.DataMatrix = np.ndarray
        self.MeanVector = np.ndarray
        self.A = np.ndarray
        self.Covariance = np.ndarray
        self.RedusedEigenFaces = np.ndarray


    def IntializeDataMatrix(self):

        # Intialize the 3D Data Matrix of Dimenssions (Number of Images , (Image Resolution)) with zeros
        self.DataMatrix = np.zeros((self.ImagesNumber, self.ImageResolution[0], self.ImageResolution[1]), dtype='float64')

        # Accessing and Reading the images in the dataset
        ImageIndex = 0
        self.Images = natsorted(glob(self.DatasetPath + '/**/*.pgm', recursive = True))
        
        for Image in self.Images:

            # To make sure that the image is in grayscale
            Img = cv2.imread(Image, cv2.IMREAD_GRAYSCALE)
            Img = cv2.resize(Img, (self.ImageResolution[1], self.ImageResolution[0]))

            # Appending the dataset images in the Data Matrix
            self.DataMatrix[ImageIndex] = np.array(Img)
            ImageIndex += 1


    def GetEigenFaces(self):

        # Reshaping the Data Matrix into 2D Matrix of Dimenssions (Number of Images , Image Hight x Image width))
        # Here each ROW represents an Image, so we Flipped the matrix so each COLUMN represents an Image
        self.DataMatrix = np.resize(self.DataMatrix, (self.ImagesNumber, self.ImageResolution[0] * self.ImageResolution[1]))
        self.DataMatrix = np.transpose(self.DataMatrix)

        # Calculating the Mean vector and reshaping it as a column so we can calculate A Matrix
        self.MeanVector = np.sum(self.DataMatrix, axis = 1, dtype="float64") / self.ImagesNumber
        self.MeanVector = np.resize(self.MeanVector, (self.ImageResolution[0] * self.ImageResolution[1], 1))
        self.A = self.DataMatrix - self.MeanVector

        # CalculatingEigenvalues, Eigenvectors 
        self.Covariance = np.dot(self.A, np.transpose(self.A)) / self.ImagesNumber
        EigenValues, EigenVectors = np.linalg.eig(self.Covariance)
       
        # get the indices of the eigenvalues by its value. Descending order.
        DescendingOrder = EigenValues.argsort()[::-1]
        # sorted eigenvalues and eigenvectors in descending order
        EigenVectors = EigenVectors[:, DescendingOrder]
        
        # Calculating EigenFaces  which is the Normalized Eigenvectors
        EigenFaces = EigenVectors / np.linalg.norm(EigenVectors)
        self.RedusedEigenFaces = np.empty(( int(0.90 * len(EigenFaces)), len(EigenFaces)), dtype = 'complex_')

        for Size in range (0, (len(EigenFaces))):
            for Index in range (0, (int(0.90 * len(EigenFaces)))):
                self.RedusedEigenFaces[Index][Size] = EigenFaces[Index][Size]


    def RecognizeFace(self,SourcePath):
        self.SourcePath=SourcePath
        TestImage = cv2.imread(SourcePath, cv2.IMREAD_GRAYSCALE)
        # Resizing and reshaping the testing image so we can calculate Phi
        TestImage = cv2.resize(TestImage, (self.ImageResolution[1], self.ImageResolution[0]))
        TestImage = np.reshape(TestImage, (TestImage.shape[0] * TestImage.shape[1], 1))
        Phi = TestImage - self.MeanVector
        OmegaTest = np.dot(self.RedusedEigenFaces, Phi)

        SubjectPath = 0
        SmallestDistance = 999999999
        Threshold = 3000

        for Image in self.Images:
            Img = cv2.imread(Image, cv2.IMREAD_GRAYSCALE)
            Img = cv2.resize(Img, (self.ImageResolution[1], self.ImageResolution[0]))
            Img = np.reshape(Img, (Img.shape[0] * Img.shape[1], 1))
            Phi = Img - self.MeanVector
            Omega_K = np.dot(self.RedusedEigenFaces, Phi)
            Difference = OmegaTest - Omega_K
            EuclideanDistance = np.sum(np.square((Difference)))

            if np.real(EuclideanDistance) < SmallestDistance:
                SmallestDistance = EuclideanDistance
                SubjectPath = Image

        if SmallestDistance < Threshold:
            print(SubjectPath)
            arrayOfImages.append(1)
            self.r=self.r+1
        else:
            print("Unknown Face!")
            arrayOfImages.append(0)
            self.r=self.r+1


# CallerM=FacesFeatures()
# CallerM.IntializeDataMatrix()
# CallerM.GetEigenFaces()

# # Using '*' pattern 
# for name in glob('./Test Images/*'):
#     print(name)
#     CallerM.RecognizeFace(name)
#     arrayOfimagesNumer.append(re.findall(r'\d+',name)[0])
          



# print(arrayOfimagesNumer)
# print(arrayOfImages)



# y = (np.array(arrayOfImages))
# pred = (np.array(arrayOfImages))
# RocCurveDisplay.from_predictions(y, pred)
# plt.show()
# cm = confusion_matrix(y, pred)

# cm_display = ConfusionMatrixDisplay(cm).plot()

# plt.show()
# if __name__ == "__main__":

#     Start = time()
#     FaceRecognizer = FacesFeatures()
#     FaceRecognizer.IntializeDataMatrix()
#     FaceRecognizer.GetEigenFaces()
#     FaceRecognizer.RecognizeFace('./Test Images/Subject1.pgm')
#     End = time()
#     print('Execution time:', (End - Start) , 'Seconds')