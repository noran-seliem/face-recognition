# Task 5 (Face Detection and Recognition)

## Team members
| Name                | Sec | BN  |
| ------------------- | --- | --- |
| Ezzeldeen Esmail    | 1   | 50  |
| Noran ElShahat      | 2   | 40  |
| Moamen Gamal        | 2   | 11  |
| Omar Sayed          | 2   | 2   |
| Abdelrahman Almahdy | 1   | 45  |



### 1- Face Detection



### 2- Face Recognition
#### 2A- Dataset
The dataset is created by AT&T Laboratories Cambridge and can be found [Here](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces)

This dataset contains 40 subject images of 10 images per subject.
The files are in PGM format. The size of each image is 92x112 pixels, with 256 grey levels per pixel.

#### 2B- Eigen Faces

Here we created a Class that holds the Data Matrix and intermediate matrices to obtain the EigenFaces Matrix
```python
class FacesFeatures:
    def __init__(self):
        self.DatasetPath = './ATT_Faces_Dataset'
        self.SubjectsNumber = 40
        self.ClassesNumber = 10 
        self.ImagesNumber = self.SubjectsNumber * self.ClassesNumber
        self.ImageResolution = (112, 92)
        self.DataMatrix = np.ndarray
        self.MeanVector = np.ndarray
        self.A = np.ndarray
        self.Covariance = np.ndarray
        self.EigenFaces = np.ndarray
```
Here we Access and Read the images in the dataset and Intialize the 3D Data Matrix
```python
def IntializeDataMatrix(self):

    self.DataMatrix = np.zeros((self.ImagesNumber, self.ImageResolution[0], self.ImageResolution[1]), dtype='float64')

    ImageIndex = 0
    Images = natsorted(glob.glob(self.DatasetPath + '/**/*.pgm', recursive = True))
    
    for Image in Images:

        Img = cv2.imread(Image, cv2.IMREAD_GRAYSCALE)
        
        self.DataMatrix[ImageIndex] = np.array(Img)
        ImageIndex += 1
```

Here we Reshape the Data Matrix into 2D Matrix till each COLUMN represents an Image
We Calculate the Mean vector and A Matrix
We then Calculate Eigenvalues, Eigenvectors and the Normalized Eigenvectors which is the EigenFaces
```python
def GetEigenFaces(self):

    self.DataMatrix = np.resize(self.DataMatrix, (self.ImagesNumber, self.ImageResolution[0] * self.ImageResolution[1]))
    self.DataMatrix = np.transpose(self.DataMatrix)

    self.MeanVector = np.sum(self.DataMatrix, axis = 1, dtype="float64") / self.ImagesNumber
    self.MeanVector = np.resize(self.MeanVector, (self.ImageResolution[0] * self.ImageResolution[1], 1))
    self.A = self.DataMatrix - self.MeanVector

    self.Covariance = np.dot(self.A, np.transpose(self.A)) / self.ImagesNumber
    Eigenvalues, Eigenvectors = np.linalg.eig(self.Covariance)
    self.EigenFaces = Eigenvectors / np.linalg.norm(Eigenvectors)
```