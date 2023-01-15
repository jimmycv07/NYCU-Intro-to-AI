import os
import cv2

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    '''
    Open the directory of the testing data, and read the grayscale version
    of face and non-face images in with a for loop.
    '''
    dataset=[]
    path= dataPath+"/face"
    fptr= os.listdir(path)
    for i in fptr:
      img= cv2.imread(path+"/"+i, cv2.IMREAD_GRAYSCALE)
      dataset.append((img,1))
    path=dataPath+"/non-face"
    fptr= os.listdir(path)
    for i in fptr:
      img=cv2.imread(path+"/"+i, cv2.IMREAD_GRAYSCALE)
      dataset.append((img,0))
    # raise NotImplementedError("To be implemented")
    # End your code (Part 1)
    return dataset

# loadImages("G:/My Drive/NCTU/Senior-2/Artificial intelligence/hw1/AI_HW1/data/test")
