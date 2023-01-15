import os
import cv2
import matplotlib.pyplot as plt

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    '''
    Read the information of detection data in, modify the image to the form that can be
    classified, and draw a rectangle on the original image depending on the classified
    result.   
    '''
    path="./"+dataPath
    fptr=open(path, 'r')
    while 1:
      temp=fptr.readline()
      if not temp:
        break
      temp=temp.split()
      img=cv2.imread("./data/detect/"+temp[0])
      imgGray=cv2.imread("./data/detect/"+temp[0], cv2.IMREAD_GRAYSCALE)
      for i in range(int(temp[1])):
        (x,y,w,h)=[int(x) for x in fptr.readline().split()]
        face=imgGray[y:y+h, x:x+w]
        face=cv2.resize(face, (19,19))
        res=clf.classify(face)
        if res:
          cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 1)
        else:
          cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 1)
      cv2.imshow("Result", img)
      cv2.waitKey(0)
    fptr.close()
    # raise NotImplementedError("To be implemented")
    # End your code (Part 4)
