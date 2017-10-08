import numpy as np
import cv2

#Enter path of the image. Eg:- C:/Python34/hello.jpg
imagePath = input("Enter Path of the Image: ")
img = cv2.imread(imagePath,1)
#Save original image in a new variable
originalImg = img
#Convert img to dtpye16 for saturation purpose
img16 = img.astype(np.int16)
print("Size Of Original Image",img.shape)
#Take input for scalar value to be operated
scalarValue = int(input('Enter a scalar value: '))
print("Press following keys on image to perform the operation:  + - * / \nPress 'r' to reset the image \n")
cv2.imshow('Original Image',img)
operation=chr(cv2.waitKey(0))
cv2.destroyAllWindows()

while(operation):
    if operation == '+':
        #Clipping the pixels of image in range [0:255] to achieve saturation
        np.clip(img16 + scalarValue, 0, 255, out=img16)
        #Converting image back to dtype uint8 for purpose of visiblity
        img = img16.astype(np.uint8)
    elif operation == '-':
        np.clip(img16 - scalarValue, 0, 255, out=img16)
        img = img16.astype(np.uint8)
    elif operation == '*':
        np.clip(img16 * scalarValue, 0, 255, out=img16)
        img = img16.astype(np.uint8)
    elif operation == '/':
        np.clip(img16 / scalarValue, 0, 255, out=img16)
        img = img16.astype(np.uint8)
    elif operation == 'r':
        img = originalImg        
    else:
        cv2.destroyAllWindows()
        break  
    
    #Resisizig image to half
    newDimension = (int(img.shape[1]/2), int(img.shape[0]/2))
    resizedImage = cv2.resize(img, newDimension, interpolation = cv2.INTER_AREA)
    
    print("You selected {} operation to be perforned".format(operation))
    print("Size of Modified Image",resizedImage.shape)
    print("Press following keys on image to perform the operation:  + - * / \nPress 'r' to reset the image \n")
    cv2.imshow('Modified Image',resizedImage)    
    operation=chr(cv2.waitKey(0))
