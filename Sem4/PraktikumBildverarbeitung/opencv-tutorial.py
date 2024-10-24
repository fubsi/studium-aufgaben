import cv2
import datetime

# Get Cam Object
cam = cv2.VideoCapture(0)

# Read the image using 'read' function
camImg = cam.read()[1]

# Display the image using 'imshow'
cv2.imshow('Webcam Image', camImg)
cv2.waitKey(0)

# Load an image using 'imread' specifying the path to image
img = cv2.imread('windows-10-dark-blue.png')

#Resize the image to 1080x720
imgS = cv2.resize(img, (1080, 720))

# Put text on the image
cv2.putText(imgS, 'Windows 10 Dark Blue', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(imgS, 'Date: ' + str(datetime.datetime.now()), (0, 715), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Increase brightness of the image
imgS = cv2.convertScaleAbs(imgS, alpha=1.5, beta=0)

# Display the image using 'imshow'
print('Image Dimensions: ', img.shape)
cv2.imshow('Windows 10 Dark Blue (Resized)', imgS)
cv2.waitKey(0)