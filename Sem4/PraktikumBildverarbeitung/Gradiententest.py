import cv2
import numpy as np

# Cam Object
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # Convert to Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Sobel Filter
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    # Magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    # Normalizatioqn
    magnitude *= 255.0 / magnitude.max()
    # Convert to uint8
    magnitude = np.uint8(magnitude)
    # Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Display
    cv2.imshow('Magnitude', magnitude)
    cv2.imshow('Laplacian', laplacian)
    cv2.imshow('Frame', frame)
    # Break
    key = cv2.waitKey(1)
    if key == ord('q'):
        break