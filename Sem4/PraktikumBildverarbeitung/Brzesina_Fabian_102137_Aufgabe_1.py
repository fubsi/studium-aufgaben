import cv2
import datetime

# Get Cam Object
cam = cv2.VideoCapture(0)

camImg = cam.read()[1]
camShape = camImg.shape[0:2]
brightness = 1
BRIGHTNESS_STEP = 0.1
contrast = 0
CONTRAST_STEP = 5

while True:
    # Read the image using 'read' function
    camImg = cam.read()[1]
    camImg = cv2.resize(camImg, (int(camShape[1]), int(camShape[0])))
    camImg = cv2.convertScaleAbs(camImg, alpha=brightness, beta=contrast)

    # Put text on the image
    cv2.putText(camImg, 'Webcam Image', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(camImg, str(datetime.datetime.now()), (0, camShape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the image on window
    cv2.imshow('Webcam Image', camImg)
    pressedKey = cv2.waitKey(1)
    if pressedKey != -1:
        print(f"Key pressed: {pressedKey}")
    if pressedKey == 113:
        # ESC pressed
        print("Escape hit, closing...")
        break
    if pressedKey == 43:
        # '+' pressed
        print("Resize Up")
        camShape = (int(camShape[0]*1.1), int(camShape[1]*1.1))
        print(f"New Shape: {camShape}")
    if pressedKey == 45:
        # '-' pressed
        print("Resize Down")
        camShape = (int(camShape[0]*0.9), int(camShape[1]*0.9))
        print(f"New Shape: {camShape}")
    if pressedKey == 97:
        # 'a' pressed
        print("Increase Contrast")
        contrast += CONTRAST_STEP
        print(f"New Beta: {contrast}")
    if pressedKey == 121:
        # 'y' pressed
        print("Decrease Contrast")
        contrast -= CONTRAST_STEP
        print(f"New Beta: {contrast}")
    if pressedKey == 115:
        # 's' pressed
        print("Increase Brightness")
        brightness += BRIGHTNESS_STEP
        print(f"New Alpha: {brightness}")
    if pressedKey == 120:
        # 'x' pressed
        print("Decrease Brightness")
        brightness -= BRIGHTNESS_STEP
        print(f"New Alpha: {brightness}")


cv2.destroyAllWindows()