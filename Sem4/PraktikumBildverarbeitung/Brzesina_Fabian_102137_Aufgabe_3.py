import cv2
import numpy as np

USED_FONT = cv2.FONT_HERSHEY_PLAIN

# ------------------------------
# --- Get Brightness Method ---
# ------------------------------
def getBrightness(c: str) -> int:
    # Returns the brightness value from the given character
    blackImg = np.zeros((100,100,3), np.uint8)
    scale = 10
    while True:
        textWidth, textHeight = cv2.getTextSize(c, USED_FONT, scale, 1)[0]
        if textWidth < blackImg.shape[1]-40 and textHeight < blackImg.shape[0]-40:
            break
        scale -= 0.01
    cv2.putText(blackImg, c, (10, 80), USED_FONT, scale, (255, 255, 255))

    brightnessSum = 0
    for row in blackImg:
        for col in row:
            brightnessSum += sum(col)
    return brightnessSum

# ------------------------------
# --- Print-Line Method ---
# ------------------------------
def printLine(img: np.ndarray, text: str, line: int):
    # Prints the given text on the given line of the image
    charWidth = img.shape[1] // len(text)
    printText = text
    x = 0
    for i, c in enumerate(printText):
        if i == 0:
            scale = 10
            while True:
                textWidth, textHeight = cv2.getTextSize("#", USED_FONT, scale, 1)[0]
                if textWidth <= charWidth:
                    break
                scale -= 0.01
        cv2.putText(img, c, (x, (20)*line), USED_FONT, scale, (255, 255, 255), 1)
        x += charWidth
    return img


chars = [chr(c) for c in range(32, 127)]
brightnesses = {c: getBrightness(c) for c in chars}
sortedBrightnesses = sorted(brightnesses.items(), key=lambda x: x[1], reverse=True)

print(sortedBrightnesses)

# Get Cam Object
cam = cv2.VideoCapture(0)

while True:
    blackImg = np.zeros((1000, 1520, 3), dtype=np.uint8)
    charWidth = blackImg.shape[1] // len(chars)
    camImg = cam.read()[1]
    camImg = cv2.resize(camImg, (1520//charWidth, 500//10))

    line = 1
    for row in camImg:
        text = ""
        for col in row:
            brightness = sum(col) * 212 #MAXVALUE of brightest ascii char / MAXVALUE of brightest pixel (255,255,255 == 765)
            for c, b in sortedBrightnesses:
                if brightness >= b:
                    text += c
                    break
        blackImg = printLine(blackImg, text, line)
        print(f"Line {line}: {text}")
        line+=1

    # Display the image on window
    cv2.imshow('Webcam Image', blackImg)
    pressedKey = cv2.waitKey(1)
    if pressedKey != -1:
        print(f"Key pressed: {pressedKey}")
    if pressedKey == 113:
        # ESC pressed
        print("Escape hit, closing...")
        break