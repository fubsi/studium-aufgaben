import cv2
import numpy as np

USED_FONT = cv2.FONT_HERSHEY_SIMPLEX

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
    # cv2.imshow('Brightnesses', blackImg)
    # cv2.waitKey(0)
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
        cv2.putText(img, c, (x, (textHeight+5)*line), USED_FONT, scale, (255, 255, 255), 1)
        x += charWidth
    return img

# ------------------------------
# --- Test Brightness Method ---
# ------------------------------
chars = [chr(c) for c in range(32, 127)]
brightnesses = {c: getBrightness(c) for c in chars}
sortedBrightnesses = sorted(brightnesses.items(), key=lambda x: x[1], reverse=True)
charsToString = ""
print("Brightnesses:")
for c, b in sortedBrightnesses:
    charsToString += c
    print(f"{c}",end="")

# ------------------------------
# --- Test Print-Line Method ---
# ------------------------------
blackImg = np.zeros((500, 1520, 3), np.uint8)
for i in range(1,6):
    blackImg = printLine(blackImg, charsToString, i)

charsToString = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt"

blackImg = printLine(blackImg, charsToString, 7)

cv2.imshow('Brightnesses', blackImg)
cv2.waitKey(0)

cv2.destroyAllWindows()