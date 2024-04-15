import cv2
import numpy as np
import pyautogui
import time
import OCR4
import pytesseract

screen_size = (1920, 1080)
half_screen_size = (screen_size[0] // 2, screen_size[1])

while True:
    img = pyautogui.screenshot(region=(0, 0, half_screen_size[0], half_screen_size[1]))
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #text = OCR4.read_caption(img)
    #frame = OCR4.see_result(img, text)
    #text = pytesseract.image_to_string(frame)
    cv2.imshow('Live Capture', frame)
    text = pytesseract.image_to_string(frame, lang="eng")
    if cv2.waitKey(1) == ord('q'):
       break

    print(text)

cv2.destroyAllWindows()