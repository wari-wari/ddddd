import pytesseract
import cv2
import matplotlib.pyplot as plt
import imutils
import re
import requests
import numpy as np

path = "images/OCR_test3.png"
image = cv2.imread(path)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
##blurred = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 75, 200)

##cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
##cnts = imutils.grab_contours(cnts)
##cnts = sorted(cnts, key=cv2.contourArea(), reverse=True)

##receiiptCnt = None

##for c in cnts:
##    peri = cv2.arcLength(c, True)
##    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

text = pytesseract.image_to_string(edged, lang='eng')

print(text)
cv2.imshow("result", edged)
cv2.waitKey()
