# import dependency files
import cv2
import numpy as np
import pytesseract

# 이미지 업로드
img = cv2.imread('images/qweqwe.png')

# 마우스를 이용하여 이미지에서 원하는 부분을 ROI로 지정한다
x, y, w, h = cv2.selectROI('img', img, False)

if w and h:
    roi = img[y:y + h, x:x + w]

# tesseract를 이용해서 ocr을 진행하도록 한다
ocr = pytesseract.image_to_string(roi, lang='kor+eng')
print(ocr)

cv2.waitKey(0)
cv2.destroyAllWindows()