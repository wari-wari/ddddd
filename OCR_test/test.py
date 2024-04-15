import cv2
import numpy as np
import pyautogui
import time
from matplotlib import pyplot as plt
import pytesseract

def preprocessing(image):
    kernel = np.ones((4, 20,), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5, 5))
    gray = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)

    th_img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
    morph = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, kernel, iterations=3)

    #cv2.imshow("th_img", th_img)
    #cv2.imshow("morph", morph)
    #cv2.waitKey()
    return image, morph

def check_size(size):
    w, h = size
    if h == 0 or w == 0: return False

    aspect = h / w if h > w else w / h
    chk1 = 3000 < (h*w) < 40000
    chk2 = 6.0 < aspect < 8.0
    return (chk1 and chk2)

def find_candidates(images):
    results = cv2.findContours(images, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = results[0] if int(cv2.__version__[0]) >= 4 else results[1]

    rects = [cv2.minAreaRect(c) for c in contours]
    candidates = [(tuple(map(int, center)), tuple(map(int, size)), angle)
                  for center, size, angle in rects if check_size(size)]

    return candidates

def color_candidate_img(image, candi_center):
    h,w = image.shape[:2]
    fill = np.zeros((h + 2, w + 2), np.uint8)
    dif1, dif2 = (25, 25, 25), (25, 25, 25)
    flags = 0xff00 + 4 + cv2.FLOODFILL_FIXED_RANGE
    flags += cv2.FLOODFILL_MASK_ONLY

    pts = np.random.randint(-15, 15, (20, 2) )
    pts = pts + candi_center
    for x,y in pts:
        if 0 <= x < w and 0 <= y < h:
            _, _, fill, _ = cv2.floodFill(image, fill, (x, y), 255, dif1, dif2, flags)

        return cv2.threshold(fill, 120, 255, cv2.THRESH_BINARY)[1]

def rotate_plate(image, rect):
    center, (h, w), angle = rect
    if w < h:
        w, h = h, w
    angle += 90

    size = image.shape[1::-1]
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rot_img = cv2.warpAffine(image, rot_mat, size, cv2.INNER_CUBIC)

    crop_img = cv2.getRectSubPix(rot_img, (w, h), center)
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(crop_img, (w, h))



#fills = [color_candidate_img(image, size) for size, _, _ in candidates]
#new_candis = [find_candidates(fill) for fill in fills]
#new_candis = [cand[0] for cand in new_candis if cand]
#candidate_img = [rotate_plate(image, cand) for cand in new_candis]

#for i, img in enumerate(candidate_img):
#    pts = np.int32(cv2.boxPoints(new_candis[i]))
#    cv2.polylines(image, [pts], True, (255, 255, 255), 2)



def read_caption(img):
    image, morph = preprocessing(img)
    candidates = find_candidates(morph)

    candidates = find_candidates(morph)
    for candidate in candidates:
        pts = np.int32(cv2.boxPoints(candidate))
        caption = image[pts[0][1] - 5:pts[2][1] + 5, pts[0][0] - 5:pts[2][0] + 5]

    gray = cv2.cvtColor(caption, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 75, 200)

    text = pytesseract.image_to_string(caption, lang='eng')
    return text

def see_result(img,text):
    extra = 8
    image, morph = preprocessing(img)
    candidates = find_candidates(morph)

    for candidate in candidates:
        pts = np.int32(cv2.boxPoints(candidate))
        cv2.rectangle(image, (pts[0][0] - extra, pts[0][1] - extra), (pts[2][0] + extra, pts[2][1] + extra), (255,255,255), cv2.FILLED)

    cv2.putText(image, text, (pts[0][0] + 1, pts[0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return image

screen_size = (1920, 1080)
half_screen_size = (screen_size[0] // 2, screen_size[1])

while True:
    img = pyautogui.screenshot(region=(0, 0, half_screen_size[0], half_screen_size[1]))
    text = read_caption(img)

    frame = np.array(img)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = see_result(img, text)
    cv2.imshow('Live Capture', frame)
    #text = pytesseract.image_to_string(frame, lang="eng")
    if cv2.waitKey(1) == ord('q'):
       break

    print(text)

