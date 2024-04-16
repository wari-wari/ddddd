from PIL import ImageGrab
import cv2
import keyboard
import mouse
import numpy as np
import pytesseract

# 실시간 화면 녹화

def set_roi(): #관심영역 지정
    global ROI_SET, x1, y1, x2, y2
    ROI_SET = False
    print("Select your ROI using mouse drag.")
    while(mouse.is_pressed() == False):
        x1, y1 = mouse.get_position()
        while(mouse.is_pressed() == True):
            x2, y2 = mouse.get_position()
            while(mouse.is_pressed() == False):
                print("Your ROI : {0}, {1}, {2}, {3}".format(x1, y1, x2, y2))
                ROI_SET = True
                return
keyboard.add_hotkey("ctrl+1", lambda: set_roi())
ROI_SET = False
x1, y1, x2, y2 = 0, 0, 0, 0

temp_ocr = ''

file = open('subtitle.txt', 'a', encoding='UTF-8') # 파일 열기

while True:  #q 버튼이 눌릴때 까지 반복 실행
    if ROI_SET == True:
        image = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2))), cv2.COLOR_BGR2RGB) # 관심영역부분
        cv2.imshow("image", image) #관심영역 선택한 부분 출력하여 보여주기
        ocr = pytesseract.image_to_string(image, lang='eng') # 문자인식
        if not ocr == temp_ocr:  # 다르면 출력
            print(ocr)
            file.write(ocr) #인식한 문장 파일에 쓰고

            # text파일로 ocr 문장 저장하는것도 넣어야함
            temp_ocr = ocr  # 기존 인식해놓은 ocr을 temp에 저장
        key = cv2.waitKey(700)
        if key == ord("q"):
            print("Quit")
            file.close()  # 파일 저장
            break
cv2.destroyAllWindows()

#1. 인식한 문자가 전에 인식한 문자와 같으면 출력하지 않기

# while문 -> 1.관심영역에서 이미지 따오기 2.이미지 출력하기 3. 이미지 ocr로 인식하기 3-1. 기존ocr로 인식해논 이미지와 ocr이 같으면 출력하지 않기 4. 이미지 출력하기 5. 관심영역 이미지 따오기 ... 반복
# 3-1번 코딩하기

#파일 저장하기

#file = open('subtitle.txt', 'w')
#file.write(ocr)
#file.close()














#2. 인식한 문자 print를 메모장에 저장 시키기