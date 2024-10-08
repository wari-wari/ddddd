# pip install transformers
# pip install datasets
# pip install sentencepiece
# pip install evaluate
# pip install sacrebleu
# pip install pytorch
# pip install torchtext
# pip install einops

# pip install pillow
# pip install tkinter
# pip instell sqlite3
import torch
from transformers import MarianTokenizer
from torch import nn
from einops import rearrange
import time
import sqlite3
import sys
from datetime import datetime

import threading

import tkinter
from tkinter import *
from tkinter import scrolledtext, messagebox
from tkinter import filedialog
import tkinter.font

import re
import cv2
import numpy as np
import pyautogui
import pytesseract
from PIL import ImageTk
from PIL import Image, ImageFont, ImageDraw

import module.db_create as db_create
import module.db_conn as db_conn
import module.vocaDB.make_db as make_db


db_create.db_create()


class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.n_heads = n_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.scale = torch.sqrt(torch.tensor(d_model / n_heads))

    def forward(self, Q, K, V, mask = None):

        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        Q = rearrange(Q, 'b n (h d) -> b h n d', h = self.n_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h = self.n_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h = self.n_heads)

        attention_score = Q @ K.transpose(-2,-1)/self.scale

        if mask is not None:
            attention_score[mask] = -1e10
        attention_weights = torch.softmax(attention_score, dim=-1)

        attention = attention_weights @ V

        x = rearrange(attention, 'b h n d -> b n (h d)')
        x = self.fc_o(x) 

        return x, attention_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_p):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(d_model, d_ff),
                                    nn.ReLU(),
                                    nn.Dropout(drop_p),
                                    nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = self.linear(x)
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.self_atten = MHA(d_model, n_heads)
        self.self_atten_LN = nn.LayerNorm(d_model)

        self.FF = FeedForward(d_model, d_ff, drop_p)
        self.FF_LN = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, enc_mask):

        residual, atten_enc = self.self_atten(x, x, x, enc_mask)
        residual = self.dropout(residual)
        x = self.self_atten_LN(x + residual)

        residual = self.FF(x)
        residual = self.dropout(residual)
        x = self.FF_LN(x + residual)

        return x, atten_enc

class Encoder(nn.Module):
    def __init__(self, input_embedding, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.scale = torch.sqrt(torch.tensor(d_model))
        self.input_embedding = input_embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(drop_p)

        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_heads, drop_p) for _ in range(n_layers)])

    def forward(self, src, mask, atten_map_save = False):

        pos = torch.arange(src.shape[1]).repeat(src.shape[0], 1).to(DEVICE)

        x = self.scale*self.input_embedding(src) + self.pos_embedding(pos)
        x = self.dropout(x)

        atten_encs = torch.tensor([]).to(DEVICE)
        for layer in self.layers:
            x, atten_enc = layer(x, mask)
            if atten_map_save is True:
                atten_encs = torch.cat([atten_encs , atten_enc[0].unsqueeze(0)], dim=0)

        return x, atten_encs


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.self_atten = MHA(d_model, n_heads)
        self.self_atten_LN = nn.LayerNorm(d_model)

        self.enc_dec_atten = MHA(d_model, n_heads)
        self.enc_dec_atten_LN = nn.LayerNorm(d_model)

        self.FF = FeedForward(d_model, d_ff, drop_p)
        self.FF_LN = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, enc_out, dec_mask, enc_dec_mask):

        residual, atten_dec = self.self_atten(x, x, x, dec_mask)
        residual = self.dropout(residual)
        x = self.self_atten_LN(x + residual)

        residual, atten_enc_dec = self.enc_dec_atten(x, enc_out, enc_out, enc_dec_mask)
        residual = self.dropout(residual)
        x = self.enc_dec_atten_LN(x + residual)

        residual = self.FF(x)
        residual = self.dropout(residual)
        x = self.FF_LN(x + residual)

        return x, atten_dec, atten_enc_dec

class Decoder(nn.Module):
    def __init__(self, input_embedding, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.scale = torch.sqrt(torch.tensor(d_model))
        self.input_embedding = input_embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(drop_p)

        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, n_heads, drop_p) for _ in range(n_layers)])

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, trg, enc_out, dec_mask, enc_dec_mask, atten_map_save = False):

        pos = torch.arange(trg.shape[1]).repeat(trg.shape[0], 1).to(DEVICE)

        x = self.scale*self.input_embedding(trg) + self.pos_embedding(pos)
        x = self.dropout(x)

        atten_decs = torch.tensor([]).to(DEVICE)
        atten_enc_decs = torch.tensor([]).to(DEVICE)
        for layer in self.layers:
            x, atten_dec, atten_enc_dec = layer(x, enc_out, dec_mask, enc_dec_mask)
            if atten_map_save is True:
                atten_decs = torch.cat([atten_decs , atten_dec[0].unsqueeze(0)], dim=0)
                atten_enc_decs = torch.cat([atten_enc_decs , atten_enc_dec[0].unsqueeze(0)], dim=0)

        x = self.fc_out(x)

        return x, atten_decs, atten_enc_decs



class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = Encoder(self.input_embedding, max_len, n_layers, d_model, d_ff, n_heads, drop_p)
        self.decoder = Decoder(self.input_embedding, max_len, n_layers, d_model, d_ff, n_heads, drop_p)

        self.n_heads = n_heads

        for m in self.modules():
            if hasattr(m,'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)

    def make_enc_mask(self, src):

        enc_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2) 
        enc_mask = enc_mask.repeat(1, self.n_heads, src.shape[1], 1) 
        return enc_mask

    def make_dec_mask(self, trg):

        trg_pad_mask = (trg.to('cpu') == pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = trg_pad_mask.repeat(1, self.n_heads, trg.shape[1], 1) 

        trg_future_mask = torch.tril(torch.ones(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[1]))==0

        dec_mask = trg_pad_mask | trg_future_mask

        return dec_mask

    def make_enc_dec_mask(self, src, trg):

        enc_dec_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)
        enc_dec_mask = enc_dec_mask.repeat(1, self.n_heads, trg.shape[1], 1)

        return enc_dec_mask

    def forward(self, src, trg):

        enc_mask = self.make_enc_mask(src)
        dec_mask = self.make_dec_mask(trg)
        enc_dec_mask = self.make_enc_dec_mask(src, trg)

        enc_out, atten_encs = self.encoder(src, enc_mask)
        out, atten_decs, atten_enc_decs = self.decoder(trg, enc_out, dec_mask, enc_dec_mask)

        return out, atten_encs, atten_decs, atten_enc_decs

class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, LR_scale = 1):
        self.optimizer = optimizer
        self.current_step = 0
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.LR_scale = LR_scale

    def step(self):
        self.current_step += 1
        lrate = self.LR_scale * (self.d_model ** -0.5) * min(self.current_step ** -0.5, self.current_step * self.warmup_steps ** -1.5)
        self.optimizer.param_groups[0]['lr'] = lrate


def translation(model, src_text, atten_map_save = False):
    model.eval()
    with torch.no_grad():
        src = tokenizer.encode(src_text, return_tensors='pt').to(DEVICE) # 1x단
        enc_mask = model.make_enc_mask(src)
        enc_out, atten_encs = model.encoder(src, enc_mask, atten_map_save)

        pred = tokenizer.encode('</s>', return_tensors='pt', add_special_tokens=False).to(DEVICE) 
        for _ in range(max_len-1): 
            dec_mask = model.make_dec_mask(pred)
            enc_dec_mask = model.make_enc_dec_mask(src, pred)
            out, atten_decs, atten_enc_decs = model.decoder(pred, enc_out, dec_mask, enc_dec_mask, atten_map_save)

            pred_word = out.argmax(dim=2)[:,-1].unsqueeze(0)

            pred = torch.cat([pred, pred_word], dim=1)

            if tokenizer.decode(pred_word.item()) == '</s>':
                break

        #print(pred)
        translated_text = tokenizer.decode(pred[0])[5:-4]
    return translated_text, atten_encs, atten_decs, atten_enc_decs


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)


tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ko-en', cache_dir="C:\pythonpractice\.cache")

eos_idx = tokenizer.eos_token_id
pad_idx = tokenizer.pad_token_id

max_len = 100

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
save_model_path = './model/TF_en.pt'
save_history_path = './model/TF_en_his.pt'
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
vocab_size = tokenizer.vocab_size



loaded = torch.load(save_model_path, map_location=DEVICE)
load_model = loaded["model"]
ep = loaded["ep"]
optimizer = loaded["optimizer"]

loaded = torch.load(save_history_path, map_location=DEVICE)
loss_history = loaded["loss_history"]
print("모델에폭:",ep)


now_lan = 'eng'
isen = False
def change_lan():
    global isen
    global save_model_path
    global save_history_path
    global now_lan

    # 모델 변경
    global load_model
    global optimizer
    global loss_history
    global ep
    if isen:
        # 모델 경로 변경
        save_model_path = './model/TF_en.pt'
        save_history_path = './model/TF_en_his.pt'      # 이후 한글 번역모델로  수정
        trans_lan["image"]=enimage

        now_lan = 'eng'
        # 모델 변경
        loaded = torch.load(save_model_path, map_location=DEVICE)
        load_model = loaded["model"]
        optimizer = loaded["optimizer"]
        ep = loaded["ep"]

        loaded = torch.load(save_history_path, map_location=DEVICE)
        loss_history = loaded["loss_history"]
        isen = False
        print(ep)
    else:
        # 모델 경로 변경
        save_model_path = './model/TF_kr.pt'
        save_history_path = './model/TF_kr_his.pt'      # 이후 수정
        trans_lan["image"]=krimage

        now_lan = 'kor'
        # 모델 변경
        loaded = torch.load(save_model_path, map_location=DEVICE)
        load_model = loaded["model"]
        optimizer = loaded["optimizer"]
        ep = loaded["ep"]

        loaded = torch.load(save_history_path, map_location=DEVICE)
        loss_history = loaded["loss_history"]
        isen = True
        print(ep)


# tt=0
# while(1):

#     src_text = str(datass.loc[tt]['ko'])
#     if(src_text =="0"):
#         break
    
#     print(f"입력: {src_text}")

#     translated_text = translation(load_model, src_text)[0]
#     print(f"AI의 번역: {translated_text}")
#     tt+=1

# trans- 데이터베이스, trans_datas- 테이블, num- int, times - varchar(12), data-varchar(500)
global conn
global cur
#db 연결
conn, cur = db_conn.db_conn()

word = ""
part_of_speech = ""
meaning = ""

tt=0
clkt = True
def print_second():
    global tt
    global draw_text
    print(tt)
    if(stop_trans):
        src_text = ocr_screen()
        print(f"입력: {src_text}")


        if (src_text==""):
            threading.Timer(1,print_second).start()
            return

        draw_text = ImageDraw.Draw(pil_image)

        draw_text.rectangle((0,0, width, height), outline=(255, 255, 255, 255), fill=(255, 255, 255, 255), width=3)
        
        src_text = src_text.replace("|", 'i')
        translated_text = translation(load_model, src_text)[0]
        
        translated_text= translated_text.replace("▁", ' ')
        translated_text= translated_text.replace("\\", '')
        translated_text= translated_text.replace("<unk>", '?')
        translated_text= translated_text.replace("  ", ' ')
        now = datetime.now()                     # 현재 시간
        time_set = now -start_time    # 경과 시간 계산(소수점버림)
        try:
            sql = "INSERT INTO trans_datas VALUES(?,?,?)"
            cur.execute(sql, (tt, "0"+str(time_set)[:-3], translated_text))
        except sqlite3.Error as e:
            print(f"Error connecting to sqlite3 Platform: {e}")
            sys.exit(1)
        print(f"AI의 번역: {translated_text}")
        print(f'{tt} : {"0"+str(time_set)[:-3]}초 경과')

        # 자막 출력하기
        draw_text.text((10, 30), translated_text, (30, 30, 30), font=font)
        imgtk = ImageTk.PhotoImage(image=pil_image)
        label_text.config(image=imgtk)
        label_text.image = imgtk

        voca, part, mean = voca_note(src_text)
        print(voca, part, mean)
        # gui에 자막 적기
        txtbox.insert(1.0, "{} - {}, 번역: {}".format(tt+1, "0"+str(time_set)[:-3], translated_text)+"\n\n")

        # gui에 단어 출력하기
        try: #단어가 있을 경우 출력
            voca, part, mean = voca_note(src_text)
            vocabox.insert(1.0, f"단어 : {voca:<15} 품사 : {part:<15} 뜻 : {mean}\n")
        except: #없을 경우 미출력
            voca = "None"
            part = "None"
            mean = "None"
            print(voca, part, mean)

        threading.Timer(1, print_second).start()             # x초 마다 반복
    else:
        conn.commit()
    tt = tt + 1
    pass

# k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# min_x = 1000000000
# max_y = 0

# def combine(image, image2):
#     contours, _ = cv2.findContours(image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     combine_image = np.zeros_like(image)
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         combine_image[y:y + h, x:x + w] = image[y:y + h, x:x + w]
#     return combine_image



def ocr_screen():
    global now_lan
    global min_x
    global max_y
    board.fill(255)
    roi = [300, 700, 770, 140]
    area = pyautogui.screenshot(region=(roi[0], roi[1], roi[2], roi[3]))
    frame = np.array(area)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th_img = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    
    # cv2.imshow('canny', th_img)

    # canny = cv2.Canny(th_img, 130, 200)
    # canny = cv2.morphologyEx(canny, cv2.MORPH_ERODE, k2)
    # canny = cv2.morphologyEx(canny, cv2.MORPH_DILATE, k2)
    # # canny = cv2.morphologyEx(canny, cv2.MORPH_DILATE, k2)
    # canny = cv2.morphologyEx(canny, cv2.MORPH_DILATE, k2)
    # canny = cv2.morphologyEx(canny, cv2.MORPH_DILATE, k2)

    # contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # combine_image = np.zeros_like(frame)
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     min_x = min(min_x, x)        # 윤곽선이 잡힌 가장 왼쪽 x값
    #     max_y = max(max_y, (y + h))  # 윤곽선이 잡힌 가장 아래쪽 y값
    #     combine_image[y:y + h, x:x + w] = frame[y:y + h, x:x + w]
    # combine_image = combine(frame, canny)

    # gray2 = cv2.cvtColor(combine_image, cv2.COLOR_BGR2GRAY)

    # _, final_image = cv2.threshold(gray2, 250, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(th_img, lang=now_lan)
    lines = text.replace('\n','')
    
    #img_t = Image.fromarray(board)
    #imgtk = ImageTk.PhotoImage(image=img_t)
    #label_text = tkinter.Label(text_show, image=imgtk)
    #label_text.pack()
    return lines


font = ImageFont.truetype('font/NanumSquareR.ttf',30)

def voca_DBcheck():
    try:
        conn = sqlite3.connect("dict.db", check_same_thread=False)
        cursor = conn.cursor()
        print("영어사전 DB 연결")
        return conn, cursor
    except sqlite3.Error as e:
        print("영어사전 DB 연결 실패")
        db_create.db_create()
        conn = sqlite3.connect("dict.db")
        cursor = conn.cursor()
        print("영어사전 DB 연결")
        return conn, cursor

#단어장 연결
voca_conn, voca_cursor = voca_DBcheck()

def voca_note(sentence):
    word =""
    part_of_speech = ""
    meaning = ""
    temp = sentence
    words = temp.split()
    
    #conn, cursor = voca_DBcheck()

    for word in words:
        try:
            voca_cursor.execute('SELECT part_of_speech, meaning FROM dict WHERE word = ?', (word,))
            result = voca_cursor.fetchone()
        except:
            continue

        if result:
            part_of_speech = result[0]
            meaning = result[1]
            print("단어 : " + word + ", " + "품사 : " + part_of_speech + ", " + "뜻 : " + meaning + " ")
            return word, part_of_speech, meaning
        else:
            continue


#word, part_of_speech, meaning = voca_note()

def print_lyric(roi):
    # 자막 출력
    global board
    global pil_image
    global text_show
    global label_text
    board = np.zeros((100 , int(width/2.3),3), np.uint8)

    board.fill(255)

    text_show=tkinter.Toplevel()

#    text_show.geometry(str(1100)+"x"+str(100)+"+"
#                       +str(roi[0])+"+"+str(roi[3]+roi[1]+5)) # roi[0], roi[3]+roi[1]+5
#                                            # w h
#                                            # x y
    text_show.geometry(str(int(width/2.3))+"x"+str(100)+"+"
                       +str(roi[0])+"+"+str(roi[3]+roi[1]+5)) # roi[0], roi[3]+roi[1]+5
                                            # w h
                                            # x y
    text_show.overrideredirect(True)
    text_show.wm_attributes("-topmost", 1)
    text_show.configure(background='#FFFFFF')


    pil_image = Image.fromarray(board)
    # PIL 이미지를 사용하여 PhotoImage 객체 생성
    imgtk = ImageTk.PhotoImage(image=pil_image)
    label_text = tkinter.Label(image=imgtk,master=text_show)
    label_text.pack(side="top")


# ocr
width, height = pyautogui.size()
screen_size = (width, height)
def text_select():
    global roi

    
    img = pyautogui.screenshot(region=(0, 0, screen_size[0], screen_size[1]))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    roi = cv2.selectROI('Select ROI', img, False)
    cv2.destroyAllWindows()
    screen_select_win.destroy()


# 스크린 설정 버튼을 누르면 자막 영역 선택 방식 고르기
def select_screen():
    global screen_select_win
    screen_select_win = tkinter.Toplevel(window)
    
    screen_select_win.iconphoto(False, icon32)
    screen_select_win.geometry(str(int(width*0.6))   +"x"+  str(int(height*0.2))    +"+"
               +str(int((width/2)-(width*0.3)))    +"+"+   str(int((height)-(height*0.27))))
                # W H
                # x y
    # 맨위
    set_frame1=tkinter.Frame(screen_select_win, bg="White", highlightthickness=1, highlightbackground="#AFAFAF", bd=1, width=int(width/2), height=int(height*0.075), background="#404040")
    set_frame1.pack(side='top', fill='both', padx=(5),pady=5)

    labelExam = tkinter.Label(set_frame1, text="자막 영역 방식을 선택해주세요.", background="#404040", fg='#FFFFFF')
    labelExam.config(font=('나눔 스퀘어', 20, 'bold'))
    labelExam.pack(fill='both', padx=(5),pady=5)

    #두번쨰
    set_framecen=tkinter.Frame(screen_select_win, bg="White", highlightthickness=1, highlightbackground="#AFAFAF", bd=1, width=int(width/2), height=int(height*0.075), background="#404040")
    set_framecen.pack(side='top', fill='both', padx=(5),pady=5)

    labelExam = tkinter.Label(set_framecen, text="전체화면을 체크하시면 재생버튼을 바로 눌러 실행하면 됩니다.\n\n"
                              +"직접설정을 체크한다면 바로 자막 영역 설정 화면이 나옵니다.\n"
                               +"마우스 휠을 이용해 자막 영역을 네모로 설정해주세요" , background="#404040", fg='#FFFFFF')
    labelExam.config(font=('나눔 스퀘어', 12, 'bold'))
    labelExam.pack(fill='both', padx=(5),pady=5)

    #3번쨰
    set_frame2=tkinter.Frame(screen_select_win, bg="White", highlightthickness=1, highlightbackground="#AFAFAF", bd=1, width=int(width/2), height=int(height*0.075), background="#404040")
    set_frame2.pack(side='bottom',fill='both', padx=(5),pady=5)
    
    global full_image_bu
    full_image_bu = Image.open("images/전체체크.png")
    full_image_bu = full_image_bu.resize((int(width*0.07), int(height*0.03)), Image.LANCZOS)
    full_image_bu = ImageTk.PhotoImage(full_image_bu)
    
    global full_image_bu_box
    full_image_bu_box = Image.open("images/전체체크박스.png")
    full_image_bu_box = full_image_bu_box.resize((int(width*0.07), int(height*0.03)), Image.LANCZOS)
    full_image_bu_box = ImageTk.PhotoImage(full_image_bu_box)
    global full_screen_sett
    full_screen_sett = tkinter.Button(set_frame2, overrelief="solid", width=int(width*0.07),height=int(height*0.03), text="한영", image=full_image_bu,
                            bd=0, command=change_screen_mod, repeatdelay=1000, repeatinterval=100
                            , highlightthickness=0, justify='center')

    full_screen_sett.pack(side='left', fill='both',padx=5, pady=10)
    
    global part_image_bu
    part_image_bu = Image.open("images/직접체크.png")
    part_image_bu = part_image_bu.resize((int(width*0.07), int(height*0.03)), Image.LANCZOS)
    part_image_bu = ImageTk.PhotoImage(part_image_bu)
    
    global part_image_bu_box
    part_image_bu_box = Image.open("images/직접체크박스.png")
    part_image_bu_box = part_image_bu_box.resize((int(width*0.07), int(height*0.03)), Image.LANCZOS)
    part_image_bu_box = ImageTk.PhotoImage(part_image_bu_box)

    global part_screen_sett
    part_screen_sett = tkinter.Button(set_frame2, overrelief="solid", width=int(width*0.07),height=int(height*0.03), text="한영", image=part_image_bu_box,
                            bd=0, command=change_screen_mod, repeatdelay=1000, repeatinterval=100
                            , highlightthickness=0 , justify='center')

    part_screen_sett.pack(side='right', fill='both',padx=5, pady=10)

    global screen_mod
    if screen_mod:
        full_screen_sett["image"]=full_image_bu
        part_screen_sett["image"]=part_image_bu_box
    else: 
        full_screen_sett["image"]=full_image_bu_box
        part_screen_sett["image"]=part_image_bu
        
    screen_select_win.configure(background='#404040')

#roi=[1070, 1086, 1068, 99]

x, y = 5, 30
lines = ""

# Main GUI
window=tkinter.Tk()
window.title("Mister Trans")
center_wid=(width)-(width*0.9)
center_hei = height-int(height*0.27)

window.geometry(str(int(width) - int(width/3.2))   +"x"+   str(int(height*0.2))   +"+"
               +str(int(center_wid))    +"+"+   str(int(center_hei)))
                # W H
                # x y
window.resizable(True, True)
window.configure(background='#404040')

icon32 = tkinter.PhotoImage(file='./images/mticon.png')

window.iconphoto(False, icon32)
count=0
file_lo="."


t_b_state=True
# b_trans
def start_trans():
    global stop_trans
    global t_b_state
    global start_time 
    global tt
    global roi
    if t_b_state:   # 재생 버튼 상태 일 때
        if screen_mod:      # 전체 화면
            roi = [int(width*0.13),int(height*0.4),int(width*0.73),int(height*0.33)]
            print_lyric(roi)
        else:               # 자막화면
            print_lyric(roi)
        txtbox.configure(state='normal')
        txtbox.delete("1.0", "end")
        vocabox.configure(state='normal')
        vocabox.delete("1.0", "end")
        b_save["state"] = DISABLED
        b_text_sel["state"] = DISABLED
        setting_bu["state"] = DISABLED
        trans_lan["state"] = DISABLED
        file_local["state"] = DISABLED


        b_trans["image"]=stopimage

        t_b_state=False
        stop_trans=True
        start_time = datetime.now()
        sql = "DELETE FROM trans_datas"
        cur.execute(sql)
        tt=0
        print_second()

    else:           # 정지 버튼 상태 일 때
        txtbox.configure(state='disabled')
        vocabox.configure(state='disabled')
        b_save["state"] = NORMAL
        b_text_sel["state"] = NORMAL
        setting_bu["state"] = NORMAL
        trans_lan["state"] = NORMAL
        file_local["state"] = NORMAL
        b_trans["image"]=playimage

        t_b_state=True
        stop_trans=False
        text_show.destroy()


# b_trans_stop
# def trans_stop():

#     txtbox.configure(state='disabled')
#     b_trans["state"] = NORMAL
#     b_save["state"] = NORMAL
#     b_text_sel["state"] = NORMAL

#     global stop_trans
#     stop_trans=False
#     global clkt
#     clkt = True
#     text_show.destroy()

def result_save():
    sql = "SELECT * from trans_datas"
    cur.execute(sql)
    global texts
    resultset = cur.fetchall()
    # label.config(text=texts)
    fir=""
    fir_b=True
    if(input_text_file_name.get()):
        file_name_text = input_text_file_name.get()
    else:
        file_name_text = "MT_lyrics"
    with open(file_lo+"/"+file_name_text+".srt",'w',encoding='UTF-8') as f:
        for i in resultset:
            if fir_b==True:
                fir = i
                fir_b=False
                continue
            f.write("{}\n{} --> {}\n{}".format(fir[0]+1, fir[1][:8]+","+fir[1][9:], i[1][:8]+","+i[1][9:], fir[2])+"\n\n")
            fir = i
        f.write("{}\n{} --> {}\n{}".format(fir[0]+1, fir[1][:8]+","+fir[1][9:], fir[1][:7]+str(int(fir[1][7])+3)+","+fir[1][9:], fir[2])+"\n\n")
    newWindow.destroy()
# 현재 자막 영역 출력 해줌
def ly_roi_show():
    global roi
    setting_window.destroy()
    time.sleep(0.3)
    if screen_mod:      # 전체 화면
        roi = [int(width*0.13),int(height*0.4),int(width*0.73),int(height*0.33)]
    area = pyautogui.screenshot(region=(roi[0], roi[1], roi[2], roi[3]))
    frame = np.array(area)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('Selected lyrics Window', frame)
    cv2.waitKey(0)


# 전체스크린, 스크린 변경
screen_mod = True
def change_screen_mod():
    global screen_mod # True 가 full/
    if screen_mod:
        full_screen_sett["image"]=full_image_bu_box
        part_screen_sett["image"]=part_image_bu
        screen_mod= False
        text_select()
    else: 
        full_screen_sett["image"]=full_image_bu
        part_screen_sett["image"]=part_image_bu_box
        screen_mod= True
        screen_select_win.destroy()
        text_show.destroy()


def file_location():
    global file_lo
    file_lo=window.dirName=filedialog.askdirectory()

# 저장을 위한 새로운 윈도우창
def createNewWindow():
    global newWindow
    newWindow = tkinter.Toplevel(window)
    
    newWindow.geometry("700x200+"+str(int(center_wid))+"+"+str(int(center_hei)))
    labelExample = tkinter.Label(newWindow, text="자막파일 이름을 입력해주세요")
    global input_text_file_name
    input_text_file_name = Entry(newWindow, width=50, font=('나눔 스퀘어', 20, 'bold'))
    result_save_B = tkinter.Button(newWindow, command=result_save, text="저장")
    newWindow.configure(background='#404040')
    labelExample.pack(fill='x')
    input_text_file_name.pack(fill='x')
    result_save_B.pack(fill='x')
    newWindow.iconphoto(False, icon32)
# 세팅화면 보여주는 윈도우
def setting_new_window():
    global setting_window
    setting_window = tkinter.Toplevel(window)
    
    setting_window.geometry(str(int(width*0.586))   +"x"+  str(int(height*0.655))    +"+"
               +str(int((width/2)-(width*0.25)))    +"+"+   str(int((height/2)-(height*0.3))))
                # W H
                # x y
    
    setting_window.iconphoto(False, icon32)
    set_frame1=tkinter.Frame(setting_window, bg="White", highlightthickness=1, highlightbackground="#AFAFAF", bd=1, width=int(width/2),height=int(int(height*0.7)*0.01), background="#404040")
    set_frame1.pack(side='top', fill='both')


    labelExample = tkinter.Label(set_frame1, text="사용 방법", background="#404040", fg='#FFFFFF')
    # labelExample.config(justify = CENTER)
    labelExample.config(font=('나눔 스퀘어', 20, 'bold'))

    main_frame1=tkinter.Frame(setting_window, bg="White", highlightthickness=1, highlightbackground="#AFAFAF", bd=1, width=int(width/2),height=int(int(height*0.7)*0.01), background="#404040")
    main_frame1.pack(side='left', fill='both')
    main_frame2=tkinter.Frame(setting_window, bg="White", highlightthickness=1, highlightbackground="#AFAFAF", bd=1, width=int(width/2),height=int(int(height*0.7)*0.01), background="#404040")
    main_frame2.pack(side='right', fill='both')
    main_frame3=tkinter.Frame(setting_window, bg="White", highlightthickness=1, highlightbackground="#AFAFAF", bd=1, width=int(width/2),height=int(int(height*0.7)*0.01), background="#404040")
    main_frame3.pack(side='left', fill='both')

    global settimg
    settimg = Image.open("images/세팅이미지.png")
    settimg = settimg.resize((int(width*0.44), int(height*0.48)), Image.LANCZOS)
    settimg = ImageTk.PhotoImage(settimg)

    
    labelExample.pack(fill='x', padx=5, pady=10)

    set_frame1=tkinter.Frame(main_frame1, highlightthickness=0, highlightbackground="#AFAFAF", bd=0, width=int(width/2),height=int(height*0.7), background="#404040")
    set_frame1.pack(side='top',expand=True, fill='both')

    set_frame2=tkinter.Frame(main_frame2, highlightthickness=0, highlightbackground="#AFAFAF", bd=0, width=int(width/2),height=int(height*0.7), background="#404040")
    set_frame2.pack(expand=True, fill='both')

    
 
    



    labelExample.pack(fill='x', padx=5, pady=10)

    labelExample2 = tkinter.Label(set_frame1, image= settimg)
    labelExample2.pack(fill='both')

    global settimg_side
    settimg_side = Image.open("images/설정화면설명.png")
    settimg_side = settimg_side.resize((int(width*0.14), int(height*0.6)), Image.LANCZOS)
    settimg_side = ImageTk.PhotoImage(settimg_side)


    labelExample3 = tkinter.Label(set_frame2, image= settimg_side)
    labelExample3.pack(fill='both')

    setting_window.configure(background='#202020')
    # result_save_B = tkinter.Button(setting_window, command=result_save, text="저장")

    global ly_roi_img
    ly_roi_img = Image.open("images/자막영역확인.png")
    ly_roi_img = ly_roi_img.resize((int(width*0.079), int(height*0.028)), Image.LANCZOS)
    ly_roi_img = ImageTk.PhotoImage(ly_roi_img)

    ly_roi_check = tkinter.Button(set_frame1, overrelief="solid", width=int(width*0.079),height=int(height*0.028), text="한영", image=ly_roi_img,
                        bd=1, command=ly_roi_show, repeatdelay=1000, repeatinterval=100
                        , highlightthickness=1, highlightbackground="#AFAFAF", )
    ly_roi_check.pack(side="left",pady=10,padx=10)


    # result_save_B.pack(fill='x')
        
def info_win():
    global info_window
    info_window = tkinter.Toplevel(window)
    
    info_window.geometry(str(int(width*0.58))   +"x"+  str(int(height*0.41))    +"+"
               +str(int((width/2)-(width*0.25)))    +"+"+   str(int((height/2)-(height*0.35))))
                # W H
                # x y
    
    info_window.iconphoto(False, icon32)

    global info_main
    info_main = Image.open("images/info.png")
    info_main = info_main.resize((int(width*0.58), int(height*0.41)), Image.LANCZOS)
    info_main = ImageTk.PhotoImage(info_main)

    labelExample3 = tkinter.Label(info_window, image= info_main)
    labelExample3.pack(fill='both')

##-----
def create(db_name): #db 생성
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS subtitles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time TEXT,
            text TEXT
        )
    ''')
    conn.commit()
    return conn

def insert(conn, start_time, end_time, text): #db에 자막시작시간, 끝나는시간, 자막 insert
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO subtitles (start_time, end_time, text)
            VALUES (?, ?, ?)
        ''', (start_time, end_time, text))
        conn.commit()
    except sqlite3.Error as e:
        messagebox.showerror("Database Error", f"자막 삽입 오류 발생: {e}")

def fetch(conn): #db에서 자막 가져오기
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT start_time, end_time, text FROM subtitles ORDER BY id')
        return cursor.fetchall()
    except sqlite3.Error as e:
        messagebox.showerror("Database Error", f"자막 가져오기 오류 발생: {e}")
        return []

def convert_time(time_str): #srt파일의 시간을 시분초로 변경
    time_parts = re.split('[:,]', time_str)
    h = int(time_parts[0])
    m = int(time_parts[1])
    s = float(time_parts[2]) + float(time_parts[3]) / 1000
    return h * 3600 + m * 60 + s

def select_file(): #파일 선택창
    global file_path
    file_path = filedialog.askopenfilename(title="자막파일 선택", filetypes=[("SRT Files", "*.srt")])
    if file_path:
        load_subtitles(file_path)

def load_subtitles(file_path):
    global draw_text, label_text, pil_image, thread1

    # 자막 출력 창 생성
    board = np.zeros((100, int(width / 2.3), 3), np.uint8)
    board.fill(255)

    text_show = tkinter.Toplevel()
    roi = [int(width * 0.13), int(height * 0.4), int(width * 0.73), int(height * 0.33)]
    text_show.geometry(f"{int(width / 2.3)}x100+{roi[0]}+{roi[3] + roi[1] + 5}")
    text_show.overrideredirect(True)
    text_show.wm_attributes("-topmost", 1)
    text_show.configure(background='#FFFFFF')

    pil_image = Image.fromarray(board)
    imgtk = ImageTk.PhotoImage(image=pil_image)
    label_text = tkinter.Label(text_show, image=imgtk)
    label_text.pack(side="top")

    draw_text = ImageDraw.Draw(pil_image)
    draw_text.rectangle((0, 0, width, height), outline=(255, 255, 255, 255), fill=(255, 255, 255, 255), width=3)

    db_name = 'subtitles.db'
    conn = create(db_name)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        messagebox.showerror("File Error", f"실행 오류 발생: {e}")
        return

    blocks = re.split(r'\n\n+', content.strip())

    def subtitle_print():  # 쓰레드 분리
        local_conn = create(db_name)
        try:
            for block in blocks:
                lines = block.splitlines()
                if len(lines) >= 3:
                    time_range = lines[1]
                    text = ' '.join(lines[2:]).replace('\n', ' ')
                    start_time, end_time = time_range.split(' --> ')
                    insert(local_conn, start_time, end_time, text)

            start_time = time.time()  # 현재 시간

            for subtitle in fetch(local_conn):
                start_time = convert_time(subtitle[0])
                end_time = convert_time(subtitle[1])

                wait_time = start_time - (time.time() - start_time)

                if wait_time > 0:
                    time.sleep(wait_time)

                print(f"[{subtitle[0]}] {subtitle[2]}")  # 삭제해도 상관없는것 (vscode에 자막 시작시간과 자막 출력)

                translated_text = f"{subtitle[2]}"
                draw_text.rectangle((0, 0, width, height), outline=(255, 255, 255, 255), fill=(255, 255, 255, 255),
                                    width=3)
                draw_text.text((10, 30), translated_text, (30, 30, 30), font=font)
                imgtk = ImageTk.PhotoImage(image=pil_image)
                label_text.config(image=imgtk)
                label_text.image = imgtk

                # label_text.update_idletasks()
                label_text.update()

                time.sleep(end_time - start_time)

        except Exception as e:
            messagebox.showerror("Error", f"자막 출력 오류 발생: {e}")
        finally:
            local_conn.close()

    thread1 = threading.Thread(target=subtitle_print)  # subtitle_print 쓰레드 분리 load_subtitle 안에서 실행
    thread1.daemon = True
    thread1.start()

def on_closing():
    window.quit()

frame1=tkinter.Frame(window, bg="White", highlightthickness=1,  highlightbackground="#AFAFAF", bd=1, width=int(width/2),height=int((height/2)*0.5), background="#404040")
frame1.pack(side="left", expand=True, padx=(5),pady=5, fill="x")

frame3=tkinter.Frame(window, bg="White", highlightthickness=1,  highlightbackground="#AFAFAF", bd=1, width=int((width/2)*0.25),height=121, background="#404040")
frame3.pack(side="right", expand=False, padx=(5),pady=5, fill="y")

frame2=tkinter.Frame(window, bg="White", highlightthickness=1,  highlightbackground="#AFAFAF", bd=1, width=int((width/2)*0.5),height=121, background="#404040")
frame2.pack(side="right", expand=False, padx=(5),pady=5, fill="y")

frame1_2=tkinter.Frame(frame1, bg="White", highlightthickness=0, highlightbackground="#AFAFAF", bd=0, height=81, width=int(width/2), background="#FFFFFF")
frame1_2.pack(side="bottom", expand=True,padx=(3), pady=3,fill='x')
frame1_1 = tkinter.Frame(frame1, bg="White", highlightthickness=0, highlightbackground="#AFAFAF", bd=0, height=40, width=int(width/2), background="#FFFFFF")
frame1_1.pack(side="top", expand=True, padx=(3), pady=3,fill='x')

frame2_1=tkinter.Frame(frame2, bg="White", highlightthickness=0, bd=0, height=121,width=int((width/2)*0.5), background="#404040")
frame2_1.pack(side='top', fill="y", expand=True)
frame2_2=tkinter.Frame(frame2, bg="White", highlightthickness=0, bd=0, height=121,width=int((width/2)*0.5), background="#404040")
frame2_2.pack(side="bottom", fill="y", expand=True)

# main_settimg = Image.open("images/메인설명.png")
# main_settimg = main_settimg.resize((int(width*0.12), int(height*0.18)), Image.LANCZOS)
# main_settimg = ImageTk.PhotoImage(main_settimg)

# labelExampleMain = tkinter.Label(frame3, image=main_settimg
#                              , background="#404040")
# labelExampleMain.pack(fill='both',expand=False)




playimage = Image.open("images/재생.png")
stopimage = Image.open("images/정지.png")
saveimage = Image.open("images/저장.png")
docuimage = Image.open("images/문서.png")
profilemage = Image.open("images/화면잡기.png")
enimage = Image.open("images/en.png")
krimage = Image.open("images/kr.png")
settingimage = Image.open("images/환경설정.png")
select_image = Image.open("images/srt_file.png")#--

# 이미지 크기 조절
widthim, heightim = int(width/25), int(width/25)  # 원하는 크기로 조절

playimage = playimage.resize((widthim, heightim), Image.LANCZOS)
stopimage = stopimage.resize((widthim, heightim), Image.LANCZOS)
saveimage = saveimage.resize((widthim, heightim), Image.LANCZOS)
docuimage = docuimage.resize((widthim, heightim), Image.LANCZOS)
profilemage = profilemage.resize((widthim, heightim), Image.LANCZOS)
enimage = enimage.resize((widthim, heightim), Image.LANCZOS)
krimage = krimage.resize((widthim, heightim), Image.LANCZOS)
settingimage = settingimage.resize((widthim, heightim), Image.LANCZOS)
select_image = select_image.resize((widthim, heightim), Image.LANCZOS)#--
# Tkinter의 PhotoImage로 변환
playimage = ImageTk.PhotoImage(playimage)
stopimage = ImageTk.PhotoImage(stopimage)
saveimage = ImageTk.PhotoImage(saveimage)
docuimage = ImageTk.PhotoImage(docuimage)
profilemage = ImageTk.PhotoImage(profilemage)
enimage = ImageTk.PhotoImage(enimage)
krimage = ImageTk.PhotoImage(krimage)
settingimage = ImageTk.PhotoImage(settingimage)
select_photo = ImageTk.PhotoImage(select_image) #--

b_trans = tkinter.Button(frame2_1, overrelief="solid", width=int(width/25),height=int(width/25), text="번역 실행", image=playimage,
                        bd=0, command=start_trans, repeatdelay=1000, repeatinterval=100
                        , highlightthickness=0 )


# b_trans_stop = tkinter.Button(frame2_1, overrelief="solid", width=100, text="번역 종료",state="disabled", image=stopimage,
#                          command=trans_stop, repeatdelay=1000, repeatinterval=100)

b_save = tkinter.Button(frame2_2, overrelief="solid", width=int(width/25),height=int(width/25), text="저장하기",state="disabled", image=saveimage,
                        bd=0, command=createNewWindow, repeatdelay=1000, repeatinterval=100
                        , highlightthickness=0 )


file_local = tkinter.Button(frame2_2, overrelief="solid", width=int(width/25), height=int(width/25), text="저장위치", image=docuimage,
                        bd=0, command=file_location, repeatdelay=1000, repeatinterval=100
                        , highlightthickness=0 )


b_text_sel = tkinter.Button(frame2_1, overrelief="solid", width=int(width/25),height=int(width/25), text="자막위치 선택", image=profilemage,
                        bd=0, command=select_screen, repeatdelay=1000, repeatinterval=100
                        , highlightthickness=0 )

trans_lan = tkinter.Button(frame2_2, overrelief="solid", width=int(width/25),height=int(width/25), text="한영", image=enimage,
                        bd=0, command=change_lan, repeatdelay=1000, repeatinterval=100
                        , highlightthickness=0 )

setting_bu = tkinter.Button(frame2_1, overrelief="solid", width=int(width/25),height=int(width/25), text="설정", image=settingimage,
                        bd=0, command=setting_new_window, repeatdelay=1000, repeatinterval=100
                        , highlightthickness=0 )

select_button = tkinter.Button(frame2_1, overrelief="solid", width=int(width/25),height=int(width/25), text="선택", image=select_photo,
                        bd=0, command=select_file, repeatdelay=1000, repeatinterval=100
                        , highlightthickness=0 )


b_trans.grid(row=0, column=0, padx=(10), pady=10)
# b_trans_stop.grid(row=0, column=1, padx=(10), pady=10)
b_save.grid(row=0, column=0, padx=(10), pady=10)
file_local.grid(row=0, column=1, padx=(10), pady=10)


b_text_sel.grid(row=0, column=1, padx=(10), pady=10)

trans_lan.grid(row=0, column=2, padx=(10), pady=10)

setting_bu.grid(row=0, column=2, padx=(10), pady=10)

select_button.grid(row=0, column=3, padx=(10), pady=10)

info_img = Image.open("images/정보.png")
help_img = Image.open("images/도움말.png")
setting_img =Image.open("images/Setting_img.jpg")
info_img = info_img.resize((int(width*0.056), int(height*0.031)), Image.LANCZOS)
help_img = help_img.resize((int(width*0.056), int(height*0.031)), Image.LANCZOS)
setting_img = setting_img.resize((int(width*0.056), int(height*0.031)), Image.LANCZOS)


info_img = ImageTk.PhotoImage(info_img)
help_img = ImageTk.PhotoImage(help_img)
setting_img = ImageTk.PhotoImage(setting_img)



help_butt = tkinter.Button(frame3, overrelief="solid", width=int(width*0.056),height=int(height*0.031), text="도움말", image = help_img,
                        command=setting_new_window, repeatdelay=1000, repeatinterval=100
                        , bg="White", highlightthickness=1,  highlightbackground="#AFAFAF", bd=1)

info_butt = tkinter.Button(frame3, overrelief="solid", width=int(width*0.056),height=int(height*0.031), text="정보", image = info_img,
                        command=info_win, repeatdelay=1000, repeatinterval=100
                        , bg="White", highlightthickness=1,  highlightbackground="#AFAFAF", bd=1)
setting = tkinter.Button(frame3, overrelief="solid", width=int(width*0.056),height=int(height*0.031), text="환경설정", image = setting_img,
                        command=info_win, repeatdelay=1000, repeatinterval=100
                        , bg="White", highlightthickness=1,  highlightbackground="#AFAFAF", bd=1)

help_butt.pack(padx=(0), pady=0)
info_butt.pack(padx=(0), pady=0)
setting.pack(padx=(0), pady=0)

vocabox = scrolledtext.ScrolledText(frame1_1, height=4)
vocabox.pack(expand=False, fill="both")
vocabox.tag_config("important", background="#ffffff", foreground="red")
vocabox.configure(state='disabled')

texts=""
txtbox = scrolledtext.ScrolledText(frame1_2, height=8)
txtbox.pack(expand=False, fill="both")
txtbox.tag_config("important", background="#ffffff", foreground="red")
txtbox.configure(state='disabled')

window.protocol("WM_DELETE_WINDOW", on_closing)
window.mainloop()
