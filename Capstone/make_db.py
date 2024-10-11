import sqlite3
import pandas as pd
import re

excel_file = 'module/vocaDB/voca.xlsx'
df = pd.read_excel(excel_file, sheet_name=None)

def check_speech(speech):
    if speech == "n":
        speech = "명사"
        return speech
    elif speech == "v":
        speech = "동사"
        return speech
    elif speech == "adj":
        speech = "형용사"
        return speech
    elif speech == "adv":
        speech = "부사"
        return speech
    elif speech == "phr":
        speech = "구"
        return speech
    else:
        speech = "None"
        return speech

def check(pre_meaning):
    part_of_speech = ""
    meaning = ""
    other = pre_meaning.split("/ ")
    if len(other) == 1:
        temp = other[0].split(".")
        part_of_speech = check_speech(temp[0])
        meaning = temp[1]

    elif len(other) > 1:
        for i in range(len(other)):
            temp = other[i].split(".")
            if i == 0:
                part_of_speech = check_speech(temp[0])
                meaning = temp[1]
            elif i > 0:
                part_of_speech = part_of_speech + "," + check_speech(temp[0])
                meaning = meaning + "," + temp[1]

    return part_of_speech, meaning

def print_result(word, part_of_speech, meaning):
    #print(f"현재 시트: {sheet_name}")
    print(f"단어: {word}   품사: {part_of_speech}   뜻: {meaning}")

def make_db():
    conn = sqlite3.connect('../../dict.db')
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS dict (
            word TEXT PRIMARY KEY,
            part_of_speech TEXT,
            meaning TEXT
            )
        ''')



    for sheet_name, data in df.items():
        for index in range(min(100, len(data))):
            word = data.iloc[index, data.columns.get_loc('단어')]
            temp = data.iloc[index, data.columns.get_loc('뜻')]
            part_of_speech, meaning = check(temp)
            print(f"현재 시트: {sheet_name}")
            print_result(word, part_of_speech, meaning)

            cursor.execute('''
                    INSERT OR IGNORE INTO dict (word, part_of_speech, meaning)
                    VALUES (?, ?, ?)
                    ''', (word, part_of_speech, meaning))


    conn.commit()
    conn.close()