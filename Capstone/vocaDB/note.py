import sqlite3

def voca_note(sentence):
    part_of_speech = ""
    meaning = ""
    words = sentence.split()

    conn = sqlite3.connect('../../dict.db')
    cursor = conn.cursor()

    for word in words:
        cursor.execute('SELECT part_of_speech, meaning FROM dict WHERE word = ?', (word,))
        result = cursor.fetchone()

        if result:
            part_of_speech, meaning = result
            print("단어 : " + word + ", " + "품사 : " + part_of_speech + ", " + "뜻 : " + meaning + " ")
        else:
            continue

temp = "What is her real name? Age? Who are her parents? Where did she grow up? The official inquiry into Alice Guo, disgraced former mayor of a small town not far from the capital Manila, has been compulsive viewing for Filipinos since it began in May."
voca_note(temp)