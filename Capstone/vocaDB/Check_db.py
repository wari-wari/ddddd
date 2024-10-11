import sqlite3

# SQLite 데이터베이스 연결
conn = sqlite3.connect('../../dict.db')
cursor = conn.cursor()

# 테이블 구조 확인
cursor.execute('PRAGMA table_info(dict)')
table_info = cursor.fetchall()
print("Table Structure:")
for column in table_info:
    print(column)

# 데이터 확인 (첫 10개 행)
cursor.execute('SELECT * FROM dict LIMIT 10')
rows = cursor.fetchall()
print("\nFirst 10 rows of data:")
for row in rows:
    print(row)