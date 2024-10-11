import openpyxl
import pandas as pd
import re


xlsx_file = 'vocabulary.xlsx'
workbook = openpyxl.load_workbook("vocabulary.xlsx")
sheet = workbook.active

second_data = []
for row in sheet.iter_rows(min_col=2, max_col=2, min_row=1, value_only = True):
    second_data.append(row[0])

print(second_data)