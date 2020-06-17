import xlrd

workbook = xlrd.open_workbook(r"ISIC-2017_Training_Part3_GroundTruth.xlsx")
sheet = workbook.sheet_by_index(0)
col_a = sheet.col_values(0,1)
col_b = sheet.col_values(1,1)

true_data = {a:b for a, b in zip(col_a, col_b)}


workbook_data = xlrd.open_workbook(r"RN50_A_data_zonder_duplicate.xlsx")
sheet_data = workbook_data.sheet_by_index(0)
c_a = sheet_data.col_values(0,1)
c_b = sheet_data.col_values(1,1)
data1 = {c:d for c, d in zip(c_a, c_b)}

workbook_data = xlrd.open_workbook(r"RN50_B_data_zonder_duplicate.xlsx")
sheet_data = workbook_data.sheet_by_index(0)
c_a = sheet_data.col_values(0,1)
c_b = sheet_data.col_values(1,1)
data2 = {c:d for c, d in zip(c_a, c_b)}


true_data_set = set(true_data)
data_set1 = set(data1)
data_set2 = set(data2)

i=0
overeenkomende_fouten = []
for name in data_set1.intersection(data_set2):
    if true_data[name] != data1[name] and true_data[name] != data2[name]:
        overeenkomende_fouten.append(name)

print(overeenkomende_fouten)
print(len(overeenkomende_fouten))