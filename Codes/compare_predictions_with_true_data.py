import xlrd
import numpy as np

workbook = xlrd.open_workbook(r"ISIC-2017_Training_Part3_GroundTruth.xlsx")
sheet = workbook.sheet_by_index(0)
col_a = sheet.col_values(0,1)
col_b = sheet.col_values(1,1)

true_data = {a:b for a, b in zip(col_a, col_b)}

network = r"V19_A"
workbook_data = xlrd.open_workbook(network+r"_data_zonder_duplicate.xlsx")
sheet_data = workbook_data.sheet_by_index(0)
c_a = sheet_data.col_values(0,1)
c_b = sheet_data.col_values(1,1)
data = {c:d for c, d in zip(c_a, c_b)}


true_data_set = set(true_data)
data_set = set(data)

fouten = []
for name in true_data_set.intersection(data_set):
    if true_data[name] != data[name]:
        fouten.append(name)

print(fouten)
print(len(fouten))

np.savetxt(network+r"_foutenlijst.txt", fouten, delimiter=" ", fmt="%s")