from sklearn.preprocessing import MinMaxScaler
from openpyxl import load_workbook

load_wb = load_workbook("./Train_Data.xlsx", data_only=True)
load_ws = load_wb['Train_Data']

get_cells = load_ws['B1':'B5919']
value_list = []

for row in get_cells:
    for cell in row:
        value_list.append([cell.value])

scale = MinMaxScaler(feature_range=(0.001, 0.999))
normal = scale.fit_transform(value_list)
print(normal)
