import pandas as pd

data_list = pd.read_excel('/Users/liaolongquan/Desktop/未命名.xlsx',sheet_name=None)
df_1 = data_list['1']
df_2 = data_list['2']
print(df_1)
print(df_2)

print(pd.merge(df_1, df_2, how='outer',on=['序号']))