import pandas as pd

df_major = pd.read_excel('/Users/liaolongquan/Desktop/替换专业-郑.xlsx',header=0)
df_major = df_major[['序号','3、专业']].sort_values(by='序号').reset_index(drop=True)
df_data = pd.read_csv('/Users/liaolongquan/Desktop/研究数据-丹丹.csv',header=0,encoding='utf-8')

df_new = pd.merge(df_data,df_major,how='left', on=['序号'])
df_new[['专业']] = df_new[['3、专业']]
df_new = df_new.drop(columns=['3、专业'])
print(df_new)

df_new.to_csv('/Users/liaolongquan/Desktop/研究数据20220119.csv',index=False,encoding='utf-8-sig')

