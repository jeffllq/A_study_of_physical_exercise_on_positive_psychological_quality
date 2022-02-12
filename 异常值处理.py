import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr




df = pd.read_csv('data/old/研究数据（含心理问卷详情）.csv', header=0)

col_names = [ '年龄', '年级', '专业', '性别', '家庭所在地', '家庭经济情况', '学习成绩', '对待体育活动的态度',
       '认知', '人际', '情感', '公正', '节制', '超越', '积极心理品质得分', '剧烈运动天数', '剧烈运动单日分钟数',
       '每周剧烈MET', '适度运动天数', '适度运动单日分钟数', '每周适度MET', '步行天数', '步行单日分钟数',
       '每周步行MET', '总MET', '总天数', '运动强度\n1轻2中3强', '工作日久坐分钟数', 'IPAQ_8']


col_names_to_process = [ '年龄', '每周剧烈MET', '每周适度MET', '每周步行MET', '总MET', '总天数',  '工作日久坐分钟数']

rows = df.shape[0]

for col_name in col_names_to_process:
    #观察离群值
    # print(col_name,df[col_name].describe())
    mean = df[col_name].mean()
    sigma = df[col_name].std()
    for i in range(0, rows):
        if df.loc[i, col_name] > (mean+3*sigma):
            df.loc[i, col_name] = (mean+3*sigma)
        if df.loc[i, col_name] < (mean-3*sigma):
            df.loc[i, col_name] = (mean-3*sigma)

    # print(col_name,df[col_name].describe())

mean = df['总MET'].mean()
sigma = df['总MET'].std()
df['总MET'] = df['每周剧烈MET']+df[ '每周适度MET']+ df['每周步行MET']
# print(col_name,df[col_name].describe())

rows = df.shape[0]
for i in range(0, rows):
    if (((df.loc[i, '总MET'] >=1500)&(df.loc[i, '剧烈运动天数'] >=3))|((df.loc[i, '总MET'] >=3000)&(df.loc[i, '剧烈运动天数']+df.loc[i, '适度运动天数']+df.loc[i, '步行天数'] >=7))):
        df.loc[i, '运动强度\n1轻2中3强'] = 3
    elif (((df.loc[i, '总MET'] >= 600)& (df.loc[i, '剧烈运动天数']+df.loc[i, '适度运动天数']+df.loc[i, '步行天数'] >=5))|
        ((df.loc[i, '适度运动天数']+df.loc[i, '步行天数'] >=5)&(df.loc[i, '适度运动单日分钟数']+df.loc[i, '步行单日分钟数'] >=30))|
        ((df.loc[i, '剧烈运动天数'] >=3)&(df.loc[i, '剧烈运动单日分钟数'] >=20))):
        df.loc[i, '运动强度\n1轻2中3强'] = 2
    else:
        df.loc[i, '运动强度\n1轻2中3强'] = 1

# df.to_csv('data/tmp.csv',index=False,encoding='utf-8-sig')

# df = df[(df['运动强度\n1轻2中3强']==2)  ]
# df = df[(df['运动强度\n1轻2中3强']==2) & (df['家庭经济情况']>2) & (df['性别']==2)  ]
print(df['总MET'].corr(df['积极心理品质得分']))


x = df['积极心理品质得分'].mean()
print('总体得分均值',x)

x = df['运动强度\n1轻2中3强'].mean()
print('总体运动强度',x)

df_test = df[df['运动强度\n1轻2中3强']==3]
x = df_test['积极心理品质得分'].mean()
print('强运动强度得分均值',x)

df_test = df[df['运动强度\n1轻2中3强']==2]
x = df_test['积极心理品质得分'].mean()
print('中运动强度得分均值',x)

df_test = df[df['运动强度\n1轻2中3强']==1]
x = df_test['积极心理品质得分'].mean()
print('弱运动强度得分均值',x)



# df_tmp = df[[ '年龄', '性别', '家庭所在地', '家庭经济情况', '学习成绩', '对待体育活动的态度',
#        '认知', '人际', '情感', '公正', '节制', '超越', '积极心理品质得分', '剧烈运动天数', '剧烈运动单日分钟数',
#        '每周剧烈MET', '适度运动天数', '适度运动单日分钟数', '每周适度MET', '步行天数', '步行单日分钟数',
#        '每周步行MET', '总MET', '总天数', '运动强度\n1轻2中3强', '工作日久坐分钟数']]
#
# # print(df_tmp.corr())
# a = df_tmp.corr()
# print(1)
# x = list(df_test['运动强度\n1轻2中3强'].values)
# y = list(df_test['积极心理品质得分'].values)
#

# plt.scatter(x, y)
# plt.show()
