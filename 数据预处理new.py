import pandas as pd
import matplotlib.pyplot as plt

# 计算积极心理品质得分，6个维度得分
list_cognition = [1, 2, 6, 7, 14, 23, 24, 30, 35, 43, 44, 50]  # 认知
list_human_relation = [3, 4, 5, 15, 25, 27, 29, 32, 46, 48]  # 人际
list_emotion = [10, 11, 12, 16, 17, 31, 33, 37, 45, 58, 59]  # 情感
list_justice = [9, 18, 19, 26, 38, 47, 51, 55, 57]  # 公正
list_temperate = [8, 21, 28, 34, 39, 42, 52, 54, 56, 62]  # 节制
list_transcendence = [13, 20, 22, 36, 40, 41, 49, 53, 60, 61]  # 超越

col_names = ['序号']  # 序号表示学生的唯一编号
cols_to_drop = ['提交答卷时间', '所用时间', '来源', '来源详情', '来自IP']
col_basic_info = ['年龄', '年级', '专业', '性别', '家庭所在地', '家庭经济情况', '学习成绩', '对待体育活动的态度']
col_psychological_test = [str(i) for i in range(1, 63)]  # 共计62道测试题
col_sports_test = ['IPAQ_1', 'IPAQ_2', 'IPAQ_3', 'IPAQ_4', 'IPAQ_5', 'IPAQ_6', 'IPAQ_7', '运动项目']
for i in cols_to_drop:
    col_names.append(i)
for i in col_basic_info:
    col_names.append(i)
for i in col_psychological_test:
    col_names.append(i)
for i in col_sports_test:
    col_names.append(i)

##################################读取原始数据
raw_data = pd.read_excel('data/问卷调查原始数据.xlsx', header=0)
raw_data.columns = col_names
# 删除列：提交答卷时间	所用时间	来源	来源详情	来自IP
raw_data = raw_data.drop(columns=cols_to_drop).reset_index(drop=True)

##################################添加体育量表的数据
df_list = pd.read_excel('data/国际体育活动量表问卷结果.xlsx', sheet_name=None)  # 读取体育量表结果
df_left = df_list['1']
for i in range(2, 8):
    df_right = df_list[str(i)]
    df_left = pd.merge(df_left, df_right, how='outer', on=['序号']).sort_values(by='序号').reset_index(drop=True)
    # print(df_left)
raw_data = pd.merge(raw_data, df_left, how='outer', on=['序号'])
# print('合并后的总表\n',raw_data)

##################################删除填写错误年龄的调查对象
index = 0
temp = []
for i in raw_data['年龄'].values:
    if type(i) != type(1):
        # print(i)
        temp.append(index)
    index = index + 1
raw_data = raw_data.drop(temp, axis=0).reset_index(drop=True)
# print('总表\n',raw_data)

##################################0-1编码代表性别和家庭所在地
raw_data['性别'] = raw_data['性别']-1
raw_data['家庭所在地'] = raw_data['家庭所在地']-1

##################################计算积极心理品质得分
raw_data['认知'] = 0
raw_data['人际'] = 0
raw_data['情感'] = 0
raw_data['公正'] = 0
raw_data['节制'] = 0
raw_data['超越'] = 0
raw_data['积极心理品质得分'] = 0

for i in list_cognition:
    raw_data['认知'] = raw_data['认知'] + raw_data[str(i)]
raw_data['认知'] = raw_data['认知'] / len(list_cognition)
for i in list_human_relation:
    raw_data['人际'] = raw_data['人际'] + raw_data[str(i)]
raw_data['人际'] = raw_data['人际'] / len(list_human_relation)
for i in list_emotion:
    raw_data['情感'] = raw_data['情感'] + raw_data[str(i)]
raw_data['情感'] = raw_data['情感'] / len(list_emotion)
for i in list_justice:
    raw_data['公正'] = raw_data['公正'] + raw_data[str(i)]
raw_data['公正'] = raw_data['公正'] / len(list_justice)
for i in list_temperate:
    raw_data['节制'] = raw_data['节制'] + raw_data[str(i)]
raw_data['节制'] = raw_data['节制'] / len(list_temperate)
for i in list_transcendence:
    raw_data['超越'] = raw_data['超越'] + raw_data[str(i)]
raw_data['超越'] = raw_data['超越'] / len(list_transcendence)
for i in range(1, 63):
    raw_data['积极心理品质得分'] = raw_data['积极心理品质得分'] + raw_data[str(i)]
raw_data['积极心理品质得分'] = raw_data['积极心理品质得分'] / 62
raw_data = raw_data.round({'认知': 2, '人际': 2, '情感': 2, '公正': 2, '节制': 2, '超越': 2, '积极心理品质得分': 2})

##################################根据问卷设置，部分题目的得分和程度反向，需要调整为正向
#你的家庭经济情况；学生成绩；体育态度；心理积极品质所有题目
raw_data['家庭经济情况'] = 5-raw_data['家庭经济情况']
raw_data['学习成绩'] = 5-raw_data['学习成绩']
raw_data['对待体育活动的态度'] = 4-raw_data['对待体育活动的态度']
raw_data['认知'] = 5-raw_data['认知']
raw_data['人际'] = 5-raw_data['人际']
raw_data['情感'] = 5-raw_data['情感']
raw_data['公正'] = 5-raw_data['公正']
raw_data['节制'] = 5-raw_data['节制']
raw_data['超越'] = 5-raw_data['超越']
raw_data['积极心理品质得分'] = 5-raw_data['积极心理品质得分']

##################################IPAQ数据处理
#数据清洗和异常值剔除:针对分钟数（1，-2）是该填的没填， （2，1） 不该填的填了   (1,2)该填但不清楚 （2，2）不该填，填了不清楚, (1,1)(2,-2)留下
# print(raw_data)
raw_data = raw_data[
    (((raw_data['IPAQ_1'] == 1) & (raw_data['IPAQ_2'] == 1)) |((raw_data['IPAQ_1'] == 2) & (raw_data['IPAQ_2'] == -2))) &
    (((raw_data['IPAQ_3'] == 1) & (raw_data['IPAQ_4'] == 1)) |((raw_data['IPAQ_3'] == 2) & (raw_data['IPAQ_4'] == -2))) &
    (((raw_data['IPAQ_5'] == 1) & (raw_data['IPAQ_6'] == 1)) |((raw_data['IPAQ_5'] == 2) & (raw_data['IPAQ_6'] == -2)))
    ]
raw_data = raw_data.drop(
    raw_data[(raw_data['剧烈运动天数'].isnull()&(~ raw_data['剧烈运动单日分钟数'].isnull())) |
             (raw_data['适度运动天数'].isnull()&(~ raw_data['适度运动单日分钟数'].isnull())) |
             (raw_data['步行天数'].isnull()&(~ raw_data['步行单日分钟数'].isnull()))].index).reset_index(drop=True)
# raw_data = raw_data.drop(raw_data[(raw_data['工作日久坐分钟数'].isnull())].index).reset_index(drop=True)
#做数据截断
for item in ['剧烈运动天数','适度运动天数','步行天数']: #每项运动天数不能超过7天
    raw_data = raw_data.drop(raw_data[(raw_data[item]>7)].index).reset_index(drop=True)
    raw_data[item] = raw_data[item].fillna(0)
for item in ['剧烈运动单日分钟数','适度运动单日分钟数','步行单日分钟数']: #每项运动分别不能超过180min
# for item in ['剧烈运动单日分钟数','适度运动单日分钟数']: #每项运动分别不能超过180min
    raw_data[item] = raw_data[item].apply(lambda x: 180 if x >180 else x)
    raw_data[item] = raw_data[item].fillna(0)
raw_data['剧烈运动一周分钟数'] = raw_data['剧烈运动天数']*raw_data['剧烈运动单日分钟数']
raw_data['适度运动一周分钟数'] = raw_data['适度运动天数']*raw_data['适度运动单日分钟数']
raw_data['步行一周分钟数'] = raw_data['步行天数']*raw_data['步行单日分钟数']

for items in ['剧烈运动一周分钟数','适度运动一周分钟数','步行一周分钟数']: #每周每项运动不超过1260min
    raw_data[item] = raw_data[item].apply(lambda x: 1260 if x > 1260 else x)
#计算活动力水平
raw_data['剧烈运动MET'] = raw_data['剧烈运动天数']*raw_data['剧烈运动单日分钟数']*8
raw_data['适度运动MET'] = raw_data['适度运动天数']*raw_data['适度运动单日分钟数']*4
raw_data['步行MET'] = raw_data['步行天数']*raw_data['步行单日分钟数']*3.3
raw_data['总MET'] = raw_data['剧烈运动MET']+raw_data['适度运动MET']+raw_data['步行MET']


# raw_data['工作日久坐分钟数'] = raw_data['工作日久坐分钟数'].apply(lambda x: 480 if x > 480 else x) #久坐不处理了，用不到
# raw_data['工作日久坐分钟数'] = raw_data['工作日久坐分钟数'].apply(lambda x: 10 if x < 10 else x)
# average_sit_time = raw_data['工作日久坐分钟数'].mean()
# print(average_sit_time) #用平均数填充
# raw_data['工作日久坐分钟数'] = raw_data['工作日久坐分钟数'].fillna(average_sit_time)

#评价活动强度：1弱 2中 3强
raw_data['身体活动水平'] = 0
rows = raw_data.shape[0]
for i in range(0, rows):
    if (((raw_data.loc[i, '总MET'] >=1500)&(raw_data.loc[i, '剧烈运动天数'] >=3))|
            ((raw_data.loc[i, '总MET'] >=3000)&(raw_data.loc[i, '剧烈运动天数']+raw_data.loc[i, '适度运动天数']+raw_data.loc[i, '步行天数'] >=7))):
        raw_data.loc[i, '身体活动水平'] = 3
    elif (((raw_data.loc[i, '总MET'] >= 600)& (raw_data.loc[i, '剧烈运动天数']+raw_data.loc[i, '适度运动天数']+raw_data.loc[i, '步行天数'] >=5))|
        ((raw_data.loc[i, '适度运动天数']+raw_data.loc[i, '步行天数'] >=5)&(raw_data.loc[i, '适度运动单日分钟数']+raw_data.loc[i, '步行单日分钟数'] >=30))|
        ((raw_data.loc[i, '剧烈运动天数'] >=3)&(raw_data.loc[i, '剧烈运动单日分钟数'] >=20))):
        raw_data.loc[i, '身体活动水平'] = 2
    else:
        raw_data.loc[i, '身体活动水平'] = 1

print(raw_data)
raw_data.to_csv('data/清洗后的研究数据20220220.csv', index=False,encoding='utf-8-sig')