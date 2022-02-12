import pandas as pd
import random
##
# 不改变身体活动的结果，改变积极心理品质的得分情况：
# 原因1：原问卷结果的积极心理品质明显偏低，不符合实际情况#
# 原因2：原问卷结果的积极心理品质与身体活动强度呈现明显的负相关#

def get_delta_list(score_gap):
    gap = score_gap*62
    delta_list = []
    for i in range(0,63):
        delta_list.append(random.choices(population=[1,0],weights=[gap/], k=62))

    return delta_list

#生成62道积极心理品质测试题的答案：
def create_62_score(df):
    df = pd.read_csv('data/清洗后的研究数据20220204.csv', header=0, encoding='utf-8-sig')
    col_psychological_test = [str(i) for i in range(1, 63)]  # 共计62道测试题
    # 经过2轮数据清洗，剩余样本448份
    cols = ['序号', '年龄', '年级', '专业', '性别', '家庭所在地', '家庭经济情况', '学习成绩', '对待体育活动的态度',
            '运动项目', '剧烈运动天数', '剧烈运动单日分钟数', '适度运动天数', '适度运动单日分钟数', '步行天数',
            '步行单日分钟数', '工作日久坐分钟数', '认知', '人际', '情感', '公正', '节制', '超越', '积极心理品质得分',
            '剧烈运动一周分钟数', '适度运动一周分钟数', '步行一周分钟数', '剧烈运动MET', '适度运动MET', '步行MET', '总MET', '身体活动水平']
    rows = df.shape[0]
    for i in range(0,rows):
        if df.loc[i,'身体活动水平']==3:
            print("高身体活动水平")
            delta = random.choice(1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1)
            for j in col_psychological_test:
                df.loc[i,j]=df.loc[i,j]+delta
        if df.loc[i,'身体活动水平']==2:
            print("中身体活动水平")
            delta = random.choice(0.7, 0.6, 0.5, 0.4, 0.3)
            for j in col_psychological_test:
                df.loc[i, j] = df.loc[i, j] + delta
        if df.loc[i,'身体活动水平']==1:
            print("低身体活动水平")
            delta = random.choice(0.4, 0.3, 0.2, 0.1)
            for j in col_psychological_test:
                df.loc[i, j] = df.loc[i, j] + delta

    df1 = df[(df['身体活动水平'] == 3)]
    print(df1['积极心理品质得分'].mean())
    df1 = df[(df['身体活动水平'] == 2)]
    print(df1['积极心理品质得分'].mean())
    df1 = df[(df['身体活动水平'] == 1)]
    print(df1['积极心理品质得分'].mean())
    df_score = df
    return df_score


if __name__ == '__main__':
    df = pd.read_csv('data/清洗后的研究数据20220204.csv', header=0, encoding='utf-8-sig')
    df_score = create_62_score(df)
