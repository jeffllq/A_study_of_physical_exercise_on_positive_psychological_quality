import pandas as pd
import statsmodels
##
# 经过2轮数据清洗，剩余样本448份
cols = ['序号', '年龄', '年级', '专业', '性别', '家庭所在地', '家庭经济情况', '学习成绩', '对待体育活动的态度',
        '运动项目', '剧烈运动天数', '剧烈运动单日分钟数', '适度运动天数', '适度运动单日分钟数', '步行天数',
        '步行单日分钟数', '工作日久坐分钟数', '认知', '人际', '情感', '公正', '节制', '超越', '积极心理品质得分',
        '剧烈运动一周分钟数', '适度运动一周分钟数', '步行一周分钟数', '剧烈运动MET', '适度运动MET', '步行MET', '总MET', '身体活动水平']


#一下为克朗巴赫系数的python实现，也可以spss操作
def cronbacha(df):
    #信度检验 克朗巴赫系数 Cronbacha信度系数
    col_psychological_test = [str(i) for i in range(1, 63)]
    # print(df)
    df = df[col_psychological_test]
    print(df)
    print(df.columns)
    total_row = df.sum(axis=1)
    print (total_row)
    sy = total_row.var()
    print (sy)
    var_column =  df.var()
    si = var_column.sum()
    print (si )
    r = (62/61)*((sy-si)/sy)
    if (r<0.98) & (r>0.7):
        print("cronbacha α系数：",r,"属于高信度")
    print (r) # 结果 0.97 。根据资料，介于 0.70－0.98 均属高信度

#效度分析直接使用spss操作，这里不python实现，因为没有找到远离解释



if __name__ == '__main__':
    df = pd.read_csv('data/清洗后的研究数据20220204.csv', encoding='utf-8-sig', header=0)
    cols_names = df.columns.tolist()
    # print(cols_names)
    # cronbacha(df) #计算积极心理品质问卷的克朗巴赫系数，检验信度
    # df_tmp = df[['年龄', '性别', '家庭所在地', '家庭经济情况', '学习成绩', '对待体育活动的态度',
    #      '认知', '人际', '情感', '公正', '节制', '超越', '积极心理品质得分',
    #     '剧烈运动一周分钟数', '适度运动一周分钟数', '步行一周分钟数', '剧烈运动MET', '适度运动MET', '步行MET', '总MET', '身体活动水平']]


    df1 = df[(df['身体活动水平']==3)]
    print(df1['积极心理品质得分'].mean())
    df1 = df[(df['身体活动水平']==2)]
    print(df1['积极心理品质得分'].mean())
    df1 = df[(df['身体活动水平']==1)]
    print(df1['积极心理品质得分'].mean())
    # print('身体活动水平均值', df1['身体活动水平'].mean())
    # print('积极心理品质得分均值', df1['积极心理品质得分'].mean())
    # df1 = df1[['年龄', '性别', '家庭所在地', '家庭经济情况',
    #      '认知', '人际', '情感', '公正', '节制', '超越', '积极心理品质得分',
    #     '剧烈运动一周分钟数', '适度运动一周分钟数', '步行一周分钟数', '剧烈运动MET', '适度运动MET', '步行MET', '总MET', '身体活动水平']]
    # corr = df1.corr()
    # print(corr)
    # corr.to_csv('tmp.csv', index=True)

    # df1 = df[(df['学习成绩']<3) & (df['对待体育活动的态度']<3)]
    # print(df1['积极心理品质得分'].corr(df1['总MET']))
    # print('身体活动水平均值', df1['身体活动水平'].mean())
    # print('积极心理品质得分均值', df1['积极心理品质得分'].mean())

