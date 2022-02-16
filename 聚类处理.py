import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
import copy

cols = ['年龄', '性别', '家庭所在地', '家庭经济情况', '学习成绩', '对待体育活动的态度']
cols1 = [ '年龄', '性别', '家庭所在地', '家庭经济情况', '学习成绩', '对待体育活动的态度', '剧烈运动天数', '剧烈运动单日分钟数', '适度运动天数', '适度运动单日分钟数', '步行天数',
        '步行单日分钟数', '工作日久坐分钟数', '认知', '人际', '情感', '公正', '节制', '超越', '积极心理品质得分',
        '剧烈运动一周分钟数', '适度运动一周分钟数', '步行一周分钟数', '剧烈运动MET', '适度运动MET', '步行MET', '总MET', '身体活动水平','类别']


def cluster_process():
    raw_data = pd.read_csv('data/清洗后的研究数据20220204.csv', header=0)
    index_list = raw_data['序号'].values.tolist()
    raw_data = raw_data.set_index('序号')

    #将性别中的2处理成0
    for i in index_list:
        if raw_data.loc[i,'性别']==2:
            raw_data.loc[i, '性别']=0

    data = raw_data[cols]
    # print(data)
    data_scaled = normalize(data)
    # print(data_scaled)
    df_data_scaled = pd.DataFrame(data_scaled, columns=cols)
    plt.figure(figsize=(10,7))
    plt.title("Dendrograms")
    Z = shc.linkage(df_data_scaled, method='ward')
    dend = shc.dendrogram(Z)
    # plt.show()
    labels_1 = shc.fcluster(Z,t=0.6 ,criterion='distance')
    print(labels_1)

    df_result = copy.deepcopy(raw_data)
    df_result['类别'] = labels_1
    print(df_result)
    df_result.to_csv('临时聚类结果.csv', index=True, encoding='utf-8-sig' )


if __name__ == '__main__':
    # cluster_process()
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('临时聚类结果.csv', header=0, index_col=['序号'])
    # df = df[cols1]
    # df1 = df[df['类别']==1]
    # df2 = df[df['类别']==2]
    # df3 = df[df['类别']==3]
    # print('类别1：积极心理品质',df1['积极心理品质得分'].mean(), '身体活动水平',df1['身体活动水平'].mean())
    # print('类别2：积极心理品质',df2['积极心理品质得分'].mean(), '身体活动水平',df2['身体活动水平'].mean())
    # print('类别3：积极心理品质',df3['积极心理品质得分'].mean(), '身体活动水平',df3['身体活动水平'].mean())
    # print(df1.describe())
    # print('************************************************************************')
    # print(df2.describe())
    # print('************************************************************************')
    # print(df3.describe())

    df1 = df[df['类别']==1]
    df2 = df[df['类别']==2]
    df3 = df[df['类别']==3]
