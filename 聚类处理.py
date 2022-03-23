import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
import copy

# cols_for_cluster = ['年龄', '性别', '家庭所在地', '家庭经济情况', '学习成绩', '对待体育活动的态度']
cols_for_cluster = [
  '剧烈运动MET', '适度运动MET', '步行MET', '总MET', '身体活动水平']


cols1 = [ '年龄', '性别', '家庭所在地', '家庭经济情况', '学习成绩', '对待体育活动的态度', '剧烈运动天数', '剧烈运动单日分钟数', '适度运动天数', '适度运动单日分钟数', '步行天数',
        '步行单日分钟数', '工作日久坐分钟数', '认知', '人际', '情感', '公正', '节制', '超越', '积极心理品质得分',
        '剧烈运动一周分钟数', '适度运动一周分钟数', '步行一周分钟数', '剧烈运动MET', '适度运动MET', '步行MET', '总MET', '身体活动水平','类别']


def cluster_process():
    raw_data = pd.read_csv('data/清洗后的研究数据20220220.csv', header=0)
    index_list = raw_data['序号'].values.tolist()
    raw_data = raw_data.set_index('序号')

    data = raw_data[cols_for_cluster]
    # print(data)
    data_scaled = normalize(data)
    # print(data_scaled)
    df_data_scaled = pd.DataFrame(data_scaled, columns=cols_for_cluster)
    plt.figure(figsize=(10,7))
    plt.title("Dendrograms")
    Z = shc.linkage(df_data_scaled, method='ward')
    dend = shc.dendrogram(Z)
    plt.show()
    labels_1 = shc.fcluster(Z, t=8, criterion='distance')
    print(labels_1)

    df_result = copy.deepcopy(raw_data)
    df_result['类别'] = labels_1
    print(df_result)
    df_result.to_csv('data/聚类结果_只有运动.csv', index=False, encoding='utf-8-sig' ) #只有2、3的类别有意义

def delete_outside_data():
    df = pd.read_csv('data/聚类结果_只有运动.csv', header=0)
    df = df.drop(df[df['类别'] == 1].index)
    df.to_csv('data/聚类结果_只有运动_剔除异常数据.csv', index=False, encoding='utf-8-sig')
    print(df)

def mynormalize():
    df = pd.read_csv('data/聚类结果20220322_剔除异常数据.csv', header=0)
    pd.set_option('display.max_rows', None)

    Input_cols = ['年龄', '性别', '家庭所在地', '家庭经济情况', '学习成绩', '对待体育活动的态度',
                  '剧烈运动MET', '适度运动MET', '步行MET', '总MET', '身体活动水平']

    Output_cols = ['认知', '人际', '情感', '公正', '节制', '超越', '积极心理品质得分']

    df[Output_cols] = df[Output_cols]/5

    print(df.max())

    # print(df)

if __name__ == '__main__':
    cluster_process()
    delete_outside_data()
    # normalize()


