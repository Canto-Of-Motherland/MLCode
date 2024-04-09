import numpy as np
import pandas as pd
from typing import Union
from util import *


def processDataTitanic(data_file: pd.DataFrame) -> pd.DataFrame:
    '''
    处理Titanic数据集
    '''
    data_file = data_file.drop(['Cabin', 'Ticket', 'Fare'], axis=1)  # 删了Cabin、Ticket、Fare三列
    age_mean = data_file['Age'].mean()  # 计算Age列的平均值
    data_file['Age'] = data_file['Age'].fillna(age_mean)  # 填充Age列的缺失值为平均值
    data_file['Embarked'] = data_file['Embarked'].fillna('S')  # 填充Embarked列的缺失值为'S'

    def title(name: str) -> str:  # 定义分割姓名并提取Title的函数
        name_1 = name.split(',')[1]
        name_2 = name_1.split('.')[0]
        name_3 = name_2.strip()
        return name_3

    data_file['Title'] = data_file['Name'].map(title)  # 分割并提取Title

    title_map = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                }  # Title与身份之间的映射集
    data_file['Title'] = data_file['Title'].map(title_map)  # 完成映射替换
    data_file = data_file.drop(['Name'], axis=1)  # 删除原Name列
    data_file['Size'] = data_file['Parch'] + data_file['SibSp'] + 1  # 新增Size列，表示家庭成员数量（包括自己）
    bins_size = [1, 2, 5, 20]  # 划分Size列的区间
    labels_size = ['Single', 'Small', 'Large']  # 划分Size列的标签
    data_file['Size'] = pd.cut(data_file['Size'], bins=bins_size, labels=labels_size, right=False)  # 划分Size列
    bins_age = [0, 10, 20, 30, 40, 50, 100]  # 划分Age列的区间
    labels_age = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-100']  # 划分Age列的标签
    data_file['Age'] = pd.cut(data_file['Age'], bins=bins_age, labels=labels_age, right=False)  # 划分Age列
    return data_file  # 返回处理后的数据（pandas.DataFrame类型）


def entropyCalculation(data: pd.DataFrame, result: str) -> Union[str, dict]:
    """
    计算信息增益并分割数据集
    """
    list_gain = []  # 存储信息增益计算结果的列表
    list_category_result = data[result].unique()  # 获得结果列的所有类别
    list_num_result = [data[result].value_counts()[category] for category in list_category_result]  # 根据所有类别名称得出每个类别下的结果数量
    num_total_result = sum(list_num_result)  # 总结果数量
    total_entropy = sum([-num / num_total_result * np.log2(num / num_total_result) for num in list_num_result])  # 计算总信息熵（未分割）
    for column in data.columns:  # 遍历所有的列
        if column != result:  # 如果不是结果咧（即所有可以用于分类的列）
            list_category = data[column].unique()  # 该列下所有类别
            entropy_column = 0.0  # 该列的信息熵（初始化）
            for category in list_category:  # 遍历该列下的所有类别
                list_category_result_column = data[data[column] == category][result].unique()  # 该类别下所有结果（因为有的结果可能在该类别中没有）
                list_num_result_column = [data[data[column] == category][result].value_counts()[category_result] if category_result in list_category_result_column else 0 for category_result in list_category_result]
                # 依据数据集中最初的结果类别，如果该类别下结果存在，则计算其数量，反之就为0
                num_total_result_column = sum(list_num_result_column)  # 计算该类别下数据总量（行）
                entropy_column_category = sum([-num / num_total_result_column * np.log2(num / num_total_result_column) if num else 0 for num in list_num_result_column])  # 计算该类别下的信息熵
                entropy_column += num_total_result_column / num_total_result * entropy_column_category  # 计算加权后的信息熵并加到该属性的分类信息熵中
            gain = total_entropy - entropy_column  # 计算该属性（列）的信息增益
            column_gain = {'column_name': column, 'column_entropy': gain}  # 存储该属性（列）的信息增益
            list_gain.append(column_gain)  # 将该属性（列）的信息增益添加到列表中
    list_gain = sorted(list_gain, key=lambda x:x['column_entropy'], reverse=True)  # 根据列表中字典项的column_gain属性进行降序排序
    column_name = list_gain[0]['column_name']  # 选出信息增益最高的分类属性并获得其名称
    groups = data.groupby(column_name)  # 依据该属性对数据集进行分割（groupby方法仅可用于完全分割）
    data_next = {category_name: category_group.drop(columns=[column_name]) for category_name, category_group in groups}  # 存储分割后的数据集
    return column_name, data_next  # 返回信息增益最高的分类属性名称和分割后的数据集


def decisionTree(data: pd.DataFrame, result: str, tree: dict = {}, node_link: str = 'root', node_id: int = 0) -> Union[dict, int]:
    """
    递归构建决策树
    """
    node_info = {'node_name': '', 'node_link': node_link, 'node_type': 0, 'node_child': []}  # 存储节点信息的字典并初始化节点类型为分支节点，节点名称和下级节点列表为空
    if len(data[result].unique()) == 1:  # 如果结果列中所有值都相同
        node_info['node_name'] = str(data[result].unique()[0])  # 获得结果列中的值并设置为该叶子节点名称
        node_info['node_type'] = 1  # 设置节点类型为叶子结点
        tree[str(node_id)] = node_info  # 存储该叶子节点信息到树中，键为节点ID
        return tree, node_id  # 返回树和现在的ID自加结果
    if len(data.columns) == 1:  # 如果只有一个属性（没有划分属性，只有结果列）
        list_category = data[result].unique()  # 获得结果列的所有类别
        list_num = [data[result].value_counts()[category] for category in list_category]  # 获得结果列中每个类别的数量
        node_info['node_name'] = str(list_category[list_num.index(max(list_num))])  # 找到数量最多的列（投票），设置为该叶子节点名称
        node_info['node_type'] = 1  # 设置节点类型为叶子结点
        tree[str(node_id)] = node_info  # 存储该叶子节点信息到树中，键为节点ID
        return tree, node_id  # 返回树和现在的ID自加结果
    best_column_name, best_data_next = entropyCalculation(data, result)  # 可以分割的情况下，就计算信息增益，获得最佳划分属性，划分数据集
    if len(best_data_next.items()) == 1:  # 如果最佳划分属性下只有一个数据集（即还是没得可分，划分属性下只有一种类别）
        list_category = data[result].unique()  # 下面的几行代码还是做投票那一套，不再赘述
        list_num = [data[result].value_counts()[category] for category in list_category]
        node_info['node_name'] = str(list_category[list_num.index(max(list_num))])
        node_info['node_type'] = 1
        tree[str(node_id)] = node_info
        return tree, node_id
    node_info['node_name'] = best_column_name  # 设置分支节点的名称为最佳划分属性
    tree[str(node_id)] = node_info  # 将分支节点存入树中（此时node_child还是空的···）
    node_id_static = node_id  # 预存ID（用于）访问刚刚的分支节点然后修改node_child
    for category_name, category_data in best_data_next.items():  # 遍历最佳划分属性下的所有类别数据集
        node_id += 1  # ID自加
        tree[str(node_id_static)]['node_child'].append(str(node_id))  # 将新建的节点ID塞到node_child里面去便于访问下级节点
        tree, node_id = decisionTree(category_data, result, tree, category_name, node_id)  # 递归调用，构建下级节点（这里的node_id赋值是防止ID重复）
    return tree, node_id  # 返回树和现在的ID自加结果


def searchNext(row_data: pd.Series, node_id: str, tree: dict) -> str:
    '''
    下级节点查找
    '''
    if tree[node_id]['node_type'] == 0:  # 如果当前节点是分支节点
        for node_next_id in tree[node_id]['node_child']:  # 遍历当前节点的所有下级节点的ID
            if tree[node_next_id]['node_link'] == row_data[tree[node_id]['node_name']]:  # 如果下级节点的链接值与本条数据的对应值相同
                return searchNext(row_data, node_next_id, tree)  # 继续递归向下查找
            else:
                continue
    else:
        return tree[node_id]['node_name']  # 如果不是分支节点（即是叶子结点），终止递归逐层返回结果


def predict(row_data: pd.Series, tree: dict) -> str:
    return searchNext(row_data, '0', tree)


def testTree(data: pd.DataFrame, result: str, tree: dict) -> None:
    '''
    测试分类准确率\n
    输入：测试集、结果列名、决策树\n
    输出：None
    '''
    i = 0
    j = 0
    for _, row in data.iterrows():  # 遍历数据集中的每一条数据
        if searchNext(row, '0', tree) == str(row[result]):  # 测试数据分类是否正确
            i += 1  # 计数
        else:
            j += 1
    print('Accuracy: ', i / (i + j))  # 打印准确率


def testTreeMatrix(data: pd.DataFrame, result: str, tree: dict) -> Union[list, list, int]:
    '''
    测试分类结果，输出每组的实际值和预测值形成二维矩阵（可扩展为多分类），用于混淆矩阵生成\n
    输入：测试集、结果列名、决策树\n
    输出：结果类别标签、各类预测结果与真实结果矩阵、数据总量
    '''
    list_num_result = [[0 for _ in data[result].unique()] for _ in data[result].unique()]  # 初始化二维矩阵用于存储信息
    list_result = [str(i) for i in data[result].unique()]  # 初始化结果类别标签集
    for _, row in data.iterrows():
        result_predict = searchNext(row, '0', tree)  # 预测结果
        list_num_result[list_result.index(str(row[result]))][list_result.index(result_predict)] += 1  # 预测结果与真实结果行列对应的值自加
    return list_result, list_num_result, len(data)  # 返回数据


data_file = readFile('train.csv')
data_file = processDataTitanic(data_file)
saveData(decisionTree(data_file, 'Survived')[0], 'titanic3-1.json')

# with open('titanic3-1.json', 'r', encoding='utf-8') as f:
#     tree = json.load(f)

# data_test = readFile('test.csv')
# data_test = processDataTitanic(data_test)
# testTree(data_test, 'Survived', tree)
# label, matrix_data, total_data = testTreeMatrix(data_test, 'Survived', tree)
# matrix_data = [[j / total_data for j in i] for i in matrix_data]
# drawConfusionMatrix(label, matrix_data)


drawGraph('titanic3-1.json', 'graph3-1')

# data_1 = readFile('train.csv')
# data_2 = readFile('test.csv')
# data_file = pd.concat([data_1, data_2])
# data_file = processDataTitanic(data_file)
# train_data, test_data = splitData(data_file, n_splits=5)
# tree = decisionTree(train_data, 'Survived')[0]
# label, matrix_data, total_data = testTreeMatrix(test_data, 'Survived', tree)
# matrix_data = [[j / total_data for j in i] for i in matrix_data]
# drawConfusionMatrix(label, matrix_data)



# for _ in range(3):
#     train_data, test_data = splitData(data_file, n_splits=7)
#     print(len(train_data), len(test_data))
#     tree = decisionTree(train_data, 'Survived')[0]
#     testTree(test_data, 'Survived', tree)