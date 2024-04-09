import numpy as np
import pandas as pd
from typing import Union
from util import *


def readFile(filename: str, index_col: int = 0, encoding: str = 'utf-8') -> pd.DataFrame:
    return pd.read_csv(filename, encoding=encoding, index_col=index_col)


def giniCalculation(data: pd.DataFrame, result: str, threshold_gini: float, is_dispersed: dict) -> Union[dict, dict]:
    '''
    计算GINI指数并返回最佳划分属性与值
    '''
    list_gini = []  # 初始化GINI指数的存储列表
    list_result_category = data[result].unique()  # 总类别表
    for column in data.columns:  # 对于每一个属性
        if column != result:  # 如果属性不是结果属性
            list_category = data[column].unique()  # 获得该属性下的所有值的类别
            if is_dispersed[column]:  #　如果是离散型数据
                for category in list_category:  # 对于该属性下每一个可划分类别
                    list_result_category_is_category = data[data[column] == category][result].unique()  # 取得该类别时对应的结果列能取到的值
                    list_num_result_is_category = [data[data[column] == category][result].value_counts()[result_category] if result_category in list_result_category_is_category else 0 for result_category in list_result_category]
                    # 上述值的数量列表
                    total_result_is_category = sum(list_num_result_is_category)  # 求和（即该类别下所有值的数量）
                    list_result_category_not_category = data[data[column] != category][result].unique()  # 不取得该类别时对应的结果列能取到的值
                    list_num_result_not_category = [data[data[column] != category][result].value_counts()[result_category] if result_category in list_result_category_not_category else 0 for result_category in list_result_category]
                    # 上述值的数量列表
                    total_result_not_category = sum(list_num_result_not_category)  # 求和（即非该类别下所有值的数量）
                    gini_category_1 = 1 - sum([(num_result_is_category / total_result_is_category) ** 2 for num_result_is_category in list_num_result_is_category]) if total_result_is_category != 0 else threshold_gini
                    # 计算是该类别下的GINI指数
                    gini_category_2 = 1 - sum([(num_result_not_category / total_result_not_category) ** 2 for num_result_not_category in list_num_result_not_category]) if total_result_not_category != 0 else threshold_gini
                    # 计算非该类别下的GINI指数
                    total_category = total_result_is_category + total_result_not_category  # 总数据量
                    gini = total_result_is_category / total_category * gini_category_1 + total_result_not_category / total_category * gini_category_2  # 加权GINI指数
                    list_gini.append({'column': column, 'dispersed':is_dispersed[column], 'category': category, 'gini': gini})  # 将该类别该取值的GINI指数添加到列表中
                    # column是列名（属性名），dispersed是该列是否为离散型数据，category是取值，gini是GINI指数
            else:  # 如果是连续型数据
                list_category = sorted(list_category, reverse=True)  # 对所有值排序
                len_list_category = len(list_category)  # 取得列表长度
                for index_category, category in enumerate(list_category):  # 遍历列表  
                    if index_category <= len_list_category - 2:  # 如果不是最后一个值
                        category_next = list_category[index_category + 1]  # 取得当前值的下一个
                        category_split = (category + category_next) / 2  # 两值求平均数作为分类依据
                        list_result_category_is_category = data[data[column] <= category_split][result].unique()  # 与离散值的计算过程类似，只不过换成≤和＞两部分，此处不再赘述
                        list_num_result_is_category = [data[data[column] <= category_split][result].value_counts()[result_category] if result_category in list_result_category_is_category else 0 for result_category in list_result_category]
                        total_result_is_category = sum(list_num_result_is_category)
                        list_result_category_not_category = data[data[column] > category_split][result].unique()
                        list_num_result_not_category = [data[data[column] > category_split][result].value_counts()[result_category] if result_category in list_result_category_not_category else 0 for result_category in list_result_category]
                        total_result_not_category = sum(list_num_result_not_category)
                        gini_category_1 = 1 - sum([(num_result_is_category / total_result_is_category) ** 2 for num_result_is_category in list_num_result_is_category]) if total_result_is_category != 0 else threshold_gini
                        gini_category_2 = 1 - sum([(num_result_not_category / total_result_not_category) ** 2 for num_result_not_category in list_num_result_not_category]) if total_result_not_category != 0 else threshold_gini
                        total_category = total_result_is_category + total_result_not_category
                        gini = total_result_is_category / total_category * gini_category_1 + total_result_not_category / total_category * gini_category_2
                        list_gini.append({'column': column, 'dispersed':is_dispersed[column], 'category': category_split, 'gini': gini})
                    else:
                        break  # 如果到最后一条了直接跳出循环，防止超出索引
    list_gini = sorted(list_gini, key=lambda x: x['gini'], reverse=False)  # 按照GINI指数升序排序
    column_name = list_gini[0]['column']  # 获得列名（切分依据）
    column_dispersed = list_gini[0]['dispersed']  # 获得数据类型
    category = list_gini[0]['category']  # 获得取值（切分依据）
    if column_dispersed:  # 如果是离散型数据
        group_is_category = data[data[column_name] == category].drop(column_name, axis=1)  # 等于与不等划分
        group_not_category = data[data[column_name] != category].drop(column_name, axis=1)
    else:
        group_is_category = data[data[column_name] <= category].drop(column_name, axis=1)  # 小于等于与大于划分
        group_not_category = data[data[column_name] > category].drop(column_name, axis=1)
    data_next = {'group_is_category': group_is_category, 'group_not_category': group_not_category}  # 存储划分后的数据
    return list_gini[0], data_next  # 返回数据


def decisionTree(data: pd.DataFrame, result: str, threshold_gini: float, threshold_sample: int, is_dispersed: dict, tree: dict = {}, node_link: str = 'root', node_id: int = 0) -> Union[dict, int]:
    '''
    递归构建CART树
    '''
    node_info = {'node_name': '', 'node_link': str(node_link), 'node_type': 0, 'node_child': []}
    if len(data) <= threshold_sample or len(data.columns) == 1:  # 如果样本数已不足以新建节点或没有可划分节点
        list_category = data[result].unique()  # 和ID-3一致的投票机制，不再赘述
        list_num = [data[result].value_counts()[category] for category in list_category]
        node_info['node_name'] = str(list_category[list_num.index(max(list_num))])
        node_info['node_type'] = 1
        tree[str(node_id)] = node_info
        return tree, node_id
    best_column, data_next = giniCalculation(data, result, threshold_gini, is_dispersed)  # 计算GINI指数并返回对应的信息和值
    if best_column['gini'] <= threshold_gini:  # 如果GINI指数低于阈值
        list_category = data[result].unique()  # 还是投票机制，不再赘述
        list_num = [data[result].value_counts()[category] for category in list_category]
        node_info['node_name'] = str(list_category[list_num.index(max(list_num))])
        node_info['node_type'] = 1
        tree[str(node_id)] = node_info
        return tree, node_id
    node_info['node_name'] = best_column['column']  # 找到最佳分类属性赋值给节点名
    tree[str(node_id)] = node_info  # 下同ID-3
    node_id_static = node_id

    data_next_is_category = data_next['group_is_category']  #去得划分后的数据（类别为是）
    if len(data_next_is_category) == 0:  # 如果数据集是空的
        return tree, node_id  # 直接返回
    node_id += 1  # 节点ID自加
    tree[str(node_id_static)]['node_child'].append(str(node_id))  # 向下级节点查找列表中直接添加入左节点
    if best_column['dispersed']:  #　如果是离散数据
        tree, node_id = decisionTree(data_next_is_category, result, threshold_gini, threshold_sample, is_dispersed, tree, best_column['category'], node_id)
        # 递归运行下级节点构建
    else:
        tree, node_id = decisionTree(data_next_is_category, result, threshold_gini, threshold_sample, is_dispersed, tree, '<= ' + str(best_column['category']), node_id)
        # 递归运行下级节点构建，不过尤其注意这里的<=，这是为了便于绘图区分节点用的，实际上不需要这个，左节点就是<=，右节点就是>
    data_next_not_category = data_next['group_not_category']
    if len(data_next_not_category) == 0:
        return tree, node_id
    node_id += 1
    tree[str(node_id_static)]['node_child'].append(str(node_id))  # 向下级节点查找列表中直接添加入右节点
    if best_column['dispersed']:
        list_node_link = data[best_column['column']].unique().tolist()  # 取得所有的类别
        list_node_link.remove(best_column['category'])  # 删掉划分类别
        list_node_link = [str(item) for item in list_node_link]  # 将其他类别列出
        node_link = ', '.join(list_node_link)  # 合并类别
        tree, node_id = decisionTree(data_next_not_category, result, threshold_gini, threshold_sample, is_dispersed, tree, node_link, node_id)
        # 递归运行下级节点构建，这里的合并类别确实是有效果的，没这个不太好做测试
    else:
        tree, node_id = decisionTree(data_next_not_category, result, threshold_gini, threshold_sample, is_dispersed, tree, '> ' + str(best_column['category']), node_id)
        # 递归运行下级节点构建，原理同连续型的部分
    return tree, node_id


def saveData(data: dict, filename: str, encoding: str = 'utf-8') -> None:
    with open(filename, 'w', encoding=encoding) as file:
        json.dump(data, file, ensure_ascii=False)


def processDataTitanic(data_file: pd.DataFrame) -> pd.DataFrame:
    '''
    处理Titanic数据集
    '''
    data_file = data_file.drop(['Cabin', 'Ticket'], axis=1)
    age_mean = data_file['Age'].mean()
    data_file['Age'] = data_file['Age'].fillna(age_mean)
    data_file['Embarked'] = data_file['Embarked'].fillna('S')

    def title(name):
        name_1 = name.split(',')[1]
        name_2 = name_1.split('.')[0]
        name_3 = name_2.strip()
        return name_3
    
    data_file['Title'] = data_file['Name'].map(title)

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
                }
    data_file['Title'] = data_file['Title'].map(title_map)
    data_file['Fare'] = data_file['Fare'].fillna(data_file['Fare'].mean())
    data_file = data_file.drop(['Name'], axis=1)
    data_file['Size'] = data_file['Parch'] + data_file['SibSp'] + 1
    data_file = data_file.drop(['Parch', 'SibSp'], axis=1)
    return data_file


def searchNext(row_data: pd.Series, node_id: str, tree: dict, is_dispersed: dict) -> str:
    if tree[node_id]['node_type'] == 0:  # 如果不是叶子节点
        node_dispersed = is_dispersed[tree[node_id]['node_name']]  # 根据节点名称检查节点是离散型还是连续型
        if node_dispersed:  # 如果节点是离散型
            for node_next_id in tree[node_id]['node_child']:  # 遍历所有子节点（其实也只有两个节点）
                if str(row_data[tree[node_id]['node_name']]) in tree[node_next_id]['node_link'].split(', '):  # 如果该条数据对应的值在节点连接分割出的类别列表中
                    return searchNext(row_data, node_next_id, tree, is_dispersed)  # 进行下一层节点查找
                else:
                    continue
        else:  # 如果节点是连续型
            node_split = float(tree[tree[node_id]['node_child'][0]]['node_link'].split(' ')[1])  # 获得≥或＜与后面的分割值
            if float(row_data[tree[node_id]['node_name']]) <= node_split:  # 如果该条数据对应的值小于等于分割值
                return searchNext(row_data, tree[node_id]['node_child'][0], tree, is_dispersed)  # 进行下一层节点查找
            else:  # 如果大于分割值
                return searchNext(row_data, tree[node_id]['node_child'][1], tree, is_dispersed)  # 进行下一层节点查找
    else:
        return tree[node_id]['node_name']  # 返回最终分类结果
    

def predict(row_data: pd.Series, tree: dict, is_dispersed: dict) -> str:
    return searchNext(row_data, '0', tree, is_dispersed)


def testTree(data: pd.DataFrame, result: str, tree: dict, is_dispersed: dict) -> None:
    i = 0
    j = 0
    for _, row in data.iterrows():
        result_predict = predict(row, tree, is_dispersed)
        if result_predict == str(row[result]):
            i += 1
        else:
            j += 1
    print('Accuracy: ', i / (i + j))
        

def testTreeMatrix(data: pd.DataFrame, result: str, tree: dict, is_dispersed: dict) -> Union[list, list, int]:
    list_num_result = [[0 for _ in data[result].unique()] for _ in data[result].unique()]
    list_result = [str(i) for i in data[result].unique()]
    for _, row in data.iterrows():
        result_predict = predict(row, tree, is_dispersed)
        list_num_result[list_result.index(str(row[result]))][list_result.index(result_predict)] += 1
    return list_result, list_num_result, len(data)
        


data_file = readFile('train.csv')
data_file = processDataTitanic(data_file)
tree, node = decisionTree(data_file, 'Survived', 0.2, 10, {'Pclass': True, 'Sex': True, 'Age': False, 'Fare': False, 'Embarked': True, 'Title': True, 'Size': False})

# data_test = readFile('test.csv')
# data_test = processDataTitanic(data_test)

# testTree(data_test, 'Survived', tree, {'Pclass': True, 'Sex': True, 'Age': False, 'Fare': False, 'Embarked': True, 'Title': True, 'Size': False})

saveData(tree, 'titanic3-2.json')
# with open('titanic3-2.json', 'r', encoding='utf-8') as f:
#     tree = json.load(f)


# data_1 = readFile('train.csv')
# data_2 = readFile('test.csv')
# data_file = pd.concat([data_1, data_2])
# data_file = processDataTitanic(data_file)
# print(data_file)
# train_data, test_data = splitData(data_file, n_splits=2)
# tree = decisionTree(train_data, 'Survived', 0.2, 10, {'Pclass': True, 'Sex': True, 'Age': False, 'Fare': False, 'Embarked': True, 'Title': True, 'Size': False})[0]
# for _ in range(3):
#     train_data, test_data = splitData(data_file, n_splits=7)
#     tree = decisionTree(train_data, 'Survived', 0.2, 10, {'Pclass': True, 'Sex': True, 'Age': False, 'Fare': False, 'Embarked': True, 'Title': True, 'Size': False})[0]
#     testTree(test_data, 'Survived', tree, {'Pclass': True, 'Sex': True, 'Age': False, 'Fare': False, 'Embarked': True, 'Title': True, 'Size': False})



# label, matrix_data, total_data = testTreeMatrix(test_data, 'Survived', tree, {'Pclass': True, 'Sex': True, 'Age': False, 'Fare': False, 'Embarked': True, 'Title': True, 'Size': False})
# matrix_data = [[j / total_data for j in i] for i in matrix_data]
# drawConfusionMatrix(label, matrix_data)
drawGraph('titanic3-2.json', 'graph3-2')