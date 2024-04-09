from graphviz import Digraph
import matplotlib.pyplot as plt
import json
from typing import Union
import random
import pandas as pd
import numpy as np


def readFile(filename: str, index_col: int = 0, encoding: str = 'utf-8') -> pd.DataFrame:
    return pd.read_csv(filename, encoding=encoding, index_col=index_col)


def saveData(data: dict, filename: str, encoding: str = 'utf-8') -> None:
    with open(filename, 'w', encoding=encoding) as file:
        json.dump(data, file, ensure_ascii=False)


def drawGraph(input_path: str, output_path: str) -> None:
    with open(input_path, 'r', encoding='utf-8') as f:  # 读取存储树的json文件
        data = json.load(f)
    
    graph = Digraph()  # 新建图
    for node_id, node_info in data.items():  # 遍历节点信息
        if node_info['node_type'] == 0:  # 如果是分支节点
            graph.node(node_id, node_info['node_name'], shape='box', style='rounded,filled', fontname='Sans Not-Rotated 14', fontcolor='#337277', fillcolor='#82d3c2', color='#337277')
            # 设置节点ID为node_id，节点信息为node_name，方形，圆角，填色，黑体，边框填充与字体颜色
        else:
            graph.node(node_id, node_info['node_name'], shape='box', style='rounded,filled', fontname='Sans Not-Rotated 14', fontcolor='#ffffff', fillcolor='#e4b67a', color='#c7965c')
            # 设置节点ID为node_id，节点信息为node_name，方形，圆角，填色，黑体，边框填充与字体颜色，为了鲜明区别分支节点，所以颜色不一样
    for node_id, node_info in data.items():  # 构建连接
        if node_info['node_child'] != []:  # 如果有下级节点
            for item in node_info['node_child']:  # 对于每个下级节点
                graph.edge(node_id, item, color='#7eb8bc', splines='ortho', label=str(data[item]['node_link']), fontcolor='#337277', fontname='Sans Not-Rotated 14')
                # 设置直线连接，起始节点ID，终止节点ID，以及连接的标签（node_link），字体颜色与字体
    graph.view()  # 预览
    graph.render(output_path, view=True)  # 保存


def splitData(data: pd.DataFrame, n_splits: int = 5, random_index: bool = True) -> Union[dict, dict]:
    len_data = len(data)
    index_head = data.iloc[1][0]
    if random_index:
        index_split = random.randint(index_head, index_head + n_splits - 1)
    else:
        index_split = 0

    train_index_list = []
    test_index_list = []
    for index in range(index_head, index_head + len_data):
        if (index - index_split) % n_splits == 0:
            test_index_list.append(index)
        else:
            train_index_list.append(index)
    train_data = data.loc[train_index_list]
    test_data = data.loc[test_index_list]
    return train_data, test_data


def drawConfusionMatrix(label: list, data: list) -> None:
    data = np.array(data, dtype=np.float64)
    plt.imshow(data, interpolation='nearest', cmap=plt.cm.Blues, origin='upper')
    plt.title('Confusion Matrix')
    plt.colorbar()
    marks = np.arange(len(label))
    plt.xticks(marks, label)
    plt.yticks(marks, label)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(len(label)):
        for j in range(len(label)):
            plt.text(j, i, '{:.2f}'.format(data[i, j]), horizontalalignment='center', verticalalignment='center')
    plt.show()


    