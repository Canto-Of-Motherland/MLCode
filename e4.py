import numpy as np
import pandas as pd
import json
from typing import Union


def readFile(filename: str, index_col: int = 0, encoding: str = 'utf-8') -> pd.DataFrame:
    return pd.read_csv(filename, encoding=encoding, index_col=index_col)



def ProbabilityCalculate(data: pd.DataFrame, result: str) -> Union[dict, dict, dict]:
    dict_conditional_prob = {}
    list_category_result = data[result].unique()
    dict_num_result = data[result].value_counts()
    all_num_prob_result = data[result].value_counts().sum()
    dict_single_prop_result = {str(category_result): dict_num_result[category_result] / all_num_prob_result for category_result in list_category_result}
    for column in data.columns:
        if column != result:
            dict_num_category_column = data[column].value_counts().to_dict()
            dict_conditional_prob_category_column = {}
            for category in dict_num_category_column:
                list_category_column_result = data[data[column] == category][result].unique()
                dict_conditional_prob_category_column[category] = {str(category_result): data[data[column] == category][result].value_counts()[category_result] / all_num_prob_result / dict_single_prop_result[str(category_result)] if category_result in list_category_column_result else 0 for category_result in list_category_result}
            dict_conditional_prob[column] = dict_conditional_prob_category_column
        else:
            continue
    return dict_single_prop_result, dict_conditional_prob


def processDataTitanic(data_file: pd.DataFrame) -> pd.DataFrame:
    data_file = data_file.drop(['Cabin', 'Ticket', 'Fare', 'Sex'], axis=1)
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
    data_file = data_file.drop(['Name'], axis=1)
    data_file['Size'] = data_file['Parch'] + data_file['SibSp'] + 1
    bins_size = [1, 2, 5, 20]
    labels_size = ['Single', 'Small', 'Large']
    data_file = data_file.drop(['Parch', 'SibSp'], axis=1)
    data_file['Size'] = pd.cut(data_file['Size'], bins=bins_size, labels=labels_size, right=False)
    bins_age = [0, 10, 20, 30, 40, 50, 100]
    labels_age = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-100']
    data_file['Age'] = pd.cut(data_file['Age'], bins=bins_age, labels=labels_age, right=False)
    return data_file


def saveData(data: dict, filename: str, encoding: str = 'utf-8') -> None:
    with open(filename, 'w', encoding=encoding) as file:
        json.dump(data, file, ensure_ascii=False)


def predict(row_data: pd.Series, result: str, dict_prob: dict) -> str:
    list_prob = []
    dict_single_prop_result = dict_prob['dict_single_prop_result']
    dict_conditional_prob = dict_prob['dict_conditional_prob']
    for key, value in dict_single_prop_result.items():
        for row_key, row_value in row_data.items():
            if row_key != result:
                value *= dict_conditional_prob[row_key][str(row_value)][str(key)]
        dict_prop_item = {'category': key, 'prop': value}
        list_prob.append(dict_prop_item)
    return max(list_prob, key=lambda x: x['prop'])['category']
        
            
def test(data: pd.DataFrame, result: str, dict_prob: dict) -> pd.DataFrame:
    i = 0
    j = 0
    for _,row in data.iterrows():
        result_predict = predict(row, result, dict_prob)
        print(result_predict, row[result])
        if result_predict == str(row[result]):
            i += 1
        else:
            j += 1
    print('Accuracy: ', i / (i + j))


# data_file = readFile('train.csv')
# data_file = processDataTitanic(data_file)
# print(data_file.head())

# dict_single_prop_result, dict_conditional_prob = ProbabilityCalculate(data_file, 'Survived')
# saveData({'dict_single_prop_result': dict_single_prop_result, 'dict_conditional_prob': dict_conditional_prob}, 'titanic4.json')

with open('titanic1.json', 'r', encoding='utf-8') as f:
    dict_prob = json.load(f)

data_test = readFile('test.csv')
data_test = processDataTitanic(data_test)

test(data_test, 'Survived', dict_prob)