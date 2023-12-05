import random
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pandas.core.computation.check import NUMEXPR_INSTALLED
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

#!/usr/bin/env python
# coding: utf-8

# 1. Dataset Overview - buyer dataset

buyer_dataset = pd.read_csv("./noon_perfumes_buyer_dataset.csv")
print(f"Rows: {buyer_dataset.shape[0]}\nColumns: {buyer_dataset.shape[1]}")
# Using with to print all columns
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(buyer_dataset.head())

buyer_dataset.info()

buyer_dataset.describe()

buyer_dataset.describe(exclude = np.number)

buyer_dataset.isnull().sum()

# [dirty data] 선호하는 향기가 존재하지 않을 경우 drop
buyer_dataset = buyer_dataset.dropna(axis=0)


# [dirty data] 빈 string 데이터 제거
buyer_dataset = buyer_dataset[buyer_dataset['preference_base_note'] != '']
buyer_dataset = buyer_dataset[buyer_dataset['preference_middle_note'] != '']


# [Data cleaning] brand와 name을 합쳐 하나의 feature로 통일

buyer_dataset['type'] = buyer_dataset['brand'] + '-' + buyer_dataset['name']
buyer_dataset = buyer_dataset.drop(['brand', 'name'], axis=1)

#SVD용 pivot table 제작
print('hihi', buyer_dataset.columns)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(buyer_dataset.head())
usr_pf_table = buyer_dataset.pivot_table(index='user_id', columns='type', values='satisfaction', aggfunc='mean').fillna(0)
pivot_table = usr_pf_table.values.T
pivot_true = buyer_dataset['satisfaction']

buyer_dataset = buyer_dataset.drop(['user_id'], axis=1)
buyer_dataset = buyer_dataset.reset_index(drop=True)

org_buyer_dataset = buyer_dataset
print(org_buyer_dataset.columns)

# 타겟 변수와 특성들을 분리
X = buyer_copy.drop('satisfaction', axis=1)
y = buyer_copy['satisfaction']

# 데이터를 훈련 세트와 테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train_copy = X_train.drop(['type'], axis=1)
# X_test_copy = X_test.drop(['type'], axis=1)

X_train_copy = X_train
X_test_copy = X_test

print(f"Rows: {X_train.shape[0]}\nColumns: {X_train.shape[1]}")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(X_train.head())

print()
print("X_train_copy")
print(f"Rows: {X_train_copy.shape[0]}\nColumns: {X_train_copy.shape[1]}")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(X_train_copy.head())

# SVD 모델 초기화 및 학습
svd = TruncatedSVD(n_components=20, random_state=42)
svd.fit(pivot_table)
print(pivot_table.shape)
latent_matrix = svd.transform(pivot_table)
print(latent_matrix.shape)
corr= np.corrcoef(latent_matrix)
print(corr)

# Heatmap으로 시각화
# plt.figure(figsize=(5, 5))
# sns.heatmap(corr, cmap='coolwarm', vmax=1.0, vmin=-1.0, linewidths=0.1, annot=False)
# plt.title('Correlation Heatmap')
# plt.show()

print(usr_pf_table.columns)

perfumes_titles = usr_pf_table.columns
perfumes_title_list = list(perfumes_titles)
print('type', perfumes_titles)
print('hi', pd.Index(perfumes_title_list))

#유저의 input data는 향수 이름
coffey_hands = perfumes_title_list.index('HUGO BOSS-Boss The Scent')
corr_coffey_hands = corr[coffey_hands]

#유저가 입력한 향수 이름을 기반으로 비슷한 노트들의 향수를 추천
result = list(perfumes_titles[(corr_coffey_hands >= 0.6)])[:50]
print(result)

#추천한 향수를 검증 및 출력하여 어느 note인지 유저에게 알려줌
for i, v in enumerate(result):
    rc_data = org_buyer_dataset[org_buyer_dataset['type'] == result[i]]
    print(rc_data)