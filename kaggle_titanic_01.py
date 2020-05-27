# import module

import pandas as pd
import numpy as np

# train 데이터 확인

train = pd.read_csv('C:/Users/schwe/git/python/python practise/kaggle_titanic/train.csv')
print(train.shape)
train.head()
train.tail()
train.dtypes

'''sibsp -> 동승한 형제/자매/배우자
   parch -> 동승한 부모/자녀
   embarked -> 탑승한 선착장'''

# test 데이터 확인

test = pd.read_csv('C:/Users/schwe/git/python/python practise/kaggle_titanic/test.csv')
print(test.shape)
test.head()
test.tail()
test.dtypes

# 실수형과 정수형을 numeric이라는 변수에 리스트 형태로 입력

numeric = ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
numeric
train[numeric]
test[numeric] # Survived라는 값이 존재하지 않기 때문에 numeric을 적용할 수 x

# Survived 값을 제거한 후 numeric 변수 정의

numeric = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
numeric
train[numeric]
test[numeric]

# import Machine Learning Algorithm Module

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

x_train = train[numeric]
x_test = test[numeric]

y_train = train['Survived']

model.fit(x_train, y_train) # NaN값이 포함되어 있어, 학습 시킬 수 x

x_train.isnull().sum()
y_train.isnull().sum()
x_test.isnull().sum()

# fillna를 이용해 결측값을 평균값으로 대체

x_train['Age'] = x_train['Age'].fillna(x_train['Age'].mean())
x_train.isnull().sum()

x_test['Age'] = x_test['Age'].fillna(x_test['Age'].mean())
x_test.isnull().sum()
x_test['Fare'] = x_test['Fare'].fillna(x_test['Fare'].mean())
x_test.isnull().sum()

model.fit(x_train, y_train) # 모델 적용
model.predict(x_test) # 모델 예측

y_test = model.predict(x_test)

submit = pd.read_csv('C:/Users/schwe/git/python/python practise/kaggle_titanic/gender_submission.csv')
submit.to_csv('result_01.csv', index = False)