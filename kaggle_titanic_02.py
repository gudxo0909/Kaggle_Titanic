# import module

import pandas as pd
import numpy as np

# 데이터 불러오기

train = pd.read_csv('C:/Users/schwe/git/python/python practise/kaggle_titanic/train.csv')
print(train.shape)
train.head()
train.tail()

test = pd.read_csv('C:/Users/schwe/git/python/python practise/kaggle_titanic/test.csv')
print(test.shape)
test.head()
test.tail()

# 데이터 타입 확인

train.dtypes
test.dtypes

train['Sex'].unique()
test['Sex'].unique()

train['Sex'].value_counts()
test['Sex'].value_counts()

train['Sex_encode'] = train['Sex'].replace('male', 0).replace('female', 1)
train['Sex_encode']
train['Sex_encode'].value_counts()

test['Sex_encode'] = test['Sex'].replace('male', 0).replace('female', 1)
test['Sex_encode']
test['Sex_encode'].value_counts()

train['Embarked'].value_counts()
test['Embarked'].value_counts()

train['Embarked_encode'] = train['Embarked'].replace('S', 0).replace('C', 1).replace('Q', 2)
test['Embarked_encode'] = test['Embarked'].replace('S', 0).replace('C', 1).replace('Q', 2)

train['Embarked_encode']
train['Embarked_encode'].value_counts()

test['Embarked_encode']
test['Embarked_encode'].value_counts()

train['Name'].unique()
train['Ticket'].unique()
train['Cabin'].unique()

numeric = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_encode', 'Embarked_encode']
numeric

x_train = train[numeric]
x_test = test[numeric]

y_train = train['Survived']

x_train.isnull().sum()
y_train.isnull().sum()
x_test.isnull().sum()

# 결측값 처리

x_train['Age'] = x_train['Age'].fillna(x_train['Age'].mean())
x_train['Embarked_encode'] = x_train['Embarked_encode'].fillna(0)

x_test['Age'] = x_test['Age'].fillna(x_test['Age'].mean())
x_test['Fare'] = x_test['Fare'].fillna(x_test['Fare'].mean())

# import machine learning module

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model

model.fit(x_train , y_train)

y_test = model.predict(x_test)

sumit = pd.read_csv('C:/Users/schwe/git/python/python practise/kaggle_titanic/gender_submission.csv')
sumit['Survived'] = y_test
sumit.to_csv('result_02.csv' , index = False)