# import module

import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
print(train.shape)
train.head()
train.tail()

test = pd.read_csv('test.csv')
print(test.shape)
test.head()
test.tail()

# 시각화를 위한 seaborn module

import seaborn as sns

sns.countplot(data = train, x = 'Sex')
sns.countplot(data = train, x = 'Sex', hue = 'Survived')

sns.countplot(data = train, x = 'Embarked')
sns.countplot(data = train, x = 'Embarked', hue = 'Survived')

# 상관관계 분석

train.corr()

train['Sex_encode'] = train['Sex'].replace('male' , 0).replace('female' , 1)
train['Sex_encode']

test['Sex_encode'] = test['Sex'].replace('male' , 0).replace('female' , 1)
test['Sex_encode']

train['Embarked_encode'] = train['Embarked'].replace('S' , 0).replace('C' , 1).replace('Q' , 2)
train['Embarked_encode']

test['Embarked_encode'] = test['Embarked'].replace('S' , 0).replace('C' , 1).replace('Q' , 2)
test['Embarked_encode']

train.corr()

# 절대값 기준 0 ~ 0.1에 해당하는 PassengerId, Age, SibSp, Parch는 생존자 예측에 도움이 안되므로 제거

numeric = ['Pclass', 'Fare', 'Sex_encode', 'Embarked_encode']
numeric

# import machine learning module

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model

x_train = train[numeric]
x_test = test[numeric]

y_train = train['Survived']

# 결측값 처리

x_train['Embarked_encode'] = x_train['Embarked_encode'].fillna(0)
x_test['Fare'] = x_test['Fare'].fillna(x_test['Fare'].mean())

model.fit(x_train, y_train)

y_test = model.predict(x_test)

submit = pd.read_csv('gender_submission.csv')
submit['Survived'] = y_test
submit.to_csv('result_03.csv' , index = False)