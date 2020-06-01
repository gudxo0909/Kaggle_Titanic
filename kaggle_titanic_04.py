# import module

import pandas as pd
import numpy as np

# 시각화를 위한 seaborn module

import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.shape)
print(test.shape)

train.dtypes
train.head()
train.tail()

test.head()
test.tail()

# 문자형 데이터인 성별 데이터 숫자형으로 변환

train['Sex_encode'] = train['Sex'].replace('male' , 0).replace('female' , 1)
train['Sex_encode']

test['Sex_encode'] = test['Sex'].replace('male' , 0).replace('female' , 1)
test['Sex_encode']

train['Embarked_encode'] = train['Embarked'].replace('S' , 0).replace('C' , 1).replace('Q' , 2)
train['Embarked_encode']

test['Embarked_encode'] = test['Embarked'].replace('S' , 0).replace('C' , 1).replace('Q' , 2)
test['Embarked_encode']

# 상관관계 분석

train.corr()

# 상관관계가 10% 이상인 열만 리스트화

numeric = ['Pclass' , 'Fare' , 'Sex_encode' , 'Embarked_encode']
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

# import machine learning module (cross_validation_score)

from sklearn.model_selection import cross_val_score
score = cross_val_score(model, x_train, y_train, scoring = 'accuracy' , cv = 10).mean()
# x_train, y_train 데이터를 사용해서 10번의 cross validation 진행 후 n개의 결과의 평균을 스코어로 예측

# import machine learning module (decisiontreeclassifier)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth = 10, max_features = 0.9)
# 최대 뿌리 10개, 전체 특성의 90%만 사용
model

from sklearn.model_selection import cross_val_score
score = cross_val_score(model, x_train, y_train, scoring = 'accuracy', cv = 10).mean()
score

model.fit(x_train, y_train)

y_test = model.predict(x_test)

submit = pd.read_csv('gender_submission.csv')
y_test = submit['Survived']
submit.to_csv('result_04.csv', index = False)