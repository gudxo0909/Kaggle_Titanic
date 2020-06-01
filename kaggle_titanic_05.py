# import module

import pandas as pd
import numpy as np
import seaborn as sns

train = pd.read_csv('train.csv')
print(train.shape)
train.head()
train.tail()

test = pd.read_csv('test.csv')
print(test.shape)
test.head()
test.tail()

train['Sex'] = train['Sex'].replace('male' , 0).replace('female' , 1)
test['Sex'] = test['Sex'].replace('male' , 0).replace('female' , 1)

# 선착장 데이터 처리 (S , Q , C 선착장을 2개의 값으로)

train['Embarked_Q'] = train['Embarked'] == 'Q'
train['Embarked_S'] = train['Embarked'] == 'S'
train['Embarked_C'] = train['Embarked'] == 'C'

train[['Embarked', 'Embarked_Q' , 'Embarked_S' , 'Embarked_C']].head()

test['Embarked_Q'] = test['Embarked'] == 'Q'
test['Embarked_S'] = test['Embarked'] == 'S'
test['Embarked_C'] = test['Embarked'] == 'C'

test[['Embarked', 'Embarked_Q' , 'Embarked_S' , 'Embarked_C']].head()

# SibSp , Parch 데이터 처리 (Family라는 변수 추가)

train['Family'] = train['SibSp'] + train['Parch'] + 1 # 가족 구성원 = 형제/자매 및 배우자 수 + 동승한 부모 및 자녀의 수 + 1 (자신)
test['Family'] = test['SibSp'] + test['Parch'] + 1

sns.countplot(data = train, x = 'Family', hue = 'Survived')

# 가족 구성원의 범위 지정

def Family_type(Family):
    '''가족 구성원이 1명이면 single
        2~4명이면 nuclear
        5명 이상이면 big'''
    if Family == 1 :
        return 'Single'
    elif Family > 1 and Family < 5 :
        return 'Nuclear'
    else :
        return 'Big'

train['Family_type'] = train['Family'].apply(Family_type)
train[['Family' , 'Family_type']].head()

test['Family_type'] = test['Family'].apply(Family_type)
test[['Family' , 'Family_type']].head()

# Family_type 변수도 Embarked 변수와 같이 처리

train['Single'] = train['Family_type'] == 'Single'
train['Nuclear'] = train['Family_type'] == 'Nuclear'
train['Big'] = train['Family_type'] == 'Big'

train[['Family_type' , 'Single' , 'Big']].head()

test['Single'] = test['Family_type'] == 'Single'
test['Nuclear'] = test['Family_type'] == 'Nuclear'
test['Big'] = test['Family_type'] == 'Big'

test[['Family_type' , 'Single' , 'Big']].head()

# 목표 변수인 Survived와 상관관계 분석

corr = train.corr()
corr['Survived']

# 결측값 처리

test['Fare'] = test['Fare'].fillna(0)

features = ['Pclass' , 'Sex' , 'Fare' , 'Embarked_Q' , 'Embarked_S' , 'Embarked_C' , 'Single' , 'Nuclear' , 'Big']
x_train = train[features]
x_test = test[features]
y_train = train['Survived']

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth = 10, max_features = 0.9)
model

from sklearn.model_selection import cross_val_score
score = cross_val_score(model, x_train, y_train, scoring = 'accuracy' , cv = 10).mean()
score

model.fit(x_train, y_train)

prediction = model.predict(x_test)
print(prediction.shape)
prediction[:5]

submit = pd.read_csv('gender_submission.csv')
prediction = submit['Survived']
submit.to_csv('result_05.csv', index = False)