''' 1. Grid Search (설정하고자 하는 파라미터 값을 임의의 값들로 묶어 모든 경우의 수로 조합하는 방법)
    - 장점 : 이미 값이 설정되어 있는 상태에서 조합하는 것이므로 빠르게 결과를 도출 할 수 있음.
    - 단점 : 이미 값을 설정하고 조합하기 때문에 더 좋은 파라미터 값을 조합하지 못한다.

    2. Random Search (설정하고자 하는 파라미터 값을 각각 일정 범위로 설정해 무작위로 조합하는 방법)
    - 장점: 파라미터 값을 범위로 설정했기 때문에 Grid Search보다 더 좋은 파라미터 값을 도출 할 수 있다.
    - 단점: 조합해야 하는 경우의 수가 많아져 결과 도출에 상당한 시간이 걸림.
    '''

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

# Grid Search Algorithm module

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
parameters = {
    'max_depth' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'max_features' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}
GS = GridSearchCV(DecisionTreeClassifier(), param_grid = parameters, cv = 10, scoring = 'accuracy')
GS.fit(x_train, y_train)

# 최적의 파라미터 값 출력

GS.best_params_ # max_depth : 8 , max_features : 1.0 일 때 최적

# Cross Validation으로 확인

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth = 8, max_features = 1.0)
model

from sklearn.model_selection import cross_val_score
score = cross_val_score(model, x_train, y_train, scoring = 'accuracy', cv = 10).mean()
score

# Random Search

from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
parameters = {
    'max_depth' : range(1,10),
    'max_features' : stats.uniform(0.1, 1.0)
}
RS = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions = parameters, cv = 10, scoring = 'accuracy', n_iter = 50)
RS.fit(x_train, y_train)

# 최적의 파라미터 값 출력

RS.best_params_ # max_depth : 9 , max_features : 1.0 일 때 최적

# Cross Validation으로 확인

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth = 9, max_features = 1.0)
model

from sklearn.model_selection import cross_val_score
score = cross_val_score(model, x_train, y_train, scoring = 'accuracy', cv = 10).mean()
score # Grid Search 보다 조금 더 스코어가 오름.

model.fit(x_train, y_train)

prediction = model.predict(x_test)
print(prediction.shape)
prediction[:5]

submit = pd.read_csv('gender_submission.csv')
prediction = submit['Survived']
submit.to_csv('final_result.csv', index = False)