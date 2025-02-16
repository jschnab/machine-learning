import numpy as np
import pandas as pd
import re as re

#import data
train = pd.read_csv("train.csv", header=0, dtype={'Age': np.float64})
test = pd.read_csv("test.csv", header=0, dtype={'Age': np.float64})
full_data = [train, test]
print(train.info())

#check the effect of class and sex on survival rates
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean(), '\n')
print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean(), '\n')

#group two features in one called family size and check effect on survival
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], \
        as_index=False).mean(),'\n')
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print(train[['IsAlone', 'Survived']].groupby(['IsAlone'], \
        as_index=False).mean())


#fill missing values with median of data and check effect on survival
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
print(train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], \
        as_index=False).mean())

#fill missing values with one of the values and check effect on survival
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
print(train[['Embarked', 'Survived']].groupby(['Embarked'], \
        as_index=False).mean(), '\n')

#fill missing data in age with random values around mean
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, \
            size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

#group Age values in categories
train['CategoricalAge'] = pd.cut(train['Age'], 5)
print(train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], \
        as_index=False).mean(), '\n')

#function to get the title from Name feature
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

#get titles from Name feature
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

#show Title and Sex association
print(pd.crosstab(train['Title'], train['Sex']), '\n')

#replace titles values and check effect on survival
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',\
            'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean(), '\n')

#data cleaning and mapping of strings to numerical values
for dataset in full_data:

    #mapping sex
    dataset['Sex'] = dataset['Sex'].map({'female':0, 'male':1}).astype(int)

    #mapping titles
    title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    #mapping embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

    #mapping fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2

    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    #mapping age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

#feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', \
        'FamilySize']
train = train.drop(drop_elements, axis=1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)

test = test.drop(drop_elements, axis=1)

print(train.head())

#save feature_engineered data as a csv file
np.savetxt("train_feat_engin.csv", train, delimiter=",", fmt="%d")
np.savetxt("test_feat_engin.csv", test, delimiter=",", fmt="%d")

