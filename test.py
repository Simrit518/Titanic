s = dataset['Fare']
for i in range(len(dataset)):
    # if s[i] < 7.910400:
    #     dataset.ix[i, 'Fare2'] = 0
    # elif s[i] < 14.454200:
    #     dataset.ix[i, 'Fare2'] = 1
    # elif s[i] < 31.000000:
    #     dataset.ix[i, 'Fare2'] = 2
    # else:
    #     dataset.ix[i, 'Fare2'] = 3
    if dataset.ix[i, 'Parch'] > 2:
        dataset.ix[i, 'Parch'] = 2
    if dataset.ix[i, 'SibSp'] > 3:
        dataset.ix[i, 'SibSp'] = 3
# 删除无用变量
del dataset['Age']
# del dataset['Fare']
del dataset['Cabin']
del dataset['Name']
del dataset['Ticket']
del dataset['Embarked']
del dataset['Sex']

def KNN(x_train, y_train, x_test):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)
    y_predict=neigh.predict(x_test)
    return y_predict
def DecisionTree(x_train, y_train, x_test):
    clf=DecisionTreeClassifier()
    clf.fit(x_train,y_train)
    y_predict=clf.predict(x_test)
    return y_predict