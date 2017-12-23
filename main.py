import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import re


#SVM分类器
def SVM(x_train, y_train, x_test):
    clf=SVC()
    clf.fit(x_train,y_train)
    y_predict=clf.predict(x_test)
    return y_predict
#随机森林分类器
def RandomForest(x_train, y_train, x_test):
    clf=RandomForestClassifier(max_depth=2,random_state=0)
    clf.fit(x_train,y_train)
    y_predict=clf.predict(x_test)
    return y_predict
#神经网络
def NeuralNetwork(x_train, y_train, x_test):
    clf = MLPClassifier(solver='adam', alpha=1e-5,
                        hidden_layer_sizes=(50,), random_state=1)
    clf.fit(x_train, y_train)
    y_predict=clf.predict(x_test)
    return y_predict



#分类器预测结果投票判决
def Vote(x_train, y_train, x_test):
    Spredict=SVM(x_train, y_train, x_test)
    Rpredict=RandomForest(x_train, y_train, x_test)
    Npredict=NeuralNetwork(x_train, y_train, x_test)
    y_predict=[]
    for i in range(len(Spredict)):
        zero = 0;one = 0
        if Spredict[i]==0:
            zero+=1
        else:
            one+=1
        if Rpredict[i]==0:
            zero+=1
        else:
            one+=1
        if Npredict[i]==0:
            zero+=1
        else:
            one+=1
        if zero>one:
            y_predict.append(0)
        else:
            y_predict.append(1)
    return y_predict
    # label = []
    # for j in y_test:
    #     label.append(j)
    # r = 0
    # for i in range(len(label)):
    #     if y_predict[i] == label[i]:
    #         r += 1
    # right = r / len(y_predict)
    # return right

#交叉验证
# def Cross_validation(x,y):
#     x_train, x_test, y_train, y_test = train_test_split(
#         x, y, test_size=0.1, random_state=0,shuffle=True)
#     print(vote(x_train, x_test, y_train, y_test))

#数据处理
def Dataprocess(file):
    dataset = pd.read_csv(file)
    dataset['Age'] .fillna(dataset['Age'].dropna().median(),inplace=True)
    dataset['Fare'].fillna(dataset['Fare'].dropna().median(),inplace=True)
    Name = dataset['Name']
    SibSp = dataset['SibSp']
    Parch = dataset['Parch']
    dataset['Isalone']=1
    for i in range(len(dataset)):
        if re.match('.*Mr', Name[i]):
            dataset.ix[i, 'Name'] = 1
        elif re.match('.*Miss', Name[i]):
            dataset.ix[i, 'Name'] = 2
        elif re.match('.*Mrs', Name[i]):
            dataset.ix[i, 'Name'] = 3
        elif re.match('.*Master', Name[i]):
            dataset.ix[i, 'Name'] = 4
        elif re.match('.*Lady|.*Countess|.*Capt|.*Col|.*Don|.*Dr|.*Major|.*Rev|.*Sir|.*Jonkheer|.*Dona', Name[i]):
            dataset.ix[i, 'Name'] = 5
        else:
            dataset.ix[i, 'Name'] = 0
        #处理Fare
        if dataset.ix[i, 'Fare'] < 7.91:
            dataset.ix[i, 'Fare'] = 0
        elif dataset.ix[i, 'Fare'] >= 7.91 and dataset.ix[i, 'Fare'] < 14.454:
            dataset.ix[i, 'Fare'] = 1
        elif dataset.ix[i, 'Fare'] >= 14.454 and dataset.ix[i, 'Fare'] < 31:
            dataset.ix[i, 'Fare'] = 2
        else:
            dataset.ix[i, 'Fare'] = 3
        #处理Parch
        if dataset.ix[i, 'Parch'] > 2:
            dataset.ix[i, 'Parch'] = 2
        # 处理SibSp
        if dataset.ix[i, 'SibSp'] > 3:
            dataset.ix[i, 'SibSp'] = 3
        # 处理Age
        if dataset.ix[i,'Age']<=16:
            dataset.ix[i, 'Age']=0
        elif dataset.ix[i,'Age']<=32:
            dataset.ix[i, 'Age']=1
        elif dataset.ix[i,'Age']<=48:
            dataset.ix[i, 'Age']=2
        elif dataset.ix[i,'Age']<=64:
            dataset.ix[i, 'Age']=3
        else:
            dataset.ix[i, 'Age']=4
        if SibSp[i] + Parch[i] == 0:
            dataset.ix[i, 'Isalone'] = 0

    #处理Embarked
    freq_port=dataset.Embarked.dropna().mode()[0]
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)  # 删除空值用S代替
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    #将Sex映射为0,1
    dataset['Sex']=dataset['Sex'].map({'male':0,'female':1}).astype(int)
    #删除无用信息
    del dataset['PassengerId']
    del dataset['Ticket']
    del dataset['Cabin']
    del dataset['SibSp']
    del dataset['Parch']
    return dataset
#生成预测结果的csv文件
def GenerateFile(pid,y_predict):
    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'PassengerId': pid, 'Survived': y_predict})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("gender_submission.csv", index=False, sep=',', header=True)

if __name__ == '__main__':
    train=Dataprocess('train.csv')
    #print(train)
    test=Dataprocess('test.csv')
    pid=pd.read_csv('test.csv')['PassengerId']
    x_train = train[['Pclass','Name', 'Sex','Age', 'Fare','Embarked']]
    y_train = train['Survived']
    x_test=test[['Pclass','Name', 'Sex','Age', 'Fare','Embarked']]
    y_predict=Vote(x_train,y_train,x_test)
    GenerateFile(pid,y_predict)




