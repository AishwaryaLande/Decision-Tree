import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
import seaborn as sns

company_Data = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Decision Tree/Company_Data.csv")
company_Data.drop_duplicates(keep='first',inplace=True)
company_Data.isnull().sum()
company_Data.Sales.describe()
company_Data.describe()
company_Data.shape

plt.hist(company_Data['Sales'])
plt.boxplot(company_Data['Sales'])

company_Data = company_Data.drop(company_Data.index[[376]])
company_Data = company_Data.drop(company_Data.index[[316]])

plt.hist(company_Data['Sales'])
plt.boxplot(company_Data['Sales'])

company_Data.columns

pd.set_option('display.expand_frame_repr', False)
company_Data['Sales'] = nm.where(company_Data['Sales'] > 7.4 ,'High','Low')
company_Data['Sales'].value_counts()

from sklearn import preprocessing
prepocess = preprocessing.LabelEncoder()
columns = ["ShelveLoc","Urban","US"];

for i in columns:
    company_Data[i] = prepocess.fit_transform(company_Data[i])
    
sns.boxplot(company_Data.CompPrice)
sns.boxplot(company_Data.Advertising)
sns.boxplot(company_Data.Population)
sns.boxplot(company_Data.Price)
sns.boxplot(company_Data.ShelveLoc)
sns.boxplot(company_Data.Education)
sns.boxplot(company_Data.Age)
sns.boxplot(company_Data.Urban)
sns.boxplot(company_Data.US)

sns.pairplot((company_Data),hue='Sales')

company_Data.head()
company_Data.info

corr = company_Data.corr()
corr
sns.heatmap(corr, annot=True)

from sklearn.model_selection import train_test_split
train,test = train_test_split(company_Data,test_size=0.3)
trainX = train.iloc[:,1:]
trainY = train.iloc[:,0]
testX = test.iloc[:,1:]
testY = test.iloc[:,0]

from sklearn.tree import DecisionTreeClassifier as DT
model1 = DT(criterion="entropy").fit(trainX,trainY)
model_pred = model1.predict(trainX)
accuracy = nm.mean(model_pred == trainY)
accuracy
model_test_pred = model1.predict(testX)
accuracy_test = nm.mean(model_test_pred == testY)
accuracy_test

company_Data['is_train'] = nm.random.uniform(0, 1, len(company_Data))<= 0.75
company_Data['is_train']

train2,test2 = company_Data[company_Data['is_train'] == True],company_Data[company_Data['is_train']==False]
trainX2 = train2.iloc[:,1:]
trainY2 = train2.iloc[:,0]
testX2 = test2.iloc[:,1:]
testY2 = test2.iloc[:,0]

model2 = DT(criterion="entropy").fit(trainX2,trainY2)
model_pred_2 = model2.predict(trainX2)
accuracy2 = nm.mean(model_pred_2 == trainY2)
accuracy2
model_test_pred2 = model2.predict(testX2)
accuracy_test2 = nm.mean(model_test_pred2 == testY2)
accuracy_test2

model3 = DT().fit(trainX,trainY)
model_pred3 = model3.predict(trainX)
accuracy = nm.mean(model_pred3 == trainY)
accuracy
model_test_pred3 = model3.predict(testX)
accuracy_test3 = nm.mean(model_test_pred3 == testY)
accuracy_test3

train4,test4 = train_test_split(company_Data,test_size=0.2)
trainX4 = train4.iloc[:,1:]
trainY4 = train4.iloc[:,0]
testX4 = test4.iloc[:,1:]
testY4 = test4.iloc[:,0]

model4 = DT(criterion="entropy").fit(trainX4,trainY4)
model_pred_4 = model4.predict(trainX4)
accuracy4 = nm.mean(model_pred_4 == trainY4)
accuracy4
model_test_pred4 = model4.predict(testX4)
accuracy_test4 = nm.mean(model_test_pred4 == testY4)
accuracy_test4
