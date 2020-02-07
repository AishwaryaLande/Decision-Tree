import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
fraud=pd.read_csv("C:\\Users\\ADMIN\\Desktop\\Data_Science_Assig\\Decision Tree\\Fraud_check.csv")
fraud.describe()
fraud.columns = ['Undergrad', 'Marital_Status', 'taxable_income', 'City_Population','Work_Experience', 'Urban']

plt.hist(fraud['taxable_income'])
plt.boxplot(fraud['taxable_income'])

fraud.drop_duplicates(keep='first',inplace=True)
fraud.drop(["City_Population","Work_Experience"],axis=1,inplace=True)

fraud['taxable_income'] = np.where(fraud['taxable_income'] <= 30000 , "Risky","Good")
fraud['taxable_income'].value_counts()

from sklearn import preprocessing
prepocess = preprocessing.LabelEncoder()
columns = ["Undergrad","Marital_Status","Urban"];

for i in columns:
    fraud[i] = prepocess.fit_transform(fraud[i])
    
fraud.columns
fraud = fraud[['Undergrad', 'Marital_Status', 'City_Population','Work_Experience', 'Urban','taxable_income']]

fraud.head()
fraud.taxable_income.value_counts()

from sklearn.model_selection import train_test_split
train,test = train_test_split(fraud,test_size=0.2)
trainX = train.iloc[:,0:5]
trainY = train.iloc[:,5]
testX = test.iloc[:,0:5]
testY = test.iloc[:,5]

colnames = list(fraud.columns)
predictors = colnames[:5]
target = colnames[5]
#start, stop, size = fraud
#np.random.uniform(start,stop,size) 
#with size = size

#DT algorithm
from sklearn.tree import DecisionTreeClassifier as DT
help(DecisionTreeClassifier)

model = DT(criterion = 'entropy')
model.fit(train[predictors],train[target])
preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)
accuracy_model=np.mean(train.taxable_income == model.predict(train[predictors]))
accuracy_model
accuracy=np.mean(preds==test.taxable_income)
accuracy
#accuracy train
#np.mean(train.taxable_income == model.predict(train[predictors]))
#np.mean(test.taxabel_income == model.predict(test[predictors]))
#np.mean(preds==test.taxable_income) 

