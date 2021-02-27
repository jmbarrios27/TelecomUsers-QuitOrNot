#importing libraries
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import joblib
plt.style.use('ggplot')
sns.set(style="whitegrid")
warnings.filterwarnings('ignore')

#reading data
telecom = pd.read_csv('C:\\Users\\Asus\\Desktop\\telecom\\data\\telecom_users.csv')
telecom.head()

#Data Sctucture check
def data_check(df):
    print('dataframe shape:', df.shape)
    print()
    print(df.describe())
    print()
    print('Dataframe "NaN" values:',sum(df.isna().sum()))
    print()
    print(telecom.info())
    

#Predicted Label Change
def checkFunc(value):
    if value == True:
        return 'Corrrectly Predicted'
    else:
        return 'Not Correctly Predicted'



#Calling data check function
data_check(telecom)

#Transforming Total charges column into integer
telecom['TotalCharges'] = telecom['TotalCharges'].convert_objects(convert_numeric=True)

#Replacing Empty values with mean
telecom.fillna(telecom.mean(),inplace=True)


telecom.describe()

# a summay of categorical variables
telecom.describe(include=[np.object])

#Creating a new dataframe to work on
data = telecom
data = data.drop(columns=['Unnamed: 0','customerID'])

#Target Variable
target_color = ['red','green']
sns.countplot(data=data, x='Churn',palette=target_color)
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()

print()
print('Umbalanced Target Class')

#Target Variable
sns.countplot(data=data, x='Churn',palette='magma',hue='SeniorCitizen')
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()

#Target Variable
sns.countplot(data=data, x='Churn',palette='viridis',hue='gender')
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()


#Target Variable
sns.countplot(data=data, x='Churn',palette='rainbow',hue='Partner')
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()

#Target Variable
sns.countplot(data=data, x='Churn',palette='Set2',hue='Dependents')
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()

#Target Variable
sns.countplot(data=data, x='Churn',palette='Set1',hue='PhoneService')
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()

#Target Variable
sns.countplot(data=data, x='Churn',palette='Pastel1',hue='MultipleLines')
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()


#Target Variable
sns.countplot(data=data, x='Churn',palette='magma',hue='InternetService')
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()


#Target Variable
sns.countplot(data=data, x='Churn',palette='Set1',hue='OnlineSecurity')
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()

#Target Variable
sns.countplot(data=data, x='Churn',palette='Set2',hue='OnlineBackup')
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()

#Target Variable
sns.countplot(data=data, x='Churn',palette='Set3',hue='DeviceProtection')
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()

#Target Variable
sns.countplot(data=data, x='Churn',palette='viridis',hue='TechSupport')
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()

#Target Variable
sns.countplot(data=data, x='Churn',palette='magma',hue='StreamingTV')
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()


#Target Variable
sns.countplot(data=data, x='Churn',palette='Set1',hue='Contract')
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()

#Target Variable
sns.countplot(data=data, x='Churn',palette='Set2',hue='PaperlessBilling')
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()

#Target Variable
sns.countplot(data=data, x='Churn',palette='rainbow',hue='PaymentMethod')
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()

#Distribution of months 
plt.figure(figsize=(10,10))
sns.distplot(data['tenure'],color='blue')
plt.title('DISTRIBUTION OF MONTHS REGISTERED')
plt.show()


#Gender Count
gender = data['gender'].value_counts()
color = ['skyblue','pink']
plt.figure(figsize=(8,6))
plt.pie(gender,labels=gender.index,autopct='%1.2f%%',explode=(0,0.1),colors=color)
plt.title('GENDER DISTRUBITION FOR CUSTOMERS')
plt.axis('equal')
plt.show()


#Gender Count
dependents = data['Dependents'].value_counts()
color = ['lightgreen','darkred']
plt.figure(figsize=(8,6))
plt.pie(dependents,labels=dependents.index,autopct='%1.2f%%',explode=(0,0.1),colors=color)
plt.title('¿CUSTOMERS HAVE DEPENDENTS?')
plt.axis('equal')
plt.show()



#phone Count
phone = data['PhoneService'].value_counts()
color = ['lightgreen','red']
plt.figure(figsize=(8,6))
plt.pie(phone,labels=phone.index,autopct='%1.2f%%',explode=(0,0.1),colors=color)
plt.title('¿CUSTOMERS HAVE PHONE SERVICE?')
plt.axis('equal')
plt.show()

#phone Count
multi = data['MultipleLines'].value_counts()
color = ['darkred','lightgreen','red']
plt.figure(figsize=(8,6))
plt.pie(multi,labels=multi.index,autopct='%1.2f%%',explode=(0.01,0.01,0.01),colors=color)
plt.title('¿CUSTOMERS HAVE MULTIPLE PHONE SERVICE')
plt.axis('equal')
plt.show()

#internet Count
internet = data['InternetService'].value_counts()
color = ['lightgreen','green','darkred']
plt.figure(figsize=(8,6))
plt.pie(internet,labels=internet.index,autopct='%1.2f%%',explode=(0,0.01,0.01),colors=color)
plt.title('¿CUSTOMERS HAVE INTERNET SERVICE?')
plt.axis('equal')
plt.show()

#osecurity Count
osecurity = data['OnlineSecurity'].value_counts()
color = ['darkred','lightgreen','red']
plt.figure(figsize=(8,6))
plt.pie(osecurity,labels=osecurity.index,autopct='%1.2f%%',explode=(0,0.01,0.01),colors=color)
plt.title('¿CUSTOMERS HAVE ONLINE SECURITY SERVICE?')
plt.axis('equal')
plt.show()

#onlinebackup Count
obackup = data['OnlineBackup'].value_counts()
color = ['red','lightgreen','darkred']
plt.figure(figsize=(8,6))
plt.pie(obackup,labels=obackup.index,autopct='%1.2f%%',explode=(0,0.01,0.01),colors=color)
plt.title('¿CUSTOMERS HAVE ONLINE BACKUP SERVICE?')
plt.axis('equal')
plt.show()

#deviceprotect Count
deviceprotect = data['DeviceProtection'].value_counts()
color = ['red','lightgreen','darkred']
plt.figure(figsize=(8,6))
plt.pie(deviceprotect,labels=deviceprotect.index,autopct='%1.2f%%',explode=(0,0.01,0.01),colors=color)
plt.title('¿CUSTOMERS HAVE DEVICE PROTECTION SERVICE?')
plt.axis('equal')
plt.show()

#TechSupport Count
techSupport = data['TechSupport'].value_counts()
color = ['red','lightgreen','darkred']
plt.figure(figsize=(8,6))
plt.pie(techSupport,labels=techSupport.index,autopct='%1.2f%%',explode=(0,0.01,0.01),colors=color)
plt.title('¿CUSTOMERS HAVE TECH SUPPORT SERVICE?')
plt.axis('equal')
plt.show()


#StreamingTV Count
streamingTV = data['StreamingTV'].value_counts()
color = ['red','green','darkred']
plt.figure(figsize=(8,6))
plt.pie(streamingTV ,labels=streamingTV .index,autopct='%1.2f%%',explode=(0,0.01,0.01),colors=color)
plt.title('¿CUSTOMERS HAVE STREAMING TV SERVICE?')
plt.axis('equal')
plt.show()


#StreamingMovies Count
streamingMovies = data['StreamingMovies'].value_counts()
color = ['red','green','darkred']
plt.figure(figsize=(8,6))
plt.pie(streamingMovies,labels=streamingMovies.index,autopct='%1.2f%%',explode=(0,0.01,0.01),colors=color)
plt.title('¿CUSTOMERS HAVE STREAMING MOVIES SERVICE?')
plt.axis('equal')
plt.show()

#Contract Count
contract = data['Contract'].value_counts()
color = ['lightblue','orange','pink']
plt.figure(figsize=(8,6))
plt.pie(contract,labels=contract.index,autopct='%1.2f%%',explode=(0,0.01,0.01),colors=color)
plt.title('¿CONTRACT TYPE?')
plt.axis('equal')
plt.show()

#PaperlessBilling Count
paperlessBilling = data['PaperlessBilling'].value_counts()
color = ['lightgreen','darkred']
plt.figure(figsize=(8,6))
plt.pie(paperlessBilling,labels=paperlessBilling.index,autopct='%1.2f%%',explode=(0.01,0.01),colors=color)
plt.title('PAPERLESS BILLING')
plt.axis('equal')
plt.show()

#PaymentMethod Count
paymentMethod = data['PaymentMethod'].value_counts()
color = ['lightyellow','lightgreen','pink','lightblue']
plt.figure(figsize=(8,6))
plt.pie(paymentMethod,labels=paymentMethod.index,autopct='%1.2f%%',explode=(0,0.01,0.01,0.01),colors=color)
plt.title('PAYMENT METHOD')
plt.axis('equal')
plt.show()

#distplot for all customers
sns.distplot(data['MonthlyCharges'],color='black')
plt.title('MONTHLY CHARGES DISTRIBUTION FOR ALL CUSTOMERES IN $')
plt.show()

#Creating dataframe for customers who renew contract or no.
nochurn_month = data[data['Churn']=='No']
churn_month = data[data['Churn']!='No']

# Customeres not renewed
sns.distplot(nochurn_month['MonthlyCharges'],color='darkred')
plt.title('MONTHLY CHARGES DISTRIBUTION FOR CUSTOMERS THAT NOT RENEWED THEIR CONTRACT IN $')
plt.show()

#Customers that renewed
sns.distplot(churn_month['MonthlyCharges'],color='blue')
plt.title('MONTHLY CHARGES DISTRIBUTION FOR CUSTOMERS THAT RENEWED THEIR CONTRACT IN $')
plt.show()

#distplot for all customers
sns.distplot(data['TotalCharges'],color='black')
plt.title('TOTAL CHARGES DISTRIBUTION FOR ALL CUSTOMERES IN $')
plt.show()

# Customeres not renewed
sns.distplot(nochurn_month['TotalCharges'],color='darkred')
plt.title('TOTAL CHARGES DISTRIBUTION FOR CUSTOMERS THAT NOT RENEWED THEIR CONTRACT IN $')
plt.show()

#Customers that renewed
sns.distplot(churn_month['TotalCharges'],color='blue')
plt.title('TOTAL CHARGES DISTRIBUTION FOR CUSTOMERS THAT RENEWED THEIR CONTRACT IN $')
plt.show()

#relation between 
plt.figure(figsize=(10,8))
sns.scatterplot(data=data, x="MonthlyCharges", y="TotalCharges",hue='Churn')
plt.title('RELATION BETWEEN MONTHLY CHARGES AND TOTAL CHARGES')
plt.show()

#Splitting Target Variable from dataframe
y = data['Churn']
X = data.drop(columns=['Churn'])

#Creating Dummy Variables for categorical data
X = pd.get_dummies(X)

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh',color='darkred')
plt.title('FEATURE IMPORTANCE')
plt.show()

#Target Variable
target_color = ['red','green']
sns.countplot(data=data, x='Churn',palette=target_color)
plt.title('PEOPLE WHO RENEWED CONTRACT')
plt.ylabel('Customer Count')
plt.show()

#Lets oversampe
smote = SMOTE(random_state = 0)
X_smote, y_smote = smote.fit_resample(X,y)
X_dev, y_dev = X_smote, y_smote


#Creating a Final test set, to test the model
X_smote = X_smote.iloc[881:]
y_smote = y_smote.iloc[881:]

#Selecting first 800 rows to final test set
X_dev = X_dev.iloc[0:880]
y_dev = y_dev.iloc[0:880]

#Checking shape
print('Input values', X_smote.shape)
print('Target value', y_smote.shape)

#Creating ranfomforest
def telecom_randomForest(X_smote,y_smote):
    from sklearn.ensemble import RandomForestClassifier
    
    #Train and Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.30,random_state=0)
    
    #RANDOM FOREST MODEL
    random_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    random_classifier.fit(X_train, y_train)
    random_prediction = random_classifier.predict(X_test)
    
    #Model Accuracy
    print('RANDOM FOREST CLASSIFIER FOR HR ANALYTICS - JOB CHANGE TRAIN DATAFRAME')
    print(confusion_matrix(y_test, random_prediction))
    print(classification_report(y_test, random_prediction))
    print('Model Accuracy: ',accuracy_score(y_test, random_prediction))
    
    #Save Model
    joblib.dump(random_classifier, 'telecom_random_forest.joblib')
    
    return random_classifier

#Calling random forest function
telecom_randomForest(X_smote,y_smote)

#Taking the Samme Model out of the function.
from sklearn.ensemble import RandomForestClassifier
    
#Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.30,random_state=0)
    
#RANDOM FOREST MODEL
random_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
random_classifier.fit(X_train, y_train)
random_prediction = random_classifier.predict(X_test)
    
#Model Accuracy
print('RANDOM FOREST CLASSIFIER FOR HR ANALYTICS - JOB CHANGE TRAIN DATAFRAME')
print(confusion_matrix(y_test, random_prediction))
print(classification_report(y_test, random_prediction))
print('Model Accuracy: ',accuracy_score(y_test, random_prediction))

#Predicting Target values for final test set
predictions = random_classifier.predict(X_dev)

#Converting into pandas object
predictions = pd.DataFrame(predictions)
predictions.columns = ['Predictions']


#Creating a Datafame for the actual values
actual_values = pd.DataFrame(y_dev)
actual_values.columns = ['ActualValues']

#Join dataframes
results = actual_values.join(predictions)
#Let´s compare the actual values with the predictions for the final test set
print(results.head(20))


# Our model is very good in general. There are some mispredictions, but in general the predictions are very good. Let´s compare predictions with actual values, and create a third columns to see how many good and bad predictions we have.

#Lets comparte the values of the columns and see if the predictions were true of vales
predicted = results['ActualValues'].str.strip().str.lower() == results['Predictions'].str.strip().str.lower()

#Transform predicted into pandas DF
predicted = pd.DataFrame(predicted)
predicted.columns = ['True or False']

#Applying check function
predicted['True or False'] = predicted['True or False'].apply(checkFunc)

#Plot correct and incorrect results
predicted_color = ['lightgreen','darkred']
sns.countplot(data=predicted, x='True or False',palette=predicted_color)
plt.xlabel('Predictions')
plt.title('PREDICTIONS CHECK')
plt.show()

#predictions %
pred = predicted['True or False'].value_counts()
color = ['lightgreen','darkred']
plt.figure(figsize=(8,6))
plt.pie(pred,labels=pred.index,autopct='%1.2f%%',explode=(0,0.1),colors=color)
plt.title('MODEL PREDICTION ON FINAL TEST SET')
plt.axis('equal')
plt.show()

