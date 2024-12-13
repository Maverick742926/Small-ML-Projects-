import numpy as np
import pandas as pd

dataset = pd.read_csv(r"C:/Users/ONE10.COMPUTER/Desktop/ML/Uber Request Data.csv")
print(dataset)

print('===SELECTED DATASET===')
selected_dataset = pd.read_csv(r"C:/Users/ONE10.COMPUTER/Desktop/selected data2.csv")
print(selected_dataset)

print('===Separate the dependent and independent variables===')
X = selected_dataset.iloc[:,:-1].values
print('===Independent Variables===')
print(X)
Y = selected_dataset.iloc[:,3].values
print('===Dependent Variables===')
print(Y)

print('###############################################################')
print('=====HANDLING MISSING DATA=====')
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 2] = imputer.fit_transform(X[:, 2].reshape(-1, 1)).flatten()
print(X)

print('###############################################################')
print('=====Categorical variables=====')
print('*****OneHot Encoder*****')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder="passthrough")
X = np.array(ct.fit_transform(X))
print(X)
print('*****Label Encoder*****')
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)
print(Y)

print('###############################################################')
print('===== Splitting the dataset =====')
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 1)
print('X_train')
print (X_train)
print ('----------------')
print('X_test')
print (X_test)
print ('----------------')
print ('Y_train')
print(Y_train)
print ('----------------')
print('Y_test')
print (Y_test)
print ('----------------')

print('###############################################################')
print('===== Feature Scaling =====')
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:,2] = sc_X.fit_transform(X_train[:,2].reshape(-1, 1)).flatten()
X_test[:,2] = sc_X.transform(X_test[:,2].reshape(-1, 1)).flatten()
print('X_train')
print (X_train)
print ('----------------')
print('X_test')
print (X_test)
print ('----------------')

print('###############################################################')
print('===== Removing Outliers =====')
lowerLimit = selected_dataset['Driver id'].quantile(0.05)
upperLimit = selected_dataset['Driver id'].quantile(0.8)
# Filter out rows based on conditions
selected_dataset_Outlier = selected_dataset[(selected_dataset['Driver id'] >= lowerLimit) 
                                    & (selected_dataset['Driver id'] <= upperLimit)]
print(selected_dataset_Outlier)

print('###############################################################')
print('===== Removing Duplicate =====')
print(selected_dataset.drop_duplicates())

print('###############################################################')
print('===== Data Integration =====')
dfIntersct = pd.merge(dataset,selected_dataset)
print(dfIntersct)
joinEmp1_Emp3 =dataset.join(selected_dataset)
print(joinEmp1_Emp3)

# print('===== Confusion Matrix =====')
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# lr = LogisticRegression()
# lr.fit(X_train, Y_train)
# y_pred = lr.predict(X_test)
# print("Predicted labels (y_pred) for x_test:", y_pred)
# accuracy = accuracy_score(Y_test, y_pred)
# print("Accuracy of logistic regression:", accuracy)
# cm = confusion_matrix(Y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
#         xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()



