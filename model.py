#%% import libraries
import os
import pickle
import pandas as pd
import numpy as np
import scipy as np
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.svm import SVC
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from DeepLearnModule import displot_graph,countplot_graph,Trad_FillNa_Con
from DeepLearnModule import Trad_FillNa_Cat,cramers_corrected_stat, LogisticReg, KNNImpute
# %%
# Loading data
CSV_PATH = os.path.join(os.getcwd(),'Data', 'diabetes.csv')
df = pd.read_csv(CSV_PATH)

cat_col = []
cat_col.append('Pregnancies')
con_col = list(df.columns[(df.dtypes=='int64') | (df.dtypes=='float64')])
con_col.remove('Pregnancies')
con_col.remove('Outcome')
# Check data
df
df.info()
df.describe().T

# Data visualization
df.boxplot()

(df['Glucose']==0).sum()
(df['Insulin']==0).sum()

for i in con_col:
   df[i] =  df[i].replace(0,np.nan)

displot_graph(con_col,df)
countplot_graph(cat_col,df)
# %% Data cleaning
# 1 bmi remove outliers, skin thickness, bp insulin
fig, ax = plt.subplots(9, 1, figsize=(15, 20))
df.plot.box(layout=(9, 1), 
            subplots=True, 
            ax=ax, 
            vert=False, 
            sharex=False)
plt.tight_layout()
plt.show()

df.describe().T

df.isna().sum()
msno.matrix(df)

columns_name = df.columns
knn_im = KNNImputer()
df = knn_im.fit_transform(df)
df = pd.DataFrame(df)
df.columns = columns_name
df.info()
df.describe().T
# Trad_FillNa_Con(df,con_col)
# Trad_FillNa_Cat(df, cat_col)

df.isna().sum()
msno.matrix(df)

df.duplicated().sum()
df = df.drop_duplicates()
df.describe().T

#%% Features Selection
for i in cat_col:
    print(i)
    confusion_matrix = pd.crosstab(df[i],df['Outcome']).to_numpy()
    print(cramers_corrected_stat(confusion_matrix))

# LogisticReg(df,con_col,'Outcome')

for i in con_col:
    print(i)
    lr = LogisticRegression(solver='liblinear')
    lr.fit(np.expand_dims(df[i],axis=-1), df['Outcome'])
    print(lr.score(np.expand_dims(df[i],axis=-1),df['Outcome']))

#%%
# All except pregnancies have 60% or more correlation
X = df.loc[:,con_col]
y = df['Outcome']
# Split train test data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3,
                                                     random_state=123)

#%%
# Finding the best model
# Pipeline
# Logistic regression
pipeline_mms_lr = Pipeline([
                        ('Min_Max_Scaler', MinMaxScaler()),
                        ('Logistic_Classifier', LogisticRegression())
                        ])

pipeline_ss_lr = Pipeline([
                        ('Standard_Scaler', StandardScaler()),
                        ('Logistic_Classifier', LogisticRegression())])

# Decision tree
pipeline_mms_dt = Pipeline([
                        ('Min_Max_Scaler', MinMaxScaler()),
                        ('DecisionTreeClassifier', DecisionTreeClassifier())
                        ])

pipeline_ss_dt = Pipeline([
                        ('Standard_Scaler', StandardScaler()),
                        ('DecisionTreeClassifier', DecisionTreeClassifier())])
# Random forest
pipeline_mms_rf = Pipeline([
                        ('Min_Max_Scaler', MinMaxScaler()),
                        ('RandomForestClassifier', RandomForestClassifier())
                        ])

pipeline_ss_rf = Pipeline([
                        ('Standard_Scaler', StandardScaler()),
                        ('RandomForestClassifier', RandomForestClassifier())])

# SVM
pipeline_mms_svm = Pipeline([
                        ('Min_Max_Scaler', MinMaxScaler()),
                        ('SVC', SVC())
                        ])

pipeline_ss_svm = Pipeline([
                        ('Standard_Scaler', StandardScaler()),
                        ('SVC', SVC())])
                        
# KNN
pipeline_mms_KNN = Pipeline([
                        ('Min_Max_Scaler', MinMaxScaler()),
                        ('KNN', KNeighborsClassifier())
                        ])

pipeline_ss_KNN = Pipeline([
                        ('Standard_Scaler', StandardScaler()),
                        ('KNN', KNeighborsClassifier())])



# Create a list to store all the pipeline
pipelines = [pipeline_mms_lr, pipeline_ss_lr, pipeline_mms_dt, pipeline_ss_dt,
             pipeline_mms_rf, pipeline_ss_rf, pipeline_mms_svm, pipeline_ss_svm,
             pipeline_mms_KNN, pipeline_ss_KNN]

for pipe in pipelines:
    pipe.fit(X_train,y_train)

# Create a dictionary for pipeline types
pipe_dict = {0: 'MMS+LogReg', 1:'SS+LogReg', 2:'MMS+DecTree', 3:'SS+DecTree',
            4:'MMS+RanForest', 5:'SS+RanForest', 6:'MMS+SVM', 7:'SS+SVM',
            8:'MMS+KNN', 9:'SS+KNN'}

best_score = 0
for i, pipe in enumerate(pipelines):
    print(pipe_dict[i])
    print(pipe.score(X_test,y_test))
    if pipe.score(X_test,y_test) > best_score:
        best_score = pipe.score(X_test,y_test)
        best_pipeline = pipe

print(f'The best scaler and classifer for this Data is {best_pipeline.steps} with score of {best_score}')
# %%
#The best scaler and classifer for this Data is standard scaler and SVC with 0.83
# Gridsearchcv
# Hyperparameters tuning
# SS and SVM found to be the optimal combination


pipeline_ss_svm = Pipeline([
                        ('Standard_Scaler', StandardScaler()),
                        ('SVC', SVC())])

grid_param = [{'SVC__C':[0.1,0.5,1,2,3],
            'SVC__kernel':['linear', 'poly', 'rbf'],
            }] #Hyperparameters

grid_search = GridSearchCV(pipeline_ss_svm, param_grid=grid_param,cv=5,
            verbose=1, n_jobs=-1)

model = grid_search.fit(X_train,y_train)
print(model.score(X_test,y_test))
print(model.best_index_)
print(model.best_params_)
#%%
BEST_ESTIMATOR_SAVE_PATH = os.path.join(os.getcwd(), 'Model', 'best_estimator.pkl')

with open(BEST_ESTIMATOR_SAVE_PATH, 'wb') as file:
    pickle.dump(model.best_estimator_,file)

# Model Analysis
y_pred = model.predict(X_test)
y_true = y_test

cm = confusion_matrix(y_true,y_pred)
cr = classification_report(y_true,y_pred)

labels = ['0','1']
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

print(cr)