import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import statsmodels.api as sm

def read_path(path):
    df = pd.read_csv(path)
    print('shape ' , df.shape)
    print()
    print(df.info())
    return df.iloc[:,0:-1],df.iloc[:,-1]


def forward_selection(data, target, significance_level=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
    data = list(data)
    for i in range(len(data)):
        if data[i] in best_features:
            data[i] = True
        else :
            data[i] = False
    return data


def indexes(X,ar):
    index_array = []
    c = 0
    for i in list(ar[0]):
        if i == True:
            index_array.append(c)
        c+=1
    # print(index_array)
    return X.iloc[:,index_array]


X,y = read_path('D:\Internship\dataset.csv')


columns = pd.DataFrame(list(X.columns))



forward_selection = pd.DataFrame(forward_selection(X, y))
X0 = indexes(X,forward_selection)

d = int(X0.shape[1]*0.80)

chi_square = pd.DataFrame(SelectKBest(score_func=chi2 , k=d).fit(X0,y).get_support())
X1 = indexes(X0,chi_square)

RFC = pd.DataFrame(SelectFromModel(RandomForestClassifier(random_state=0)).fit(X1, y).get_support())
X2 = indexes(X1,RFC)

lss = pd.DataFrame(SelectFromModel(Lasso(alpha=0.05,random_state=0)).fit(X2, y).get_support())
X3 = indexes(X2,lss)


print(X.columns)
print(X0.columns)
print(X1.columns)
print(X2.columns)
print(X3.columns)

print(X.shape)
print(X0.shape)
print(X1.shape)
print(X2.shape)
print(X3.shape)


# result = pd.concat([columns,chi_square,RFC,lss,lr,forward_selection],axis=1)
# result = pd.concat([result,result.sum(axis=1)],axis=1)
# result.columns = ['Specs','Chi-square','Random Forest','Lasso','Logistic Regression','Forward Selection','Total']

# result.to_csv('result.csv')

# print('Saved Result.csv')
 
