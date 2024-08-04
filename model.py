import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("PCOS_data_without_infertility(2).csv")
df = df.drop(columns='Unnamed: 44')
df= df.drop(columns=['Sl. No','Patient File No.'])


df["BMI"] = df["Weight (Kg)"] / np.square(df["Height(Cm) "] / 100)
# Round to two decimal places
df["BMI"] = df["BMI"].round(2)

df["FSH/LH"] = df["FSH(mIU/mL)"] / df["LH(mIU/mL)"]
# Round the result to two decimal places
df["FSH/LH"] = df["FSH/LH"].round(2)

df["Waist:Hip Ratio"] = df["Waist(inch)"] / df["Hip(inch)"]
df["Waist:Hip Ratio"] = df["Waist:Hip Ratio"].round(2)



#replace with binary values
df.loc[df["Cycle(R/I)"] == 5, "Cycle(R/I)"] = 4
df.loc[df["Cycle(R/I)"].isin([2, 4]), "Cycle(R/I)"] = df["Cycle(R/I)"].replace({2: 0, 4: 1})
df["II    beta-HCG(mIU/mL)"] = df["II    beta-HCG(mIU/mL)"].replace({"1.99.": 1.99})
#convert object datatype to float
df["II    beta-HCG(mIU/mL)"] = df["II    beta-HCG(mIU/mL)"].astype(float)
df.drop(df.loc[df["AMH(ng/mL)"]== "a"].index, inplace=True);
#convert object dt to float dt
df["AMH(ng/mL)"] = df["AMH(ng/mL)"].astype(float)

z_scores = np.abs((df - df.mean()) / df.std())
threshold = 3

# Identify outliers
outliers = df[(z_scores > threshold).any(axis=1)]

# Remove outliers
clean_data = df[(z_scores <= threshold).all(axis=1)]

import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
features=clean_data.drop('PCOS (Y/N)',axis=1)
target=clean_data['PCOS (Y/N)']
x_resamp,y_resamp=SMOTE(random_state=42).fit_resample(features,target)

clean_data = pd.concat([x_resamp, y_resamp], axis=1)

import seaborn as sns
import matplotlib.pyplot as plt

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs((corr_matrix.iloc[i, j])) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

#feature matrix
xc=clean_data.drop("PCOS (Y/N)",axis=1)
#target variable
yc=clean_data["PCOS (Y/N)"]


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(
        xc,   #feature matrix
        yc,   #target var
        test_size=0.3,
        random_state=42
      )

rf = RandomForestClassifier(max_depth=44,random_state=42)
rf.fit(X_train,y_train)

clf_cat = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6)
clf_cat.fit(X_train, y_train,verbose=False)

xgb_classifier = xgb.XGBClassifier(n_estimators=100,max_depth=5,learning_rate=0.01,enable_categorical=True)
xgb_classifier.fit(X_train, y_train)

classifiers = [('Random Forest',rf ),('XgBoost',xgb_classifier),('Catboost', clf_cat)]

# Create a VotingClassifier with soft voting
voting_clf = VotingClassifier(estimators=classifiers, voting='soft')

# Train the VotingClassifier
voting_clf.fit(X_train, y_train)

#inp=(25,74.0,152.0,32.02,17,72,18,11.7,1,2,7.0,1,0,1214.23,1214.23,2.0,1.51,1.32,45,40,0.88,6.51,7.94,22.43,31.4,0.3,125.0,1,1,1,1,1,1.0,1,120,80,15,8,20.0,21.0,8.0)


import pickle
pickle.dump(voting_clf, open('model.pkl', 'wb'))
