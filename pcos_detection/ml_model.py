#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


import warnings
warnings.ignore = True


# In[ ]:


l = pd.read_csv("PCOS_data.csv")


# In[ ]:


#l.head()


# In[ ]:


#l.info()


# In[ ]:


l.columns


# In[ ]:


#round(l.describe(),2).T


# In[ ]:


df = l.drop('Patient File No.',axis=1)
df = df.drop('Sl. No',axis=1)
#df


# # preprocessing

# In[ ]:


df[' Age (yrs)'] = pd.to_numeric(df[' Age (yrs)'], errors='coerce')
df['Weight (Kg)'] = pd.to_numeric(df['Weight (Kg)'], errors='coerce')
df['Height(Cm) '] = pd.to_numeric(df['Height(Cm) '], errors='coerce')
df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
df['Blood Group'] = pd.to_numeric(df['Blood Group'], errors='coerce')
df['Pulse rate(bpm) '] = pd.to_numeric(df['Pulse rate(bpm) '], errors='coerce')
df['RR (breaths/min)'] = pd.to_numeric(df['RR (breaths/min)'], errors='coerce')
df['Hb(g/dl)'] = pd.to_numeric(df['Hb(g/dl)'], errors='coerce')
df['Cycle(R/I)'] = pd.to_numeric(df['Cycle(R/I)'], errors='coerce')
df['Cycle length(days)'] = pd.to_numeric(df['Cycle length(days)'], errors='coerce')
df['Marraige Status (Yrs)'] = pd.to_numeric(df['Marraige Status (Yrs)'], errors='coerce')
df['Pregnant(Y/N)'] = pd.to_numeric(df['Pregnant(Y/N)'], errors='coerce')
df['No. of abortions'] = pd.to_numeric(df['No. of abortions'], errors='coerce')
df['  I   beta-HCG(mIU/mL)'] = pd.to_numeric(df['  I   beta-HCG(mIU/mL)'], errors='coerce')
df['II    beta-HCG(mIU/mL)'] = pd.to_numeric(df['II    beta-HCG(mIU/mL)'], errors='coerce')
df['FSH(mIU/mL)'] = pd.to_numeric(df['FSH(mIU/mL)'], errors='coerce')
df['LH(mIU/mL)'] = pd.to_numeric(df['LH(mIU/mL)'], errors='coerce')
df['FSH/LH'] = pd.to_numeric(df['FSH/LH'], errors='coerce')
df['Hip(inch)'] = pd.to_numeric(df['Hip(inch)'], errors='coerce')
df['Waist(inch)'] = pd.to_numeric(df['Waist(inch)'], errors='coerce')
df['Waist:Hip Ratio'] = pd.to_numeric(df['Waist:Hip Ratio'], errors='coerce')
df['TSH (mIU/L)'] = pd.to_numeric(df['TSH (mIU/L)'], errors='coerce')
df['AMH(ng/mL)'] = pd.to_numeric(df['AMH(ng/mL)'], errors='coerce')
df['PRL(ng/mL)'] = pd.to_numeric(df['PRL(ng/mL)'], errors='coerce')
df['Vit D3 (ng/mL)'] = pd.to_numeric(df['Vit D3 (ng/mL)'], errors='coerce')
df['PRG(ng/mL)'] = pd.to_numeric(df['PRG(ng/mL)'], errors='coerce')
df['RBS(mg/dl)'] = pd.to_numeric(df['RBS(mg/dl)'], errors='coerce')
df['Weight gain(Y/N)'] = pd.to_numeric(df['Weight gain(Y/N)'], errors='coerce')
#df['LH'] = pd.to_numeric(df['LH'], errors='coerce')
df['hair growth(Y/N)'] = pd.to_numeric(df['hair growth(Y/N)'], errors='coerce')
df['Skin darkening (Y/N)'] = pd.to_numeric(df['Skin darkening (Y/N)'], errors='coerce')
df['Hair loss(Y/N)'] = pd.to_numeric(df['Hair loss(Y/N)'], errors='coerce')
df['Pimples(Y/N)'] = pd.to_numeric(df['Pimples(Y/N)'], errors='coerce')
df['Fast food (Y/N)'] = pd.to_numeric(df['Fast food (Y/N)'], errors='coerce')
df['Reg.Exercise(Y/N)'] = pd.to_numeric(df['Reg.Exercise(Y/N)'], errors='coerce')
df['BP _Systolic (mmHg)'] = pd.to_numeric(df['BP _Systolic (mmHg)'], errors='coerce')
df['BP _Diastolic (mmHg)'] = pd.to_numeric(df['BP _Diastolic (mmHg)'], errors='coerce')
df['Follicle No. (L)'] = pd.to_numeric(df['Follicle No. (L)'], errors='coerce')
df['Follicle No. (R)'] = pd.to_numeric(df['Follicle No. (R)'], errors='coerce')
df['Avg. F size (L) (mm)'] = pd.to_numeric(df['Avg. F size (L) (mm)'], errors='coerce')
df['Avg. F size (R) (mm)'] = pd.to_numeric(df['Avg. F size (R) (mm)'], errors='coerce')
df['Endometrium (mm)'] = pd.to_numeric(df['Endometrium (mm)'], errors='coerce')
df['Unnamed: 44'] = pd.to_numeric(df['Unnamed: 44'], errors='coerce')


# In[ ]:


age_mean = df[' Age (yrs)'].mean()
wt_mean = df['Weight (Kg)'].mean()
ht_mean = df['Height(Cm) '] .mean()
bmi_mean = df['BMI'].mean()
bg_mean = df['Blood Group'].mean()
pr_mean = df['Pulse rate(bpm) '].mean()
rr_mean = df['RR (breaths/min)'].mean()
hb_mean = df['Hb(g/dl)'].mean()
cycle_mean = df['Cycle(R/I)'].mean()
cyclel_mean = df['Cycle length(days)'].mean()
ms_mean = df['Marraige Status (Yrs)'].mean()
p_mean = df['Pregnant(Y/N)'] .mean()
noa_mean = df['No. of abortions'].mean()
IB_mean = df['  I   beta-HCG(mIU/mL)'].mean()
IIB_mean = df['II    beta-HCG(mIU/mL)'].mean()
fsh_mean = df['FSH(mIU/mL)'].mean()
lh_mean = df['LH(mIU/mL)'].mean()
fshlh_mean = df['FSH/LH'].mean()
hip_mean = df['Hip(inch)'].mean()
waist_mean = df['Waist(inch)'].mean()
whr_mean = df['Waist:Hip Ratio'] .mean()
tsh_mean = df['TSH (mIU/L)'].mean()
amh_mean = df['AMH(ng/mL)'].mean()
prl_mean = df['PRL(ng/mL)'].mean()
vit_mean = df['Vit D3 (ng/mL)'].mean()
prg_mean = df['PRG(ng/mL)'].mean()
rbs_mean = df['RBS(mg/dl)'].mean()
wg_mean = df['Weight gain(Y/N)'].mean()
hg_mean = df['hair growth(Y/N)'].mean()
sd_mean = df['Skin darkening (Y/N)'].mean()
hl_mean = df['Hair loss(Y/N)'].mean()
pim_mean = df['Pimples(Y/N)'].mean()
ff_mean = df['Fast food (Y/N)'].mean()
re_mean = df['Reg.Exercise(Y/N)'].mean()
bps_mean = df['BP _Systolic (mmHg)'].mean()
bpd_mean = df['BP _Diastolic (mmHg)'].mean()
fnl_mean = df['Follicle No. (L)'].mean()
fnr_mean = df['Follicle No. (R)'].mean()
afsl_mean = df['Avg. F size (L) (mm)'].mean()
afsr_mean = df['Avg. F size (R) (mm)'].mean()
end_mean = df['Endometrium (mm)'].mean()
unn_mean = df['Unnamed: 44'].mean()


# In[ ]:


df[' Age (yrs)'] = df[' Age (yrs)'].fillna(age_mean)
df['Weight (Kg)'] = df['Weight (Kg)'].fillna(wt_mean)
df['Height(Cm) '] = df['Height(Cm) '].fillna(ht_mean)
df['BMI'] = df['BMI'].fillna(bmi_mean)
df['Blood Group'] = df['Blood Group'].fillna(bg_mean)
df['Pulse rate(bpm) '] = df['Pulse rate(bpm) '].fillna(pr_mean)
df['RR (breaths/min)'] = df['RR (breaths/min)'].fillna(rr_mean)
df['Hb(g/dl)'] = df['Hb(g/dl)'].fillna(hb_mean)
df['Cycle(R/I)'] = df['Cycle(R/I)'].fillna(cycle_mean)
df['Cycle length(days)'] = df['Cycle length(days)'].fillna(cyclel_mean)
df['Marraige Status (Yrs)'] = df['Marraige Status (Yrs)'].fillna(ms_mean)
df['Pregnant(Y/N)'] = df['Pregnant(Y/N)'].fillna(p_mean)
df['No. of abortions'] = df['No. of abortions'].fillna(noa_mean)
df['  I   beta-HCG(mIU/mL)'] = df['  I   beta-HCG(mIU/mL)'].fillna(IB_mean)
df['II    beta-HCG(mIU/mL)'] = df['II    beta-HCG(mIU/mL)'].fillna(IIB_mean)
df['FSH(mIU/mL)'] = df['FSH(mIU/mL)'].fillna(fsh_mean)
df['LH(mIU/mL)'] = df['LH(mIU/mL)'].fillna(lh_mean)
df['FSH/LH'] = df['FSH/LH'].fillna(fshlh_mean)
df['Hip(inch)'] = df['Hip(inch)'].fillna(hip_mean)
df['Waist(inch)'] = df['Waist(inch)'].fillna(waist_mean)
df['Waist:Hip Ratio'] = df['Waist:Hip Ratio'].fillna(whr_mean)
df['TSH (mIU/L)'] = df['TSH (mIU/L)'] .fillna(tsh_mean)
df['AMH(ng/mL)'] = df['AMH(ng/mL)'].fillna(amh_mean)
df['PRL(ng/mL)'] = df['PRL(ng/mL)'].fillna(prl_mean)
df['Vit D3 (ng/mL)'] = df['Vit D3 (ng/mL)'].fillna(vit_mean)
df['PRG(ng/mL)'] = df['PRG(ng/mL)'].fillna(prg_mean)
df['RBS(mg/dl)'] = df['RBS(mg/dl)'].fillna(rbs_mean)
df['Weight gain(Y/N)'] = df['Weight gain(Y/N)'].fillna(wg_mean)
#df['LH'] = pd.to_numeric(df['LH'], errors='coerce')
df['hair growth(Y/N)'] = df['hair growth(Y/N)'] .fillna(hg_mean)
df['Skin darkening (Y/N)'] = df['Skin darkening (Y/N)'].fillna(sd_mean)
df['Hair loss(Y/N)'] = df['Hair loss(Y/N)'].fillna(hl_mean)
df['Pimples(Y/N)'] = df['Pimples(Y/N)'].fillna(pim_mean)
df['Fast food (Y/N)'] = df['Fast food (Y/N)'].fillna(ff_mean)
df['Reg.Exercise(Y/N)'] = df['Reg.Exercise(Y/N)'].fillna(re_mean)
df['BP _Systolic (mmHg)'] = df['BP _Systolic (mmHg)'].fillna(bps_mean)
df['BP _Diastolic (mmHg)'] = df['BP _Diastolic (mmHg)'].fillna(bpd_mean)
df['Follicle No. (L)'] = df['Follicle No. (L)'].fillna(fnl_mean)
df['Follicle No. (R)'] = df['Follicle No. (R)'].fillna(fnr_mean)
df['Avg. F size (L) (mm)'] = df['Avg. F size (L) (mm)'].fillna(afsl_mean)
df['Avg. F size (R) (mm)'] = df['Avg. F size (R) (mm)'].fillna(afsr_mean)
df['Endometrium (mm)'] = df['Endometrium (mm)'].fillna(end_mean)
df['Unnamed: 44'] = df['Unnamed: 44'].fillna(unn_mean)


# In[ ]:


df.to_csv('processed_dataset.csv',index=False)


# In[ ]:


# df.shape
# round(df.corr(),2)


# In[ ]:


x = df.iloc[:,1:]
x


# In[ ]:


y = df['PCOS (Y/N)']
y


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20, random_state=2)


# In[ ]:


# x_train.shape


# In[ ]:


# y_train.shape


# In[ ]:





# In[ ]:


from statsmodels import api as sm 
model_stats = sm.OLS(y_train,x_train)
#If the p-value is very small (usually less than 0.05 or 0.01), 
#then we can reject the null hypothesis in favor of an alternative hypothesis.
#If the p-value is large, we fail to reject the null hypothesis.


# In[ ]:


model_stats= model_stats.fit()
model_stats.summary()


# In[ ]:


x_train.columns


# In[ ]:


model_updated = sm.OLS(y_train,x_train.drop([' Age (yrs)','Weight (Kg)','Height(Cm) ','BMI','Hb(g/dl)','Pregnant(Y/N)','FSH(mIU/mL)','Vit D3 (ng/mL)'], axis=1))
model_updated=model_updated.fit()
model_updated.summary()


# In[ ]:


model_updated.fittedvalues[:5]


# In[ ]:


y_train[:5]


# In[ ]:


model_updated.resid[:5]


# In[ ]:


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_train,model_updated.predict(x_train.drop([' Age (yrs)','Weight (Kg)','Height(Cm) ','BMI','Hb(g/dl)','Pregnant(Y/N)','FSH(mIU/mL)','Vit D3 (ng/mL)'], axis=1))))


# In[ ]:


df.drop([' Age (yrs)','Weight (Kg)','Height(Cm) ','BMI','Hb(g/dl)','Pregnant(Y/N)','FSH(mIU/mL)','Vit D3 (ng/mL)'], axis=1, inplace=True)
df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lr = LinearRegression()


# In[ ]:


model = lr.fit(x_train,y_train)


# In[ ]:


model


# In[ ]:


model.coef_[12]


# In[ ]:


model.coef_


# In[ ]:


df.head()


# In[ ]:


dir(model)


# In[ ]:


model.fit_intercept


# In[ ]:


model.intercept_


# In[ ]:


p = model.intercept_+(model.coef_[0]*5.82115)+(model.coef_[1]*0.00000)+(model.coef_[2]*18.10000)+(model.coef_[3]*0.00000)+(model.coef_[4]*0.71300)+(model.coef_[5]*6.51300)+(model.coef_[6]*89.90000)+(model.coef_[7]*2.80160)+(model.coef_[8]*24.00000)+(model.coef_[9]*666.00000)+(model.coef_[10]*20.20000)+(model.coef_[11]*393.82000)+(model.coef_[12]*10.29000)


# In[ ]:


pred_values = model.predict(x_test)


# In[ ]:


x_test.iloc[0]


# In[ ]:


y_test.iloc[0]


# In[ ]:


pred_values


# In[ ]:


x = df.iloc[:,1:]


# In[ ]:


y = df['PCOS (Y/N)']


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20, random_state=2)


# In[ ]:


x_train.shape


# In[ ]:


y_train.shape


# In[ ]:


x_train


# In[ ]:


# Assumption 2 --> Avoid high multi collinearity
# VIF  --> Variance Inflation factor, if it is higher than 10 generally variable seems to be 
# highly correlated


# In[ ]:





from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:



vif = pd.DataFrame()


# In[ ]:


vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]


# In[ ]:


vif["Predictor"] = x.columns


# In[ ]:


vif["VIF Factor"]


# In[ ]:


vif["Predictor"]


# In[ ]:


threshold = 10
while vif["VIF Factor"].max() > threshold:
    # Get the index of the predictor variable with the highest VIF value
    idx = vif["VIF Factor"].argmax()
    
    # Remove the predictor variable from the X DataFrame
    x.drop(x.columns[idx], axis=1, inplace=True)
    
    # Recalculate the VIF for the remaining predictor variables
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif["Predictor"] = x.columns

# Fit a linear regression model using the remaining predictor variables
model = sm.OLS(y, x).fit()


# In[ ]:


x.columns


# In[ ]:


df.columns


# In[ ]:


df.drop(['Weight gain(Y/N)'], axis=1, inplace=True)


# In[ ]:


y = df['PCOS (Y/N)']


# In[ ]:


feature_names = ['Cycle(R/I)', 'Cycle length(days)', 'Marraige Status (Yrs)',
       'No. of abortions', '  I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)',
       'LH(mIU/mL)', 'FSH/LH', 'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)',
       'PRG(ng/mL)', 'hair growth(Y/N)',
       'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)',
       'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'Follicle No. (L)',
       'Follicle No. (R)']
x = df[feature_names]


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y)


# In[ ]:


from sklearn import svm
classifier = svm.SVC(kernel = 'linear')
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(y_test,y_pred))


# In[ ]:





# In[ ]:


threshold = 15
while vif["VIF Factor"].max() > threshold:
    # Get the index of the predictor variable with the highest VIF value
    idx = vif["VIF Factor"].argmax()
    
    # Remove the predictor variable from the X DataFrame
    x.drop(x.columns[idx], axis=1, inplace=True)
    
    # Recalculate the VIF for the remaining predictor variables
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif["Predictor"] = x.columns

# Fit a linear regression model using the remaining predictor variables
model = sm.OLS(y, x).fit()


# In[ ]:


x.columns


# In[ ]:


df.columns


# In[ ]:


df.drop([' Age (yrs)', 'Weight (Kg)', 'Height(Cm) ', 'BMI', 'Blood Group','Hb(g/dl)','Pregnant(Y/N)','FSH(mIU/mL)','Vit D3 (ng/mL)'], axis=1, inplace=True)


# In[ ]:


feature_names = [' Age (yrs)', 'Weight (Kg)', 'Height(Cm) ', 'BMI', 'Blood Group',
       'Pulse rate(bpm) ', 'RR (breaths/min)', 'Hb(g/dl)', 'Cycle(R/I)',
       'Cycle length(days)', 'Marraige Status (Yrs)', 'Pregnant(Y/N)',
       'No. of abortions', '  I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)',
       'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)', 'Waist(inch)',
       'Waist:Hip Ratio', 'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)',
       'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 'Weight gain(Y/N)',
       'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)',
       'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)',
       'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 'Follicle No. (L)',
       'Follicle No. (R)', 'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)',
       'Endometrium (mm)']
x = df[feature_names]


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y)


# In[ ]:


from sklearn import svm
classifier = svm.SVC(kernel = 'linear')
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


threshold = 25
while vif["VIF Factor"].max() > threshold:
    # Get the index of the predictor variable with the highest VIF value
    idx = vif["VIF Factor"].argmax()
    
    # Remove the predictor variable from the X DataFrame
    x.drop(x.columns[idx], axis=1, inplace=True)
    
    # Recalculate the VIF for the remaining predictor variables
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif["Predictor"] = x.columns

# Fit a linear regression model using the remaining predictor variables
model = sm.OLS(y, x).fit()


# In[ ]:


x.columns


# In[ ]:


df.drop(['Blood Group', 'Pulse rate(bpm) ', 'RR (breaths/min)','Hip(inch)', 'Waist(inch)', 'Waist:Hip Ratio','RBS(mg/dl)','BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 'Avg. F size (L) (mm)',
       'Avg. F size (R) (mm)', 'Endometrium (mm)', 'Unnamed: 44'], axis=1, inplace=True)


# In[ ]:


y = df['PCOS (Y/N)']


# In[ ]:


feature_names = ['Cycle(R/I)', 'Cycle length(days)', 'Marraige Status (Yrs)',
       'No. of abortions', '  I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)',
       'LH(mIU/mL)', 'FSH/LH', 'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)',
       'PRG(ng/mL)', 'Weight gain(Y/N)', 'hair growth(Y/N)',
       'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)',
       'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'Follicle No. (L)',
       'Follicle No. (R)']


# In[ ]:


x = df[feature_names]


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y)


# In[ ]:


from sklearn import svm
classifier = svm.SVC(kernel = 'linear')
classifier.fit(x_train,y_train)


# In[ ]:


y_pred = classifier.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(y_test,y_pred))


# In[ ]:


import pickle
# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(classifier, file)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# NAIVE BAYES

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


# In[ ]:


model=GaussianNB()
model.fit(x_train,y_train)


# In[ ]:


y_pred=model.predict(x_test)


# In[ ]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


classifier=KNeighborsClassifier(n_neighbors=7)
classifier.fit(x_train, y_train)


# In[ ]:


y_pred=classifier.predict(x_test)
from sklearn import metrics


# In[ ]:


y_pred = classifier.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


# KMeans

# In[ ]:


pip install xgboost


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression


# In[ ]:



model_1 = LinearRegression()
model_2 = xgb.XGBRegressor()
model_3 = RandomForestRegressor()


# In[ ]:


model_1.fit(x_train, y_train)
model_2.fit(x_train, y_train)
model_3.fit(x_train, y_train)


# In[ ]:



pred_1 = model_1.predict(x_test)
pred_2 = model_2.predict(x_test)
pred_3 = model_3.predict(x_test)


# In[ ]:


pred_final = (pred_1+pred_2+pred_3)/3.0


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


print(mean_squared_error(y_test, pred_final))


# ENSEMBLE METHODS

# In[ ]:


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# create the individual classifiers
svm_clf = SVC(kernel='linear', probability=True)
knn_clf = KNeighborsClassifier(n_neighbors=5)
rf_clf = RandomForestClassifier(n_estimators=100)

# create the ensemble voting classifier
voting_clf = VotingClassifier(
    estimators=[('svm', svm_clf), ('knn', knn_clf), ('rf', rf_clf)],
    voting='soft')

# fit the voting classifier on the training data
voting_clf.fit(x_train, y_train)

# predict on the test data
y_pred = voting_clf.predict(x_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


# # Bagging

# In[ ]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Create a base decision tree model
base_model = DecisionTreeClassifier()

# Create a BaggingClassifier model
bagging_model = BaggingClassifier(base_estimator=base_model, n_estimators=10, random_state=42)

# Fit the model to the training data
bagging_model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = bagging_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)


# In[ ]:


print(accuracy)


# # Boosting

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

# Create an AdaBoostClassifier model
adaboost_model = AdaBoostClassifier(n_estimators=100, random_state=42)

# Fit the model to the training data
adaboost_model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = adaboost_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)


# In[ ]:


print(accuracy)


# # Hyper Parameter Tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
# Define the parameter grid to search over
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
        'criterion':['gini','entropy']

}

# Create a random forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)

# Fit the model to the training data
grid_search.fit(x_train, y_train)
with open('model.pkl', 'wb') as f:
    pickle.dump(grid_search, f)
# Print the best hyperparameters
print(grid_search.best_params_)

# Make predictions on the test data using the best model
y_pred = grid_search.best_estimator_.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)


# In[ ]:


x_test


# In[ ]:





# In[ ]:


print(accuracy)


# RANDOM FOREST ALGORITHM

# In[ ]:


# Import the required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# Split the data into training and testing sets
# Import the required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('PCOS (Y/N)', axis=1),
                                                    df['PCOS (Y/N)'],
                                                    test_size=0.3,
                                                    random_state=42)

# Create a Random Forest Classifier with 100 trees
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
rfc.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rfc.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# Create a Random Forest Classifier with 100 trees
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
rfc.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rfc.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# DECISION TREE

# In[ ]:




# Import the required libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('PCOS (Y/N)', axis=1),
                                                    df['PCOS (Y/N)'],
                                                    test_size=0.3,
                                                    random_state=42)

# Create a Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
dtc.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = dtc.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# XG BOOST

# In[ ]:





# Import the required libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('PCOS (Y/N)', axis=1),
                                                    df['PCOS (Y/N)'],
                                                    test_size=0.3,
                                                    random_state=42)

# Convert the data into DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set the XGBoost parameters
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Train the XGBoost model
model = xgb.train(params, dtrain, num_boost_round=100)

# Make predictions on the testing data
y_pred = model.predict(dtest)
y_pred = [round(value) for value in y_pred]

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# In[ ]:





# In[ ]:




