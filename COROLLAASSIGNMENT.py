#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[2]:


car=pd.read_csv(r"E:\TUSHAR\ASSINMENT DONE BY TUSHAR\ASSIGNMENT NO 5 (MULTILINEAR REGRESSION)\ToyotaCorolla.csv",encoding='latin1')
car.head(2)


# In[3]:


cars=pd.concat([car.iloc[:,2:4],car.iloc[:,6:7],car.iloc[:,8:9],car.iloc[:,12:14],car.iloc[:,15:18]],axis=1)


# In[4]:


cars.isna().sum()


# In[5]:


cars.head(2)


# In[6]:


car1=cars.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'TAX'},axis=1)
car1.head(2)


# In[7]:


car1[car1.duplicated()]


# In[8]:


car2=car1.drop_duplicates().reset_index(drop=True)
car2


# In[9]:


car2.corr()


# In[10]:


sns.pairplot(car2)


# In[11]:


model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+TAX+Weight',data=car2).fit()


# In[12]:


model.params


# In[13]:


print(model.tvalues,'\n',np.round(model.pvalues,4))


# In[14]:


model.rsquared,model.rsquared_adj


# In[15]:


ml_c=smf.ols("Price~CC",data=car2).fit()
ml_c.pvalues


# In[16]:


ml_d=smf.ols("Price~Doors",data=car2).fit()
ml_d.pvalues


# In[17]:


ml_cd=smf.ols("Price~CC+Doors",data=car2).fit()
ml_cd.pvalues


# In[18]:


rsq_c=smf.ols("CC~Doors+Age+KM+HP+TAX+Weight+Gears",data=car2).fit().rsquared
vif_c=1/(1-rsq_c)
vif_c


# In[19]:


rsq_d=smf.ols("Doors~CC+Age+KM+HP+TAX+Weight+Gears",data=car2).fit().rsquared
vif_d=1/(1-rsq_d)
vif_d


# In[20]:


rsq_a=smf.ols("Age~Doors+CC+KM+HP+TAX+Weight+Gears",data=car2).fit().rsquared
vif_a=1/(1-rsq_a)
vif_a


# In[21]:


rsq_k=smf.ols("KM~Doors+CC+Age+HP+TAX+Weight+Gears",data=car2).fit().rsquared
vif_k=1/(1-rsq_k)
vif_k
rsq_h=smf.ols("HP~Doors+CC+KM+Age+TAX+Weight+Gears",data=car2).fit().rsquared
vif_h=1/(1-rsq_h)
vif_h
rsq_t=smf.ols("TAX~Doors+CC+KM+HP+Age+Weight+Gears",data=car2).fit().rsquared
vif_t=1/(1-rsq_t)
vif_t
rsq_w=smf.ols("Weight~Doors+CC+KM+HP+TAX+Age+Gears",data=car2).fit().rsquared
vif_w=1/(1-rsq_w)
vif_w
rsq_g=smf.ols("Gears~Doors+CC+KM+HP+TAX+Weight+Age",data=car2).fit().rsquared
vif_g=1/(1-rsq_g)
vif_g


# In[22]:


df={'Variables':['CC','Doors','Age','KM','HP','TAX','Weight','Gears'],
    'VIF':[vif_c,vif_d,vif_a,vif_k,vif_h,vif_t,vif_w,vif_g]}
vif=pd.DataFrame(df)
vif


# In[23]:


qqplot=sm.qqplot(model.resid,line='q')


# In[24]:


def standard_values(vals):
    return(vals-vals.mean())/vals.std()
plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))


# In[25]:


fig=plt.figure(figsize=(16,8))
fig=sm.graphics.plot_regress_exog(model,'CC',fig=fig)


# In[26]:


fig=plt.figure(figsize=(16,8))
fig=sm.graphics.plot_regress_exog(model,'KM',fig=fig)


# In[27]:


fig=plt.figure(figsize=(16,8))
fig=sm.graphics.plot_regress_exog(model,'Weight',fig=fig)


# In[28]:


fig=plt.figure(figsize=(16,8))
fig=sm.graphics.plot_regress_exog(model,'Doors',fig=fig)


# In[29]:


fig=plt.figure(figsize=(16,8))
fig=sm.graphics.plot_regress_exog(model,'Age',fig=fig)


# In[30]:


fig=plt.figure(figsize=(16,8))
fig=sm.graphics.plot_regress_exog(model,'Gears',fig=fig)


# In[31]:


fig=plt.figure(figsize=(16,8))
fig=sm.graphics.plot_regress_exog(model,'TAX',fig=fig)


# In[32]:


fig=plt.figure(figsize=(16,8))
fig=sm.graphics.plot_regress_exog(model,'HP',fig=fig)


# In[33]:


influence_plot(model)
plt.show()


# In[34]:


car2[car2.index.isin([80])]


# In[35]:


car3=car2.drop(car2.index[[80]],axis=0).reset_index(drop=True)


# In[36]:


car3.head(2)


# In[37]:


final_ml=smf.ols("Price~CC+HP+KM+Weight+Age+Gears+TAX",data=car3).fit()
final_ml.rsquared,final_ml.aic


# In[38]:


final_ml=smf.ols("Price~HP+KM+Weight+Age+Gears+TAX+Doors",data=car3).fit()
final_ml.rsquared,final_ml.aic


# In[44]:


new_data=pd.DataFrame({'Age':20,'KM':5000,'HP':80,'CC':2500,'Doors':4,'Gears':4,'TAX':250,'Weight':1200},index=[1])
final_ml.predict(new_data)


# In[45]:


pred_y=final_ml.predict(car3)
pred_y

