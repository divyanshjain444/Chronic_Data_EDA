import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('kidney_disease.csv')
print(df.head(10))

# df = df.drop('id' , axis = 1)
# print(df)

# print(df.columns)

# print(df.info())

# print(df['pcv'].unique())
# print(df['pcv'].isna().sum())
df['pcv'] = pd.to_numeric(df['pcv'],errors = 'coerce')
# print(df.info())

df['wc'] = pd.to_numeric(df['wc'],errors = 'coerce')
df['rc'] = pd.to_numeric(df['rc'],errors = 'coerce')
# print(df.info())

print(df.columns)
categorical_data= [col for col in df.columns if df[col].dtype  == 'object']
# print("THIS DATA IS BELONGS WITH CATEGORY",categorical_data)

numerical_data = [col for col in df.columns if df[col].dtype != 'object']
# print("this data is belongs with integer",numerical_data)

# for col in categorical_data:
#     print(f"{col} has {df[col].unique()} values \n")

df['dm'] = df['dm'].replace(to_replace = {' yes':'yes' , '\tyes':'yes', '\tno':'no'})
# print(df['dm'].unique())

df['cad'] = df['cad'].replace(to_replace = {'\tno':'no'})
# print(df['cad'].unique())

df['classification'] = df['classification'].replace(to_replace = {'ckd\t':'ckd'})
# print(df['classification'].unique())
df['classification'] = df['classification'].map({'ckd':0 , 'notckd':1})
# print(df['classification'].unique())

# for col in categorical_data:
#     print(f"{col} has {df[col].unique()} values \n")

# sns.countplot(x = 'htn' , data =df)
# plt.show()

# sns.boxplot(x = 'classification',y = 'bu',data =df,palette = 'viridis')
# plt.show()

# sns.violinplot(x ='classification',y = 'sc',data =df,palette = 'muted' )
# plt.show()

# sns.countplot(x = 'ane',data =df)
# plt.show()

# df.appet.value_counts().plot.pie(autopct = '%1.1f%%')
# plt.show()

# sns.countplot(x = 'pcc', data =df,palette='viridis')
# plt.show()

# sns.histplot(df['wc'].dropna(),bins = 20,kde = True)
# plt.show()

# df.dm.value_counts().plot.pie(autopct = '%1.1f%%',wedgeprops = dict(width = 0.3))
# plt.title('diabetes')
# plt.show()

# sns.countplot(x = 'ba' , data = df,palette='muted')
# plt.show()

# bivariate analysis

# sns.scatterplot(x = 'age', y = 'bp' , data = df , hue = 'classification' , palette='coolwarm')
# plt.show()

# sns.boxplot(x = 'dm', y = 'al',data = df,palette='muted')
# plt.show()

# dia_hyper = pd.crosstab(df['dm'] , df['htn'])
# dia_hyper.plot(kind='bar',stacked= True)
# plt.show()

# cols = ['age' , 'bgr' , 'sc' , 'classification']
# g = sns.PairGrid(df[cols] , hue = "classification" , palette='coolwarm')
# g.map_upper(sns.scatterplot)
# g.map_lower(sns.kdeplot , cmap = 'Blues_d')
# g.map_diag(sns.histplot)
# g.add_legend()
# plt.title('Pairgrid for selected columns')
# plt.show()

# fig = px.scatter_3d(df , x = 'age' , y = 'bp',z = 'sc' , color='classification', hover_data=['sc' , 'hemo'])
# fig.show()


