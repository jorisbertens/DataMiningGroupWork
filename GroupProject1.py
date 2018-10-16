# import modules
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import sklearn 
from sklearn import preprocessing

# connect to the database
my_path = 'C:/Users/jojo/Documents/Uni/Data Mining/Databases/insurance.db'
conn = sqlite3.connect(my_path)
cursor = conn.cursor()

# show tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

# create tables and save in df
query_LOB= """select * from LOB"""
query_Engage = """select * from Engage"""
query_Merge = """select * from LOB join Engage using ('Customer Identity')"""

df_LOB= pd.read_sql_query(query_LOB,conn)
df_Engage = pd.read_sql_query(query_Engage,conn)
df_Merge= pd.read_sql_query(query_Merge,conn)

#Clean data
df_Merge.drop(columns='index',inplace=True)
df_Merge = df_Merge.rename(columns={'Brithday Year': 'Birthday Year', 'Premiums in LOB:  Life':'Premiums in LOB: Life'})
df_Merge.replace(to_replace={'2 - High School':2,'1 - Basic' :1,'3 - BSc/MSc':3, '4 - PhD':4}, value = None, inplace = True)
df_Merge['Educational Degree String'] = df_Merge['Educational Degree']
df_Merge['Educational Degree String'] .replace(to_replace={2:'High School',1:'Basic',3:'BSc/MSc',4: 'PhD'}, value = None, inplace = True)
df_Merge = df_Merge.iloc[:,:-1]


# general description of data set
df_describe = df_Merge.describe()
df_Merge.shape[0]

df_Merge
# count null values
df_Merge.isnull().sum()

# drop null values
df_Merge = df_Merge.dropna()

# Count values
df_Merge['Birthday Year'].value_counts()
df_Merge['First Policy´s Year'].value_counts()
df_Merge['Educational Degree'].value_counts()
df_Merge['Has Children (Y=1)'].value_counts()
df_Merge['Educational Degree'].value_counts()

# Show columns
df_Merge.columns
df_Merge

# create boxplots to show outliers
# How to do this with a for loop?
sns.boxplot(x=df_Merge['Customer Identity'])
sns.boxplot(x=df_Merge['Premiums in LOB: Motor'])
sns.boxplot(x=df_Merge['Premiums in LOB: Household'])
sns.boxplot(x=df_Merge['Premiums in LOB: Health'])
sns.boxplot(x=df_Merge['Premiums in LOB: Life'])
sns.boxplot(x=df_Merge['Premiums in LOB: Work Compensations'])
sns.boxplot(x=df_Merge['First Policy´s Year'])
sns.boxplot(x=df_Merge['Birthday Year'])
sns.boxplot(x=df_Merge['Educational Degree'])

# Calculate z value
z = np.abs(stats.zscore(df_Merge))
print(z)
# Define threshold
threshold = 3
print(np.where(z > 3))
# Only take the values where z < threshold
df_Merge_clean = df_Merge[(z < 3).all(axis=1)]

# How many values were dropped --> 400
df_Merge_clean.shape[0]
df_Merge.shape[0]

# Boxplots after cleaning
sns.boxplot(x=df_Merge_clean['Customer Identity'])
sns.boxplot(x=df_Merge_clean['Premiums in LOB: Motor'])
sns.boxplot(x=df_Merge_clean['Premiums in LOB: Household'])
sns.boxplot(x=df_Merge_clean['Premiums in LOB: Health'])
sns.boxplot(x=df_Merge_clean['Premiums in LOB: Life'])
sns.boxplot(x=df_Merge_clean['Premiums in LOB: Work Compensations'])
sns.boxplot(x=df_Merge_clean['First Policy´s Year'])
sns.boxplot(x=df_Merge_clean['Birthday Year'])
sns.boxplot(x=df_Merge_clean['Educational Degree'])


# IQR method --> doesn't work yet
#Q1 = df_Merge.quantile(0.25)
#Q3 = df_Merge.quantile(0.75)
#IQR = Q3 - Q1
#print(IQR)
#
#upper_bound = Q3 + 1.5 * IQR
#lower_bound = Q1 - 1.5 * IQR
#print(df_Merge < lower_bound |(df_Merge > upper_bound))

# Pearson correlations
corr=df_Merge_clean.corr(method='pearson')
corr_iloc = corr.iloc[1:,1:]

# Graph correlations
mask = np.zeros_like(corr_iloc)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr_iloc, mask=mask, vmax=.3, square=True)
    
# Select all pairs with absolute correlation > 0,3
attrs = corr_iloc
print(attrs)
threshold = 0.3
important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]) \
    .unstack().dropna().to_dict()
print(important_corrs)

# Unique important correlations
unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])), columns=['attribute pair', 'correlation'])
# sorted by absolute value
unique_important_corrs = unique_important_corrs.ix[
    abs(unique_important_corrs['correlation']).argsort()[::-1]]
print(unique_important_corrs)

# Why is policy year < birthday year?
Policy_val= df_Merge_clean["First Policy´s Year"]-df_Merge_clean["Birthday Year"]
Policy_value_error = df_Merge_clean[Policy_val<=0]

pd.DataFrame.hist(df_Merge_clean,figsize = (15,20))
#for column in df_Merge:
#    x, y = df_Merge['Customer Identity'], df_Merge['Birthday Year']
#    plt.scatter(x, y, alpha=0.5)
#    
#for i in  ('Premiums in LOB: Motor',
#       'Premiums in LOB: Household', 'Premiums in LOB: Health',
#       'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations'):
#    x, y = df_Merge['Customer Identity'], df_Merge[i]
#    plt.scatter(x, y, alpha=0.5)
#
#
#
#for i in ('Premiums in LOB: Motor',
#       'Premiums in LOB: Household', 'Premiums in LOB: Health',
#       'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations'):
#    sns.boxplot(x=df_Merge[i])


#

#df_Merge['Educational Degree'].replace({Null: 'None'})
# df_outlier = df_Merge.loc[df_Merge['Birthday Year'] < 1900]
#df_without = df_Merge[~df_Merge.index.isin([7195])]
#pd.DataFrame.hist(df_without, figsize = (15,20))


##ax = sns.heatmap(corr, annot=True, fmt="d")
#
#Insurance year < Birthday?
#Insurance premiums < 0 --> customer canceled the insurance
#Dimensionality reduction --> Policy first year (just kick it out, since correlation is not important), geographical living area
#IQR vs. Z to detect & remove outliers?
#NORMALIZATION?
# 1. Get data
# 2. Explore data
# 3. Modify data
# 4. Segment data: 2 segmentations --> one for LOB and one for Engage
# 5. Pivot table for the clusters: customers belong to 1/1 (Lob/Engage) or 1/2 (Lob/Engage)
# 6. Marketing: e.g. cross-selling. Don't discount on something they already have but discount other insurances
# NOrmalization: 
# z-score --> (value - mean)/constant
# minimax --> (value - min)/ max - min  --> not robust to outliers
# Remove outliers first, then normalize
#    
#    

    
