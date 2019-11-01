# -*- coding: utf-8 -*-
"""
Created on Fri Feb 08 09:40:53 2019

@author: Stephen
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

#------------------------------------------------------------------------------
#df = pd.read_csv('D:\\Python Scripts\\managers vs analyst\\cosine_distance_df_bas_big_aff_5.csv')
#data = pd.read_csv('D:\\Python Scripts\\earnings_drift\\preliminary_tests_clean_qtrly_ret.csv')
data = pd.read_csv('D:\\Python Scripts\\earnings_drift\\preliminary_test_90_lagprc.csv')

# load in and structure returns data
ff5 = pd.read_csv('D:\\programs\\Dropbox\\python_scripts\\recursive_component\\data\\ff5_monthly.csv', index_col=0)
mom = pd.read_csv('D:\\programs\\Dropbox\\python_scripts\\recursive_component\\data\\momentum.csv', index_col=0)

def summary(p):
    s = p.describe().T
    s['tstat'] = s['mean']/(s['std']/np.sqrt(s['count']))
    return s[['mean','std','tstat']].T

#------------------------------------------------------------------------------
data['booklev'] = (data['dlttq'] + data['dlcq'] ) / data['atq']
data['cashflow'] = (data['ibq'] + data['dpq']) / (data['atq'] - data['cheq'])
data['mtb'] = (data['prccq'] * data['cshoq']) / (data['atq'] - data['ltq'])
data['size'] = np.log(data['prccq'] * data['cshoq'])

qtr_list = []
for i in range(len(data)):
    qtr = str(data['year'][i]) + str(data['fqtr'][i])
    qtr_list.append(int(qtr))
data['qtr'] = qtr_list

#data['qtr'] = data['qtr_x']
positive = data[data['surprise'] > 0]
negative = data[data['surprise'] < 0]

# Quintiles
buckets = lambda x: pd.Series(pd.qcut(x,5,labels=False), index=x.index)
data['quintile'] = data.groupby('qtr')['cd_mt_at_list'].apply(buckets)
positive['quintile'] = positive.groupby('qtr')['cd_mt_at_list'].apply(buckets)
negative['quintile'] = negative.groupby('qtr')['cd_mt_at_list'].apply(buckets)
# Terciles
buckets = lambda x: pd.Series(pd.qcut(x,[.0,.3,.7,1.0],labels=False), index=x.index)
data['squint'] = data.groupby('qtr')['surprise'].apply(buckets)
positive['squint'] = positive.groupby('qtr')['surprise'].apply(buckets)
negative['squint'] = negative.groupby('qtr_x')['surprise'].apply(buckets)

#------------------------------------------------------------------------------
# TABLE 6
# book leverage
table = data.pivot_table(index='qtr', columns='quintile', values='booklev')
table.columns = ['info_' + str(int(x)) for x in table.columns]
table['spread'] = table['info_0'] - table['info_4']
t = summary(table)
# cashflow
table = data.pivot_table(index='qtr', columns='quintile', values='cashflow')
table.columns = ['info_' + str(int(x)) for x in table.columns]
table['spread'] = table['info_0'] - table['info_4']
t = summary(table)
# market-to-book
table = data.pivot_table(index='qtr', columns='quintile', values='mtb')
table.columns = ['info_' + str(int(x)) for x in table.columns]
table['spread'] = table['info_0'] - table['info_4']
t = summary(table)
# ln(market cap)
table = data.pivot_table(index='qtr', columns='quintile', values='size')
table.columns = ['info_' + str(int(x)) for x in table.columns]
table['spread'] = table['info_0'] - table['info_4']
t = summary(table)

#------------------------------------------------------------------------------
# TABLE 7
# quarterly excess returns
table = data.pivot_table(index='qtr', columns='quintile', values='ex_ret')
table.columns = ['info_' + str(int(x)) for x in table.columns]
table['spread'] = table['info_0'] - table['info_4']
t = summary(table*100.0)
# drift
table = data.pivot_table(index='qtr', columns='quintile', values='DRIFT')
table.columns = ['info_' + str(int(x)) for x in table.columns]
table['spread'] = table['info_0'] - table['info_4']
t = summary(table*100.0)

#------------------------------------------------------------------------------
# TABLE 8
# regression analysis
merged = table*100.0
ports = ff5.join(mom).reset_index()
ports = ports.merge(data[['yyyymm','qtr']], how='inner', left_on='date', right_on='yyyymm')
del ports['yyyymm'], ports['date']
ports = ports.groupby('qtr').mean()
merged = merged.join(ports)
merged['MOM'] = merged['Mom   ']
del merged['Mom   ']
merged['exmkt'] = merged['Mkt-RF']
merged['mkt'] = merged['exmkt'] + merged['RF']
del merged['Mkt-RF']

reg = smf.ols('spread ~ exmkt', data=merged).fit(cov_type='HAC', cov_kwds={'maxlags':4})
reg = smf.ols('spread ~ exmkt + SMB + HML', data=merged).fit(cov_type='HAC', cov_kwds={'maxlags':4})
reg = smf.ols('spread ~ exmkt + SMB + HML + RMW + CMA', data=merged).fit(cov_type='HAC', cov_kwds={'maxlags':4})
reg = smf.ols('spread ~ exmkt + SMB + HML + RMW + CMA + MOM', data=merged).fit(cov_type='HAC', cov_kwds={'maxlags':4})
# Annualize the monthly Alpha
print(reg.summary())
alpha = reg.params[0] / 100
print('Annualize Alpha: ' + str((((1 + alpha)**(4)) - 1)*100) + ' with t-stat: ' + str(reg.tvalues[0]))


#------------------------------------------------------------------------------
# TABLE 9 (8 and 9 from old draft)
table = positive.pivot_table(index='qtr', columns=['quintile','squint'], values='DRIFT')
# calculate spread for low_surp/low_info - low_surp/high_info
table['lsli_lshi'] = (table.iloc[:, table.columns.get_level_values(0)==0][0][0]) - (table.iloc[:, table.columns.get_level_values(0)==4][4][0])
# spread for high_surp/low_info - high_surp/high_info
table['hsli_hshi'] = (table.iloc[:, table.columns.get_level_values(0)==0][0][2]) - (table.iloc[:, table.columns.get_level_values(0)==4][4][2])
# spread for high_surp/low_info - low_surp/low_info
table['hsli_lsli'] = (table.iloc[:, table.columns.get_level_values(0)==0][0][2]) - (table.iloc[:, table.columns.get_level_values(0)==0][0][0])
# spread for high_surp/high_info - low_surp/high_info
table['hshi_lshi'] = (table.iloc[:, table.columns.get_level_values(0)==4][4][2]) - (table.iloc[:, table.columns.get_level_values(0)==4][4][0])
t = summary(table*100)

table = negative.pivot_table(index='qtr', columns=['quintile','squint'], values='DRIFT')
# calculate spread for low_surp/low_info - low_surp/high_info
table['lsli_lshi'] = (table.iloc[:, table.columns.get_level_values(0)==0][0][0]) - (table.iloc[:, table.columns.get_level_values(0)==4][4][0])
# spread for high_surp/low_info - high_surp/high_info
table['hsli_hshi'] = (table.iloc[:, table.columns.get_level_values(0)==0][0][2]) - (table.iloc[:, table.columns.get_level_values(0)==4][4][2])
# spread for high_surp/low_info - low_surp/low_info
table['hsli_lsli'] = (table.iloc[:, table.columns.get_level_values(0)==0][0][2]) - (table.iloc[:, table.columns.get_level_values(0)==0][0][0])
# spread for high_surp/high_info - low_surp/high_info
table['hshi_lshi'] = (table.iloc[:, table.columns.get_level_values(0)==4][4][2]) - (table.iloc[:, table.columns.get_level_values(0)==4][4][0])
t = summary(table*100)






















