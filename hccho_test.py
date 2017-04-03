#-*- coding: utf-8 -*-
import datetime

import pandas as pd
from pandas_datareader import data as web
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

def download_stock_data(file_name,company_code,year1,month1,date1,year2,month2,date2):
	start = datetime.datetime(year1, month1, date1)
	end = datetime.datetime(year2, month2, date2)
	df = web.DataReader("%s.KS" % (company_code), "yahoo", start, end)

	df.to_pickle(file_name)

	return df
 
def load_stock_data(file_name):
	df = pd.read_pickle(file_name)
	return df
 
 

#df = download_stock_data('Y:\\TeamMember\\hccho\\PythonTest\\MachineLearning\\lg.data','066570',2015,1,1,2017,2,28)
#df2 = download_stock_data('Y:\\TeamMember\\hccho\\PythonTest\\MachineLearning\\samsung.data','005930',2015,1,1,2017,2,28)


df = load_stock_data('Y:\\TeamMember\\hccho\\PythonTest\\MachineLearning\\lg.data')
df2 = load_stock_data('Y:\\TeamMember\\hccho\\PythonTest\\MachineLearning\\samsung.data')
df['Open'].plot()
plt.axhline(df['Open'].mean(),color='red')
plt.show()
print(df.describe())
print(df.quantile([0.25,0.5,0.75]))



(n, bins, patched) = plt.hist(df['Open'])
df['Open'].plot(kind='kde')
plt.axvline(df['Open'].mean(),color='red')
plt.show()

for index in range(len(n)):
	print ("Bin : %0.f, Frequency = %0.f" % (bins[index],n[index]))
 
 
scatter_matrix(df[['Open','High','Low','Close']], alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.show() 

df[['Open','High','Low','Close','Adj Close']].plot(kind='box')
plt.show() 

print("Cov: ", df['Close'].cov(df2['Close']))
print("Corr: ", df['Close'].corr(df2['Close']))