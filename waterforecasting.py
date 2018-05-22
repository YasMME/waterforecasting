import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt #to show, plt.show()
import tensorflow as tf

#read in csv file
data = pd.read_csv('Water_Consumption_And_Cost__2013_-_2017_.csv',
        header=0)

#drop unused columns
data = data.drop(['Development Name', 'Account Name', 'Location', 
    'Funding Source', 'Meter AMR', 'Meter Scope', 'TDS #', 'EDP', 
    'RC Code', 'AMP #', 'Vendor Name', 'UMIS BILL ID','Meter Number',
    'Estimated', 'Rate Class', 'Bill Analyzed', 'Other Charges'],
    axis=1)

#filter based on days of service
data = data.loc[(data['# days'] > 26) & (data['# days'] < 34)]

#combine all data points for a borough in a month

#get list of boroughs
boroughs = pd.Series(data['Borough'].values).unique()
#get revenue months
revenue_months = pd.Series(data['Revenue Month'].values).unique()

queens_data = pd.DataFrame()
for m in revenue_months:
    #get all data points for a month, sum
print(queens_data)




