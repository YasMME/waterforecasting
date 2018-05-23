import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt #to show, plt.show()
import tensorflow as tf
from dateutil.parser import parse
import datetime

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


monthly_data = pd.DataFrame(columns=('Month', 'Borough',
    'Consumption (HCF)'))
for b in boroughs:
    temp_borough = data.loc[(data['Borough'] == b)]
    for m in revenue_months:
        #strip extra date info
        d = parse(m, fuzzy=True)
        d = d.date() 

        #sum consumption at all locations in a borough
        monthly = temp_borough.loc[(data['Revenue Month'] == m)]
        total = monthly['Consumption (HCF)'].sum()

        monthly_data = monthly_data.append({'Month':d,
                'Borough':b, 'Consumption (HCF)':total},
                ignore_index=True)

#put data in chronological order
monthly_data = monthly_data.sort_values(by='Month', ascending=True)
