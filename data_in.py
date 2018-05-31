#read in water data from CSV file
import numpy as np
import pandas as pd
from dateutil.parser import parse
import datetime

def data_in():
    data = pd.read_csv('Water_Consumption_And_Cost__2013_-_2017_.csv',
            header=0)

    #filter based on days of service
    data = data.loc[(data['# days'] > 26) & (data['# days'] < 34)]

    #get list of boroughs
    boroughs = pd.Series(data['Borough'].values).unique()
    #get revenue months
    revenue_months = pd.Series(data['Revenue Month'].values).unique()

    ### Create a dictionary of DataFrames, keyed by Borough ###
    frames_by_borough = {}
    for b in boroughs:
        temp_borough = data.loc[(data['Borough'] == b)]
        monthly_data = pd.DataFrame(columns=('Month', 'Consumption (HCF)'))
        for m in revenue_months:
            #convert string to date, strip extra info
            d = parse(m, fuzzy=True)
            d = d.date()
            #sum consumption at all locations in the borough
            monthly = temp_borough.loc[(data['Revenue Month'] == m)]
            total = monthly['Consumption (HCF)'].sum()
            monthly_data = monthly_data.append({'Month':d,
                    'Consumption (HCF)':total},
                    ignore_index=True)
            monthly_data = monthly_data.sort_values(by='Month',
                    ascending=True)
        frames_by_borough[b] = monthly_data
    return frames_by_borough
