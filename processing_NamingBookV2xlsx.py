import pandas as pd

## load in multiple sheets
housedata = pd.read_excel('NamingBookV2.xlsx', sheet_name='Houses', header=None, skiprows=1)
columndata = pd.read_excel('NamingBookV2.xlsx', sheet_name='Columns', header=None, skiprows=2)
alldata = pd.concat([housedata, columndata])
## disregard images/final col
alldata_subset = alldata[[0,1,2,3,4]]
## save to csv file
alldata_subset.to_csv('NamingBookV2.csv',index=False)