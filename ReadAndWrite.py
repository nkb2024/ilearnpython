import pandas as pd
# Read a table from a web page
dfs=pd.read_html('https://www.ussoccer.com/uswnt-stats')
#print(dfs[0].head())
#head is used to display the first 5 rows of the dataframe
# dfs[0].to_csv('uswnt-stats.csv')
# print("------------")
# print("uswnt-stats.csv created")
# print("------------")
# print(dfs[1].head())
#read_csv is used to read a csv file
gender = pd.read_csv('weight-height.csv')
#print(gender.head())
#loc is used to access a group of rows and columns by label(s) or a boolean array.
female= gender.loc[gender['Gender']=='Female']
#print(female)
cols = ['Name', 'Age', 'City']
data = [['Tom', 10, 'New York'], ['Nick', 15, 'California'], ['Juli', 14, 'Los Angeles']]
df2 = pd.DataFrame(data, columns=cols)
print(df2)
print("------------------------------------------------------------------------------------")
print(df2['Name']=='Tom')
print("--------------------------------------------------------------------------------------")
print(df2['Name'].min)
print("--------------------------------------------------------------------------------------")
print(df2['Name'].loc[2])
print("--------------------------------------------------------------------------------------")
print(df2['Name'].iloc[0])
print("--------------------------------------------------------------------------------------")
print(df2.tail(1))
print("--------------------------------------------------------------------------------------")
print(df2.describe())
print("--------------------------------------------------------------------------------------")
print(df2.dtypes)