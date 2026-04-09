import pandas as pd

data_dict = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}

# Converting dictionary to DataFrame
df = pd.DataFrame(data_dict)
print(df)

name_column = df['Name']
print(name_column)

first_row = df.loc[0]
print(first_row)

# Modifying age of the first record
df.at[0, 'Age'] = 26
print(df)
# Adding a new column 'Country'
df['Country'] = ['USA', 'USA', 'USA']
print(df)
