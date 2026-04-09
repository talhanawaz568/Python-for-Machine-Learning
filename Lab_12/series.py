import pandas as pd
# Creating a Series from a list
data = [10, 20, 30, 40, 50]
series = pd.Series(data)
print(series)


series_with_index = pd.Series(data, index=['a', 'b', 'c', 'd', 'e'])
print(series_with_index)
