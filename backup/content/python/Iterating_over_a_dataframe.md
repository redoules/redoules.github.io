Title: Iterating over a DataFrame
Slug: Iterating_over_a_dataframe
Summary: Iterating over a Pandas DataFrame with a generator
Date: 2017-10-14 20:33  
Category: Python  
Tags: Data Wrangling
Authors: Guillaume Redoulès

### Create a sample dataframe


```python
# Import modules
import pandas as pd
```


```python
# Example dataframe

raw_data  = {'fruit': ['Banana', 'Orange', 'Apple', 'lemon', "lime", "plum"], 
        'color': ['yellow', 'orange', 'red', 'yellow', "green", "purple"], 
        'kcal': [89, 47, 52, 15, 30, 28]
    }

df = pd.DataFrame(raw_data, columns = ['fruit', 'color', 'kcal'])
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fruit</th>
      <th>color</th>
      <th>kcal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Banana</td>
      <td>yellow</td>
      <td>89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Orange</td>
      <td>orange</td>
      <td>47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Apple</td>
      <td>red</td>
      <td>52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>lemon</td>
      <td>yellow</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>lime</td>
      <td>green</td>
      <td>30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>plum</td>
      <td>purple</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>



### Using the iterrows method

Pandas DataFrames can return a generator with the iterrrows method. It can then be used to loop over the rows of the DataFrame




```python
for index, row in df.iterrows():
    print("At line {0} there is a {1} which is {2} and contains {3} kcal".format(index, row["fruit"], row["color"], row["kcal"]))
```

    At line 0 there is a Banana which is yellow and contains 89 kcal
    At line 1 there is a Orange which is orange and contains 47 kcal
    At line 2 there is a Apple which is red and contains 52 kcal
    At line 3 there is a lemon which is yellow and contains 15 kcal
    At line 4 there is a lime which is green and contains 30 kcal
    At line 5 there is a plum which is purple and contains 28 kcal
    
