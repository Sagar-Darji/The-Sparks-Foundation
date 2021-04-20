## The Sparks Foundation - Task no. 1 - Prediction using supervised Machine Learning
[![](https://img.shields.io/badge/author-@SagarDarji-blue.svg?style=flat)](https://www.linkedin.com/in/sagar-darji-7b7011165/)

## Problem Statement:  Predict the percentage of an student based on the no. of study hours.*

#### Objective - What should be the predicted score if student studies for 9.5 hours/day?

### Step no. 1. Importing the Data -
In this step, we will import all the required libraries and required dataset


```python
#Importing all required important libraries -
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
%matplotlib inline
import seaborn as sns
```


```python
#Importing the data from remote -
url = "http://bit.ly/w-data"
data = pd.read_csv(url) #Read the dataset in a variable (here it is "data")
```


```python
#View the data 
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.5</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.1</td>
      <td>47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.2</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.5</td>
      <td>75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.5</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>2.7</td>
      <td>30</td>
    </tr>
    <tr>
      <th>21</th>
      <td>4.8</td>
      <td>54</td>
    </tr>
    <tr>
      <th>22</th>
      <td>3.8</td>
      <td>35</td>
    </tr>
    <tr>
      <th>23</th>
      <td>6.9</td>
      <td>76</td>
    </tr>
    <tr>
      <th>24</th>
      <td>7.8</td>
      <td>86</td>
    </tr>
  </tbody>
</table>
</div>




```python
#For getting the information regarding rows and colunms 
data.shape
```




    (25, 2)




```python
#For more information about the given data 
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 25 entries, 0 to 24
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   Hours   25 non-null     float64
     1   Scores  25 non-null     int64  
    dtypes: float64(1), int64(1)
    memory usage: 528.0 bytes
    


```python
#For Statical Summary of the given data 
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>25.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.012000</td>
      <td>51.480000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.525094</td>
      <td>25.286887</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.100000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.700000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.800000</td>
      <td>47.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.400000</td>
      <td>75.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.200000</td>
      <td>95.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Check for any missing value in the given data
data.isnull().sum()
```




    Hours     0
    Scores    0
    dtype: int64



###  Step 2. Data Visualization - 

In this step, we visualize the given dataset and try to see if their any direct corelation exist between the two variables or not.


```python
# Plotting the distribution of scores
plt.rcParams["figure.figsize"]=[12,7]
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')
plt.grid()
plt.show()
```


![png](output_15_0.png)


*From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score. So, we can use the linear regression supervised machine learning model on it to predict the further values.*


```python
#We can also see corelation between the variables using corr()
data.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Hours</th>
      <td>1.000000</td>
      <td>0.976191</td>
    </tr>
    <tr>
      <th>Scores</th>
      <td>0.976191</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Step no. 3 **Preparing the data**

The next step is to divide the data into Train and Test data.


```python
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  
```


```python
X
```




    array([[2.5],
           [5.1],
           [3.2],
           [8.5],
           [3.5],
           [1.5],
           [9.2],
           [5.5],
           [8.3],
           [2.7],
           [7.7],
           [5.9],
           [4.5],
           [3.3],
           [1.1],
           [8.9],
           [2.5],
           [1.9],
           [6.1],
           [7.4],
           [2.7],
           [4.8],
           [3.8],
           [6.9],
           [7.8]])




```python
y
```




    array([21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30,
           24, 67, 69, 30, 54, 35, 76, 86], dtype=int64)



The next step is to split this data into training and test sets. This can be done by using Scikit-Learn's built-in train_test_split() method:


```python
#Split the dataset into train and test data within 70:30 Ratio
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 
```

### Step no. 4 **Training the Model**
We have split our data into training and testing sets, and now is finally the time to train our algorithm. 


```python
#Trained the Model
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
```




    LinearRegression()



### Step no. 5 **Visualize The Model**


```python
# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the trained data
plt.rcParams["figure.figsize"]=[12,7]
plt.scatter(X_train, y_train)
plt.plot(X, line, color='Orange');
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')
plt.grid()
plt.show()
```


![png](output_27_0.png)



```python
# Plotting for the test data
plt.rcParams["figure.figsize"]=[12,7]
plt.scatter(X_test, y_test)
plt.plot(X, line, color='Orange');
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')
plt.grid()
plt.show()
```


![png](output_28_0.png)


### Step no. 6 **Making Predictions**
Now that we have trained our algorithm, it's time to make some predictions.


```python
#Test the Model
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores
```

    [[1.5]
     [3.2]
     [7.4]
     [2.5]
     [5.9]
     [3.8]
     [1.9]
     [7.8]]
    


```python
# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>17.053665</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27</td>
      <td>33.694229</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69</td>
      <td>74.806209</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>26.842232</td>
    </tr>
    <tr>
      <th>4</th>
      <td>62</td>
      <td>60.123359</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35</td>
      <td>39.567369</td>
    </tr>
    <tr>
      <th>6</th>
      <td>24</td>
      <td>20.969092</td>
    </tr>
    <tr>
      <th>7</th>
      <td>86</td>
      <td>78.721636</td>
    </tr>
  </tbody>
</table>
</div>




```python
# You can also test with your own data
hours = 9.25
own_pred = regressor.predict([[hours]])
print("the predicted score if student studies for", hours, "hours/day is", own_pred[0])
```

    the predicted score if student studies for 9.25 hours/day is 92.91505723477056
    

### Step no. 7 **Evaluating the model**
The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error and mean Absolute error.


```python
from sklearn import metrics  
print('Mean Squared Error:', 
      metrics.mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 
```

    Mean Squared Error: 22.96509721270044
    Mean Absolute Error: 4.4197278080276545
    

***Task no. 1, Completed.
Thanks!***
