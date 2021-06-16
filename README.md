## Salary Prediction Linear Regression Model

I wanted to provide a quick introduction to building models in Python, and what better place to start than one of the very basic models, linear regression? This will be the first post about machine learning and I plan to write about more complex models in the future. Stay tuned! But for right now, let’s focus on linear regression.

#import your libraries
​
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
​
Import Dataset
salary_df = pd.read_csv('salary.csv')
salary_df['Salary'].max()
139465
isnull( )
NULL is a special marker used in SQL to indicate that a data value does not exist in the database. If the chart below was populated would indicate:"missing information and inapplicable information"

# check if there are any Null Values
sns.heatmap(salary_df.isnull(), yticklabels = False, cbar = False, cmap= "Blues")
<matplotlib.axes._subplots.AxesSubplot at 0x7ffe644d0b00>

#check the dataframe info
​
salary_df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 35 entries, 0 to 34
Data columns (total 2 columns):
YearsExperience    35 non-null float64
Salary             35 non-null int64
dtypes: float64(1), int64(1)
memory usage: 640.0 bytes
Describe ( )
Pandas describe( ) is used to view some basic statistical details like percentile, mean, std etc.

#Statistical summary of the dataframe
salary_df.describe()
YearsExperience	Salary
count	35.000000	35.000000
mean	6.308571	83945.600000
std	3.618610	32162.673003
min	1.100000	37731.000000
25%	3.450000	57019.000000
50%	5.300000	81363.000000
75%	9.250000	113223.500000
max	13.500000	139465.000000
Maximum ( )
The max( ) function returns the item with the highest value, or the item with the highest value in an iterable.

max = salary_df[salary_df['Salary'] == salary_df['Salary'].max()]
max = salary_df[salary_df['Salary'] == salary_df['Salary'].max()]
max
YearsExperience	Salary
34	13.5	139465
Minimum ( )
The min( ) function returns the item with the lowest value, or the item with the lowest value in an iterable.

in
min = salary_df[salary_df['Salary'] == salary_df['Salary'].min()]
min
min
YearsExperience	Salary
2	1.5	37731
Histogram
A histogram is a graphical display of data using bars of different heights.

In a histogram, each bar groups numbers into ranges. Taller bars show that more data falls in that range

salary_df.hist(bins = 30, figsize = (20,10), color = 'r')
array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7ffe645c9080>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x7ffe64697240>]],
      dtype=object)

  )
## Pair Plot (  )
​
A pairs plot allows us to see both distribution of single variables and relationships between two variables. 
​
Pair plots are a great method to identify trends for follow-up analysis and, fortunately, are easily implemented in Python!
#plot pairplot
​
sns.pairplot(salary_df)
<seaborn.axisgrid.PairGrid at 0x7ffe64248e80>

Correlation values
Correlation values range between -1 and 1.

There are two key components of a correlation value:

magnitude – The larger the magnitude (closer to 1 or -1), the stronger the correlation sign – If negative, there is an inverse correlation. If positive, there is a regular correlation.

#plotting correlations 
corr_matrix = salary_df.corr()
sns.heatmap(corr_matrix, annot = True)
plt.show()

sns.regplot(x = 'YearsExperience', y = 'Salary', data = salary_df)
/Users/sorushdovlatabadi/anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
<matplotlib.axes._subplots.AxesSubplot at 0x7ffe65dd5e80>

X = salary_df[['YearsExperience']]
y = salary_df[['Salary']]
Independent Variables
Independent variables (also referred to as Features) are the input for a process that is being analyzes. Dependent variables are the output of the process.

#Independent variable
X
YearsExperience
0	1.1
1	1.3
2	1.5
3	2.0
4	2.2
5	2.9
6	3.0
7	3.2
8	3.2
9	3.7
10	3.9
11	4.0
12	4.0
13	4.1
14	4.5
15	4.9
16	5.1
17	5.3
18	5.9
19	6.0
20	6.8
21	7.1
22	7.9
23	8.2
24	8.7
25	9.0
26	9.5
27	9.6
28	10.3
29	10.5
30	11.2
31	11.5
32	12.3
33	12.9
34	13.5
Dependent Variable
a variable (often denoted by y ) whose value depends on that of another.

#Dependent variable
#Dependent variable
y
Salary
0	39343
1	46205
2	37731
3	43525
4	39891
5	56642
6	60150
7	54445
8	64445
9	57189
10	63218
11	55794
12	56957
13	57081
14	61111
15	67938
16	66029
17	83088
18	81363
19	93940
20	91738
21	98273
22	101302
23	113812
24	109431
25	105582
26	116969
27	112635
28	122391
29	121872
30	127345
31	126756
32	128765
33	135675
34	139465
X.shape
X.shape
(35, 1)
shape
y.shape
(35, 1)
Float32
float32 Single precision float

Example:

'float' 58682.7578125

'numpy.float32' 58682.8

#Conversion of data into ('float32')
​
X = np.array(X).astype('float32')
y = np.array(y).astype('float32')
#only take the numberical variables and scale them
X
array([[ 1.1],
       [ 1.3],
       [ 1.5],
       [ 2. ],
       [ 2.2],
       [ 2.9],
       [ 3. ],
       [ 3.2],
       [ 3.2],
       [ 3.7],
       [ 3.9],
       [ 4. ],
       [ 4. ],
       [ 4.1],
       [ 4.5],
       [ 4.9],
       [ 5.1],
       [ 5.3],
       [ 5.9],
       [ 6. ],
       [ 6.8],
       [ 7.1],
       [ 7.9],
       [ 8.2],
       [ 8.7],
       [ 9. ],
       [ 9.5],
       [ 9.6],
       [10.3],
       [10.5],
       [11.2],
       [11.5],
       [12.3],
       [12.9],
       [13.5]], dtype=float32)
Test set & Training Set
The procedure involves taking a dataset and dividing it into two subsets.

The first subset is used to fit the model and is referred to as the training dataset. The second subset is not used to train the model; instead, the input element of the dataset is provided to the model, then predictions are made and compared to the expected values. This second dataset is referred to as the test dataset.

Train Dataset: Used to fit the machine learning model. Test Dataset: Used to evaluate the fit machine learning model. The objective is to estimate the performance of the machine learning model on new data: data not used to train the model.

This is how we expect to use the model in practice.

#split the data into test and train sets
from sklearn.model_selection import train_test_split
​
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# verify that the split was successful by obtaining the shape of both X_train and X_test
X_train.shape
(28, 1)
X_test.shape
X_test.shape
(7, 1)
#The order of the data gets shuffled naturally.
X_train
array([[ 9.6],
       [ 4.9],
       [13.5],
       [ 5.3],
       [ 9.5],
       [ 4. ],
       [11.2],
       [ 4. ],
       [10.5],
       [ 7.1],
       [ 2.9],
       [ 5.9],
       [12.3],
       [ 8.2],
       [ 4.5],
       [ 5.1],
       [10.3],
       [ 2. ],
       [ 4.1],
       [ 6.8],
       [ 3. ],
       [ 3.2],
       [ 9. ],
       [ 1.5],
       [11.5],
       [ 6. ],
       [ 7.9],
       [ 1.1]], dtype=float32)
X_test
array([[ 2.2],
       [ 3.9],
       [ 3.7],
       [ 8.7],
       [12.9],
       [ 3.2],
       [ 1.3]], dtype=float32)
fit_intercept ( )
Here is a link describing the fit_intercept function.

https://stackoverflow.com/questions/46779605/in-the-linearregression-method-in-sklearn-what-exactly-is-the-fit-intercept-par

#using Linear regrssion model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
​
#fit intercept (True) lets your graph have a y intercept, b parameter (y = mx + b)
#if intercept (False) forces b=0, force the line through the origine 
regression_model_sklearn = LinearRegression(fit_intercept = True)
regression_model_sklearn.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
         normalize=False)
Score ( )
Test accuracy of the model

regression_model_sklearn_accuracy = regression_model_sklearn.score(X_test, y_test)
regression_model_sklearn_accuracy
0.9675636612666972
y = mx + b Coefficients
Linear Regression Coefficients

After instantiating and fitting the model, use the .coef_ attribute to view the coefficients.

# find our coefficients m & b
print('Linear Model Coefficient (m): ', regression_model_sklearn.coef_)
print('Linear Model Coefficient (b): ', regression_model_sklearn.intercept_)
Linear Model Coefficient (m):  [[8847.595]]
Linear Model Coefficient (b):  [28075.777]
Predict ( )
The predict() function accepts only a single argument which is usually the data to be tested.

It returns the labels of the data passed as argument based upon the learned or trained data obtained from the model.

Thus, the predict() function works on top of the trained model and makes use of the learned label to map and predict the labels for the data to be tested.

 #evaluate trained model performance
y_predict = regression_model_sklearn.predict(X_test)
_predict
y_predict
array([[ 47540.484],
       [ 62581.4  ],
       [ 60811.88 ],
       [105049.84 ],
       [142209.75 ],
       [ 56388.08 ],
       [ 39577.65 ]], dtype=float32)
Chart
The assembly of the chart is self explanatory.

plt.scatter(X_train, y_train, color = 'gray')
plt.plot(X_train, regression_model_sklearn.predict(X_train), color = 'red')
plt.ylabel('Salary')
plt.xlabel('Number of Years of Experience')
plt.title('Salary vs. Years of Expereince')
Text(0.5, 1.0, 'Salary vs. Years of Expereince')

Predict ( )
Using our model to predict a salary with 5 years experience

num_years_exp = [[5]]
salary = regression_model_sklearn.predict(num_years_exp)
salary
array([[72313.75097656]])
