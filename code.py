# add libraries we need
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# no longer using
# import ipywidgets as ipy

# load the data from a raw string path (csv)
df = pd.read_csv('Population of all US Cities 2024.csv')

# Check data size/shape
df.shape

# Check for missing values
df.isnull().sum()

# Check the beginning of the table
df.head()

# show the table datatypes
df.dtypes

# gives descriptive statistics including those that summarize 
# the central tendency, dispersion and shape of a dataset’s distribution...
df.describe()

# make sure to include only number columns for correlation analysis
num_df = df.select_dtypes(include=[np.number])
cor_matrix = num_df.corr()

# plotting the map
plt.figure(figsize=(11, 8))
sns.heatmap(cor_matrix, annot=True, cmap='Spectral', linewidths=1.25)
plt.title('Correlation Heatmap')
plt.show()

# Highest populated cities for 2024
highest2024 = df.sort_values('Population 2024', ascending=False).head(10)
plt.title('Highest Populated Cities - 2024')
sns.barplot(x='Population 2024', y='US City', data=highest2024)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}M'.format(x / 1000000)))
plt.show()

# Highest density cities for 2024
highDensity2024 = df.sort_values('Density (mile2)', ascending=False).head(10)
plt.title('Highest Density Cities - 2024')
sns.barplot(x='Density (mile2)', y='US City', data=highDensity2024)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
plt.show()

# Lowest populated cities for 2024
lowest2024 = df.sort_values('Population 2024').head(10)
plt.title('Lowest Populated Cities - 2024')
sns.barplot(x='Population 2024', y='US City', data=lowest2024)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
plt.show()

# Lowest density cities for 2024
lowDensity2024 = df.sort_values('Density (mile2)').head(10)
plt.title('Lowest Density Cities - 2024')
sns.barplot(x='Density (mile2)', y='US City', data=lowDensity2024)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
plt.show()

# Relationship between pop and area
plt.title('Relationship between Population and Area')
sns.regplot(x='Area (mile2)', y='Population 2024', data=df)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}M'.format(x / 1000000)))
plt.show()

# Split into X and y
# Feature selection
X = df[['Population 2020', 'Population 2024', 'Density (mile2)', 'Area (mile2)']]
y = df['Annual Change']

# Split the data into training and testing sets
# random_state 4 = random seed to get same results for testing purposes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=214)

# KNeighborsRegressor
# Regression model
model = KNeighborsRegressor(
    algorithm="auto", 
    leaf_size=20, 
    metric="euclidean", 
    n_neighbors=2, 
    weights="distance")

# fit predict and score(R^2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model.score(X_test, y_test)

# Evaluate the model
# R^2 and mean squared eror
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot the results
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
plt.xlabel('Actual Annual Change')
plt.ylabel('Predicted Annual Change')
plt.title('Actual vs Predicted Annual Change')
plt.show()

# To create a prediction based on any arbitrary city 
# Please follow the instructions below

# 1. Please input the numbers with any valid numbers for the 2020 population, 2024 population, density, area, and prediction year (more below).
# 2. For accurate results, please take a look at the table at the top of the notebook.
# 3. Then please input a prediction year to get your desired population prediction for that year. 
# 4. The prediction year needs to be greater than the current year 2024.

pop2020 =  177693.5
pop2024 = 178620.5
density = 3340
area = 62
prediction_year = 2028

# used to generate population
years_of_growth = prediction_year - 2024
ppop = pop2024

# let's predict the 2024 population for input
# Setting up column names to avoid a warning for missing feature names
column_names = ['Population 2020', 'Population 2024', 'Density (mile2)', 'Area (mile2)']

# Creates a dataframe from a single element for input. 
Ynew = (model.predict(pd.DataFrame(np.array([[pop2020, pop2024, density, area]]), columns = column_names)))

# Loop for each year in years_of_growth 
for x in range(years_of_growth):
    ppop *= 1.0 + Ynew[0]

# Output the annual growth rate as a decimal and the population
print(f'Predicated annual growth rate: {Ynew}')
print(f'Predicated {prediction_year} population: {ppop}')
