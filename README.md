# Mileage-per-Gallon
1. Import Libraries: All the libraries used in this task need to be imported. These range from those useful in
data manipulation like pandas and numpy to those for visualization like matplotlib and seaborn and then to
those for machine learning like scikit-learn.
2. Load the Dataset: The script loads the auto-mpg dataset from a specified file path into a pandas DataFrame.
3. Initial Data Exploration:
 - It then displays the first few rows of the dataset.
 - Further, it shows summary statistics of each numeric column.
1. Dataset: Check for missing values within the dataset.
4. Cleaning of Data:
 • The horsepower column is to be converted into a numeric one, replacing the non-numeric entries with
NaN.
 • Fill the NaN horse power values with the mean of the column.
 • Create dummy variables for the categorical origin variable.
- The column of car names is useless in prediction; thus, drop it.
5. Data Visualization:
Pair plots to understand how the different features vary against the target variable, mpg, and compute and
visualize the correlation matrix to show relationships between variables.
6. Feature Selection: Finalize the following useful features: cylinders, displacement, horsepower, weight,
acceleration, model year, origin_2, and origin_3.
7. Train-Test Split: The data should be split into two different sets: training and testing, in a ratio of 80% vs.
20%.
8. Training the Model: A Random Forest Regressor shou
