"# Linear-Regression-model" 

Linear regression is a foundational statistical method used for modeling the relationship between a dependent variable (often denoted as \( Y \)) and one or more independent variables (often denoted as \( X \)). It assumes a linear relationship between the independent variables and the dependent variable. The main goal of linear regression is to find the best-fitting linear equation that describes the relationship between the variables.

The general form of a linear regression model with one independent variable is:

\[ Y = \beta_0 + \beta_1 X + \varepsilon \]

Where:
- \( Y \) is the dependent variable (or target variable) we want to predict.
- \( X \) is the independent variable (or predictor variable) that influences \( Y \).
- \( \beta_0 \) is the intercept, representing the value of \( Y \) when \( X = 0 \).
- \( \beta_1 \) is the slope, representing the change in \( Y \) for a one-unit change in \( X \).
- \( \varepsilon \) is the error term, representing the difference between the observed and predicted values of \( Y \).

The objective of linear regression is to estimate the values of the coefficients \( \beta_0 \) and \( \beta_1 \) that minimize the sum of squared differences between the observed and predicted values of the dependent variable. This process is often referred to as "fitting the model" to the data.

Linear regression can be extended to multiple independent variables, resulting in a multiple linear regression model:

\[ Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_n X_n + \varepsilon \]

Where:
- \( X_1, X_2, \ldots, X_n \) are the independent variables.
- \( \beta_0 \) is the intercept.
- \( \beta_1, \beta_2, \ldots, \beta_n \) are the coefficients corresponding to each independent variable.
- \( \varepsilon \) is the error term.

The coefficients \( \beta_0, \beta_1, \ldots, \beta_n \) are estimated using various optimization techniques, such as ordinary least squares (OLS), which minimizes the sum of squared residuals.

Linear regression is widely used in various fields, including economics, finance, social sciences, and machine learning. It serves as the basis for more complex regression techniques and provides valuable insights into the relationships between variables.

Steps Involve to build the Linear regression model:
After Importing necessary libraries:

Sure, let's go through each of these tasks in more detail:

1. **Importing a dataset (advertising) from a CSV file:**
   - This task involves using pandas' `read_csv()` function to load the data from a CSV file named 'Advertising.csv' into a DataFrame called `advertising`.
   - The `advertising` DataFrame will contain the dataset, which presumably includes information about advertising spending on different media channels and corresponding sales figures.

2. **Observing the dataset and its properties using methods like info() and describe():**
   - After loading the dataset, the `info()` method is used to get a concise summary of the DataFrame, including information about the data types, number of non-null values, and memory usage.
   - The `describe()` method provides descriptive statistics for numerical columns in the DataFrame, such as count, mean, standard deviation, minimum, and maximum values.

3. **Visualizing the dataset using pairplots and heatmaps:**
   - Pairplots are created using seaborn's `pairplot()` function to visualize pairwise relationships between variables. In this case, pairplots are generated for the variables 'TV', 'Radio', 'Newspaper', and 'Sales' (target variable).
   - Heatmaps are generated using seaborn's `heatmap()` function to visualize the correlation matrix between variables. This helps in understanding the correlation between predictor variables and their influence on the target variable.

4. **Analyzing multicollinearity among predictor variables:**
   - Multicollinearity refers to the presence of high correlations between predictor variables in a regression model. To analyze multicollinearity, the variance inflation factor (VIF) is calculated for each predictor variable using the `variance_inflation_factor()` function from the `statsmodels.stats.outliers_influence` module.
   - The VIF values are then examined to identify any predictors with high multicollinearity, which may require further investigation or elimination from the model.

5. **Splitting the dataset into training and testing subsets:**
   - The dataset is split into training and testing subsets using scikit-learn's `train_test_split()` function. The `train_size` parameter specifies the proportion of the dataset to include in the training set, while the remaining data is allocated to the test set.
   - The split is performed on the predictor variables ('TV', 'Radio', 'Newspaper') and the target variable ('Sales').

6. **Scaling the numerical features:**
   - Numerical features are scaled using scikit-learn's `StandardScaler()` to ensure that all features have a similar scale. This is important for algorithms that are sensitive to the scale of the features, such as linear regression.
   - The `fit_transform()` method is applied to the training data to compute the mean and standard deviation used for scaling, while the `transform()` method is applied to the test data using the same parameters obtained from the training set.

7. **Training and evaluating a simple linear regression model using only the 'TV' feature:**
   - A simple linear regression model is instantiated using scikit-learn's `LinearRegression()` class.
   - The model is trained using the training data, where the predictor variable is 'TV' and the target variable is 'Sales'.
   - The trained model is then used to make predictions on both the training and testing data.
   - Evaluation metrics such as R-squared score are calculated to assess the performance of the model on both training and testing data.

8. **Training and evaluating a multiple linear regression model using all three features ('TV', 'Radio', 'Newspaper'):**
   - Similar to the previous step, a multiple linear regression model is instantiated and trained using all three predictor variables ('TV', 'Radio', 'Newspaper') and the target variable ('Sales').
   - The model is evaluated using the same evaluation metrics to compare its performance with the simple linear regression model.

9. **Plotting the regression lines and scatter plots for visualization:**
   - Scatter plots are created to visualize the relationship between predictor variables and the target variable.
   - Regression lines are plotted on the scatter plots to depict the relationship between predictor variables and the target variable as modeled by the linear regression models.
   - This visualization helps in understanding how well the models fit the data and how they predict the target variable based on the predictor variables.
   
