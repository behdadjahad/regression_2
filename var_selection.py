# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

from sklearn.metrics import r2_score


def ols_model_eval(X, y, mse=True):

    ols_model = LinearRegression()
    ols_model.fit(X, y)


    # Calculate predictions
    y_pred_ols = ols_model.predict(X)

    mse_ols = np.mean((y - y_pred_ols)**2)
    if mse:    
        print(f'Mean Squared Error of OLS: {mse_ols}')
    
    return y_pred_ols, ols_model.coef_.T, mse_ols

def cal_r2_score(y, y_pred):
    return r2_score(y, y_pred)


def backward_elimination(X, y, significance_level=0.05):
    """
    Perform backward elimination for feature selection in multiple linear regression.
    
    Parameters:
    - X: Independent variables (features).
    - y: Dependent variable (target).
    - significance_level: Threshold p-value for feature removal (default is 0.05).
    
    Returns:
    - selected_features: List of selected features.
    """
    n_features = X.shape[1]
    selected_features = list(range(n_features))
    
    while len(selected_features) > 0:
        X_subset = X[:, selected_features]
        
        # Fit a linear regression model using the selected features
        X_subset = sm.add_constant(X_subset)  # Add a constant term for the intercept
        model = sm.OLS(y, X_subset).fit()
        
        # Get the p-values for the coefficients
        p_values = model.pvalues[1:]  # Exclude the constant term
        
        # Find the feature with the highest p-value
        max_p_value = np.max(p_values)
        max_p_value_index = np.argmax(p_values)
        
        # Check if the highest p-value is above the significance level
        if max_p_value > significance_level:
            # Remove the feature with the highest p-value
            selected_features.pop(max_p_value_index)
        else:
            # Stop the elimination process if all remaining features have p-values below the significance level
            break
    
    return selected_features

def forward_selection(X, y):
    """
    Perform forward selection for feature selection in multiple linear regression.
    
    Parameters:
    - X: Independent variables (features).
    - y: Dependent variable (target).
    
    Returns:
    - selected_features: List of selected features.
    """
    n_features = X.shape[1]
    selected_features = []
    remaining_features = list(range(n_features))
    
    while remaining_features:
        best_score = -np.inf
        best_feature = None
        
        for feature in remaining_features:
            candidate_features = selected_features + [feature]
            X_subset = X[:, candidate_features]
            
            # Fit a linear regression model using the candidate features
            model = LinearRegression()
            model.fit(X_subset, y)
            
            # Evaluate the model using some performance metric (e.g., R^2 score)
            score = model.score(X_subset, y)
            
            # Update the best feature if the score improves
            if score > best_score:
                best_score = score
                best_feature = feature
        
        # Add the best feature to the list of selected features
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        
    return selected_features

def cal_adjusted_r2(X, r_squared):

    # adjusted R2

    # Get the number of observations (n) and number of predictors (k)
    n = X.shape[0]
    k = X.shape[1]

    # Calculate the adjusted R^2
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    return adjusted_r_squared


def cal_cp(X, y, y_pred):
    
    n = X.shape[0]
    d = X.shape[1]
    # Mallows ’ s Cp Statistic

    # Calculate the sum of squared errors (SSE)
    SSE = np.sum((y - y_pred)**2)

    # Estimate the error variance (sigma^2)
    sigma_squared_hat = SSE / (n - d - 1)

    # Calculate Mallows' Cp statistic
    Cp = (1 / n) * (SSE + 2 * d * sigma_squared_hat)

    return Cp

def cal_vif(X):
    """
    Calculate the Variance Inflation Factor (VIF) for each feature in X.
    
    Parameters:
    - X: DataFrame containing the independent variables (features).
    
    Returns:
    - vif: Series containing the VIF for each feature.
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data

# Load the Boston Housing Dataset
housing = fetch_california_housing()
# Create a DataFrame from the dataset
housing_df = pd.DataFrame(data=housing.data, columns=housing.feature_names)

# Add the target variable to the DataFrame
housing_df['MEDV'] = housing.target
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Display the first few rows of the DataFrame
print(housing_df.head())

# Statistical summary of the dataset
print(housing_df.describe())


# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(housing_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Visualize the distribution of target variable (MEDV)
plt.figure(figsize=(10, 6))
sns.histplot(housing_df['MEDV'], kde=True, color='blue', bins=30)
plt.title('Distribution of MEDV')
plt.xlabel('MEDV')
plt.ylabel('Frequency')
plt.show()
