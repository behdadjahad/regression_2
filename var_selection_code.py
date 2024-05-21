import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import itertools
from statsmodels.formula.api import ols

from sklearn.feature_selection import RFE




def ols_model_eval(X, y):

    ols_model = LinearRegression()
    ols_model.fit(X, y)


    # Calculate predictions
    y_pred_ols = ols_model.predict(X)
    
    return y_pred_ols

def ridge_model_eval(X, y):
    # Perform Ridge Regression
    alpha = 1.0  # Regularization strength
    ridge_reg = Ridge(alpha=alpha)
    ridge_reg.fit(X, y)

    y_pred_ridge = ridge_reg.predict(X)
    mse_ridge = mean_squared_error(y, y_pred_ridge)

    return y_pred_ridge, mse_ridge

def cal_adjusetd_r_squared(X, y, y_pred):

    n = X.shape[0]
    k = X.shape[1]

    # Calculate the coefficient of determination (R^2)
    r_squared = r2_score(y, y_pred)

    # Calculate the adjusted R^2
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    return adjusted_r_squared

def cal_cp(X, y, y_pred):

    n = X.shape[0]
    d = X.shape[1]
    # Calculate the sum of squared errors (SSE)
    SSE = np.sum((y - y_pred)**2)

    # Estimate the error variance (sigma^2)
    sigma_squared_hat = SSE / (n - d - 1)

    # Calculate Mallows' Cp statistic
    Cp = (1 / n) * (SSE + 2 * d * sigma_squared_hat)

    return Cp


def calculate_vif(X):
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


def backward_elimination(X, y, significance_level=0.05):
    """
    Perform backward elimination for feature selection in multiple linear regression.
    
    Parameters:
    - X: Independent variables (f# Display the first few rows of the DataFrame
print(california_df.head())

print(california_df['MedInc'])
eatures).
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


def print_data(name, X, y, header=False):
    y_pred = ols_model_eval(X, y)
    if header:
        print("features\t\tr2\t\t\tr2_d\t\t\tmse\t\t\tsse\t\t\tcp")
    r_square = r2_score(y, y_pred)
    r2_d = cal_adjusetd_r_squared(X, y, y_pred)
    mse = mean_squared_error(y, y_pred)
    sse = np.sum((y - y_pred)**2)
    cp = cal_cp(X, y, y_pred)
    
    string_res = f"{name}\t\t{r_square}\t{r2_d}\t{mse}\t{sse}\t{cp}"
    print(string_res)


data = np.loadtxt('/home/behdad/Desktop/workspace/regression_2/regression_2/CaliforniaHousing/cal_housing.data', delimiter=',')
domain = np.loadtxt('/home/behdad/Desktop/workspace/regression_2/regression_2/CaliforniaHousing/cal_housing.y', delimiter=',', dtype=str)

X_domain = domain[:-1]
y_domain = domain[-1]

missing_rows = np.isnan(data).any(axis=1)
# Remove rows with missing values
clean_data = data[~missing_rows]


scaler = StandardScaler()
std_data = scaler.fit_transform(clean_data)

# Separate features and target
X = std_data[:, :-1]  # Features (all columns except the last one)
y = std_data[:, -1]   # Target (last column)


x1 = X[:, 0:1]
x2 = X[:, 1:2]
x3 = X[:, 2:3]
# x4 = X[:, 3:4]

X = np.concatenate((x1, x2, x3), axis=1)


# Fit the linear model
model = sm.OLS(y, X).fit()

# Get the variance-covariance matrix
vcov_matrix = model.cov_params()
print("Variance-Covariance Matrix:\n", vcov_matrix)




# Calculate the correlation matrix
correlation_matrix = np.corrcoef(X, rowvar=False)
print("Correlation matrix:\n", correlation_matrix)

# Optionally, round the correlation matrix for better readability
rounded_correlation_matrix = np.round(correlation_matrix, 3)
print("Rounded correlation matrix:\n", rounded_correlation_matrix)


X1 = x1
X2 = x2
X3 = x3
X1_X2 = np.concatenate((x1, x2), axis=1)
X1_X3 = np.concatenate((x1, x3), axis=1)
X2_X3 = np.concatenate((x2, x3), axis=1)
X1_X2_X3 = np.concatenate((x1, x2, x3), axis=1)

print_data('x1', X1, y, True)
print_data('x2', X2, y)
print_data('x3', X3, y)
print_data('x1x2', X1_X2, y)
print_data('x1x3', X1_X3, y)
print_data('x2x3', X2_X3, y)
print_data('x1x2x3', X1_X2_X3, y)


print("\nForward selection:")
print(forward_selection(X, y))
print("\nBackward elimination:")
print(backward_elimination(X, y))

