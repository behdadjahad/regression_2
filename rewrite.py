import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


from statsmodels.formula.api import ols


# Load the CSV file
file_path_data = "/home/behdad/Desktop/workspace/regression_2/regression_2/CaliforniaHousing/cal_housing.data"
file_path_y = "/home/behdad/Desktop/workspace/regression_2/regression_2/CaliforniaHousing/cal_housing.y"
data = np.genfromtxt(file_path_data, delimiter=',')
y = np.genfromtxt(file_path_y, delimiter=',', dtype=str)
























# Extract the relevant columns
y = data[:1000, -1]
x1 = data[:1000, 1]
x2 = data[:1000, 2]
x3 = data[:1000, 3]

# Fit the linear model using formula API for ANOVA
# Create a dictionary for the data
data_dict = {'y': y, 'x3': x3}

# Convert dictionary to structured array
structured_data = np.core.records.fromarrays([y, x3], names='y,x3')

# Fit the model
model_formula = ols('y ~ x3', data=structured_data).fit()
print(model_formula.summary())

# ANOVA
anova_table = sm.stats.anova_lm(model_formula, typ=2)
print(anova_table)

# Variance-covariance matrix
vcov_matrix = model_formula.cov_params()
print(vcov_matrix)

# Create the design matrix for feature selection
Z = np.column_stack((x1, x2, x3))

# Perform recursive feature elimination
estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=1, step=1)
selector = selector.fit(Z, y)
print("Ranking of features: ", selector.ranking_)
print("Selected features: ", selector.support_)

# Calculate the inverse of the cross-product matrix
Z_transpose_Z_inv = np.linalg.inv(Z.T @ Z)
print(Z_transpose_Z_inv)

# Create matrix R for correlation calculations
R = np.column_stack((x1, x2, x3, y))
cor_matrix = np.corrcoef(R.T)
print(np.round(cor_matrix, 3))

# Additional features
TH = x1 * x2
TC = x1 * x3
CH = x3 * x2

x1x2 = (TH - np.mean(TH)) / np.sqrt(np.sum((TH - np.mean(TH)) ** 2))
x1x3 = (TC - np.mean(TC)) / np.sqrt(np.sum((TC - np.mean(TC)) ** 2))
x2x3 = (CH - np.mean(CH)) / np.sqrt(np.sum((CH - np.mean(CH)) ** 2))

x11 = (x1 ** 2 - np.mean(x1 ** 2)) / np.sqrt(np.sum((x1 ** 2 - np.mean(x1 ** 2)) ** 2))
x22 = (x2 ** 2 - np.mean(x2 ** 2)) / np.sqrt(np.sum((x2 ** 2 - np.mean(x2 ** 2)) ** 2))
x33 = (x3 ** 2 - np.mean(x3 ** 2)) / np.sqrt(np.sum((x3 ** 2 - np.mean(x3 ** 2)) ** 2))

# Matrix for extended feature selection
extended_Z = np.column_stack((x1, x2, x3, x1x2, x1x3, x2x3, x11, x22, x33))

# Fit the model with extended features
X = sm.add_constant(extended_Z)
extended_model = sm.OLS(y, X).fit()
print(extended_model.summary())

# Calculate VIF
vif_data = np.array([variance_inflation_factor(X, i) for i in range(X.shape[1])])
features = ["const", "x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x11", "x22", "x33"]
print("VIF Data:")
for feature, vif in zip(features, vif_data):
    print(f"{feature}: {vif}")

# Calculate the eigenvalues and the inverse of the G matrix
G = X.T @ X
eigenvalues = np.linalg.eigvals(G)
G_inv = np.linalg.inv(G)
print("Eigenvalues of G:", eigenvalues)
print("Inverse of G:", G_inv)






































# # Fit the linear model
# X = sm.add_constant(x3)
# model = sm.OLS(y, X).fit()
# print(model.summary())

# # ANOVA
# anova_table = sm.stats.anova_lm(model, typ=2)
# print(anova_table)

# # Variance-covariance matrix
# vcov_matrix = model.cov_params()
# print(vcov_matrix)

# # Create the design matrix for feature selection
# Z = np.column_stack((x1, x2, x3))

# # Perform leaps (all subsets regression)
# estimator = LinearRegression()
# selector = RFE(estimator, n_features_to_select=1, step=1)
# selector = selector.fit(Z, y)
# print("Ranking of features: ", selector.ranking_)
# print("Selected features: ", selector.support_)

# # Calculate the inverse of the cross-product matrix
# Z_transpose_Z_inv = np.linalg.inv(Z.T @ Z)
# print(Z_transpose_Z_inv)

# # Create matrix R for correlation calculations
# R = np.column_stack((x1, x2, x3, y))
# cor_matrix = np.corrcoef(R.T)
# print(np.round(cor_matrix, 3))

# # Additional features
# TH = x1 * x2
# TC = x1 * x3
# CH = x3 * x2

# x1x2 = (TH - np.mean(TH)) / np.sqrt(np.sum((TH - np.mean(TH)) ** 2))
# x1x3 = (TC - np.mean(TC)) / np.sqrt(np.sum((TC - np.mean(TC)) ** 2))
# x2x3 = (CH - np.mean(CH)) / np.sqrt(np.sum((CH - np.mean(CH)) ** 2))

# x11 = (x1 ** 2 - np.mean(x1 ** 2)) / np.sqrt(np.sum((x1 ** 2 - np.mean(x1 ** 2)) ** 2))
# x22 = (x2 ** 2 - np.mean(x2 ** 2)) / np.sqrt(np.sum((x2 ** 2 - np.mean(x2 ** 2)) ** 2))
# x33 = (x3 ** 2 - np.mean(x3 ** 2)) / np.sqrt(np.sum((x3 ** 2 - np.mean(x3 ** 2)) ** 2))

# # Matrix for extended feature selection
# extended_Z = np.column_stack((x1, x2, x3, x1x2, x1x3, x2x3, x11, x22, x33))

# # Fit the model with extended features
# X = sm.add_constant(extended_Z)
# extended_model = sm.OLS(y, X).fit()
# print(extended_model.summary())

# # Calculate VIF
# vif_data = np.array([variance_inflation_factor(X, i) for i in range(X.shape[1])])
# features = ["const", "x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x11", "x22", "x33"]
# print("VIF Data:")
# for feature, vif in zip(features, vif_data):
#     print(f"{feature}: {vif}")

# # Calculate the eigenvalues and the inverse of the G matrix
# G = X.T @ X
# eigenvalues = np.linalg.eigvals(G)
# G_inv = np.linalg.inv(G)
# print("Eigenvalues of G:", eigenvalues)
# print("Inverse of G:", G_inv)
