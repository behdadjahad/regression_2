import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler









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




# Load the CSV file
# Z = pd.read_csv("C:/proghe/New folder/p.csv")

# Extract the relevant columns
y = y.iloc[:1000, 1]
x1 = X.iloc[:1000, 2]
x2 = X.iloc[:1000, 3]
x3 = X.iloc[:1000, 4]

# Fit the linear model
X = sm.add_constant(x3)
model = sm.OLS(y, X).fit()
print(model.summary())

# ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# Variance-covariance matrix
vcov_matrix = model.cov_params()
print(vcov_matrix)

# Feature selection using stepwise regression (backward)
X = np.column_stack((x1, x2, x3))
y = np.array(y)

# Perform recursive feature elimination
estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=1, step=1)
selector = selector.fit(X, y)

print("Ranking of features: ", selector.ranking_)
print("Selected features: ", selector.support_)

# Correlation matrix
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

# Matrix for feature selection
Z = np.column_stack((x1, x2, x3, x1x2, x1x3, x2x3, x11, x22, x33))

# Linear model and VIF calculation
X = sm.add_constant(Z)
model = sm.OLS(y, X).fit()
print(model.summary())

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = ["const", "x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x11", "x22", "x33"]
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
print(vif_data)
