import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from sklearn.decomposition import PCA
import statsmodels.api as sm
from scipy import stats
from scipy.stats import t
from sklearn.pipeline import Pipeline 

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
    corr_matrix = np.corrcoef(X, rowvar=False)
    inv_corr_matrix = np.linalg.inv(corr_matrix)
    vif = np.diag(inv_corr_matrix)
    return vif



def p_value_ridge_lasso(X, y, coef, alpha):

    n_bootstraps = 1000
    bootstrapped_coefs = np.zeros((n_bootstraps, X.shape[1]))

    for i in range(n_bootstraps):
        X_resampled, Y_resampled = resample(X, y)
        lasso_resampled = Lasso(alpha=alpha).fit(X_resampled, Y_resampled)
        bootstrapped_coefs[i, :] = lasso_resampled.coef_

    coef_means = np.mean(bootstrapped_coefs, axis=0)
    coef_se = np.std(bootstrapped_coefs, axis=0)

    t_stats = coef_means / coef_se

    df = n_bootstraps - 1  # degrees of freedom
    p_values = 2 * (1 - t.cdf(np.abs(t_stats), df))
    return p_values


def p_values_pcr(X, y, coef):
    n_bootstraps = 1000
    bootstrapped_coefs = np.zeros((n_bootstraps, X.shape[1]))

    for i in range(n_bootstraps):
        X_resampled, Y_resampled = resample(X, y)
        reg_resampled = LinearRegression().fit(X_resampled, Y_resampled)
        bootstrapped_coefs[i, :] = reg_resampled.coef_

    # Calculate the standard error of the coefficients
    coef_means = np.mean(bootstrapped_coefs, axis=0)
    coef_se = np.std(bootstrapped_coefs, axis=0)

    # Compute the t-statistic
    t_stats = coef_means / coef_se

    # Determine the p-values
    df = n_bootstraps - 1  # degrees of freedom
    p_values = 2 * (1 - t.cdf(np.abs(t_stats), df))
    return p_values

# Load the data
data = np.loadtxt('/home/behdad/Desktop/workspace/regression_2/regression_2/CaliforniaHousing/cal_housing.data', delimiter=',')



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
x4 = X[:, 3:4]
x5 = X[:, 4:5]
x6 = X[:, 5:6]
x7 = X[:, 6:7]
x8 = X[:, 7:8]
# x4 = X[:, 3:4]


X = np.concatenate((x1, x2, x3), axis=1)



# Building and fitting the model
model = sm.OLS(y, X).fit()
print(model.summary())

# Correlation matrix
correlation_matrix = np.corrcoef(X, rowvar=False)
print("Correlation matrix:\n", correlation_matrix.round(3))

# Variance Inflation Factor (VIF)
vif = calculate_vif(X)
print("VIF:\n", vif)

# Eigenvalues and Condition Number
G = np.dot(X.T, X)
eigenvalues, _ = np.linalg.eig(G)
print("Eigenvalues:\n", eigenvalues)

# Condition Number
lambdamax = max(eigenvalues)
lambdamin = min(eigenvalues)
condition_number = lambdamax / lambdamin
print("Condition Number:", condition_number)

# Durbin-Watson test for autocorrelation in residuals
dw = durbin_watson(model.resid)
print("Durbin-Watson statistic:", dw)

# Performing regression analysis for a single predictor as an example
ols_model = LinearRegression()
print(ols_model.fit(X, y))

# Durbin-Watson test for single predictor model
# dw_single = durbin_watson(model_single.resid)
# print("Durbin-Watson statistic for single predictor model:", dw_single)



#corr max
#VIF
#Ridge
# Choose a range of lambda values to test
alphas = np.logspace(-4, 4, 100)
# print("alphas:", alphas)
# Perform cross-validation to find the best lambda
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X, y)

# Get the best lambda
best_alpha = ridge_cv.alpha_
print("Best lambda:", best_alpha)
ridge = Ridge(alpha=best_alpha)
ridge.fit(X, y)
y_pred_ridge = ridge.predict(X)
print("ridge model r2 score:")
r2 = r2_score(y, y_pred_ridge)
print(r2)
print("ridge model 2r adjusted score:")
print(cal_adjusetd_r_squared(X, y, y_pred_ridge))
print("ridge model mse:")
print(mean_squared_error(y, y_pred_ridge))
print("ridge model cp:")
print(cal_cp(X, y, y_pred_ridge))

residuals = y - y_pred_ridge

# Degrees of freedom: n - p (number of observations - number of parameters)
n, p = X.shape
residual_variance = np.sum(residuals**2) / (n - p)

# Calculate the covariance matrix of the coefficients
cov_matrix = np.linalg.inv(X.T @ X) * residual_variance

# Calculate the standard errors of the coefficients
std_errors = np.sqrt(np.diag(cov_matrix))

# Calculate the t-statistics
t_stats = ridge.coef_ / std_errors

# Calculate the p-values
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p))
print("ridge p-val:")
print(p_values)
print("****************************")
ols = LinearRegression()
ols.fit(X, y)
y_pred_ols = ols.predict(X)
print("ols model r2 score:")
r2 = r2_score(y, y_pred_ols)
print(r2)
print("ols model 2r adjusted score:")
print(cal_adjusetd_r_squared(X, y, y_pred_ols))
print("ols model mse:")
print(mean_squared_error(y, y_pred_ols))
print("ols model cp:")
print(cal_cp(X, y, y_pred_ols))
print("ridge p_values:")
print(p_value_ridge_lasso(X, y, ridge.coef_, best_alpha))

print("****************")
# Perform cross-validation to find the best alpha
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X, y)

# Get the best alpha
best_alpha = lasso_cv.alpha_
print("Best alpha:", best_alpha)

# Fit the final model with the best alpha
final_lasso = Lasso(alpha=best_alpha)
final_lasso.fit(X, y)
# Predict and evaluate as before
y_pred_lasso = final_lasso.predict(X)
print("lasso model r2 score:")
r2 = r2_score(y, y_pred_lasso)
print(r2)
print("lasso model 2r adjusted score:")
print(cal_adjusetd_r_squared(X, y, y_pred_lasso))
print("lasso model mse:")
print(mean_squared_error(y, y_pred_lasso))
print("lasso model cp:")
print(cal_cp(X, y, y_pred_lasso))
print("lasso p-val:")
print(p_value_ridge_lasso(X, y, final_lasso.coef_, best_alpha))
print("*******************\n")
#Eigenvalue
#Lasso
#PCR
# Create a pipeline with PCA and linear regression 
pca = PCA(n_components=3)
reg = LinearRegression() 
pipeline = Pipeline(steps=[('pca', pca), 
                           ('reg', reg)]) 
pipeline.fit(X, y)
y_pred_pcr = pipeline.predict(X)
print("pcr model r2 score:")
r2 = r2_score(y, y_pred_pcr)
print(r2)
print("\npcr model 2r adjusted score:")
print(cal_adjusetd_r_squared(X, y, y_pred_pcr))
print("\npcr model mse:")
print(mean_squared_error(y, y_pred_pcr))
print("\npcr model cp:")
print(cal_cp(X, y, y_pred_pcr))
print("pcr p-val:")
print(p_values_pcr(X, y, reg.coef_))
print("\n*******************\n")
# Keep only the first six principal components 



#rho

e = y - y_pred_ols
sum_e2 = 0
sum_ee = 0
for i in range(len(e)):
    if i > 0:
        sum_ee = e[i] * e[i-1]
    sum_e2 += e[i]**2

rho = sum_ee / sum_e2
print("rho:", rho)
#Durbin-Watson


