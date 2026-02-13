import numpy as np
import scipy.stats as stats
from scipy import linalg

class LinearRegression:
    def __init__(self, X, y):
        self.coefficients = None
        self.intercept = None

        #Parameters for the models
        self.X = None
        self.y = None
        self.b = None

        #Parameters for testing 
        self.n = None
        self.d = None
        self.df = None
        self.cov = None
        self.sigma2 = None
        self.y_hat = None
        self.var = None
        self.p = None
        
        #For F-test & R-squared
        self.SSR = None
        self.SSE = None
        self.Syy = None
        self.r2 = None
        self.fp_value = None

        #For T-test
        self.XtX_inv = None
        self.SE = None
        self.t_stats = None
        self.p_values = None


    def _add_intercept(self, X):
        one_column = np.ones((X.shape[0], 1))
        return np.hstack((one_column, X))

    def fit(self, X, y):
        # Setup Data
        X_raw = np.array(X, dtype=float) 
        y = np.array(y, dtype=float)
        
        # Capture Dimensions
        self.n = X_raw.shape[0]
        self.d = X_raw.shape[1]
        
        # Degrees of freedom for later: (n - d - 1)
        self.df = self.n - self.d - 1

        # Add Intercept
        # Transform X to include the 1s
        self.X = self._add_intercept(X_raw)
        
        # Store them for later (Metrics/Analysis)
        self.X = np.asanyarray(self.X, dtype=float)
        self.y = np.asanyarray(y, dtype=float)

        # The Normal Equation: b = (X.T * X)^-1 * X.T * Y
        Xt = self.X.T
        XtX = Xt @ self.X
        Xty = Xt @ self.y
        
        # Explicitly using inverse as per assignment formula
        try:
            self.XtX_inv = np.linalg.inv(XtX)
            self.b = self.XtX_inv @ Xty
        except np.linalg.LinAlgError:
            self.b = None
            raise ValueError("Error: Matrix is Singular (non-invertible).")

        # Store Components
        self.intercept = self.b[0]
        self.coefficients = self.b[1:]

        self._calculate_training_metrics()

    def _calculate_training_metrics(self):
        #Calculate predicted values
        self.y_hat = self.X @ self.b

        #Calculate Residuals
        residuals = self.y - self.y_hat

        #Calculate SSE Sum of Squares
        self.SSE = np.sum(residuals ** 2)

        #Syy Total Sum of Squares
        self.Syy = np.sum((self.y - np.mean(self.y)) ** 2)

        #SSR Sum of Squares Regression
        self.SSR = self.Syy - self.SSE

        #R-squared
        self.r2 = self.SSR / self.Syy

        #Variance of residuals (sigma^2)
        self.sigma2 = self.SSE / self.df


    def predict(self, X):
        X_raw = np.array(X, dtype=float)
        X_with_intercept = self._add_intercept(X_raw)
        return X_with_intercept @ self.b
    
    def sse_value(self):
        return self.SSE
    
    def ssy_value(self):
        return self.Syy
    
    def ssr_value(self):
        return self.SSR

    def R_squared(self):
        return self.r2

    def sigma_square(self):
        return self.sigma2
    
    def RMSE(self):
        return np.sqrt(self.sigma2)
    
    def standard_deviation_y(self):
        return np.sqrt(self.Syy / (self.n - 1))
    
    def variance_y(self):
        return self.Syy
    
    def sum_of_squares_regression(self):
        SSR = np.sum((self.y_hat - np.mean(self.y)) ** 2)
        self.SSR = SSR
        return SSR
    
    def r_squared(self):
        r2 = self.SSR / self.Syy
        self.r2 = r2
        return r2
    
    # Model error variance (sigma^2) is the unbiased estimator of the variance of the error term
    def model_error_variance(self):
        return self.SSE / self.df
    
    #Target variable variance (sigma_y^2) is the variance of the target variable y
    def target_variable_variance(self):
        return self.Syy / self.n - 1
    
    # F-test statistic for overall model significance: F = (SSR / d) / (SSE / (n - d - 1))
    
    def mean_squared_regression(self):
        return self.SSR / self.d
    
    def f_test_statistic(self):
        msr = self.mean_squared_regression()
        F = msr / (self.SSE / self.df)
        return F
    
    def f_p_value(self):
        F_stat = self.f_test_statistic()
        fp_value = stats.f.sf(F_stat, self.d, self.df)
        return fp_value
    
    # T-test for individual coefficients: t = (b_j - 0) / SE(b_j)
    def standard_error(self):
        # Calculate standard errors for each coefficient
        C = self.sigma2 * self.XtX_inv
        self.SE = np.sqrt(np.diagonal(C))
        return self.SE
        
    def t_statistics(self):
        if self.SE is None:
            self.standard_error()
        self.t_stats = self.b / self.SE
        return self.t_stats
    
    def t_p_values(self):
        if self.t_stats is None:
            self.t_statistics()
        self.p_values = 2 * stats.t.sf(np.abs(self.t_stats), df=self.df)
        return self.p_values
    
    def missing_percentage(self, df):
        # Calculate percentage of rows with at least one missing value
        rows_with_missing = len(df.loc[df.isnull().any(axis=1)].index)
        percent = 100 * (rows_with_missing / len(df))
        print("\nColumns with missing data:")
        missing_cols = df.isnull().sum()[df.isnull().sum() > 0]
        for col, count in missing_cols.items():
            print(f"{col} : {count} ({round(100 * count / len(df), 3)}%)")