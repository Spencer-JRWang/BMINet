import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeCV, Ridge, ElasticNetCV, ElasticNet, LinearRegression
from sklearn.preprocessing import LabelEncoder

def Lasso_Formula(df, disease_pairs=("A", "B"), cv=5, alpha=None):
    """
    Use LassoCV for feature selection and generate a classification formula.

    Parameters:
    df (pd.DataFrame): DataFrame containing disease labels and features, where the first column 
                       should be the disease label (0-1), and the remaining columns are features.
    disease_pairs (tuple): A tuple of two disease labels to filter the DataFrame. Only rows 
                           with these disease labels are considered.
    cv (int): Number of cross-validation folds for LassoCV.
    alpha (float or None): Regularization parameter for Lasso. If None, the best alpha found 
                           by LassoCV is used.

    Returns:
    str: A string representation of the classification formula based on selected features.
    """

    # Filter the DataFrame to include only rows with specified disease labels
    df = df[df['Disease'].isin(disease_pairs)]

    # Separate labels and features
    X = df.iloc[:, 1:]  # Features
    y = df.iloc[:, 0]   # Disease label
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df.iloc[:, 0]) * 100

    # Use LassoCV for feature selection with cross-validation
    lasso_cv = LassoCV(cv=cv, random_state=42)
    lasso_cv.fit(X, y)

    # Select features with non-zero coefficients
    selected_features = X.columns[(lasso_cv.coef_ != 0)]
    X_selected = X[selected_features]

    # Perform Lasso regression with the best alpha or provided alpha
    if alpha:
        lasso = Lasso(alpha=alpha)
    else:
        lasso = Lasso(alpha=lasso_cv.alpha_)
    lasso.fit(X_selected, y)

    # Generate the classification formula based on selected features
    formula = f"Disease = {lasso.intercept_:.4f} + " + " + ".join(
        f"{coef:.4f}*{name}" for coef, name in zip(lasso.coef_, selected_features)
    )
    return formula

def Ridge_Formula(df, disease_pairs=("A", "B"), cv=5, alpha=None):
    """
    Use RidgeCV for feature selection and generate a classification formula.
    """
    # Filter the DataFrame to include only rows with specified disease labels
    df = df[df['Disease'].isin(disease_pairs)]

    # Separate labels and features
    X = df.iloc[:, 1:]  # Features
    y = df.iloc[:, 0]   # Disease label
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df.iloc[:, 0]) * 100

    # Use RidgeCV for feature selection with cross-validation
    ridge_cv = RidgeCV(cv=cv)
    ridge_cv.fit(X, y)

    # Perform Ridge regression with the best alpha or provided alpha
    if alpha:
        ridge = Ridge(alpha=alpha)
    else:
        ridge = Ridge(alpha=ridge_cv.alpha_)
    ridge.fit(X, y)

    # Generate the classification formula
    formula = f"Disease = {ridge.intercept_:.4f} + " + " + ".join(
        f"{coef:.4f}*{name}" for coef, name in zip(ridge.coef_, X.columns)
    )
    return formula

def ElasticNet_Formula(df, disease_pairs=("A", "B"), cv=5, alpha=None, l1_ratio=0.5):
    """
    Use ElasticNetCV for feature selection and generate a classification formula.
    """
    # Filter the DataFrame to include only rows with specified disease labels
    df = df[df['Disease'].isin(disease_pairs)]

    # Separate labels and features
    X = df.iloc[:, 1:]  # Features
    y = df.iloc[:, 0]   # Disease label
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df.iloc[:, 0]) * 100

    # Use ElasticNetCV for feature selection with cross-validation
    elastic_cv = ElasticNetCV(cv=cv, l1_ratio=l1_ratio, random_state=42)
    elastic_cv.fit(X, y)

    # Perform ElasticNet regression with the best alpha or provided alpha
    if alpha:
        elastic = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    else:
        elastic = ElasticNet(alpha=elastic_cv.alpha_, l1_ratio=l1_ratio)
    elastic.fit(X, y)

    # Generate the classification formula
    selected_features = X.columns[(elastic.coef_ != 0)]
    formula = f"Disease = {elastic.intercept_:.4f} + " + " + ".join(
        f"{coef:.4f}*{name}" for coef, name in zip(elastic.coef_, X.columns) if coef != 0
    )
    return formula

def Linear_Formula(df, disease_pairs=("A", "B")):
    """
    Use Linear Regression to generate a classification formula.
    """
    # Filter the DataFrame to include only rows with specified disease labels
    df = df[df['Disease'].isin(disease_pairs)]

    # Separate labels and features
    X = df.iloc[:, 1:]  # Features
    y = df.iloc[:, 0]   # Disease label
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df.iloc[:, 0]) * 100

    # Use Linear Regression for feature selection
    linear_model = LinearRegression()
    linear_model.fit(X, y)

    # Generate the classification formula
    formula = f"Disease = {linear_model.intercept_:.4f} + " + " + ".join(
        f"{coef:.4f}*{name}" for coef, name in zip(linear_model.coef_, X.columns)
    )
    return formula