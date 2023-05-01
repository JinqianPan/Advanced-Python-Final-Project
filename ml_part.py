from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

sqf_data = pd.read_csv('./data/sqf_data.csv')

# Define predictor variables and target variable
X = sqf_data.drop(columns=['found_weapon', 'year'])
y = sqf_data['found_weapon']

# Define numeric and categorical features
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)])

# Create a pipeline with the preprocessor and logistic regression estimator
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Split the dataset into training and test seta
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Get the logistic regression model from the pipeline
logistic_reg = pipeline.named_steps['classifier']

# Get the feature names after one-hot encoding
feature_names = numeric_cols.tolist()
for col, cats in zip(categorical_cols, preprocessor.named_transformers_['cat'].categories_):
    feature_names.extend([f'{col}_{cat}' for cat in cats[1:]])  # [1:] is due to drop='first'

# Create a dataframe of coefficients
coef_df = pd.DataFrame(logistic_reg.coef_[0], index=feature_names, columns=['Coefficient'])
print(coef_df)