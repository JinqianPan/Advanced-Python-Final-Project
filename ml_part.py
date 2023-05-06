import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


sqf_data = pd.read_csv('./data/sqf_data.csv')


# Define predictor variables and target variable
X = sqf_data.drop(columns=['found_weapon', 'year'])
y = sqf_data['found_weapon']

# Define numeric and categorical features
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

# Define class weight
class_weights = {0: 1, 1: len(sqf_data[sqf_data.found_weapon == 0])/len(sqf_data[sqf_data.found_weapon == 1])}


# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)])

# Create a pipeline with the preprocessor and logistic regression estimator
logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight=class_weights))
])

# Create a pipeline with the preprocessor and random forest estimator
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight=class_weights))
])

# Create a pipeline with the preprocessor and XGBoost estimator
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(scale_pos_weight=class_weights[0]/class_weights[1]))
])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the logistic regression pipeline to the training data
logreg_pipeline.fit(X_train, y_train)

# Get the logistic regression model from the pipeline
logistic_reg = logreg_pipeline.named_steps['classifier']

# Get the feature names after one-hot encoding
feature_names = numeric_cols.tolist()
for col, cats in zip(categorical_cols, preprocessor.named_transformers_['cat'].categories_):
    feature_names.extend([f'{col}_{cat}' for cat in cats[1:]])  # [1:] is due to drop='first'

# Create a dataframe of logistic regression coefficients
logreg_coef_df = pd.DataFrame(logistic_reg.coef_[0], index=feature_names, columns=['Coefficient'])
print('Logistic Regression Coefficients:\n', logreg_coef_df)

# Fit the random forest pipeline to the training data
rf_pipeline.fit(X_train, y_train)

# Get the random forest model from the pipeline
random_forest = rf_pipeline.named_steps['classifier']

# Get feature importances from the random forest model
rf_feature_importances = pd.DataFrame(random_forest.feature_importances_, index=feature_names, columns=['Importance'])
print('Random Forest Feature Importances:\n', rf_feature_importances)

# Fit the XGBoost pipeline to the training data
xgb_pipeline.fit(X_train, y_train)

# Get the XGBoost model from the pipeline
xgb_model = xgb_pipeline.named_steps['classifier']

# Get feature importances from the XGBoost model
xgb_feature_importances = pd.DataFrame(xgb_model.feature_importances_, index=feature_names, columns=['Importance'])
print('XGBoost Feature Importances:\n', xgb_feature_importances)

# Generate predictions using each model and calculate accuracy
logreg_pred = logreg_pipeline.predict(X_test)
logreg_eval = classification_report(y_test, logreg_pred)
print('Logistic Regression report:', logreg_eval)

rf_pred = rf_pipeline.predict(X_test)
rf_eval = classification_report(y_test, rf_pred)
print('RFM report:', rf_eval)

xgb_pred = xgb_pipeline.predict(X_test)
xgb_eval = classification_report(y_test, xgb_pred)
print('Xgboost report:', rf_eval)
