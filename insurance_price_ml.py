import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_excel('Health_insurance_cost.xlsx')

# EDA
data.sample(10)

data.isnull().sum()

data.shape

sns.histplot(x='health_insurance_price', data=data, kde=True)

#postitively skewed distribution

# checking for 'age'-outliers
data["age"].plot(kind="box")

data.smoking_status.value_counts()

sns.set_style("whitegrid")

#box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='smoking_status', y='health_insurance_price', data=data)
plt.title('Effect of Smoking Status on Health Insurance Price')
plt.xlabel('Smoker Status')
#plt.xticks(ticks=[0, 1], labels=['Non-Smoker', 'Smoker'])
plt.ylabel('Health Insurance Price')
plt.tight_layout()
plt.show()

# histogram/density plot

plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='health_insurance_price', hue='smoking_status', kde=True, bins=30, alpha=0.6, palette='Set2')
plt.title('Effect of Smoking Status on Health Insurance Price')
plt.xlabel('Health Insurance Price')
plt.ylabel('Frequency')
#plt.legend(title='Smoker Status', labels=['Non-Smoker', 'Smoker'])
plt.tight_layout()
plt.show()

# Create a figure and set of subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Create scatter plots for each independent variable
sns.scatterplot(x='age', y='health_insurance_price', data=data, ax=axes[0, 0])
axes[0, 0].set_title('Age vs. Insurance Cost')

sns.boxplot(x='gender', y='health_insurance_price', data=data, ax=axes[0, 1])
axes[0, 1].set_title('Gender vs. Insurance Cost')

sns.scatterplot(x='BMI', y='health_insurance_price', data=data, ax=axes[0, 2])
axes[0, 2].set_title('BMI vs. Insurance Cost')

sns.boxplot(x='Children', y='health_insurance_price', data=data, ax=axes[1, 0])
axes[1, 0].set_title('Children vs. Insurance Cost')

sns.boxplot(x='location', y='health_insurance_price', data=data, ax=axes[1, 1])
axes[1, 1].set_title('Region vs. Insurance Cost')

# Remove the empty subplot
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()

# Data pre-processing
#fill null values of age (no outliers here) & BMI columns with mean

data['age']= data['age'].fillna(data['age'].mean())

data['BMI']= data['BMI'].fillna(data['BMI'].mean())

data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

data= pd.get_dummies(data, columns=['gender','smoking_status','location'], drop_first=True)

data.sample(5)

# determine error metrics /linear regression

X = data.drop('health_insurance_price', axis=1)
y = data['health_insurance_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1 - r2_score(y_test, y_pred)) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1)


print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Adjusted r2:", adjusted_r2)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Support Vector Regression': SVR()
}

# Train and evaluate models
results = {}
for name, model in models.items():      ## key, value
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse,'R-squared': r2}

# results
for name, result in results.items():
    print(f"Model: {name}")
    print(f"MAE: {result['MAE']}")
    print(f"MSE: {result['MSE']}")
    print(f"RMSE: {result['RMSE']}")
    print(f"R-squared: {result['R-squared']}")
    print()

# Visualize performance
scaler = MinMaxScaler()
metrics_df = pd.DataFrame(results).T    ## T-transposes the DataFrame, which swaps the rows and columns for visualization
metrics_df_scaled = pd.DataFrame(scaler.fit_transform(metrics_df), columns=metrics_df.columns, index=metrics_df.index)
metrics_df_scaled.plot(kind='bar', figsize=(10, 6),alpha=0.5)


plt.title('Performance Comparison of Regression Models')
plt.xlabel('Model')
plt.ylabel('Error Metrics')
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

""" Gradient Boosting model is the best one among the rest!"""



"""**Perform the necessary steps required to improve the accuracy of your model.**"""

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gbm = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(gbm, param_grid, cv=5, scoring='neg_mean_squared_error')   ##grid_search=model
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best GBM model
best_gbm = grid_search.best_estimator_

# Evaluate the best GBM model
y_pred_train = best_gbm.predict(X_train_scaled)
y_pred_test = best_gbm.predict(X_test_scaled)

train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Train R-squared Score:", train_r2)
print("Test R-squared Score:", test_r2)

# Perform cross-validation (5 fold)

cv_scores = cross_val_score(gbm, X, y, cv=5, scoring='neg_mean_squared_error') ## entire X,y given for validation

# Convert negative MSE scores to positive values
cv_scores = -cv_scores

# cross-validation scores
print("Cross-Validation Scores (MSE):", cv_scores)
print("Average MSE:", cv_scores.mean())
