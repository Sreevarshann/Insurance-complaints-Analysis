import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Load and prepare the dataset
file_path = '/Users/sreevarshansathiyamurthy/Downloads/Insurance_complaints__All_data.csv'
data = pd.read_csv(file_path, parse_dates=['Received date', 'Closed date'])

# Prepare data for logistic regression
data['Duration'] = (data['Closed date'] - data['Received date']).dt.days
X = data.drop(['Confirmed complaint', 'Received date', 'Closed date'], axis=1)
y = data['Confirmed complaint'].apply(lambda x: 1 if x == 'Yes' else 0)

# Set up preprocessing
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Create and train the model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])

# Split and fit data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test)

# Calculate accuracy metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print accuracy metrics
print("MODEL ACCURACY METRICS")
print("=====================")
print(f"Accuracy Score: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print("\n" + "="*50 + "\n")

# Time Series Forecasting
monthly_complaints = data.set_index('Received date').resample('ME').size()

# Fit SARIMAX model
model_ts = SARIMAX(monthly_complaints,
                   order=(2, 1, 2),
                   seasonal_order=(1, 1, 1, 12))
fitted_model = model_ts.fit(disp=False)

# Generate forecast
last_date = monthly_complaints.index[-1]
future_dates = pd.date_range(start=last_date, periods=25, freq='ME')[1:]
forecast_res = fitted_model.get_forecast(steps=24)
forecast_mean = forecast_res.predicted_mean
forecast_ci = forecast_res.conf_int()

# Create visualization
plt.figure(figsize=(15, 8))

# Plot historical data
plt.plot(monthly_complaints.index, monthly_complaints.values, 
         label='Historical Monthly Complaints', color='blue')

# Plot forecast
plt.plot(forecast_mean.index, forecast_mean.values, 
         label='Forecasted Complaints (2 Years)', 
         color='red', linestyle='--')

# Add confidence intervals
plt.fill_between(forecast_mean.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1],
                 color='red', alpha=0.2,
                 label='95% Confidence Interval')

# Enhance plot appearance
plt.title('Insurance Complaints Forecast for Next 2 Years (2025-2027)', fontsize=14, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Complaints', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize=10)
plt.xticks(rotation=45)

# Add forecast insights
avg_historical = monthly_complaints[-12:].mean()
avg_forecast = forecast_mean.mean()
trend_direction = "Increasing" if forecast_mean[-1] > forecast_mean[0] else "Decreasing"

# Add text box with insights and accuracy
textstr = f'Model Insights:\n' \
          f'Accuracy: {accuracy:.2%}\n' \
          f'Recent Avg: {avg_historical:.0f}\n' \
          f'Forecast Avg: {avg_forecast:.0f}\n' \
          f'Trend: {trend_direction}'
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8),
         verticalalignment='top', fontsize=10)

plt.tight_layout()
plt.show()

# Print forecast statistics
print("FORECAST STATISTICS")
print("==================")
print(f"Average monthly complaints (last 12 months): {avg_historical:.2f}")
print(f"Average monthly complaints (forecasted): {avg_forecast:.2f}")
print(f"Predicted trend: {trend_direction}")
print(f"Minimum forecasted complaints: {forecast_mean.min():.2f}")
print(f"Maximum forecasted complaints: {forecast_mean.max():.2f}")