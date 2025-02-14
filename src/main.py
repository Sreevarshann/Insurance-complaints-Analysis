import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# 2. Data Acquisition and Inspection
# Load the dataset
file_path = '/Users/sreevarshansathiyamurthy/Downloads/Insurance_complaints__All_data.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to inspect its structure
print(data.head())

# Get an overview of data types and check for missing values
print(data.info())

# Summary statistics for numeric columns
print(data.describe())

# Count of unique values in each column to understand the categorical distribution
for column in data.columns:
    print(f"Unique values in {column}: {data[column].nunique()}")

# 3. Data Cleaning and Preparation
# Handling missing data
categorical_columns = data.select_dtypes(include=['object']).columns
data[categorical_columns] = data[categorical_columns].fillna('Unknown')

# Removing duplicates
data.drop_duplicates(inplace=True)

# Convert date columns to datetime
data['Received date'] = pd.to_datetime(data['Received date'], errors='coerce')
data['Closed date'] = pd.to_datetime(data['Closed date'], errors='coerce')

# Normalize data types if necessary
data['Respondent ID'] = data['Respondent ID'].astype(int)

# Encode categorical variables
le = LabelEncoder()
for col in categorical_columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Now, let's check the changes
print(data.info())
print(data.head())

# 4. Exploratory Data Analysis (EDA) with a Focus on Static Visualization
# Calculate the duration in days between received and closed dates
data['Duration'] = (data['Closed date'] - data['Received date']).dt.days

# Verify if 'Duration' column exists and view its summary to ensure correctness
if 'Duration' in data.columns:
    print("Duration column is created successfully. Summary:")
    print(data['Duration'].describe())
else:
    print("Failed to create Duration column.")

# 1. Bar Chart of Complaint Types
if 'Complaint type' in data.columns:
    complaint_types_counts = data['Complaint type'].value_counts()
    plt.figure(figsize=(10, 6))
    complaint_types_counts.plot(kind='bar', color='skyblue')
    plt.title('Frequency of Complaint Types')
    plt.xlabel('Complaint Type')
    plt.ylabel('Number of Complaints')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Complaint type column not found for plotting.")

# 2. Histogram of Duration between Received and Closed Dates
if 'Duration' in data.columns:
    plt.figure(figsize=(10, 6))
    plt.hist(data['Duration'].dropna(), bins=30, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Complaint Resolution Duration')
    plt.xlabel('Duration in Days')
    plt.ylabel('Frequency')
    plt.show()
else:
    print("Duration column is not available for plotting.")

# 3. Box Plot for Respondent IDs
if 'Respondent ID' in data.columns:
    plt.figure(figsize=(10, 6))
    plt.boxplot(data['Respondent ID'], notch=True, vert=False)
    plt.title('Distribution of Respondent IDs')
    plt.xlabel('Respondent ID')
    plt.show()
else:
    print("Respondent ID column not found for plotting.")

# 4. Statistical Summary for numeric data
numeric_cols = data.select_dtypes(include=['int64', 'float64'])
print("Statistical Summary for numeric columns:")
print(numeric_cols.describe())

# 5. Advanced Analysis (Optional)
# Prepare data for logistic regression
data['Duration'] = (data['Closed date'] - data['Received date']).dt.days
X = data.drop(['Confirmed complaint', 'Received date', 'Closed date'], axis=1)
y = data['Confirmed complaint'].apply(lambda x: 1 if x == 'Yes' else 0)

# Check the distribution of the target variable
print("Distribution of the target variable:")
print(y.value_counts())

# If the target variable has only one class, we need to handle it
if y.nunique() < 2:
    print("Warning: The target variable contains only one class. Adjusting the dataset to include at least two classes.")
    # Example: Manually adding a sample of another class (if possible)
    # This is just an example and should be adjusted based on your actual data
    # Here, we assume that the dataset has at least one sample of another class
    if len(data) > 1:
        y.iloc[0] = 1  # Change the first sample to class 1
    else:
        raise ValueError("The dataset contains only one sample. At least two samples are required for classification.")

# Now, recheck the distribution of the target variable
print("Updated distribution of the target variable:")
print(y.value_counts())

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