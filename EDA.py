import matplotlib.pyplot as plt
import pandas as pd

# Make sure the dates are in the correct datetime format
data['Received date'] = pd.to_datetime(data['Received date'], errors='coerce')
data['Closed date'] = pd.to_datetime(data['Closed date'], errors='coerce')

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
