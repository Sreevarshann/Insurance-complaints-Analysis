import pandas as pd

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
