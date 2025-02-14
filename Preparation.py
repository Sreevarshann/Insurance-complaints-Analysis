
# Handling missing data
# Filling missing values with 'Unknown' or the most frequent value for categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
data[categorical_columns] = data[categorical_columns].fillna('Unknown')

# Removing duplicates
data.drop_duplicates(inplace=True)

# Convert date columns to datetime
data['Received date'] = pd.to_datetime(data['Received date'], errors='coerce')
data['Closed date'] = pd.to_datetime(data['Closed date'], errors='coerce')

# Normalize data types if necessary
# For example, ensuring all IDs are int (if not already)
data['Respondent ID'] = data['Respondent ID'].astype(int)

# Encode categorical variables
# Using label encoding for simplicity, one could also use one-hot encoding if needed
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in categorical_columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Now, let's check the changes
print(data.info())
print(data.head())