# This module handles data loading, preprocessing, and splitting for the rabbit position prediction task.
# Import necessary libraries
def load_data(filepath):
    import pandas as pd
    # Load the dataset from a CSV file
    data = pd.read_csv(filepath)
    return data

# This function preprocesses the data by normalizing pressure values and encoding rabbit position classes.
def preprocess_data(data):
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder

    # Normalize pressure data
    pressure_cols = [col for col in data.columns if col.startswith('Pressure_')]
    scaler = StandardScaler()
    '''Normalizes pressure data using StandardScaler. This scales the pressure values to have a mean of 0 and a standard deviation of 1.'''
    data[pressure_cols] = scaler.fit_transform(data[pressure_cols])

    # Encode rabbit position classes
    label_encoder = LabelEncoder()
    data['RabbitPosition'] = label_encoder.fit_transform(data['RabbitPosition'])

    return data, label_encoder

#This function splits the dataset into training and testing sets.
def split_data(data, test_size=0.2):
    from sklearn.model_selection import train_test_split
    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, test_data

# This function extracts features and labels from the dataset.
def get_features_and_labels(data):
    # Separate features and labels
    X = data[[col for col in data.columns if col.startswith('Pressure_')]].values
    y = data['RabbitPosition'].values
    return X, y