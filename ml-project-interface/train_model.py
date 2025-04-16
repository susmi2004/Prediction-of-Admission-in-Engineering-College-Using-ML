import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    print("Loading data...")
    # Load the dataset
    df = pd.read_csv('college_data.csv')

    print("Preprocessing data...")
    # Initialize label encoders
    le_gender = LabelEncoder()
    le_category = LabelEncoder()
    le_branch = LabelEncoder()
    le_college = LabelEncoder()

    # Fit and transform the data
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Category'] = le_category.fit_transform(df['Category'])
    df['Branch'] = le_branch.fit_transform(df['Branch'])
    df['College'] = le_college.fit_transform(df['College'])

    # Prepare features and target
    X = df[['Gender', 'Category', 'Rank', 'Branch', 'College']]
    y = df['Eligible']

    print("Training model...")
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    print("Saving model and encoders...")
    # Save the model and encoders
    model_data = {
        'model': model,
        'le_gender': le_gender,
        'le_category': le_category,
        'le_branch': le_branch,
        'le_college': le_college,
        'feature_names': X.columns.tolist()
    }

    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model training and saving completed!")

if __name__ == "__main__":
    train_and_save_model()
