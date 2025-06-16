# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import joblib
import os
from generate_dataset import generate_attendance_data

def balance_dataset(df):
    # Separate majority and minority classes
    df_majority = df[df.attendance_status == 'Absent']
    df_minority = df[df.attendance_status == 'Present']
    
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                   replace=True,     # sample with replacement
                                   n_samples=len(df_majority),    # to match majority class
                                   random_state=42)  # reproducible results
    
    # Combine majority and upsampled minority
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_balanced

def train_and_save_model():
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Generate or load dataset
    print("Generating dataset...")
    df = generate_attendance_data(1000)  # Generate 1000 samples
    
    # Balance the dataset
    print("Balancing dataset...")
    df = balance_dataset(df)
    
    # Print class distribution
    print("\nClass distribution after balancing:")
    print(df['attendance_status'].value_counts())
    
    # Prepare features and target
    categorical_features = [
        'health_status', 'transportation_mode', 'family_support',
        'extracurricular_activities', 'internet_access', 'class_time',
        'subject_difficulty', 'family_income_level'
    ]
    
    numerical_features = [
        'previous_attendance', 'study_hours', 'sleep_hours',
        'previous_grades', 'distance', 'previous_absence_count'
    ]
    
    # Initialize label encoders
    label_encoders = {}
    
    # Transform categorical features
    X_categorical = pd.DataFrame()
    for feature in categorical_features:
        le = LabelEncoder()
        X_categorical[feature] = le.fit_transform(df[feature])
        label_encoders[feature] = le
        # Save encoder
        joblib.dump(le, f'model/{feature}_encoder.pkl')
        print(f"Saved encoder for {feature}")
    
    # Combine features
    X = pd.concat([X_categorical, df[numerical_features]], axis=1)
    y = df['attendance_status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model with class weights
    print("\nTraining model...")
    model = RandomForestClassifier(
        n_estimators=200,  # Increased number of trees
        max_depth=15,      # Increased depth
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Add class weights
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"\nTrain accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Print class distribution in predictions
    y_pred = model.predict(X_test)
    print("\nPrediction distribution on test set:")
    print(pd.Series(y_pred).value_counts())
    
    # Save model
    joblib.dump(model, 'model/student_attendance_model.pkl')
    print("\nModel saved successfully")
    
    # Save feature names
    feature_names = list(X.columns)
    joblib.dump(feature_names, 'model/feature_names.pkl')
    print("Feature names saved")
    
    # Save feature importances
    feature_importances = dict(zip(feature_names, model.feature_importances_))
    joblib.dump(feature_importances, 'model/feature_importances.pkl')
    print("Feature importances saved")
    
    return model, label_encoders, feature_names

if __name__ == '__main__':
    print("Starting model training...")
    model, label_encoders, feature_names = train_and_save_model()
    print("\nModel training completed successfully!") 