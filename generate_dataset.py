# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_attendance_data(n_samples=1000):
    np.random.seed(42)
    
    # Define possible values for categorical features
    health_status = ['Good', 'Fair', 'Poor']
    transportation_mode = ['Car', 'Bus', 'Walking', 'Bicycle']
    extracurricular_activities = ['Yes', 'No']
    class_time = ['Morning', 'Afternoon', 'Evening']
    subject_difficulty = ['Easy', 'Medium', 'Hard']
    
    # Generate data
    data = {
        'health_status': np.random.choice(health_status, n_samples),
        'transportation_mode': np.random.choice(transportation_mode, n_samples),
        'extracurricular_activities': np.random.choice(extracurricular_activities, n_samples),
        'class_time': np.random.choice(class_time, n_samples),
        'subject_difficulty': np.random.choice(subject_difficulty, n_samples),
    }
    
    # Generate numerical features with realistic distributions
    data['previous_attendance'] = np.random.beta(2, 2, n_samples)  # Beta distribution for attendance
    data['study_hours'] = np.random.normal(0.4, 0.1, n_samples).clip(0, 1)  # Normal distribution for study hours
    data['sleep_hours'] = np.random.normal(0.33, 0.05, n_samples).clip(0, 1)  # Normal distribution for sleep
    data['previous_grades'] = np.random.beta(2, 1, n_samples)  # Beta distribution for grades
    data['distance'] = np.random.beta(1, 2, n_samples)  # Beta distribution for distance
    data['previous_absence_count'] = np.random.beta(1, 3, n_samples)  # Beta distribution for absences
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate attendance status based on features
    # Higher probability of Present for:
    # - Good health
    # - High family support
    # - High previous attendance
    # - High previous grades
    # - Low distance
    # - Low previous absences
    
    present_prob = (
        (df['health_status'] == 'Good').astype(int) * 0.2 +
        df['previous_attendance'] * 0.2 +
        df['previous_grades'] * 0.15 +
        (1 - df['distance']) * 0.1 +
        (1 - df['previous_absence_count']) * 0.2
    )
    
    # Add some randomness
    present_prob += np.random.normal(0, 0.1, n_samples)
    present_prob = present_prob.clip(0, 1)
    
    # Generate attendance status
    df['attendance_status'] = np.where(present_prob > 0.5, 'Present', 'Absent')
    
    # Ensure balanced classes
    present_count = (df['attendance_status'] == 'Present').sum()
    absent_count = (df['attendance_status'] == 'Absent').sum()
    
    if present_count > absent_count:
        # Downsample Present class
        present_df = df[df['attendance_status'] == 'Present']
        present_df = present_df.sample(n=absent_count, random_state=42)
        df = pd.concat([present_df, df[df['attendance_status'] == 'Absent']])
    elif absent_count > present_count:
        # Downsample Absent class
        absent_df = df[df['attendance_status'] == 'Absent']
        absent_df = absent_df.sample(n=present_count, random_state=42)
        df = pd.concat([absent_df, df[df['attendance_status'] == 'Present']])
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

if __name__ == '__main__':
    # Generate and save dataset
    df = generate_attendance_data(1000)
    print("\nDataset generated successfully!")
    print("\nClass distribution:")
    print(df['attendance_status'].value_counts())
    print("\nSample of the dataset:")
    print(df.head()) 