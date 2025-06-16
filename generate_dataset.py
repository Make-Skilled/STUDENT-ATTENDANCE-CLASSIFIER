import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_attendance_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate dates for the last academic year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate student IDs
    student_ids = [f'STU{i:04d}' for i in range(1, n_samples + 1)]
    
    # Generate data
    data = {
        'student_id': np.random.choice(student_ids, n_samples),
        'date': np.random.choice(dates, n_samples),
        
        # Academic Performance
        'previous_attendance': np.random.normal(85, 10, n_samples).clip(0, 100),
        'previous_grades': np.random.normal(75, 15, n_samples).clip(0, 100),
        'study_hours': np.random.normal(4, 1.5, n_samples).clip(0, 12),
        
        # Physical Factors
        'distance': np.random.exponential(5, n_samples).clip(0.1, 30),
        'sleep_hours': np.random.normal(7, 1, n_samples).clip(4, 12),
        'health_status': np.random.choice(['Good', 'Fair', 'Poor'], n_samples, p=[0.7, 0.2, 0.1]),
        
        # Transportation
        'transportation_mode': np.random.choice(
            ['Bus', 'Car', 'Walking', 'Bicycle'],
            n_samples,
            p=[0.4, 0.3, 0.2, 0.1]
        ),
        
        # Social Factors
        'family_support': np.random.choice(
            ['High', 'Medium', 'Low'],
            n_samples,
            p=[0.6, 0.3, 0.1]
        ),
        'extracurricular_activities': np.random.choice(
            ['Yes', 'No'],
            n_samples,
            p=[0.7, 0.3]
        ),
        
        # Infrastructure
        'internet_access': np.random.choice(
            ['Yes', 'No'],
            n_samples,
            p=[0.9, 0.1]
        ),
        
        # Additional Features
        'weather_condition': np.random.choice(
            ['Sunny', 'Rainy', 'Cloudy', 'Snowy'],
            n_samples,
            p=[0.4, 0.3, 0.2, 0.1]
        ),
        'class_time': np.random.choice(
            ['Morning', 'Afternoon', 'Evening'],
            n_samples,
            p=[0.5, 0.3, 0.2]
        ),
        'subject_difficulty': np.random.choice(
            ['Easy', 'Moderate', 'Difficult'],
            n_samples,
            p=[0.3, 0.4, 0.3]
        ),
        'previous_absence_count': np.random.poisson(3, n_samples),
        'family_income_level': np.random.choice(
            ['Low', 'Middle', 'High'],
            n_samples,
            p=[0.2, 0.6, 0.2]
        )
    }
    
    # Generate attendance status based on features
    attendance_prob = np.zeros(n_samples)
    
    # Factors affecting attendance probability
    attendance_prob += (data['previous_attendance'] - 50) / 50  # Previous attendance
    attendance_prob += (data['previous_grades'] - 50) / 100  # Grades
    attendance_prob += (data['study_hours'] - 2) / 10  # Study hours
    attendance_prob -= data['distance'] / 30  # Distance
    attendance_prob += (data['sleep_hours'] - 6) / 6  # Sleep hours
    
    # Categorical factors
    health_map = {'Good': 0.2, 'Fair': 0, 'Poor': -0.3}
    attendance_prob += np.array([health_map[h] for h in data['health_status']])
    
    transport_map = {'Bus': -0.1, 'Car': 0.1, 'Walking': -0.05, 'Bicycle': 0}
    attendance_prob += np.array([transport_map[t] for t in data['transportation_mode']])
    
    support_map = {'High': 0.2, 'Medium': 0, 'Low': -0.2}
    attendance_prob += np.array([support_map[s] for s in data['family_support']])
    
    # Normalize probability
    attendance_prob = (attendance_prob - attendance_prob.min()) / (attendance_prob.max() - attendance_prob.min())
    attendance_prob = attendance_prob * 0.8 + 0.1  # Scale to 0.1-0.9 range
    
    # Generate attendance status
    data['attendance_status'] = np.random.choice(
        ['Present', 'Absent', 'Late'],
        n_samples,
        p=[0.8, 0.15, 0.05]
    )
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some correlations
    df.loc[df['health_status'] == 'Poor', 'attendance_status'] = np.random.choice(
        ['Present', 'Absent', 'Late'],
        size=len(df[df['health_status'] == 'Poor']),
        p=[0.4, 0.5, 0.1]
    )
    
    df.loc[df['distance'] > 15, 'attendance_status'] = np.random.choice(
        ['Present', 'Absent', 'Late'],
        size=len(df[df['distance'] > 15]),
        p=[0.6, 0.3, 0.1]
    )
    
    return df

if __name__ == '__main__':
    # Generate dataset
    df = generate_attendance_data(1000)
    
    # Save to CSV
    df.to_csv('attendance.csv', index=False)
    print("Dataset generated and saved to attendance.csv")
    print("\nDataset Statistics:")
    print(f"Total records: {len(df)}")
    print("\nFeature distributions:")
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"\n{col}:")
            print(df[col].value_counts())
        else:
            print(f"\n{col}:")
            print(f"Mean: {df[col].mean():.2f}")
            print(f"Std: {df[col].std():.2f}")
            print(f"Min: {df[col].min():.2f}")
            print(f"Max: {df[col].max():.2f}") 