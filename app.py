from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from generate_dataset import generate_attendance_data

app = Flask(__name__)
CORS(app)

# Define feature lists
categorical_features = [
    'health_status',
    'transportation_mode',
    'family_support',
    'extracurricular_activities',
    'internet_access',
    'weather_condition',
    'class_time',
    'subject_difficulty',
    'family_income_level'
]

numerical_features = [
    'previous_attendance',
    'previous_grades',
    'study_hours',
    'distance',
    'sleep_hours',
    'previous_absence_count'
]

# Initialize global variables
model = None
label_encoders = {}
target_encoder = None
scaler = None

# Load and preprocess data
def load_data():
    if not os.path.exists('attendance.csv'):
        # If dataset doesn't exist, generate it
        df = generate_attendance_data(1000)
        df.to_csv('attendance.csv', index=False)
    return pd.read_csv('attendance.csv')

# Train model
def train_model():
    df = load_data()
    
    # Convert categorical variables to numerical
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Scale numerical features
    for col in numerical_features:
        scaler.fit_transform(df[[col]])
    
    # Prepare features and target
    X = df.drop(['attendance_status', 'student_id', 'date'], axis=1)
    y = df['attendance_status']
    
    # Convert target to numerical
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and encoders
    joblib.dump(model, 'attendance_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(le_target, 'target_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return model, label_encoders, le_target, scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Validate required fields
        required_fields = categorical_features + numerical_features
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Transform categorical features
        for feature in categorical_features:
            if feature in input_data.columns:
                try:
                    le = label_encoders[feature]
                    input_data[feature] = le.transform(input_data[feature])
                except Exception as e:
                    return jsonify({'error': f'Error processing {feature}: {str(e)}'}), 400
        
        # Scale numerical features
        try:
            numerical_data = input_data[numerical_features]
            scaled_data = scaler.transform(numerical_data)
            input_data[numerical_features] = scaled_data
        except Exception as e:
            return jsonify({'error': f'Error scaling numerical features: {str(e)}'}), 400
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
        except Exception as e:
            return jsonify({'error': f'Error making prediction: {str(e)}'}), 400
        
        # Get feature importance
        try:
            feature_importance = dict(zip(model.feature_names_in_, model.feature_importances_))
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            return jsonify({'error': f'Error calculating feature importance: {str(e)}'}), 400
        
        # Calculate risk factors
        risk_factors = []
        for feature, value in data.items():
            if feature in ['previous_attendance', 'study_hours', 'sleep_hours']:
                if float(value) < 0.5:  # Low values are risk factors
                    risk_factors.append(f"{feature.replace('_', ' ').title()}: {value}")
            elif feature in ['distance', 'previous_absence_count']:
                if float(value) > 0.7:  # High values are risk factors
                    risk_factors.append(f"{feature.replace('_', ' ').title()}: {value}")
        
        return jsonify({
            'prediction': prediction,
            'probabilities': {
                'Present': float(probabilities[0]),
                'Absent': float(probabilities[1]),
                'Late': float(probabilities[2])
            },
            'feature_importance': sorted_importance,
            'risk_factors': risk_factors,
            'confidence': float(max(probabilities))
        })
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 400

@app.route('/api/features', methods=['GET'])
def get_features():
    try:
        df = load_data()
        features = df.drop(['attendance_status', 'student_id', 'date'], axis=1).columns.tolist()
        return jsonify({
            'success': True,
            'features': features
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    try:
        df = load_data()
        model_info = {
            'total_samples': len(df),
            'feature_names': df.drop(['attendance_status', 'student_id', 'date'], axis=1).columns.tolist(),
            'target_distribution': df['attendance_status'].value_counts().to_dict(),
            'feature_types': {
                'numerical': numerical_features,
                'categorical': categorical_features
            }
        }
        return jsonify({
            'success': True,
            'model_info': model_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True) 