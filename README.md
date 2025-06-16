# Student Attendance Prediction System

This system uses machine learning to predict student attendance patterns based on historical data. It provides a web interface for making predictions and analyzing attendance trends.

## Features

- Machine learning-based attendance prediction
- Modern web interface using Tailwind CSS
- RESTful API endpoints
- Real-time predictions
- Confidence scores for predictions

## Prerequisites

- Python 3.7+
- pip (Python package manager)
- Modern web browser

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd student-attendance-classifier
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the attendance.csv file in the root directory with the following columns:
- previous_attendance
- study_hours
- distance
- previous_grades
- attendance_status

## Running the Application

1. Start the Flask backend:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Fill in the prediction form with student data:
   - Previous Attendance Rate
   - Study Hours
   - Distance from School
   - Previous Grades

2. Click "Predict Attendance" to get the prediction results

3. View the predicted attendance status and confidence score

## API Endpoints

- `POST /api/predict`: Make attendance predictions
- `GET /api/features`: Get available features for prediction

## Technologies Used

- Backend:
  - Python
  - Flask
  - scikit-learn
  - pandas
  - numpy

- Frontend:
  - HTML5
  - Tailwind CSS
  - JavaScript (Vanilla)

## Contributing

Feel free to submit issues and enhancement requests!