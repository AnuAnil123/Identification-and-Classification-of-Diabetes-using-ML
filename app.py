from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
import logging
from filelock import FileLock
import os
from werkzeug.utils import secure_filename


# Initialize the Flask application
app = Flask(__name__)

# Security: Secret key should be stored in an environment variable for production
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_secret_key')  # Update for production

# Paths to Excel file and machine learning models
excel_file_path = r'C:\Users\HP\Downloads\patients.xlsx'
random_forest_model_path = r'C:\Users\HP\Downloads\random_forest_model.pkl'
xgb_model_path = r'C:\Users\HP\Downloads\xgboost_model.pkl'
# Define the upload folder and ensure it exists
UPLOAD_FOLDER = 'C:/Users/HP/Downloads/d/upload'  # Replace this with the path where you want to save files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the models
rf_model = joblib.load(random_forest_model_path)
xgb_model = joblib.load(xgb_model_path)

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

# Helper functions for Excel operations
def load_patients_from_excel():
    try:
        return pd.read_excel(excel_file_path).to_dict(orient='records')
    except FileNotFoundError:
        logging.error(f"Excel file not found: {excel_file_path}")
        return []
    except Exception as e:
        logging.error(f"Error loading Excel file: {str(e)}")
        return []

def save_patients_to_excel(patients_list):
    try:
        df = pd.DataFrame(patients_list)
        lock = FileLock(f"{excel_file_path}.lock")
        with lock:
            df.to_excel(excel_file_path, index=False)
    except Exception as e:
        logging.error(f"Error saving Excel file: {str(e)}")

def make_prediction(features, model_choice):
    if not all(isinstance(f, (int, float)) for f in features):
        logging.error("Invalid input features")
        return "Invalid input data"
    
    logging.debug(f"Features: {features}, Model Choice: {model_choice}")
    
    df = pd.DataFrame([features], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 
        'SkinThickness', 'Insulin', 'BMI', 
        'DiabetesPedigreeFunction', 'Age'
    ])
    
    if model_choice == 'random_forest':
        prediction = rf_model.predict(df)[0]
    elif model_choice == 'xgb':
        prediction = xgb_model.predict(df)[0]
    
    logging.debug(f"Prediction: {prediction}")
    return "Diabetes" if prediction == 1 else "No Diabetes"

# Initialize Excel file if not present
def initialize_excel():
    try:
        pd.read_excel(excel_file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['id', 'name', 'prediction'])
        df.to_excel(excel_file_path, index=False)

initialize_excel()

# Routes
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_type = request.form['user_type']
        password = request.form['password']
        if user_type == 'user' and password == 'u':
            session['user_type'] = 'user'
            return redirect(url_for('user_dashboard'))
        elif user_type == 'admin' and password == 'a':
            session['user_type'] = 'admin'
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid credentials. Please try again.')
    return render_template('login.html')

@app.route('/user_dashboard')
def user_dashboard():
    if 'user_type' not in session or session['user_type'] != 'user':
        return redirect(url_for('home'))
    return render_template('user_dashboard.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'user_type' not in session or session['user_type'] != 'admin':
        return redirect(url_for('home'))
    
    patients = load_patients_from_excel()
    return render_template('admin_dashboard.html', patients=patients)

@app.route('/user_predict', methods=['GET', 'POST'])
def user_predict():
    if 'user_type' in session and session['user_type'] == 'user':
        if request.method == 'POST':
            patient_id = request.form['patient_id']
            patient_name = request.form['patient_name']
            model_choice = request.form['model_choice']

            try:
                features = [
                    float(request.form['Pregnancies']),
                    float(request.form['Glucose']),
                    float(request.form['BloodPressure']),
                    float(request.form['SkinThickness']),
                    float(request.form['Insulin']),
                    float(request.form['BMI']),
                    float(request.form['DiabetesPedigreeFunction']),
                    float(request.form['Age'])
                ]
            except ValueError:
                flash("Invalid input data. Please ensure all inputs are numeric.")
                return redirect(url_for('user_predict'))

            result = make_prediction(features, model_choice)
            accuracy = "98%" if model_choice == 'random_forest' else "78%"

            patients = load_patients_from_excel()
            patients.append({'id': patient_id, 'name': patient_name, 'prediction': result})
            save_patients_to_excel(patients)

            return render_template('prediction_result.html', 
                                   prediction=result, 
                                   accuracy=accuracy, 
                                   patient_id=patient_id, 
                                   patient_name=patient_name)
    return render_template('user_predict.html')

@app.route('/admin_predict', methods=['GET', 'POST'])
def admin_predict():
    if 'user_type' in session and session['user_type'] == 'admin':
        if request.method == 'POST':
            logging.debug("Admin predict form submitted.")
            patient_id = request.form['patient_id']
            patient_name = request.form['patient_name']
            model_choice = request.form['model_choice']

            try:
                features = [
                    float(request.form['Pregnancies']),
                    float(request.form['Glucose']),
                    float(request.form['BloodPressure']),
                    float(request.form['SkinThickness']),
                    float(request.form['Insulin']),
                    float(request.form['BMI']),
                    float(request.form['DiabetesPedigreeFunction']),
                    float(request.form['Age'])
                ]
            except ValueError:
                flash("Invalid input data. Please ensure all inputs are numeric.")
                return redirect(url_for('admin_predict'))

            result = make_prediction(features, model_choice)
            accuracy = "98%" if model_choice == 'random_forest' else "78%"

            patients = load_patients_from_excel()
            patients.append({'id': patient_id, 'name': patient_name, 'prediction': result})
            save_patients_to_excel(patients)

            return render_template('prediction_result.html', 
                                   prediction=result, 
                                   accuracy=accuracy, 
                                   patient_id=patient_id, 
                                   patient_name=patient_name)
    return render_template('admin_predict.html')
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Generate file preview (e.g., for CSV or Excel files)
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)
                preview = df.head().to_html(classes='table table-bordered')
                return render_template('admin_dashboard.html', preview=preview)
            except Exception as e:
                flash(f'Error processing file: {e}')
                return redirect(url_for('admin_dashboard'))

    return render_template('admin_dashboard.html')


@app.route('/add_patient', methods=['POST'])
def add_patient():
    if 'user_type' not in session or session['user_type'] != 'admin':
        return redirect(url_for('home'))

    patient_id = request.form['patient_id']
    patient_name = request.form['patient_name']

    patients = load_patients_from_excel()
    patients.append({'id': patient_id, 'name': patient_name, 'prediction': None})
    save_patients_to_excel(patients)

    flash('Patient added successfully.')  # Display success message
    return redirect(url_for('admin_dashboard'))

@app.route('/delete_patient/<int:patient_id>', methods=['POST'])
def delete_patient(patient_id):
    # Load existing patients
    patients = load_patients_from_excel()
    
    # Find the patient by id
    patients = [patient for patient in patients if patient['id'] != patient_id]

    # Save the updated patient list back to the Excel file
    save_patients_to_excel(patients)

    flash('Patient deleted successfully.')  # Feedback for the user
    return redirect(url_for('admin_dashboard'))  # Redirect back to the dashboard

@app.route('/delete_all_patients', methods=['POST'])
def delete_all_patients():
    try:
        df = pd.DataFrame(columns=['id', 'name', 'prediction'])
        lock = FileLock(f"{excel_file_path}.lock")
        with lock:
            df.to_excel(excel_file_path, index=False)
        flash('All patients deleted successfully!', 'success')  # Display success message
    except Exception as e:
        logging.error(f'Error deleting patients: {str(e)}')
        flash(f'Error deleting patients: {str(e)}', 'error')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/logout')
def logout():
    session.pop('user_type', None)
    return redirect(url_for('home'))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
