from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os 
import json
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a secure secret key for session management
    
# Global variables to store models and encoders
models = None
le_inst = None
le_branch = None
df_boys = None
df_girls = None
college_data = None

# Database setup
def get_db_connection():
    conn = sqlite3.connect('user_database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    with open('schema.sql') as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

# Initialize the database if it doesn't exist
if not os.path.exists('user_database.db'):
    init_db()

def load_and_preprocess_data():
    global df_boys, df_girls, le_inst, le_branch, college_data
    
    # Reading the dataset
    df = pd.read_csv('viewFile5.csv')
    
    # Clean up column names by removing whitespace and newlines
    df.columns = df.columns.str.strip()
    
    # Convert rank columns to numeric
    rank_columns = [col for col in df.columns if 'BOYS' in col or 'GIRLS' in col]
    for col in rank_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean text columns
    if 'NAME OF THE INSTITUTION' in df.columns:
        df['NAME OF THE INSTITUTION'] = df['NAME OF THE INSTITUTION'].str.replace('\n', ' ').str.strip()
    elif 'NAMEOFTHEINSTITUTION' in df.columns:
        df['NAMEOFTHEINSTITUTION'] = df['NAMEOFTHEINSTITUTION'].str.replace('\n', ' ').str.strip()
        df.rename(columns={'NAMEOFTHEINSTITUTION': 'NAME OF THE INSTITUTION'}, inplace=True)
    
    if 'PLACE' in df.columns:
        df['PLACE'] = df['PLACE'].str.strip()
    
    if 'DIST' in df.columns:
        df['DIST'] = df['DIST'].str.strip()

    # Create a simplified target variable (eligible/not eligible) for each category
    def create_eligibility(row, category_col):
        rank_columns = [col for col in df.columns if category_col in col]
        if rank_columns and any(pd.notna(row[col]) for col in rank_columns):
            min_rank = min([row[col] for col in rank_columns if pd.notna(row[col])])
        else:
            min_rank = np.nan
        return 1 if pd.notna(min_rank) else 0

    # Prepare features and targets
    features = ['INSTCODE', 'branch_code']
    category_columns = ['OC', 'SC', 'ST', 'BCA', 'BCB', 'BCC', 'BCD', 'BCE', 'OC_EWS']

    # Create separate datasets for boys and girls
    df_boys = df.copy()
    df_girls = df.copy()

    for cat in category_columns:
        df_boys[f'{cat}_eligible'] = df_boys.apply(
            lambda row: create_eligibility(row, f'{cat}_BOYS'), axis=1)
        df_girls[f'{cat}_eligible'] = df_girls.apply(
            lambda row: create_eligibility(row, f'{cat}_GIRLS'), axis=1)

    # Encode categorical variables
    le_inst = LabelEncoder()
    le_branch = LabelEncoder()

    df_boys['INSTCODE'] = le_inst.fit_transform(df_boys['INSTCODE'])
    df_boys['branch_code'] = le_branch.fit_transform(df_boys['branch_code'])
    df_girls['INSTCODE'] = le_inst.transform(df_girls['INSTCODE'])
    df_girls['branch_code'] = le_branch.transform(df_girls['branch_code'])
    
    # Store college data for dropdowns
    college_data = df[['INSTCODE', 'NAME OF THE INSTITUTION', 'branch_code']].drop_duplicates()

    return df_boys, df_girls, le_inst, le_branch

def train_models(df_boys, df_girls):
    models = {}
    X_boys = df_boys[['INSTCODE', 'branch_code']]
    X_girls = df_girls[['INSTCODE', 'branch_code']]

    category_columns = ['OC', 'SC', 'ST', 'BCA', 'BCB', 'BCC', 'BCD', 'BCE', 'OC_EWS']

    for cat in category_columns:
        # Boys model
        y_boys = df_boys[f'{cat}_eligible']
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
            X_boys, y_boys, test_size=0.2, random_state=42)

        rf_boys = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_boys.fit(X_train_b, y_train_b)

        # Girls model
        y_girls = df_girls[f'{cat}_eligible']
        X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
            X_girls, y_girls, test_size=0.2, random_state=42)

        rf_girls = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_girls.fit(X_train_g, y_train_g)

        models[cat] = {'boys': rf_boys, 'girls': rf_girls}

    # Save the models and encoders
    with open('models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'le_inst': le_inst,
            'le_branch': le_branch
        }, f)

    return models

def load_models():
    global models, le_inst, le_branch
    
    # Load the models and encoders
    with open('models.pkl', 'rb') as f:
        saved_data = pickle.load(f)
        models = saved_data['models']
        le_inst = saved_data['le_inst']
        le_branch = saved_data['le_branch']

def predict_admission(rank, gender, category, preferred_college, preferred_branch):
    global models, le_inst, le_branch, df_boys, df_girls
    
    # Select appropriate dataset based on gender
    df = df_boys if gender.lower() == 'male' else df_girls

    # Clean and transform input
    preferred_college = preferred_college.strip().upper()
    preferred_branch = preferred_branch.strip().upper()
    
    print(f"\nExamining inputs:")
    print(f"Rank: {rank}")
    print(f"Gender: {gender}")
    print(f"Category: {category}")
    print(f"Preferred College: {preferred_college}")
    print(f"Preferred Branch: {preferred_branch}")

    # Encode inputs
    try:
        inst_code = preferred_college  # Already using the code from dropdown
        branch_code = preferred_branch  # Already using the code from dropdown
        
        # Convert to numeric for model input
        inst_code_num = le_inst.transform([inst_code])[0]
        branch_code_num = le_branch.transform([branch_code])[0]
    except Exception as e:
        return {"status": "error", "message": f"Invalid college code or branch code: {str(e)}"}

    # Get the appropriate model based on gender and category
    gender_key = 'boys' if gender.lower() == 'male' else 'girls'
    
    # Make sure category is in the correct format
    if category not in ['OC', 'SC', 'ST', 'BCA', 'BCB', 'BCC', 'BCD', 'BCE', 'OC_EWS']:
        return {"status": "error", "message": f"Invalid category: {category}"}
    
    try:
        model = models[category][gender_key]
    except KeyError:
        return {"status": "error", "message": f"No model available for category: {category} and gender: {gender}"}

    # Prepare input for prediction
    input_data = np.array([[inst_code_num, branch_code_num]])

    # Predict eligibility
    prediction = model.predict(input_data)[0]

    # Get rank threshold from original data
    gender_suffix = 'BOYS' if gender.lower() == 'male' else 'GIRLS'
    category_col = f'{category}_{gender_suffix}'

    # Filter the dataframe to get the specific college-branch combination
    college_branch_data = df[
        (df['INSTCODE'] == inst_code_num) &
        (df['branch_code'] == branch_code_num)
    ]

    if college_branch_data.empty:
        return {"status": "error", "message": "No data available for this college-branch combination"}

    # Get the rank threshold
    try:
        rank_threshold = college_branch_data[category_col].iloc[0]
    except KeyError:
        return {"status": "error", "message": f"No data available for category: {category}"}
    
    # Get college details from the original dataframe
    try:
        original_df = pd.read_csv('viewFile5.csv')
        
        # Clean up column names by removing whitespace
        original_df.columns = original_df.columns.str.strip()
        
        # Check if the column names exist in the dataframe
        if 'INSTCODE' not in original_df.columns or 'branch_code' not in original_df.columns:
            # Try to find alternative column names
            if 'INSTCODE' not in original_df.columns and 'INST CODE' in original_df.columns:
                original_df['INSTCODE'] = original_df['INST CODE']
            if 'branch_code' not in original_df.columns and 'BRANCH CODE' in original_df.columns:
                original_df['branch_code'] = original_df['BRANCH CODE']
        
        name_col = 'NAME OF THE INSTITUTION' if 'NAME OF THE INSTITUTION' in original_df.columns else 'NAMEOFTHEINSTITUTION'
        place_col = 'PLACE' if 'PLACE' in original_df.columns else 'LOCATION'
        dist_col = 'DIST' if 'DIST' in original_df.columns else 'DISTRICT'
        fee_col = 'COLLFEE' if 'COLLFEE' in original_df.columns else 'FEE'
        
        college_details_df = original_df[
            (original_df['INSTCODE'] == inst_code) &
            (original_df['branch_code'] == branch_code)
        ]
        
        if college_details_df.empty:
            return {"status": "error", "message": "Could not find college details"}
        
        # Get college details and clean the data
        college_details = {
            'name': str(college_details_df[name_col].iloc[0]).replace('\n', ' ').strip(),
            'place': str(college_details_df[place_col].iloc[0]).strip() if place_col in college_details_df.columns else "N/A",
            'district': str(college_details_df[dist_col].iloc[0]).strip() if dist_col in college_details_df.columns else "N/A",
            'fee': int(college_details_df[fee_col].iloc[0]) if fee_col in college_details_df.columns and pd.notna(college_details_df[fee_col].iloc[0]) else 0
        }
    except Exception as e:
        # If there's an error getting college details, continue with prediction but without details
        college_details = {
            'name': "College information unavailable",
            'place': "N/A",
            'district': "N/A",
            'fee': 0
        }
        print(f"Error getting college details: {str(e)}")

    try:
        rank_threshold_numeric = pd.to_numeric(rank_threshold, errors='coerce')
    except:
        return {"status": "error", "message": "Error: Rank threshold in data is not in a valid numeric format"}

    # Check eligibility for preferred branch
    if pd.notna(rank_threshold_numeric) and rank <= rank_threshold_numeric:
        result = {
            "status": "success", 
            "message": f"Eligible for preferred branch at {college_details['name']}",
            "college_details": college_details
        }
    else:
        result = {
            "status": "success", 
            "message": f"Not eligible for {preferred_branch} at {college_details['name']}",
            "college_details": college_details
        }

    # Find alternative branches and colleges
    try:
        # Find alternative branches in the same college
        other_branches_df = original_df[original_df['INSTCODE'] == inst_code]
        eligible_branches = []

        for _, row in other_branches_df.iterrows():
            try:
                if category_col in row and pd.notna(row[category_col]):
                    row_rank = pd.to_numeric(row[category_col], errors='coerce')
                    if pd.notna(row_rank) and rank <= row_rank:
                        branch = row['branch_code']
                        branch_name = {
                            'CSE': 'Computer Science & Engineering',
                            'ECE': 'Electronics & Communication Engineering',
                            'MEC': 'Mechanical Engineering',
                            'CIV': 'Civil Engineering',
                            'EEE': 'Electrical & Electronics Engineering',
                            'CSD': 'Computer Science & Data Science',
                            'CSM': 'Computer Science & Machine Learning',
                            'CSO': 'Computer Science & Operations',
                            # Add more mappings as needed
                        }.get(branch, branch)
                        
                        if branch != preferred_branch:
                            eligible_branches.append({
                                'code': branch,
                                'name': branch_name,
                                'rank': int(row_rank) if pd.notna(row_rank) else 'N/A'
                            })
            except (KeyError, TypeError, ValueError) as e:
                print(f"Error processing branch: {str(e)}")
                continue
        
        # Find other colleges in the same location for the preferred branch
        preferred_place = college_details['place']
        preferred_college_name = college_details['name']
        
        eligible_colleges = []
        try:
            other_colleges_in_place = original_df[
                (original_df[place_col] == preferred_place) &
                (original_df['INSTCODE'] != inst_code) &
                (original_df['branch_code'] == branch_code)
            ]
            
            for _, row in other_colleges_in_place.iterrows():
                try:
                    if category_col in row and pd.notna(row[category_col]):
                        row_rank = pd.to_numeric(row[category_col], errors='coerce')
                        if pd.notna(row_rank) and rank <= row_rank:
                            college_name = row[name_col]
                            college_code = row['INSTCODE']
                            college_place = row[place_col] if place_col in row else 'N/A'
                            college_district = row[dist_col] if dist_col in row else 'N/A'
                            college_fee = row[fee_col] if fee_col in row and pd.notna(row[fee_col]) else 0
                            
                            eligible_colleges.append({
                                'code': college_code,
                                'name': college_name,
                                'place': college_place,
                                'district': college_district,
                                'fee': college_fee,
                                'rank': int(row_rank) if pd.notna(row_rank) else 'N/A'
                            })
                except (KeyError, TypeError, ValueError) as e:
                    print(f"Error processing college: {str(e)}")
                    continue
        except Exception as e:
            print(f"Error finding other colleges: {str(e)}")
        
        # Add alternative options to result
        if eligible_branches:
            result["alternative_branches"] = eligible_branches
        else:
            result["alternative_branches"] = []
            
        if eligible_colleges:
            result["eligible_colleges"] = eligible_colleges
        else:
            result["eligible_colleges"] = []
            
    except Exception as e:
        print(f"Error finding alternatives: {str(e)}")
        result["alternative_branches"] = []
        result["eligible_colleges"] = []
        
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['full_name']
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'error')
            return render_template('login.html')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        location = request.form['location']
        tenth_percentage = request.form['tenth_percentage']
        inter_percentage = request.form['inter_percentage']
        category = request.form['category']
        student_mobile = request.form['student_mobile']
        parent_mobile = request.form['parent_mobile']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')
        
        hashed_password = generate_password_hash(password)
        
        conn = get_db_connection()
        try:
            conn.execute(
                'INSERT INTO users (full_name, email, location, tenth_percentage, inter_percentage, category, student_mobile, parent_mobile, password) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (full_name, email, location, tenth_percentage, inter_percentage, category, student_mobile, parent_mobile, hashed_password)
            )
            conn.commit()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists', 'error')
        finally:
            conn.close()
    
    return render_template('signup.html')

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/get_colleges')
def get_colleges():
    global college_data
    if college_data is None:
        # If college_data is not loaded yet, load it
        load_and_preprocess_data()
    
    # Get unique colleges with their names
    colleges = college_data[['INSTCODE', 'NAME OF THE INSTITUTION']].drop_duplicates().sort_values('NAME OF THE INSTITUTION')
    
    # Convert to list of dictionaries for JSON response
    college_list = [{'code': row['INSTCODE'], 'name': row['NAME OF THE INSTITUTION']} 
                   for _, row in colleges.iterrows()]
    
    return jsonify(college_list)

@app.route('/get_branches/<college_code>')
def get_branches(college_code):
    global college_data, le_branch
    if college_data is None:
        # If college_data is not loaded yet, load it
        load_and_preprocess_data()
    
    # Filter branches for the selected college
    branches = college_data[college_data['INSTCODE'] == college_code]['branch_code'].unique()
    
    # Map branch codes to human-readable names
    branch_mapping = {
        'CSE': 'Computer Science & Engineering',
        'ECE': 'Electronics & Communication Engineering',
        'MEC': 'Mechanical Engineering',
        'CIV': 'Civil Engineering',
        'EEE': 'Electrical & Electronics Engineering',
        'CSD': 'Computer Science & Data Science',
        'CSM': 'Computer Science & Machine Learning',
        'CSO': 'Computer Science & Operations',
        # Add more mappings as needed
    }
    
    # Convert to list of dictionaries for JSON response
    branch_list = [{'code': branch, 'name': branch_mapping.get(branch, branch)} 
                  for branch in sorted(branches)]
    
    return jsonify(branch_list)

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    try:
        if request.is_json:
            data = request.get_json()
            return jsonify(predict_admission(
                int(data.get('rank', 0)),
                data.get('gender', ''),
                data.get('category', ''),
                data.get('preferredCollege', ''),
                data.get('preferredBranch', '')
            ))
        else:
            data = request.form.to_dict()
            
            rank = int(data.get('rank', 0))
            gender = data.get('gender', '')
            category = data.get('category', '')
            preferred_college = data.get('preferredCollege', '')
            preferred_branch = data.get('preferredBranch', '')
            
            # Store prediction data in session for the results page
            result = predict_admission(rank, gender, category, preferred_college, preferred_branch)
            
            # Store all necessary data in session
            session['prediction_result'] = result
            session['prediction_input'] = {
                'rank': rank,
                'gender': gender,
                'category': category,
                'preferred_college': preferred_college,
                'preferred_branch': preferred_branch
            }
            
            # Redirect to results page
            return redirect(url_for('results'))
            
    except Exception as e:
        flash(f"An error occurred: {str(e)}", "error")
        return redirect(url_for('home'))

@app.route('/results')
def results():
    if 'user_id' not in session:
        flash("Please login first", "error")
        return redirect(url_for('login'))
        
    if 'prediction_result' not in session:
        flash("No prediction data available. Please make a prediction first.", "error")
        return redirect(url_for('home'))
    
    result = session.get('prediction_result', {})
    input_data = session.get('prediction_input', {})
    
    # Check if there was an error in the prediction
    if result.get('status') == 'error':
        flash(result.get('message', 'An error occurred during prediction'), "error")
        return redirect(url_for('home'))
    
    # Get original branch name for display
    branch_code = input_data.get('preferred_branch', '')
    branch_mapping = {
        'CSE': 'Computer Science & Engineering',
        'ECE': 'Electronics & Communication Engineering',
        'MEC': 'Mechanical Engineering',
        'CIV': 'Civil Engineering',
        'EEE': 'Electrical & Electronics Engineering',
        'CSD': 'Computer Science & Data Science',
        'CSM': 'Computer Science & Machine Learning',
        'CSO': 'Computer Science & Operations',
        # Add more mappings as needed
    }
    branch_name = branch_mapping.get(branch_code, branch_code)
    
    # Check if eligible
    is_eligible = False
    if result.get('status') == 'success' and result.get('message').startswith('Eligible'):
        is_eligible = True
    
    # Get college name for display
    college_code = input_data.get('preferred_college', '')
    college_name = None
    
    try:
        # Try to get the college name from the original data
        if college_data is not None:
            college_info = college_data[college_data['INSTCODE'] == college_code]
            if not college_info.empty:
                college_name = college_info['NAME OF THE INSTITUTION'].iloc[0]
    except Exception as e:
        print(f"Error getting college name: {str(e)}")
    
    # Format alternative branches and colleges for display
    alternative_branches = result.get('alternative_branches', [])
    eligible_colleges = result.get('eligible_colleges', [])
    
    # Create cards for alternative branches
    alternative_branches_html = ''
    if alternative_branches:
        for branch in alternative_branches:
            alternative_branches_html += f'''
            <div class="branch-card">
                <div class="branch-name">{branch["name"]}</div>
                <div class="branch-code">Branch Code: {branch["code"]}</div>
                <div class="branch-rank">Closing Rank: {branch["rank"]}</div>
            </div>
            '''
    
    # Create a table for eligible colleges
    eligible_colleges_table = ''
    if eligible_colleges:
        eligible_colleges_table = '<table><tr><th>College Code</th><th>College Name</th><th>Place</th><th>District</th><th>Fee</th><th>Rank</th></tr>'
        for college in eligible_colleges:
            eligible_colleges_table += f'<tr><td>{college["code"]}</td><td>{college["name"]}</td><td>{college["place"]}</td><td>{college["district"]}</td><td>{college["fee"]}</td><td>{college["rank"]}</td></tr>'
        eligible_colleges_table += '</table>'
    
    # Determine display flags based on eligibility
    has_eligible_branches = len(alternative_branches) > 0
    has_eligible_colleges = len(eligible_colleges) > 0
    
    # Set the result status for UI display
    if is_eligible:
        result_status = "eligible"
        # When eligible, we only show other colleges
        show_alternative_branches = False
        show_eligible_colleges = has_eligible_colleges
    else:
        # When not eligible, check if we have any alternatives
        if has_eligible_branches or has_eligible_colleges:
            result_status = "not_eligible_with_alternatives"
            show_alternative_branches = has_eligible_branches
            show_eligible_colleges = has_eligible_colleges
        else:
            result_status = "not_eligible_no_alternatives"
            show_alternative_branches = False
            show_eligible_colleges = False
    
    return render_template('results.html',
        is_eligible=is_eligible,
        rank=input_data.get('rank'),
        gender=input_data.get('gender'),
        category=input_data.get('category'),
        preferred_branch=branch_name,
        college_details=result.get('college_details'),
        message=result.get('message'),
        alternative_branches=alternative_branches_html,
        eligible_colleges=eligible_colleges_table,
        college_name=college_name,
        result_status=result_status,
        show_alternative_branches=show_alternative_branches,
        show_eligible_colleges=show_eligible_colleges,
        has_eligible_branches=has_eligible_branches,
        has_eligible_colleges=has_eligible_colleges
    )

if __name__ == '__main__':
    print("Loading and preprocessing data...")
    df_boys, df_girls, le_inst, le_branch = load_and_preprocess_data()
    
    if os.path.exists('models.pkl'):
        print("Loading pre-trained models...")
        load_models()
    else:
        print("Training models...")
        models = train_models(df_boys, df_girls)
    
    app.run(debug=True)
