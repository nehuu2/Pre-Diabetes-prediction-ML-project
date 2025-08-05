import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import traceback
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="DiabetesPredict - Early Diabetes Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Beautiful, modern CSS
st.markdown("""
<style>
    /* Global styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom headers */
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Beautiful form container */
    .prediction-form {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Section headers */
    .section-header {
        color: #34495e;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    /* Result boxes with animations */
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .high-risk {
        background: linear-gradient(145deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        border: none;
        box-shadow: 0 10px 30px rgba(255,107,107,0.3);
    }
    
    .low-risk {
        background: linear-gradient(145deg, #51cf66 0%, #40c057 100%);
        color: white;
        border: none;
        box-shadow: 0 10px 30px rgba(81,207,102,0.3);
    }
    
    /* Metric cards */
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(102,126,234,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102,126,234,0.4);
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        padding: 0.75rem;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Success and info messages */
    .stSuccess {
        background: linear-gradient(145deg, #51cf66 0%, #40c057 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stInfo {
        background: linear-gradient(145deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Health tips styling */
    .health-tips {
        background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        import os
        model_path = 'notebooks/logistic_model.pkl'
        scaler_path = 'notebooks/scaler.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return model, scaler
        
    except Exception as e:
        return None, None

# Load data for visualization
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Data/cleaned_diabetes_data2.csv')
        return df
    except Exception as e:
        return None

# Prediction function
def predict_diabetes(input_data, model, scaler):
    try:
        # Validate input data
        if len(input_data) != 9:
            st.error(f"Expected 9 features, got {len(input_data)}")
            return None, None
        
        # Check for invalid values
        for i, value in enumerate(input_data):
            if not isinstance(value, (int, float)) or np.isnan(value):
                st.error(f"Invalid value at index {i}: {value}")
                return None, None
        
        # Convert to numpy array
        input_array = np.array(input_data).reshape(1, -1)
        
        # Scale the data
        scaled_data = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0]
        
        return prediction, probability
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Main app
def main():
    # Load model and data
    model, scaler = load_model()
    df = load_data()
    
    if model is None or scaler is None:
        st.error("❌ Failed to load the model. Please check if the model files exist.")
        return
    
    # Initialize session state for navigation
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Home"
    
    # Simple navigation
    st.sidebar.title("🏥 DiabetesPredict")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔮 Prediction", "📊 Analysis", "ℹ️ About"],
        index=["🏠 Home", "🔮 Prediction", "📊 Analysis", "ℹ️ About"].index(st.session_state.page)
    )
    
    # Update session state when page changes
    if page != st.session_state.page:
        st.session_state.page = page
        st.rerun()
    
    # Route to appropriate page
    if st.session_state.page == "🏠 Home":
        show_home_page()
    elif st.session_state.page == "🔮 Prediction":
        show_prediction_page(model, scaler)
    elif st.session_state.page == "📊 Analysis":
        show_analysis_page(df)
    elif st.session_state.page == "ℹ️ About":
        show_about_page()

def show_home_page():
    st.markdown('<h1 class="main-header">🏥 DiabetesPredict</h1>', unsafe_allow_html=True)
    
    # Hero section
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(145deg, rgba(255,255,255,0.9) 0%, rgba(248,249,250,0.9) 100%); border-radius: 20px; margin: 2rem 0;">
        <h2 style="color: #2c3e50; font-size: 1.8rem; margin-bottom: 1rem;">AI-Powered Diabetes Risk Assessment</h2>
        <p style="color: #7f8c8d; font-size: 1.1rem; line-height: 1.6;">
            Get instant, accurate predictions using our trained machine learning model. 
            Simply enter your health parameters and receive personalized risk assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("### 🚀 Key Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
            <h4 style="color: #3498db; margin-bottom: 0.5rem;">🤖 AI-Powered</h4>
            <p style="color: #7f8c8d; margin: 0;">Trained logistic regression model with 76% accuracy</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
            <h4 style="color: #3498db; margin-bottom: 0.5rem;">⚡ Instant Results</h4>
            <p style="color: #7f8c8d; margin: 0;">Get predictions in seconds with detailed probability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
            <h4 style="color: #3498db; margin-bottom: 0.5rem;">🔒 Privacy First</h4>
            <p style="color: #7f8c8d; margin: 0;">All data processed locally, nothing stored</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
            <h4 style="color: #3498db; margin-bottom: 0.5rem;">📊 Data Insights</h4>
            <p style="color: #7f8c8d; margin: 0;">Interactive visualizations and analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works section
    st.markdown("### 📋 How It Works")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(145deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 1rem 0;">
            <h3 style="margin-bottom: 0.5rem;">1️⃣ Enter Data</h3>
            <p style="margin: 0; opacity: 0.9;">Input your health parameters like glucose, blood pressure, BMI, etc.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(145deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 1rem 0;">
            <h3 style="margin-bottom: 0.5rem;">2️⃣ AI Analysis</h3>
            <p style="margin: 0; opacity: 0.9;">Our model analyzes your data against thousands of samples</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(145deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 1rem 0;">
            <h3 style="margin-bottom: 0.5rem;">3️⃣ Get Results</h3>
            <p style="margin: 0; opacity: 0.9;">Receive personalized risk assessment with health tips</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(145deg, #51cf66 0%, #40c057 100%); border-radius: 20px; margin: 2rem 0; color: white;">
        <h3 style="margin-bottom: 1rem;">Ready to Predict Your Risk?</h3>
        <p style="margin-bottom: 1.5rem; opacity: 0.9;">Click the button below to start your health assessment!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Predict Now", use_container_width=True, type="primary"):
            # Update sidebar selection
            st.session_state.page = "🔮 Prediction"
            st.rerun()

def show_prediction_page(model, scaler):
    st.markdown('<h1 class="main-header">🔮 Diabetes Prediction</h1>', unsafe_allow_html=True)
    
    # Back to home button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.page = "🏠 Home"
            st.rerun()
    
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(145deg, rgba(255,255,255,0.9) 0%, rgba(248,249,250,0.9) 100%); border-radius: 15px; margin: 1rem 0;">
        <p style="color: #2c3e50; font-size: 1.1rem; margin: 0;">
            Enter your health parameters below to get your personalized diabetes risk assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick test button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🧪 Quick Test (Sample Data)", use_container_width=True):
            st.session_state.quick_test = True
            st.rerun()
    
    # Initialize form values
    if st.session_state.get('quick_test', False):
        pregnancies_val = 6
        glucose_val = 148
        blood_pressure_val = 72
        skin_thickness_val = 35
        insulin_val = 125
        bmi_val = 33.6
        diabetes_pedigree_val = 0.627
        age_val = 50
        st.success("🧪 Using sample data for testing!")
    else:
        pregnancies_val = 1
        glucose_val = 120
        blood_pressure_val = 80
        skin_thickness_val = 20
        insulin_val = 80
        bmi_val = 25.0
        diabetes_pedigree_val = 0.5
        age_val = 30
    
    with st.container():
        st.markdown('<div class="prediction-form">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(145deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <h3 style="margin: 0; text-align: center;">👤 Personal Information</h3>
            </div>
            """, unsafe_allow_html=True)
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=pregnancies_val)
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=age_val)
            
            st.markdown("""
            <div style="background: linear-gradient(145deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                <h3 style="margin: 0; text-align: center;">🩸 Blood Tests</h3>
            </div>
            """, unsafe_allow_html=True)
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=glucose_val)
            insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=insulin_val)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(145deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <h3 style="margin: 0; text-align: center;">📏 Physical Measurements</h3>
            </div>
            """, unsafe_allow_html=True)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=blood_pressure_val)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=skin_thickness_val)
            bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=bmi_val, step=0.1)
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=diabetes_pedigree_val, step=0.001)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction button
        if st.button("🔮 Predict Risk", use_container_width=True, type="primary"):
            # Prepare input data - need to include index column (0) as first feature
            input_data = [
                0,  # Index column (always 0 for new predictions)
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, diabetes_pedigree, age
            ]
            
            st.info(f"🔍 Processing prediction with data: {input_data}")
            
            # Make prediction
            prediction, probability = predict_diabetes(input_data, model, scaler)
            
            if prediction is not None:
                st.markdown("---")
                st.markdown("## 📊 Prediction Results")
                
                # Display result
                if prediction == 1:
                    st.markdown("""
                    <div class="result-box high-risk">
                        <h2>⚠️ HIGH RISK</h2>
                        <p>Our analysis indicates a higher probability of diabetes.</p>
                        <p><strong>Recommendation:</strong> Please consult with a healthcare professional.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-box low-risk">
                        <h2>✅ LOW RISK</h2>
                        <p>Our analysis indicates a lower probability of diabetes.</p>
                        <p><strong>Recommendation:</strong> Continue maintaining a healthy lifestyle.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show probability
                risk_probability = probability[1] if prediction == 1 else probability[0]
                st.metric("Risk Probability", f"{risk_probability:.1%}")
                
                # Show input summary
                st.markdown("### 📋 Your Input Summary")
                input_df = pd.DataFrame({
                    'Parameter': ['Index', 'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                                'Insulin', 'BMI', 'Diabetes Pedigree', 'Age'],
                    'Value': input_data
                })
                st.dataframe(input_df, use_container_width=True)
                
                # Health tips
                st.markdown("### 💡 Health Tips")
                if prediction == 1:
                    st.markdown("""
                    <div class="health-tips">
                        <h4 style="color: #e74c3c; margin-bottom: 1rem;">⚠️ Important Recommendations:</h4>
                        <ul style="color: #2c3e50; line-height: 1.8;">
                            <li><strong>Monitor blood sugar</strong> regularly</li>
                            <li><strong>Exercise</strong> at least 30 minutes daily</li>
                            <li><strong>Eat a balanced diet</strong> low in processed sugars</li>
                            <li><strong>Maintain healthy weight</strong></li>
                            <li><strong>Schedule regular check-ups</strong> with your doctor</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="health-tips">
                        <h4 style="color: #27ae60; margin-bottom: 1rem;">✅ Keep Up the Good Work:</h4>
                        <ul style="color: #2c3e50; line-height: 1.8;">
                            <li><strong>Continue healthy habits</strong></li>
                            <li><strong>Regular exercise</strong> and balanced diet</li>
                            <li><strong>Annual health check-ups</strong></li>
                            <li><strong>Monitor family history</strong></li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("❌ Prediction failed. Please check your input values and try again.")
        
        # Reset quick test after processing
        if st.session_state.get('quick_test', False):
            st.session_state.quick_test = False

def show_analysis_page(df):
    if df is None:
        st.error("Data not available for analysis.")
        return
    
    st.markdown('<h1 class="main-header">📊 Data Analysis</h1>', unsafe_allow_html=True)
    
    # Back to home button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.page = "🏠 Home"
            st.rerun()
    
    # Basic statistics
    st.markdown("## 📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Diabetes Cases", df['Outcome'].sum())
    with col3:
        st.metric("Non-Diabetes Cases", len(df) - df['Outcome'].sum())
    with col4:
        st.metric("Diabetes Rate", f"{(df['Outcome'].mean()*100):.1f}%")
    
    # Key metrics distribution
    st.markdown("## 📊 Key Health Metrics")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Glucose Distribution', 'BMI Distribution', 'Age Distribution', 'Blood Pressure Distribution')
    )
    
    fig.add_trace(go.Histogram(x=df['Glucose'], name='Glucose'), row=1, col=1)
    fig.add_trace(go.Histogram(x=df['BMI'], name='BMI'), row=1, col=2)
    fig.add_trace(go.Histogram(x=df['Age'], name='Age'), row=2, col=1)
    fig.add_trace(go.Histogram(x=df['BloodPressure'], name='Blood Pressure'), row=2, col=2)
    
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("## 🔗 Feature Correlations")
    correlation_matrix = df.corr()
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",
        title="Feature Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    st.markdown('<h1 class="main-header">ℹ️ About DiabetesPredict</h1>', unsafe_allow_html=True)
    
    # Back to home button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.page = "🏠 Home"
            st.rerun()
    
    st.markdown("""
    ## 🎯 Project Overview
    
    DiabetesPredict is a machine learning tool designed to assess diabetes risk based on health parameters.
    
    ### 🔬 Model Information
    - **Algorithm**: Logistic Regression
    - **Accuracy**: 76.04%
    - **Features**: 8 health parameters
    - **Dataset**: Pima Indians Diabetes Database
    
    ### 📋 Features Used
    1. **Pregnancies**: Number of times pregnant
    2. **Glucose**: Plasma glucose concentration (mg/dL)
    3. **Blood Pressure**: Diastolic blood pressure (mm Hg)
    4. **Skin Thickness**: Triceps skin fold thickness (mm)
    5. **Insulin**: 2-Hour serum insulin (mu U/ml)
    6. **BMI**: Body mass index
    7. **Diabetes Pedigree Function**: Diabetes family history
    8. **Age**: Age in years
    
    ### ⚠️ Important Disclaimer
    This tool is for **educational and research purposes only**. It is **NOT a substitute for professional medical advice**. 
    Always consult with a qualified healthcare provider for any health concerns or decisions.
    
    ### 🛡️ Privacy & Security
    - All data entered is processed locally
    - No personal information is stored or transmitted
    - Results are for educational purposes only
    
    ### 🛠️ Technical Details
    - **Framework**: Streamlit
    - **Language**: Python
    - **ML Library**: Scikit-learn
    - **Visualization**: Plotly
    - **Data Processing**: Pandas, NumPy
    """)

if __name__ == "__main__":
    main() 