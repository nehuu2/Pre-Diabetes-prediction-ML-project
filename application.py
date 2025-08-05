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

# Simple, clean CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-form {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .low-risk {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
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
    
    # Simple navigation
    st.sidebar.title("🏥 DiabetesPredict")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔮 Prediction", "📊 Analysis", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Prediction":
        show_prediction_page(model, scaler)
    elif page == "📊 Analysis":
        show_analysis_page(df)
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    st.markdown('<h1 class="main-header">🏥 DiabetesPredict</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to DiabetesPredict
    
    This application uses machine learning to predict the risk of diabetes based on health parameters.
    
    ### How it works:
    1. **Enter your health data** in the prediction form
    2. **Get instant results** with risk assessment
    3. **View detailed analysis** of your data
    
    ### Features:
    - 🤖 **AI-Powered Prediction**: Uses trained logistic regression model
    - 📊 **Data Visualization**: Interactive charts and analysis
    - 🔒 **Privacy First**: All data processed locally
    - ⚡ **Instant Results**: Get predictions in seconds
    
    ---
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📝 Step 1
        Enter your health parameters like glucose, blood pressure, BMI, etc.
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Step 2
        Our AI model analyzes your data against thousands of samples.
        """)
    
    with col3:
        st.markdown("""
        ### ✅ Step 3
        Get your risk assessment with detailed probability.
        """)
    
    st.markdown("---")
    st.markdown("### Ready to predict?")
    st.markdown("Use the sidebar navigation to go to the **🔮 Prediction** page and test your ML model!")

def show_prediction_page(model, scaler):
    st.markdown('<h1 class="main-header">🔮 Diabetes Prediction</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Enter your health parameters below to get your diabetes risk assessment.
    """)
    
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
            st.markdown("### Personal Information")
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=pregnancies_val)
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=age_val)
            
            st.markdown("### Blood Tests")
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=glucose_val)
            insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=insulin_val)
        
        with col2:
            st.markdown("### Physical Measurements")
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
                    - **Monitor blood sugar** regularly
                    - **Exercise** at least 30 minutes daily
                    - **Eat a balanced diet** low in processed sugars
                    - **Maintain healthy weight**
                    - **Schedule regular check-ups** with your doctor
                    """)
                else:
                    st.markdown("""
                    - **Continue healthy habits**
                    - **Regular exercise** and balanced diet
                    - **Annual health check-ups**
                    - **Monitor family history**
                    """)
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