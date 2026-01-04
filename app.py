import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score)
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Student Impact Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

custom_css = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #f1f5f9;
    }
    .main-header {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
        text-align: center;
    }
    .prediction-card-pass {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border-left: 8px solid #10b981;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4);
        margin: 15px 0;
    }
    .prediction-card-fail {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border-left: 8px solid #ef4444;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.4);
        margin: 15px 0;
    }
    h1 {
        color: #ffffff;
        font-weight: 800;
        margin: 0;
    }
    p {
        font-size: 1.1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        padding: 12px 32px !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        width: 100% !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

@st.cache_resource
def train_model():
    # Load the specific dataset
    try:
        df = pd.read_csv('ai_impact_student_performance_dataset.csv')
    except FileNotFoundError:
        st.error("Dataset 'ai_impact_student_performance_dataset.csv' not found. Please upload it.")
        return None, None, None, None, None, None

    # Create target variable: Pass if final_score >= 50
    df['passed'] = (df['final_score'] >= 50).astype(int)
    
    # Separate features and target
    # Exclude final_score (target proxy) and passed (target)
    X = df.drop(['passed', 'final_score'], axis=1)
    y = df['passed']
    
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Handle missing values if any
    X = X.fillna(X.mean(numeric_only=True))
    
    scaler = StandardScaler()
    feature_names = X.columns.tolist()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return model, feature_names, label_encoders, metrics, test_data, scaler

model, feature_names, label_encoders, metrics, test_data, scaler = train_model()

def prepare_input(input_df):
    X_input = input_df.copy()
    
    # Encode categorical inputs using loaded encoders
    for col in label_encoders:
        if col in X_input.columns:
            try:
                # Handle unknown labels by defaulting to 0 or handling exception
                val = X_input[col].iloc[0]
                if val in label_encoders[col].classes_:
                    X_input[col] = label_encoders[col].transform(X_input[col].astype(str))
                else:
                    X_input[col] = 0 # Default/Unknown
            except:
                X_input[col] = 0
    
    X_input = X_input.fillna(0)
    
    # Ensure all features exist
    for feat in feature_names:
        if feat not in X_input.columns:
            X_input[feat] = 0
            
    # Reorder columns to match training
    X_input = X_input[feature_names]
    
    # Scale features
    X_scaled = scaler.transform(X_input)
    
    return pd.DataFrame(X_scaled, columns=feature_names)

def get_top_features(n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:n]
    return [(feature_names[i], importances[i]) for i in indices]

st.markdown('<div class="main-header"><h1>AI Student Impact Predictor</h1><p>Advanced ML Model based on Real Student Data</p></div>', unsafe_allow_html=True)

if model is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Analytics", "Features", "About"])

    with tab1:
        st.header("Student Profile")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics & Academic")
            grade = st.selectbox("Grade Level", ['10th', '11th', '12th', '1st Year', '2nd Year', '3rd Year'])
            gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
            age = st.number_input("Age", 15, 35, 20)
            concept = st.slider("Concept Understanding (1-10)", 1, 10, 6)
            consistency = st.slider("Study Consistency (1-10)", 1.0, 10.0, 5.5)
            particip = st.slider("Class Participation (1-10)", 1, 10, 6)
            improve = st.number_input("Improvement Rate (%)", -20.0, 50.0, 10.0)

        with col2:
            st.subheader("Lifestyle & Habits")
            study = st.number_input("Study Hours/Day", 0.0, 15.0, 3.5)
            sleep = st.number_input("Sleep Hours/Day", 0.0, 12.0, 7.0)
            social = st.number_input("Social Media Hours/Day", 0.0, 12.0, 2.5)
            tutor = st.number_input("Tutoring Hours/Week", 0.0, 20.0, 2.0)
            
        with col3:
            st.subheader("AI Usage Profile")
            is_ai_user = st.checkbox("Self-Reported AI User", value=True)
            ai_tool = st.selectbox("Primary AI Tool", ['ChatGPT', 'Gemini', 'Copilot', 'Claude', 'ChatGPT+Gemini', 'Unknown'])
            ai_purpose = st.selectbox("Usage Purpose", ['Exam Prep', 'Notes', 'Doubt Solving', 'Coding', 'Homework', 'Unknown'])
            ai_time = st.number_input("AI Usage (Min/Day)", 0, 300, 60)
            ai_prompts = st.number_input("Prompts per Week", 0, 200, 50)
            ai_content = st.slider("AI Generated Content %", 0, 100, 30)
            ai_depend = st.slider("AI Dependency Score (1-10)", 1, 10, 5)
            ai_ethics = st.slider("AI Ethics Score (1-10)", 1, 10, 5)

        st.divider()
        
        # Prepare input dictionary
        uses_ai_val = 1 if is_ai_user else 0
        
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'grade_level': [grade],
            'study_hours_per_day': [study],
            'uses_ai': [uses_ai_val],
            'ai_usage_time_minutes': [ai_time],
            'ai_tools_used': [ai_tool],
            'ai_usage_purpose': [ai_purpose],
            'ai_dependency_score': [ai_depend],
            'ai_generated_content_percentage': [ai_content],
            'ai_prompts_per_week': [ai_prompts],
            'ai_ethics_score': [ai_ethics],
            'concept_understanding_score': [concept],
            'study_consistency_index': [consistency],
            'improvement_rate': [improve],
            'sleep_hours': [sleep],
            'social_media_hours': [social],
            'tutoring_hours': [tutor],
            'class_participation_score': [particip]
        })
        
        if st.button("Run Prediction", use_container_width=True):
            prepared = prepare_input(input_data)
            pred = model.predict(prepared)[0]
            proba = model.predict_proba(prepared)[0]
            conf = max(proba)
            
            st.divider()
            
            # Using 50 as pass mark proxy
            if pred == 1:
                st.markdown('<div class="prediction-card-pass"><h2 style="color: #ffffff;">LIKELY TO PASS - {:.1%} Confidence</h2><p>Based on the provided metrics, the student is projected to score ≥ 50%.</p></div>'.format(conf), unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-card-fail"><h2 style="color: #ffffff;">AT RISK - {:.1%} Risk</h2><p>The model suggests a high risk of scoring < 50%.</p></div>'.format(1-conf), unsafe_allow_html=True)
            
            col_prob1, col_prob2 = st.columns(2)
            
            with col_prob1:
                fig = go.Figure(data=[go.Bar(x=['Fail Risk', 'Pass Prob'], y=[proba[0], proba[1]], marker_color=['#ef4444', '#10b981'])])
                fig.update_layout(title="Prediction Probability", template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_prob2:
                fig = go.Figure(go.Indicator(mode="gauge+number", value=conf*100, title="Confidence Level", gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#10b981" if pred==1 else "#ef4444"))))
                fig.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Top Influencing Factors")
            top = get_top_features(4)
            cols = st.columns(4)
            for i, (feat, imp) in enumerate(top):
                cols[i].metric(feat.replace('_', ' ').title(), f"{imp:.1%}")

    with tab2:
        st.header("Model Performance")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        m2.metric("Precision", f"{metrics['precision']:.2%}")
        m3.metric("Recall", f"{metrics['recall']:.2%}")
        m4.metric("F1-Score", f"{metrics['f1']:.2%}")
        m5.metric("AUC", f"{metrics['auc']:.3f}")
        
        st.divider()
        
        col_feat, col_conf = st.columns(2)
        
        with col_feat:
            top_all = get_top_features(15)
            fig = go.Figure(data=[go.Bar(y=[f[0].replace('_',' ').title() for f in top_all], x=[f[1] for f in top_all], orientation='h', marker_color=[f[1] for f in top_all], marker_colorscale='Viridis')])
            fig.update_layout(title="Feature Importance", xaxis_title="Importance", height=500, template="plotly_dark", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        
        with col_conf:
            y_test = test_data['y_test']
            y_pred = test_data['y_pred']
            cm = confusion_matrix(y_test, y_pred)
            fig = go.Figure(data=go.Heatmap(z=cm, x=['Fail (Pred)', 'Pass (Pred)'], y=['Fail (True)', 'Pass (True)'], text=cm, texttemplate='%{text}', colorscale='Blues'))
            fig.update_layout(title="Confusion Matrix", height=400, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, test_data['y_pred_proba'])
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line=dict(color='#10b981', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='gray', dash='dash')))
            fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=300, template="plotly_dark")
            st.plotly_chart(fig_roc, use_container_width=True)

    with tab3:
        st.header("Dataset Features")
        st.markdown("The model is trained on 19 features extracted from `ai_impact_student_performance_dataset.csv`.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Academic")
            st.write("- **Grade Level**: 10th - 3rd Year")
            st.write("- **Concept Understanding**: 1-10 Scale")
            st.write("- **Study Consistency**: 1-10 Index")
            st.write("- **Improvement Rate**: % Change")
            st.write("- **Class Participation**: 1-10 Score")
            
        with c2:
            st.subheader("AI Usage")
            st.write("- **Tools**: ChatGPT, Gemini, Copilot, etc.")
            st.write("- **Usage Time**: Minutes per day")
            st.write("- **Dependency**: 1-10 Score")
            st.write("- **Content %**: % of work AI-generated")
            st.write("- **Ethics Score**: 1-10 Scale")
            
        with c3:
            st.subheader("Personal")
            st.write("- **Study Hours**: Daily avg")
            st.write("- **Sleep Hours**: Daily avg")
            st.write("- **Social Media**: Daily avg")
            st.write("- **Tutoring**: Hours per week")

    with tab4:
        st.header("About")
        st.markdown("""
        ### AI Student Impact Predictor
        This application uses a Random Forest Classifier to predict student performance based on the `ai_impact_student_performance_dataset.csv`.
        
        **Target Variable:**
        The model predicts if a student will **PASS** (Final Score ≥ 50) or **FAIL** (Final Score < 50).
        
        **Model Details:**
        - **Algorithm**: Random Forest (200 Trees)
        - **Features**: 19 Independent Variables
        - **Data Source**: Uploaded CSV Dataset
        """)

else:
    st.error("Model could not be trained. Please ensure the dataset file is present.")