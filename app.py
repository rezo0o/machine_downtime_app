import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Machine Downtime Analysis", layout="wide")

# Title and description
st.title("Machine Downtime Analysis Dashboard")
st.markdown("""
This dashboard analyzes machine downtime data and provides insights through various visualizations 
and machine learning models. Use the sidebar to navigate through different analyses.
""")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('machine_downtime.csv')
    return df

# Load data
df = load_data()

# Sidebar navigation
page = st.sidebar.selectbox(
    "Choose a page", 
    ["Data Overview", "Exploratory Analysis", "Machine Learning Models"]
)

if page == "Data Overview":
    st.header("Data Overview")
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Machines", df['Machine_ID'].nunique())
    with col3:
        st.metric("Assembly Lines", df['Assembly_Line_No'].nunique())
    
    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    # Show basic statistics
    st.subheader("Basic Statistics")
    st.dataframe(df.describe())
    
    # Distribution of machine downtime
    st.subheader("Distribution of Machine Downtime")
    fig = px.pie(df, names='Downtime', title='Distribution of Machine Downtime')
    st.plotly_chart(fig)

elif page == "Exploratory Analysis":
    st.header("Exploratory Analysis")
    
    # Select features for analysis
    numerical_cols = df.select_dtypes(include=['float64']).columns
    
    # Correlation heatmap
    st.subheader("Correlation Analysis")
    corr = df[numerical_cols].corr()
    fig = px.imshow(corr, 
                    title="Feature Correlation Heatmap",
                    color_continuous_scale='RdBu_r')
    st.plotly_chart(fig)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select Feature", numerical_cols)
    
    fig = px.histogram(df, x=feature, color='Downtime',
                      title=f"Distribution of {feature} by Downtime Status",
                      marginal="box")
    st.plotly_chart(fig)
    
    # Time series analysis
    st.subheader("Machine Status Over Time")
    df['Date'] = pd.to_datetime(df['Date'])
    daily_failures = df[df['Downtime'] == 'Machine_Failure'].groupby('Date').size().reset_index()
    daily_failures.columns = ['Date', 'Failures']
    
    fig = px.line(daily_failures, x='Date', y='Failures',
                  title='Machine Failures Over Time')
    st.plotly_chart(fig)

elif page == "Machine Learning Models":
    st.header("Machine Learning Models")
    
    # Prepare data for modeling
    X = df.drop(['Downtime', 'Date'], axis=1)
    y = df['Downtime']
    
    # Model selection
    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    )
    
    # Train model button
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create preprocessing pipeline
            numerical_features = X.select_dtypes(include=['float64']).columns
            categorical_features = X.select_dtypes(include=['object']).columns
            
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer([
                ('numerical', numerical_pipeline, numerical_features),
                ('categorical', categorical_pipeline, categorical_features)
            ])
            
            # Select model
            if model_choice == "Logistic Regression":
                model = LogisticRegression(random_state=42)
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            else:
                model = GradientBoostingClassifier(random_state=42)
            
            # Create and train pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)
            
            # Show results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
            
            with col2:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, 
                              labels=dict(x="Predicted", y="Actual"),
                              title="Confusion Matrix")
                st.plotly_chart(fig)
            
            # ROC Curve
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test == 'Machine_Failure', 
                                  y_pred_proba[:, 1])
            fig = px.line(x=fpr, y=tpr, 
                         title=f'ROC Curve (AUC = {auc(fpr, tpr):.3f})',
                         labels={'x': 'False Positive Rate', 
                                'y': 'True Positive Rate'})
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            st.plotly_chart(fig)
            
            # Feature importance
            if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                importances = pipeline.named_steps['classifier'].feature_importances_
            elif hasattr(pipeline.named_steps['classifier'], 'coef_'):
                importances = np.abs(pipeline.named_steps['classifier'].coef_[0])
            
            # Get feature names after preprocessing
            feature_names = (list(numerical_features) + 
                           list(categorical_features))
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            st.subheader("Feature Importance")
            fig = px.bar(importance_df, x='importance', y='feature',
                        title='Feature Importance',
                        orientation='h')
            st.plotly_chart(fig)

# Add footer
st.markdown("""
---
Created with Streamlit for Machine Downtime Analysis
""")