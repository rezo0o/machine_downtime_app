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
    df['Date'] = pd.to_datetime(df['Date'])
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
    
    # Date range
    st.subheader("Data Time Range")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Start Date", df['Date'].min().strftime('%Y-%m-%d'))
    with col2:
        st.metric("End Date", df['Date'].max().strftime('%Y-%m-%d'))
    
    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    # Show basic statistics
    st.subheader("Basic Statistics")
    st.dataframe(df.describe())
    
    # Distribution of machine downtime
    st.subheader("Distribution of Machine Downtime")
    fig = px.pie(df, names='Downtime', title='Distribution of Machine Downtime')
    st.plotly_chart(fig, use_container_width=True)
    
    # Downtime by assembly line
    st.subheader("Downtime by Assembly Line")
    downtime_by_line = df.groupby(['Assembly_Line_No', 'Downtime']).size().unstack()
    fig = px.bar(downtime_by_line, 
                 title='Machine Downtime Distribution by Assembly Line',
                 barmode='group')
    st.plotly_chart(fig, use_container_width=True)

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
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    col1, col2 = st.columns(2)
    with col1:
        feature = st.selectbox("Select Feature", numerical_cols)
    with col2:
        bin_count = st.slider("Number of bins", min_value=10, max_value=100, value=30)
    
    fig = px.histogram(df, x=feature, color='Downtime',
                      title=f"Distribution of {feature} by Downtime Status",
                      nbins=bin_count,
                      marginal="box")
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    st.subheader("Machine Status Over Time")
    # Resample data by day
    daily_failures = df[df['Downtime'] == 'Machine_Failure'].groupby(
        pd.Grouper(key='Date', freq='D')
    ).size().reset_index()
    daily_failures.columns = ['Date', 'Failures']
    
    fig = px.line(daily_failures, x='Date', y='Failures',
                  title='Machine Failures Over Time')
    fig.update_traces(line_color='red')
    st.plotly_chart(fig, use_container_width=True)
    
    # Machine-wise analysis
    st.subheader("Machine-wise Analysis")
    machine_stats = df.groupby('Machine_ID').agg({
        'Downtime': lambda x: (x == 'Machine_Failure').mean(),
        'Hydraulic_Pressure(bar)': 'mean',
        'Coolant_Temperature': 'mean',
        'Spindle_Vibration': 'mean'
    }).round(3)
    machine_stats.columns = ['Failure Rate', 'Avg Hydraulic Pressure', 
                           'Avg Coolant Temp', 'Avg Spindle Vibration']
    st.dataframe(machine_stats)

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
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
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
                              title="Confusion Matrix",
                              color_continuous_scale='RdBu')
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
                # For logistic regression, normalize the coefficients
                coef = pipeline.named_steps['classifier'].coef_[0]
                importances = np.abs(coef) / np.sum(np.abs(coef))  # Normalize to sum to 1
            
            # Get feature names after preprocessing
            cat_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['categorical'].get_feature_names_out(categorical_features)
            feature_names = list(numerical_features) + list(cat_feature_names)
            
            # Ensure lengths match
            if len(feature_names) == len(importances):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                st.subheader("Feature Importance")
                fig = px.bar(importance_df.head(10), x='importance', y='feature',
                            title='Top 10 Most Important Features',
                            orientation='h')
                st.plotly_chart(fig)
            else:
                st.warning("Feature importance visualization unavailable due to dimensionality mismatch")

# Add footer
st.markdown("""
---
Created with Streamlit for Machine Downtime Analysis
""")
