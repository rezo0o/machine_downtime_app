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

# Page configuration
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
    """
    Load and perform basic preprocessing on the machine downtime data.
    Uses Streamlit's caching to avoid reloading on each interaction.
    """
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

# Data Overview page
if page == "Data Overview":
    st.header("Data Overview")
    
    # Basic statistics in metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Machines", df['Machine_ID'].nunique())
    with col3:
        st.metric("Assembly Lines", df['Assembly_Line_No'].nunique())
    
    # Date range information
    st.subheader("Data Time Range")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Start Date", df['Date'].min().strftime('%Y-%m-%d'))
    with col2:
        st.metric("End Date", df['Date'].max().strftime('%Y-%m-%d'))
    
    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    # Show basic statistics for numerical columns
    st.subheader("Basic Statistics")
    st.dataframe(df.describe())
    
    # Distribution of machine downtime (pie chart)
    st.subheader("Distribution of Machine Downtime")
    fig = px.pie(df, names='Downtime', title='Distribution of Machine Downtime',
                color_discrete_map={'Machine_Failure': 'red', 'No_Machine_Failure': 'green'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Downtime by assembly line (bar chart)
    st.subheader("Downtime by Assembly Line")
    downtime_by_line = df.groupby(['Assembly_Line_No', 'Downtime']).size().unstack()
    fig = px.bar(downtime_by_line, 
                title='Machine Downtime Distribution by Assembly Line',
                barmode='group',
                color_discrete_map={'Machine_Failure': 'red', 'No_Machine_Failure': 'green'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Downtime by machine
    st.subheader("Downtime by Machine")
    downtime_by_machine = df.groupby(['Machine_ID', 'Downtime']).size().unstack()
    fig = px.bar(downtime_by_machine,
                title='Machine Downtime Distribution by Machine ID',
                barmode='group',
                color_discrete_map={'Machine_Failure': 'red', 'No_Machine_Failure': 'green'})
    st.plotly_chart(fig, use_container_width=True)

# Exploratory Analysis page
elif page == "Exploratory Analysis":
    st.header("Exploratory Analysis")
    
    # Select features for analysis
    numerical_cols = df.select_dtypes(include=['float64']).columns
    
    # Correlation heatmap
    st.subheader("Correlation Analysis")
    corr = df[numerical_cols].corr()
    fig = px.imshow(corr, 
                    title="Feature Correlation Heatmap",
                    color_continuous_scale='RdBu_r',
                    labels=dict(color="Correlation"))
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions with user selection
    st.subheader("Feature Distributions")
    col1, col2 = st.columns(2)
    with col1:
        feature = st.selectbox("Select Feature", numerical_cols)
    with col2:
        bin_count = st.slider("Number of bins", min_value=10, max_value=100, value=30)
    
    fig = px.histogram(df, x=feature, color='Downtime',
                      title=f"Distribution of {feature} by Downtime Status",
                      nbins=bin_count,
                      marginal="box",
                      color_discrete_map={'Machine_Failure': 'red', 'No_Machine_Failure': 'green'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot matrix for selected features
    st.subheader("Feature Relationships")
    selected_features = st.multiselect(
        "Select features for scatter plot matrix (2-4 recommended)", 
        options=list(numerical_cols),
        default=list(numerical_cols)[:3]
    )
    
    if len(selected_features) > 1:
        fig = px.scatter_matrix(
            df, 
            dimensions=selected_features, 
            color='Downtime',
            color_discrete_map={'Machine_Failure': 'red', 'No_Machine_Failure': 'green'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least 2 features for the scatter plot matrix")
    
    # Time series analysis of failures
    st.subheader("Machine Status Over Time")
    # Resample data by day and count failures
    daily_failures = df[df['Downtime'] == 'Machine_Failure'].groupby(
        pd.Grouper(key='Date', freq='D')
    ).size().reset_index()
    daily_failures.columns = ['Date', 'Failures']
    
    # Add a date range selector
    time_options = ["All Time", "Last 3 Months", "Last 6 Months"]
    time_selection = st.selectbox("Select Time Period", time_options)
    
    # Filter based on selection
    if time_selection == "Last 3 Months":
        date_threshold = daily_failures['Date'].max() - pd.Timedelta(days=90)
        filtered_failures = daily_failures[daily_failures['Date'] > date_threshold]
    elif time_selection == "Last 6 Months":
        date_threshold = daily_failures['Date'].max() - pd.Timedelta(days=180)
        filtered_failures = daily_failures[daily_failures['Date'] > date_threshold]
    else:
        filtered_failures = daily_failures
    
    fig = px.line(filtered_failures, x='Date', y='Failures',
                 title='Machine Failures Over Time')
    fig.update_traces(line_color='red')
    st.plotly_chart(fig, use_container_width=True)
    
    # Machine-wise analysis with detailed statistics
    st.subheader("Machine-wise Analysis")
    machine_stats = df.groupby('Machine_ID').agg({
        'Downtime': lambda x: (x == 'Machine_Failure').mean(),
        'Hydraulic_Pressure(bar)': ['mean', 'std'],
        'Coolant_Temperature': ['mean', 'std'],
        'Spindle_Vibration': ['mean', 'std'],
        'Tool_Vibration': ['mean', 'std']
    }).round(3)
    
    # Flatten the multi-index columns for better display
    machine_stats.columns = [f"{col[0]}_{col[1]}" if col[1] != "" else col[0] 
                           for col in machine_stats.columns]
    
    # Rename columns for clarity
    machine_stats = machine_stats.rename(columns={'Downtime_<lambda_0>': 'Failure_Rate'})
    
    st.dataframe(machine_stats)
    
    # Visualize machine failure rates
    fig = px.bar(machine_stats.reset_index(), x='Machine_ID', y='Failure_Rate',
                title='Machine Failure Rates',
                color='Failure_Rate',
                color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)

# Machine Learning Models page
elif page == "Machine Learning Models":
    st.header("Machine Learning Models")
    
    # Prepare data for modeling
    X = df.drop(['Downtime', 'Date'], axis=1)
    y = df['Downtime']
    
    # Model selection with hyperparameters
    model_container = st.container()
    with model_container:
        col1, col2 = st.columns(2)
        
        with col1:
            model_choice = st.selectbox(
                "Select Model",
                ["Logistic Regression", "Random Forest", "Gradient Boosting"]
            )
        
        with col2:
            test_size = st.slider("Test set size (%)", 10, 50, 20, 5) / 100
    
    # Advanced options container
    with st.expander("Advanced Model Options"):
        if model_choice == "Logistic Regression":
            C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0, 0.01)
            max_iter = st.slider("Maximum iterations", 100, 1000, 100, 100)
            solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
        
        elif model_choice == "Random Forest":
            n_estimators = st.slider("Number of trees", 10, 500, 100, 10)
            max_depth = st.slider("Maximum tree depth", 2, 30, 10, 1)
            min_samples_split = st.slider("Minimum samples to split", 2, 20, 2, 1)
        
        elif model_choice == "Gradient Boosting":
            n_estimators = st.slider("Number of boosting stages", 10, 500, 100, 10)
            learning_rate = st.slider("Learning rate", 0.01, 1.0, 0.1, 0.01)
            max_depth = st.slider("Maximum tree depth", 2, 15, 3, 1)
    
    # Train model button
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            # Split data with stratification to maintain class balance
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Create preprocessing pipeline for numerical and categorical features
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
            
            # Create label encoder for target
            le = LabelEncoder()
            le.fit(['No_Machine_Failure', 'Machine_Failure'])
            y_train_encoded = le.transform(y_train)
            y_test_encoded = le.transform(y_test)
            
            # Select model with configured hyperparameters
            if model_choice == "Logistic Regression":
                model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=42)
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
            else:  # Gradient Boosting
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=42
                )
            
            # Create and train pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train_encoded)
            
            # Make predictions
            y_pred_encoded = pipeline.predict(X_test)
            y_pred = le.inverse_transform(y_pred_encoded)
            y_pred_proba = pipeline.predict_proba(X_test)
            
            # Show results in multiple columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
            
            with col2:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(
                    cm, 
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['No Failure', 'Failure'],
                    y=['No Failure', 'Failure'],
                    title="Confusion Matrix",
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # ROC Curve - Fixed implementation
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fig = px.line(
                x=fpr, y=tpr, 
                title=f'ROC Curve (AUC = {roc_auc:.3f})',
                labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                width=700, height=500
            )
            
            # Add the diagonal reference line
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            
            # Style the ROC curve line
            fig.update_traces(line=dict(color='blue', width=2))
            
            # Display the ROC curve
            st.plotly_chart(fig, use_container_width=True)
            
            # Extract feature importance
            if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                # For tree-based models (Random Forest, Gradient Boosting)
                importances = pipeline.named_steps['classifier'].feature_importances_
            elif hasattr(pipeline.named_steps['classifier'], 'coef_'):
                # For logistic regression, normalize the coefficients
                coef = pipeline.named_steps['classifier'].coef_[0]
                importances = np.abs(coef) / np.sum(np.abs(coef))  # Normalize to sum to 1
            
            # Get feature names after preprocessing
            cat_cols = pipeline.named_steps['preprocessor'].transformers_[1][2]
            cat_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['categorical'].named_steps['encoder'].get_feature_names_out(cat_cols)
            feature_names = list(numerical_features) + list(cat_feature_names)
            
            # Ensure lengths match
            if len(feature_names) == len(importances):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                st.subheader("Feature Importance")
                fig = px.bar(
                    importance_df.head(10), 
                    x='importance', 
                    y='feature',
                    title='Top 10 Most Important Features',
                    orientation='h',
                    color='importance',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Print all feature importances in expandable section
                with st.expander("View all feature importances"):
                    st.dataframe(importance_df)
            else:
                st.warning("Feature importance visualization unavailable due to dimensionality mismatch")
            
            # Model performance metrics
            st.subheader("Model Performance Summary")
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric("Accuracy", f"{report['accuracy']:.3f}")
            with metrics_cols[1]:
                st.metric("Precision (Failure)", f"{report['Machine_Failure']['precision']:.3f}")
            with metrics_cols[2]:
                st.metric("Recall (Failure)", f"{report['Machine_Failure']['recall']:.3f}")
            with metrics_cols[3]:
                st.metric("F1 Score (Failure)", f"{report['Machine_Failure']['f1-score']:.3f}")

# Add footer
st.markdown("""
---
Created with Streamlit for Machine Downtime Analysis
""")
