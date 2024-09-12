import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.title("E-commerce Sales Prediction System")

uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        st.write("⚠️ Unable to decode the file with UTF-8 encoding. Trying with 'ISO-8859-1'...")
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    st.write("### Dataset Preview")
    st.write(df.head())

    st.write("### Column Names in the Dataset")
    st.write(df.columns.tolist())  

    target_column = st.selectbox("Select the target column to predict:", df.columns)

    # Step 2: Data Preprocessing and Cleaning
    st.write("## Data Preprocessing and Cleaning")

    st.write("### Missing Values")
    st.write(df.isnull().sum())
    df = df.dropna()  

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    st.write("### Data After Cleaning and Encoding")
    st.write(df.head())


    st.write("### Feature Selection")
    X = df.drop(target_column, axis=1) 
    y = df[target_column]

  
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Step 4: Model Training
    st.write("## Model Training")

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 5: Model Evaluation
    st.write("### Model Evaluation")

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared Score: {r2:.2f}")

    # Step 6: Visualization
    st.write("## Data Visualization")
    
    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("### Feature Importance")
    importance = model.coef_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    st.write(feature_importance)

    # Step 7: Prediction on New Data
    st.write("## Make Predictions")
    user_input = [st.number_input(f"Enter {col}", value=0) for col in X.columns]
    
    if st.button("Predict"):
        user_input = np.array(user_input).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input)
        prediction = model.predict(user_input_scaled)
        st.write(f"Predicted Sales: {prediction[0]:.2f}")
