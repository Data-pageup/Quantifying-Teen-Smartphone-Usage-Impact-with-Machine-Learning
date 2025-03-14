import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

# Title
st.title("Regression Analysis App")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.write(data.head())
    
    if data.shape[1] >= 2:
        st.header("Data Analysis")
        
        # Step 2: Select Input (X) and Target (Y)
        x_column = st.selectbox("Select X (Input) Column:", data.columns)
        y_column = st.selectbox("Select Y (Target) Column:", data.columns)
        
        # Step 3: Select Regression Type
        regression_type = st.radio("Select Regression Type:", ["Linear Regression", "Logistic Regression"])
        
        if regression_type == "Linear Regression":
            st.subheader("Linear Regression Analysis")
            
            # Split data
            X = data[[x_column]]
            y = data[y_column]
            model = LinearRegression()
            model.fit(X, y)
            
            # Prediction
            st.write("Model trained successfully!")
            y_pred = model.predict(X)
            
            # Display RMSE
            rmse = mean_squared_error(y, y_pred, squared=False)
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
            
            # Visualization
            plt.figure(figsize=(10, 6))
            plt.scatter(X, y, label="Actual Data", color="blue")
            plt.plot(X, y_pred, label="Prediction", color="red")
            plt.legend()
            st.pyplot(plt)
            
            # User Prediction
            user_input = st.number_input("Enter a value for X to predict Y:")
            if user_input:
                prediction = model.predict([[user_input]])[0]
                st.write(f"Predicted Y value: {prediction:.3f}")
        
        elif regression_type == "Logistic Regression":
            st.subheader("Logistic Regression Analysis")
            
            # Split data
            X = data[[x_column]]
            y = data[y_column]
            
            # Ensure binary target
            if len(y.unique()) != 2:
                st.error("Logistic Regression requires a binary target variable (0/1). Please select an appropriate column.")
            else:
                # Train Test Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = LogisticRegression()
                model.fit(X_train, y_train)
                
                # Prediction and Accuracy
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
                
                # User Prediction
                user_input = st.number_input("Enter a value for X to predict probability:")
                if user_input:
                    probability = model.predict_proba([[user_input]])[0][1]
                    predicted_class = model.predict([[user_input]])[0]
                    st.write(f"Predicted Probability (Class 1): {probability:.3f}")
                    st.write(f"Predicted Class: {predicted_class}")

