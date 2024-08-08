import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('Naval_Vessel_Condition\data.csv')
features = ['LeverPosition', 'GasTurbineShaftTorque[kNm]', 'GT_rateofrevolutions(GTn)[rpm]',
             'GasGeneratorRateofRevolutions(GGn)[rpm]', 'StarboardPropellerTorque(Ts)[kN]',
             'PortPropellerTorque(Tp)[kN]', 'HightPressure(HP)TurbineExitTemperature(T48)[C]',
             'GTCompressorInletAirTemperature(T1)[C]', 'GTCompressorOutletAirTemperature(T2)[C]',
             'HPTurbineExitPressure(P48)[bar]', 'GTCompressorInletAirPressure(P1)[bar]', 
             'GTExhaustGasPressure(Pexh)[bar]']

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Feature Selection"])

    if page == "Prediction":
        st.title("Prediction Page")
        st.write("Input feature values to predict the target value.")

        # Input fields for prediction
        input_features = {feature: st.number_input(feature, value=0.0) for feature in features}
        
        if st.button("Predict"):
            if data is not None:
                try:
                    X = data[features]
                    y = data['GTTurbineDecayStateCoefficient']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Predict based on input
                    input_df = pd.DataFrame([input_features], columns=features)
                    prediction = model.predict(input_df)
                    st.write(f"Predicted GTTurbineDecayStateCoefficient: {prediction[0]}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.error("Data is not loaded properly.")

                

    elif page == "Feature Selection":
        st.title("Feature Selection Page")
        st.write("Select a feature to be the new target. Remaining features will be used to predict this new target.")

        selected_target = st.selectbox("Select New Target Feature", features)
        
        if selected_target:
            st.write("Preparing model...")
            # Show progress
            st.spinner("Training model...")
            
            # Prepare model using original data
            remaining_features = [f for f in features if f != selected_target]
            X = data[remaining_features]
            y = data[selected_target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Input fields for prediction
            st.write(f"Input values for features to predict {selected_target}:")
            input_features = {feature: st.number_input(feature, value=0.0) for feature in remaining_features}
            
            if st.button("Predict"):
                try:
                    input_df = pd.DataFrame([input_features], columns=remaining_features)
                    prediction = model.predict(input_df)
                    st.write(f"Predicted {selected_target}: {prediction[0]}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()