import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the sigmoid function
def sigmoid(x, k):
    return 1 / (1 + np.exp(-k * x))

# Define the enthalpy of the alloy as a function of temperature
def enthalpy(T, A1, A2, Tm, DeltaHf, k, H298):
    H = A1 * T + A2 * (T - Tm) + DeltaHf * sigmoid(T - Tm, k) + H298
    return H

def main():
    st.title("Enthalpy-Temperature Relationship")

    # File upload
    st.header("Upload CSV file with 'S.N.', 'T' and 'H' columns")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    # Sidebar for initial guess coefficients and mole fractions
    st.sidebar.header("Initial Guess for Coefficients")
    A1_guess = st.sidebar.number_input("A1 (J/(mol K))", value=1.0)
    A2_guess = st.sidebar.number_input("A2 (J/(mol K))", value=1.0)
    Tm_guess = st.sidebar.number_input("Tm (K)", value=300.0)
    DeltaHf_guess = st.sidebar.number_input("DeltaHf (J/mol)", value=100.0)
    k_guess = st.sidebar.number_input("k", value=0.01)
    H298_guess = st.sidebar.number_input("H298 (J/mol)", value=0.0)

    # Sidebar for mole fractions
    st.sidebar.header("Mole Fractions of Alloy Components (The mole fraction must represent the same alloy composition of the H-T dataset)")
    mole_fractions = {}
    elements = ['Ag', 'Al', 'Au','Bi', 'Cu', 'In', 'Ni', 'Pb',  'Sn', 'Ti', 'V']
    for element in elements:
        mole_fractions[element] = st.sidebar.number_input(f"{element} mole fraction", value=0.0, min_value=0.0, max_value=1.0)

    if uploaded_file is not None:
        # Load data from the uploaded CSV file
        data = pd.read_csv(uploaded_file)
        T_data = data['T']
        H_data = data['H']

        # Perform curve fitting to deduce the coefficients
        initial_guess = [A1_guess, A2_guess, Tm_guess, DeltaHf_guess, k_guess, H298_guess]
        fit_params, _ = curve_fit(enthalpy, T_data, H_data, p0=initial_guess)

        # Unpack the fitted coefficients
        A1_fit, A2_fit, Tm_fit, DeltaHf_fit, k_fit, H298_fit = fit_params

        # Generate fitted H values
        H_fit = enthalpy(T_data, A1_fit, A2_fit, Tm_fit, DeltaHf_fit, k_fit, H298_fit)

        # Plot the fitted curve and the original scatter data
        plt.figure(figsize=(8, 6))
        plt.scatter(T_data, H_data, label='Original Data', color='blue', marker='o')
        plt.plot(T_data, H_fit, label='Fitted Curve', color='red', linewidth=2)
        plt.xlabel('Temperature (K)')
        plt.ylabel('Enthalpy (J/mol)')
        plt.legend()
        plt.grid(True)
        plt.title('Enthalpy-Temperature Relationship')
        st.pyplot(plt)

        # Print the deduced coefficients with appropriate units or as unitless
        st.header("Deduced coefficients:")
        st.write(f"A1 = {A1_fit} (J/mol*K)")
        st.write(f"A2 = {A2_fit} (J/mol)")
        st.write(f"Tm = {Tm_fit} (K)")
        st.write(f"DeltaHf = {DeltaHf_fit} (J/mol)")
        st.write(f"k = {k_fit}")
        st.write(f"H298 = {H298_fit} (J/mol)")

        # Combine mole fractions with coefficients into a single row DataFrame
        mole_fractions_list = [mole_fractions[element] for element in elements]
        combined_data = {
            'xAg': mole_fractions_list[0],
            'xAl': mole_fractions_list[1],
            'xAu': mole_fractions_list[2],
            'xBi': mole_fractions_list[3],
            'xCu': mole_fractions_list[4],
            'xIn': mole_fractions_list[5],
            'xNi': mole_fractions_list[6],
            'xPb': mole_fractions_list[7],
            'xSn': mole_fractions_list[8],
            'xTi': mole_fractions_list[9],
            'xV': mole_fractions_list[10],
            'A1': A1_fit,
            'A2': A2_fit,
            'Tm (K)': Tm_fit,
            'DeltaHf (J/mol)': DeltaHf_fit,
            'k': k_fit,
            'H298 (J/mol)': H298_fit
        }

        # Create DataFrame and allow user to download as CSV
        df_combined_data = pd.DataFrame([combined_data])
        st.header("Coefficients with Mole Fractions:")
        st.dataframe(df_combined_data)
        st.download_button(
            label="Download Coefficients with Mole Fractions as CSV",
            data=df_combined_data.to_csv(index=False),
            file_name="coefficients_with_mole_fractions.csv",
            mime="text/csv"
        )

        # Allow user to download H-T data of the fitted curve as CSV
        df_fitted_data = pd.DataFrame({'Temperature (K)': T_data, 'Enthalpy (J/mol)': H_fit})
        st.download_button(
            label="Download Fitted H-T Data as CSV",
            data=df_fitted_data.to_csv(index=False),
            file_name="fitted_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
