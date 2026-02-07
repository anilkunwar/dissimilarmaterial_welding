import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import xarray as xr
from pycalphad import Database, equilibrium, variables as v
import os
import tempfile
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Thermodynamic Enthalpy Analyzer",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .tab-container {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .download-button {
        background-color: #4CAF50;
        color: white;
        padding: 8px 16px;
        border-radius: 5px;
        border: none;
        margin: 5px;
    }
    .phase-list {
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Define molar weights for common elements (can be extended)
MOLAR_WEIGHTS = {
    'AG': 107.8682, 'AL': 26.9815386, 'AU': 196.966569, 'BI': 208.98040,
    'CU': 63.546, 'IN': 114.818, 'NI': 58.6934, 'PB': 207.2,
    'SN': 118.71, 'TI': 47.867, 'V': 50.9415, 'FE': 55.845,
    'CR': 51.9961, 'MO': 95.95, 'W': 183.84, 'MN': 54.938044,
    'SI': 28.0855, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'H': 1.00794, 'HE': 4.002602, 'LI': 6.941, 'BE': 9.012182,
    'B': 10.811, 'MG': 24.305, 'P': 30.973762, 'S': 32.065,
    'CL': 35.453, 'K': 39.0983, 'CA': 40.078, 'SC': 44.955912,
    'CO': 58.933194, 'ZN': 65.38, 'GA': 69.723, 'GE': 72.64,
    'AS': 74.92160, 'SE': 78.96, 'BR': 79.904, 'KR': 83.798,
    'RB': 85.4678, 'SR': 87.62, 'Y': 88.90585, 'ZR': 91.224,
    'NB': 92.90638, 'TC': 98.0, 'RU': 101.07, 'RH': 102.90550,
    'PD': 106.42, 'CD': 112.411, 'SB': 121.76, 'TE': 127.60,
    'XE': 131.293, 'CS': 132.90545, 'BA': 137.327, 'LA': 138.90547,
    'CE': 140.116, 'PR': 140.90765, 'ND': 144.242, 'PM': 145.0,
    'SM': 150.36, 'EU': 151.964, 'GD': 157.25, 'TB': 158.92535,
    'DY': 162.500, 'HO': 164.93032, 'ER': 167.259, 'TM': 168.93421,
    'YB': 173.04, 'LU': 174.9668, 'HF': 178.49, 'TA': 180.94788,
    'RE': 186.207, 'OS': 190.23, 'IR': 192.217, 'PT': 195.084,
    'TL': 204.3833, 'PB': 207.2, 'BI': 208.98040, 'PO': 209.0,
    'AT': 210.0, 'RN': 222.0, 'FR': 223.0, 'RA': 226.0,
    'AC': 227.0, 'TH': 232.03806, 'PA': 231.03588, 'U': 238.02891,
    'NP': 237.0, 'PU': 244.0, 'AM': 243.0, 'CM': 247.0,
    'BK': 247.0, 'CF': 251.0, 'ES': 252.0, 'FM': 257.0,
    'MD': 258.0, 'NO': 259.0, 'LR': 262.0, 'RF': 267.0,
    'DB': 270.0, 'SG': 271.0, 'BH': 270.0, 'HS': 277.0,
    'MT': 276.0, 'DS': 281.0, 'RG': 280.0, 'CN': 285.0,
    'NH': 284.0, 'FL': 289.0, 'MC': 288.0, 'LV': 293.0,
    'TS': 294.0, 'OG': 294.0
}

class EnthalpyAnalyzer:
    def __init__(self):
        self.results_history = []
        self.fitting_results = []
        
    def calculate_alloy_molar_weight(self, composition):
        """Calculate molar weight of alloy from composition dictionary"""
        molar_weight = 0.0
        for element, fraction in composition.items():
            element_upper = element.upper()
            if element_upper in MOLAR_WEIGHTS:
                molar_weight += fraction * MOLAR_WEIGHTS[element_upper]
            else:
                st.warning(f"Molar weight for {element} not found. Using default value of 50 g/mol.")
                molar_weight += fraction * 50.0
        return molar_weight
    
    def convert_to_specific_enthalpy(self, df, composition):
        """Convert molar enthalpy to specific enthalpy"""
        molar_weight = self.calculate_alloy_molar_weight(composition)
        df['H_specific'] = df['H'] / molar_weight * 1000  # J/mol to J/kg
        return df
    
    def sigmoid(self, x, k):
        """Sigmoid function for enthalpy fitting"""
        return 1 / (1 + np.exp(-k * x))
    
    def enthalpy_equation(self, T, A1, A2, Tm, DeltaHf, k, H298):
        """Enthalpy equation for curve fitting"""
        return A1 * T + A2 * (T - Tm) + DeltaHf * self.sigmoid(T - Tm, k) + H298

def main():
    st.title("🔥 Thermodynamic Enthalpy Analyzer")
    st.markdown("---")
    
    analyzer = EnthalpyAnalyzer()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Enthalpy Computation",
        "📈 Curve Fitting",
        "🔍 Multi-Material Comparison",
        "⚙️ Settings & Help"
    ])
    
    # Tab 1: Enthalpy Computation
    with tab1:
        st.header("Enthalpy Computation from TDB Files")
        
        # Create two columns for file selection
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Select TDB File")
            
            # Option 1: Browse existing TDB files
            database_dir = "databases"
            os.makedirs(database_dir, exist_ok=True)
            
            # Get list of existing TDB files
            existing_tdb_files = [f for f in os.listdir(database_dir) if f.lower().endswith('.tdb')]
            
            tdb_option = st.radio(
                "Choose TDB source:",
                ["Select from database", "Upload new TDB file"]
            )
            
            tdb_path = None
            
            if tdb_option == "Select from database" and existing_tdb_files:
                selected_file = st.selectbox(
                    "Available TDB files:",
                    existing_tdb_files,
                    help="Select a TDB file from the databases directory"
                )
                tdb_path = os.path.join(database_dir, selected_file)
                st.success(f"Selected: {selected_file}")
            elif tdb_option == "Upload new TDB file":
                uploaded_file = st.file_uploader(
                    "Upload your TDB file",
                    type=["tdb", "TDB"],
                    help="Upload a thermodynamic database file"
                )
                if uploaded_file is not None:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.tdb') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tdb_path = tmp_file.name
                    st.success(f"Uploaded: {uploaded_file.name}")
                    
                    # Option to save to database directory
                    if st.checkbox("Save to database directory for future use"):
                        save_path = os.path.join(database_dir, uploaded_file.name)
                        with open(save_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        st.info(f"Saved to: {save_path}")
        
        with col2:
            if tdb_path:
                try:
                    dbf = Database(tdb_path)
                    st.subheader("Database Information")
                    
                    # Display database info
                    st.write(f"**Elements:** {', '.join(sorted(dbf.elements))}")
                    st.write(f"**Phases:** {len(dbf.phases)} phases available")
                    
                    # Show phases in a scrollable box
                    with st.expander("View All Phases"):
                        phases_list = list(dbf.phases.keys())
                        cols = st.columns(3)
                        for i, phase in enumerate(sorted(phases_list)):
                            cols[i % 3].write(f"• {phase}")
                    
                    # Get user inputs
                    st.subheader("Composition Settings")
                    
                    # Element selection
                    available_elements = [e for e in dbf.elements if e != 'VA']
                    selected_elements = st.multiselect(
                        "Select elements (exclude VA):",
                        sorted(available_elements),
                        default=sorted(available_elements)[:min(3, len(available_elements))]
                    )
                    
                    if selected_elements:
                        # Composition input
                        st.write("**Enter composition (mole fractions):**")
                        composition = {}
                        remaining_fraction = 1.0
                        
                        cols = st.columns(len(selected_elements))
                        for idx, element in enumerate(selected_elements):
                            with cols[idx]:
                                max_val = min(1.0, remaining_fraction)
                                default_val = 1.0/len(selected_elements) if len(selected_elements) > 0 else 0.0
                                fraction = st.number_input(
                                    f"{element}",
                                    min_value=0.0,
                                    max_value=max_val,
                                    value=default_val,
                                    step=0.01,
                                    key=f"comp_{element}"
                                )
                                composition[element] = fraction
                                if idx < len(selected_elements) - 1:
                                    remaining_fraction -= fraction
                        
                        # Temperature range
                        st.subheader("Temperature Settings")
                        col_temp1, col_temp2, col_temp3 = st.columns(3)
                        with col_temp1:
                            T_start = st.number_input("Start Temp (K)", 300, 5000, 300, 10)
                        with col_temp2:
                            T_end = st.number_input("End Temp (K)", T_start+10, 5000, 1500, 10)
                        with col_temp3:
                            T_step = st.number_input("Step Size (K)", 1, 100, 10)
                        
                        # Phase selection
                        st.subheader("Phase Selection")
                        available_phases = list(dbf.phases.keys())
                        selected_phases = st.multiselect(
                            "Select phases for equilibrium:",
                            sorted(available_phases),
                            default=sorted(available_phases)[:min(2, len(available_phases))]
                        )
                        
                        # Pressure
                        P = st.number_input("Pressure (Pa)", 101325, 1000000, 101325)
                        
                        if st.button("Compute Enthalpy", type="primary"):
                            with st.spinner("Computing equilibrium..."):
                                try:
                                    # Prepare conditions
                                    conditions = {
                                        v.T: (T_start, T_end, T_step),
                                        v.P: P
                                    }
                                    
                                    # Add composition conditions
                                    for element, fraction in composition.items():
                                        conditions[v.X(element)] = fraction
                                    
                                    # Add VA to elements
                                    elements_with_va = selected_elements + ['VA']
                                    
                                    # Perform equilibrium calculation
                                    eq_result = equilibrium(
                                        dbf, 
                                        elements_with_va, 
                                        selected_phases, 
                                        conditions, 
                                        output='HM'
                                    )
                                    
                                    # Extract and process results
                                    T_values = eq_result.T.values.flatten()
                                    HM_values = eq_result.HM.values.flatten()
                                    
                                    # Create DataFrame
                                    result_df = pd.DataFrame({
                                        'Temperature_K': T_values,
                                        'Enthalpy_J_mol': HM_values
                                    })
                                    
                                    # Add specific enthalpy
                                    result_df = analyzer.convert_to_specific_enthalpy(result_df, composition)
                                    
                                    # Store results
                                    result_info = {
                                        'name': f"{', '.join(selected_elements)}",
                                        'composition': composition,
                                        'data': result_df,
                                        'phases': selected_phases,
                                        'tdb_file': os.path.basename(tdb_path)
                                    }
                                    analyzer.results_history.append(result_info)
                                    
                                    # Display results
                                    st.success("Calculation completed!")
                                    
                                    # Show data preview
                                    with st.expander("View Data Preview"):
                                        st.dataframe(result_df.head(10))
                                    
                                    # Plot results
                                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                                    
                                    # Molar enthalpy plot
                                    ax1.plot(result_df['Temperature_K'], result_df['Enthalpy_J_mol'], 
                                            'b-', linewidth=2, label='Molar Enthalpy')
                                    ax1.set_xlabel('Temperature (K)')
                                    ax1.set_ylabel('Enthalpy (J/mol)')
                                    ax1.set_title('Molar Enthalpy vs Temperature')
                                    ax1.grid(True, alpha=0.3)
                                    ax1.legend()
                                    
                                    # Specific enthalpy plot
                                    ax2.plot(result_df['Temperature_K'], result_df['H_specific'], 
                                            'r-', linewidth=2, label='Specific Enthalpy')
                                    ax2.set_xlabel('Temperature (K)')
                                    ax2.set_ylabel('Enthalpy (J/kg)')
                                    ax2.set_title('Specific Enthalpy vs Temperature')
                                    ax2.grid(True, alpha=0.3)
                                    ax2.legend()
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Download options
                                    st.subheader("📥 Download Results")
                                    
                                    col_dl1, col_dl2 = st.columns(2)
                                    
                                    with col_dl1:
                                        # CSV format
                                        csv_data = result_df.to_csv(index=False)
                                        st.download_button(
                                            label="Download CSV (All Data)",
                                            data=csv_data,
                                            file_name=f"enthalpy_data_{'_'.join(selected_elements)}.csv",
                                            mime="text/csv"
                                        )
                                        
                                        # Molar enthalpy only
                                        molar_df = result_df[['Temperature_K', 'Enthalpy_J_mol']]
                                        molar_csv = molar_df.to_csv(index=False)
                                        st.download_button(
                                            label="Download CSV (Molar Only)",
                                            data=molar_csv,
                                            file_name=f"molar_enthalpy_{'_'.join(selected_elements)}.csv",
                                            mime="text/csv"
                                        )
                                    
                                    with col_dl2:
                                        # DAT format
                                        dat_content = "# Temperature(K) Enthalpy(J/mol) Enthalpy(J/kg)\n"
                                        for _, row in result_df.iterrows():
                                            dat_content += f"{row['Temperature_K']:.2f} {row['Enthalpy_J_mol']:.4f} {row['H_specific']:.4f}\n"
                                        
                                        st.download_button(
                                            label="Download DAT Format",
                                            data=dat_content,
                                            file_name=f"enthalpy_data_{'_'.join(selected_elements)}.dat",
                                            mime="text/plain"
                                        )
                                        
                                        # Specific enthalpy only
                                        specific_df = result_df[['Temperature_K', 'H_specific']]
                                        specific_csv = specific_df.to_csv(index=False)
                                        st.download_button(
                                            label="Download CSV (Specific Only)",
                                            data=specific_csv,
                                            file_name=f"specific_enthalpy_{'_'.join(selected_elements)}.csv",
                                            mime="text/csv"
                                        )
                                    
                                except Exception as e:
                                    st.error(f"Error in calculation: {str(e)}")
                    
                except Exception as e:
                    st.error(f"Error loading database: {str(e)}")
    
    # Tab 2: Curve Fitting
    with tab2:
        st.header("Curve Fitting and Analysis")
        
        # File upload or use computed data
        data_source = st.radio(
            "Choose data source:",
            ["Upload CSV file", "Use computed data from Tab 1"]
        )
        
        if data_source == "Upload CSV file":
            uploaded_file = st.file_uploader(
                "Upload enthalpy data (CSV with Temperature and Enthalpy columns)",
                type=['csv']
            )
            if uploaded_file:
                data = pd.read_csv(uploaded_file)
                st.write("Data preview:")
                st.dataframe(data.head())
                
                # Allow user to specify column names
                col1, col2 = st.columns(2)
                with col1:
                    temp_col = st.selectbox("Select temperature column", data.columns)
                with col2:
                    enthalpy_col = st.selectbox("Select enthalpy column", data.columns)
                
                T_data = data[temp_col].values
                H_data = data[enthalpy_col].values
        
        else:  # Use computed data
            if analyzer.results_history:
                # Let user select which computed result to use
                result_names = [f"{i+1}: {res['name']} ({res['tdb_file']})" 
                              for i, res in enumerate(analyzer.results_history)]
                selected_result = st.selectbox("Select computed result:", result_names)
                
                if selected_result:
                    idx = int(selected_result.split(':')[0]) - 1
                    result_data = analyzer.results_history[idx]['data']
                    T_data = result_data['Temperature_K'].values
                    H_data = result_data['Enthalpy_J_mol'].values
            else:
                st.info("No computed data available. Please compute some data in Tab 1 first.")
                st.stop()
        
        if 'T_data' in locals() and 'H_data' in locals():
            st.subheader("Fitting Parameters")
            
            # Parameter inputs
            col1, col2 = st.columns(2)
            
            with col1:
                A1_guess = st.number_input("A1 initial guess (J/(mol·K))", 0.0, 100.0, 30.0, 0.1)
                A2_guess = st.number_input("A2 initial guess (J/(mol·K))", -100.0, 100.0, 10.0, 0.1)
                Tm_guess = st.number_input("Tm initial guess (K)", 0.0, 5000.0, 500.0, 10.0)
            
            with col2:
                DeltaHf_guess = st.number_input("ΔHf initial guess (J/mol)", -100000.0, 100000.0, 10000.0, 100.0)
                k_guess = st.number_input("k initial guess", 0.0, 1.0, 0.01, 0.001)
                H298_guess = st.number_input("H298 initial guess (J/mol)", -100000.0, 100000.0, 0.0, 100.0)
            
            # Composition for record keeping
            st.subheader("Alloy Composition (for reference)")
            composition_cols = st.columns(4)
            elements = ['Ag', 'Al', 'Au', 'Bi', 'Cu', 'In', 'Ni', 'Pb', 'Sn', 'Ti', 'V']
            mole_fractions = {}
            
            for i, element in enumerate(elements):
                with composition_cols[i % 4]:
                    mole_fractions[element] = st.number_input(
                        f"x{element}", 0.0, 1.0, 0.0, 0.01, key=f"fit_{element}"
                    )
            
            if st.button("Perform Curve Fit", type="primary"):
                with st.spinner("Fitting curve..."):
                    try:
                        # Perform curve fitting
                        initial_guess = [A1_guess, A2_guess, Tm_guess, DeltaHf_guess, k_guess, H298_guess]
                        fit_params, pcov = curve_fit(
                            analyzer.enthalpy_equation, 
                            T_data, 
                            H_data, 
                            p0=initial_guess,
                            maxfev=5000
                        )
                        
                        A1_fit, A2_fit, Tm_fit, DeltaHf_fit, k_fit, H298_fit = fit_params
                        
                        # Calculate fitted curve
                        T_fit = np.linspace(T_data.min(), T_data.max(), 500)
                        H_fit = analyzer.enthalpy_equation(T_fit, *fit_params)
                        
                        # Calculate R²
                        H_pred = analyzer.enthalpy_equation(T_data, *fit_params)
                        residuals = H_data - H_pred
                        ss_res = np.sum(residuals**2)
                        ss_tot = np.sum((H_data - np.mean(H_data))**2)
                        r_squared = 1 - (ss_res / ss_tot)
                        
                        # Display results
                        st.success(f"Fitting completed! R² = {r_squared:.6f}")
                        
                        # Plot results
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                        
                        # Main fit plot
                        ax1.scatter(T_data, H_data, alpha=0.5, label='Original Data', color='blue')
                        ax1.plot(T_fit, H_fit, 'r-', linewidth=2, label='Fitted Curve')
                        ax1.set_xlabel('Temperature (K)')
                        ax1.set_ylabel('Enthalpy (J/mol)')
                        ax1.set_title('Enthalpy-Temperature Curve Fitting')
                        ax1.grid(True, alpha=0.3)
                        ax1.legend()
                        
                        # Residual plot
                        ax2.scatter(T_data, residuals, alpha=0.6, color='green')
                        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                        ax2.set_xlabel('Temperature (K)')
                        ax2.set_ylabel('Residuals (J/mol)')
                        ax2.set_title('Residual Plot')
                        ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display coefficients
                        st.subheader("Fitted Parameters")
                        col_coeff1, col_coeff2 = st.columns(2)
                        
                        with col_coeff1:
                            st.metric("A1", f"{A1_fit:.4f} J/(mol·K)")
                            st.metric("A2", f"{A2_fit:.4f} J/(mol·K)")
                            st.metric("Tm", f"{Tm_fit:.2f} K")
                        
                        with col_coeff2:
                            st.metric("ΔHf", f"{DeltaHf_fit:.2f} J/mol")
                            st.metric("k", f"{k_fit:.6f}")
                            st.metric("H298", f"{H298_fit:.2f} J/mol")
                        
                        # Store fitting results
                        fit_result = {
                            'coefficients': {
                                'A1': A1_fit,
                                'A2': A2_fit,
                                'Tm': Tm_fit,
                                'DeltaHf': DeltaHf_fit,
                                'k': k_fit,
                                'H298': H298_fit
                            },
                            'r_squared': r_squared,
                            'composition': mole_fractions,
                            'temperature_range': [T_data.min(), T_data.max()]
                        }
                        analyzer.fitting_results.append(fit_result)
                        
                        # Download options
                        st.subheader("📥 Download Fitting Results")
                        
                        # Create DataFrame for coefficients
                        coeff_df = pd.DataFrame([{
                            'A1_J_per_mol_K': A1_fit,
                            'A2_J_per_mol_K': A2_fit,
                            'Tm_K': Tm_fit,
                            'DeltaHf_J_per_mol': DeltaHf_fit,
                            'k': k_fit,
                            'H298_J_per_mol': H298_fit,
                            'R_squared': r_squared,
                            **{f'x{element}': mole_fractions[element] for element in elements}
                        }])
                        
                        # Create DataFrame for fitted curve
                        fitted_curve_df = pd.DataFrame({
                            'Temperature_K': T_fit,
                            'Enthalpy_J_per_mol': H_fit
                        })
                        
                        col_fit1, col_fit2 = st.columns(2)
                        
                        with col_fit1:
                            st.download_button(
                                label="Download Coefficients (CSV)",
                                data=coeff_df.to_csv(index=False),
                                file_name="fitting_coefficients.csv",
                                mime="text/csv"
                            )
                            
                        with col_fit2:
                            st.download_button(
                                label="Download Fitted Curve (CSV)",
                                data=fitted_curve_df.to_csv(index=False),
                                file_name="fitted_enthalpy_curve.csv",
                                mime="text/csv"
                            )
                        
                        # Save as JSON
                        json_data = json.dumps(fit_result, indent=4)
                        st.download_button(
                            label="Download Full Results (JSON)",
                            data=json_data,
                            file_name="fitting_results.json",
                            mime="application/json"
                        )
                        
                    except Exception as e:
                        st.error(f"Error in curve fitting: {str(e)}")
    
    # Tab 3: Multi-Material Comparison
    with tab3:
        st.header("Multi-Material Comparison")
        
        if not analyzer.results_history:
            st.info("No data available for comparison. Please compute some data in Tab 1 first.")
        else:
            # Select results to compare
            st.subheader("Select Materials for Comparison")
            
            result_options = []
            for i, result in enumerate(analyzer.results_history):
                name = f"{result['name']} (TDB: {result['tdb_file']})"
                result_options.append((i, name))
            
            selected_indices = st.multiselect(
                "Select up to 5 materials:",
                options=[opt[1] for opt in result_options],
                default=[opt[1] for opt in result_options[:min(3, len(result_options))]]
            )
            
            if selected_indices:
                # Get indices of selected results
                selected_ids = []
                for selected in selected_indices:
                    for idx, name in result_options:
                        if name == selected:
                            selected_ids.append(idx)
                            break
                
                # Plot comparison
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                colors = plt.cm.tab10(np.linspace(0, 1, len(selected_ids)))
                
                for i, idx in enumerate(selected_ids):
                    result = analyzer.results_history[idx]
                    data = result['data']
                    
                    # Molar enthalpy comparison
                    ax1.plot(data['Temperature_K'], data['Enthalpy_J_mol'],
                            color=colors[i], linewidth=2, 
                            label=f"{result['name']}")
                    
                    # Specific enthalpy comparison
                    ax2.plot(data['Temperature_K'], data['H_specific'],
                            color=colors[i], linewidth=2, 
                            label=f"{result['name']}")
                
                ax1.set_xlabel('Temperature (K)')
                ax1.set_ylabel('Molar Enthalpy (J/mol)')
                ax1.set_title('Molar Enthalpy Comparison')
                ax1.grid(True, alpha=0.3)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                ax2.set_xlabel('Temperature (K)')
                ax2.set_ylabel('Specific Enthalpy (J/kg)')
                ax2.set_title('Specific Enthalpy Comparison')
                ax2.grid(True, alpha=0.3)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display comparison table
                st.subheader("Comparison Summary")
                
                summary_data = []
                for idx in selected_ids:
                    result = analyzer.results_history[idx]
                    data = result['data']
                    
                    summary_data.append({
                        'Material': result['name'],
                        'TDB File': result['tdb_file'],
                        'Phases': ', '.join(result['phases']),
                        'Min Temp (K)': data['Temperature_K'].min(),
                        'Max Temp (K)': data['Temperature_K'].max(),
                        'Min H (J/mol)': data['Enthalpy_J_mol'].min(),
                        'Max H (J/mol)': data['Enthalpy_J_mol'].max(),
                        'ΔH (J/mol)': data['Enthalpy_J_mol'].max() - data['Enthalpy_J_mol'].min()
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df)
                
                # Download comparison data
                st.subheader("📥 Download Comparison Data")
                
                # Combine all selected data
                combined_data = {}
                for i, idx in enumerate(selected_ids):
                    result = analyzer.results_history[idx]
                    data = result['data']
                    material_name = result['name'].replace(', ', '_')
                    
                    combined_data[f'Temperature_K'] = data['Temperature_K'].values
                    combined_data[f'{material_name}_H_molar'] = data['Enthalpy_J_mol'].values
                    combined_data[f'{material_name}_H_specific'] = data['H_specific'].values
                
                combined_df = pd.DataFrame(combined_data)
                
                col_comp1, col_comp2 = st.columns(2)
                
                with col_comp1:
                    st.download_button(
                        label="Download Combined Data (CSV)",
                        data=combined_df.to_csv(index=False),
                        file_name="multi_material_comparison.csv",
                        mime="text/csv"
                    )
                
                with col_comp2:
                    st.download_button(
                        label="Download Summary Table (CSV)",
                        data=summary_df.to_csv(index=False),
                        file_name="comparison_summary.csv",
                        mime="text/csv"
                    )
    
    # Tab 4: Settings & Help
    with tab4:
        st.header("Settings & User Guide")
        
        col_set1, col_set2 = st.columns(2)
        
        with col_set1:
            st.subheader("📖 User Guide")
            st.markdown("""
            ### How to Use This Application:
            
            1. **Tab 1: Enthalpy Computation**
               - Select or upload a TDB file
               - Choose elements and set compositions
               - Define temperature range and phases
               - Compute and download results
            
            2. **Tab 2: Curve Fitting**
               - Upload data or use computed results
               - Set initial parameters for fitting
               - Fit the enthalpy equation
               - Download fitting results
            
            3. **Tab 3: Multi-Material Comparison**
               - Compare multiple computed datasets
               - Visualize differences
               - Export comparison tables
            
            ### Tips:
            - Use consistent units throughout
            - Start with wide parameter ranges for fitting
            - Save interesting results for later comparison
            """)
        
        with col_set2:
            st.subheader("⚙️ Application Settings")
            
            # Display session info
            st.write(f"**Results in memory:** {len(analyzer.results_history)}")
            st.write(f"**Fitting results:** {len(analyzer.fitting_results)}")
            
            # Clear data button
            if st.button("Clear All Session Data"):
                analyzer.results_history = []
                analyzer.fitting_results = []
                st.success("Session data cleared!")
                st.rerun()
            
            # Database management
            st.subheader("Database Management")
            database_dir = "databases"
            if os.path.exists(database_dir):
                tdb_files = [f for f in os.listdir(database_dir) if f.endswith('.tdb')]
                if tdb_files:
                    st.write("**Available TDB files:**")
                    for tdb in tdb_files:
                        col_file1, col_file2, col_file3 = st.columns([3, 1, 1])
                        with col_file1:
                            st.write(f"• {tdb}")
                        with col_file2:
                            if st.button("View", key=f"view_{tdb}"):
                                st.info(f"File: {tdb}")
                        with col_file3:
                            if st.button("Delete", key=f"del_{tdb}"):
                                os.remove(os.path.join(database_dir, tdb))
                                st.rerun()
            
            # Add element molar weight
            st.subheader("Add Custom Element Molar Weight")
            col_elem1, col_elem2, col_elem3 = st.columns(3)
            with col_elem1:
                new_element = st.text_input("Element symbol", key="new_elem")
            with col_elem2:
                new_weight = st.number_input("Molar weight (g/mol)", 0.0, 500.0, 0.0, 0.1)
            with col_elem3:
                if st.button("Add Element"):
                    if new_element:
                        MOLAR_WEIGHTS[new_element.upper()] = new_weight
                        st.success(f"Added {new_element.upper()}: {new_weight} g/mol")
        
        # Equation reference
        st.subheader("📚 Equation Reference")
        st.latex(r"H(T) = A_1 T + A_2 (T - T_m) + \Delta H_f \cdot \frac{1}{1 + e^{-k(T - T_m)}} + H_{298}")
        st.markdown("""
        Where:
        - $H(T)$: Enthalpy at temperature T (J/mol)
        - $A_1$, $A_2$: Linear coefficients (J/(mol·K))
        - $T_m$: Melting temperature (K)
        - $\Delta H_f$: Heat of fusion (J/mol)
        - $k$: Sigmoid steepness parameter
        - $H_{298}$: Enthalpy at 298 K (J/mol)
        """)

if __name__ == "__main__":
    main()
