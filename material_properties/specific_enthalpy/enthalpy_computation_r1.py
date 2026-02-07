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
import shutil
from datetime import datetime

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🔥 Thermodynamic Enthalpy Analyzer Pro",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .download-section {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .phase-container {
        max-height: 250px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 8px;
        background-color: #fafafa;
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff8e1;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FFC107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Comprehensive molar weights database (expanded)
MOLAR_WEIGHTS = {
    # Common elements in alloys
    'AG': 107.8682, 'AL': 26.9815386, 'AU': 196.966569, 'BI': 208.98040,
    'CU': 63.546, 'IN': 114.818, 'NI': 58.6934, 'PB': 207.2,
    'SN': 118.71, 'TI': 47.867, 'V': 50.9415, 'FE': 55.845,
    'CR': 51.9961, 'MO': 95.95, 'W': 183.84, 'MN': 54.938044,
    'SI': 28.0855, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'H': 1.00794, 'MG': 24.305, 'ZN': 65.38, 'CO': 58.933194,
    'PD': 106.42, 'PT': 195.084, 'RH': 102.90550, 'IR': 192.217,
    'RU': 101.07, 'OS': 190.23, 'RE': 186.207, 'TA': 180.94788,
    'NB': 92.90638, 'ZR': 91.224, 'HF': 178.49, 'TH': 232.03806,
    'U': 238.02891, 'Y': 88.90585, 'LA': 138.90547, 'CE': 140.116,
    'PR': 140.90765, 'ND': 144.242, 'SM': 150.36, 'EU': 151.964,
    'GD': 157.25, 'TB': 158.92535, 'DY': 162.500, 'HO': 164.93032,
    'ER': 167.259, 'TM': 168.93421, 'YB': 173.04, 'LU': 174.9668,
    'SC': 44.955912, 'GA': 69.723, 'GE': 72.64, 'AS': 74.92160,
    'SE': 78.96, 'BR': 79.904, 'KR': 83.798, 'RB': 85.4678,
    'SR': 87.62, 'BA': 137.327, 'RA': 226.0, 'AC': 227.0,
    'TH': 232.03806, 'PA': 231.03588, 'NP': 237.0, 'PU': 244.0,
    'AM': 243.0, 'CM': 247.0, 'BK': 247.0, 'CF': 251.0,
    'ES': 252.0, 'FM': 257.0, 'MD': 258.0, 'NO': 259.0,
    'LR': 262.0
}

class EnthalpyAnalyzer:
    def __init__(self):
        self.results_history = []
        self.fitting_results = []
        self.database_dir = Path("databases")
        self.database_dir.mkdir(exist_ok=True)
    
    def get_available_tdb_files(self):
        """Retrieve all TDB files from the databases directory"""
        try:
            return sorted([f.name for f in self.database_dir.glob("*.tdb")], key=str.lower)
        except Exception as e:
            st.error(f"Error accessing databases directory: {str(e)}")
            return []
    
    def save_uploaded_tdb(self, uploaded_file):
        """Save uploaded TDB file to databases directory"""
        try:
            save_path = self.database_dir / uploaded_file.name
            # Check if file already exists
            if save_path.exists():
                st.warning(f"File '{uploaded_file.name}' already exists in database. Overwriting...")
            
            with open(save_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            return str(save_path)
        except Exception as e:
            st.error(f"Error saving TDB file: {str(e)}")
            return None
    
    def calculate_alloy_molar_weight(self, composition):
        """Calculate molar weight of alloy from composition dictionary"""
        molar_weight = 0.0
        missing_elements = []
        
        for element, fraction in composition.items():
            if fraction <= 0:
                continue
                
            element_upper = element.upper().strip()
            if element_upper in MOLAR_WEIGHTS:
                molar_weight += fraction * MOLAR_WEIGHTS[element_upper]
            else:
                missing_elements.append(element_upper)
                # Use reasonable default for unknown elements
                molar_weight += fraction * 50.0
        
        if missing_elements:
            st.warning(f"Molar weights not found for elements: {', '.join(missing_elements)}. Using default 50 g/mol.")
        
        return molar_weight if molar_weight > 0 else 50.0  # Prevent division by zero
    
    def convert_to_specific_enthalpy(self, df, composition):
        """Convert molar enthalpy to specific enthalpy (J/kg)"""
        if 'Enthalpy_J_mol' not in df.columns:
            raise ValueError("DataFrame must contain 'Enthalpy_J_mol' column")
        
        molar_weight = self.calculate_alloy_molar_weight(composition)
        df['Enthalpy_J_kg'] = df['Enthalpy_J_mol'] / (molar_weight / 1000.0)  # Convert g/mol to kg/mol
        return df
    
    def sigmoid(self, x, k):
        """Sigmoid function for enthalpy fitting"""
        # Avoid overflow in exp for large values
        kx = np.clip(-k * x, -700, 700)  # np.exp overflows beyond ~700
        return 1 / (1 + np.exp(kx))
    
    def enthalpy_equation(self, T, A1, A2, Tm, DeltaHf, k, H298):
        """Enthalpy equation for curve fitting with numerical stability"""
        T = np.asarray(T)
        sigmoid_term = DeltaHf * self.sigmoid(T - Tm, k)
        linear_term = A1 * T + A2 * np.maximum(T - Tm, 0)  # Ensure non-negative for solid phase
        return linear_term + sigmoid_term + H298
    
    def format_dat_file(self, df, composition, metadata=None):
        """Format data in DAT file format with headers"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dat_lines = [
            "# Enthalpy Data File",
            f"# Generated: {timestamp}",
            f"# Composition: {', '.join([f'{e}={f:.4f}' for e, f in composition.items()])}"
        ]
        
        if metadata:
            for key, value in metadata.items():
                dat_lines.append(f"# {key}: {value}")
        
        dat_lines.append("#" + "-"*60)
        dat_lines.append("# Temperature(K)    Enthalpy(J/mol)    Enthalpy(J/kg)")
        dat_lines.append("#" + "-"*60)
        
        for _, row in df.iterrows():
            dat_lines.append(f"{row['Temperature_K']:15.2f} {row['Enthalpy_J_mol']:18.4f} {row['Enthalpy_J_kg']:18.4f}")
        
        return "\n".join(dat_lines)

def create_enhanced_visualization(df, composition, material_name="Alloy"):
    """Create publication-quality dual-axis visualization"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), dpi=100)
    
    # Color scheme based on composition
    cmap = plt.cm.viridis
    color_idx = hash(material_name) % 256 / 256.0
    line_color = cmap(color_idx)
    
    # Molar enthalpy plot
    ax1.plot(df['Temperature_K'], df['Enthalpy_J_mol'], 
             color=line_color, linewidth=2.5, marker='o', markersize=4, 
             markevery=max(1, len(df)//20), label=f'{material_name} (Molar)')
    ax1.set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Enthalpy (J/mol)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Molar Enthalpy vs Temperature - {material_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=10)
    
    # Specific enthalpy plot
    ax2.plot(df['Temperature_K'], df['Enthalpy_J_kg'], 
             color=line_color, linewidth=2.5, marker='s', markersize=4,
             markevery=max(1, len(df)//20), label=f'{material_name} (Specific)')
    ax2.set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Specific Enthalpy (J/kg)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Specific Enthalpy vs Temperature - {material_name}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=10)
    
    # Add composition annotation
    comp_text = ', '.join([f'{e}={f:.3f}' for e, f in list(composition.items())[:4]])
    if len(composition) > 4:
        comp_text += f", ... (+{len(composition)-4} more)"
    
    fig.text(0.5, 0.02, f'Composition: {comp_text}', 
             ha='center', fontsize=10, style='italic', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

def create_comparison_visualization(analyzer, selected_indices):
    """Create enhanced multi-material comparison visualization"""
    if not selected_indices or not analyzer.results_history:
        return None
    
    fig = plt.figure(figsize=(14, 10), dpi=100)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])  # Molar enthalpy comparison (full width)
    ax2 = fig.add_subplot(gs[1, 0])  # Specific enthalpy comparison
    ax3 = fig.add_subplot(gs[1, 1])  # ΔH comparison bar chart
    
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(selected_indices))))
    
    # Plot molar and specific enthalpy
    delta_h_values = []
    material_names = []
    
    for i, idx in enumerate(selected_indices):
        if idx >= len(analyzer.results_history):
            continue
            
        result = analyzer.results_history[idx]
        data = result['data']
        name = result['name']
        
        # Molar enthalpy
        ax1.plot(data['Temperature_K'], data['Enthalpy_J_mol'],
                color=colors[i], linewidth=2.5, label=name, alpha=0.9)
        
        # Specific enthalpy
        ax2.plot(data['Temperature_K'], data['Enthalpy_J_kg'],
                color=colors[i], linewidth=2.5, label=name, alpha=0.9)
        
        # Calculate ΔH for bar chart
        delta_h = data['Enthalpy_J_mol'].max() - data['Enthalpy_J_mol'].min()
        delta_h_values.append(delta_h)
        material_names.append(name)
    
    # Format molar enthalpy plot
    ax1.set_xlabel('Temperature (K)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Enthalpy (J/mol)', fontsize=11, fontweight='bold')
    ax1.set_title('Molar Enthalpy Comparison Across Materials', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=9, ncol=max(1, len(selected_indices)//4))
    
    # Format specific enthalpy plot
    ax2.set_xlabel('Temperature (K)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Specific Enthalpy (J/kg)', fontsize=11, fontweight='bold')
    ax2.set_title('Specific Enthalpy Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=9)
    
    # ΔH bar chart
    bars = ax3.barh(material_names, delta_h_values, color=colors[:len(material_names)])
    ax3.set_xlabel('ΔH (J/mol)', fontsize=11, fontweight='bold')
    ax3.set_title('Total Enthalpy Change (ΔH)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels on bars
    for bar, value in zip(bars, delta_h_values):
        ax3.text(value, bar.get_y() + bar.get_height()/2, 
                f' {value:,.0f}', va='center', fontsize=9)
    
    plt.suptitle('Multi-Material Enthalpy Comparison Dashboard', 
                fontsize=16, fontweight='bold', y=0.995)
    
    return fig

def main():
    st.markdown('<h1 class="main-header">🔥 Thermodynamic Enthalpy Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive tool for thermodynamic calculations, curve fitting, and multi-material comparison")
    st.markdown("---")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnthalpyAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Create tabs with intuitive icons
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Enthalpy Computation",
        "📊 Curve Fitting & Analysis", 
        "🔄 Multi-Material Comparison",
        "ℹ️ Help & Settings"
    ])
    
    # ==================== TAB 1: Enthalpy Computation ====================
    with tab1:
        st.header("🔬 Enthalpy Computation from TDB Files")
        
        # File selection section
        st.subheader("📁 TDB File Selection")
        col_file1, col_file2 = st.columns([1.2, 1])
        
        with col_file1:
            # Get available TDB files
            available_tdb_files = analyzer.get_available_tdb_files()
            
            if available_tdb_files:
                tdb_source = st.radio(
                    "Source:",
                    ["Select from database directory", "Upload new TDB file"],
                    horizontal=True,
                    key="tdb_source"
                )
            else:
                st.info("No TDB files found in 'databases' directory. Please upload a file.")
                tdb_source = "Upload new TDB file"
            
            tdb_path = None
            
            if tdb_source == "Select from database directory" and available_tdb_files:
                selected_file = st.selectbox(
                    "Available TDB files in 'databases' directory:",
                    available_tdb_files,
                    help="Select a thermodynamic database file"
                )
                tdb_path = str(analyzer.database_dir / selected_file)
                st.success(f"✓ Selected: **{selected_file}**")
                
                # Show file info
                file_size = os.path.getsize(tdb_path) / 1024  # KB
                st.caption(f"File size: {file_size:.1f} KB")
            
            else:  # Upload option
                uploaded_file = st.file_uploader(
                    "Upload TDB file",
                    type=["tdb", "TDB"],
                    help="Upload a thermodynamic database file (.tdb)",
                    key="uploader1"
                )
                
                if uploaded_file is not None:
                    # Save to temp file for immediate use
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.tdb') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tdb_path = tmp_file.name
                    
                    st.success(f"✓ Uploaded: **{uploaded_file.name}**")
                    
                    # Option to save to database
                    if st.checkbox("💾 Save to 'databases' directory for future use", value=True):
                        saved_path = analyzer.save_uploaded_tdb(uploaded_file)
                        if saved_path:
                            st.info(f"Saved to: `{saved_path}`")
        
        # Database information and settings
        with col_file2:
            if tdb_path and os.path.exists(tdb_path):
                try:
                    with st.spinner("Loading thermodynamic database..."):
                        dbf = Database(tdb_path)
                    
                    st.subheader("🗄️ Database Information")
                    st.markdown(f"""
                    - **Elements**: {', '.join(sorted([e for e in dbf.elements if e != 'VA']))}
                    - **Phases**: {len(dbf.phases)} available
                    - **File**: `{os.path.basename(tdb_path)}`
                    """)
                    
                    # Show phases in expandable section
                    with st.expander("📋 View All Phases", expanded=False):
                        phases_list = sorted(dbf.phases.keys())
                        cols = st.columns(3)
                        for i, phase in enumerate(phases_list):
                            cols[i % 3].write(f"`{phase}`")
                
                except Exception as e:
                    st.error(f"❌ Error loading database: {str(e)}")
                    st.stop()
            else:
                st.info("👈 Please select or upload a TDB file to continue")
                st.stop()
        
        # Composition settings
        st.markdown("---")
        st.subheader("⚙️ Calculation Settings")
        
        col_set1, col_set2 = st.columns([1, 1.2])
        
        with col_set1:
            # Element selection
            available_elements = sorted([e for e in dbf.elements if e != 'VA'])
            if not available_elements:
                st.error("No valid elements found in database (excluding VA)")
                st.stop()
            
            selected_elements = st.multiselect(
                "Select alloy elements:",
                available_elements,
                default=available_elements[:min(3, len(available_elements))],
                help="Select elements to include in your alloy composition"
            )
            
            if not selected_elements:
                st.warning("⚠️ Please select at least one element")
                st.stop()
            
            # Composition input with smart distribution
            st.markdown("**Composition (mole fractions):**")
            composition = {}
            cols_comp = st.columns(len(selected_elements))
            
            # Calculate default values that sum to 1.0
            default_frac = 1.0 / len(selected_elements)
            
            for idx, element in enumerate(selected_elements):
                with cols_comp[idx % len(cols_comp)]:
                    # For last element, calculate to ensure sum=1.0
                    if idx == len(selected_elements) - 1:
                        remaining = 1.0 - sum(composition.values())
                        remaining = max(0.0, min(1.0, remaining))  # Clamp to [0,1]
                        fraction = st.number_input(
                            f"X({element})",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(f"{remaining:.4f}"),
                            step=0.01,
                            key=f"comp_{element}"
                        )
                    else:
                        fraction = st.number_input(
                            f"X({element})",
                            min_value=0.0,
                            max_value=1.0,
                            value=default_frac,
                            step=0.01,
                            key=f"comp_{element}"
                        )
                    
                    composition[element] = fraction
            
            # Validate composition sum
            comp_sum = sum(composition.values())
            if not (0.99 <= comp_sum <= 1.01):
                st.warning(f"⚠️ Composition sum = {comp_sum:.4f} (should be ≈1.0). Results may be inaccurate.")
        
        with col_set2:
            # Temperature settings
            st.markdown("**Temperature Range:**")
            col_temp1, col_temp2, col_temp3 = st.columns(3)
            with col_temp1:
                T_start = st.number_input("Start (K)", 100, 5000, 300, 10)
            with col_temp2:
                T_end = st.number_input("End (K)", T_start+10, 6000, 1500, 10)
            with col_temp3:
                T_step = st.number_input("Step (K)", 1, 200, 10)
            
            if T_end <= T_start:
                st.error("❌ End temperature must be greater than start temperature")
                st.stop()
            
            # Phase selection
            st.markdown("**Phase Selection:**")
            available_phases = sorted(dbf.phases.keys())
            default_phases = available_phases[:min(2, len(available_phases))]
            selected_phases = st.multiselect(
                "Equilibrium phases:",
                available_phases,
                default=default_phases,
                help="Select phases to consider in equilibrium calculation"
            )
            
            if not selected_phases:
                st.warning("⚠️ Please select at least one phase")
                st.stop()
            
            # Pressure setting
            P = st.number_input("Pressure (Pa)", 1000, 10000000, 101325, 1000)
        
        # Compute button
        st.markdown("---")
        if st.button("🚀 Compute Enthalpy", type="primary", use_container_width=True):
            with st.spinner("🔄 Performing equilibrium calculation... This may take a moment"):
                try:
                    # Prepare conditions
                    conditions = {
                        v.T: (T_start, T_end, T_step),
                        v.P: P
                    }
                    
                    # Add composition conditions (only for selected elements)
                    for element, fraction in composition.items():
                        if fraction > 0:
                            conditions[v.X(element)] = fraction
                    
                    # Add VA to elements list for calculation
                    elements_with_va = selected_elements + ['VA']
                    
                    # Perform equilibrium calculation
                    eq_result = equilibrium(
                        dbf,
                        elements_with_va,
                        selected_phases,
                        conditions,
                        output='HM',
                        verbose=False
                    )
                    
                    # Extract and process results
                    T_values = eq_result.T.values.flatten()
                    HM_values = eq_result.HM.values.flatten()
                    
                    # Create DataFrame
                    result_df = pd.DataFrame({
                        'Temperature_K': T_values,
                        'Enthalpy_J_mol': HM_values
                    })
                    
                    # Remove NaN values and sort by temperature
                    result_df = result_df.dropna().sort_values('Temperature_K').reset_index(drop=True)
                    
                    if len(result_df) == 0:
                        st.error("❌ Calculation returned no valid results. Try adjusting parameters.")
                        st.stop()
                    
                    # Add specific enthalpy
                    result_df = analyzer.convert_to_specific_enthalpy(result_df, composition)
                    
                    # Store results with metadata
                    material_name = "-".join([f"{e}{composition[e]:.2f}" for e in selected_elements])
                    result_info = {
                        'name': material_name,
                        'composition': composition,
                        'data': result_df,
                        'phases': selected_phases,
                        'tdb_file': os.path.basename(tdb_path),
                        'temperature_range': (T_start, T_end, T_step),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    analyzer.results_history.append(result_info)
                    
                    # Display success message
                    st.success(f"✅ Calculation completed successfully! Generated {len(result_df)} data points.")
                    
                    # Create and display visualization
                    fig = create_enhanced_visualization(result_df, composition, material_name)
                    st.pyplot(fig)
                    
                    # Display key metrics
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.metric("Min Enthalpy (J/mol)", f"{result_df['Enthalpy_J_mol'].min():,.0f}")
                    with col_m2:
                        st.metric("Max Enthalpy (J/mol)", f"{result_df['Enthalpy_J_mol'].max():,.0f}")
                    with col_m3:
                        st.metric("ΔH (J/mol)", f"{result_df['Enthalpy_J_mol'].max() - result_df['Enthalpy_J_mol'].min():,.0f}")
                    with col_m4:
                        st.metric("Data Points", len(result_df))
                    
                    # Download section
                    st.markdown('<div class="download-section">', unsafe_allow_html=True)
                    st.subheader("📥 Download Results")
                    
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        st.markdown("**CSV Formats:**")
                        
                        # Full data CSV
                        csv_full = result_df.to_csv(index=False)
                        st.download_button(
                            "📄 Download Full Data (CSV)",
                            data=csv_full,
                            file_name=f"enthalpy_{material_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            key=f"dl_csv_full_{len(analyzer.results_history)}"
                        )
                        
                        # Molar only
                        molar_df = result_df[['Temperature_K', 'Enthalpy_J_mol']]
                        st.download_button(
                            "📄 Download Molar Enthalpy Only (CSV)",
                            data=molar_df.to_csv(index=False),
                            file_name=f"molar_enthalpy_{material_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            key=f"dl_csv_molar_{len(analyzer.results_history)}"
                        )
                    
                    with col_dl2:
                        st.markdown("**DAT Format:**")
                        
                        # DAT format with metadata
                        metadata = {
                            'TDB File': os.path.basename(tdb_path),
                            'Phases': ', '.join(selected_phases),
                            'Pressure (Pa)': P,
                            'Temperature Range': f"{T_start}-{T_end} K"
                        }
                        dat_content = analyzer.format_dat_file(result_df, composition, metadata)
                        
                        st.download_button(
                            "📄 Download DAT Format (with metadata)",
                            data=dat_content,
                            file_name=f"enthalpy_{material_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.dat",
                            mime="text/plain",
                            key=f"dl_dat_{len(analyzer.results_history)}"
                        )
                        
                        # Specific enthalpy only
                        specific_df = result_df[['Temperature_K', 'Enthalpy_J_kg']]
                        st.download_button(
                            "📄 Download Specific Enthalpy Only (CSV)",
                            data=specific_df.to_csv(index=False),
                            file_name=f"specific_enthalpy_{material_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            key=f"dl_csv_specific_{len(analyzer.results_history)}"
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Data preview
                    with st.expander("🔍 View Complete Data Table"):
                        st.dataframe(result_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Calculation error: {str(e)}")
                    st.exception(e)  # Show full traceback in development
    
    # ==================== TAB 2: Curve Fitting ====================
    with tab2:
        st.header("📊 Curve Fitting & Analysis")
        
        if not analyzer.results_history:
            st.info("💡 No computed data available. Please perform calculations in the 'Enthalpy Computation' tab first.")
            st.stop()
        
        # Data source selection
        data_source = st.radio(
            "Select data source:",
            ["Use computed results from Tab 1", "Upload external CSV file"],
            horizontal=True
        )
        
        if data_source == "Use computed results from Tab 1":
            # Select from computed results
            result_options = [
                f"{i+1}. {res['name']} | {res['tdb_file']} | {res['temperature_range'][0]}-{res['temperature_range'][1]}K"
                for i, res in enumerate(analyzer.results_history)
            ]
            
            selected_result_idx = st.selectbox(
                "Select computed result:",
                range(len(result_options)),
                format_func=lambda x: result_options[x]
            )
            
            result_data = analyzer.results_history[selected_result_idx]['data']
            T_data = result_data['Temperature_K'].values
            H_data = result_data['Enthalpy_J_mol'].values
            material_name = analyzer.results_history[selected_result_idx]['name']
            composition_ref = analyzer.results_history[selected_result_idx]['composition']
            
        else:  # Upload CSV
            uploaded_csv = st.file_uploader(
                "Upload CSV with Temperature and Enthalpy columns",
                type=['csv'],
                key="uploader_csv"
            )
            
            if uploaded_csv is None:
                st.info("Please upload a CSV file to continue")
                st.stop()
            
            try:
                df_upload = pd.read_csv(uploaded_csv)
                st.write("Preview of uploaded data:")
                st.dataframe(df_upload.head(), use_container_width=True)
                
                # Column selection
                col1, col2 = st.columns(2)
                with col1:
                    temp_col = st.selectbox("Temperature column", df_upload.columns)
                with col2:
                    enthalpy_col = st.selectbox("Enthalpy column (J/mol)", df_upload.columns)
                
                T_data = df_upload[temp_col].values
                H_data = df_upload[enthalpy_col].values
                material_name = "Uploaded Data"
                composition_ref = {}
                
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
                st.stop()
        
        # Fitting parameters section
        st.markdown("---")
        st.subheader("⚙️ Fitting Parameters")
        
        # Smart initial guesses based on data
        T_mid = np.median(T_data)
        H_range = H_data.max() - H_data.min()
        H_slope = H_range / (T_data.max() - T_data.min())
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            A1_guess = st.number_input(
                "A₁ initial guess (J/mol·K)", 
                -100.0, 100.0, 
                float(min(50.0, max(5.0, H_slope * 0.7))), 
                0.1,
                help="Sensible heat coefficient for solid phase"
            )
            A2_guess = st.number_input(
                "A₂ initial guess (J/mol·K)", 
                -100.0, 100.0, 
                float(min(30.0, max(1.0, H_slope * 0.3))), 
                0.1,
                help="Additional sensible heat coefficient for liquid phase"
            )
            Tm_guess = st.number_input(
                "Tₘ initial guess (K)", 
                float(T_data.min()), float(T_data.max()), 
                float(T_mid), 
                1.0,
                help="Melting temperature"
            )
        
        with col_p2:
            DeltaHf_guess = st.number_input(
                "ΔHf initial guess (J/mol)", 
                -50000.0, 50000.0, 
                float(min(30000.0, max(5000.0, H_range * 0.6))), 
                100.0,
                help="Heat of fusion"
            )
            k_guess = st.number_input(
                "k initial guess (1/K)", 
                0.0001, 1.0, 
                0.01, 
                0.001,
                help="Sigmoid steepness parameter (controls melting transition sharpness)"
            )
            H298_guess = st.number_input(
                "H₂₉₈ initial guess (J/mol)", 
                -50000.0, 50000.0, 
                float(H_data.min() * 0.9), 
                100.0,
                help="Reference enthalpy at 298 K"
            )
        
        # Composition reference (for metadata)
        st.markdown("---")
        st.subheader("ℹ️ Composition Reference (for metadata only)")
        st.caption("These values are stored with fitting results but don't affect the fitting calculation")
        
        elements_common = ['Ag', 'Al', 'Au', 'Bi', 'Cu', 'In', 'Ni', 'Pb', 'Sn', 'Ti', 'V', 'Fe', 'Cr']
        comp_cols = st.columns(4)
        mole_fractions = {}
        
        # Pre-fill if composition available from computed data
        for i, element in enumerate(elements_common):
            with comp_cols[i % 4]:
                default_val = composition_ref.get(element, 0.0) if composition_ref else 0.0
                mole_fractions[element] = st.number_input(
                    f"X({element})",
                    0.0, 1.0, 
                    float(default_val),
                    0.01,
                    key=f"fit_comp_{element}"
                )
        
        # Fit button
        st.markdown("---")
        if st.button("🎯 Perform Curve Fit", type="primary", use_container_width=True):
            with st.spinner("Fitting curve to data..."):
                try:
                    # Perform curve fitting with robust settings
                    initial_guess = [A1_guess, A2_guess, Tm_guess, DeltaHf_guess, k_guess, H298_guess]
                    
                    # Bounds to ensure physical meaning
                    lower_bounds = [-100, -100, T_data.min(), 0, 1e-6, -1e6]
                    upper_bounds = [100, 100, T_data.max(), 1e6, 1.0, 1e6]
                    
                    fit_params, pcov = curve_fit(
                        analyzer.enthalpy_equation,
                        T_data,
                        H_data,
                        p0=initial_guess,
                        bounds=(lower_bounds, upper_bounds),
                        maxfev=10000
                    )
                    
                    A1_fit, A2_fit, Tm_fit, DeltaHf_fit, k_fit, H298_fit = fit_params
                    
                    # Generate smooth fitted curve
                    T_fit = np.linspace(T_data.min(), T_data.max(), 1000)
                    H_fit = analyzer.enthalpy_equation(T_fit, *fit_params)
                    
                    # Calculate statistics
                    H_pred = analyzer.enthalpy_equation(T_data, *fit_params)
                    residuals = H_data - H_pred
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((H_data - np.mean(H_data))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    rmse = np.sqrt(np.mean(residuals**2))
                    
                    # Store fitting results
                    fit_result = {
                        'material_name': material_name,
                        'coefficients': {
                            'A1': A1_fit,
                            'A2': A2_fit,
                            'Tm': Tm_fit,
                            'DeltaHf': DeltaHf_fit,
                            'k': k_fit,
                            'H298': H298_fit
                        },
                        'statistics': {
                            'r_squared': r_squared,
                            'rmse': rmse,
                            'data_points': len(T_data)
                        },
                        'composition': mole_fractions,
                        'temperature_range': [float(T_data.min()), float(T_data.max())],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    analyzer.fitting_results.append(fit_result)
                    
                    # Display results
                    st.success(f"✅ Fitting completed! R² = {r_squared:.6f}, RMSE = {rmse:.2f} J/mol")
                    
                    # Create comprehensive visualization
                    fig = plt.figure(figsize=(14, 10), dpi=100)
                    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
                    
                    # Main fit plot
                    ax1 = fig.add_subplot(gs[0:2, :])
                    ax1.scatter(T_data, H_data, alpha=0.6, label='Original Data', 
                               color='#1E88E5', s=40, edgecolors='white', linewidth=0.5)
                    ax1.plot(T_fit, H_fit, 'r-', linewidth=3, label='Fitted Curve', alpha=0.9)
                    ax1.axvline(Tm_fit, color='green', linestyle='--', alpha=0.7, 
                               label=f'Melting Point Tₘ = {Tm_fit:.1f} K')
                    ax1.set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Enthalpy (J/mol)', fontsize=12, fontweight='bold')
                    ax1.set_title(f'Enthalpy-Temperature Curve Fitting - {material_name}', 
                                 fontsize=14, fontweight='bold')
                    ax1.grid(True, alpha=0.3, linestyle='--')
                    ax1.legend(loc='best', fontsize=10)
                    
                    # Residual plot
                    ax2 = fig.add_subplot(gs[2, 0])
                    ax2.scatter(T_data, residuals, alpha=0.7, color='#FF7043', s=30)
                    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
                    ax2.set_xlabel('Temperature (K)', fontsize=10)
                    ax2.set_ylabel('Residuals (J/mol)', fontsize=10)
                    ax2.set_title('Residual Plot', fontsize=11, fontweight='bold')
                    ax2.grid(True, alpha=0.3, linestyle='--')
                    
                    # Coefficient summary
                    ax3 = fig.add_subplot(gs[2, 1])
                    ax3.axis('off')
                    
                    coeff_text = (
                        f"Fitted Parameters:\n\n"
                        f"A₁ = {A1_fit:.3f} J/(mol·K)\n"
                        f"A₂ = {A2_fit:.3f} J/(mol·K)\n"
                        f"Tₘ = {Tm_fit:.1f} K\n"
                        f"ΔHf = {DeltaHf_fit:,.0f} J/mol\n"
                        f"k = {k_fit:.5f} 1/K\n"
                        f"H₂₉₈ = {H298_fit:,.0f} J/mol\n\n"
                        f"Goodness of Fit:\n"
                        f"R² = {r_squared:.6f}\n"
                        f"RMSE = {rmse:.2f} J/mol"
                    )
                    
                    ax3.text(0.1, 0.95, coeff_text, transform=ax3.transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Download section
                    st.markdown('<div class="download-section">', unsafe_allow_html=True)
                    st.subheader("📥 Download Fitting Results")
                    
                    col_f1, col_f2, col_f3 = st.columns(3)
                    
                    # Coefficients CSV
                    with col_f1:
                        coeff_df = pd.DataFrame([{
                            'Material': material_name,
                            'A1_J_per_mol_K': A1_fit,
                            'A2_J_per_mol_K': A2_fit,
                            'Tm_K': Tm_fit,
                            'DeltaHf_J_per_mol': DeltaHf_fit,
                            'k_1_per_K': k_fit,
                            'H298_J_per_mol': H298_fit,
                            'R_squared': r_squared,
                            'RMSE_J_per_mol': rmse,
                            'Data_Points': len(T_data),
                            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            **{f'X_{element}': mole_fractions[element] for element in elements_common}
                        }])
                        
                        st.download_button(
                            "📄 Coefficients (CSV)",
                            data=coeff_df.to_csv(index=False),
                            file_name=f"fitting_coeffs_{material_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    
                    # Fitted curve data
                    with col_f2:
                        fitted_df = pd.DataFrame({
                            'Temperature_K': T_fit,
                            'Enthalpy_Fitted_J_mol': H_fit
                        })
                        
                        st.download_button(
                            "📄 Fitted Curve (CSV)",
                            data=fitted_df.to_csv(index=False),
                            file_name=f"fitted_curve_{material_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    
                    # Complete results JSON
                    with col_f3:
                        json_data = json.dumps(fit_result, indent=4)
                        st.download_button(
                            "📄 Full Results (JSON)",
                            data=json_data,
                            file_name=f"fitting_results_{material_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json"
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Equation display
                    st.markdown("---")
                    st.subheader("🧮 Fitted Equation")
                    st.latex(rf"""
                    H(T) = {A1_fit:.3f} \cdot T + {A2_fit:.3f} \cdot \max(T - {Tm_fit:.1f}, 0) + 
                    {DeltaHf_fit:,.0f} \cdot \frac{{1}}{{1 + e^{{-{k_fit:.5f}(T - {Tm_fit:.1f})}}}} + {H298_fit:,.0f}
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Fitting error: {str(e)}")
                    st.exception(e)
    
    # ==================== TAB 3: Multi-Material Comparison ====================
    with tab3:
        st.header("🔄 Multi-Material Comparison")
        
        if not analyzer.results_history:
            st.info("💡 No computed data available for comparison. Please perform calculations in the 'Enthalpy Computation' tab first.")
            st.stop()
        
        # Selection interface
        st.subheader("✅ Select Materials to Compare")
        
        # Create selection options with informative labels
        selection_options = []
        for i, res in enumerate(analyzer.results_history):
            label = f"{res['name']} | {res['tdb_file']} | {len(res['data'])} pts"
            selection_options.append((i, label))
        
        # Multiselect with max 6 materials for clarity
        selected_labels = st.multiselect(
            "Select up to 6 materials for comparison:",
            [label for _, label in selection_options],
            default=[selection_options[i][1] for i in range(min(3, len(selection_options)))],
            max_selections=6
        )
        
        if not selected_labels:
            st.warning("⚠️ Please select at least one material for comparison")
            st.stop()
        
        # Get indices of selected materials
        selected_indices = []
        for label in selected_labels:
            for idx, opt_label in selection_options:
                if opt_label == label:
                    selected_indices.append(idx)
                    break
        
        # Create and display comparison visualization
        with st.spinner("Generating comparison visualization..."):
            fig = create_comparison_visualization(analyzer, selected_indices)
            if fig:
                st.pyplot(fig)
        
        # Comparison table
        st.markdown("---")
        st.subheader("📊 Comparison Summary Table")
        
        summary_data = []
        for idx in selected_indices:
            res = analyzer.results_history[idx]
            data = res['data']
            
            # Calculate metrics
            min_h = data['Enthalpy_J_mol'].min()
            max_h = data['Enthalpy_J_mol'].max()
            delta_h = max_h - min_h
            avg_slope = delta_h / (data['Temperature_K'].max() - data['Temperature_K'].min())
            
            summary_data.append({
                'Material': res['name'],
                'TDB File': res['tdb_file'],
                'Phases': ', '.join(res['phases'][:3]) + ('...' if len(res['phases']) > 3 else ''),
                'Temp Range (K)': f"{res['temperature_range'][0]}-{res['temperature_range'][1]}",
                'Data Points': len(data),
                'Min H (J/mol)': f"{min_h:,.0f}",
                'Max H (J/mol)': f"{max_h:,.0f}",
                'ΔH (J/mol)': f"{delta_h:,.0f}",
                'Avg dH/dT (J/mol·K)': f"{avg_slope:.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Download options
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
        st.subheader("📥 Download Comparison Data")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            # Combined data CSV
            combined_data = {'Temperature_K': analyzer.results_history[selected_indices[0]]['data']['Temperature_K'].values}
            
            for idx in selected_indices:
                res = analyzer.results_history[idx]
                name_clean = res['name'].replace(' ', '_').replace('-', '_')
                combined_data[f"{name_clean}_H_molar"] = res['data']['Enthalpy_J_mol'].values
                combined_data[f"{name_clean}_H_specific"] = res['data']['Enthalpy_J_kg'].values
            
            combined_df = pd.DataFrame(combined_data)
            
            st.download_button(
                "📄 Combined Data (CSV)",
                data=combined_df.to_csv(index=False),
                file_name=f"multi_material_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        with col_c2:
            # Summary table CSV
            st.download_button(
                "📄 Summary Table (CSV)",
                data=summary_df.to_csv(index=False),
                file_name=f"comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== TAB 4: Help & Settings ====================
    with tab4:
        st.header("ℹ️ Help & Application Settings")
        
        col_h1, col_h2 = st.columns([1, 1.2])
        
        with col_h1:
            st.subheader("📖 User Guide")
            st.markdown("""
            ### Quick Start Guide
            
            **1. Enthalpy Computation Tab**
            - Select or upload a TDB file
            - Choose elements and set compositions (sum to 1.0)
            - Define temperature range and select phases
            - Click "Compute Enthalpy" to generate results
            - Download results in CSV or DAT format
            
            **2. Curve Fitting Tab**
            - Select computed data or upload CSV
            - Adjust initial parameter guesses if needed
            - Click "Perform Curve Fit" to fit the equation:
            
            """)
            
            st.latex(r"""
            H(T) = A_1 T + A_2 (T - T_m) + \Delta H_f \cdot \frac{1}{1 + e^{-k(T - T_m)}} + H_{298}
            """)
            
            st.markdown("""
            **3. Multi-Material Comparison**
            - Compute multiple materials in Tab 1
            - Select materials to compare in this tab
            - Analyze differences in enthalpy behavior
            
            ### Tips for Best Results
            - Start with binary or ternary systems for faster calculations
            - For curve fitting, ensure your data covers both solid and liquid regions
            - Use the residual plot to assess fit quality
            - Save interesting TDB files to the database directory for reuse
            """)
        
        with col_h2:
            st.subheader("⚙️ Application Management")
            
            # Session data status
            st.markdown("#### Current Session Status")
            st.metric("Computed Results", len(analyzer.results_history))
            st.metric("Fitting Results", len(analyzer.fitting_results))
            st.metric("TDB Files in Database", len(analyzer.get_available_tdb_files()))
            
            # Clear session data
            if st.button("🗑️ Clear All Session Data", type="secondary"):
                analyzer.results_history = []
                analyzer.fitting_results = []
                st.success("✅ Session data cleared successfully!")
                st.rerun()
            
            # Database management
            st.markdown("#### Database Directory Management")
            
            tdb_files = analyzer.get_available_tdb_files()
            if tdb_files:
                st.write(f"**Found {len(tdb_files)} TDB files:**")
                
                for tdb in tdb_files:
                    col_f1, col_f2, col_f3 = st.columns([3, 1, 1])
                    with col_f1:
                        st.caption(f"`{tdb}`")
                    with col_f2:
                        file_path = analyzer.database_dir / tdb
                        if st.button("ℹ️", key=f"info_{tdb}", help="File info"):
                            size_kb = os.path.getsize(file_path) / 1024
                            st.info(f"{tdb}\nSize: {size_kb:.1f} KB")
                    with col_f3:
                        if st.button("🗑️", key=f"delete_{tdb}", help="Delete file"):
                            try:
                                os.remove(file_path)
                                st.success(f"Deleted {tdb}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting {tdb}: {str(e)}")
            else:
                st.info("No TDB files in database directory. Upload files in Tab 1 to populate.")
            
            # Custom element addition
            st.markdown("#### Add Custom Element Molar Weight")
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                new_element = st.text_input("Element symbol (e.g., 'RE')", key="new_elem")
            with col_e2:
                new_weight = st.number_input("Molar weight (g/mol)", 0.0, 500.0, 50.0, 0.1)
            with col_e3:
                if st.button("➕ Add Element"):
                    if new_element.strip():
                        elem_key = new_element.strip().upper()
                        MOLAR_WEIGHTS[elem_key] = new_weight
                        st.success(f"✅ Added {elem_key}: {new_weight} g/mol")
                    else:
                        st.warning("⚠️ Please enter an element symbol")
            
            # About section
            st.markdown("---")
            st.subheader("ℹ️ About")
            st.markdown("""
            **Thermodynamic Enthalpy Analyzer Pro**
            
            A comprehensive tool for thermodynamic calculations using the CALPHAD method.
            
            - **Core Libraries**: pycalphad, scipy, xarray
            - **Visualization**: matplotlib, streamlit
            - **Data Formats**: CSV, DAT with metadata
            
            © 2026 Thermodynamic Analysis Toolkit
            """)
    
    # Footer
    st.markdown("---")
    st.caption("🔥 Thermodynamic Enthalpy Analyzer Pro | Powered by pycalphad & Streamlit | Session ID: " + 
               datetime.now().strftime("%Y%m%d%H%M%S"))

if __name__ == "__main__":
    main()
