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
import base64
import io
from typing import Dict, List, Optional, Tuple

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
        background: linear-gradient(90deg, #1E88E5, #4A00E0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding: 12px 24px;
        font-weight: 600;
        border: 1px solid #ddd;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
        border-color: #1E88E5;
        box-shadow: 0 4px 6px rgba(30, 136, 229, 0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        margin-bottom: 15px;
        color: white;
        border: none;
    }
    .download-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 25px;
        border-radius: 15px;
        margin-top: 25px;
        border: 1px solid #ddd;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .phase-container {
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #28a745;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(40, 167, 69, 0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #ffc107;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(255, 193, 7, 0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #17a2b8;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(23, 162, 184, 0.1);
    }
    .element-badge {
        display: inline-block;
        padding: 4px 12px;
        margin: 2px;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .phase-badge {
        display: inline-block;
        padding: 4px 12px;
        margin: 2px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .data-point {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        text-align: center;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Comprehensive molar weights database (expanded)
MOLAR_WEIGHTS = {
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
    'LR': 262.0, 'LI': 6.941, 'BE': 9.012182, 'B': 10.811,
    'NA': 22.989769, 'P': 30.973762, 'S': 32.065, 'CL': 35.453,
    'K': 39.0983, 'CA': 40.078, 'F': 18.9984032
}

class EnthalpyAnalyzer:
    def __init__(self):
        self.results_history = []
        self.fitting_results = []
        self.database_dir = Path("databases")
        self.database_dir.mkdir(exist_ok=True)
        self._ensure_default_tdb()
    
    def _ensure_default_tdb(self):
        """Create a default TDB file if none exists"""
        default_tdb_files = list(self.database_dir.glob("*.tdb"))
        
        if not default_tdb_files:
            # Create a simple example TDB file
            example_tdb_content = """$ Example TDB File for Binary Al-Cu System
$ Created by Thermodynamic Enthalpy Analyzer Pro
$
 ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8322E+01!
 ELEMENT CU   FCC_A1                    6.3546E+01  5.0041E+03  3.3150E+01!
$
 FUNCTION GHSERAL    2.98150E+02  -7976.15+137.093038*T-24.3671976*T*LN(T)
     -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1);  7.00000E+02  Y
     -11276.24+223.048446*T-38.5844296*T*LN(T)+.018531982*T**2
     -5.764227E-06*T**3+74092*T**(-1);  9.33600E+02  Y
     -11278.378+188.684153*T-31.748192*T*LN(T)-1.230524E+28*T**(-9);  2.90000E+03  N !
 FUNCTION GHSERCU    2.98150E+02  -7770.458+130.485235*T-24.112392*T*LN(T)
     -.00265684*T**2+1.29223E-07*T**3+52478*T**(-1);  1.35777E+03  Y
     -13542.026+183.803828*T-31.38*T*LN(T)+2.64313E+31*T**(-9);  3.20000E+03  N !
$
 TYPE_DEFINITION % SEQ *!
$
 PHASE LIQUID %  1  1.0  !
 CONSTITUENT LIQUID :AL,CU : !
$
 PARAMETER G(LIQUID,AL;0)  2.98150E+02  +11005.029-11.840849*T
      +7.9337E-20*T**7+GHSERAL#;  9.33600E+02  Y
      +10482.382-11.253974*T+1.231E+28*T**(-9)+GHSERAL#;  2.90000E+03  N !
 PARAMETER G(LIQUID,CU;0)  2.98150E+02  +12964.735-9.511904*T
      +5.8494E-21*T**7+GHSERCU#;  1.35777E+03  Y
      +13924.446-9.511904*T+2.64313E+31*T**(-9)+GHSERCU#;  3.20000E+03  N !
 PARAMETER G(LIQUID,AL,CU;0)  2.98150E+02  -47046.58+6.75*T;  6.00000E+03  N !
 PARAMETER G(LIQUID,AL,CU;1)  2.98150E+02  +21202.8352-9.67484*T;  6.00000E+03  N !
$
 PHASE FCC_A1 %  2  1.0  1.0  !
 CONSTITUENT FCC_A1 :AL,CU : VA : !
$
 PARAMETER G(FCC_A1,AL:VA;0)  2.98150E+02  +GHSERAL#;  6.00000E+03  N !
 PARAMETER G(FCC_A1,CU:VA;0)  2.98150E+02  +GHSERCU#;  6.00000E+03  N !
 PARAMETER G(FCC_A1,AL,CU:VA;0)  2.98150E+02  -12282.6+2.63791*T;  6.00000E+03  N !
 PARAMETER G(FCC_A1,AL,CU:VA;1)  2.98150E+02  +4580.9-1.7352*T;  6.00000E+03  N !
$
 LIST_OF_REFERENCES
 NUMBER  SOURCE
    1    'Example TDB for Al-Cu System - Thermodynamic Enthalpy Analyzer Pro'
$"""
            
            default_tdb_path = self.database_dir / "ALCU_EXAMPLE.tdb"
            with open(default_tdb_path, 'w') as f:
                f.write(example_tdb_content)
            
            # Create additional example TDB files
            self._create_additional_examples()
    
    def _create_additional_examples(self):
        """Create additional example TDB files for common systems"""
        # Fe-C example
        fec_tdb = """$ Fe-C Example TDB
 ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT FE   BCC_A2                    5.5847E+01  4.4890E+03  2.7280E+01!
 ELEMENT C    GRAPHITE                  1.2011E+01  1.0500E+03  5.7420E+00!
$
 PHASE LIQUID %  1  1.0  !
 CONSTITUENT LIQUID :FE,C : !
$
 PARAMETER G(LIQUID,FE;0)  2.98150E+02  +12040.17-6.55843*T+GHSERFE#;  6.00000E+03  N !
 PARAMETER G(LIQUID,C;0)  2.98150E+02  +117230.0-24.373*T+GHSERCC#;  6.00000E+03  N !
$
 PHASE BCC_A2 %  2  1.0  3.0  !
 CONSTITUENT BCC_A2 :FE:C : VA : !
$
 PARAMETER G(BCC_A2,FE:VA;0)  2.98150E+02  +GHSERFE#;  6.00000E+03  N !
$
 LIST_OF_REFERENCES
 NUMBER  SOURCE
    1    'Example TDB for Fe-C System'
$"""
        
        fec_path = self.database_dir / "FEC_EXAMPLE.tdb"
        with open(fec_path, 'w') as f:
            f.write(fec_tdb)
    
    def get_available_tdb_files(self):
        """Retrieve all TDB files from the databases directory"""
        try:
            tdb_files = []
            for ext in ['*.tdb', '*.TDB']:
                tdb_files.extend(self.database_dir.glob(ext))
            
            # Sort alphabetically, with example files first
            sorted_files = sorted(tdb_files, key=lambda x: (not 'EXAMPLE' in x.name.upper(), x.name.lower()))
            return [f.name for f in sorted_files]
        except Exception as e:
            st.error(f"Error accessing databases directory: {str(e)}")
            return []
    
    def save_uploaded_tdb(self, uploaded_file):
        """Save uploaded TDB file to databases directory"""
        try:
            save_path = self.database_dir / uploaded_file.name
            
            # Check if file already exists
            if save_path.exists():
                # Create a unique filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name_parts = uploaded_file.name.rsplit('.', 1)
                new_name = f"{name_parts[0]}_{timestamp}.{name_parts[1]}" if len(name_parts) > 1 else f"{uploaded_file.name}_{timestamp}"
                save_path = self.database_dir / new_name
            
            with open(save_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ TDB file saved successfully: `{save_path.name}`")
            return str(save_path)
        except Exception as e:
            st.error(f"❌ Error saving TDB file: {str(e)}")
            return None
    
    def calculate_alloy_molar_weight(self, composition: Dict[str, float]) -> float:
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
            st.warning(f"⚠️ Molar weights not found for elements: {', '.join(missing_elements)}. Using default 50 g/mol.")
        
        return molar_weight if molar_weight > 0 else 50.0
    
    def normalize_composition(self, composition: Dict[str, float]) -> Dict[str, float]:
        """Normalize composition so sum of fractions = 1.0"""
        total = sum(composition.values())
        if total == 0:
            return composition
        
        normalized = {k: v/total for k, v in composition.items()}
        return normalized
    
    def convert_to_specific_enthalpy(self, df: pd.DataFrame, composition: Dict[str, float]) -> pd.DataFrame:
        """Convert molar enthalpy to specific enthalpy (J/kg)"""
        if 'Enthalpy_J_mol' not in df.columns:
            raise ValueError("DataFrame must contain 'Enthalpy_J_mol' column")
        
        molar_weight = self.calculate_alloy_molar_weight(composition)
        df['Enthalpy_J_kg'] = df['Enthalpy_J_mol'] / (molar_weight / 1000.0)
        return df
    
    def sigmoid(self, x, k):
        """Sigmoid function for enthalpy fitting"""
        kx = np.clip(-k * x, -700, 700)
        return 1 / (1 + np.exp(kx))
    
    def enthalpy_equation(self, T, A1, A2, Tm, DeltaHf, k, H298):
        """Enthalpy equation for curve fitting"""
        T = np.asarray(T)
        sigmoid_term = DeltaHf * self.sigmoid(T - Tm, k)
        linear_term = A1 * T + A2 * np.maximum(T - Tm, 0)
        return linear_term + sigmoid_term + H298
    
    def format_dat_file(self, df, composition, metadata=None):
        """Format data in DAT file format with headers"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dat_lines = [
            "# =========================================================",
            "# ENTHALPY DATA FILE",
            "# =========================================================",
            f"# Generated: {timestamp}",
            f"# Software: Thermodynamic Enthalpy Analyzer Pro",
            "# =========================================================",
            "# COMPOSITION (Mole Fractions):"
        ]
        
        for element, fraction in composition.items():
            dat_lines.append(f"#   {element:3s}: {fraction:10.6f}")
        
        if metadata:
            dat_lines.append("# =========================================================")
            dat_lines.append("# METADATA:")
            for key, value in metadata.items():
                dat_lines.append(f"#   {key}: {value}")
        
        dat_lines.append("# =========================================================")
        dat_lines.append("# Temperature(K)    Enthalpy(J/mol)    Enthalpy(J/kg)")
        dat_lines.append("# =========================================================")
        
        for _, row in df.iterrows():
            dat_lines.append(f"  {row['Temperature_K']:15.2f} {row['Enthalpy_J_mol']:18.4f} {row['Enthalpy_J_kg']:18.4f}")
        
        return "\n".join(dat_lines)
    
    def detect_phase_transitions(self, df: pd.DataFrame, threshold: float = 1000) -> List[Dict]:
        """Detect phase transitions from enthalpy data"""
        transitions = []
        dH = np.diff(df['Enthalpy_J_mol'].values)
        dT = np.diff(df['Temperature_K'].values)
        derivatives = dH / dT
        
        # Find significant changes in derivative
        significant_changes = np.where(np.abs(np.diff(derivatives)) > threshold)[0]
        
        for idx in significant_changes:
            T_transition = (df['Temperature_K'].iloc[idx] + df['Temperature_K'].iloc[idx+1]) / 2
            H_change = df['Enthalpy_J_mol'].iloc[idx+1] - df['Enthalpy_J_mol'].iloc[idx]
            
            transitions.append({
                'temperature': T_transition,
                'enthalpy_change': H_change,
                'index': idx,
                'type': 'melting' if H_change > 0 else 'solidification'
            })
        
        return transitions

def create_enhanced_visualization(df, composition, material_name="Alloy", phase_transitions=None):
    """Create publication-quality dual-axis visualization"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=150)
    
    # Create a color gradient based on temperature
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(df)))
    
    # Molar enthalpy plot
    scatter1 = ax1.scatter(df['Temperature_K'], df['Enthalpy_J_mol'], 
                          c=colors, s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    line1, = ax1.plot(df['Temperature_K'], df['Enthalpy_J_mol'], 
                     color='#1E88E5', linewidth=2.5, alpha=0.9, label=f'{material_name}')
    
    ax1.set_xlabel('Temperature (K)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Enthalpy (J/mol)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Molar Enthalpy Evolution - {material_name}', fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add phase transition markers if available
    if phase_transitions:
        for trans in phase_transitions:
            ax1.axvline(trans['temperature'], color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax1.annotate(f"{trans['type'].title()}\n{trans['temperature']:.0f} K",
                        xy=(trans['temperature'], df['Enthalpy_J_mol'].min()),
                        xytext=(10, 20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=9, color='red', fontweight='bold')
    
    # Specific enthalpy plot
    scatter2 = ax2.scatter(df['Temperature_K'], df['Enthalpy_J_kg'], 
                          c=colors, s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    line2, = ax2.plot(df['Temperature_K'], df['Enthalpy_J_kg'], 
                     color='#FF7043', linewidth=2.5, alpha=0.9, label=f'{material_name}')
    
    ax2.set_xlabel('Temperature (K)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Specific Enthalpy (J/kg)', fontsize=13, fontweight='bold')
    ax2.set_title(f'Specific Enthalpy Evolution - {material_name}', fontsize=15, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = plt.Normalize(df['Temperature_K'].min(), df['Temperature_K'].max())
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Temperature (K)', fontsize=11, fontweight='bold')
    
    # Add composition annotation with badges
    comp_text = 'Composition: '
    for i, (element, fraction) in enumerate(list(composition.items())[:5]):
        comp_text += f'<span class="element-badge">{element}: {fraction:.3f}</span> '
    if len(composition) > 5:
        comp_text += f'<span class="element-badge">+{len(composition)-5} more</span>'
    
    fig.text(0.5, 0.01, comp_text, 
             ha='center', fontsize=11, style='italic', alpha=0.9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5),
             transform=fig.transFigure)
    
    plt.tight_layout(rect=[0, 0.05, 0.9, 0.97])
    return fig

def create_comparison_visualization(analyzer, selected_indices):
    """Create enhanced multi-material comparison visualization"""
    if not selected_indices or not analyzer.results_history:
        return None
    
    fig = plt.figure(figsize=(16, 12), dpi=150)
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
    
    ax1 = fig.add_subplot(gs[0:2, :])  # Main comparison plot
    ax2 = fig.add_subplot(gs[2, 0])     # ΔH comparison
    ax3 = fig.add_subplot(gs[2, 1])     # Cp comparison
    ax4 = fig.add_subplot(gs[2, 2])     # Material properties
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(selected_indices)))
    
    delta_h_values = []
    material_names = []
    avg_cp_values = []
    molar_weights = []
    
    for i, idx in enumerate(selected_indices):
        if idx >= len(analyzer.results_history):
            continue
            
        result = analyzer.results_history[idx]
        data = result['data']
        name = result['name']
        composition = result['composition']
        
        # Molar enthalpy comparison
        ax1.plot(data['Temperature_K'], data['Enthalpy_J_mol'],
                color=colors[i], linewidth=2.5, label=name, alpha=0.9,
                marker='o', markersize=3, markevery=len(data)//20)
        
        # Calculate ΔH
        delta_h = data['Enthalpy_J_mol'].max() - data['Enthalpy_J_mol'].min()
        delta_h_values.append(delta_h)
        material_names.append(name)
        
        # Calculate average Cp (dH/dT)
        dH = np.diff(data['Enthalpy_J_mol'].values)
        dT = np.diff(data['Temperature_K'].values)
        avg_cp = np.mean(dH/dT) if len(dH) > 0 else 0
        avg_cp_values.append(avg_cp)
        
        # Calculate molar weight
        molar_weight = analyzer.calculate_alloy_molar_weight(composition)
        molar_weights.append(molar_weight)
    
    # Format main plot
    ax1.set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Enthalpy (J/mol)', fontsize=12, fontweight='bold')
    ax1.set_title('Molar Enthalpy Comparison Across Materials', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=10, ncol=2, framealpha=0.9)
    
    # ΔH bar chart
    bars1 = ax2.barh(material_names, delta_h_values, color=colors, alpha=0.8)
    ax2.set_xlabel('ΔH (J/mol)', fontsize=11, fontweight='bold')
    ax2.set_title('Total Enthalpy Change', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    for bar, value in zip(bars1, delta_h_values):
        ax2.text(value, bar.get_y() + bar.get_height()/2, 
                f' {value:,.0f}', va='center', fontsize=9, fontweight='bold')
    
    # Cp bar chart
    bars2 = ax3.barh(material_names, avg_cp_values, color=colors, alpha=0.8)
    ax3.set_xlabel('Average Cp (J/mol·K)', fontsize=11, fontweight='bold')
    ax3.set_title('Average Heat Capacity', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    for bar, value in zip(bars2, avg_cp_values):
        ax3.text(value, bar.get_y() + bar.get_height()/2, 
                f' {value:.1f}', va='center', fontsize=9, fontweight='bold')
    
    # Material properties table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create table data
    table_data = []
    for i, idx in enumerate(selected_indices):
        result = analyzer.results_history[idx]
        table_data.append([
            material_names[i],
            f"{molar_weights[i]:.1f}",
            f"{delta_h_values[i]:,.0f}",
            f"{avg_cp_values[i]:.1f}",
            f"{len(result['data'])}"
        ])
    
    column_labels = ['Material', 'M.W.\n(g/mol)', 'ΔH\n(J/mol)', 'Avg Cp\n(J/mol·K)', 'Points']
    
    table = ax4.table(cellText=table_data,
                     colLabels=column_labels,
                     cellLoc='center',
                     loc='center',
                     colColours=['#1E88E5'] * 5,
                     colWidths=[0.25, 0.15, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    ax4.set_title('Material Properties Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Multi-Material Enthalpy Analysis Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    return fig

def create_composition_input_interface(available_elements, default_composition=None):
    """Create an interactive composition input interface"""
    if default_composition is None:
        default_composition = {}
    
    # Create columns for element inputs
    num_columns = min(6, len(available_elements))
    cols = st.columns(num_columns)
    
    composition = {}
    element_sliders = {}
    
    # First pass: collect user inputs
    for idx, element in enumerate(available_elements):
        col_idx = idx % num_columns
        with cols[col_idx]:
            default_val = default_composition.get(element, 0.0)
            
            # Use slider for better user experience
            fraction = st.slider(
                f"X({element})",
                min_value=0.0,
                max_value=1.0,
                value=float(default_val),
                step=0.01,
                key=f"comp_slider_{element}",
                help=f"Mole fraction of {element}"
            )
            
            element_sliders[element] = fraction
    
    # Calculate sum and normalize if needed
    total = sum(element_sliders.values())
    
    if total > 0:
        # Normalize to sum = 1
        if abs(total - 1.0) > 0.001:
            st.info(f"Composition sum = {total:.3f}. Auto-normalizing to 1.0")
            for element in element_sliders:
                element_sliders[element] /= total
        
        composition = element_sliders
    else:
        # Distribute equally if all zeros
        equal_fraction = 1.0 / len(available_elements)
        composition = {element: equal_fraction for element in available_elements}
        st.warning("All fractions set to 0. Using equal distribution.")
    
    # Display normalized composition
    st.markdown("**Normalized Composition:**")
    norm_cols = st.columns(min(8, len(composition)))
    
    for i, (element, fraction) in enumerate(list(composition.items())[:8]):
        with norm_cols[i % len(norm_cols)]:
            st.markdown(f'<div class="element-badge">{element}: {fraction:.3f}</div>', 
                       unsafe_allow_html=True)
    
    if len(composition) > 8:
        st.caption(f"... and {len(composition)-8} more elements")
    
    return composition

def main():
    st.markdown('<h1 class="main-header">🔥 Thermodynamic Enthalpy Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Computational Thermodynamics for Multi-Component Systems")
    st.markdown("*Enthalpy calculations, curve fitting, and material comparison using CALPHAD methodology*")
    st.markdown("---")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnthalpyAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Create tabs with enhanced layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Dashboard",
        "🔬 Enthalpy Computation", 
        "📊 Curve Fitting",
        "🔄 Multi-Material",
        "⚙️ Settings & Help"
    ])
    
    # ==================== TAB 1: Dashboard ====================
    with tab1:
        st.header("📊 Dashboard Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Computed Materials", len(analyzer.results_history))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("TDB Files", len(analyzer.get_available_tdb_files()))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Fitting Results", len(analyzer.fitting_results))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Active Session", "Online", delta="Now")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("---")
        st.subheader("🚀 Quick Actions")
        
        quick_cols = st.columns(5)
        
        with quick_cols[0]:
            if st.button("📁 Upload TDB", use_container_width=True):
                st.session_state.active_tab = "🔬 Enthalpy Computation"
                st.rerun()
        
        with quick_cols[1]:
            if st.button("🔄 Compute Example", use_container_width=True):
                # Auto-run example calculation
                st.info("Running example calculation...")
                st.session_state.run_example = True
                st.rerun()
        
        with quick_cols[2]:
            if st.button("📈 View All Results", use_container_width=True):
                st.session_state.active_tab = "🔄 Multi-Material"
                st.rerun()
        
        with quick_cols[3]:
            if st.button("⚙️ Settings", use_container_width=True):
                st.session_state.active_tab = "⚙️ Settings & Help"
                st.rerun()
        
        with quick_cols[4]:
            if st.button("🔄 Clear Session", type="secondary", use_container_width=True):
                analyzer.results_history = []
                analyzer.fitting_results = []
                st.success("Session cleared!")
                st.rerun()
        
        # Recent computations
        if analyzer.results_history:
            st.markdown("---")
            st.subheader("📋 Recent Computations")
            
            for i, result in enumerate(analyzer.results_history[-3:]):
                with st.expander(f"{i+1}. {result['name']} | {result['tdb_file']}", expanded=False):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**Composition:**")
                        for elem, frac in list(result['composition'].items())[:5]:
                            st.write(f"- {elem}: {frac:.4f}")
                    
                    with col_b:
                        st.write("**Statistics:**")
                        data = result['data']
                        st.write(f"Temperature: {result['temperature_range'][0]} - {result['temperature_range'][1]} K")
                        st.write(f"Data points: {len(data)}")
                        st.write(f"ΔH: {data['Enthalpy_J_mol'].max() - data['Enthalpy_J_mol'].min():,.0f} J/mol")
    
    # ==================== TAB 2: Enthalpy Computation ====================
    with tab2:
        st.header("🔬 Enthalpy Computation from TDB Files")
        
        # File management section
        with st.container():
            st.subheader("📁 TDB File Management")
            
            col_file1, col_file2 = st.columns([1.2, 1])
            
            with col_file1:
                # Get available TDB files
                available_tdb_files = analyzer.get_available_tdb_files()
                
                if available_tdb_files:
                    st.markdown("**Available Thermodynamic Databases:**")
                    tdb_options = ["Select from database"] + available_tdb_files
                    selected_tdb_option = st.selectbox(
                        "Choose TDB file:",
                        tdb_options,
                        index=0,
                        help="Select a thermodynamic database file from your local database directory"
                    )
                    
                    if selected_tdb_option != "Select from database":
                        tdb_path = str(analyzer.database_dir / selected_tdb_option)
                        st.success(f"✅ Selected: **{selected_tdb_option}**")
                        
                        # Show file info
                        try:
                            file_size = os.path.getsize(tdb_path) / 1024
                            st.caption(f"File size: {file_size:.1f} KB | Path: `{tdb_path}`")
                        except:
                            pass
                    else:
                        tdb_path = None
                else:
                    st.warning("⚠️ No TDB files found in the 'databases' directory.")
                    tdb_path = None
                
                # File upload section
                st.markdown("**Or upload new TDB file:**")
                uploaded_file = st.file_uploader(
                    "Drag and drop or click to upload",
                    type=["tdb", "TDB"],
                    help="Upload a thermodynamic database file (.tdb)",
                    key="uploader_tab2"
                )
                
                if uploaded_file is not None:
                    # Save to temp file for immediate use
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.tdb', mode='wb') as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tdb_path = tmp_file.name
                    
                    st.success(f"✅ Uploaded: **{uploaded_file.name}**")
                    
                    # Auto-save option
                    if st.checkbox("💾 Save to database directory", value=True):
                        saved_path = analyzer.save_uploaded_tdb(uploaded_file)
                        if saved_path:
                            st.info(f"Saved to: `{saved_path}`")
            
            with col_file2:
                if tdb_path and os.path.exists(tdb_path):
                    try:
                        with st.spinner("🔍 Loading database..."):
                            dbf = Database(tdb_path)
                        
                        st.markdown("**📊 Database Information**")
                        
                        # Display elements
                        elements_list = sorted([e for e in dbf.elements if e != 'VA'])
                        st.markdown(f"**Elements ({len(elements_list)}):**")
                        elements_html = ""
                        for elem in elements_list[:10]:
                            elements_html += f'<span class="element-badge">{elem}</span> '
                        if len(elements_list) > 10:
                            elements_html += f'<span class="element-badge">+{len(elements_list)-10}</span>'
                        st.markdown(elements_html, unsafe_allow_html=True)
                        
                        # Display phases
                        phases_list = sorted(dbf.phases.keys())
                        st.markdown(f"**Phases ({len(phases_list)}):**")
                        with st.expander("📋 View all phases", expanded=False):
                            cols_phases = st.columns(3)
                            for i, phase in enumerate(phases_list):
                                cols_phases[i % 3].markdown(f'<span class="phase-badge">{phase}</span>', 
                                                          unsafe_allow_html=True)
                        
                        # Database metadata
                        st.markdown("**📝 Metadata:**")
                        st.write(f"- Database elements: {', '.join(elements_list[:5])}{'...' if len(elements_list) > 5 else ''}")
                        st.write(f"- Total phases: {len(phases_list)}")
                        
                    except Exception as e:
                        st.error(f"❌ Error loading database: {str(e)}")
                        st.info("Please check if the TDB file is valid and try again.")
                        st.stop()
                else:
                    if not tdb_path:
                        st.info("👈 Please select or upload a TDB file to begin")
                    else:
                        st.error(f"❌ File not found: {tdb_path}")
                    st.stop()
        
        # Calculation parameters
        st.markdown("---")
        st.subheader("⚙️ Calculation Parameters")
        
        col_params1, col_params2 = st.columns([1, 1])
        
        with col_params1:
            # Element selection with search
            elements_list = sorted([e for e in dbf.elements if e != 'VA'])
            if not elements_list:
                st.error("No valid elements found in database (excluding VA)")
                st.stop()
            
            st.markdown("**Select Elements:**")
            search_element = st.text_input("Search elements:", "", 
                                          help="Type to filter elements")
            
            if search_element:
                filtered_elements = [e for e in elements_list 
                                   if search_element.upper() in e.upper()]
            else:
                filtered_elements = elements_list
            
            selected_elements = st.multiselect(
                "Choose elements for your alloy:",
                filtered_elements,
                default=elements_list[:min(3, len(elements_list))],
                help="Select elements to include in your alloy composition"
            )
            
            if not selected_elements:
                st.warning("⚠️ Please select at least one element")
                st.stop()
            
            # Composition input
            st.markdown("**Enter Composition:**")
            st.caption("For n-component system, provide n-1 fractions. The nth component will be calculated automatically.")
            
            # Smart composition input
            composition = {}
            remaining_fraction = 1.0
            
            # Create input fields for n-1 components
            for i, element in enumerate(selected_elements[:-1]):
                max_val = min(1.0, remaining_fraction)
                
                fraction = st.number_input(
                    f"Mole fraction X({element})",
                    min_value=0.0,
                    max_value=float(max_val),
                    value=float(1.0/len(selected_elements) if i == 0 else 0.0),
                    step=0.01,
                    key=f"comp_input_{element}"
                )
                
                composition[element] = fraction
                remaining_fraction -= fraction
            
            # Last element gets the remaining fraction
            last_element = selected_elements[-1]
            composition[last_element] = remaining_fraction
            
            # Display composition summary
            st.markdown("**📊 Composition Summary:**")
            comp_df = pd.DataFrame({
                'Element': list(composition.keys()),
                'Mole Fraction': list(composition.values())
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            # Validate composition
            comp_sum = sum(composition.values())
            if not (0.999 <= comp_sum <= 1.001):
                st.error(f"❌ Composition sum = {comp_sum:.4f} (must be exactly 1.0)")
                st.stop()
        
        with col_params2:
            # Temperature settings
            st.markdown("**🌡️ Temperature Range:**")
            col_temp1, col_temp2, col_temp3 = st.columns(3)
            with col_temp1:
                T_start = st.number_input("Start (K)", 100, 5000, 300, 10,
                                         help="Starting temperature in Kelvin")
            with col_temp2:
                T_end = st.number_input("End (K)", T_start+10, 6000, 1500, 10,
                                       help="End temperature in Kelvin")
            with col_temp3:
                T_step = st.number_input("Step (K)", 1, 200, 10,
                                        help="Temperature step size")
            
            if T_end <= T_start:
                st.error("❌ End temperature must be greater than start temperature")
                st.stop()
            
            # Phase selection
            st.markdown("**Select Phases for Equilibrium:**")
            available_phases = sorted(dbf.phases.keys())
            
            # Smart phase selection
            default_phases = []
            for phase in available_phases:
                if 'LIQUID' in phase.upper():
                    default_phases.append(phase)
                    break
            if not default_phases and available_phases:
                default_phases = available_phases[:min(2, len(available_phases))]
            
            selected_phases = st.multiselect(
                "Equilibrium phases to consider:",
                available_phases,
                default=default_phases,
                help="Select phases to include in the equilibrium calculation"
            )
            
            if not selected_phases:
                st.warning("⚠️ Please select at least one phase")
                st.stop()
            
            # Advanced settings
            with st.expander("⚙️ Advanced Settings", expanded=False):
                P = st.number_input("Pressure (Pa)", 1000, 10000000, 101325, 1000,
                                   help="System pressure in Pascals")
                output_grid = st.number_input("Output grid density", 10, 1000, 100,
                                             help="Number of points in output grid")
                
                calc_mode = st.selectbox(
                    "Calculation mode",
                    ["Fast", "Accurate", "Very Accurate"],
                    index=1,
                    help="Trade-off between speed and accuracy"
                )
        
        # Compute button with progress
        st.markdown("---")
        compute_col1, compute_col2, compute_col3 = st.columns([2, 1, 1])
        
        with compute_col2:
            compute_button = st.button("🚀 Compute Enthalpy", 
                                      type="primary", 
                                      use_container_width=True)
        
        with compute_col3:
            example_button = st.button("📚 Run Example", 
                                      type="secondary", 
                                      use_container_width=True,
                                      help="Run a pre-configured example calculation")
        
        if compute_button or example_button or getattr(st.session_state, 'run_example', False):
            if 'run_example' in st.session_state:
                del st.session_state.run_example
            
            with st.spinner("🔄 Performing equilibrium calculation... This may take a moment"):
                try:
                    # Prepare conditions
                    conditions = {
                        v.T: (T_start, T_end, T_step),
                        v.P: P,
                        v.N: 1
                    }
                    
                    # Add composition conditions
                    for element, fraction in composition.items():
                        if fraction > 0:
                            conditions[v.X(element)] = fraction
                    
                    # Add VA to elements list for calculation
                    elements_with_va = selected_elements + ['VA']
                    
                    # Perform equilibrium calculation
                    progress_bar = st.progress(0)
                    
                    # Step 1: Setup
                    progress_bar.progress(20)
                    
                    # Step 2: Calculate equilibrium
                    eq_result = equilibrium(
                        dbf,
                        elements_with_va,
                        selected_phases,
                        conditions,
                        output='HM',
                        verbose=False
                    )
                    
                    progress_bar.progress(80)
                    
                    # Extract and process results
                    T_values = eq_result.T.values.flatten()
                    HM_values = eq_result.HM.values.flatten()
                    
                    # Create DataFrame
                    result_df = pd.DataFrame({
                        'Temperature_K': T_values,
                        'Enthalpy_J_mol': HM_values
                    })
                    
                    # Remove NaN values and sort
                    result_df = result_df.dropna().sort_values('Temperature_K').reset_index(drop=True)
                    
                    if len(result_df) == 0:
                        st.error("❌ Calculation returned no valid results. Try adjusting parameters.")
                        st.stop()
                    
                    # Add specific enthalpy
                    result_df = analyzer.convert_to_specific_enthalpy(result_df, composition)
                    
                    # Detect phase transitions
                    phase_transitions = analyzer.detect_phase_transitions(result_df)
                    
                    progress_bar.progress(100)
                    
                    # Store results with metadata
                    material_name = f"{'-'.join([f'{e}' for e in selected_elements[:3]])}"
                    if len(selected_elements) > 3:
                        material_name += f"-{len(selected_elements)-3}more"
                    
                    result_info = {
                        'name': material_name,
                        'composition': composition,
                        'data': result_df,
                        'phases': selected_phases,
                        'tdb_file': os.path.basename(tdb_path) if tdb_path else 'uploaded.tdb',
                        'temperature_range': (T_start, T_end, T_step),
                        'phase_transitions': phase_transitions,
                        'molar_weight': analyzer.calculate_alloy_molar_weight(composition),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    analyzer.results_history.append(result_info)
                    
                    # Display success message
                    st.success(f"✅ Calculation completed successfully! Generated {len(result_df)} data points.")
                    
                    # Create and display visualization
                    fig = create_enhanced_visualization(result_df, composition, material_name, phase_transitions)
                    st.pyplot(fig)
                    
                    # Display key metrics in cards
                    st.markdown("### 📊 Key Metrics")
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    
                    with col_m1:
                        st.markdown('<div class="data-point">', unsafe_allow_html=True)
                        st.metric("Min Enthalpy", f"{result_df['Enthalpy_J_mol'].min():,.0f} J/mol")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_m2:
                        st.markdown('<div class="data-point">', unsafe_allow_html=True)
                        st.metric("Max Enthalpy", f"{result_df['Enthalpy_J_mol'].max():,.0f} J/mol")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_m3:
                        st.markdown('<div class="data-point">', unsafe_allow_html=True)
                        delta_h = result_df['Enthalpy_J_mol'].max() - result_df['Enthalpy_J_mol'].min()
                        st.metric("Total ΔH", f"{delta_h:,.0f} J/mol")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_m4:
                        st.markdown('<div class="data-point">', unsafe_allow_html=True)
                        molar_weight = analyzer.calculate_alloy_molar_weight(composition)
                        st.metric("Molar Weight", f"{molar_weight:.2f} g/mol")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display phase transition information
                    if phase_transitions:
                        st.markdown("### 🔥 Phase Transitions Detected")
                        for trans in phase_transitions:
                            col_t1, col_t2 = st.columns([1, 3])
                            with col_t1:
                                st.metric("Transition Temp", f"{trans['temperature']:.1f} K")
                            with col_t2:
                                st.write(f"**{trans['type'].title()}** | Enthalpy change: {trans['enthalpy_change']:,.0f} J/mol")
                    
                    # Enhanced download section
                    st.markdown('<div class="download-section">', unsafe_allow_html=True)
                    st.subheader("📥 Download Results")
                    
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    
                    with col_dl1:
                        st.markdown("**📊 Full Data (CSV)**")
                        csv_full = result_df.to_csv(index=False)
                        
                        st.download_button(
                            "📄 Download Full Dataset",
                            data=csv_full,
                            file_name=f"enthalpy_{material_name.replace(' ', '_')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_dl2:
                        st.markdown("**📈 Plot Data (CSV)**")
                        plot_df = result_df[['Temperature_K', 'Enthalpy_J_mol', 'Enthalpy_J_kg']]
                        
                        st.download_button(
                            "📄 Download Plot Data",
                            data=plot_df.to_csv(index=False),
                            file_name=f"plot_data_{material_name.replace(' ', '_')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_dl3:
                        st.markdown("**📝 DAT Format**")
                        metadata = {
                            'TDB File': os.path.basename(tdb_path) if tdb_path else 'uploaded.tdb',
                            'Phases': ', '.join(selected_phases),
                            'Pressure (Pa)': P,
                            'Temperature Range': f"{T_start}-{T_end} K, step={T_step}",
                            'Molar Weight (g/mol)': analyzer.calculate_alloy_molar_weight(composition)
                        }
                        dat_content = analyzer.format_dat_file(result_df, composition, metadata)
                        
                        st.download_button(
                            "📄 Download DAT File",
                            data=dat_content,
                            file_name=f"enthalpy_{material_name.replace(' ', '_')}.dat",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    # Additional export options
                    with st.expander("🔄 Additional Export Formats", expanded=False):
                        col_exp1, col_exp2 = st.columns(2)
                        
                        with col_exp1:
                            # JSON export
                            json_data = json.dumps({
                                'material': material_name,
                                'composition': composition,
                                'data': result_df.to_dict('records'),
                                'metadata': metadata
                            }, indent=4)
                            
                            st.download_button(
                                "📁 JSON Format",
                                data=json_data,
                                file_name=f"enthalpy_{material_name.replace(' ', '_')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        with col_exp2:
                            # Excel export
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                result_df.to_excel(writer, sheet_name='Enthalpy Data', index=False)
                                # Add summary sheet
                                summary_df = pd.DataFrame([{
                                    'Material': material_name,
                                    'Molar Weight (g/mol)': analyzer.calculate_alloy_molar_weight(composition),
                                    'Min Temp (K)': T_start,
                                    'Max Temp (K)': T_end,
                                    'Data Points': len(result_df)
                                }])
                                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                            
                            st.download_button(
                                "📗 Excel Format",
                                data=output.getvalue(),
                                file_name=f"enthalpy_{material_name.replace(' ', '_')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Data preview
                    with st.expander("🔍 View Complete Data Table", expanded=False):
                        st.dataframe(result_df, use_container_width=True, height=300)
                
                except Exception as e:
                    st.error(f"❌ Calculation error: {str(e)}")
                    st.exception(e)
    
    # ==================== TAB 3: Curve Fitting ====================
    with tab3:
        st.header("📊 Curve Fitting & Analysis")
        
        if not analyzer.results_history:
            st.info("💡 No computed data available. Please perform calculations in the 'Enthalpy Computation' tab first.")
            st.stop()
        
        # Data source selection
        data_source = st.radio(
            "Select data source:",
            ["Use computed results", "Upload external CSV file"],
            horizontal=True,
            index=0
        )
        
        if data_source == "Use computed results":
            # Select from computed results
            result_options = [
                f"{i+1}. {res['name']} | {res['tdb_file']} | {res['temperature_range'][0]}-{res['temperature_range'][1]}K | {len(res['data'])} pts"
                for i, res in enumerate(analyzer.results_history)
            ]
            
            selected_result_idx = st.selectbox(
                "Select computed result:",
                range(len(result_options)),
                format_func=lambda x: result_options[x],
                index=len(result_options)-1 if result_options else 0
            )
            
            result_data = analyzer.results_history[selected_result_idx]['data']
            T_data = result_data['Temperature_K'].values
            H_data = result_data['Enthalpy_J_mol'].values
            material_name = analyzer.results_history[selected_result_idx]['name']
            composition_ref = analyzer.results_history[selected_result_idx]['composition']
            
            st.success(f"✅ Loaded: {material_name} with {len(T_data)} data points")
        
        else:  # Upload CSV
            uploaded_csv = st.file_uploader(
                "Upload CSV with Temperature and Enthalpy columns",
                type=['csv', 'txt', 'dat'],
                key="uploader_csv_tab3"
            )
            
            if uploaded_csv is None:
                st.info("Please upload a CSV file to continue")
                st.stop()
            
            try:
                df_upload = pd.read_csv(uploaded_csv)
                
                st.write("**Preview of uploaded data:**")
                st.dataframe(df_upload.head(), use_container_width=True)
                
                # Column selection
                col1, col2 = st.columns(2)
                with col1:
                    temp_col = st.selectbox("Temperature column", df_upload.columns, index=0)
                with col2:
                    enthalpy_col = st.selectbox("Enthalpy column (J/mol)", df_upload.columns, index=1)
                
                # Validate data
                if len(df_upload) < 10:
                    st.warning("⚠️ Very few data points. Curve fitting may be unstable.")
                
                T_data = df_upload[temp_col].values
                H_data = df_upload[enthalpy_col].values
                material_name = "Uploaded Data"
                composition_ref = {}
                
                # Basic data validation
                if len(T_data) != len(H_data):
                    st.error("❌ Temperature and enthalpy arrays must have same length")
                    st.stop()
                
                st.success(f"✅ Loaded {len(T_data)} data points from uploaded file")
                
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
                st.stop()
        
        # Fitting parameters with smart initialization
        st.markdown("---")
        st.subheader("⚙️ Fitting Parameters")
        
        # Calculate smart initial guesses
        if len(T_data) > 0 and len(H_data) > 0:
            T_range = T_data.max() - T_data.min()
            H_range = H_data.max() - H_data.min()
            H_slope = H_range / T_range if T_range > 0 else 1.0
            T_mid = (T_data.max() + T_data.min()) / 2
            
            # Default guesses
            A1_guess_default = max(5.0, min(50.0, H_slope * 0.5))
            A2_guess_default = max(1.0, min(30.0, H_slope * 0.2))
            Tm_guess_default = T_mid
            DeltaHf_guess_default = max(5000.0, min(30000.0, H_range * 0.3))
            k_guess_default = 0.01
            H298_guess_default = H_data.min() if len(H_data) > 0 else 0
        else:
            A1_guess_default = 20.0
            A2_guess_default = 10.0
            Tm_guess_default = 1000.0
            DeltaHf_guess_default = 15000.0
            k_guess_default = 0.01
            H298_guess_default = 0.0
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.markdown("**🔧 Initial Parameter Guesses:**")
            A1_guess = st.number_input(
                "A₁ (J/mol·K)", 
                -100.0, 100.0, 
                float(A1_guess_default), 
                0.1,
                help="Sensible heat coefficient for solid phase"
            )
            A2_guess = st.number_input(
                "A₂ (J/mol·K)", 
                -100.0, 100.0, 
                float(A2_guess_default), 
                0.1,
                help="Additional sensible heat coefficient for liquid phase"
            )
            Tm_guess = st.number_input(
                "Tₘ (K)", 
                float(T_data.min()) if len(T_data) > 0 else 300.0, 
                float(T_data.max()) if len(T_data) > 0 else 3000.0, 
                float(Tm_guess_default), 
                1.0,
                help="Melting temperature"
            )
        
        with col_p2:
            st.markdown("**🎯 Additional Parameters:**")
            DeltaHf_guess = st.number_input(
                "ΔHf (J/mol)", 
                -50000.0, 50000.0, 
                float(DeltaHf_guess_default), 
                100.0,
                help="Heat of fusion"
            )
            k_guess = st.number_input(
                "k (1/K)", 
                0.0001, 1.0, 
                float(k_guess_default), 
                0.001,
                help="Sigmoid steepness parameter"
            )
            H298_guess = st.number_input(
                "H₂₉₈ (J/mol)", 
                -100000.0, 100000.0, 
                float(H298_guess_default), 
                100.0,
                help="Reference enthalpy at 298 K"
            )
        
        # Advanced fitting options
        with st.expander("⚙️ Advanced Fitting Options", expanded=False):
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                max_iterations = st.number_input("Max iterations", 100, 10000, 5000, 100)
                fit_method = st.selectbox("Fitting method", ["trf", "lm"], index=0)
            
            with col_adv2:
                loss_function = st.selectbox(
                    "Loss function",
                    ["linear", "soft_l1", "huber", "cauchy", "arctan"],
                    index=0
                )
                weight_data = st.checkbox("Weight by temperature", value=False)
        
        # Fit button
        st.markdown("---")
        if st.button("🎯 Perform Curve Fit", type="primary", use_container_width=True):
            with st.spinner("🔄 Fitting curve to data..."):
                try:
                    # Prepare data
                    if weight_data:
                        # Weight by temperature (higher weight at extreme temperatures)
                        weights = 1.0 / (1.0 + np.abs(T_data - np.mean(T_data)) / np.std(T_data))
                    else:
                        weights = None
                    
                    # Initial guess
                    initial_guess = [A1_guess, A2_guess, Tm_guess, DeltaHf_guess, k_guess, H298_guess]
                    
                    # Bounds for physical constraints
                    lower_bounds = [-200, -200, T_data.min() * 0.8, 0, 1e-6, -1e6]
                    upper_bounds = [200, 200, T_data.max() * 1.2, 1e7, 1.0, 1e6]
                    
                    # Perform curve fitting
                    fit_params, pcov = curve_fit(
                        analyzer.enthalpy_equation,
                        T_data,
                        H_data,
                        p0=initial_guess,
                        bounds=(lower_bounds, upper_bounds),
                        maxfev=max_iterations,
                        method=fit_method,
                        sigma=weights
                    )
                    
                    A1_fit, A2_fit, Tm_fit, DeltaHf_fit, k_fit, H298_fit = fit_params
                    
                    # Calculate parameter uncertainties
                    perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(6)
                    
                    # Generate fitted curve
                    T_fit = np.linspace(T_data.min(), T_data.max(), 1000)
                    H_fit = analyzer.enthalpy_equation(T_fit, *fit_params)
                    
                    # Calculate statistics
                    H_pred = analyzer.enthalpy_equation(T_data, *fit_params)
                    residuals = H_data - H_pred
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((H_data - np.mean(H_data))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    rmse = np.sqrt(np.mean(residuals**2))
                    mae = np.mean(np.abs(residuals))
                    
                    # Calculate AIC and BIC
                    n = len(T_data)
                    k_params = 6
                    aic = n * np.log(ss_res/n) + 2 * k_params
                    bic = n * np.log(ss_res/n) + k_params * np.log(n)
                    
                    # Store fitting results
                    fit_result = {
                        'material_name': material_name,
                        'coefficients': {
                            'A1': A1_fit,
                            'A1_uncertainty': float(perr[0]),
                            'A2': A2_fit,
                            'A2_uncertainty': float(perr[1]),
                            'Tm': Tm_fit,
                            'Tm_uncertainty': float(perr[2]),
                            'DeltaHf': DeltaHf_fit,
                            'DeltaHf_uncertainty': float(perr[3]),
                            'k': k_fit,
                            'k_uncertainty': float(perr[4]),
                            'H298': H298_fit,
                            'H298_uncertainty': float(perr[5])
                        },
                        'statistics': {
                            'r_squared': r_squared,
                            'rmse': rmse,
                            'mae': mae,
                            'aic': aic,
                            'bic': bic,
                            'data_points': n
                        },
                        'composition': composition_ref,
                        'temperature_range': [float(T_data.min()), float(T_data.max())],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    analyzer.fitting_results.append(fit_result)
                    
                    # Display results
                    st.success(f"✅ Fitting completed! R² = {r_squared:.6f}, RMSE = {rmse:.2f} J/mol")
                    
                    # Create comprehensive visualization
                    fig = plt.figure(figsize=(16, 12), dpi=150)
                    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
                    
                    # Main fit plot
                    ax1 = fig.add_subplot(gs[0:2, 0:2])
                    ax1.scatter(T_data, H_data, alpha=0.6, label='Original Data', 
                               color='#1E88E5', s=60, edgecolors='white', linewidth=1)
                    ax1.plot(T_fit, H_fit, 'r-', linewidth=3, label='Fitted Curve', alpha=0.9)
                    ax1.fill_between(T_fit, 
                                    analyzer.enthalpy_equation(T_fit, *(fit_params - perr)),
                                    analyzer.enthalpy_equation(T_fit, *(fit_params + perr)),
                                    color='red', alpha=0.2, label='Uncertainty')
                    
                    ax1.axvline(Tm_fit, color='green', linestyle='--', alpha=0.7, 
                               label=f'Melting Point Tₘ = {Tm_fit:.1f} ± {perr[2]:.1f} K')
                    
                    ax1.set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Enthalpy (J/mol)', fontsize=12, fontweight='bold')
                    ax1.set_title(f'Enthalpy-Temperature Curve Fitting\n{material_name}', 
                                 fontsize=14, fontweight='bold', pad=15)
                    ax1.grid(True, alpha=0.3, linestyle='--')
                    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
                    
                    # Residual plot
                    ax2 = fig.add_subplot(gs[2, 0])
                    ax2.scatter(T_data, residuals, alpha=0.7, color='#FF7043', s=40)
                    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
                    ax2.fill_between([T_data.min(), T_data.max()], 
                                    [-rmse, -rmse], [rmse, rmse], 
                                    color='gray', alpha=0.2, label='±RMSE')
                    ax2.set_xlabel('Temperature (K)', fontsize=10)
                    ax2.set_ylabel('Residuals (J/mol)', fontsize=10)
                    ax2.set_title('Residual Plot', fontsize=11, fontweight='bold')
                    ax2.grid(True, alpha=0.3, linestyle='--')
                    ax2.legend(loc='upper right', fontsize=9)
                    
                    # Residual histogram
                    ax3 = fig.add_subplot(gs[2, 1])
                    ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7, 
                            color='#4CAF50', density=True)
                    ax3.axvline(x=0, color='r', linestyle='--', alpha=0.7)
                    ax3.set_xlabel('Residuals (J/mol)', fontsize=10)
                    ax3.set_ylabel('Density', fontsize=10)
                    ax3.set_title('Residual Distribution', fontsize=11, fontweight='bold')
                    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
                    
                    # Coefficient summary
                    ax4 = fig.add_subplot(gs[0:2, 2])
                    ax4.axis('off')
                    
                    coeff_text = (
                        f"**Fitted Parameters**\n\n"
                        f"A₁ = {A1_fit:.3f} ± {perr[0]:.3f} J/(mol·K)\n"
                        f"A₂ = {A2_fit:.3f} ± {perr[1]:.3f} J/(mol·K)\n"
                        f"Tₘ = {Tm_fit:.1f} ± {perr[2]:.1f} K\n"
                        f"ΔHf = {DeltaHf_fit:,.0f} ± {perr[3]:,.0f} J/mol\n"
                        f"k = {k_fit:.5f} ± {perr[4]:.5f} 1/K\n"
                        f"H₂₉₈ = {H298_fit:,.0f} ± {perr[5]:,.0f} J/mol\n\n"
                        f"**Goodness of Fit**\n\n"
                        f"R² = {r_squared:.6f}\n"
                        f"RMSE = {rmse:.2f} J/mol\n"
                        f"MAE = {mae:.2f} J/mol\n"
                        f"AIC = {aic:.2f}\n"
                        f"BIC = {bic:.2f}\n"
                        f"N = {n} points"
                    )
                    
                    ax4.text(0.05, 0.95, coeff_text, transform=ax4.transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', 
                                     alpha=0.8, pad=10))
                    
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
                            'A1_uncertainty': perr[0],
                            'A2_J_per_mol_K': A2_fit,
                            'A2_uncertainty': perr[1],
                            'Tm_K': Tm_fit,
                            'Tm_uncertainty': perr[2],
                            'DeltaHf_J_per_mol': DeltaHf_fit,
                            'DeltaHf_uncertainty': perr[3],
                            'k_1_per_K': k_fit,
                            'k_uncertainty': perr[4],
                            'H298_J_per_mol': H298_fit,
                            'H298_uncertainty': perr[5],
                            'R_squared': r_squared,
                            'RMSE_J_per_mol': rmse,
                            'MAE_J_per_mol': mae,
                            'AIC': aic,
                            'BIC': bic,
                            'Data_Points': n,
                            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }])
                        
                        st.download_button(
                            "📄 Download Coefficients (CSV)",
                            data=coeff_df.to_csv(index=False),
                            file_name=f"fitting_coeffs_{material_name.replace(' ', '_')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    # Fitted curve data
                    with col_f2:
                        fitted_df = pd.DataFrame({
                            'Temperature_K': T_fit,
                            'Enthalpy_Fitted_J_mol': H_fit,
                            'Enthalpy_Lower_Bound': analyzer.enthalpy_equation(T_fit, *(fit_params - perr)),
                            'Enthalpy_Upper_Bound': analyzer.enthalpy_equation(T_fit, *(fit_params + perr))
                        })
                        
                        st.download_button(
                            "📄 Download Fitted Curve (CSV)",
                            data=fitted_df.to_csv(index=False),
                            file_name=f"fitted_curve_{material_name.replace(' ', '_')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    # Complete results JSON
                    with col_f3:
                        json_data = json.dumps(fit_result, indent=4, default=str)
                        st.download_button(
                            "📄 Download Full Results (JSON)",
                            data=json_data,
                            file_name=f"fitting_results_{material_name.replace(' ', '_')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Equation display
                    st.markdown("---")
                    st.subheader("🧮 Fitted Equation")
                    
                    col_eq1, col_eq2 = st.columns([2, 1])
                    
                    with col_eq1:
                        st.latex(rf"""
                        H(T) = {A1_fit:.3f} \cdot T + {A2_fit:.3f} \cdot \max(T - {Tm_fit:.1f}, 0) + 
                        {DeltaHf_fit:,.0f} \cdot \frac{{1}}{{1 + e^{{-{k_fit:.5f}(T - {Tm_fit:.1f})}}}} + {H298_fit:,.0f}
                        """)
                    
                    with col_eq2:
                        st.markdown("**Equation Components:**")
                        st.markdown("""
                        - **A₁·T**: Solid phase sensible heat
                        - **A₂·max(T-Tₘ,0)**: Liquid phase sensible heat
                        - **ΔHf·sigmoid(T-Tₘ)**: Phase transition enthalpy
                        - **H₂₉₈**: Reference enthalpy at 298K
                        """)
                
                except Exception as e:
                    st.error(f"❌ Fitting error: {str(e)}")
                    st.exception(e)
    
    # ==================== TAB 4: Multi-Material Comparison ====================
    with tab4:
        st.header("🔄 Multi-Material Comparison Dashboard")
        
        if not analyzer.results_history:
            st.info("💡 No computed data available for comparison. Please perform calculations in the 'Enthalpy Computation' tab first.")
            st.stop()
        
        # Selection interface
        st.subheader("✅ Select Materials for Comparison")
        
        # Create selection cards
        result_cards = []
        for i, res in enumerate(analyzer.results_history):
            card_html = f"""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0; 
                        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);">
                <h4 style="margin: 0; color: #1E88E5;">{res['name']}</h4>
                <p style="margin: 5px 0; font-size: 0.9em; color: #666;">
                    📁 {res['tdb_file']}<br>
                    🌡️ {res['temperature_range'][0]}-{res['temperature_range'][1]} K<br>
                    📊 {len(res['data'])} data points
                </p>
            </div>
            """
            result_cards.append((i, card_html))
        
        # Display cards in columns
        num_cols = 3
        cols = st.columns(num_cols)
        
        selected_indices = []
        for idx, card_html in enumerate(result_cards):
            with cols[idx % num_cols]:
                st.markdown(card_html[1], unsafe_allow_html=True)
                selected = st.checkbox(f"Select {analyzer.results_history[card_html[0]]['name']}", 
                                      key=f"select_{card_html[0]}")
                if selected:
                    selected_indices.append(card_html[0])
        
        if not selected_indices:
            st.warning("⚠️ Please select at least one material for comparison")
            st.stop()
        
        # Limit to 6 materials for clarity
        if len(selected_indices) > 6:
            st.warning(f"⚠️ Too many materials selected ({len(selected_indices)}). Limiting to first 6.")
            selected_indices = selected_indices[:6]
        
        # Create comparison visualization
        with st.spinner("🔄 Generating comparison visualization..."):
            fig = create_comparison_visualization(analyzer, selected_indices)
            if fig:
                st.pyplot(fig)
        
        # Detailed comparison table
        st.markdown("---")
        st.subheader("📊 Detailed Comparison Table")
        
        comparison_data = []
        for idx in selected_indices:
            res = analyzer.results_history[idx]
            data = res['data']
            
            # Calculate comprehensive metrics
            min_h = data['Enthalpy_J_mol'].min()
            max_h = data['Enthalpy_J_mol'].max()
            delta_h = max_h - min_h
            
            # Calculate Cp from derivative
            dH = np.diff(data['Enthalpy_J_mol'].values)
            dT = np.diff(data['Temperature_K'].values)
            cp_values = dH / dT if len(dH) > 0 else [0]
            
            # Find melting temperature (max slope)
            if len(cp_values) > 0:
                max_cp_idx = np.argmax(cp_values)
                Tm_est = data['Temperature_K'].iloc[max_cp_idx]
                max_cp = cp_values[max_cp_idx]
            else:
                Tm_est = 0
                max_cp = 0
            
            avg_cp = np.mean(cp_values) if len(cp_values) > 0 else 0
            
            comparison_data.append({
                'Material': res['name'],
                'TDB File': res['tdb_file'],
                'Composition': ', '.join([f"{k}{v:.2f}" for k, v in list(res['composition'].items())[:2]]),
                'Temp Range (K)': f"{res['temperature_range'][0]}-{res['temperature_range'][1]}",
                'Data Points': len(data),
                'Min H (J/mol)': f"{min_h:,.0f}",
                'Max H (J/mol)': f"{max_h:,.0f}",
                'ΔH (J/mol)': f"{delta_h:,.0f}",
                'Tₘ (K)': f"{Tm_est:.0f}",
                'Max Cp (J/mol·K)': f"{max_cp:.1f}",
                'Avg Cp (J/mol·K)': f"{avg_cp:.1f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Statistical analysis
        st.markdown("---")
        st.subheader("📈 Statistical Analysis")
        
        if len(selected_indices) >= 2:
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                # Calculate correlation matrix
                enthalpy_data = []
                common_T = analyzer.results_history[selected_indices[0]]['data']['Temperature_K'].values
                
                for idx in selected_indices:
                    data = analyzer.results_history[idx]['data']
                    # Interpolate to common temperature grid
                    H_interp = np.interp(common_T, data['Temperature_K'].values, data['Enthalpy_J_mol'].values)
                    enthalpy_data.append(H_interp)
                
                correlation_matrix = np.corrcoef(enthalpy_data)
                
                # Display correlation heatmap
                fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                im = ax_corr.imshow(correlation_matrix, cmap='RdYlBu', vmin=-1, vmax=1)
                
                # Add text annotations
                for i in range(len(selected_indices)):
                    for j in range(len(selected_indices)):
                        text = ax_corr.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                          ha="center", va="center", color="black")
                
                material_names_short = [analyzer.results_history[idx]['name'] for idx in selected_indices]
                ax_corr.set_xticks(range(len(material_names_short)))
                ax_corr.set_yticks(range(len(material_names_short)))
                ax_corr.set_xticklabels(material_names_short, rotation=45, ha='right')
                ax_corr.set_yticklabels(material_names_short)
                ax_corr.set_title('Enthalpy Correlation Matrix', fontsize=14, fontweight='bold')
                plt.colorbar(im, ax=ax_corr)
                plt.tight_layout()
                st.pyplot(fig_corr)
            
            with col_stat2:
                # Calculate relative differences
                st.markdown("**Relative Differences (%)**")
                
                base_idx = selected_indices[0]
                base_name = analyzer.results_history[base_idx]['name']
                base_delta_h = comparison_df.loc[0, 'ΔH (J/mol)'].replace(',', '')
                base_delta_h = float(base_delta_h) if base_delta_h != '—' else 0
                
                diff_data = []
                for i, idx in enumerate(selected_indices[1:], 1):
                    mat_name = analyzer.results_history[idx]['name']
                    delta_h_str = comparison_df.loc[i, 'ΔH (J/mol)'].replace(',', '')
                    delta_h = float(delta_h_str) if delta_h_str != '—' else 0
                    
                    if base_delta_h != 0:
                        diff_percent = ((delta_h - base_delta_h) / base_delta_h) * 100
                        diff_data.append({
                            'Material': mat_name,
                            'ΔH Difference (%)': f"{diff_percent:.1f}",
                            'Compared to': base_name
                        })
                
                if diff_data:
                    diff_df = pd.DataFrame(diff_data)
                    st.dataframe(diff_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Insufficient data for difference calculation")
        
        # Download options
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
        st.subheader("📥 Download Comparison Data")
        
        col_c1, col_c2, col_c3 = st.columns(3)
        
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
                mime="text/csv",
                use_container_width=True
            )
        
        with col_c2:
            # Comparison summary
            st.download_button(
                "📄 Summary Table (CSV)",
                data=comparison_df.to_csv(index=False),
                file_name=f"comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_c3:
            # Export all as Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                combined_df.to_excel(writer, sheet_name='Combined Data', index=False)
                comparison_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Add individual sheets
                for idx in selected_indices:
                    res = analyzer.results_history[idx]
                    res['data'].to_excel(writer, sheet_name=res['name'][:30], index=False)
            
            st.download_button(
                "📗 Excel Workbook",
                data=output.getvalue(),
                file_name=f"comparison_workbook_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== TAB 5: Settings & Help ====================
    with tab5:
        st.header("⚙️ Settings & Help Center")
        
        col_h1, col_h2 = st.columns([1, 1.2])
        
        with col_h1:
            st.subheader("📖 User Guide")
            
            with st.expander("🎯 Quick Start", expanded=True):
                st.markdown("""
                **1. First-Time Setup:**
                - The app automatically creates example TDB files in the `databases/` directory
                - You can upload your own TDB files or use the provided examples
                
                **2. Basic Workflow:**
                1. **Tab 1 (Dashboard):** Overview and quick actions
                2. **Tab 2 (Enthalpy Computation):** 
                   - Select/upload TDB file
                   - Choose elements and set composition (sum to 1.0)
                   - Define temperature range
                   - Compute and visualize enthalpy
                3. **Tab 3 (Curve Fitting):**
                   - Fit mathematical models to enthalpy data
                   - Analyze phase transitions
                   - Export fitting parameters
                4. **Tab 4 (Multi-Material):**
                   - Compare multiple materials
                   - Statistical analysis
                   - Generate comprehensive reports
                
                **3. Composition Rules:**
                - For n-component system, provide n-1 mole fractions
                - The nth component is automatically calculated to ensure sum = 1.0
                - Both mole fractions and weight fractions are supported
                """)
            
            with st.expander("🔬 Advanced Features", expanded=False):
                st.markdown("""
                **Phase Transition Detection:**
                - Automatic detection of melting/solidification points
                - Enthalpy change calculation at transitions
                - Visual markers on plots
                
                **Curve Fitting:**
                - Advanced sigmoid-based enthalpy equation
                - Parameter uncertainty estimation
                - Multiple goodness-of-fit metrics (R², RMSE, AIC, BIC)
                - Confidence intervals for predictions
                
                **Data Export:**
                - Multiple formats: CSV, DAT, JSON, Excel
                - Publication-quality plots
                - Complete metadata preservation
                - Batch export capabilities
                """)
            
            with st.expander("⚠️ Troubleshooting", expanded=False):
                st.markdown("""
                **Common Issues:**
                
                1. **"No TDB file found"**
                   - Check `databases/` directory exists
                   - Upload a TDB file or use provided examples
                   - Ensure file has .tdb or .TDB extension
                
                2. **Calculation errors**
                   - Verify element selection matches TDB file
                   - Check composition sums to 1.0
                   - Ensure temperature range is appropriate
                   - Reduce number of phases if calculation is slow
                
                3. **Memory issues**
                   - Reduce temperature range or step size
                   - Limit number of phases in equilibrium
                   - Clear session data periodically
                
                4. **Visualization problems**
                   - Update matplotlib to latest version
                   - Check data contains no NaN values
                   - Ensure sufficient data points for smooth plots
                """)
        
        with col_h2:
            st.subheader("⚙️ Application Management")
            
            # Session management
            st.markdown("#### 🗑️ Session Management")
            
            col_sess1, col_sess2 = st.columns(2)
            
            with col_sess1:
                if st.button("Clear All Data", type="secondary", use_container_width=True):
                    analyzer.results_history = []
                    analyzer.fitting_results = []
                    st.success("✅ All session data cleared!")
                    st.rerun()
            
            with col_sess2:
                if st.button("Reset to Defaults", type="secondary", use_container_width=True):
                    # Clear session state
                    for key in list(st.session_state.keys()):
                        if key != 'analyzer':
                            del st.session_state[key]
                    st.success("✅ Application reset to defaults!")
                    st.rerun()
            
            # Database management
            st.markdown("#### 📁 Database Management")
            
            tdb_files = analyzer.get_available_tdb_files()
            
            if tdb_files:
                st.write(f"**Found {len(tdb_files)} TDB files:**")
                
                for tdb in tdb_files:
                    col_f1, col_f2, col_f3, col_f4 = st.columns([3, 1, 1, 1])
                    with col_f1:
                        st.caption(f"`{tdb}`")
                    with col_f2:
                        file_path = analyzer.database_dir / tdb
                        file_size = os.path.getsize(file_path) / 1024
                        st.caption(f"{file_size:.1f} KB")
                    with col_f3:
                        if st.button("📋", key=f"copy_{tdb}", help="Copy path"):
                            st.code(str(file_path))
                    with col_f4:
                        if st.button("🗑️", key=f"delete_{tdb}", help="Delete file"):
                            try:
                                os.remove(file_path)
                                st.success(f"Deleted {tdb}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            else:
                st.info("No TDB files in database directory.")
            
            # TDB upload section
            st.markdown("#### 📤 Upload TDB Files")
            
            uploaded_new = st.file_uploader(
                "Upload new TDB file",
                type=["tdb", "TDB"],
                key="uploader_settings",
                help="Upload thermodynamic database files"
            )
            
            if uploaded_new is not None:
                saved_path = analyzer.save_uploaded_tdb(uploaded_new)
                if saved_path:
                    st.rerun()
            
            # Custom element management
            st.markdown("#### ⚛️ Custom Element Database")
            
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                new_element = st.text_input("Element symbol", key="new_elem_sym")
            with col_e2:
                new_weight = st.number_input("Molar weight (g/mol)", 1.0, 500.0, 50.0, 0.1)
            with col_e3:
                if st.button("➕ Add Element", use_container_width=True):
                    if new_element.strip():
                        elem_key = new_element.strip().upper()
                        MOLAR_WEIGHTS[elem_key] = new_weight
                        st.success(f"✅ Added {elem_key}: {new_weight} g/mol")
                    else:
                        st.warning("⚠️ Please enter an element symbol")
            
            # Display custom elements
            custom_elements = [k for k in MOLAR_WEIGHTS.keys() if k not in [
                'AG', 'AL', 'AU', 'BI', 'CU', 'IN', 'NI', 'PB', 'SN', 'TI', 
                'V', 'FE', 'CR', 'MO', 'W', 'MN', 'SI', 'C', 'N', 'O', 'H'
            ]]
            
            if custom_elements:
                st.markdown("**Custom Elements:**")
                for elem in sorted(custom_elements):
                    st.write(f"- {elem}: {MOLAR_WEIGHTS[elem]} g/mol")
            
            # About section
            st.markdown("---")
            st.subheader("ℹ️ About")
            
            st.markdown("""
            **Thermodynamic Enthalpy Analyzer Pro**
            
            *Version 2.0 | Advanced Edition*
            
            A comprehensive tool for thermodynamic calculations using the CALPHAD method.
            
            **Core Features:**
            - Multi-component system support
            - Automatic phase transition detection
            - Advanced curve fitting with uncertainty
            - Multi-material comparison
            - Publication-quality visualizations
            
            **Technical Stack:**
            - **CALPHAD Engine:** pycalphad
            - **Numerical Methods:** scipy, numpy
            - **Visualization:** matplotlib, plotly
            - **Interface:** Streamlit
            - **Data Handling:** pandas, xarray
            
            **License:** MIT Open Source
            **Developed by:** Thermodynamic Analysis Research Group
            
            © 2026 Thermodynamic Analysis Toolkit. All rights reserved.
            """)
    
    # Footer
    st.markdown("---")
    
    footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])
    
    with footer_col1:
        st.caption(f"🔥 Thermodynamic Enthalpy Analyzer Pro v2.0 | Session: {datetime.now().strftime('%Y%m%d-%H%M%S')}")
    
    with footer_col2:
        st.caption(f"📊 Materials: {len(analyzer.results_history)} | Fits: {len(analyzer.fitting_results)}")
    
    with footer_col3:
        if st.button("🔄 Refresh Session", type="secondary"):
            st.rerun()

if __name__ == "__main__":
    main()
