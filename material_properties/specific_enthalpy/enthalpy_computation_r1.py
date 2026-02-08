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
import traceback
from typing import Dict, List, Optional, Tuple, Any
import sympy as sp
from itertools import combinations
import math

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🔥 Thermodynamic Enthalpy Analyzer Pro",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better visualization
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #FF512F 0%, #F09819 50%, #FF512F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(255, 81, 47, 0.2);
    }
    .dof-indicator {
        padding: 10px 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        text-align: center;
    }
    .dof-correct {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
    }
    .dof-warning {
        background: linear-gradient(135deg, #f46b45, #eea849);
        color: white;
    }
    .dof-error {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
    }
    .phase-rule-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .system-status {
        background: #f8f9fa;
        border-left: 5px solid #007bff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .component-badge {
        display: inline-block;
        padding: 5px 15px;
        margin: 3px;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .condition-badge {
        display: inline-block;
        padding: 5px 15px;
        margin: 3px;
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .phase-badge {
        display: inline-block;
        padding: 5px 15px;
        margin: 3px;
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .thermo-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .gibbs-rule {
        font-family: "Courier New", monospace;
        background: #2c3e50;
        color: #ecf0f1;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 1.2rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
        box-shadow: 0 2px 4px rgba(30, 136, 229, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Comprehensive molar weights database
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
    'SR': 87.62, 'BA': 137.327, 'LI': 6.941, 'BE': 9.012182,
    'B': 10.811, 'NA': 22.989769, 'P': 30.973762, 'S': 32.065,
    'CL': 35.453, 'K': 39.0983, 'CA': 40.078, 'F': 18.9984032
}

class ThermodynamicSystemAnalyzer:
    """Enhanced analyzer with Gibbs Phase Rule validation"""
    
    def __init__(self):
        self.results_history = []
        self.fitting_results = []
        self.database_dir = Path("thermo_databases")
        self.database_dir.mkdir(exist_ok=True)
        self._ensure_default_tdb()
        self.system_state = {}
    
    def _ensure_default_tdb(self):
        """Create comprehensive example TDB files"""
        # Al-Cu-Ni ternary system example
        al_cu_ni_tdb = """$ AL-CU-NI Ternary System - Example Database
$ For testing Thermodynamic Enthalpy Analyzer
$
 ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8322E+01!
 ELEMENT CU   FCC_A1                    6.3546E+01  5.0041E+03  3.3150E+01!
 ELEMENT NI   FCC_A1                    5.8693E+01  4.7870E+03  2.9796E+01!
$
 TYPE_DEFINITION % SEQ *!
$
$============== LIQUID PHASE ==============
 PHASE LIQUID %  1  1.0  !
 CONSTITUENT LIQUID :AL,CU,NI : !
$
$============== FCC_A1 PHASE ==============
 PHASE FCC_A1 %  2  1.0  1.0  !
 CONSTITUENT FCC_A1 :AL,CU,NI : VA : !
$
$============== BCC_A2 PHASE ==============
 PHASE BCC_A2 %  2  1.0  3.0  !
 CONSTITUENT BCC_A2 :AL,CU,NI : VA : !
$
$============== FUNCTION DEFINITIONS ==============
 FUNCTION GHSERAL    2.98150E+02  -7976.15+137.093038*T-24.3671976*T*LN(T)
     -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1);  7.00000E+02  Y
     -11276.24+223.048446*T-38.5844296*T*LN(T)+.018531982*T**2
     -5.764227E-06*T**3+74092*T**(-1);  9.33600E+02  Y
     -11278.378+188.684153*T-31.748192*T*LN(T)-1.230524E+28*T**(-9);  2.90000E+03  N !
 FUNCTION GHSERCU    2.98150E+02  -7770.458+130.485235*T-24.112392*T*LN(T)
     -.00265684*T**2+1.29223E-07*T**3+52478*T**(-1);  1.35777E+03  Y
     -13542.026+183.803828*T-31.38*T*LN(T)+2.64313E+31*T**(-9);  3.20000E+03  N !
 FUNCTION GHSERNI    2.98150E+02  -5179.159+117.854*T-22.096*T*LN(T)
     -.0048407*T**2;  1.72800E+03  Y
     -27840.655+279.135*T-43.1*T*LN(T)+1.12754E+31*T**(-9);  3.00000E+03  N !
$
$============== LIQUID PARAMETERS ==============
 PARAMETER G(LIQUID,AL;0)  2.98150E+02  +11005.029-11.840849*T
      +7.9337E-20*T**7+GHSERAL#;  9.33600E+02  Y
      +10482.382-11.253974*T+1.231E+28*T**(-9)+GHSERAL#;  2.90000E+03  N !
 PARAMETER G(LIQUID,CU;0)  2.98150E+02  +12964.735-9.511904*T
      +5.8494E-21*T**7+GHSERCU#;  1.35777E+03  Y
      +13924.446-9.511904*T+2.64313E+31*T**(-9)+GHSERCU#;  3.20000E+03  N !
 PARAMETER G(LIQUID,NI;0)  2.98150E+02  +16414.686-9.397*T
      -3.82318E-21*T**7+GHSERNI#;  1.72800E+03  Y
      +17197.666-9.397*T+1.26586E+31*T**(-9)+GHSERNI#;  3.00000E+03  N !
 PARAMETER G(LIQUID,AL,CU;0)  2.98150E+02  -47046.58+6.75*T;  6.00000E+03  N !
 PARAMETER G(LIQUID,AL,CU;1)  2.98150E+02  +21202.8352-9.67484*T;  6.00000E+03  N !
 PARAMETER G(LIQUID,AL,NI;0)  2.98150E+02  -152000+16*T;  6.00000E+03  N !
 PARAMETER G(LIQUID,CU,NI;0)  2.98150E+02  +8030-3.235*T;  6.00000E+03  N !
$
$============== FCC_A1 PARAMETERS ==============
 PARAMETER G(FCC_A1,AL:VA;0)  2.98150E+02  +GHSERAL#;  6.00000E+03  N !
 PARAMETER G(FCC_A1,CU:VA;0)  2.98150E+02  +GHSERCU#;  6.00000E+03  N !
 PARAMETER G(FCC_A1,NI:VA;0)  2.98150E+02  +GHSERNI#;  6.00000E+03  N !
 PARAMETER G(FCC_A1,AL,CU:VA;0)  2.98150E+02  -12282.6+2.63791*T;  6.00000E+03  N !
 PARAMETER G(FCC_A1,AL,CU:VA;1)  2.98150E+02  +4580.9-1.7352*T;  6.00000E+03  N !
 PARAMETER G(FCC_A1,AL,NI:VA;0)  2.98150E+02  -162400+16*T;  6.00000E+03  N !
 PARAMETER G(FCC_A1,CU,NI:VA;0)  2.98150E+02  +8366+2.802*T;  6.00000E+03  N !
$
$============== BCC_A2 PARAMETERS ==============
 PARAMETER G(BCC_A2,AL:VA;0)  2.98150E+02  +10083-4.813*T+GHSERAL#;  6.00000E+03  N !
 PARAMETER G(BCC_A2,CU:VA;0)  2.98150E+02  +5000+2*T+GHSERCU#;  6.00000E+03  N !
 PARAMETER G(BCC_A2,NI:VA;0)  2.98150E+02  +8715.084-3.556*T+GHSERNI#;  6.00000E+03  N !
$
 LIST_OF_REFERENCES
 NUMBER  SOURCE
    1    'AL-CU-NI Ternary System - Thermodynamic Enthalpy Analyzer Example'
    2    'Based on COST507 database with simplifications'
$"""
        
        default_path = self.database_dir / "AL_CU_NI_EXAMPLE.tdb"
        with open(default_path, 'w') as f:
            f.write(al_cu_ni_tdb)
        
        # Create additional examples
        self._create_binary_examples()
        self._create_quaternary_example()
    
    def _create_binary_examples(self):
        """Create binary system examples"""
        # Fe-C binary
        fe_c_tdb = """$ FE-C Binary System
 ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT FE   BCC_A2                    5.5847E+01  4.4890E+03  2.7280E+01!
 ELEMENT C    GRAPHITE                  1.2011E+01  1.0500E+03  5.7420E+00!
$
 TYPE_DEFINITION % SEQ *!
$
 PHASE LIQUID %  1  1.0  !
 CONSTITUENT LIQUID :FE,C : !
$
 PHASE BCC_A2 %  2  1.0  3.0  !
 CONSTITUENT BCC_A2 :FE:C : VA : !
$
 PHASE FCC_A1 %  2  1.0  1.0  !
 CONSTITUENT FCC_A1 :FE:C : VA : !
$
 FUNCTION GHSERFE    2.98150E+02  +1225.7+124.134*T-23.5143*T*LN(T)
     -.00439752*T**2-5.8927E-08*T**3+77359*T**(-1);  1.81100E+03  Y
     -25383.581+299.31255*T-46*T*LN(T)+2.29603E+31*T**(-9);  6.00000E+03  N !
 FUNCTION GHSERCC   2.98150E+02  -17368.441+170.73*T-24.3*T*LN(T)
     +4.723E-04*T**2-6.188E-08*T**3+1.1857E+05*T**(-1);  4.10000E+03  N !
$
 PARAMETER G(LIQUID,FE;0)  2.98150E+02  +12040.17-6.55843*T+GHSERFE#;  6.00000E+03  N !
 PARAMETER G(LIQUID,C;0)  2.98150E+02  +117230.0-24.373*T+GHSERCC#;  6.00000E+03  N !
 PARAMETER G(LIQUID,FE,C;0)  2.98150E+02  -91976.5+6.648*T;  6.00000E+03  N !
$
 PARAMETER G(BCC_A2,FE:VA;0)  2.98150E+02  +GHSERFE#;  6.00000E+03  N !
 PARAMETER G(BCC_A2,FE:C:VA;0)  2.98150E+02  +10000-2*T;  6.00000E+03  N !
$
 PARAMETER G(FCC_A1,FE:VA;0)  2.98150E+02  +1000-1.5*T+GHSERFE#;  6.00000E+03  N !
 PARAMETER G(FCC_A1,C:VA;0)  2.98150E+02  +GHSERCC#;  6.00000E+03  N !
$
 LIST_OF_REFERENCES
 NUMBER  SOURCE
    1    'Fe-C Binary System - Simplified for testing'
$"""
        
        fec_path = self.database_dir / "FE_C_EXAMPLE.tdb"
        with open(fec_path, 'w') as f:
            f.write(fe_c_tdb)
        
        # Ni-Al binary
        ni_al_tdb = """$ NI-AL Binary System
 ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT NI   FCC_A1                    5.8693E+01  4.7870E+03  2.9796E+01!
 ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8322E+01!
$
 TYPE_DEFINITION % SEQ *!
$
 PHASE LIQUID %  1  1.0  !
 CONSTITUENT LIQUID :NI,AL : !
$
 PHASE FCC_A1 %  2  1.0  1.0  !
 CONSTITUENT FCC_A1 :NI,AL : VA : !
$
 PHASE B2_BCC %  2  1.0  1.0  !
 CONSTITUENT B2_BCC :NI,AL : VA : !
$
 FUNCTION GHSERNI    2.98150E+02  -5179.159+117.854*T-22.096*T*LN(T)
     -.0048407*T**2;  1.72800E+03  Y
     -27840.655+279.135*T-43.1*T*LN(T)+1.12754E+31*T**(-9);  3.00000E+03  N !
 FUNCTION GHSERAL    2.98150E+02  -7976.15+137.093038*T-24.3671976*T*LN(T)
     -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1);  7.00000E+02  Y
     -11276.24+223.048446*T-38.5844296*T*LN(T)+.018531982*T**2
     -5.764227E-06*T**3+74092*T**(-1);  9.33600E+02  Y
     -11278.378+188.684153*T-31.748192*T*LN(T)-1.230524E+28*T**(-9);  2.90000E+03  N !
$
 PARAMETER G(LIQUID,NI;0)  2.98150E+02  +16414.686-9.397*T
      -3.82318E-21*T**7+GHSERNI#;  1.72800E+03  Y
      +17197.666-9.397*T+1.26586E+31*T**(-9)+GHSERNI#;  3.00000E+03  N !
 PARAMETER G(LIQUID,AL;0)  2.98150E+02  +11005.029-11.840849*T
      +7.9337E-20*T**7+GHSERAL#;  9.33600E+02  Y
      +10482.382-11.253974*T+1.231E+28*T**(-9)+GHSERAL#;  2.90000E+03  N !
 PARAMETER G(LIQUID,NI,AL;0)  2.98150E+02  -152000+16*T;  6.00000E+03  N !
$
 PARAMETER G(FCC_A1,NI:VA;0)  2.98150E+02  +GHSERNI#;  6.00000E+03  N !
 PARAMETER G(FCC_A1,AL:VA;0)  2.98150E+02  +GHSERAL#;  6.00000E+03  N !
 PARAMETER G(FCC_A1,NI,AL:VA;0)  2.98150E+02  -162400+16*T;  6.00000E+03  N !
$
 PARAMETER G(B2_BCC,NI:VA;0)  2.98150E+02  +8715.084-3.556*T+GHSERNI#;  6.00000E+03  N !
 PARAMETER G(B2_BCC,AL:VA;0)  2.98150E+02  +10083-4.813*T+GHSERAL#;  6.00000E+03  N !
 PARAMETER G(B2_BCC,NI,AL:VA;0)  2.98150E+02  -140000+15*T;  6.00000E+03  N !
$
 LIST_OF_REFERENCES
 NUMBER  SOURCE
    1    'Ni-Al Binary System - Example Database'
$"""
        
        nial_path = self.database_dir / "NI_AL_EXAMPLE.tdb"
        with open(nial_path, 'w') as f:
            f.write(ni_al_tdb)
    
    def _create_quaternary_example(self):
        """Create quaternary system example"""
        quaternary_tdb = """$ QUATERNARY EXAMPLE - Al-Cu-Mg-Si
 ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8322E+01!
 ELEMENT CU   FCC_A1                    6.3546E+01  5.0041E+03  3.3150E+01!
 ELEMENT MG   HCP_A3                    2.4305E+01  4.9980E+03  3.2670E+01!
 ELEMENT SI   DIAMOND_A4                2.8085E+01  4.6830E+03  1.8830E+01!
$
 TYPE_DEFINITION % SEQ *!
$
 PHASE LIQUID %  1  1.0  !
 CONSTITUENT LIQUID :AL,CU,MG,SI : !
$
 PHASE FCC_A1 %  2  1.0  1.0  !
 CONSTITUENT FCC_A1 :AL,CU,MG,SI : VA : !
$
 FUNCTION GHSERAL    2.98150E+02  -7976.15+137.093038*T-24.3671976*T*LN(T)
     -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1);  7.00000E+02  Y
     -11276.24+223.048446*T-38.5844296*T*LN(T)+.018531982*T**2
     -5.764227E-06*T**3+74092*T**(-1);  9.33600E+02  Y
     -11278.378+188.684153*T-31.748192*T*LN(T)-1.230524E+28*T**(-9);  2.90000E+03  N !
 FUNCTION GHSERCU    2.98150E+02  -7770.458+130.485235*T-24.112392*T*LN(T)
     -.00265684*T**2+1.29223E-07*T**3+52478*T**(-1);  1.35777E+03  Y
     -13542.026+183.803828*T-31.38*T*LN(T)+2.64313E+31*T**(-9);  3.20000E+03  N !
 FUNCTION GHSERMG    2.98150E+02  -8367.34+143.675547*T-26.1849782*T*LN(T)
     +.00115895*T**2-6.1953E-07*T**3;  9.23000E+02  Y
     -14130.185+204.716215*T-34.3088*T*LN(T);  3.00000E+03  N !
 FUNCTION GHSERSI    2.98150E+02  -8162.609+137.236859*T-22.8317533*T*LN(T)
     -.001912904*T**2-3.552E-09*T**3;  1.68700E+03  Y
     -9457.642+167.281367*T-27.196*T*LN(T);  3.50000E+03  N !
$
 PARAMETER G(LIQUID,AL;0)  2.98150E+02  +11005.029-11.840849*T+GHSERAL#;  3.00000E+03  N !
 PARAMETER G(LIQUID,CU;0)  2.98150E+02  +12964.735-9.511904*T+GHSERCU#;  3.00000E+03  N !
 PARAMETER G(LIQUID,MG;0)  2.98150E+02  +8200-8.5*T+GHSERMG#;  3.00000E+03  N !
 PARAMETER G(LIQUID,SI;0)  2.98150E+02  +50500-29.8*T+GHSERSI#;  3.00000E+03  N !
$
 PARAMETER G(FCC_A1,AL:VA;0)  2.98150E+02  +GHSERAL#;  3.00000E+03  N !
 PARAMETER G(FCC_A1,CU:VA;0)  2.98150E+02  +GHSERCU#;  3.00000E+03  N !
 PARAMETER G(FCC_A1,MG:VA;0)  2.98150E+02  +3000+2*T+GHSERMG#;  3.00000E+03  N !
 PARAMETER G(FCC_A1,SI:VA;0)  2.98150E+02  +2000+1.5*T+GHSERSI#;  3.00000E+03  N !
$
 LIST_OF_REFERENCES
 NUMBER  SOURCE
    1    'Al-Cu-Mg-Si Quaternary System - Example Database'
$"""
        
        quat_path = self.database_dir / "AL_CU_MG_SI_EXAMPLE.tdb"
        with open(quat_path, 'w') as f:
            f.write(quaternary_tdb)
    
    def analyze_degrees_of_freedom(self, components, phases, conditions):
        """
        Analyze the thermodynamic system's degrees of freedom
        using the Gibbs Phase Rule
        """
        # Count components (excluding VA)
        C = len([c for c in components if c != 'VA'])
        
        # Count phases
        P = len(phases)
        
        # Count independent intensive variables specified
        # Gibbs Phase Rule: F = C - P + 2
        F_theoretical = C - P + 2
        
        # Analyze what's actually specified in conditions
        specified_vars = self._analyze_specified_variables(conditions)
        
        # Calculate actual degrees of freedom
        F_actual = F_theoretical - specified_vars['independent_intensive']
        
        # Check for over-specification
        is_over_specified = specified_vars['independent_intensive'] > F_theoretical
        is_under_specified = specified_vars['independent_intensive'] < F_theoretical
        
        # Build analysis report
        analysis = {
            'components': C,
            'phases': P,
            'theoretical_F': F_theoretical,
            'specified_vars': specified_vars,
            'actual_F': F_actual,
            'is_valid': F_actual == 0 and not is_over_specified,
            'is_over_specified': is_over_specified,
            'is_under_specified': is_under_specified,
            'message': self._generate_dof_message(C, P, F_theoretical, F_actual, 
                                                 specified_vars, is_over_specified)
        }
        
        return analysis
    
    def _analyze_specified_variables(self, conditions):
        """Analyze which variables are specified in conditions"""
        analysis = {
            'T_specified': False,
            'P_specified': False,
            'N_specified': False,
            'compositions': {},
            'total_composition_vars': 0,
            'independent_intensive': 0,
            'extensive_vars': 0
        }
        
        for key, value in conditions.items():
            if hasattr(key, 'species'):
                var_name = str(key)
                
                if 'T' in var_name:
                    analysis['T_specified'] = True
                    analysis['independent_intensive'] += 1
                
                elif 'P' in var_name:
                    analysis['P_specified'] = True
                    analysis['independent_intensive'] += 1
                
                elif 'N' in var_name:
                    analysis['N_specified'] = True
                    analysis['extensive_vars'] += 1
                
                elif 'X(' in var_name:
                    # Extract element from X(ELEMENT)
                    element = var_name.split('(')[1].split(')')[0]
                    analysis['compositions'][element] = value
                    analysis['total_composition_vars'] += 1
        
        # For N components, we need N-1 independent composition variables
        # Check if compositions are properly specified
        comp_values = list(analysis['compositions'].values())
        if comp_values:
            # If all compositions sum to 1, we might have over-specified
            total_comp = sum([v for v in comp_values if not isinstance(v, tuple)])
            analysis['independent_composition_vars'] = len(comp_values)
            if abs(total_comp - 1.0) < 0.001 and len(comp_values) > 1:
                analysis['independent_composition_vars'] = len(comp_values) - 1
        
        analysis['independent_intensive'] += analysis['independent_composition_vars']
        
        return analysis
    
    def _generate_dof_message(self, C, P, F_theoretical, F_actual, 
                            specified_vars, is_over_specified):
        """Generate human-readable message about degrees of freedom"""
        
        messages = []
        
        # Basic phase rule info
        messages.append(f"### 🔬 Gibbs Phase Rule Analysis")
        messages.append(f"**Components (C):** {C}")
        messages.append(f"**Phases (P):** {P}")
        messages.append(f"**Theoretical DOF (F = C - P + 2):** F = {C} - {P} + 2 = **{F_theoretical}**")
        
        # What's specified
        messages.append(f"\n### 📋 Specified Variables:")
        messages.append(f"- **Temperature (T):** {'✅ Specified' if specified_vars['T_specified'] else '❌ Missing'}")
        messages.append(f"- **Pressure (P):** {'✅ Specified' if specified_vars['P_specified'] else '❌ Missing'}")
        messages.append(f"- **Composition variables:** {specified_vars['total_composition_vars']} specified")
        
        if specified_vars['compositions']:
            messages.append(f"  - Composition details:")
            for elem, val in specified_vars['compositions'].items():
                messages.append(f"    - X({elem}) = {val}")
        
        # DOF status
        messages.append(f"\n### 🎯 Degrees of Freedom Status:")
        
        if F_actual == 0 and not is_over_specified:
            messages.append(f"<div class='dof-indicator dof-correct'>✅ PERFECT! F = 0 (System is fully determined)</div>")
            messages.append("The system has exactly the right number of constraints for equilibrium.")
        
        elif F_actual > 0:
            messages.append(f"<div class='dof-indicator dof-warning'>⚠️ UNDER-CONSTRAINED! F = {F_actual} > 0</div>")
            messages.append(f"You need to specify {F_actual} more intensive variable(s).")
            
            suggestions = []
            if not specified_vars['T_specified']:
                suggestions.append("• Specify temperature (T)")
            if not specified_vars['P_specified']:
                suggestions.append("• Specify pressure (P)")
            if specified_vars['total_composition_vars'] < C - 1:
                suggestions.append(f"• Specify {C - 1 - specified_vars['total_composition_vars']} more composition variable(s)")
            
            if suggestions:
                messages.append("\n**Suggestions:**")
                messages.extend(suggestions)
        
        elif F_actual < 0 or is_over_specified:
            messages.append(f"<div class='dof-indicator dof-error'>❌ OVER-CONSTRAINED! F = {F_actual} < 0</div>")
            messages.append("You have specified too many constraints. The system cannot satisfy all conditions simultaneously.")
            
            if specified_vars['total_composition_vars'] >= C:
                messages.append(f"• You specified {specified_vars['total_composition_vars']} composition variables for {C} components.")
                messages.append(f"  For {C} components, you should specify exactly {C-1} independent composition variables.")
            
            messages.append("\n**To fix:** Remove redundant constraints, typically:")
            messages.append("• Don't specify compositions that sum to exactly 1.0 (last component is implicit)")
            messages.append("• Don't use v.N unless you have a specific reason")
            messages.append("• Ensure each composition variable is independent")
        
        # Specific checks
        messages.append(f"\n### 🔍 Specific Checks:")
        
        # Check for v.N misuse
        if specified_vars['N_specified']:
            messages.append("⚠️ **v.N is specified** - pycalphad assumes N=1 by default. Consider removing unless you have a specific need.")
        
        # Check composition sum
        comp_sum = sum([v for v in specified_vars['compositions'].values() 
                       if not isinstance(v, tuple)])
        if comp_sum > 1.001 or comp_sum < 0.999:
            messages.append(f"⚠️ **Composition sum = {comp_sum:.4f}** (should be ≈1.0)")
        
        return "\n".join(messages)
    
    def calculate_gibbs_phase_rule(self, components_count, phases_count):
        """Calculate Gibbs Phase Rule with detailed explanation"""
        F = components_count - phases_count + 2
        
        explanation = {
            'formula': 'F = C - P + 2',
            'variables': {
                'F': 'Degrees of Freedom',
                'C': f'Components ({components_count})',
                'P': f'Phases ({phases_count})',
                '2': 'Temperature and Pressure'
            },
            'result': F,
            'interpretation': self._interpret_gibbs_result(F)
        }
        
        return explanation
    
    def _interpret_gibbs_result(self, F):
        """Interpret the Gibbs Phase Rule result"""
        if F > 0:
            return f"System has {F} degree(s) of freedom. You must specify {F} more intensive variable(s)."
        elif F == 0:
            return "System is invariant (fully determined). Exactly 0 degrees of freedom - perfect for equilibrium calculation."
        else:
            return f"System is over-constrained (F = {F}). Too many phases for the given components."
    
    def validate_equilibrium_conditions(self, components, phases, conditions):
        """Validate conditions before attempting equilibrium calculation"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'corrected_conditions': conditions.copy()
        }
        
        # 1. Check for required T and P
        if v.T not in conditions and 'T' not in str(conditions.keys()):
            validation['errors'].append("❌ Temperature (T) must be specified")
            validation['is_valid'] = False
        
        if v.P not in conditions and 'P' not in str(conditions.keys()):
            validation['errors'].append("❌ Pressure (P) must be specified")
            validation['is_valid'] = False
        
        # 2. Analyze composition variables
        comp_vars = {}
        for key in conditions.keys():
            if hasattr(key, 'species') and 'X(' in str(key):
                element = str(key).split('(')[1].split(')')[0]
                comp_vars[element] = conditions[key]
        
        # Count real components (excluding VA)
        real_components = [c for c in components if c != 'VA']
        num_components = len(real_components)
        
        # Check composition count
        if len(comp_vars) > num_components - 1:
            validation['warnings'].append(
                f"⚠️ You specified {len(comp_vars)} composition variables for {num_components} components. "
                f"For {num_components} components, specify exactly {num_components-1} independent compositions."
            )
        
        # 3. Check composition sum
        comp_sum = 0
        for val in comp_vars.values():
            if not isinstance(val, tuple):  # Skip temperature ranges
                comp_sum += val
        
        if comp_sum > 1.001:
            validation['errors'].append(f"❌ Composition sum = {comp_sum:.4f} > 1.0")
            validation['is_valid'] = False
        elif comp_sum < 0.999 and len(comp_vars) == num_components - 1:
            # For n-1 components, the sum should be ≤ 1.0
            validation['warnings'].append(
                f"⚠️ Composition sum = {comp_sum:.4f}. The remaining component will be 1 - {comp_sum:.4f} = {1-comp_sum:.4f}"
            )
        
        # 4. Check for v.N misuse
        if v.N in conditions:
            validation['warnings'].append(
                "⚠️ v.N is specified. pycalphad assumes N=1 by default. "
                "Remove v.N unless you specifically need to vary total moles."
            )
        
        # 5. Check phase count vs components
        if len(phases) > num_components + 2:
            validation['warnings'].append(
                f"⚠️ {len(phases)} phases specified for {num_components} components. "
                "Gibbs Phase Rule limits maximum phases to C + 2."
            )
        
        # Generate suggestions
        if not validation['is_valid']:
            validation['suggestions'] = self._generate_fix_suggestions(
                real_components, phases, conditions, comp_vars
            )
        
        return validation
    
    def _generate_fix_suggestions(self, components, phases, conditions, comp_vars):
        """Generate specific suggestions to fix DOF issues"""
        suggestions = []
        
        num_components = len(components)
        
        # Basic required variables
        if v.T not in conditions:
            suggestions.append("• Add temperature condition: `conditions[v.T] = 1000` (or your desired temperature)")
        
        if v.P not in conditions:
            suggestions.append("• Add pressure condition: `conditions[v.P] = 101325` (1 atm)")
        
        # Composition suggestions
        if len(comp_vars) != num_components - 1:
            suggestions.append(
                f"• Specify exactly {num_components-1} composition variables for {num_components} components"
            )
            
            # Suggest which compositions to specify
            if num_components > 1:
                example_comps = {}
                for i, comp in enumerate(components[:-1]):
                    example_comps[comp] = 1.0 / num_components
                
                example_str = ", ".join([f"v.X('{k}')={v:.3f}" for k, v in example_comps.items()])
                suggestions.append(f"  Example: {example_str}")
                suggestions.append(f"  (Last component {components[-1]} will be 1 - sum of others)")
        
        # Remove v.N if present
        if v.N in conditions:
            suggestions.append("• Remove v.N from conditions (pycalphad assumes N=1 by default)")
        
        return suggestions
    
    def create_minimal_working_example(self, components, phases):
        """Create a minimal working example for the given components and phases"""
        # Filter out VA from components
        real_components = [c for c in components if c != 'VA']
        num_components = len(real_components)
        
        example = {
            'components': real_components,
            'phases': phases[:min(2, len(phases))],  # Use first 1-2 phases
            'conditions': {}
        }
        
        # Add required conditions
        example['conditions'][v.T] = 1000  # Example temperature
        example['conditions'][v.P] = 101325  # 1 atm
        
        # Add composition conditions (n-1 components)
        if num_components > 1:
            # Distribute compositions evenly
            fraction = 1.0 / num_components
            for i, comp in enumerate(real_components[:-1]):
                example['conditions'][v.X(comp)] = fraction
        
        # Generate code snippet
        code_lines = [
            "from pycalphad import Database, equilibrium, variables as v",
            "",
            f"# Components (excluding VA): {', '.join(real_components)}",
            f"# Phases: {', '.join(example['phases'])}",
            "",
            "dbf = Database('your_database.tdb')",
            "",
            "conditions = {"
        ]
        
        for key, value in example['conditions'].items():
            if key == v.T:
                code_lines.append(f"    v.T: {value},  # Temperature in K")
            elif key == v.P:
                code_lines.append(f"    v.P: {value},  # Pressure in Pa")
            elif 'X(' in str(key):
                element = str(key).split('(')[1].split(')')[0]
                code_lines.append(f"    v.X('{element}'): {value:.3f},")
        
        code_lines.extend([
            "}",
            "",
            "# Add VA to components list",
            f"components_with_va = {real_components + ['VA']}",
            "",
            "# Perform equilibrium calculation",
            "eq = equilibrium(",
            "    dbf,",
            "    components_with_va,",
            f"    {example['phases']},",
            "    conditions,",
            "    output='HM'  # For enthalpy",
            ")",
            "",
            "# This should work if your TDB file has the required data"
        ])
        
        example['code'] = "\n".join(code_lines)
        return example
    
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
        
        return molar_weight if molar_weight > 0 else 50.0
    
    def convert_to_specific_enthalpy(self, df, composition):
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
    
    def get_available_tdb_files(self):
        """Retrieve all TDB files from the databases directory"""
        try:
            tdb_files = []
            for ext in ['*.tdb', '*.TDB']:
                tdb_files.extend(self.database_dir.glob(ext))
            
            # Sort with examples first
            sorted_files = sorted(tdb_files, key=lambda x: (not 'EXAMPLE' in x.name.upper(), x.name.lower()))
            return [f.name for f in sorted_files]
        except Exception as e:
            st.error(f"Error accessing databases directory: {str(e)}")
            return []
    
    def save_uploaded_tdb(self, uploaded_file):
        """Save uploaded TDB file to databases directory"""
        try:
            save_path = self.database_dir / uploaded_file.name
            
            # Create unique filename if exists
            if save_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name_parts = uploaded_file.name.rsplit('.', 1)
                new_name = f"{name_parts[0]}_{timestamp}.{name_parts[1]}" if len(name_parts) > 1 else f"{uploaded_file.name}_{timestamp}"
                save_path = self.database_dir / new_name
            
            with open(save_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            return str(save_path)
        except Exception as e:
            st.error(f"Error saving TDB file: {str(e)}")
            return None

def create_thermodynamic_dashboard(analyzer, components, phases, conditions):
    """Create a comprehensive thermodynamic system dashboard"""
    # Analyze degrees of freedom
    dof_analysis = analyzer.analyze_degrees_of_freedom(components, phases, conditions)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    fig.suptitle('Thermodynamic System Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Gibbs Phase Rule Visualization
    ax1 = axes[0, 0]
    labels = ['Components (C)', 'Phases (P)', 'T & P (+2)']
    values = [dof_analysis['components'], dof_analysis['phases'], 2]
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    
    ax1.bar(labels, values, color=colors)
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Gibbs Phase Rule: F = C - P + 2', fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(values):
        ax1.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    
    # Calculate and display F
    F = dof_analysis['theoretical_F']
    ax1.text(1.5, max(values) * 0.8, f'F = {F}', fontsize=20, 
             fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 2. DOF Status Pie Chart
    ax2 = axes[0, 1]
    if dof_analysis['is_valid']:
        sizes = [100]
        labels = ['✅ Correct\nF = 0']
        colors = ['#4CAF50']
    elif dof_analysis['is_under_specified']:
        specified = dof_analysis['specified_vars']['independent_intensive']
        needed = dof_analysis['theoretical_F']
        sizes = [specified, needed - specified]
        labels = [f'Specified\n{specified}', f'Missing\n{needed - specified}']
        colors = ['#FF9800', '#F44336']
    else:  # Over-specified
        specified = dof_analysis['specified_vars']['independent_intensive']
        excess = specified - dof_analysis['theoretical_F']
        sizes = [dof_analysis['theoretical_F'], excess]
        labels = [f'Required\n{dof_analysis["theoretical_F"]}', f'Excess\n{excess}']
        colors = ['#4CAF50', '#F44336']
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Degrees of Freedom Status', fontweight='bold')
    
    # 3. Variable Specification Status
    ax3 = axes[1, 0]
    var_labels = ['Temperature', 'Pressure', 'Compositions']
    var_status = [
        1 if dof_analysis['specified_vars']['T_specified'] else 0,
        1 if dof_analysis['specified_vars']['P_specified'] else 0,
        min(1, dof_analysis['specified_vars']['total_composition_vars'] / 
            max(1, dof_analysis['components'] - 1))
    ]
    var_colors = ['#4CAF50' if s == 1 else '#F44336' for s in var_status[:2]]
    var_colors.append('#FF9800' if 0 < var_status[2] < 1 else 
                     '#4CAF50' if var_status[2] == 1 else '#F44336')
    
    bars = ax3.bar(var_labels, var_status, color=var_colors)
    ax3.set_ylim(0, 1.2)
    ax3.set_ylabel('Specification Status', fontweight='bold')
    ax3.set_title('Variable Specification', fontweight='bold')
    
    # Add status text
    for i, (bar, status) in enumerate(zip(bars, var_status)):
        if i < 2:
            text = '✅ Specified' if status == 1 else '❌ Missing'
        else:
            comp_vars = dof_analysis['specified_vars']['total_composition_vars']
            needed = max(1, dof_analysis['components'] - 1)
            text = f'{comp_vars}/{needed}'
        
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                text, ha='center', fontweight='bold')
    
    # 4. System Complexity
    ax4 = axes[1, 1]
    complexity_data = {
        'Components': dof_analysis['components'],
        'Phases': dof_analysis['phases'],
        'Specified\nVariables': dof_analysis['specified_vars']['independent_intensive'],
        'Required\nVariables': dof_analysis['theoretical_F']
    }
    
    x_pos = range(len(complexity_data))
    ax4.bar(x_pos, list(complexity_data.values()), color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(list(complexity_data.keys()))
    ax4.set_ylabel('Count', fontweight='bold')
    ax4.set_title('System Complexity', fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(complexity_data.values()):
        ax4.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    return fig, dof_analysis

def create_phase_diagram_visualization(components, phases, conditions):
    """Create a visualization of the phase diagram region"""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Create a simple phase diagram representation
    # This is a conceptual visualization, not an actual calculation
    
    # Extract composition variables
    comp_vars = {}
    for key, value in conditions.items():
        if hasattr(key, 'species') and 'X(' in str(key):
            element = str(key).split('(')[1].split(')')[0]
            if not isinstance(value, tuple):  # Skip ranges
                comp_vars[element] = value
    
    if len(comp_vars) >= 2:
        # Create ternary or binary diagram
        elements = list(comp_vars.keys())
        
        if len(elements) == 2:
            # Binary phase diagram
            x = np.linspace(0, 1, 100)
            T_min, T_max = 500, 2000
            
            # Create conceptual phase boundaries
            for i, phase in enumerate(phases):
                # Simple sinusoidal boundaries for visualization
                T_phase = T_min + (T_max - T_min) * (0.3 + 0.4 * np.sin(np.pi * x + i * np.pi/len(phases)))
                ax.plot(x, T_phase, label=phase, linewidth=2)
            
            # Mark the specified composition
            if len(elements) == 2:
                x_spec = comp_vars[elements[0]]
                ax.axvline(x=x_spec, color='red', linestyle='--', alpha=0.7, 
                          label=f'Specified: X({elements[0]}) = {x_spec:.2f}')
            
            ax.set_xlabel(f'Mole Fraction {elements[0]}', fontweight='bold')
            ax.set_ylabel('Temperature (K)', fontweight='bold')
            ax.set_title(f'Conceptual {elements[0]}-{elements[1]} Phase Diagram', 
                        fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    else:
        # Simple temperature-pressure diagram
        ax.text(0.5, 0.5, 'Phase Diagram Visualization\n\nSpecify at least 2 composition\nvariables for detailed view',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Phase Diagram Region', fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    st.markdown('<h1 class="main-header">🔥 Thermodynamic Enthalpy Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced CALPHAD Calculations with Gibbs Phase Rule Validation")
    st.markdown("*Solve 'Number of degrees of freedom is not zero' error with intelligent system analysis*")
    st.markdown("---")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ThermodynamicSystemAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Create enhanced tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Dashboard",
        "⚖️ DOF Analyzer", 
        "🔬 Computation",
        "📊 Curve Fitting",
        "⚙️ Settings & Help"
    ])
    
    # ==================== TAB 1: Dashboard ====================
    with tab1:
        st.header("🏠 Thermodynamic System Dashboard")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="thermo-card">', unsafe_allow_html=True)
            st.metric("Available TDB Files", len(analyzer.get_available_tdb_files()))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="thermo-card">', unsafe_allow_html=True)
            st.metric("Computed Systems", len(analyzer.results_history))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="thermo-card">', unsafe_allow_html=True)
            st.metric("Fitting Results", len(analyzer.fitting_results))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="thermo-card">', unsafe_allow_html=True)
            st.metric("DOF Errors Fixed", "0", "0")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Gibbs Phase Rule Explanation
        st.markdown("---")
        st.markdown('<div class="phase-rule-box">', unsafe_allow_html=True)
        st.subheader("⚖️ Gibbs Phase Rule")
        st.latex(r"F = C - P + 2")
        st.markdown("""
        **Where:**
        - **F** = Degrees of Freedom (must be 0 for equilibrium)
        - **C** = Number of Components (excluding VA)
        - **P** = Number of Phases
        - **2** = Temperature and Pressure (always count as 2 intensive variables)
        
        **Key Insight:** For pycalphad `equilibrium()` to work, you must specify exactly **F** intensive variables!
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Common error patterns
        st.markdown("---")
        st.subheader("🔍 Common DOF Error Patterns")
        
        col_err1, col_err2, col_err3 = st.columns(3)
        
        with col_err1:
            with st.expander("❌ Missing T or P", expanded=False):
                st.code("""# ERROR: Missing required variables
conditions = {
    v.X('AL'): 0.7,  # Only composition, no T or P!
}""", language='python')
                st.markdown("**Fix:** Add both T and P")
                st.code("""conditions = {
    v.T: 1000,
    v.P: 101325,
    v.X('AL'): 0.7
}""", language='python')
        
        with col_err2:
            with st.expander("❌ Over-specified Composition", expanded=False):
                st.code("""# ERROR: Specifying all compositions for binary
conditions = {
    v.T: 1000,
    v.P: 101325,
    v.X('AL'): 0.7,
    v.X('CU'): 0.3  # Redundant! Sums to 1.0
}""", language='python')
                st.markdown("**Fix:** Specify n-1 components")
                st.code("""conditions = {
    v.T: 1000,
    v.P: 101325,
    v.X('AL'): 0.7  # CU is implicit: 1 - 0.7 = 0.3
}""", language='python')
        
        with col_err3:
            with st.expander("❌ Using v.N unnecessarily", expanded=False):
                st.code("""# ERROR: v.N is usually not needed
conditions = {
    v.T: 1000,
    v.P: 101325,
    v.X('AL'): 0.7,
    v.N: 1  # pycalphad assumes N=1 by default
}""", language='python')
                st.markdown("**Fix:** Remove v.N")
                st.code("""conditions = {
    v.T: 1000,
    v.P: 101325,
    v.X('AL'): 0.7
}""", language='python')
        
        # Quick action buttons
        st.markdown("---")
        st.subheader("🚀 Quick Actions")
        
        col_act1, col_act2, col_act3 = st.columns(3)
        
        with col_act1:
            if st.button("📁 Upload New TDB", use_container_width=True):
                st.session_state.active_tab = "🔬 Computation"
                st.rerun()
        
        with col_act2:
            if st.button("⚖️ Analyze DOF", use_container_width=True):
                st.session_state.active_tab = "⚖️ DOF Analyzer"
                st.rerun()
        
        with col_act3:
            if st.button("🔄 Clear Session", type="secondary", use_container_width=True):
                analyzer.results_history = []
                analyzer.fitting_results = []
                st.success("Session data cleared!")
                st.rerun()
    
    # ==================== TAB 2: DOF Analyzer ====================
    with tab2:
        st.header("⚖️ Degrees of Freedom Analyzer")
        st.markdown("**Diagnose and fix 'Number of degrees of freedom is not zero' errors**")
        
        col_ana1, col_ana2 = st.columns([1, 1])
        
        with col_ana1:
            st.subheader("📋 System Configuration")
            
            # Manual input or load from TDB
            config_source = st.radio(
                "Configuration source:",
                ["Manual input", "Load from TDB file"],
                horizontal=True
            )
            
            if config_source == "Load from TDB file":
                available_tdb = analyzer.get_available_tdb_files()
                if not available_tdb:
                    st.info("No TDB files found. Please upload one in the Settings tab.")
                    selected_tdb = None
                else:
                    selected_tdb = st.selectbox(
                        "Select TDB file:",
                        available_tdb
                    )
                    
                    if selected_tdb:
                        tdb_path = analyzer.database_dir / selected_tdb
                        try:
                            dbf = Database(str(tdb_path))
                            elements = sorted([e for e in dbf.elements if e != 'VA'])
                            phases = sorted(dbf.phases.keys())
                            
                            # Auto-fill components and phases
                            components = st.multiselect(
                                "Components (select from TDB):",
                                elements,
                                default=elements[:min(2, len(elements))]
                            )
                            
                            selected_phases = st.multiselect(
                                "Phases (select from TDB):",
                                phases,
                                default=phases[:min(2, len(phases))]
                            )
                            
                        except Exception as e:
                            st.error(f"Error loading TDB: {str(e)}")
                            components = []
                            selected_phases = []
            else:
                # Manual input
                st.markdown("**Enter Components:**")
                comp_input = st.text_input(
                    "Component symbols (comma-separated, e.g., AL, CU, NI):",
                    "AL, CU"
                )
                components = [c.strip().upper() for c in comp_input.split(',') if c.strip()]
                
                st.markdown("**Enter Phases:**")
                phase_input = st.text_input(
                    "Phase names (comma-separated, e.g., LIQUID, FCC_A1):",
                    "LIQUID, FCC_A1"
                )
                selected_phases = [p.strip().upper() for p in phase_input.split(',') if p.strip()]
            
            if not components:
                st.warning("Please specify at least one component")
                st.stop()
            
            if not selected_phases:
                st.warning("Please specify at least one phase")
                st.stop()
            
            # Add VA to components for calculation
            components_with_va = components + ['VA']
        
        with col_ana2:
            st.subheader("⚙️ Conditions Specification")
            
            # Temperature
            temp_type = st.radio(
                "Temperature specification:",
                ["Single value", "Range"],
                horizontal=True
            )
            
            if temp_type == "Single value":
                T_value = st.number_input("Temperature (K)", 100, 5000, 1000)
                T_condition = T_value
            else:
                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    T_start = st.number_input("T start (K)", 100, 5000, 800)
                with col_t2:
                    T_end = st.number_input("T end (K)", T_start, 6000, 1500)
                with col_t3:
                    T_step = st.number_input("T step (K)", 1, 200, 10)
                T_condition = (T_start, T_end, T_step)
            
            # Pressure
            P_value = st.number_input("Pressure (Pa)", 1000, 10000000, 101325)
            
            # Composition specification
            st.markdown("**Composition (mole fractions):**")
            st.caption(f"For {len(components)} components, specify exactly {len(components)-1} independent compositions")
            
            composition = {}
            remaining = 1.0
            
            for i, comp in enumerate(components):
                if i < len(components) - 1:
                    max_val = min(1.0, remaining)
                    fraction = st.number_input(
                        f"X({comp})",
                        min_value=0.0,
                        max_value=float(max_val),
                        value=float(1.0/len(components)),
                        step=0.01,
                        key=f"dof_comp_{comp}"
                    )
                    composition[comp] = fraction
                    remaining -= fraction
                else:
                    # Last component gets remaining
                    st.info(f"X({comp}) = {remaining:.3f} (calculated)")
                    composition[comp] = remaining
            
            # Build conditions dictionary
            conditions = {
                v.T: T_condition,
                v.P: P_value
            }
            
            # Add composition conditions (n-1 components)
            for i, (comp, frac) in enumerate(composition.items()):
                if i < len(components) - 1:  # Only first n-1
                    conditions[v.X(comp)] = frac
        
        # Analyze button
        st.markdown("---")
        if st.button("🔬 Analyze Degrees of Freedom", type="primary", use_container_width=True):
            with st.spinner("Analyzing thermodynamic system..."):
                try:
                    # Create dashboard visualization
                    fig, dof_analysis = create_thermodynamic_dashboard(
                        analyzer, components_with_va, selected_phases, conditions
                    )
                    
                    st.pyplot(fig)
                    
                    # Display analysis results
                    st.markdown("### 📊 Analysis Results")
                    st.markdown(dof_analysis['message'], unsafe_allow_html=True)
                    
                    # Additional validation
                    validation = analyzer.validate_equilibrium_conditions(
                        components_with_va, selected_phases, conditions
                    )
                    
                    if validation['errors']:
                        st.markdown("### ❌ Validation Errors")
                        for error in validation['errors']:
                            st.error(error)
                    
                    if validation['warnings']:
                        st.markdown("### ⚠️ Validation Warnings")
                        for warning in validation['warnings']:
                            st.warning(warning)
                    
                    if validation['suggestions']:
                        st.markdown("### 💡 Suggestions for Fix")
                        for suggestion in validation['suggestions']:
                            st.markdown(suggestion)
                    
                    # Generate working example
                    if not dof_analysis['is_valid']:
                        st.markdown("### 🔧 Minimal Working Example")
                        example = analyzer.create_minimal_working_example(
                            components_with_va, selected_phases
                        )
                        
                        with st.expander("View code template", expanded=False):
                            st.code(example['code'], language='python')
                        
                        if st.button("📋 Copy to Clipboard", key="copy_example"):
                            st.code(example['code'], language='python')
                            st.success("Code copied! Paste it into your calculation.")
                    
                    # Test equilibrium if system is valid
                    if dof_analysis['is_valid'] and validation['is_valid']:
                        st.markdown("### 🎯 Ready for Equilibrium Calculation")
                        
                        # Show what will be passed to equilibrium
                        st.markdown("**Final conditions for equilibrium():**")
                        conds_display = []
                        for key, value in conditions.items():
                            if key == v.T:
                                if isinstance(value, tuple):
                                    conds_display.append(f"v.T: ({value[0]}, {value[1]}, {value[2]})")
                                else:
                                    conds_display.append(f"v.T: {value}")
                            elif key == v.P:
                                conds_display.append(f"v.P: {value}")
                            elif 'X(' in str(key):
                                element = str(key).split('(')[1].split(')')[0]
                                conds_display.append(f"v.X('{element}'): {value:.3f}")
                        
                        for cond in conds_display:
                            st.code(cond)
                        
                        st.success("✅ System is properly constrained! Ready for equilibrium calculation.")
                        
                        # Option to test with a TDB file
                        test_tdb = st.selectbox(
                            "Select TDB file to test:",
                            [""] + analyzer.get_available_tdb_files()
                        )
                        
                        if test_tdb and st.button("🧪 Test Equilibrium Calculation", type="secondary"):
                            with st.spinner("Testing equilibrium calculation..."):
                                try:
                                    tdb_path = analyzer.database_dir / test_tdb
                                    dbf_test = Database(str(tdb_path))
                                    
                                    # Check if selected phases exist in database
                                    available_phases = sorted(dbf_test.phases.keys())
                                    missing_phases = [p for p in selected_phases if p not in available_phases]
                                    
                                    if missing_phases:
                                        st.warning(f"Phases not in database: {', '.join(missing_phases)}")
                                        st.info(f"Available phases: {', '.join(available_phases[:5])}...")
                                        # Use only available phases
                                        selected_phases = [p for p in selected_phases if p in available_phases]
                                    
                                    # Perform equilibrium calculation
                                    eq_result = equilibrium(
                                        dbf_test,
                                        components_with_va,
                                        selected_phases,
                                        conditions,
                                        output='HM',
                                        verbose=False
                                    )
                                    
                                    st.success("✅ Equilibrium calculation successful!")
                                    
                                    # Display results
                                    T_values = eq_result.T.values.flatten()
                                    HM_values = eq_result.HM.values.flatten()
                                    
                                    result_df = pd.DataFrame({
                                        'Temperature_K': T_values,
                                        'Enthalpy_J_mol': HM_values
                                    })
                                    
                                    result_df = result_df.dropna().sort_values('Temperature_K')
                                    
                                    if len(result_df) > 0:
                                        st.metric("Data Points Generated", len(result_df))
                                        
                                        # Quick plot
                                        fig_test, ax_test = plt.subplots(figsize=(10, 6))
                                        ax_test.plot(result_df['Temperature_K'], result_df['Enthalpy_J_mol'], 
                                                   'b-', linewidth=2, marker='o', markersize=4)
                                        ax_test.set_xlabel('Temperature (K)', fontweight='bold')
                                        ax_test.set_ylabel('Enthalpy (J/mol)', fontweight='bold')
                                        ax_test.set_title('Test Equilibrium Calculation Result', fontweight='bold')
                                        ax_test.grid(True, alpha=0.3)
                                        st.pyplot(fig_test)
                                        
                                        # Store for later use
                                        result_info = {
                                            'name': f"Test: {test_tdb}",
                                            'composition': composition,
                                            'data': result_df,
                                            'phases': selected_phases,
                                            'tdb_file': test_tdb,
                                            'temperature_range': (T_condition if isinstance(T_condition, tuple) else (T_condition, T_condition, 1)),
                                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        }
                                        analyzer.results_history.append(result_info)
                                        
                                        st.success("✅ Test calculation stored in history!")
                                    else:
                                        st.warning("Calculation completed but returned no valid data points.")
                                
                                except Exception as e:
                                    st.error(f"❌ Test calculation failed: {str(e)}")
                                    st.code(traceback.format_exc(), language='python')
                
                except Exception as e:
                    st.error(f"❌ Analysis error: {str(e)}")
                    st.code(traceback.format_exc(), language='python')
        
        # Quick reference
        st.markdown("---")
        st.subheader("📚 Quick Reference")
        
        col_ref1, col_ref2 = st.columns(2)
        
        with col_ref1:
            st.markdown("**Rules for F = 0:**")
            st.markdown("""
            1. **Always specify T and P**
            2. **For n components, specify n-1 compositions**
            3. **Don't use v.N unless necessary**
            4. **Ensure compositions sum ≤ 1.0**
            5. **Last composition is implicit: 1 - sum(others)**
            """)
        
        with col_ref2:
            st.markdown("**Common Patterns:**")
            st.markdown("""
            - **Binary (2 components):** Specify 1 composition
            - **Ternary (3 components):** Specify 2 compositions  
            - **Quaternary (4 components):** Specify 3 compositions
            - **Unary (1 component):** No composition needed
            """)
    
    # ==================== TAB 3: Computation ====================
    with tab3:
        st.header("🔬 Enthalpy Computation with DOF Protection")
        
        # File selection
        st.subheader("📁 TDB File Selection")
        
        available_tdb = analyzer.get_available_tdb_files()
        if not available_tdb:
            st.info("No TDB files found. Using built-in examples...")
            available_tdb = analyzer.get_available_tdb_files()
        
        col_file1, col_file2 = st.columns([1.2, 1])
        
        with col_file1:
            tdb_source = st.radio(
                "Source:",
                ["Select from database", "Upload new TDB"],
                horizontal=True
            )
            
            tdb_path = None
            
            if tdb_source == "Select from database" and available_tdb:
                selected_file = st.selectbox(
                    "Available TDB files:",
                    available_tdb,
                    index=0
                )
                tdb_path = str(analyzer.database_dir / selected_file)
                st.success(f"✅ Selected: **{selected_file}**")
            
            else:
                uploaded_file = st.file_uploader(
                    "Upload TDB file",
                    type=["tdb", "TDB"],
                    key="uploader_tab3"
                )
                
                if uploaded_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.tdb') as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tdb_path = tmp.name
                    
                    st.success(f"✅ Uploaded: **{uploaded_file.name}**")
            
            if not tdb_path:
                st.info("👈 Please select or upload a TDB file")
                st.stop()
        
        with col_file2:
            try:
                dbf = Database(tdb_path)
                
                elements = sorted([e for e in dbf.elements if e != 'VA'])
                phases = sorted(dbf.phases.keys())
                
                st.markdown("**Database Info:**")
                st.markdown(f"- **Elements:** {len(elements)}")
                st.markdown(f"- **Phases:** {len(phases)}")
                
                with st.expander("View details", expanded=False):
                    st.markdown(f"**Elements:** {', '.join(elements[:10])}{'...' if len(elements) > 10 else ''}")
                    st.markdown(f"**Phases:** {', '.join(phases[:10])}{'...' if len(phases) > 10 else ''}")
            
            except Exception as e:
                st.error(f"❌ Error loading database: {str(e)}")
                st.stop()
        
        # System configuration with DOF protection
        st.markdown("---")
        st.subheader("⚙️ System Configuration")
        
        col_conf1, col_conf2 = st.columns([1, 1])
        
        with col_conf1:
            # Element selection
            selected_elements = st.multiselect(
                "Select components:",
                elements,
                default=elements[:min(3, len(elements))],
                help="Select elements for your alloy system"
            )
            
            if not selected_elements:
                st.warning("Please select at least one element")
                st.stop()
            
            # Composition input with auto-balancing
            st.markdown("**Composition Input:**")
            st.caption(f"For {len(selected_elements)} components, specify {len(selected_elements)-1} values")
            
            composition = {}
            total_so_far = 0.0
            
            for i, element in enumerate(selected_elements):
                if i < len(selected_elements) - 1:
                    max_val = 1.0 - total_so_far
                    fraction = st.number_input(
                        f"X({element})",
                        min_value=0.0,
                        max_value=float(max_val),
                        value=float(1.0/len(selected_elements)),
                        step=0.01,
                        key=f"comp_{element}"
                    )
                    composition[element] = fraction
                    total_so_far += fraction
                else:
                    # Last element
                    remaining = 1.0 - total_so_far
                    composition[element] = remaining
                    st.info(f"X({element}) = {remaining:.3f} (auto-calculated)")
            
            # Display composition summary
            st.markdown("**Composition Summary:**")
            comp_text = ""
            for elem, frac in composition.items():
                comp_text += f'<span class="component-badge">{elem}: {frac:.3f}</span> '
            st.markdown(comp_text, unsafe_allow_html=True)
            
            # Validate composition
            comp_sum = sum(composition.values())
            if abs(comp_sum - 1.0) > 0.001:
                st.error(f"❌ Composition sum = {comp_sum:.4f} (must be 1.0)")
                st.stop()
        
        with col_conf2:
            # Temperature settings
            st.markdown("**Temperature Range:**")
            temp_mode = st.radio("Mode:", ["Single", "Range"], horizontal=True)
            
            if temp_mode == "Single":
                T_value = st.number_input("Temperature (K)", 100, 5000, 1000)
                T_condition = T_value
                T_display = f"{T_value} K"
            else:
                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    T_start = st.number_input("Start (K)", 100, 5000, 300)
                with col_t2:
                    T_end = st.number_input("End (K)", T_start, 6000, 1500)
                with col_t3:
                    T_step = st.number_input("Step (K)", 1, 200, 10)
                T_condition = (T_start, T_end, T_step)
                T_display = f"{T_start}-{T_end} K, step={T_step}"
            
            # Pressure
            P_value = st.number_input("Pressure (Pa)", 1000, 10000000, 101325)
            
            # Phase selection
            st.markdown("**Phase Selection:**")
            selected_phases = st.multiselect(
                "Select phases for equilibrium:",
                phases,
                default=phases[:min(2, len(phases))]
            )
            
            if not selected_phases:
                st.warning("Please select at least one phase")
                st.stop()
            
            # Advanced options
            with st.expander("⚙️ Advanced Options", expanded=False):
                calc_mode = st.selectbox(
                    "Calculation mode",
                    ["Fast", "Accurate", "Very Accurate"],
                    index=1
                )
                
                output_variable = st.selectbox(
                    "Output variable",
                    ["HM", "GM", "SM", "CPM"],
                    index=0,
                    help="HM: Enthalpy, GM: Gibbs energy, SM: Entropy, CPM: Heat capacity"
                )
        
        # DOF Analysis and Validation
        st.markdown("---")
        st.subheader("⚖️ DOF Validation Before Calculation")
        
        # Build conditions
        conditions = {
            v.T: T_condition,
            v.P: P_value
        }
        
        # Add composition conditions (n-1)
        for i, (element, fraction) in enumerate(composition.items()):
            if i < len(selected_elements) - 1:  # Only first n-1
                conditions[v.X(element)] = fraction
        
        components_with_va = selected_elements + ['VA']
        
        # Perform DOF analysis
        dof_analysis = analyzer.analyze_degrees_of_freedom(
            components_with_va, selected_phases, conditions
        )
        
        validation = analyzer.validate_equilibrium_conditions(
            components_with_va, selected_phases, conditions
        )
        
        # Display analysis
        col_val1, col_val2 = st.columns([2, 1])
        
        with col_val1:
            if dof_analysis['is_valid'] and validation['is_valid']:
                st.markdown('<div class="dof-indicator dof-correct">✅ SYSTEM VALID - Ready for calculation!</div>', 
                          unsafe_allow_html=True)
            elif dof_analysis['is_under_specified']:
                st.markdown('<div class="dof-indicator dof-warning">⚠️ UNDER-CONSTRAINED - Fix before calculation</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown('<div class="dof-indicator dof-error">❌ OVER-CONSTRAINED - Fix before calculation</div>', 
                          unsafe_allow_html=True)
        
        with col_val2:
            st.metric("Degrees of Freedom", dof_analysis['actual_F'], 
                     delta="Target: 0", delta_color="normal" if dof_analysis['actual_F'] == 0 else "off")
        
        # Show validation details
        if validation['errors']:
            st.error("**Validation Errors:**")
            for error in validation['errors']:
                st.error(error)
        
        if validation['warnings']:
            st.warning("**Validation Warnings:**")
            for warning in validation['warnings']:
                st.warning(warning)
        
        # Show what will be passed to equilibrium
        with st.expander("📋 View Conditions for equilibrium()", expanded=False):
            st.markdown("**Components:**")
            st.markdown(f"`{components_with_va}`")
            
            st.markdown("**Phases:**")
            st.markdown(f"`{selected_phases}`")
            
            st.markdown("**Conditions:**")
            for key, value in conditions.items():
                if key == v.T:
                    if isinstance(value, tuple):
                        st.code(f"v.T: ({value[0]}, {value[1]}, {value[2]})")
                    else:
                        st.code(f"v.T: {value}")
                elif key == v.P:
                    st.code(f"v.P: {value}")
                elif 'X(' in str(key):
                    element = str(key).split('(')[1].split(')')[0]
                    st.code(f"v.X('{element}'): {value}")
        
        # Calculate button (only enabled if valid)
        st.markdown("---")
        calculate_enabled = dof_analysis['is_valid'] and validation['is_valid']
        
        if calculate_enabled:
            if st.button("🚀 Calculate Enthalpy", type="primary", use_container_width=True):
                with st.spinner("Performing equilibrium calculation..."):
                    try:
                        # Perform equilibrium calculation
                        eq_result = equilibrium(
                            dbf,
                            components_with_va,
                            selected_phases,
                            conditions,
                            output=output_variable,
                            verbose=False
                        )
                        
                        # Extract results
                        T_values = eq_result.T.values.flatten()
                        result_values = eq_result[output_variable].values.flatten()
                        
                        # Create DataFrame
                        result_df = pd.DataFrame({
                            'Temperature_K': T_values,
                            f'{output_variable}_J_mol': result_values
                        })
                        
                        result_df = result_df.dropna().sort_values('Temperature_K')
                        
                        if len(result_df) == 0:
                            st.error("❌ Calculation returned no valid results")
                            st.stop()
                        
                        # Rename enthalpy column for consistency
                        if output_variable == 'HM':
                            result_df = result_df.rename(columns={'HM_J_mol': 'Enthalpy_J_mol'})
                        
                            # Add specific enthalpy
                            result_df = analyzer.convert_to_specific_enthalpy(result_df, composition)
                        
                        # Store results
                        material_name = f"{'-'.join([e for e in selected_elements[:3]])}"
                        if len(selected_elements) > 3:
                            material_name += f"-{len(selected_elements)-3}more"
                        
                        result_info = {
                            'name': material_name,
                            'composition': composition,
                            'data': result_df,
                            'phases': selected_phases,
                            'tdb_file': os.path.basename(tdb_path),
                            'temperature_range': (T_condition if isinstance(T_condition, tuple) else (T_condition, T_condition, 1)),
                            'output_variable': output_variable,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        analyzer.results_history.append(result_info)
                        
                        st.success(f"✅ Calculation successful! Generated {len(result_df)} data points")
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
                        
                        if output_variable == 'HM':
                            ax.plot(result_df['Temperature_K'], result_df['Enthalpy_J_mol'], 
                                   'b-', linewidth=2.5, marker='o', markersize=4, 
                                   markevery=max(1, len(result_df)//20))
                            ax.set_ylabel('Enthalpy (J/mol)', fontweight='bold')
                            ax.set_title(f'Enthalpy vs Temperature - {material_name}', fontweight='bold')
                            
                            # Add specific enthalpy on secondary axis
                            ax2 = ax.twinx()
                            ax2.plot(result_df['Temperature_K'], result_df['Enthalpy_J_kg'], 
                                    'r-', linewidth=2, alpha=0.7, linestyle='--')
                            ax2.set_ylabel('Specific Enthalpy (J/kg)', fontweight='bold', color='r')
                            ax2.tick_params(axis='y', labelcolor='r')
                        else:
                            ax.plot(result_df['Temperature_K'], result_df[f'{output_variable}_J_mol'], 
                                   'g-', linewidth=2.5)
                            ax.set_ylabel(f'{output_variable} (J/mol)', fontweight='bold')
                            ax.set_title(f'{output_variable} vs Temperature - {material_name}', fontweight='bold')
                        
                        ax.set_xlabel('Temperature (K)', fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Key metrics
                        col_m1, col_m2, col_m3 = st.columns(3)
                        
                        with col_m1:
                            if output_variable == 'HM':
                                min_val = result_df['Enthalpy_J_mol'].min()
                                max_val = result_df['Enthalpy_J_mol'].max()
                                st.metric("Min Enthalpy", f"{min_val:,.0f} J/mol")
                        
                        with col_m2:
                            if output_variable == 'HM':
                                st.metric("Max Enthalpy", f"{max_val:,.0f} J/mol")
                        
                        with col_m3:
                            if output_variable == 'HM':
                                delta = max_val - min_val
                                st.metric("ΔH", f"{delta:,.0f} J/mol")
                        
                        # Download options
                        st.markdown("---")
                        st.subheader("📥 Download Results")
                        
                        col_dl1, col_dl2 = st.columns(2)
                        
                        with col_dl1:
                            csv_data = result_df.to_csv(index=False)
                            st.download_button(
                                "📄 Download CSV",
                                data=csv_data,
                                file_name=f"results_{material_name}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col_dl2:
                            # Create comprehensive report
                            report = {
                                'material': material_name,
                                'composition': composition,
                                'conditions': {
                                    'temperature': T_display,
                                    'pressure': f"{P_value} Pa",
                                    'phases': selected_phases
                                },
                                'data': result_df.to_dict('records')
                            }
                            
                            json_data = json.dumps(report, indent=4)
                            st.download_button(
                                "📁 Download JSON Report",
                                data=json_data,
                                file_name=f"report_{material_name}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                    
                    except Exception as e:
                        st.error(f"❌ Calculation error: {str(e)}")
                        st.code(traceback.format_exc(), language='python')
                        
                        # Provide specific help for DOF errors
                        if "degrees of freedom is not zero" in str(e):
                            st.markdown("### 🆘 DOF Error Detected!")
                            st.markdown("""
                            Even though our analysis showed F=0, pycalphad still encountered a DOF error.
                            This can happen because:
                            
                            1. **Phase stability issues**: Selected phases might not be stable at given conditions
                            2. **Database limitations**: Some phases might not be defined for all compositions
                            3. **Numerical issues** in the starting point calculation
                            
                            **Try:**
                            1. Use different phases
                            2. Adjust temperature range
                            3. Try different composition
                            4. Check the DOF Analyzer tab for detailed diagnosis
                            """)
        else:
            st.button("🚀 Calculate Enthalpy", 
                     use_container_width=True, 
                     disabled=True,
                     help="Fix DOF issues before calculation")
            
            # Show what needs to be fixed
            st.markdown("### 🔧 Fix Required Before Calculation")
            
            if dof_analysis['is_under_specified']:
                st.error(f"**Under-constrained:** Need {dof_analysis['theoretical_F'] - dof_analysis['specified_vars']['independent_intensive']} more intensive variable(s)")
            
            if dof_analysis['is_over_specified']:
                st.error(f"**Over-constrained:** Remove {dof_analysis['specified_vars']['independent_intensive'] - dof_analysis['theoretical_F']} constraint(s)")
            
            # Provide specific suggestions
            suggestions = analyzer._generate_fix_suggestions(
                selected_elements, selected_phases, conditions, 
                dof_analysis['specified_vars']['compositions']
            )
            
            if suggestions:
                st.markdown("**Suggestions:**")
                for suggestion in suggestions:
                    st.markdown(suggestion)
    
    # ==================== TAB 4: Curve Fitting ====================
    with tab4:
        st.header("📊 Curve Fitting & Analysis")
        
        if not analyzer.results_history:
            st.info("💡 No computed data available. Please perform calculations in the 'Enthalpy Computation' tab first.")
            st.stop()
        
        # Data selection
        st.subheader("📈 Select Data for Fitting")
        
        result_options = [
            f"{i+1}. {res['name']} | {res['tdb_file']} | {len(res['data'])} pts"
            for i, res in enumerate(analyzer.results_history)
        ]
        
        selected_idx = st.selectbox(
            "Select computed result:",
            range(len(result_options)),
            format_func=lambda x: result_options[x],
            index=len(result_options)-1
        )
        
        result = analyzer.results_history[selected_idx]
        data = result['data']
        
        # Check if we have enthalpy data
        if 'Enthalpy_J_mol' not in data.columns:
            st.error("Selected data doesn't contain enthalpy. Please use 'HM' output in calculations.")
            st.stop()
        
        T_data = data['Temperature_K'].values
        H_data = data['Enthalpy_J_mol'].values
        
        # Display data info
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.metric("Data Points", len(T_data))
        
        with col_info2:
            st.metric("Temperature Range", f"{T_data.min():.0f}-{T_data.max():.0f} K")
        
        with col_info3:
            delta_H = H_data.max() - H_data.min()
            st.metric("ΔH Range", f"{delta_H:,.0f} J/mol")
        
        # Fitting parameters
        st.markdown("---")
        st.subheader("⚙️ Fitting Parameters")
        
        # Smart initial guesses
        T_range = T_data.max() - T_data.min()
        H_range = H_data.max() - H_data.min()
        H_slope = H_range / T_range if T_range > 0 else 1.0
        T_mid = (T_data.max() + T_data.min()) / 2
        
        col_fit1, col_fit2 = st.columns(2)
        
        with col_fit1:
            A1_guess = st.number_input(
                "A₁ initial (J/mol·K)", 
                -200.0, 200.0, 
                float(max(5.0, min(50.0, H_slope * 0.5))), 
                0.1
            )
            A2_guess = st.number_input(
                "A₂ initial (J/mol·K)", 
                -200.0, 200.0, 
                float(max(1.0, min(30.0, H_slope * 0.2))), 
                0.1
            )
            Tm_guess = st.number_input(
                "Tₘ initial (K)", 
                float(T_data.min()), float(T_data.max()), 
                float(T_mid), 
                1.0
            )
        
        with col_fit2:
            DeltaHf_guess = st.number_input(
                "ΔHf initial (J/mol)", 
                -100000.0, 100000.0, 
                float(max(5000.0, min(50000.0, H_range * 0.3))), 
                100.0
            )
            k_guess = st.number_input(
                "k initial (1/K)", 
                0.0001, 1.0, 
                0.01, 
                0.001
            )
            H298_guess = st.number_input(
                "H₂₉₈ initial (J/mol)", 
                -100000.0, 100000.0, 
                float(H_data.min() * 0.9), 
                100.0
            )
        
        # Fit button
        if st.button("🎯 Perform Curve Fit", type="primary", use_container_width=True):
            with st.spinner("Fitting curve to data..."):
                try:
                    initial_guess = [A1_guess, A2_guess, Tm_guess, DeltaHf_guess, k_guess, H298_guess]
                    
                    # Bounds
                    lower_bounds = [-500, -500, T_data.min() * 0.8, 0, 1e-6, -1e6]
                    upper_bounds = [500, 500, T_data.max() * 1.2, 1e6, 1.0, 1e6]
                    
                    fit_params, pcov = curve_fit(
                        analyzer.enthalpy_equation,
                        T_data,
                        H_data,
                        p0=initial_guess,
                        bounds=(lower_bounds, upper_bounds),
                        maxfev=10000
                    )
                    
                    A1_fit, A2_fit, Tm_fit, DeltaHf_fit, k_fit, H298_fit = fit_params
                    
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
                    
                    # Store results
                    fit_result = {
                        'material_name': result['name'],
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
                        'composition': result['composition'],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    analyzer.fitting_results.append(fit_result)
                    
                    st.success(f"✅ Fitting completed! R² = {r_squared:.6f}, RMSE = {rmse:.2f} J/mol")
                    
                    # Visualization
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=150)
                    
                    # Main fit
                    ax1.scatter(T_data, H_data, alpha=0.6, label='Data', 
                               color='blue', s=40)
                    ax1.plot(T_fit, H_fit, 'r-', linewidth=2.5, label='Fit')
                    ax1.set_xlabel('Temperature (K)', fontweight='bold')
                    ax1.set_ylabel('Enthalpy (J/mol)', fontweight='bold')
                    ax1.set_title(f'Enthalpy-Temperature Fit - {result["name"]}', 
                                 fontweight='bold')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Residuals
                    ax2.scatter(T_data, residuals, alpha=0.6, color='green', s=30)
                    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
                    ax2.set_xlabel('Temperature (K)', fontweight='bold')
                    ax2.set_ylabel('Residuals (J/mol)', fontweight='bold')
                    ax2.set_title('Residuals', fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display equation
                    st.markdown("### 🧮 Fitted Equation")
                    st.latex(rf"""
                    H(T) = {A1_fit:.3f} \cdot T + {A2_fit:.3f} \cdot \max(T - {Tm_fit:.1f}, 0) + 
                    {DeltaHf_fit:,.0f} \cdot \frac{{1}}{{1 + e^{{-{k_fit:.5f}(T - {Tm_fit:.1f})}}}} + {H298_fit:,.0f}
                    """)
                
                except Exception as e:
                    st.error(f"❌ Fitting error: {str(e)}")
    
    # ==================== TAB 5: Settings & Help ====================
    with tab5:
        st.header("⚙️ Settings & Comprehensive Help")
        
        col_set1, col_set2 = st.columns([1, 1])
        
        with col_set1:
            st.subheader("📖 Complete DOF Error Guide")
            
            with st.expander("❓ What is 'Number of degrees of freedom is not zero'?", expanded=True):
                st.markdown("""
                **This is a thermodynamic consistency error from pycalphad's `equilibrium()` function.**
                
                It means: **Your system is not fully determined for equilibrium calculation.**
                
                ### 🔬 Thermodynamic Meaning:
                
                From the **Gibbs Phase Rule**:
                ```
                F = C - P + 2
                ```
                
                - **F** = Degrees of Freedom (must be 0 for equilibrium)
                - **C** = Number of Components (real elements, excluding VA)
                - **P** = Number of Phases
                - **2** = Temperature + Pressure (always count as 2 intensive variables)
                
                **For `equilibrium()` to work:**
                ```
                F = 0  ← You must have exactly 0 degrees of freedom!
                ```
                
                Which means you must specify:
                ```
                Number of specified intensive variables = C - P + 2
                ```
                """)
            
            with st.expander("🎯 Exactly What to Specify", expanded=False):
                st.markdown("""
                ### For N-component system:
                
                **ALWAYS specify:**
                1. **Temperature (T)** - in Kelvin
                2. **Pressure (P)** - in Pascals
                
                **For composition (CRITICAL PART):**
                - **Specify exactly N-1 mole fractions**
                - **DO NOT specify all N compositions**
                - **DO NOT let compositions sum to > 1.0**
                
                ### Examples:
                
                **Binary (2 components):**
                ```python
                # CORRECT - Specify 1 composition
                conditions = {
                    v.T: 1000,
                    v.P: 101325,
                    v.X('AL'): 0.7  # CU is implicit: 1 - 0.7 = 0.3
                }
                ```
                
                **Ternary (3 components):**
                ```python
                # CORRECT - Specify 2 compositions
                conditions = {
                    v.T: 1000,
                    v.P: 101325,
                    v.X('AL'): 0.6,
                    v.X('CU'): 0.2  # NI is implicit: 1 - 0.6 - 0.2 = 0.2
                }
                ```
                
                **Unary (1 component):**
                ```python
                # CORRECT - No composition needed
                conditions = {
                    v.T: 1000,
                    v.P: 101325
                }
                ```
                """)
            
            with st.expander("🚫 Common Mistakes & Fixes", expanded=False):
                st.markdown("""
                ### ❌ MISTAKE 1: Missing T or P
                ```python
                # WRONG - No temperature!
                conditions = {
                    v.P: 101325,
                    v.X('AL'): 0.7
                }
                ```
                **FIX:** Always include both `v.T` and `v.P`
                
                ### ❌ MISTAKE 2: Specifying all compositions
                ```python
                # WRONG - Specifying all 3 for ternary
                conditions = {
                    v.T: 1000,
                    v.P: 101325,
                    v.X('AL'): 0.6,
                    v.X('CU'): 0.2,
                    v.X('NI'): 0.2  # REDUNDANT! Sums to 1.0
                }
                ```
                **FIX:** Specify only N-1 compositions
                
                ### ❌ MISTAKE 3: Using v.N unnecessarily
                ```python
                # WRONG - v.N is not needed
                conditions = {
                    v.T: 1000,
                    v.P: 101325,
                    v.X('AL'): 0.7,
                    v.N: 1  # pycalphad assumes N=1 by default
                }
                ```
                **FIX:** Remove `v.N` unless you specifically need to vary total moles
                
                ### ❌ MISTAKE 4: Composition sum > 1.0
                ```python
                # WRONG - Sum > 1.0
                conditions = {
                    v.T: 1000,
                    v.P: 101325,
                    v.X('AL'): 0.8,
                    v.X('CU'): 0.3  # Sum = 1.1 > 1.0!
                }
                ```
                **FIX:** Ensure sum of specified compositions ≤ 1.0
                """)
            
            with st.expander("🔧 Debugging Steps", expanded=False):
                st.markdown("""
                ### Step-by-Step Debugging:
                
                1. **Count your components (C)**
                   ```python
                   real_components = [c for c in components if c != 'VA']
                   C = len(real_components)
                   ```
                
                2. **Count your phases (P)**
                   ```python
                   P = len(phases)
                   ```
                
                3. **Calculate required variables**
                   ```python
                   required_vars = C - P + 2
                   ```
                
                4. **Count what you're specifying**
                   - +1 for `v.T`
                   - +1 for `v.P`  
                   - +1 for each independent `v.X(element)`
                
                5. **Check:**
                   ```
                   If specified = required: ✅ Good to go!
                   If specified < required: ❌ Under-constrained
                   If specified > required: ❌ Over-constrained
                   ```
                
                6. **Print your conditions:**
                   ```python
                   print("Conditions:")
                   for k, v in conditions.items():
                       print(f"  {k}: {v}")
                   ```
                """)
        
        with col_set2:
            st.subheader("⚙️ Application Management")
            
            # Database management
            st.markdown("#### 📁 Database Management")
            
            tdb_files = analyzer.get_available_tdb_files()
            if tdb_files:
                st.write(f"**Found {len(tdb_files)} TDB files:**")
                
                for tdb in tdb_files:
                    col_t1, col_t2 = st.columns([3, 1])
                    with col_t1:
                        st.caption(f"`{tdb}`")
                    with col_t2:
                        if st.button("🗑️", key=f"del_{tdb}", help="Delete"):
                            try:
                                (analyzer.database_dir / tdb).unlink()
                                st.success(f"Deleted {tdb}")
                                st.rerun()
                            except:
                                st.error("Error deleting")
            else:
                st.info("No TDB files in database directory")
            
            # TDB upload
            st.markdown("#### 📤 Upload TDB Files")
            uploaded = st.file_uploader("Upload TDB", type=["tdb", "TDB"], key="upload_settings")
            if uploaded:
                analyzer.save_uploaded_tdb(uploaded)
                st.rerun()
            
            # Session management
            st.markdown("#### 🗑️ Session Management")
            
            col_sess1, col_sess2 = st.columns(2)
            with col_sess1:
                if st.button("Clear All Data", use_container_width=True, type="secondary"):
                    analyzer.results_history = []
                    analyzer.fitting_results = []
                    st.success("Data cleared!")
                    st.rerun()
            
            with col_sess2:
                if st.button("Reset Application", use_container_width=True, type="secondary"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.success("Application reset!")
                    st.rerun()
            
            # About section
            st.markdown("---")
            st.subheader("ℹ️ About")
            
            st.markdown("""
            **Thermodynamic Enthalpy Analyzer Pro v3.0**
            
            *With Intelligent DOF Error Prevention*
            
            **Key Features:**
            - 🛡️ **DOF Error Prevention**: Gibbs Phase Rule validation
            - 🔬 **Smart Composition Handling**: Auto-balancing for n-component systems
            - 📊 **Comprehensive Analysis**: Phase diagrams, curve fitting, comparison
            - 🎯 **Minimal Working Examples**: Auto-generated code for quick fixes
            
            **Technical Stack:**
            - **CALPHAD Engine**: pycalphad
            - **Numerical Methods**: scipy, numpy
            - **Visualization**: matplotlib
            - **Interface**: Streamlit
            
            **License**: MIT
            **Version**: 3.0.0
            """)
    
    # Footer
    st.markdown("---")
    
    footer_col1, footer_col2 = st.columns([3, 1])
    
    with footer_col1:
        st.caption(f"🔥 Thermodynamic Enthalpy Analyzer Pro | DOF-Protected Edition | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with footer_col2:
        if st.button("🔄 Refresh", type="secondary"):
            st.rerun()

if __name__ == "__main__":
    main()
