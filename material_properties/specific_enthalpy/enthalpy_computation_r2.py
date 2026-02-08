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
import sys
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🔥 Thermodynamic Enthalpy Analyzer Pro",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling with enhanced features
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #1E88E5, #E53935, #FF9800);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
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
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1E88E5, #0D47A1);
        color: white;
        box-shadow: 0 4px 8px rgba(30, 136, 229, 0.3);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        border-left: 4px solid #1E88E5;
    }
    .download-section {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border: 1px solid #90caf9;
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
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    .gradient-bg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .customization-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# COMPREHENSIVE PERIODIC TABLE DATA (118 elements with complete information)
PERIODIC_TABLE = {
    # Element: [Symbol, Name, AtomicNumber, MolarWeight(g/mol), Density(g/cm³), MeltingPoint(K), BoilingPoint(K), Group, Period, Block]
    'H': ['H', 'Hydrogen', 1, 1.008, 0.0000899, 14.01, 20.28, 1, 1, 's'],
    'HE': ['He', 'Helium', 2, 4.0026, 0.0001785, 0.95, 4.22, 18, 1, 's'],
    'LI': ['Li', 'Lithium', 3, 6.94, 0.534, 453.65, 1603, 1, 2, 's'],
    'BE': ['Be', 'Beryllium', 4, 9.0122, 1.85, 1560, 2742, 2, 2, 's'],
    'B': ['B', 'Boron', 5, 10.81, 2.34, 2349, 4200, 13, 2, 'p'],
    'C': ['C', 'Carbon', 6, 12.011, 2.267, 3823, 4098, 14, 2, 'p'],
    'N': ['N', 'Nitrogen', 7, 14.007, 0.001251, 63.15, 77.36, 15, 2, 'p'],
    'O': ['O', 'Oxygen', 8, 15.999, 0.001429, 54.36, 90.20, 16, 2, 'p'],
    'F': ['F', 'Fluorine', 9, 18.998, 0.001696, 53.53, 85.03, 17, 2, 'p'],
    'NE': ['Ne', 'Neon', 10, 20.180, 0.0008999, 24.56, 27.07, 18, 2, 'p'],
    'NA': ['Na', 'Sodium', 11, 22.990, 0.968, 370.87, 1156, 1, 3, 's'],
    'MG': ['Mg', 'Magnesium', 12, 24.305, 1.738, 923, 1363, 2, 3, 's'],
    'AL': ['Al', 'Aluminium', 13, 26.982, 2.70, 933.47, 2792, 13, 3, 'p'],
    'SI': ['Si', 'Silicon', 14, 28.085, 2.3296, 1687, 3538, 14, 3, 'p'],
    'P': ['P', 'Phosphorus', 15, 30.974, 1.823, 317.3, 553.7, 15, 3, 'p'],
    'S': ['S', 'Sulfur', 16, 32.06, 2.067, 388.36, 717.87, 16, 3, 'p'],
    'CL': ['Cl', 'Chlorine', 17, 35.45, 0.003214, 171.6, 239.11, 17, 3, 'p'],
    'AR': ['Ar', 'Argon', 18, 39.95, 0.001784, 83.80, 87.30, 18, 3, 'p'],
    'K': ['K', 'Potassium', 19, 39.098, 0.856, 336.53, 1032, 1, 4, 's'],
    'CA': ['Ca', 'Calcium', 20, 40.078, 1.55, 1115, 1757, 2, 4, 's'],
    'SC': ['Sc', 'Scandium', 21, 44.956, 2.985, 1814, 3109, 3, 4, 'd'],
    'TI': ['Ti', 'Titanium', 22, 47.867, 4.506, 1941, 3560, 4, 4, 'd'],
    'V': ['V', 'Vanadium', 23, 50.942, 6.11, 2183, 3680, 5, 4, 'd'],
    'CR': ['Cr', 'Chromium', 24, 51.996, 7.15, 2180, 2944, 6, 4, 'd'],
    'MN': ['Mn', 'Manganese', 25, 54.938, 7.21, 1519, 2334, 7, 4, 'd'],
    'FE': ['Fe', 'Iron', 26, 55.845, 7.874, 1811, 3134, 8, 4, 'd'],
    'CO': ['Co', 'Cobalt', 27, 58.933, 8.86, 1768, 3200, 9, 4, 'd'],
    'NI': ['Ni', 'Nickel', 28, 58.693, 8.912, 1728, 3186, 10, 4, 'd'],
    'CU': ['Cu', 'Copper', 29, 63.546, 8.96, 1357.77, 2835, 11, 4, 'd'],
    'ZN': ['Zn', 'Zinc', 30, 65.38, 7.134, 692.68, 1180, 12, 4, 'd'],
    'GA': ['Ga', 'Gallium', 31, 69.723, 5.907, 302.91, 2477, 13, 4, 'p'],
    'GE': ['Ge', 'Germanium', 32, 72.63, 5.323, 1211.40, 3106, 14, 4, 'p'],
    'AS': ['As', 'Arsenic', 33, 74.922, 5.776, 1090, 887, 15, 4, 'p'],
    'SE': ['Se', 'Selenium', 34, 78.971, 4.809, 494, 958, 16, 4, 'p'],
    'BR': ['Br', 'Bromine', 35, 79.904, 3.122, 265.8, 332.0, 17, 4, 'p'],
    'KR': ['Kr', 'Krypton', 36, 83.798, 0.003733, 115.79, 119.93, 18, 4, 'p'],
    'RB': ['Rb', 'Rubidium', 37, 85.468, 1.532, 312.46, 961, 1, 5, 's'],
    'SR': ['Sr', 'Strontium', 38, 87.62, 2.64, 1050, 1655, 2, 5, 's'],
    'Y': ['Y', 'Yttrium', 39, 88.906, 4.469, 1799, 3609, 3, 5, 'd'],
    'ZR': ['Zr', 'Zirconium', 40, 91.224, 6.506, 2128, 4682, 4, 5, 'd'],
    'NB': ['Nb', 'Niobium', 41, 92.906, 8.57, 2750, 5017, 5, 5, 'd'],
    'MO': ['Mo', 'Molybdenum', 42, 95.95, 10.22, 2896, 4912, 6, 5, 'd'],
    'TC': ['Tc', 'Technetium', 43, 98.0, 11.5, 2430, 4538, 7, 5, 'd'],
    'RU': ['Ru', 'Ruthenium', 44, 101.07, 12.37, 2607, 4423, 8, 5, 'd'],
    'RH': ['Rh', 'Rhodium', 45, 102.91, 12.41, 2237, 3968, 9, 5, 'd'],
    'PD': ['Pd', 'Palladium', 46, 106.42, 12.02, 1828.05, 3236, 10, 5, 'd'],
    'AG': ['Ag', 'Silver', 47, 107.87, 10.49, 1234.93, 2435, 11, 5, 'd'],
    'CD': ['Cd', 'Cadmium', 48, 112.41, 8.69, 594.22, 1040, 12, 5, 'd'],
    'IN': ['In', 'Indium', 49, 114.82, 7.31, 429.75, 2345, 13, 5, 'p'],
    'SN': ['Sn', 'Tin', 50, 118.71, 7.287, 505.08, 2875, 14, 5, 'p'],
    'SB': ['Sb', 'Antimony', 51, 121.76, 6.685, 903.78, 1860, 15, 5, 'p'],
    'TE': ['Te', 'Tellurium', 52, 127.60, 6.232, 722.66, 1261, 16, 5, 'p'],
    'I': ['I', 'Iodine', 53, 126.90, 4.93, 386.85, 457.4, 17, 5, 'p'],
    'XE': ['Xe', 'Xenon', 54, 131.29, 0.005887, 161.4, 165.03, 18, 5, 'p'],
    'CS': ['Cs', 'Caesium', 55, 132.91, 1.873, 301.59, 944, 1, 6, 's'],
    'BA': ['Ba', 'Barium', 56, 137.33, 3.594, 1000, 2170, 2, 6, 's'],
    'LA': ['La', 'Lanthanum', 57, 138.91, 6.145, 1193, 3737, 3, 6, 'f'],
    'CE': ['Ce', 'Cerium', 58, 140.12, 6.770, 1068, 3716, 3, 6, 'f'],
    'PR': ['Pr', 'Praseodymium', 59, 140.91, 6.773, 1208, 3793, 3, 6, 'f'],
    'ND': ['Nd', 'Neodymium', 60, 144.24, 7.007, 1297, 3347, 3, 6, 'f'],
    'PM': ['Pm', 'Promethium', 61, 145.0, 7.26, 1315, 3273, 3, 6, 'f'],
    'SM': ['Sm', 'Samarium', 62, 150.36, 7.52, 1345, 2067, 3, 6, 'f'],
    'EU': ['Eu', 'Europium', 63, 151.96, 5.243, 1099, 1802, 3, 6, 'f'],
    'GD': ['Gd', 'Gadolinium', 64, 157.25, 7.895, 1585, 3546, 3, 6, 'f'],
    'TB': ['Tb', 'Terbium', 65, 158.93, 8.229, 1629, 3503, 3, 6, 'f'],
    'DY': ['Dy', 'Dysprosium', 66, 162.50, 8.55, 1680, 2840, 3, 6, 'f'],
    'HO': ['Ho', 'Holmium', 67, 164.93, 8.795, 1734, 2993, 3, 6, 'f'],
    'ER': ['Er', 'Erbium', 68, 167.26, 9.066, 1802, 3141, 3, 6, 'f'],
    'TM': ['Tm', 'Thulium', 69, 168.93, 9.321, 1818, 2223, 3, 6, 'f'],
    'YB': ['Yb', 'Ytterbium', 70, 173.05, 6.965, 1097, 1469, 3, 6, 'f'],
    'LU': ['Lu', 'Lutetium', 71, 174.97, 9.84, 1925, 3675, 3, 6, 'f'],
    'HF': ['Hf', 'Hafnium', 72, 178.49, 13.31, 2506, 4876, 4, 6, 'd'],
    'TA': ['Ta', 'Tantalum', 73, 180.95, 16.69, 3290, 5731, 5, 6, 'd'],
    'W': ['W', 'Tungsten', 74, 183.84, 19.25, 3695, 5828, 6, 6, 'd'],
    'RE': ['Re', 'Rhenium', 75, 186.21, 21.02, 3459, 5869, 7, 6, 'd'],
    'OS': ['Os', 'Osmium', 76, 190.23, 22.59, 3306, 5285, 8, 6, 'd'],
    'IR': ['Ir', 'Iridium', 77, 192.22, 22.56, 2719, 4701, 9, 6, 'd'],
    'PT': ['Pt', 'Platinum', 78, 195.08, 21.45, 2041.4, 4098, 10, 6, 'd'],
    'AU': ['Au', 'Gold', 79, 196.97, 19.3, 1337.33, 3129, 11, 6, 'd'],
    'HG': ['Hg', 'Mercury', 80, 200.59, 13.534, 234.32, 629.88, 12, 6, 'd'],
    'TL': ['Tl', 'Thallium', 81, 204.38, 11.85, 577, 1746, 13, 6, 'p'],
    'PB': ['Pb', 'Lead', 82, 207.2, 11.34, 600.61, 2022, 14, 6, 'p'],
    'BI': ['Bi', 'Bismuth', 83, 208.98, 9.78, 544.7, 1837, 15, 6, 'p'],
    'PO': ['Po', 'Polonium', 84, 209.0, 9.196, 527, 1235, 16, 6, 'p'],
    'AT': ['At', 'Astatine', 85, 210.0, 7.0, 575, 610, 17, 6, 'p'],
    'RN': ['Rn', 'Radon', 86, 222.0, 0.00973, 202, 211.3, 18, 6, 'p'],
    'FR': ['Fr', 'Francium', 87, 223.0, 1.87, 300, 950, 1, 7, 's'],
    'RA': ['Ra', 'Radium', 88, 226.0, 5.5, 973, 2010, 2, 7, 's'],
    'AC': ['Ac', 'Actinium', 89, 227.0, 10.07, 1323, 3471, 3, 7, 'f'],
    'TH': ['Th', 'Thorium', 90, 232.04, 11.72, 2023, 5061, 3, 7, 'f'],
    'PA': ['Pa', 'Protactinium', 91, 231.04, 15.37, 1841, 4300, 3, 7, 'f'],
    'U': ['U', 'Uranium', 92, 238.03, 19.1, 1405.3, 4404, 3, 7, 'f'],
    'NP': ['Np', 'Neptunium', 93, 237.0, 20.45, 917, 4273, 3, 7, 'f'],
    'PU': ['Pu', 'Plutonium', 94, 244.0, 19.84, 912.5, 3501, 3, 7, 'f'],
    'AM': ['Am', 'Americium', 95, 243.0, 13.69, 1449, 2880, 3, 7, 'f'],
    'CM': ['Cm', 'Curium', 96, 247.0, 13.51, 1613, 3383, 3, 7, 'f'],
    'BK': ['Bk', 'Berkelium', 97, 247.0, 14.0, 1259, 2900, 3, 7, 'f'],
    'CF': ['Cf', 'Californium', 98, 251.0, 15.1, 1173, 1743, 3, 7, 'f'],
    'ES': ['Es', 'Einsteinium', 99, 252.0, 8.84, 1133, 1269, 3, 7, 'f'],
    'FM': ['Fm', 'Fermium', 100, 257.0, 9.7, 1800, 1100, 3, 7, 'f'],
    'MD': ['Md', 'Mendelevium', 101, 258.0, 10.3, 1100, 1100, 3, 7, 'f'],
    'NO': ['No', 'Nobelium', 102, 259.0, 9.9, 1100, 1100, 3, 7, 'f'],
    'LR': ['Lr', 'Lawrencium', 103, 262.0, 15.6, 1900, 1100, 3, 7, 'f'],
    'RF': ['Rf', 'Rutherfordium', 104, 267.0, 23.2, 2400, 5800, 4, 7, 'd'],
    'DB': ['Db', 'Dubnium', 105, 268.0, 29.3, 1100, 1100, 5, 7, 'd'],
    'SG': ['Sg', 'Seaborgium', 106, 269.0, 35.0, 1100, 1100, 6, 7, 'd'],
    'BH': ['Bh', 'Bohrium', 107, 270.0, 37.1, 1100, 1100, 7, 7, 'd'],
    'HS': ['Hs', 'Hassium', 108, 269.0, 40.7, 1100, 1100, 8, 7, 'd'],
    'MT': ['Mt', 'Meitnerium', 109, 278.0, 37.4, 1100, 1100, 9, 7, 'd'],
    'DS': ['Ds', 'Darmstadtium', 110, 281.0, 34.8, 1100, 1100, 10, 7, 'd'],
    'RG': ['Rg', 'Roentgenium', 111, 282.0, 28.7, 1100, 1100, 11, 7, 'd'],
    'CN': ['Cn', 'Copernicium', 112, 285.0, 23.7, 1100, 1100, 12, 7, 'd'],
    'NH': ['Nh', 'Nihonium', 113, 286.0, 16.0, 700, 1400, 13, 7, 'p'],
    'FL': ['Fl', 'Flerovium', 114, 289.0, 14.0, 340, 420, 14, 7, 'p'],
    'MC': ['Mc', 'Moscovium', 115, 290.0, 13.5, 700, 1400, 15, 7, 'p'],
    'LV': ['Lv', 'Livermorium', 116, 293.0, 12.9, 700, 1100, 16, 7, 'p'],
    'TS': ['Ts', 'Tennessine', 117, 294.0, 7.2, 700, 883, 17, 7, 'p'],
    'OG': ['Og', 'Oganesson', 118, 294.0, 5.0, 320, 350, 18, 7, 'p']
}

# Extract molar weights for compatibility
MOLAR_WEIGHTS = {element: data[3] for element, data in PERIODIC_TABLE.items()}

# Comprehensive colormap list with over 50 options
COLORMAPS = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'twilight', 'twilight_shifted', 'turbo', 'rainbow',
    'jet', 'nipy_spectral', 'gist_ncar', 'gist_rainbow',
    'hsv', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
    'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
    'cubehelix', 'brg', 'gist_heat', 'coolwarm', 'cool',
    'Wistia', 'hot', 'afmhot', 'gist_yarg', 'bone',
    'pink', 'spring', 'summer', 'autumn', 'winter',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
    'pastel1', 'pastel2', 'Paired', 'Accent', 'Dark2',
    'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20c',
    'Greys', 'Reds', 'Blues', 'Greens', 'Oranges',
    'Purples', 'RdYlBu', 'Spectral', 'PiYG', 'PRGn',
    'BrBG', 'RdGy', 'PuOr', 'Set3', 'flag_r'
]

class EnthalpyAnalyzer:
    def __init__(self):
        self.results_history = []
        self.fitting_results = []
        self.history_thumbnails = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.database_dir = Path(os.path.join(script_dir, "databases"))
        self.database_dir.mkdir(exist_ok=True)
        self.plot_customizations = {
            'curve_thickness': 2.5,
            'box_thickness': 1.0,
            'font_size': 12,
            'title_font_size': 14,
            'legend_font_size': 10,
            'legend_location': 'best',
            'colormap': 'viridis',
            'marker_size': 6,
            'grid_alpha': 0.3
        }
    
    def get_available_tdb_files(self):
        """Retrieve all TDB files from the databases directory"""
        try:
            tdb_files = []
            for ext in ["*.tdb", "*.TDB"]:
                tdb_files.extend(self.database_dir.glob(ext))
            return sorted([f.name for f in tdb_files], key=str.lower)
        except Exception as e:
            st.error(f"Error accessing databases directory: {str(e)}")
            return []
    
    def save_uploaded_tdb(self, uploaded_file):
        """Save uploaded TDB file to databases directory"""
        try:
            save_path = self.database_dir / uploaded_file.name
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
                molar_weight += fraction * 50.0
        
        if missing_elements:
            st.warning(f"Molar weights not found for elements: {', '.join(missing_elements)}. Using default 50 g/mol.")
        
        return molar_weight if molar_weight > 0 else 50.0
    
    def convert_composition(self, composition, from_type='mole', to_type='weight'):
        """Convert composition between mole fraction and weight fraction"""
        if from_type == to_type:
            return composition
        
        if from_type == 'mole' and to_type == 'weight':
            total_weight = sum(composition[el] * MOLAR_WEIGHTS.get(el.upper(), 50.0) for el in composition)
            converted = {}
            for element, fraction in composition.items():
                element_upper = element.upper()
                molar_weight = MOLAR_WEIGHTS.get(element_upper, 50.0)
                converted[element] = (fraction * molar_weight) / total_weight
            return converted
        
        elif from_type == 'weight' and to_type == 'mole':
            total_moles = sum(composition[el] / MOLAR_WEIGHTS.get(el.upper(), 50.0) for el in composition)
            converted = {}
            for element, fraction in composition.items():
                element_upper = element.upper()
                molar_weight = MOLAR_WEIGHTS.get(element_upper, 50.0)
                converted[element] = (fraction / molar_weight) / total_moles
            return converted
        
        return composition
    
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
        """Enthalpy equation for curve fitting with numerical stability"""
        T = np.asarray(T)
        sigmoid_term = DeltaHf * self.sigmoid(T - Tm, k)
        linear_term = A1 * T + A2 * np.maximum(T - Tm, 0)
        return linear_term + sigmoid_term + H298
    
    def specific_enthalpy_equation(self, T, A1, A2, Tm, DeltaHf, k, H298, molar_weight):
        """Specific enthalpy equation (J/kg)"""
        molar_enthalpy = self.enthalpy_equation(T, A1, A2, Tm, DeltaHf, k, H298)
        return molar_enthalpy / (molar_weight / 1000.0)
    
    def create_thumbnail(self, fig, size=(300, 200)):
        """Create thumbnail image from matplotlib figure"""
        try:
            fig.set_size_inches(size[0]/100, size[1]/100)
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()
        except:
            return None
    
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

def create_enhanced_visualization(df, composition, material_name="Alloy", customizations=None, Tm=None):
    """Create publication-quality dual-axis visualization with enhanced features"""
    if customizations is None:
        customizations = {}
    
    # Apply customizations
    curve_thickness = customizations.get('curve_thickness', 2.5)
    box_thickness = customizations.get('box_thickness', 1.0)
    font_size = customizations.get('font_size', 12)
    title_font_size = customizations.get('title_font_size', 14)
    legend_font_size = customizations.get('legend_font_size', 10)
    legend_location = customizations.get('legend_location', 'best')
    colormap = customizations.get('colormap', 'viridis')
    marker_size = customizations.get('marker_size', 6)
    grid_alpha = customizations.get('grid_alpha', 0.3)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=150)
    
    # Generate color based on composition hash
    color_hash = hash(str(sorted(composition.items()))) % 256 / 256.0
    if colormap in plt.colormaps():
        cmap = plt.get_cmap(colormap)
        line_color = cmap(color_hash)
    else:
        line_color = plt.cm.viridis(color_hash)
    
    # Enhanced marker styles
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D']
    marker = markers[int(color_hash * len(markers))]
    
    # Molar enthalpy plot
    ax1.plot(df['Temperature_K'], df['Enthalpy_J_mol'], 
             color=line_color, linewidth=curve_thickness, 
             marker=marker, markersize=marker_size,
             markevery=max(1, len(df)//20), 
             label=f'{material_name} (Molar)',
             markerfacecolor='white', markeredgewidth=1.5)
    
    # Highlight melting temperature if provided
    if Tm is not None:
        ax1.axvline(Tm, color='red', linestyle='--', linewidth=2.5, alpha=0.8, 
                   label=f'Melting Point: {Tm:.1f} K')
        # Add shaded region around Tm
        ax1.axvspan(Tm-50, Tm+50, alpha=0.15, color='red', label='Melting Region')
    
    ax1.set_xlabel('Temperature (K)', fontsize=font_size, fontweight='bold')
    ax1.set_ylabel('Enthalpy (J/mol)', fontsize=font_size, fontweight='bold')
    ax1.set_title(f'Molar Enthalpy vs Temperature - {material_name}', 
                  fontsize=title_font_size, fontweight='bold', pad=20)
    ax1.grid(True, alpha=grid_alpha, linestyle='--')
    ax1.legend(loc=legend_location, fontsize=legend_font_size, framealpha=0.9)
    
    # Set box thickness
    for spine in ax1.spines.values():
        spine.set_linewidth(box_thickness)
    
    # Specific enthalpy plot
    ax2.plot(df['Temperature_K'], df['Enthalpy_J_kg'], 
             color=line_color, linewidth=curve_thickness, 
             marker=marker, markersize=marker_size,
             markevery=max(1, len(df)//20), 
             label=f'{material_name} (Specific)',
             markerfacecolor='white', markeredgewidth=1.5)
    
    if Tm is not None:
        ax2.axvline(Tm, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
        ax2.axvspan(Tm-50, Tm+50, alpha=0.15, color='red')
    
    ax2.set_xlabel('Temperature (K)', fontsize=font_size, fontweight='bold')
    ax2.set_ylabel('Specific Enthalpy (J/kg)', fontsize=font_size, fontweight='bold')
    ax2.set_title(f'Specific Enthalpy vs Temperature - {material_name}', 
                  fontsize=title_font_size, fontweight='bold', pad=20)
    ax2.grid(True, alpha=grid_alpha, linestyle='--')
    ax2.legend(loc=legend_location, fontsize=legend_font_size, framealpha=0.9)
    
    # Set box thickness
    for spine in ax2.spines.values():
        spine.set_linewidth(box_thickness)
    
    # Add composition annotation
    comp_text = ', '.join([f'{e}={f:.3f}' for e, f in list(composition.items())[:4]])
    if len(composition) > 4:
        comp_text += f", ... (+{len(composition)-4} more)"
    
    fig.text(0.5, 0.02, f'Composition: {comp_text}', 
             ha='center', fontsize=font_size-2, style='italic', alpha=0.7)
    
    # Add statistics box
    stats_text = f"""
    Statistics:
    • ΔH (J/mol): {(df['Enthalpy_J_mol'].max() - df['Enthalpy_J_mol'].min()):,.0f}
    • Avg dH/dT: {(df['Enthalpy_J_mol'].diff().mean()/df['Temperature_K'].diff().mean()):.2f} J/(mol·K)
    • Points: {len(df)}
    """
    
    fig.text(0.02, 0.98, stats_text, transform=fig.transFigure,
             fontsize=font_size-2, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

def create_comparison_visualization(analyzer, selected_indices, customizations=None):
    """Create enhanced multi-material comparison visualization"""
    if not selected_indices or not analyzer.results_history:
        return None
    
    if customizations is None:
        customizations = {}
    
    # Apply customizations
    curve_thickness = customizations.get('curve_thickness', 2.5)
    box_thickness = customizations.get('box_thickness', 1.0)
    font_size = customizations.get('font_size', 12)
    title_font_size = customizations.get('title_font_size', 14)
    legend_font_size = customizations.get('legend_font_size', 10)
    legend_location = customizations.get('legend_location', 'best')
    colormap = customizations.get('colormap', 'tab10')
    grid_alpha = customizations.get('grid_alpha', 0.3)
    
    # Create figure with enhanced layout
    fig = plt.figure(figsize=(16, 12), dpi=150)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    ax1 = fig.add_subplot(gs[0, :])  # Molar enthalpy comparison
    ax2 = fig.add_subplot(gs[1, 0:2])  # Specific enthalpy comparison
    ax3 = fig.add_subplot(gs[1, 2])   # ΔH comparison bar chart
    ax4 = fig.add_subplot(gs[2, :])   # Phase fraction comparison
    
    # Get colormap
    if colormap in plt.colormaps():
        cmap = plt.get_cmap(colormap)
    else:
        cmap = plt.get_cmap('tab10')
    
    colors = cmap(np.linspace(0, 1, min(10, len(selected_indices))))
    
    # Plot data
    delta_h_values = []
    material_names = []
    melting_temps = []
    
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H']
    
    for i, idx in enumerate(selected_indices):
        if idx >= len(analyzer.results_history):
            continue
            
        result = analyzer.results_history[idx]
        data = result['data']
        name = result['name']
        
        # Molar enthalpy
        ax1.plot(data['Temperature_K'], data['Enthalpy_J_mol'],
                color=colors[i], linewidth=curve_thickness,
                marker=markers[i % len(markers)], markersize=5,
                markevery=max(1, len(data)//30),
                label=name, alpha=0.9)
        
        # Specific enthalpy
        ax2.plot(data['Temperature_K'], data['Enthalpy_J_kg'],
                color=colors[i], linewidth=curve_thickness,
                marker=markers[i % len(markers)], markersize=5,
                markevery=max(1, len(data)//30),
                label=name, alpha=0.9)
        
        # Calculate ΔH for bar chart
        delta_h = data['Enthalpy_J_mol'].max() - data['Enthalpy_J_mol'].min()
        delta_h_values.append(delta_h)
        material_names.append(name)
        
        # Check for melting temperature in fitting results
        Tm = None
        for fit_result in analyzer.fitting_results:
            if fit_result['material_name'] == name:
                Tm = fit_result['coefficients'].get('Tm')
                break
        
        melting_temps.append(Tm)
        
        # Highlight melting temperature
        if Tm is not None:
            ax1.axvline(Tm, color=colors[i], linestyle='--', alpha=0.7, linewidth=1.5)
            ax2.axvline(Tm, color=colors[i], linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Format molar enthalpy plot
    ax1.set_xlabel('Temperature (K)', fontsize=font_size, fontweight='bold')
    ax1.set_ylabel('Enthalpy (J/mol)', fontsize=font_size, fontweight='bold')
    ax1.set_title('Molar Enthalpy Comparison', fontsize=title_font_size, fontweight='bold')
    ax1.grid(True, alpha=grid_alpha, linestyle='--')
    ax1.legend(loc=legend_location, fontsize=legend_font_size, ncol=2, framealpha=0.9)
    
    # Format specific enthalpy plot
    ax2.set_xlabel('Temperature (K)', fontsize=font_size, fontweight='bold')
    ax2.set_ylabel('Specific Enthalpy (J/kg)', fontsize=font_size, fontweight='bold')
    ax2.set_title('Specific Enthalpy Comparison', fontsize=title_font_size, fontweight='bold')
    ax2.grid(True, alpha=grid_alpha, linestyle='--')
    ax2.legend(loc=legend_location, fontsize=legend_font_size, framealpha=0.9)
    
    # ΔH bar chart
    bars = ax3.barh(range(len(delta_h_values)), delta_h_values, color=colors[:len(material_names)])
    ax3.set_yticks(range(len(material_names)))
    ax3.set_yticklabels(material_names, fontsize=legend_font_size)
    ax3.set_xlabel('ΔH (J/mol)', fontsize=font_size, fontweight='bold')
    ax3.set_title('Total Enthalpy Change', fontsize=title_font_size, fontweight='bold')
    ax3.grid(True, alpha=grid_alpha, axis='x', linestyle='--')
    
    # Add value labels on bars
    for bar, value in zip(bars, delta_h_values):
        ax3.text(value, bar.get_y() + bar.get_height()/2, 
                f' {value:,.0f}', va='center', fontsize=legend_font_size-1)
    
    # Melting temperature comparison
    valid_temps = [(name, temp) for name, temp in zip(material_names, melting_temps) if temp is not None]
    if valid_temps:
        names, temps = zip(*valid_temps)
        bars_tm = ax4.bar(range(len(temps)), temps, color=colors[:len(temps)])
        ax4.set_xticks(range(len(temps)))
        ax4.set_xticklabels(names, rotation=45, fontsize=legend_font_size)
        ax4.set_ylabel('Melting Temperature (K)', fontsize=font_size, fontweight='bold')
        ax4.set_title('Melting Temperature Comparison (from Fitted Equation)', 
                     fontsize=title_font_size, fontweight='bold')
        ax4.grid(True, alpha=grid_alpha, axis='y', linestyle='--')
        
        # Add value labels
        for bar, temp in zip(bars_tm, temps):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{temp:.1f} K', ha='center', va='bottom', fontsize=legend_font_size-1)
    else:
        ax4.text(0.5, 0.5, 'No melting temperature data available.\nPerform curve fitting first.',
                ha='center', va='center', transform=ax4.transAxes, fontsize=font_size)
        ax4.set_title('Melting Temperature Comparison', fontsize=title_font_size, fontweight='bold')
    
    # Set box thickness for all axes
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_linewidth(box_thickness)
    
    plt.suptitle('Multi-Material Enthalpy Comparison Dashboard', 
                fontsize=title_font_size+2, fontweight='bold', y=0.98)
    
    return fig

def create_interactive_plotly_visualization(df, composition, material_name, fitted_params=None):
    """Create interactive Plotly visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Molar Enthalpy vs Temperature', 
                       'Specific Enthalpy vs Temperature',
                       'Temperature Derivative',
                       'Phase Diagram'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # Molar enthalpy
    fig.add_trace(
        go.Scatter(
            x=df['Temperature_K'],
            y=df['Enthalpy_J_mol'],
            mode='lines+markers',
            name='Molar Enthalpy',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            hovertemplate='Temp: %{x:.1f} K<br>Enthalpy: %{y:,.0f} J/mol<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Specific enthalpy
    fig.add_trace(
        go.Scatter(
            x=df['Temperature_K'],
            y=df['Enthalpy_J_kg'],
            mode='lines+markers',
            name='Specific Enthalpy',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            hovertemplate='Temp: %{x:.1f} K<br>Enthalpy: %{y:,.0f} J/kg<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Temperature derivative
    if len(df) > 1:
        dT = df['Temperature_K'].diff()
        dH = df['Enthalpy_J_mol'].diff()
        dHdT = dH / dT
        dHdT.iloc[0] = dHdT.iloc[1]  # Handle NaN
        
        fig.add_trace(
            go.Scatter(
                x=df['Temperature_K'],
                y=dHdT,
                mode='lines',
                name='dH/dT',
                line=dict(color='green', width=2),
                hovertemplate='Temp: %{x:.1f} K<br>dH/dT: %{y:.2f} J/(mol·K)<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'Enthalpy Analysis - {material_name}',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(title_text='Temperature (K)', row=1, col=1)
    fig.update_yaxes(title_text='Enthalpy (J/mol)', row=1, col=1)
    fig.update_xaxes(title_text='Temperature (K)', row=1, col=2)
    fig.update_yaxes(title_text='Enthalpy (J/kg)', row=1, col=2)
    fig.update_xaxes(title_text='Temperature (K)', row=2, col=1)
    fig.update_yaxes(title_text='dH/dT (J/(mol·K))', row=2, col=1)
    
    # Add melting temperature if available
    if fitted_params and 'Tm' in fitted_params:
        Tm = fitted_params['Tm']
        fig.add_vline(x=Tm, line_dash="dash", line_color="red", 
                     annotation_text=f"Tm = {Tm:.1f} K", 
                     annotation_position="top right",
                     row=1, col=1)
        fig.add_vline(x=Tm, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_vline(x=Tm, line_dash="dash", line_color="red", row=2, col=1)
    
    return fig

def main():
    st.markdown('<h1 class="main-header">🔥 Thermodynamic Enthalpy Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive tool for thermodynamic calculations, curve fitting, and multi-material comparison")
    st.markdown("---")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnthalpyAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Create tabs with enhanced icons
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔬 Enthalpy Computation",
        "📊 Curve Fitting & Analysis", 
        "🔄 Multi-Material Comparison",
        "🎨 Visualization Customization",
        "ℹ️ Help & Settings"
    ])
    
    # ==================== TAB 1: Enthalpy Computation ====================
    with tab1:
        st.header("🔬 Enthalpy Computation from TDB Files")
        
        # File selection section
        st.subheader("📁 TDB File Selection")
        col_file1, col_file2 = st.columns([1.2, 1])
        
        with col_file1:
            available_tdb_files = analyzer.get_available_tdb_files()
            
            if available_tdb_files:
                tdb_source = st.radio(
                    "Source:",
                    ["Select from database directory", "Upload new TDB file"],
                    horizontal=True,
                    key="tdb_source"
                )
            else:
                st.info(f"No TDB files found in '{analyzer.database_dir}'. Please upload a file.")
                tdb_source = "Upload new TDB file"
            
            tdb_path = None
            
            if tdb_source == "Select from database directory" and available_tdb_files:
                selected_file = st.selectbox(
                    f"Available TDB files in '{analyzer.database_dir}' directory:",
                    available_tdb_files,
                    help="Select a thermodynamic database file"
                )
                tdb_path = str(analyzer.database_dir / selected_file)
                st.success(f"✓ Selected: **{selected_file}**")
                
                if os.path.exists(tdb_path):
                    file_size = os.path.getsize(tdb_path) / 1024
                    st.caption(f"File size: {file_size:.1f} KB")
            
            else:
                uploaded_file = st.file_uploader(
                    "Upload TDB file",
                    type=["tdb", "TDB"],
                    help="Upload a thermodynamic database file (.tdb)",
                    key="uploader1"
                )
                
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.tdb') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tdb_path = tmp_file.name
                    
                    st.success(f"✓ Uploaded: **{uploaded_file.name}**")
                    
                    if st.checkbox("💾 Save to 'databases' directory for future use", value=True):
                        saved_path = analyzer.save_uploaded_tdb(uploaded_file)
                        if saved_path:
                            st.info(f"Saved to: `{saved_path}`")
        
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
            
            # Enhanced composition input
            st.markdown("**Composition Input:**")
            fraction_type = st.radio(
                "Fraction type:",
                ["Mole Fraction", "Weight Fraction"],
                horizontal=True,
                key="comp_type_tab1"
            )
            
            composition = {}
            remaining = 1.0
            
            for i, element in enumerate(selected_elements[:-1]):
                col1, col2 = st.columns([3, 1])
                with col1:
                    fraction = st.slider(
                        f"{element} {fraction_type}",
                        0.0, 1.0, 
                        0.0 if i > 0 else 0.33,
                        0.01,
                        key=f"comp_slider_{element}"
                    )
                with col2:
                    fraction = st.number_input(
                        f"Value",
                        0.0, 1.0,
                        fraction,
                        0.01,
                        key=f"comp_num_{element}"
                    )
                
                composition[element] = fraction
                remaining -= fraction
            
            # Last element
            last_element = selected_elements[-1]
            composition[last_element] = max(0.0, remaining)
            
            st.markdown(f"""
            <div class="info-box">
            <strong>Auto-calculated:</strong> {last_element} {fraction_type} = {composition[last_element]:.4f}
            <br><strong>Total:</strong> {sum(composition.values()):.4f}
            </div>
            """, unsafe_allow_html=True)
            
            # Convert to mole fraction if needed for calculation
            if fraction_type == "Weight Fraction":
                composition_mole = analyzer.convert_composition(composition, 'weight', 'mole')
            else:
                composition_mole = composition.copy()
        
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
            
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                output_quantity = st.selectbox(
                    "Output quantity:",
                    ["HM (Molar Enthalpy)", "GM (Gibbs Energy)", "SM (Entropy)", "CP (Heat Capacity)"],
                    help="Select thermodynamic quantity to calculate"
                )
                
                output_map = {
                    "HM (Molar Enthalpy)": "HM",
                    "GM (Gibbs Energy)": "GM",
                    "SM (Entropy)": "SM",
                    "CP (Heat Capacity)": "CP"
                }
                output_key = output_map[output_quantity]
        
        # Compute button
        st.markdown("---")
        if st.button("🚀 Compute Enthalpy", type="primary", use_container_width=True):
            with st.spinner("🔄 Performing equilibrium calculation... This may take a moment"):
                try:
                    conditions = {
                        v.T: (T_start, T_end, T_step),
                        v.P: P
                    }
                    
                    # Add composition conditions
                    elements_with_composition = list(composition_mole.keys())
                    
                    for i, element in enumerate(elements_with_composition[:-1]):
                        if composition_mole[element] > 0:
                            conditions[v.X(element)] = composition_mole[element]
                    
                    elements_with_va = selected_elements + ['VA']
                    
                    eq_result = equilibrium(
                        dbf,
                        elements_with_va,
                        selected_phases,
                        conditions,
                        output=output_key,
                        verbose=False,
                        broadcast=False
                    )
                    
                    # Extract results
                    T_values = eq_result.T.values.flatten()
                    result_values = eq_result[output_key].values.flatten()
                    
                    result_df = pd.DataFrame({
                        'Temperature_K': T_values,
                        f'{output_key}_value': result_values
                    })
                    
                    result_df = result_df.dropna().sort_values('Temperature_K').reset_index(drop=True)
                    
                    if len(result_df) == 0:
                        st.error("❌ Calculation returned no valid results. Try adjusting parameters.")
                        st.stop()
                    
                    # Rename column for consistency
                    result_df = result_df.rename(columns={f'{output_key}_value': 'Enthalpy_J_mol'})
                    
                    # Add specific enthalpy
                    result_df = analyzer.convert_to_specific_enthalpy(result_df, composition_mole)
                    
                    # Store results
                    material_name = "-".join([f"{e}{composition_mole[e]:.2f}" for e in selected_elements[:3]])
                    if len(selected_elements) > 3:
                        material_name += f"-{len(selected_elements)-3}more"
                    
                    result_info = {
                        'name': material_name,
                        'composition': composition_mole,
                        'composition_type': fraction_type,
                        'data': result_df,
                        'phases': selected_phases,
                        'tdb_file': os.path.basename(tdb_path),
                        'temperature_range': (T_start, T_end, T_step),
                        'output_quantity': output_key,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    analyzer.results_history.append(result_info)
                    
                    # Create visualization
                    fig = create_enhanced_visualization(
                        result_df, 
                        composition_mole, 
                        material_name,
                        analyzer.plot_customizations
                    )
                    
                    # Create thumbnail
                    thumbnail = analyzer.create_thumbnail(fig)
                    if thumbnail:
                        analyzer.history_thumbnails.append({
                            'name': material_name,
                            'thumbnail': thumbnail,
                            'timestamp': result_info['timestamp']
                        })
                    
                    st.success(f"✅ Calculation completed! Generated {len(result_df)} data points.")
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
                        csv_full = result_df.to_csv(index=False)
                        st.download_button(
                            "📄 Download Full Data (CSV)",
                            data=csv_full,
                            file_name=f"enthalpy_{material_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    
                    with col_dl2:
                        metadata = {
                            'TDB File': os.path.basename(tdb_path),
                            'Phases': ', '.join(selected_phases),
                            'Pressure (Pa)': P,
                            'Temperature Range': f"{T_start}-{T_end} K",
                            'Output Quantity': output_key,
                            'Composition Type': fraction_type
                        }
                        dat_content = analyzer.format_dat_file(result_df, composition_mole, metadata)
                        
                        st.download_button(
                            "📄 Download DAT Format (with metadata)",
                            data=dat_content,
                            file_name=f"enthalpy_{material_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.dat",
                            mime="text/plain"
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Data preview
                    with st.expander("🔍 View Complete Data Table"):
                        st.dataframe(result_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Calculation error: {str(e)}")
                    st.exception(e)
    
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
            composition_type_ref = analyzer.results_history[selected_result_idx].get('composition_type', 'Mole Fraction')
            
            # Display composition info
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.subheader("📝 Composition Reference (Auto-populated)")
            
            col_comp1, col_comp2 = st.columns(2)
            with col_comp1:
                display_type = st.radio(
                    "Display composition as:",
                    ["Mole Fraction", "Weight Fraction"],
                    horizontal=True,
                    key="display_comp_type"
                )
            
            with col_comp2:
                if display_type == "Weight Fraction" and composition_type_ref == "Mole Fraction":
                    display_comp = analyzer.convert_composition(composition_ref, 'mole', 'weight')
                elif display_type == "Mole Fraction" and composition_type_ref == "Weight Fraction":
                    display_comp = analyzer.convert_composition(composition_ref, 'weight', 'mole')
                else:
                    display_comp = composition_ref
                
                st.write("**Composition:**")
                for element, fraction in display_comp.items():
                    st.write(f"• {element}: {fraction:.4f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Store for later use
            composition_for_fitting = composition_ref
            composition_type_for_fitting = composition_type_ref
            
        else:
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
                
                col1, col2 = st.columns(2)
                with col1:
                    temp_col = st.selectbox("Temperature column", df_upload.columns)
                with col2:
                    enthalpy_col = st.selectbox("Enthalpy column (J/mol)", df_upload.columns)
                
                T_data = df_upload[temp_col].values
                H_data = df_upload[enthalpy_col].values
                material_name = "Uploaded Data"
                
                # Manual composition input
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.subheader("📝 Composition Reference (Manual Input)")
                
                col_man1, col_man2 = st.columns(2)
                with col_man1:
                    comp_type = st.radio(
                        "Input composition as:",
                        ["Mole Fraction", "Weight Fraction"],
                        horizontal=True,
                        key="manual_comp_type"
                    )
                
                with col_man2:
                    elements = st.multiselect(
                        "Select elements:",
                        sorted(PERIODIC_TABLE.keys()),
                        default=['AL', 'CU', 'NI']
                    )
                
                if elements:
                    composition_for_fitting = {}
                    cols = st.columns(len(elements))
                    for idx, element in enumerate(elements):
                        with cols[idx]:
                            fraction = st.number_input(
                                f"{element}",
                                0.0, 1.0, 
                                1.0/len(elements) if idx < len(elements)-1 else 0.0,
                                0.01,
                                key=f"manual_comp_{element}"
                            )
                            composition_for_fitting[element] = fraction
                    
                    # Calculate last element
                    if len(elements) > 1:
                        total = sum(composition_for_fitting.values())
                        if total < 1.0:
                            last_element = elements[-1]
                            composition_for_fitting[last_element] = 1.0 - total + composition_for_fitting.get(last_element, 0)
                
                composition_type_for_fitting = comp_type
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
                st.stop()
        
        # Fitting parameters section
        st.markdown("---")
        st.subheader("⚙️ Fitting Parameters")
        
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
        
        # Fit button
        st.markdown("---")
        if st.button("🎯 Perform Curve Fit", type="primary", use_container_width=True):
            with st.spinner("Fitting curve to data..."):
                try:
                    initial_guess = [A1_guess, A2_guess, Tm_guess, DeltaHf_guess, k_guess, H298_guess]
                    
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
                    
                    # Calculate molar weight for specific enthalpy equation
                    molar_weight = analyzer.calculate_alloy_molar_weight(composition_for_fitting)
                    
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
                            'data_points': len(T_data),
                            'molar_weight_g_per_mol': molar_weight
                        },
                        'composition': composition_for_fitting,
                        'composition_type': composition_type_for_fitting,
                        'temperature_range': [float(T_data.min()), float(T_data.max())],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    analyzer.fitting_results.append(fit_result)
                    
                    st.success(f"✅ Fitting completed! R² = {r_squared:.6f}, RMSE = {rmse:.2f} J/mol")
                    
                    # Create comprehensive visualization
                    fig = plt.figure(figsize=(16, 12), dpi=150)
                    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
                    
                    # Main fit plot
                    ax1 = fig.add_subplot(gs[0:2, :])
                    ax1.scatter(T_data, H_data, alpha=0.6, label='Original Data', 
                               color='#1E88E5', s=50, edgecolors='white', linewidth=1)
                    ax1.plot(T_fit, H_fit, 'r-', linewidth=3, label='Fitted Curve', alpha=0.9)
                    
                    # Highlight melting temperature with colorful style
                    ax1.axvline(Tm_fit, color='red', linestyle='--', alpha=0.9, linewidth=2.5,
                              label=f'Melting Point Tₘ = {Tm_fit:.1f} K')
                    
                    # Add shaded melting region
                    ax1.axvspan(Tm_fit-50, Tm_fit+50, alpha=0.2, color='orange', label='Melting Region')
                    
                    # Add text annotation for Tm
                    ax1.text(Tm_fit, ax1.get_ylim()[0] + 0.1*(ax1.get_ylim()[1]-ax1.get_ylim()[0]),
                            f'Tm = {Tm_fit:.1f} K', rotation=90, verticalalignment='bottom',
                            fontsize=12, fontweight='bold', color='red')
                    
                    ax1.set_xlabel('Temperature (K)', fontsize=14, fontweight='bold')
                    ax1.set_ylabel('Enthalpy (J/mol)', fontsize=14, fontweight='bold')
                    ax1.set_title(f'Enthalpy-Temperature Curve Fitting - {material_name}', 
                                 fontsize=16, fontweight='bold', pad=20)
                    ax1.grid(True, alpha=0.3, linestyle='--')
                    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
                    
                    # Residual plot
                    ax2 = fig.add_subplot(gs[2, 0])
                    ax2.scatter(T_data, residuals, alpha=0.7, color='#FF7043', s=40)
                    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=2)
                    ax2.set_xlabel('Temperature (K)', fontsize=12)
                    ax2.set_ylabel('Residuals (J/mol)', fontsize=12)
                    ax2.set_title('Residual Plot', fontsize=13, fontweight='bold')
                    ax2.grid(True, alpha=0.3, linestyle='--')
                    
                    # Specific enthalpy fit
                    ax3 = fig.add_subplot(gs[2, 1])
                    if 'Enthalpy_J_kg' in result_data.columns:
                        H_specific_data = result_data['Enthalpy_J_kg'].values
                        H_specific_fit = analyzer.specific_enthalpy_equation(T_fit, *fit_params, molar_weight)
                        ax3.scatter(T_data, H_specific_data, alpha=0.6, color='#4CAF50', s=40)
                        ax3.plot(T_fit, H_specific_fit, 'purple', linewidth=2.5, alpha=0.9)
                        ax3.axvline(Tm_fit, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                        ax3.set_xlabel('Temperature (K)', fontsize=12)
                        ax3.set_ylabel('Enthalpy (J/kg)', fontsize=12)
                        ax3.set_title('Specific Enthalpy Fit', fontsize=13, fontweight='bold')
                    else:
                        ax3.text(0.5, 0.5, 'Specific enthalpy data not available',
                                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                    ax3.grid(True, alpha=0.3, linestyle='--')
                    
                    # Coefficient summary
                    ax4 = fig.add_subplot(gs[2, 2])
                    ax4.axis('off')
                    
                    coeff_text = (
                        f"🔧 Fitted Parameters:\n\n"
                        f"• A₁ = {A1_fit:.4f} J/(mol·K)\n"
                        f"• A₂ = {A2_fit:.4f} J/(mol·K)\n"
                        f"• Tₘ = {Tm_fit:.2f} K\n"
                        f"• ΔHf = {DeltaHf_fit:,.0f} J/mol\n"
                        f"• k = {k_fit:.6f} 1/K\n"
                        f"• H₂₉₈ = {H298_fit:,.0f} J/mol\n"
                        f"• M = {molar_weight:.2f} g/mol\n\n"
                        f"📊 Goodness of Fit:\n"
                        f"• R² = {r_squared:.6f}\n"
                        f"• RMSE = {rmse:.2f} J/mol\n"
                        f"• Data Points = {len(T_data)}"
                    )
                    
                    ax4.text(0.1, 0.95, coeff_text, transform=ax4.transAxes,
                            fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=10))
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display equations
                    st.markdown("---")
                    col_eq1, col_eq2 = st.columns(2)
                    
                    with col_eq1:
                        st.subheader("🧮 Molar Enthalpy Equation (J/mol)")
                        st.latex(rf"""
                        H_{{molar}}(T) = {A1_fit:.4f} \cdot T + {A2_fit:.4f} \cdot \max(T - {Tm_fit:.2f}, 0) + 
                        {DeltaHf_fit:,.0f} \cdot \frac{{1}}{{1 + e^{{-{k_fit:.6f}(T - {Tm_fit:.2f})}}}} + {H298_fit:,.0f}
                        """)
                    
                    with col_eq2:
                        st.subheader("🧮 Specific Enthalpy Equation (J/kg)")
                        st.latex(rf"""
                        H_{{specific}}(T) = \frac{{1}}{{{molar_weight:.4f} \times 10^{{-3}}}} \times \left[{A1_fit:.4f} \cdot T + {A2_fit:.4f} \cdot \max(T - {Tm_fit:.2f}, 0) + 
                        {DeltaHf_fit:,.0f} \cdot \frac{{1}}{{1 + e^{{-{k_fit:.6f}(T - {Tm_fit:.2f})}}}} + {H298_fit:,.0f}\right]
                        """)
                    
                    # Download section
                    st.markdown('<div class="download-section">', unsafe_allow_html=True)
                    st.subheader("📥 Download Fitting Results")
                    
                    col_f1, col_f2, col_f3 = st.columns(3)
                    
                    with col_f1:
                        coeff_df = pd.DataFrame([{
                            'Material': material_name,
                            'A1_J_per_mol_K': A1_fit,
                            'A2_J_per_mol_K': A2_fit,
                            'Tm_K': Tm_fit,
                            'DeltaHf_J_per_mol': DeltaHf_fit,
                            'k_1_per_K': k_fit,
                            'H298_J_per_mol': H298_fit,
                            'Molar_Weight_g_per_mol': molar_weight,
                            'R_squared': r_squared,
                            'RMSE_J_per_mol': rmse,
                            'Data_Points': len(T_data),
                            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }])
                        
                        st.download_button(
                            "📄 Coefficients (CSV)",
                            data=coeff_df.to_csv(index=False),
                            file_name=f"fitting_coeffs_{material_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    
                    with col_f2:
                        fitted_df = pd.DataFrame({
                            'Temperature_K': T_fit,
                            'Enthalpy_Fitted_J_mol': H_fit,
                            'Enthalpy_Fitted_J_kg': analyzer.specific_enthalpy_equation(T_fit, *fit_params, molar_weight)
                        })
                        
                        st.download_button(
                            "📄 Fitted Curve (CSV)",
                            data=fitted_df.to_csv(index=False),
                            file_name=f"fitted_curve_{material_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    
                    with col_f3:
                        json_data = json.dumps(fit_result, indent=4)
                        st.download_button(
                            "📄 Full Results (JSON)",
                            data=json_data,
                            file_name=f"fitting_results_{material_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json"
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Fitting error: {str(e)}")
                    st.exception(e)
    
    # ==================== TAB 3: Multi-Material Comparison ====================
    with tab3:
        st.header("🔄 Multi-Material Comparison")
        
        if not analyzer.results_history:
            st.info("💡 No computed data available for comparison. Please perform calculations in the 'Enthalpy Computation' tab first.")
            st.stop()
        
        # Display history with thumbnails
        st.subheader("📚 Calculation History")
        
        if analyzer.history_thumbnails:
            cols = st.columns(min(4, len(analyzer.history_thumbnails)))
            for idx, thumb_info in enumerate(analyzer.history_thumbnails[:4]):
                with cols[idx % 4]:
                    st.markdown(f"**{thumb_info['name']}**")
                    st.markdown(f"<small>{thumb_info['timestamp']}</small>", unsafe_allow_html=True)
                    st.image(f"data:image/png;base64,{thumb_info['thumbnail']}", use_column_width=True)
        
        # Selection interface
        st.subheader("✅ Select Materials to Compare")
        
        selection_options = []
        for i, res in enumerate(analyzer.results_history):
            label = f"{res['name']} | {res['tdb_file']} | {len(res['data'])} pts"
            selection_options.append((i, label))
        
        selected_labels = st.multiselect(
            "Select up to 8 materials for comparison:",
            [label for _, label in selection_options],
            default=[selection_options[i][1] for i in range(min(3, len(selection_options)))],
            max_selections=8
        )
        
        if not selected_labels:
            st.warning("⚠️ Please select at least one material for comparison")
            st.stop()
        
        selected_indices = []
        for label in selected_labels:
            for idx, opt_label in selection_options:
                if opt_label == label:
                    selected_indices.append(idx)
                    break
        
        # Apply customizations from Tab 4
        customizations = analyzer.plot_customizations
        
        # Create and display comparison visualization
        with st.spinner("Generating comparison visualization..."):
            fig = create_comparison_visualization(analyzer, selected_indices, customizations)
            if fig:
                st.pyplot(fig)
        
        # Interactive Plotly visualization
        st.subheader("📈 Interactive Visualization")
        if st.checkbox("Show interactive plot (Plotly)", value=True):
            # Create interactive plot for first selected material
            if selected_indices:
                result = analyzer.results_history[selected_indices[0]]
                data = result['data']
                composition = result['composition']
                material_name = result['name']
                
                # Find fitting results for this material
                fitted_params = None
                for fit_result in analyzer.fitting_results:
                    if fit_result['material_name'] == material_name:
                        fitted_params = fit_result['coefficients']
                        break
                
                plotly_fig = create_interactive_plotly_visualization(
                    data, composition, material_name, fitted_params
                )
                st.plotly_chart(plotly_fig, use_container_width=True)
        
        # Comparison table
        st.markdown("---")
        st.subheader("📊 Comparison Summary Table")
        
        summary_data = []
        for idx in selected_indices:
            res = analyzer.results_history[idx]
            data = res['data']
            
            min_h = data['Enthalpy_J_mol'].min()
            max_h = data['Enthalpy_J_mol'].max()
            delta_h = max_h - min_h
            avg_slope = delta_h / (data['Temperature_K'].max() - data['Temperature_K'].min())
            
            # Find melting temperature from fitting results
            Tm = None
            for fit_result in analyzer.fitting_results:
                if fit_result['material_name'] == res['name']:
                    Tm = fit_result['coefficients'].get('Tm')
                    break
            
            summary_data.append({
                'Material': res['name'],
                'TDB File': res['tdb_file'],
                'Phases': ', '.join(res['phases'][:3]) + ('...' if len(res['phases']) > 3 else ''),
                'Temp Range (K)': f"{res['temperature_range'][0]}-{res['temperature_range'][1]}",
                'Data Points': len(data),
                'Min H (J/mol)': f"{min_h:,.0f}",
                'Max H (J/mol)': f"{max_h:,.0f}",
                'ΔH (J/mol)': f"{delta_h:,.0f}",
                'Avg dH/dT (J/mol·K)': f"{avg_slope:.2f}",
                'Tm (K)': f"{Tm:.1f}" if Tm else "N/A"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Download options
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
        st.subheader("📥 Download Comparison Data")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
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
            st.download_button(
                "📄 Summary Table (CSV)",
                data=summary_df.to_csv(index=False),
                file_name=f"comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== TAB 4: Visualization Customization ====================
    with tab4:
        st.header("🎨 Visualization Customization")
        st.markdown("Customize the appearance of all plots in the application.")
        
        col_cust1, col_cust2 = st.columns(2)
        
        with col_cust1:
            st.subheader("📊 Line & Marker Settings")
            
            analyzer.plot_customizations['curve_thickness'] = st.slider(
                "Curve Thickness",
                0.5, 5.0, 2.5, 0.1,
                help="Thickness of plot lines"
            )
            
            analyzer.plot_customizations['box_thickness'] = st.slider(
                "Box/Spine Thickness",
                0.5, 5.0, 1.0, 0.1,
                help="Thickness of plot borders"
            )
            
            analyzer.plot_customizations['marker_size'] = st.slider(
                "Marker Size",
                2, 15, 6, 1,
                help="Size of data point markers"
            )
            
            analyzer.plot_customizations['grid_alpha'] = st.slider(
                "Grid Transparency",
                0.0, 1.0, 0.3, 0.05,
                help="Transparency of grid lines"
            )
        
        with col_cust2:
            st.subheader("📝 Text & Legend Settings")
            
            analyzer.plot_customizations['font_size'] = st.slider(
                "Font Size",
                8, 20, 12, 1,
                help="Base font size for labels"
            )
            
            analyzer.plot_customizations['title_font_size'] = st.slider(
                "Title Font Size",
                10, 24, 14, 1,
                help="Font size for titles"
            )
            
            analyzer.plot_customizations['legend_font_size'] = st.slider(
                "Legend Font Size",
                8, 16, 10, 1,
                help="Font size for legend text"
            )
            
            analyzer.plot_customizations['legend_location'] = st.selectbox(
                "Legend Location",
                ['best', 'upper right', 'upper left', 'lower left', 'lower right',
                 'right', 'center left', 'center right', 'lower center', 
                 'upper center', 'center'],
                help="Position of the legend"
            )
        
        # Colormap selection
        st.subheader("🌈 Colormap Selection")
        col_cmap1, col_cmap2 = st.columns([2, 1])
        
        with col_cmap1:
            analyzer.plot_customizations['colormap'] = st.selectbox(
                "Select Colormap",
                COLORMAPS,
                index=COLORMAPS.index('viridis'),
                help="Color scheme for plots"
            )
        
        with col_cmap2:
            # Display colormap preview
            fig_cmap, ax_cmap = plt.subplots(figsize=(8, 1))
            cmap = plt.get_cmap(analyzer.plot_customizations['colormap'])
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            ax_cmap.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, 0, 1])
            ax_cmap.set_xticks([])
            ax_cmap.set_yticks([])
            ax_cmap.set_title(f"Preview: {analyzer.plot_customizations['colormap']}", fontsize=10)
            st.pyplot(fig_cmap)
        
        # Save and reset buttons
        st.markdown("---")
        col_save1, col_save2, col_save3 = st.columns(3)
        
        with col_save1:
            if st.button("💾 Save Customizations", use_container_width=True):
                st.success("✅ Customizations saved!")
                st.rerun()
        
        with col_save2:
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                analyzer.plot_customizations = {
                    'curve_thickness': 2.5,
                    'box_thickness': 1.0,
                    'font_size': 12,
                    'title_font_size': 14,
                    'legend_font_size': 10,
                    'legend_location': 'best',
                    'colormap': 'viridis',
                    'marker_size': 6,
                    'grid_alpha': 0.3
                }
                st.success("✅ Customizations reset to defaults!")
                st.rerun()
        
        with col_save3:
            if st.button("🎨 Apply to All Plots", use_container_width=True):
                st.info("Customizations will be applied to all new plots.")
                st.rerun()
        
        # Export/Import customizations
        st.markdown("---")
        st.subheader("📤 Export/Import Customizations")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            # Export
            json_custom = json.dumps(analyzer.plot_customizations, indent=4)
            st.download_button(
                "📄 Export Settings (JSON)",
                data=json_custom,
                file_name=f"plot_customizations_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
        
        with col_exp2:
            # Import
            uploaded_custom = st.file_uploader(
                "Import settings from JSON",
                type=['json'],
                key="upload_custom"
            )
            
            if uploaded_custom is not None:
                try:
                    imported = json.load(uploaded_custom)
                    analyzer.plot_customizations.update(imported)
                    st.success("✅ Customizations imported successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error importing customizations: {str(e)}")
    
    # ==================== TAB 5: Help & Settings ====================
    with tab5:
        st.header("ℹ️ Help & Application Settings")
        
        col_h1, col_h2 = st.columns([1, 1.2])
        
        with col_h1:
            st.subheader("📖 User Guide")
            
            with st.expander("🚀 Quick Start", expanded=True):
                st.markdown("""
                1. **Enthalpy Computation Tab**
                   - Select/upload TDB file
                   - Choose elements and set composition
                   - Define temperature range
                   - Click "Compute Enthalpy"
                
                2. **Curve Fitting Tab**
                   - Select computed data
                   - Adjust fitting parameters
                   - Get fitted equations
                
                3. **Multi-Material Comparison**
                   - Select multiple materials
                   - Compare enthalpy curves
                   - Analyze melting temperatures
                
                4. **Visualization Customization**
                   - Customize plot appearance
                   - Choose colormaps
                   - Adjust font sizes
                """)
            
            with st.expander("🔬 N-Component System", expanded=False):
                st.markdown("""
                **For n-component systems:**
                - Enter fractions for n-1 components
                - The nth component is automatically calculated
                - Fractions are normalized to sum = 1.0
                
                **Example (Ternary system A-B-C):**
                - Enter X(A) = 0.3, X(B) = 0.4
                - X(C) is automatically: 1.0 - (0.3 + 0.4) = 0.3
                
                **Conversion:**
                - Switch between mole and weight fractions
                - Automatic conversion using element molar weights
                """)
            
            with st.expander("📊 Fitting Equation", expanded=False):
                st.markdown("""
                **Molar Enthalpy Equation:**
                ```
                H(T) = A₁·T + A₂·max(T - Tₘ, 0) + ΔHf·[1/(1 + exp(-k·(T - Tₘ)))] + H₂₉₈
                ```
                
                **Parameters:**
                - A₁: Sensible heat coefficient (solid)
                - A₂: Additional coefficient (liquid)
                - Tₘ: Melting temperature
                - ΔHf: Heat of fusion
                - k: Sigmoid steepness
                - H₂₉₈: Reference enthalpy at 298K
                
                **Specific Enthalpy:**
                - Automatically calculated from molar enthalpy
                - Uses alloy molar weight
                """)
        
        with col_h2:
            st.subheader("⚙️ Application Management")
            
            # Session data status
            st.markdown("#### 📈 Current Session Status")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Computed Results", len(analyzer.results_history))
            with col_stat2:
                st.metric("Fitting Results", len(analyzer.fitting_results))
            with col_stat3:
                st.metric("TDB Files", len(analyzer.get_available_tdb_files()))
            
            # Data management
            st.markdown("#### 💾 Data Management")
            col_data1, col_data2 = st.columns(2)
            
            with col_data1:
                if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
                    analyzer.results_history = []
                    analyzer.fitting_results = []
                    analyzer.history_thumbnails = []
                    st.success("✅ All session data cleared!")
                    st.rerun()
            
            with col_data2:
                if st.button("💾 Export Session", type="secondary", use_container_width=True):
                    session_data = {
                        'results_history': analyzer.results_history,
                        'fitting_results': analyzer.fitting_results,
                        'customizations': analyzer.plot_customizations
                    }
                    json_data = json.dumps(session_data, indent=4)
                    st.download_button(
                        "📥 Download Session",
                        data=json_data,
                        file_name=f"enthalpy_analyzer_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            # Database management
            st.markdown("#### 🗄️ Database Management")
            tdb_files = analyzer.get_available_tdb_files()
            
            if tdb_files:
                st.write(f"**Found {len(tdb_files)} TDB files:**")
                
                for tdb in tdb_files:
                    col_db1, col_db2, col_db3 = st.columns([3, 1, 1])
                    with col_db1:
                        st.caption(f"`{tdb}`")
                    with col_db2:
                        file_path = analyzer.database_dir / tdb
                        if st.button("ℹ️", key=f"info_{tdb}"):
                            size_kb = os.path.getsize(file_path) / 1024
                            st.info(f"**{tdb}**\nSize: {size_kb:.1f} KB")
                    with col_db3:
                        if st.button("🗑️", key=f"delete_{tdb}"):
                            try:
                                os.remove(file_path)
                                st.success(f"Deleted {tdb}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            else:
                st.info(f"No TDB files in `{analyzer.database_dir}`")
            
            # Periodic table explorer
            st.markdown("---")
            st.subheader("🧪 Periodic Table Explorer")
            
            selected_element = st.selectbox(
                "Explore element properties:",
                sorted(PERIODIC_TABLE.keys()),
                format_func=lambda x: f"{PERIODIC_TABLE[x][0]} - {PERIODIC_TABLE[x][1]}"
            )
            
            if selected_element:
                elem_data = PERIODIC_TABLE[selected_element]
                col_elem1, col_elem2 = st.columns(2)
                with col_elem1:
                    st.markdown(f"""
                    **{elem_data[1]} ({elem_data[0]})**
                    
                    • **Atomic Number:** {elem_data[2]}
                    • **Molar Weight:** {elem_data[3]} g/mol
                    • **Density:** {elem_data[4]} g/cm³
                    """)
                with col_elem2:
                    st.markdown(f"""
                    **Physical Properties:**
                    
                    • **Melting Point:** {elem_data[5]} K
                    • **Boiling Point:** {elem_data[6]} K
                    • **Group:** {elem_data[7]}
                    • **Period:** {elem_data[8]}
                    • **Block:** {elem_data[9]}
                    """)
            
            # About section
            st.markdown("---")
            st.subheader("ℹ️ About")
            st.markdown("""
            **Thermodynamic Enthalpy Analyzer Pro v2.0**
            
            A comprehensive tool for thermodynamic calculations using CALPHAD method.
            
            **Features:**
            - TDB file processing with pycalphad
            - Advanced curve fitting with dual equations
            - Multi-material comparison with melting temperature highlights
            - Extensive visualization customization
            - Complete periodic table database
            
            **Core Libraries:**
            - pycalphad, scipy, xarray
            - matplotlib, plotly, streamlit
            
            © 2026 Thermodynamic Analysis Toolkit
            """)
    
    # Footer
    st.markdown("---")
    st.caption("🔥 Thermodynamic Enthalpy Analyzer Pro v2.0 | Powered by pycalphad & Streamlit | " + 
               datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
