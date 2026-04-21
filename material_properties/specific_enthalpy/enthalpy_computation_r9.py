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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
warnings.filterwarnings('ignore')

# Initialize session state for caching
if 'simulation_cache' not in st.session_state:
    st.session_state.simulation_cache = {}
if 'widget_state' not in st.session_state:
    st.session_state.widget_state = {}

# Page configuration with enhanced settings
st.set_page_config(
    page_title="🔥 Thermodynamic Enthalpy Analyzer Pro",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/thermo-analyzer',
        'Report a bug': "https://github.com/thermo-analyzer/issues",
        'About': "### Thermodynamic Enthalpy Analyzer Pro v3.0\nAdvanced thermodynamic analysis tool for materials science"
    }
)

# Custom CSS for professional styling with enhanced features
st.markdown("""
<style>
.main-header {
    font-size: 3.2rem;
    background: linear-gradient(90deg, #1E88E5, #E53935, #FF9800, #43A047);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1.5rem;
    font-weight: 800;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    letter-spacing: -0.5px;
}
.sub-header {
    font-size: 1.8rem;
    color: #2C3E50;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    border-bottom: 3px solid #1E88E5;
    padding-bottom: 0.5rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    padding: 0 10px;
}
.stTabs [data-baseweb="tab"] {
    height: 55px;
    white-space: pre-wrap;
    background-color: #f8f9fa;
    border-radius: 8px 8px 0 0;
    border: 1px solid #dee2e6;
    padding: 12px 20px;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);
    color: white;
    box-shadow: 0 6px 12px rgba(30, 136, 229, 0.4);
    transform: translateY(-2px);
    border: none;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background-color: #e9ecef;
    transform: translateY(-1px);
}
.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    border-left: 6px solid #1E88E5;
    transition: transform 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}
.download-section {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 25px;
    border-radius: 12px;
    margin-top: 25px;
    border: 2px solid #90caf9;
    box-shadow: 0 4px 12px rgba(144, 202, 249, 0.3);
}
.phase-container {
    max-height: 300px;
    overflow-y: auto;
    border: 2px solid #e0e0e0;
    padding: 15px;
    border-radius: 10px;
    background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
    scrollbar-width: thin;
    scrollbar-color: #1E88E5 #f0f0f0;
}
.phase-container::-webkit-scrollbar {
    width: 8px;
}
.phase-container::-webkit-scrollbar-track {
    background: #f0f0f0;
    border-radius: 4px;
}
.phase-container::-webkit-scrollbar-thumb {
    background: #1E88E5;
    border-radius: 4px;
}
.success-box {
    background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
    padding: 20px;
    border-radius: 10px;
    border-left: 6px solid #4CAF50;
    margin: 15px 0;
    box-shadow: 0 3px 8px rgba(76, 175, 80, 0.2);
}
.warning-box {
    background: linear-gradient(135deg, #fff8e1 0%, #fff3cd 100%);
    padding: 20px;
    border-radius: 10px;
    border-left: 6px solid #FFC107;
    margin: 15px 0;
    box-shadow: 0 3px 8px rgba(255, 193, 7, 0.2);
}
.info-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 20px;
    border-radius: 10px;
    border-left: 6px solid #2196F3;
    margin: 15px 0;
    box-shadow: 0 3px 8px rgba(33, 150, 243, 0.2);
}
.gradient-bg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 25px;
    box-shadow: 0 6px 15px rgba(102, 126, 234, 0.4);
}
.customization-box {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    padding: 20px;
    border-radius: 10px;
    border: 2px solid #e0e0e0;
    margin: 15px 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}
.column-selector {
    background: linear-gradient(135deg, #f0f8ff 0%, #e6f7ff 100%);
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
    border-left: 6px solid #4CAF50;
    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.2);
}
.plot-container {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    margin: 20px 0;
    border: 1px solid #e0e0e0;
}
.session-info {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #dee2e6;
    font-size: 0.9rem;
    color: #6c757d;
}
.data-table {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border: 1px solid #e0e0e0;
}
.stButton > button {
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    transition: all 0.3s ease;
    border: none;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
.stNumberInput > div > div > input,
.stTextInput > div > div > input,
.stSelectbox > div > div > div {
    border-radius: 8px !important;
    border: 2px solid #e0e0e0 !important;
    transition: all 0.3s ease !important;
}
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus,
.stSelectbox > div > div > div:focus {
    border-color: #1E88E5 !important;
    box-shadow: 0 0 0 3px rgba(30, 136, 229, 0.1) !important;
}
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #1E88E5, #0D47A1) !important;
}
.tooltip-icon {
    display: inline-block;
    width: 18px;
    height: 18px;
    background-color: #6c757d;
    color: white;
    border-radius: 50%;
    text-align: center;
    line-height: 18px;
    font-size: 12px;
    margin-left: 5px;
    cursor: help;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# COMPREHENSIVE PERIODIC TABLE DATA (118 elements with complete information)
PERIODIC_TABLE = {
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
        # Initialize session state if not exists
        if 'results_history' not in st.session_state:
            st.session_state.results_history = []
        if 'fitting_results' not in st.session_state:
            st.session_state.fitting_results = []
        if 'viscosity_fitting_results' not in st.session_state:
            st.session_state.viscosity_fitting_results = []
        if 'history_thumbnails' not in st.session_state:
            st.session_state.history_thumbnails = []
        if 'plot_customizations' not in st.session_state:
            st.session_state.plot_customizations = {
                'curve_thickness': 2.5,
                'box_thickness': 1.0,
                'font_size': 12,
                'title_font_size': 14,
                'legend_font_size': 10,
                'legend_location': 'best',
                'colormap': 'viridis',
                'marker_size': 6,
                'grid_alpha': 0.3,
                'figure_width': 14,
                'figure_height': 10,
                'tick_size': 11,
                'label_fontsize': 12,
                'plot_style': 'default',
                'dpi': 150,
                'transparent_bg': False,
                'annotation_fontsize': 9
            }
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.database_dir = Path(os.path.join(script_dir, "databases"))
        self.database_dir.mkdir(exist_ok=True)
    
    @property
    def results_history(self):
        return st.session_state.results_history
    
    @results_history.setter
    def results_history(self, value):
        st.session_state.results_history = value
    
    @property
    def fitting_results(self):
        return st.session_state.fitting_results
    
    @fitting_results.setter
    def fitting_results(self, value):
        st.session_state.fitting_results = value
    
    @property
    def viscosity_fitting_results(self):
        return st.session_state.viscosity_fitting_results
    
    @viscosity_fitting_results.setter
    def viscosity_fitting_results(self, value):
        st.session_state.viscosity_fitting_results = value
    
    @property
    def history_thumbnails(self):
        return st.session_state.history_thumbnails
    
    @history_thumbnails.setter
    def history_thumbnails(self, value):
        st.session_state.history_thumbnails = value
    
    @property
    def plot_customizations(self):
        return st.session_state.plot_customizations
    
    @plot_customizations.setter
    def plot_customizations(self, value):
        st.session_state.plot_customizations = value
    
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
    
    # ========== VISCOSITY MODELING METHODS ==========
    
    def arrhenius_viscosity(self, T, A_l, E_a):
        """
        Arrhenius equation for dynamic viscosity in liquid phase
        μ = A_l * exp(E_a / (R * T))
        
        Parameters:
        T: Temperature (K)
        A_l: Pre-exponential factor (Pa·s)
        E_a: Activation energy (J/mol)
        """
        T = np.asarray(T)
        R = 8.314  # Universal gas constant (J/mol·K)
        return A_l * np.exp(E_a / (R * T))
    
    def asymptotic_viscosity(self, T, A_l, E_a, A_ls, phi_s, n=2.5):
        """
        Asymptotic viscosity model for mushy zone
        μ_∞ = μ(T) * (1 - φ_s)^(-n)
        
        Parameters:
        T: Temperature (K)
        A_l: Pre-exponential factor for liquid (Pa·s)
        E_a: Activation energy (J/mol)
        A_ls: Pre-exponential factor for mushy zone (Pa·s)
        phi_s: Solid fraction (0-1)
        n: Exponent (typically 2.5)
        """
        T = np.asarray(T)
        R = 8.314
        mu_T = A_ls * np.exp(E_a / (R * T))
        return mu_T * (1 - phi_s) ** (-n)
    
    def solid_fraction(self, T, Tm, delta_T_s=1.0, delta_T_l=2.0):
        """
        Calculate solid fraction as a piecewise linear function
        """
        T = np.asarray(T)
        phi_s = np.zeros_like(T, dtype=float)
        
        # Solid region
        mask_solid = T <= Tm - delta_T_s
        phi_s[mask_solid] = 1.0
        
        # Mushy zone
        mask_mushy = (T > Tm - delta_T_s) & (T < Tm + delta_T_l)
        if np.any(mask_mushy):
            phi_s[mask_mushy] = (Tm + delta_T_l - T[mask_mushy]) / (delta_T_s + delta_T_l)
        
        # Liquid region
        mask_liquid = T >= Tm + delta_T_l
        phi_s[mask_liquid] = 0.0
        
        return phi_s
    
    def unified_viscosity_model(self, T, A_l, E_a, A_ls, Tm, delta_T_s=1.0, delta_T_l=2.0, n=2.5, mu_solid_eff=1000.0):
        """
        Complete viscosity model across all phases
        
        Parameters:
        T: Temperature array (K)
        A_l: Pre-exponential factor for liquid (Pa·s)
        E_a: Activation energy (J/mol)
        A_ls: Pre-exponential factor for mushy zone (Pa·s)
        Tm: Melting temperature (K)
        delta_T_s: Solid side mushy zone width (K)
        delta_T_l: Liquid side mushy zone width (K)
        n: Asymptotic exponent
        mu_solid_eff: Effective viscosity in solid phase (Pa·s)
        """
        T = np.asarray(T)
        R = 8.314
        mu = np.zeros_like(T, dtype=float)
        
        # Calculate solid fraction
        phi_s = self.solid_fraction(T, Tm, delta_T_s, delta_T_l)
        
        # Liquid phase viscosity (Arrhenius)
        mu_liquid = A_l * np.exp(E_a / (R * T))
        
        # Mushy zone viscosity (asymptotic)
        mu_mushy_ref = A_ls * np.exp(E_a / (R * T))
        mu_mushy = mu_mushy_ref * (1 - phi_s) ** (-n)
        
        # Combine based on phase
        for i, temp in enumerate(T):
            if temp <= Tm - delta_T_s:
                mu[i] = mu_solid_eff
            elif temp >= Tm + delta_T_l:
                mu[i] = mu_liquid[i]
            else:
                mu[i] = mu_mushy[i]
        
        return mu, phi_s
    
    def viscosity_equation_for_fitting(self, T, A_l, E_a, A_ls, Tm):
        """
        Simplified viscosity equation for curve fitting
        Uses fixed parameters for mushy zone
        """
        mu, _ = self.unified_viscosity_model(T, A_l, E_a, A_ls, Tm, 
                                              delta_T_s=1.0, delta_T_l=2.0, 
                                              n=2.5, mu_solid_eff=1000.0)
        return mu
    
    def create_viscosity_data_from_enthalpy_fit(self, Tm_from_enthalpy, T_range=None, n_points=100):
        """
        Generate viscosity data using Tm from enthalpy fitting
        Uses reference coefficients from Sn-0.7Cu solder
        """
        if T_range is None:
            T_range = (Tm_from_enthalpy - 100, Tm_from_enthalpy + 100)
        
        T = np.linspace(T_range[0], T_range[1], n_points)
        
        # Reference coefficients for Sn-0.7Cu (from the paper)
        A_l_ref = 9.79e-4  # Pa·s
        E_a_ref = 2962.997  # J/mol
        A_ls_ref = 0.091  # Pa·s
        
        # Generate viscosity data
        mu, phi_s = self.unified_viscosity_model(T, A_l_ref, E_a_ref, A_ls_ref, Tm_from_enthalpy)
        
        df = pd.DataFrame({
            'Temperature_K': T,
            'Viscosity_Pa_s': mu,
            'Solid_Fraction': phi_s
        })
        
        return df, (A_l_ref, E_a_ref, A_ls_ref)
    
    def fit_viscosity_data(self, T_data, mu_data, Tm_fixed=None, initial_guess=None):
        """
        Fit viscosity data to Arrhenius + asymptotic model
        
        Parameters:
        T_data: Temperature data (K)
        mu_data: Viscosity data (Pa·s)
        Tm_fixed: Fixed melting temperature from enthalpy fit (optional)
        initial_guess: [A_l, E_a, A_ls, Tm] or [A_l, E_a, A_ls] if Tm_fixed
        """
        R = 8.314
        
        if Tm_fixed is not None:
            # Fit with fixed Tm
            def viscosity_fit_func(T, A_l, E_a, A_ls):
                return self.viscosity_equation_for_fitting(T, A_l, E_a, A_ls, Tm_fixed)
            
            if initial_guess is None:
                initial_guess = [1e-3, 3000.0, 0.1]
            
            lower_bounds = [1e-6, 100.0, 1e-6]
            upper_bounds = [1e-1, 1e5, 10.0]
            
            fit_params, pcov = curve_fit(
                viscosity_fit_func,
                T_data,
                mu_data,
                p0=initial_guess,
                bounds=(lower_bounds, upper_bounds),
                method='trf',
                maxfev=5000
            )
            
            A_l_fit, E_a_fit, A_ls_fit = fit_params
            Tm_fit = Tm_fixed
            
        else:
            # Fit with Tm as parameter
            def viscosity_fit_func_full(T, A_l, E_a, A_ls, Tm):
                return self.viscosity_equation_for_fitting(T, A_l, E_a, A_ls, Tm)
            
            if initial_guess is None:
                Tm_guess = np.median(T_data)
                initial_guess = [1e-3, 3000.0, 0.1, Tm_guess]
            
            lower_bounds = [1e-6, 100.0, 1e-6, T_data.min()]
            upper_bounds = [1e-1, 1e5, 10.0, T_data.max()]
            
            fit_params, pcov = curve_fit(
                viscosity_fit_func_full,
                T_data,
                mu_data,
                p0=initial_guess,
                bounds=(lower_bounds, upper_bounds),
                method='trf',
                maxfev=5000
            )
            
            A_l_fit, E_a_fit, A_ls_fit, Tm_fit = fit_params
        
        # Calculate statistics
        mu_pred = self.viscosity_equation_for_fitting(T_data, A_l_fit, E_a_fit, A_ls_fit, Tm_fit)
        residuals = mu_data - mu_pred
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((mu_data - np.mean(mu_data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Calculate confidence intervals
        if pcov is not None:
            perr = np.sqrt(np.diag(pcov))
            confidence_intervals = []
            for i in range(len(fit_params)):
                ci = 1.96 * perr[i]
                confidence_intervals.append((fit_params[i] - ci, fit_params[i] + ci))
        else:
            confidence_intervals = None
        
        return {
            'A_l': A_l_fit,
            'E_a': E_a_fit,
            'A_ls': A_ls_fit,
            'Tm': Tm_fit,
            'r_squared': r_squared,
            'rmse': rmse,
            'residuals': residuals,
            'confidence_intervals': confidence_intervals,
            'pcov': pcov
        }
    
    def create_thumbnail(self, fig, size=(300, 200)):
        """Create thumbnail image from matplotlib figure"""
        try:
            thumb_fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
            for ax in fig.axes:
                new_ax = thumb_fig.add_subplot(111)
                new_ax.set_xlabel(ax.get_xlabel(), fontsize=8)
                new_ax.set_ylabel(ax.get_ylabel(), fontsize=8)
                new_ax.set_title(ax.get_title(), fontsize=10)
                for line in ax.lines:
                    xdata = line.get_xdata()
                    ydata = line.get_ydata()
                    new_line, = new_ax.plot(xdata, ydata,
                                          color=line.get_color(),
                                          linewidth=line.get_linewidth()/2,
                                          linestyle=line.get_linestyle(),
                                          marker=line.get_marker(),
                                          markersize=line.get_markersize()/2)
                for collection in ax.collections:
                    offsets = collection.get_offsets()
                    if len(offsets) > 0:
                        new_ax.scatter(offsets[:, 0], offsets[:, 1],
                                     color=collection.get_facecolor(),
                                     s=collection.get_sizes()/4,
                                     alpha=collection.get_alpha())
                new_ax.grid(True, alpha=0.3)
                new_ax.tick_params(labelsize=6)
                break
            thumb_fig.tight_layout(pad=0.5)
            buf = BytesIO()
            thumb_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(thumb_fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()
        except Exception as e:
            st.warning(f"Could not create thumbnail: {str(e)}")
            return None
    
    def format_dat_file(self, df, composition, metadata=None, columns=None):
        """Format data in DAT file format with headers and customizable columns"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dat_lines = [
            "# Enthalpy Data File",
            f"# Generated: {timestamp}",
            f"# Application: Thermodynamic Enthalpy Analyzer Pro v3.0",
            f"# Composition: {', '.join([f'{e}={f:.6f}' for e, f in composition.items()])}"
        ]
        
        if metadata:
            for key, value in metadata.items():
                dat_lines.append(f"# {key}: {value}")
        
        dat_lines.append("#" + "-"*80)
        
        if columns is None:
            columns = ['Temperature_K', 'Enthalpy_J_mol', 'Enthalpy_J_kg']
        
        header_mapping = {
            'Temperature_K': 'Temperature(K)',
            'Enthalpy_J_mol': 'Enthalpy(J/mol)',
            'Enthalpy_J_kg': 'Specific_Enthalpy(J/kg)',
            'Heat_Capacity_J_per_mol_K': 'Heat_Capacity(J/(mol·K))',
            'Phase': 'Phase'
        }
        
        header_parts = [header_mapping.get(col, col) for col in columns]
        dat_lines.append("# " + "  ".join([f"{h:<20}" for h in header_parts]))
        dat_lines.append("#" + "-"*80)
        
        for _, row in df.iterrows():
            line_parts = []
            for col in columns:
                if col in df.columns:
                    value = row[col]
                    if col == 'Temperature_K':
                        line_parts.append(f"{value:15.2f}")
                    elif 'Enthalpy' in col:
                        line_parts.append(f"{value:18.6f}")
                    elif 'Heat_Capacity' in col:
                        line_parts.append(f"{value:18.6f}")
                    else:
                        line_parts.append(f"{value:20}")
            dat_lines.append(" ".join(line_parts))
        
        return "\n".join(dat_lines)
    
    def create_enhanced_visualization(self, df, composition, material_name="Alloy", customizations=None, Tm=None, show_heat_capacity=True):
        """Create publication-quality dual-axis visualization with enhanced features and dynamic positioning"""
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
        figure_width = customizations.get('figure_width', 14)
        figure_height = customizations.get('figure_height', 10)
        tick_size = customizations.get('tick_size', 11)
        label_fontsize = customizations.get('label_fontsize', 12)
        plot_style = customizations.get('plot_style', 'default')
        dpi = customizations.get('dpi', 150)
        
        # Apply plot style
        if plot_style != 'default':
            plt.style.use(plot_style)
        
        # Create figure with dynamic spacing based on whether heat capacity is shown
        n_subplots = 3 if show_heat_capacity else 2
        fig, axes = plt.subplots(n_subplots, 1, figsize=(figure_width, figure_height),
                                dpi=dpi, constrained_layout=True)
        
        if n_subplots == 2:
            ax1, ax2 = axes
        else:
            ax1, ax2, ax3 = axes
        
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
        
        # Calculate statistics for annotations
        delta_H = df['Enthalpy_J_mol'].max() - df['Enthalpy_J_mol'].min()
        avg_dHdT = df['Enthalpy_J_mol'].diff().mean() / df['Temperature_K'].diff().mean()
        
        # Molar enthalpy plot with enhanced padding
        ax1.plot(df['Temperature_K'], df['Enthalpy_J_mol'],
                color=line_color, linewidth=curve_thickness,
                marker=marker, markersize=marker_size,
                markevery=max(1, len(df)//20),
                label=f'{material_name}',
                markerfacecolor='white', markeredgewidth=1.5,
                markeredgecolor=line_color)
        
        # Highlight melting temperature if provided
        if Tm is not None:
            ax1.axvline(Tm, color='red', linestyle='--', linewidth=2.5, alpha=0.8,
                       label=f'Melting Point: {Tm:.1f} K')
            ax1.axvspan(Tm-50, Tm+50, alpha=0.15, color='red', label='Melting Region')
        
        # Enhanced label placement with increased padding
        ax1.set_xlabel('Temperature (K)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
        ax1.set_ylabel('Enthalpy (J/mol)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
        ax1.set_title(f'Molar Enthalpy vs Temperature - {material_name}',
                     fontsize=title_font_size, fontweight='bold', pad=20)
        ax1.grid(True, alpha=grid_alpha, linestyle='--', which='both')
        
        # SMART legend placement to avoid overlap
        legend1 = ax1.legend(fontsize=legend_font_size, framealpha=0.95, loc='upper left',
                            bbox_to_anchor=(0.01, 0.99), shadow=True, borderpad=1,
                            fancybox=True, edgecolor='black')
        legend1.get_frame().set_linewidth(1.5)
        
        # Set tick parameters with improved size and padding
        ax1.tick_params(axis='both', which='major', labelsize=tick_size, pad=10)
        ax1.tick_params(axis='both', which='minor', labelsize=tick_size-2, pad=8)
        
        # Set box thickness
        for spine in ax1.spines.values():
            spine.set_linewidth(box_thickness)
        
        # Specific enthalpy plot with enhanced padding
        ax2.plot(df['Temperature_K'], df['Enthalpy_J_kg'],
                color=line_color, linewidth=curve_thickness,
                marker=marker, markersize=marker_size,
                markevery=max(1, len(df)//20),
                label=f'{material_name}',
                markerfacecolor='white', markeredgewidth=1.5,
                markeredgecolor=line_color)
        
        if Tm is not None:
            ax2.axvline(Tm, color='red', linestyle='--', linewidth=2.5, alpha=0.8,
                       label=f'Tm = {Tm:.1f} K')
            ax2.axvspan(Tm-50, Tm+50, alpha=0.15, color='red')
        
        # Enhanced label placement
        ax2.set_xlabel('Temperature (K)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
        ax2.set_ylabel('Specific Enthalpy (J/kg)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
        ax2.set_title(f'Specific Enthalpy vs Temperature - {material_name}',
                     fontsize=title_font_size, fontweight='bold', pad=20)
        ax2.grid(True, alpha=grid_alpha, linestyle='--', which='both')
        
        # SMART legend placement
        legend2 = ax2.legend(fontsize=legend_font_size, framealpha=0.95, loc='upper left',
                            bbox_to_anchor=(0.01, 0.99), shadow=True, borderpad=1,
                            fancybox=True, edgecolor='black')
        legend2.get_frame().set_linewidth(1.5)
        
        # Set tick parameters
        ax2.tick_params(axis='both', which='major', labelsize=tick_size, pad=10)
        ax2.tick_params(axis='both', which='minor', labelsize=tick_size-2, pad=8)
        
        # Set box thickness
        for spine in ax2.spines.values():
            spine.set_linewidth(box_thickness)
        
        # Heat capacity plot (dH/dT) if requested
        if show_heat_capacity:
            # Calculate heat capacity using finite differences
            dT = df['Temperature_K'].diff()
            dH = df['Enthalpy_J_mol'].diff()
            heat_capacity = dH / dT
            
            # Smooth the heat capacity data
            window_size = min(5, len(heat_capacity) // 10)
            if window_size > 1:
                heat_capacity_smooth = heat_capacity.rolling(window=window_size, center=True, min_periods=1).mean()
            else:
                heat_capacity_smooth = heat_capacity
            
            ax3.plot(df['Temperature_K'], heat_capacity_smooth,
                    color='green', linewidth=curve_thickness,
                    marker='s', markersize=marker_size-2,
                    markevery=max(1, len(df)//20),
                    label='Heat Capacity (dH/dT)',
                    markerfacecolor='white', markeredgewidth=1.5,
                    markeredgecolor='green')
            
            if Tm is not None:
                ax3.axvline(Tm, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
                ax3.axvspan(Tm-50, Tm+50, alpha=0.15, color='red')
            
            ax3.set_xlabel('Temperature (K)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
            ax3.set_ylabel('Heat Capacity (J/(mol·K))', fontsize=label_fontsize, fontweight='bold', labelpad=15)
            ax3.set_title(f'Heat Capacity vs Temperature - {material_name}',
                         fontsize=title_font_size, fontweight='bold', pad=20)
            ax3.grid(True, alpha=grid_alpha, linestyle='--', which='both')
            
            # Add heat capacity statistics
            max_cp = heat_capacity_smooth.max()
            min_cp = heat_capacity_smooth.min()
            avg_cp = heat_capacity_smooth.mean()
            
            ax3.legend(fontsize=legend_font_size, framealpha=0.95, loc='upper left',
                      bbox_to_anchor=(0.01, 0.99), shadow=True, borderpad=1,
                      fancybox=True, edgecolor='black')
            ax3.tick_params(axis='both', which='major', labelsize=tick_size, pad=10)
            ax3.tick_params(axis='both', which='minor', labelsize=tick_size-2, pad=8)
            
            for spine in ax3.spines.values():
                spine.set_linewidth(box_thickness)
        
        # Add composition annotation with DYNAMIC positioning
        comp_text = ', '.join([f'{e}={f:.4f}' for e, f in list(composition.items())[:4]])
        if len(composition) > 4:
            comp_text += f", ... (+{len(composition)-4} more)"
        
        # Position composition text at bottom with buffer
        fig.text(0.5, 0.01, f'Composition: {comp_text}',
                ha='center', fontsize=font_size-1, style='italic', alpha=0.9,
                fontweight='medium', bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat',
                                              alpha=0.3, edgecolor='goldenrod', linewidth=1))
        
        # Add statistics box with DYNAMIC positioning to avoid overlap
        if show_heat_capacity:
            stats_text = f"""Statistics:
• ΔH (J/mol): {delta_H:,.0f}
• Avg dH/dT: {avg_dHdT:.3f} J/(mol·K)
• Max Cₚ: {max_cp:.3f} J/(mol·K)
• Min Cₚ: {min_cp:.3f} J/(mol·K)
• Points: {len(df)}"""
        else:
            stats_text = f"""Statistics:
• ΔH (J/mol): {delta_H:,.0f}
• Avg dH/dT: {avg_dHdT:.3f} J/(mol·K)
• Points: {len(df)}"""
        
        # Position stats box in top-right corner with padding
        fig.text(0.98, 0.98, stats_text, transform=fig.transFigure,
                fontsize=font_size-1, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.7,
                         linewidth=1.5, edgecolor='navy'))
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.02, 0.98, f"Generated: {timestamp}", transform=fig.transFigure,
                fontsize=8, verticalalignment='top', alpha=0.6)
        
        # Tight layout with extra padding
        plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
        
        return fig
    
    def create_viscosity_visualization(self, df, fit_results=None, Tm_from_enthalpy=None, 
                                       customizations=None, material_name="Alloy"):
        """Create enhanced viscosity visualization with fitted curves"""
        if customizations is None:
            customizations = {}
        
        curve_thickness = customizations.get('curve_thickness', 2.5)
        box_thickness = customizations.get('box_thickness', 1.0)
        font_size = customizations.get('font_size', 12)
        title_font_size = customizations.get('title_font_size', 14)
        legend_font_size = customizations.get('legend_font_size', 10)
        colormap = customizations.get('colormap', 'viridis')
        marker_size = customizations.get('marker_size', 6)
        grid_alpha = customizations.get('grid_alpha', 0.3)
        figure_width = customizations.get('figure_width', 14)
        figure_height = customizations.get('figure_height', 10)
        tick_size = customizations.get('tick_size', 11)
        label_fontsize = customizations.get('label_fontsize', 12)
        dpi = customizations.get('dpi', 150)
        
        fig = plt.figure(figsize=(figure_width, figure_height), dpi=dpi, constrained_layout=True)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :])  # Viscosity vs T
        ax2 = fig.add_subplot(gs[1, 0])  # Solid fraction
        ax3 = fig.add_subplot(gs[1, 1])  # Log viscosity
        ax4 = fig.add_subplot(gs[2, :])  # Residuals or fit quality
        
        # Color scheme
        if colormap in plt.colormaps():
            cmap = plt.get_cmap(colormap)
        else:
            cmap = plt.cm.viridis
        
        # Plot 1: Viscosity vs Temperature
        ax1.plot(df['Temperature_K'], df['Viscosity_Pa_s'],
                color=cmap(0.3), linewidth=curve_thickness,
                marker='o', markersize=marker_size,
                label='Viscosity Data')
        
        if Tm_from_enthalpy is not None:
            ax1.axvline(Tm_from_enthalpy, color='red', linestyle='--', 
                       linewidth=2, alpha=0.8, label=f'Tm = {Tm_from_enthalpy:.1f} K')
            ax1.axvspan(Tm_from_enthalpy-2, Tm_from_enthalpy+2, 
                       alpha=0.2, color='red', label='Mushy Zone')
        
        ax1.set_xlabel('Temperature (K)', fontsize=label_fontsize, fontweight='bold')
        ax1.set_ylabel('Dynamic Viscosity (Pa·s)', fontsize=label_fontsize, fontweight='bold')
        ax1.set_title(f'Dynamic Viscosity vs Temperature - {material_name}',
                     fontsize=title_font_size, fontweight='bold')
        ax1.grid(True, alpha=grid_alpha, linestyle='--')
        ax1.legend(fontsize=legend_font_size)
        ax1.tick_params(labelsize=tick_size)
        
        # Plot 2: Solid fraction
        if 'Solid_Fraction' in df.columns:
            ax2.plot(df['Temperature_K'], df['Solid_Fraction'],
                    color=cmap(0.7), linewidth=curve_thickness,
                    marker='s', markersize=marker_size)
            ax2.set_xlabel('Temperature (K)', fontsize=label_fontsize, fontweight='bold')
            ax2.set_ylabel('Solid Fraction', fontsize=label_fontsize, fontweight='bold')
            ax2.set_title('Solid Fraction vs Temperature',
                         fontsize=title_font_size, fontweight='bold')
            ax2.grid(True, alpha=grid_alpha, linestyle='--')
            ax2.tick_params(labelsize=tick_size)
        
        # Plot 3: Log viscosity
        ax3.semilogy(df['Temperature_K'], df['Viscosity_Pa_s'],
                    color=cmap(0.5), linewidth=curve_thickness,
                    marker='^', markersize=marker_size)
        if Tm_from_enthalpy is not None:
            ax3.axvline(Tm_from_enthalpy, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax3.set_xlabel('Temperature (K)', fontsize=label_fontsize, fontweight='bold')
        ax3.set_ylabel('Dynamic Viscosity (Pa·s) [Log Scale]', fontsize=label_fontsize, fontweight='bold')
        ax3.set_title('Viscosity (Log Scale)',
                     fontsize=title_font_size, fontweight='bold')
        ax3.grid(True, alpha=grid_alpha, linestyle='--', which='both')
        ax3.tick_params(labelsize=tick_size)
        
        # Plot 4: Fitted curve if available
        if fit_results is not None:
            T_fit = np.linspace(df['Temperature_K'].min(), df['Temperature_K'].max(), 200)
            mu_fit = self.viscosity_equation_for_fitting(T_fit,
                                                         fit_results['A_l'],
                                                         fit_results['E_a'],
                                                         fit_results['A_ls'],
                                                         fit_results['Tm'])
            ax1.plot(T_fit, mu_fit, 'r-', linewidth=curve_thickness+1,
                    label=f'Fitted Curve (R²={fit_results["r_squared"]:.4f})')
            ax3.plot(T_fit, mu_fit, 'r-', linewidth=curve_thickness+1)
            
            # Residuals
            if 'residuals' in fit_results:
                ax4.scatter(df['Temperature_K'], fit_results['residuals'],
                           alpha=0.6, color='red', s=40)
                ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax4.set_xlabel('Temperature (K)', fontsize=label_fontsize, fontweight='bold')
                ax4.set_ylabel('Residuals (Pa·s)', fontsize=label_fontsize, fontweight='bold')
                ax4.set_title(f'Fit Residuals (RMSE = {fit_results["rmse"]:.2e} Pa·s)',
                             fontsize=title_font_size, fontweight='bold')
                ax4.grid(True, alpha=grid_alpha, linestyle='--')
                ax4.tick_params(labelsize=tick_size)
        
        # Set box thickness
        for ax in [ax1, ax2, ax3, ax4]:
            for spine in ax.spines.values():
                spine.set_linewidth(box_thickness)
        
        # Add statistics
        if 'Viscosity_Pa_s' in df.columns:
            stats_text = f"""Viscosity Statistics:
• Min μ: {df['Viscosity_Pa_s'].min():.2e} Pa·s
• Max μ: {df['Viscosity_Pa_s'].max():.2e} Pa·s
• Mean μ: {df['Viscosity_Pa_s'].mean():.2e} Pa·s
• Points: {len(df)}"""
            
            if fit_results is not None:
                stats_text += f"\n\nFit Quality:\n• R²: {fit_results['r_squared']:.6f}\n• RMSE: {fit_results['rmse']:.2e}"
            
            fig.text(0.98, 0.98, stats_text, transform=fig.transFigure,
                    fontsize=font_size-1, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.7,
                             linewidth=1.5, edgecolor='navy'))
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.02, 0.98, f"Generated: {timestamp}", transform=fig.transFigure,
                fontsize=8, verticalalignment='top', alpha=0.6)
        
        return fig

# Continue with the rest of the main application code...
# (The remaining code would include the main() function and all tab implementations)
# For brevity, I'll show the key additions for viscosity in the tabs section

def main():
    st.markdown('<h1 class="main-header">🔥 Thermodynamic Enthalpy Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced thermodynamic analysis tool for materials science and engineering")
    st.markdown("---")
    
    # Initialize analyzer with session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnthalpyAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Display session info
    with st.expander("📊 Session Information", expanded=False):
        col_sess1, col_sess2, col_sess3, col_sess4 = st.columns(4)
        with col_sess1:
            st.metric("Computed Results", len(analyzer.results_history))
        with col_sess2:
            st.metric("Fitting Results", len(analyzer.fitting_results))
        with col_sess3:
            st.metric("Viscosity Fits", len(analyzer.viscosity_fitting_results))
        with col_sess4:
            st.metric("Available TDB Files", len(analyzer.get_available_tdb_files()))
    
    # Create tabs with enhanced icons - ADDED VISCOSITY TAB
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔬 Enthalpy Computation",
        "📊 Curve Fitting & Analysis",
        "💧 Viscosity Modeling",  # NEW TAB
        "🔄 Multi-Material Comparison",
        "📈 Phase Diagram Analysis",
        "🎨 Visualization Customization",
        "ℹ️ Help & Settings"
    ])
    
    # ==================== TAB 1: Enthalpy Computation ====================
    with tab1:
        # ... (existing enthalpy computation code remains the same)
        st.markdown('<div class="sub-header">🔬 Enthalpy Computation from TDB Files</div>', unsafe_allow_html=True)
        # [Existing code from original file - unchanged]
        pass  # Placeholder - use original Tab 1 code
    
    # ==================== TAB 2: Curve Fitting ====================
    with tab2:
        # ... (existing curve fitting code remains the same)
        st.markdown('<div class="sub-header">📊 Curve Fitting & Analysis</div>', unsafe_allow_html=True)
        # [Existing code from original file - unchanged]
        pass  # Placeholder - use original Tab 2 code
    
    # ==================== TAB 3: VISCOSITY MODELING (NEW) ====================
    with tab3:
        st.markdown('<div class="sub-header">💧 Dynamic Viscosity Modeling</div>', unsafe_allow_html=True)
        st.markdown("Compute and fit dynamic viscosity using melting temperature from enthalpy analysis")
        
        if not analyzer.fitting_results:
            st.warning("⚠️ No enthalpy fitting results available. Please perform enthalpy curve fitting in Tab 2 first.")
            st.info("💡 The viscosity model uses the melting temperature (Tm) obtained from enthalpy fitting.")
        else:
            # Select enthalpy fit result to use Tm
            st.markdown("#### 📋 Select Enthalpy Fit Result")
            fit_options = [
                f"{i+1}. {fit['material_name']} | Tm = {fit['coefficients']['Tm']:.2f} K"
                for i, fit in enumerate(analyzer.fitting_results)
            ]
            
            selected_fit_idx = st.selectbox(
                "Choose enthalpy fit (Tm will be used for viscosity model):",
                range(len(fit_options)),
                format_func=lambda x: fit_options[x],
                key="tab3_fit_select"
            )
            
            selected_fit = analyzer.fitting_results[selected_fit_idx]
            Tm_from_enthalpy = selected_fit['coefficients']['Tm']
            material_name = selected_fit['material_name']
            composition = selected_fit['composition']
            
            st.success(f"✅ Selected: {material_name} with Tm = {Tm_from_enthalpy:.2f} K")
            
            # Viscosity computation options
            st.markdown("#### ⚙️ Viscosity Model Parameters")
            
            col_visc1, col_visc2 = st.columns(2)
            
            with col_visc1:
                st.markdown("**Temperature Range:**")
                T_start_visc = st.number_input("Start (K)", 100, 5000, 
                                               int(Tm_from_enthalpy - 50), 10,
                                               key="tab3_T_start")
                T_end_visc = st.number_input("End (K)", T_start_visc+10, 6000,
                                            int(Tm_from_enthalpy + 50), 10,
                                            key="tab3_T_end")
                T_step_visc = st.number_input("Step (K)", 1, 100, 5,
                                             key="tab3_T_step")
            
            with col_visc2:
                st.markdown("**Mushy Zone Parameters:**")
                delta_T_s = st.number_input("Solid side width (K)", 0.1, 10.0, 1.0, 0.1,
                                           key="tab3_delta_Ts")
                delta_T_l = st.number_input("Liquid side width (K)", 0.1, 10.0, 2.0, 0.1,
                                           key="tab3_delta_Tl")
                n_exponent = st.number_input("Asymptotic exponent (n)", 1.0, 5.0, 2.5, 0.1,
                                            key="tab3_n_exp")
            
            # Reference coefficients (Sn-0.7Cu)
            st.markdown("#### 🔧 Viscosity Coefficients")
            
            col_coeff1, col_coeff2 = st.columns(2)
            
            with col_coeff1:
                st.markdown("**Liquid Phase (Arrhenius):**")
                A_l = st.number_input("A_l (Pa·s)", 1e-6, 1.0, 9.79e-4, 1e-6,
                                     format="%.6f", key="tab3_A_l")
                E_a = st.number_input("E_a (J/mol)", 100.0, 100000.0, 2962.997, 10.0,
                                     key="tab3_E_a")
            
            with col_coeff2:
                st.markdown("**Mushy Zone:**")
                A_ls = st.number_input("A_ls (Pa·s)", 1e-6, 10.0, 0.091, 0.001,
                                      format="%.6f", key="tab3_A_ls")
                mu_solid_eff = st.number_input("μ_solid (Pa·s)", 10.0, 10000.0, 1000.0, 10.0,
                                              key="tab3_mu_solid")
            
            # Generate viscosity data
            if st.button("🚀 Compute Viscosity", type="primary", key="tab3_compute"):
                with st.spinner("Computing dynamic viscosity..."):
                    try:
                        T_visc = np.arange(T_start_visc, T_end_visc + T_step_visc, T_step_visc)
                        
                        # Compute viscosity
                        mu_visc, phi_s = analyzer.unified_viscosity_model(
                            T_visc, A_l, E_a, A_ls, Tm_from_enthalpy,
                            delta_T_s, delta_T_l, n_exponent, mu_solid_eff
                        )
                        
                        df_visc = pd.DataFrame({
                            'Temperature_K': T_visc,
                            'Viscosity_Pa_s': mu_visc,
                            'Solid_Fraction': phi_s
                        })
                        
                        st.success(f"✅ Viscosity computed for {len(df_visc)} temperature points")
                        
                        # Display visualization
                        fig_visc = analyzer.create_viscosity_visualization(
                            df_visc, None, Tm_from_enthalpy,
                            analyzer.plot_customizations, material_name
                        )
                        
                        with st.container():
                            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                            st.pyplot(fig_visc)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display key metrics
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("Min Viscosity", f"{mu_visc.min():.2e} Pa·s")
                        with col_m2:
                            st.metric("Max Viscosity", f"{mu_visc.max():.2e} Pa·s")
                        with col_m3:
                            st.metric("At Tm", f"{mu_visc[np.argmin(np.abs(T_visc - Tm_from_enthalpy))]:.2e} Pa·s")
                        
                        # Store for fitting
                        st.session_state['current_viscosity_data'] = {
                            'df': df_visc,
                            'Tm': Tm_from_enthalpy,
                            'material_name': material_name,
                            'composition': composition,
                            'coefficients': {
                                'A_l': A_l,
                                'E_a': E_a,
                                'A_ls': A_ls
                            }
                        }
                        
                        # Download options
                        st.markdown('<div class="download-section">', unsafe_allow_html=True)
                        st.markdown("#### 📥 Download Viscosity Data")
                        
                        csv_data = df_visc.to_csv(index=False)
                        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                        
                        st.download_button(
                            "📄 Download CSV",
                            data=csv_data,
                            file_name=f"viscosity_{material_name.replace(' ', '_')}_{timestamp_str}.csv",
                            mime="text/csv",
                            key=f"tab3_visc_csv_{timestamp_str}"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error computing viscosity: {str(e)}")
                        st.exception(e)
            
            # Viscosity fitting section
            st.markdown("---")
            st.markdown("#### 📊 Fit Viscosity Data")
            
            if 'current_viscosity_data' in st.session_state:
                visc_data = st.session_state['current_viscosity_data']
                
                fit_option = st.radio(
                    "Fitting mode:",
                    ["Use Tm from enthalpy (fixed)", "Fit Tm as parameter"],
                    horizontal=True,
                    key="tab3_fit_option"
                )
                
                col_fit1, col_fit2 = st.columns(2)
                
                with col_fit1:
                    A_l_guess = st.number_input("A_l guess", 1e-6, 1.0, 
                                               visc_data['coefficients']['A_l']*0.9, 1e-6,
                                               format="%.6f", key="tab3_A_l_guess")
                    E_a_guess = st.number_input("E_a guess", 100.0, 100000.0,
                                               visc_data['coefficients']['E_a']*0.9, 10.0,
                                               key="tab3_E_a_guess")
                
                with col_fit2:
                    A_ls_guess = st.number_input("A_ls guess", 1e-6, 10.0,
                                                visc_data['coefficients']['A_ls']*0.9, 0.001,
                                                format="%.6f", key="tab3_A_ls_guess")
                
                if st.button("🎯 Fit Viscosity Model", type="primary", key="tab3_fit_visc"):
                    with st.spinner("Fitting viscosity model..."):
                        try:
                            T_data = visc_data['df']['Temperature_K'].values
                            mu_data = visc_data['df']['Viscosity_Pa_s'].values
                            
                            if fit_option == "Use Tm from enthalpy (fixed)":
                                Tm_fixed = visc_data['Tm']
                                initial_guess = [A_l_guess, E_a_guess, A_ls_guess]
                            else:
                                Tm_fixed = None
                                initial_guess = [A_l_guess, E_a_guess, A_ls_guess, visc_data['Tm']]
                            
                            fit_results = analyzer.fit_viscosity_data(
                                T_data, mu_data, Tm_fixed, initial_guess
                            )
                            
                            st.success(f"✅ Viscosity fitting completed! R² = {fit_results['r_squared']:.6f}")
                            
                            # Display results
                            col_r1, col_r2, col_r3 = st.columns(3)
                            with col_r1:
                                st.metric("R²", f"{fit_results['r_squared']:.6f}")
                            with col_r2:
                                st.metric("RMSE", f"{fit_results['rmse']:.2e} Pa·s")
                            with col_r3:
                                st.metric("Tm (fitted)", f"{fit_results['Tm']:.2f} K")
                            
                            # Create visualization with fit
                            T_fit = np.linspace(T_data.min(), T_data.max(), 200)
                            mu_fit = analyzer.viscosity_equation_for_fitting(
                                T_fit, fit_results['A_l'], fit_results['E_a'],
                                fit_results['A_ls'], fit_results['Tm']
                            )
                            
                            df_fit = pd.DataFrame({
                                'Temperature_K': T_fit,
                                'Viscosity_Pa_s': mu_fit,
                                'Solid_Fraction': analyzer.solid_fraction(
                                    T_fit, fit_results['Tm']
                                )
                            })
                            
                            fig_fit = analyzer.create_viscosity_visualization(
                                df_fit, fit_results, fit_results['Tm'],
                                analyzer.plot_customizations, material_name
                            )
                            
                            with st.container():
                                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                                st.pyplot(fig_fit)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Display equations
                            st.markdown("#### 🧮 Viscosity Equations")
                            
                            col_eq1, col_eq2 = st.columns(2)
                            
                            with col_eq1:
                                st.markdown("**Liquid Phase (Arrhenius):**")
                                st.latex(rf"""
                                \mu_{{liquid}}(T) = {A_l_guess:.2e} \cdot \exp\left(\frac{{{E_a_guess:.1f}}}{{8.314 \cdot T}}\right)
                                """)
                            
                            with col_eq2:
                                st.markdown("**Mushy Zone (Asymptotic):**")
                                st.latex(rf"""
                                \mu_{{mushy}}(T) = \mu(T) \cdot (1 - \phi_s)^{{-{n_exponent}}}
                                """)
                            
                            # Store fitting results
                            fit_result_dict = {
                                'material_name': material_name,
                                'Tm_from_enthalpy': visc_data['Tm'],
                                'coefficients': {
                                    'A_l': fit_results['A_l'],
                                    'E_a': fit_results['E_a'],
                                    'A_ls': fit_results['A_ls'],
                                    'Tm': fit_results['Tm']
                                },
                                'statistics': {
                                    'r_squared': fit_results['r_squared'],
                                    'rmse': fit_results['rmse']
                                },
                                'composition': composition,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            analyzer.viscosity_fitting_results.append(fit_result_dict)
                            
                            # Download fitted parameters
                            st.markdown('<div class="download-section">', unsafe_allow_html=True)
                            st.markdown("#### 📥 Download Fitted Parameters")
                            
                            params_df = pd.DataFrame([{
                                'Material': material_name,
                                'A_l_Pa_s': fit_results['A_l'],
                                'E_a_J_per_mol': fit_results['E_a'],
                                'A_ls_Pa_s': fit_results['A_ls'],
                                'Tm_K': fit_results['Tm'],
                                'R_squared': fit_results['r_squared'],
                                'RMSE_Pa_s': fit_results['rmse'],
                                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }])
                            
                            st.download_button(
                                "📄 Download Parameters (CSV)",
                                data=params_df.to_csv(index=False),
                                file_name=f"viscosity_params_{material_name.replace(' ', '_')}_{timestamp_str}.csv",
                                mime="text/csv",
                                key=f"tab3_params_csv_{timestamp_str}"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"❌ Fitting error: {str(e)}")
                            st.exception(e)
            else:
                st.info("👈 Compute viscosity first to enable fitting")
    
    # ==================== TAB 4: Multi-Material Comparison ====================
    with tab4:
        # ... (existing comparison code)
        st.markdown('<div class="sub-header">🔄 Multi-Material Comparison</div>', unsafe_allow_html=True)
        pass  # Placeholder - use original Tab 3 code
    
    # ==================== TAB 5: Phase Diagram Analysis ====================
    with tab5:
        # ... (existing phase diagram code)
        st.markdown('<div class="sub-header">📈 Phase Diagram Analysis</div>', unsafe_allow_html=True)
        pass  # Placeholder - use original Tab 4 code
    
    # ==================== TAB 6: Visualization Customization ====================
    with tab6:
        # ... (existing customization code)
        st.markdown('<div class="sub-header">🎨 Visualization Customization</div>', unsafe_allow_html=True)
        pass  # Placeholder - use original Tab 5 code
    
    # ==================== TAB 7: Help & Settings ====================
    with tab7:
        # ... (existing help code)
        st.markdown('<div class="sub-header">ℹ️ Help & Application Settings</div>', unsafe_allow_html=True)
        pass  # Placeholder - use original Tab 6 code

if __name__ == "__main__":
    main()
