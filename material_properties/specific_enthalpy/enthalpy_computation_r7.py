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

def create_enhanced_visualization(df, composition, material_name="Alloy", customizations=None, Tm=None, show_heat_capacity=True):
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

def create_comparison_visualization(analyzer, selected_indices, customizations=None):
    """Create enhanced multi-material comparison visualization with improved label placement"""
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
    figure_width = customizations.get('figure_width', 16)
    figure_height = customizations.get('figure_height', 12)
    tick_size = customizations.get('tick_size', 11)
    label_fontsize = customizations.get('label_fontsize', 12)
    dpi = customizations.get('dpi', 150)
    
    # Create figure with enhanced layout
    fig = plt.figure(figsize=(figure_width, figure_height), dpi=dpi, constrained_layout=True)
    
    # Create 2x2 grid with different sizes
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Define axes positions
    ax1 = fig.add_subplot(gs[0, :2])  # Molar enthalpy comparison
    ax2 = fig.add_subplot(gs[0, 2:])  # Specific enthalpy comparison
    ax3 = fig.add_subplot(gs[1, :2])  # ΔH comparison bar chart
    ax4 = fig.add_subplot(gs[1, 2:])  # Heat capacity comparison
    ax5 = fig.add_subplot(gs[2, :])   # Material properties summary
    
    # Get colormap
    if colormap in plt.colormaps():
        cmap = plt.get_cmap(colormap)
    else:
        cmap = plt.get_cmap('tab10')
    
    colors = cmap(np.linspace(0, 1, min(20, len(selected_indices))))
    
    # Plot data
    delta_h_values = []
    material_names = []
    melting_temps = []
    max_heat_capacities = []
    
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    
    for i, idx in enumerate(selected_indices):
        if idx >= len(analyzer.results_history):
            continue
            
        result = analyzer.results_history[idx]
        data = result['data']
        name = result['name']
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        line_style = line_styles[i % len(line_styles)]
        
        # Molar enthalpy
        ax1.plot(data['Temperature_K'], data['Enthalpy_J_mol'],
                color=color, linewidth=curve_thickness,
                linestyle=line_style,
                marker=marker, markersize=5,
                markevery=max(1, len(data)//30),
                label=name, alpha=0.9)
        
        # Specific enthalpy
        ax2.plot(data['Temperature_K'], data['Enthalpy_J_kg'],
                color=color, linewidth=curve_thickness,
                linestyle=line_style,
                marker=marker, markersize=5,
                markevery=max(1, len(data)//30),
                label=name, alpha=0.9)
        
        # Calculate ΔH for bar chart
        delta_h = data['Enthalpy_J_mol'].max() - data['Enthalpy_J_mol'].min()
        delta_h_values.append(delta_h)
        material_names.append(name)
        
        # Calculate heat capacity
        dT = data['Temperature_K'].diff()
        dH = data['Enthalpy_J_mol'].diff()
        heat_capacity = dH / dT
        max_heat_capacities.append(heat_capacity.max() if len(heat_capacity) > 0 else 0)
        
        # Check for melting temperature in fitting results
        Tm = None
        for fit_result in analyzer.fitting_results:
            if fit_result['material_name'] == name:
                Tm = fit_result['coefficients'].get('Tm')
                break
        
        melting_temps.append(Tm)
        
        # Highlight melting temperature
        if Tm is not None:
            ax1.axvline(Tm, color=color, linestyle=':', alpha=0.7, linewidth=1.5)
            ax2.axvline(Tm, color=color, linestyle=':', alpha=0.7, linewidth=1.5)
        
        # Plot heat capacity
        if len(heat_capacity) > 0:
            ax4.plot(data['Temperature_K'].iloc[1:], heat_capacity.iloc[1:],
                    color=color, linewidth=curve_thickness-0.5,
                    linestyle=line_style, alpha=0.7,
                    label=name)
    
    # Format molar enthalpy plot with improved label placement
    ax1.set_xlabel('Temperature (K)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
    ax1.set_ylabel('Enthalpy (J/mol)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
    ax1.set_title('Molar Enthalpy Comparison', fontsize=title_font_size, fontweight='bold', pad=20)
    ax1.grid(True, alpha=grid_alpha, linestyle='--')
    ax1.legend(loc='upper left', fontsize=legend_font_size-1, ncol=1, framealpha=0.9,
              shadow=True, borderpad=1, bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax1.tick_params(axis='both', labelsize=tick_size, pad=8)
    
    # Format specific enthalpy plot
    ax2.set_xlabel('Temperature (K)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
    ax2.set_ylabel('Specific Enthalpy (J/kg)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
    ax2.set_title('Specific Enthalpy Comparison', fontsize=title_font_size, fontweight='bold', pad=20)
    ax2.grid(True, alpha=grid_alpha, linestyle='--')
    ax2.legend(loc='upper left', fontsize=legend_font_size-1, ncol=1, framealpha=0.9,
              shadow=True, borderpad=1, bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax2.tick_params(axis='both', labelsize=tick_size, pad=8)
    
    # ΔH bar chart
    bars = ax3.barh(range(len(delta_h_values)), delta_h_values, 
                   color=colors[:len(material_names)], edgecolor='black', linewidth=1)
    ax3.set_yticks(range(len(material_names)))
    ax3.set_yticklabels(material_names, fontsize=legend_font_size)
    ax3.set_xlabel('ΔH (J/mol)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
    ax3.set_title('Total Enthalpy Change', fontsize=title_font_size, fontweight='bold', pad=15)
    ax3.grid(True, alpha=grid_alpha, axis='x', linestyle='--')
    ax3.tick_params(axis='both', labelsize=tick_size, pad=8)
    
    # Add value labels on bars with improved positioning
    for bar, value in zip(bars, delta_h_values):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2, 
                f' {value:,.0f}', va='center', fontsize=legend_font_size-1,
                fontweight='bold')
    
    # Heat capacity comparison
    ax4.set_xlabel('Temperature (K)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
    ax4.set_ylabel('Heat Capacity (J/(mol·K))', fontsize=label_fontsize, fontweight='bold', labelpad=15)
    ax4.set_title('Heat Capacity Comparison', fontsize=title_font_size, fontweight='bold', pad=20)
    ax4.grid(True, alpha=grid_alpha, linestyle='--')
    ax4.legend(loc='upper left', fontsize=legend_font_size-1, ncol=1, framealpha=0.9,
              shadow=True, borderpad=1, bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax4.tick_params(axis='both', labelsize=tick_size, pad=8)
    
    # Material properties summary table
    ax5.axis('off')
    
    # Create summary data
    summary_data = []
    for i, idx in enumerate(selected_indices):
        if idx < len(analyzer.results_history):
            result = analyzer.results_history[idx]
            data = result['data']
            
            delta_h = data['Enthalpy_J_mol'].max() - data['Enthalpy_J_mol'].min()
            dT = data['Temperature_K'].diff()
            dH = data['Enthalpy_J_mol'].diff()
            avg_cp = (dH / dT).mean() if len(dH) > 1 else 0
            max_cp = max_heat_capacities[i] if i < len(max_heat_capacities) else 0
            Tm = melting_temps[i]
            
            summary_data.append([
                material_names[i],
                f"{delta_h:,.0f}",
                f"{avg_cp:.2f}",
                f"{max_cp:.2f}",
                f"{Tm:.1f}" if Tm else "N/A",
                len(data)
            ])
    
    # Create table
    col_labels = ['Material', 'ΔH (J/mol)', 'Avg Cₚ', 'Max Cₚ', 'Tₘ (K)', 'Points']
    
    # Create a table in the axis
    table = ax5.table(cellText=summary_data,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(font_size-1)
    table.scale(1, 1.5)
    
    # Color header row
    for j, col in enumerate(col_labels):
        table[(0, j)].set_facecolor('#1E88E5')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data) + 1):
        if i % 2 == 0:
            for j in range(len(col_labels)):
                table[(i, j)].set_facecolor('#f8f9fa')
    
    ax5.set_title('Material Properties Summary', fontsize=title_font_size, fontweight='bold', pad=20)
    
    # Set box thickness for all axes
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_linewidth(box_thickness)
    
    plt.suptitle('Multi-Material Enthalpy Comparison Dashboard', 
                fontsize=title_font_size+4, fontweight='bold', y=0.98)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.02, 0.02, f"Generated: {timestamp}", fontsize=8, alpha=0.6)
    
    return fig

def create_interactive_plotly_visualization(df, composition, material_name, fitted_params=None):
    """Create interactive Plotly visualization with enhanced features"""
    # Create subplots with different layout
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Molar Enthalpy', 
                       'Specific Enthalpy',
                       'Heat Capacity (dH/dT)',
                       'Enthalpy Derivative',
                       'Phase Analysis',
                       'Data Statistics'),
        vertical_spacing=0.12,
        horizontal_spacing=0.15,
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'table'}]]
    )
    
    # Color scheme
    colors = px.colors.qualitative.Set1
    
    # 1. Molar enthalpy
    fig.add_trace(
        go.Scatter(
            x=df['Temperature_K'],
            y=df['Enthalpy_J_mol'],
            mode='lines+markers',
            name='Molar Enthalpy',
            line=dict(color=colors[0], width=3),
            marker=dict(size=8, symbol='circle', line=dict(width=1, color='white')),
            hovertemplate='<b>Temperature:</b> %{x:.1f} K<br><b>Enthalpy:</b> %{y:,.0f} J/mol<br><extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Specific enthalpy
    fig.add_trace(
        go.Scatter(
            x=df['Temperature_K'],
            y=df['Enthalpy_J_kg'],
            mode='lines+markers',
            name='Specific Enthalpy',
            line=dict(color=colors[1], width=3),
            marker=dict(size=8, symbol='square', line=dict(width=1, color='white')),
            hovertemplate='<b>Temperature:</b> %{x:.1f} K<br><b>Specific Enthalpy:</b> %{y:,.0f} J/kg<br><extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Heat capacity (dH/dT)
    if len(df) > 1:
        dT = df['Temperature_K'].diff()
        dH = df['Enthalpy_J_mol'].diff()
        heat_capacity = dH / dT
        
        # Smooth the data
        window_size = min(5, len(heat_capacity) // 10)
        if window_size > 1:
            heat_capacity_smooth = heat_capacity.rolling(window=window_size, center=True, min_periods=1).mean()
        else:
            heat_capacity_smooth = heat_capacity
        
        fig.add_trace(
            go.Scatter(
                x=df['Temperature_K'],
                y=heat_capacity_smooth,
                mode='lines+markers',
                name='Heat Capacity',
                line=dict(color=colors[2], width=3),
                marker=dict(size=6, symbol='diamond', line=dict(width=1, color='white')),
                hovertemplate='<b>Temperature:</b> %{x:.1f} K<br><b>Cₚ:</b> %{y:.3f} J/(mol·K)<br><extra></extra>'
            ),
            row=2, col=1
        )
    
    # 4. Enthalpy derivative (second derivative)
    if len(df) > 2:
        d2H = df['Enthalpy_J_mol'].diff().diff()
        dT2 = df['Temperature_K'].diff().diff()
        second_derivative = d2H / dT2
        
        fig.add_trace(
            go.Scatter(
                x=df['Temperature_K'].iloc[2:],
                y=second_derivative.iloc[2:],
                mode='lines',
                name='d²H/dT²',
                line=dict(color=colors[3], width=2, dash='dash'),
                hovertemplate='<b>Temperature:</b> %{x:.1f} K<br><b>d²H/dT²:</b> %{y:.3f} J/(mol·K²)<br><extra></extra>'
            ),
            row=2, col=2
        )
    
    # 5. Phase analysis (placeholder)
    fig.add_trace(
        go.Scatter(
            x=[df['Temperature_K'].min(), df['Temperature_K'].max()],
            y=[0, 1],
            mode='lines',
            name='Phase Fraction',
            line=dict(color=colors[4], width=2),
            visible='legendonly'
        ),
        row=3, col=1
    )
    
    # 6. Data statistics table
    stats_data = [
        ['Property', 'Value'],
        ['Data Points', str(len(df))],
        ['Min Temperature', f"{df['Temperature_K'].min():.1f} K"],
        ['Max Temperature', f"{df['Temperature_K'].max():.1f} K"],
        ['ΔH (molar)', f"{df['Enthalpy_J_mol'].max() - df['Enthalpy_J_mol'].min():,.0f} J/mol"],
        ['Avg dH/dT', f"{(df['Enthalpy_J_mol'].diff() / df['Temperature_K'].diff()).mean():.3f} J/(mol·K)"],
        ['Composition', ', '.join([f'{k}:{v:.3f}' for k, v in list(composition.items())[:3]])]
    ]
    
    # FIXED: Properly structured table trace
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Property', 'Value'],
                fill_color='#1E88E5',
                align='center',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[[row[0] for row in stats_data], [row[1] for row in stats_data]],
                fill_color='white',
                align='center',
                font=dict(color='black', size=11)
            ),
            columnwidth=[0.4, 0.6]
        ),
        row=3, col=2
    )
    
    # Add melting temperature line if available
    if fitted_params and 'Tm' in fitted_params:
        Tm = fitted_params['Tm']
        
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_vline(x=Tm, line_dash="dot", line_color="red", 
                            annotation_text=f"Tm = {Tm:.1f} K", 
                            annotation_position="top right",
                            row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title=dict(text=f'Enthalpy Analysis - {material_name}', 
                  font=dict(size=24, color='#2C3E50')),
        height=1000,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text='Temperature (K)', row=1, col=1)
    fig.update_yaxes(title_text='Enthalpy (J/mol)', row=1, col=1)
    fig.update_xaxes(title_text='Temperature (K)', row=1, col=2)
    fig.update_yaxes(title_text='Enthalpy (J/kg)', row=1, col=2)
    fig.update_xaxes(title_text='Temperature (K)', row=2, col=1)
    fig.update_yaxes(title_text='Cₚ (J/(mol·K))', row=2, col=1)
    fig.update_xaxes(title_text='Temperature (K)', row=2, col=2)
    fig.update_yaxes(title_text='d²H/dT² (J/(mol·K²))', row=2, col=2)
    fig.update_xaxes(title_text='Temperature (K)', row=3, col=1)
    fig.update_yaxes(title_text='Phase Fraction', row=3, col=1)
    
    return fig

def create_curve_fitting_visualization(T_data, H_data, T_fit, H_fit, fit_params, residuals, 
                                      r_squared, rmse, material_name, composition, 
                                      molar_weight, customizations=None):
    """Create comprehensive curve fitting visualization with enhanced label placement"""
    if customizations is None:
        customizations = {}
    
    # Apply customizations
    curve_thickness = customizations.get('curve_thickness', 2.5)
    box_thickness = customizations.get('box_thickness', 1.0)
    font_size = customizations.get('font_size', 12)
    title_font_size = customizations.get('title_font_size', 14)
    legend_font_size = customizations.get('legend_font_size', 10)
    colormap = customizations.get('colormap', 'viridis')
    grid_alpha = customizations.get('grid_alpha', 0.3)
    figure_width = customizations.get('figure_width', 16)
    figure_height = customizations.get('figure_height', 12)
    tick_size = customizations.get('tick_size', 11)
    label_fontsize = customizations.get('label_fontsize', 12)
    dpi = customizations.get('dpi', 150)
    
    A1_fit, A2_fit, Tm_fit, DeltaHf_fit, k_fit, H298_fit = fit_params
    
    # Create figure with enhanced layout
    fig = plt.figure(figsize=(figure_width, figure_height), dpi=dpi, constrained_layout=True)
    
    # Create grid for complex layout
    gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.25)
    
    # Main fit plot
    ax1 = fig.add_subplot(gs[0:2, :])
    
    # Enhanced scatter plot
    scatter = ax1.scatter(T_data, H_data, alpha=0.7, label='Original Data', 
                         color='#1E88E5', s=60, edgecolors='white', linewidth=1.5,
                         zorder=5)
    
    # Enhanced fitted curve
    ax1.plot(T_fit, H_fit, 'r-', linewidth=curve_thickness+1, 
            label='Fitted Curve', alpha=0.9, zorder=4)
    
    # Calculate confidence interval (simplified)
    n = len(T_data)
    p = len(fit_params)
    t_value = 2.0  # Approximate for 95% confidence
    
    # Calculate standard error of the fit
    residuals_std = np.std(residuals)
    conf_interval = t_value * residuals_std * np.sqrt(1/n + (T_fit - np.mean(T_data))**2 / np.sum((T_data - np.mean(T_data))**2))
    
    # Add confidence interval
    ax1.fill_between(T_fit, H_fit - conf_interval, H_fit + conf_interval,
                    alpha=0.2, color='red', label='95% Confidence Interval')
    
    # Highlight melting temperature with colorful style
    ax1.axvline(Tm_fit, color='red', linestyle='--', alpha=0.9, linewidth=2.5,
              label=f'Melting Point Tₘ = {Tm_fit:.1f} K', zorder=3)
    
    # Add shaded melting region
    ax1.axvspan(Tm_fit-50, Tm_fit+50, alpha=0.2, color='orange', label='Melting Region', zorder=2)
    
    # Enhanced labels and titles
    ax1.set_xlabel('Temperature (K)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
    ax1.set_ylabel('Enthalpy (J/mol)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
    ax1.set_title(f'Enthalpy-Temperature Curve Fitting - {material_name}', 
                 fontsize=title_font_size+2, fontweight='bold', pad=25)
    ax1.grid(True, alpha=grid_alpha, linestyle='--', zorder=1)
    
    # Enhanced legend placement
    ax1.legend(loc='upper left', fontsize=legend_font_size, framealpha=0.95, 
              borderpad=1, labelspacing=0.5, handlelength=2,
              shadow=True, fancybox=True, edgecolor='black')
    
    # Set tick parameters
    ax1.tick_params(axis='both', which='major', labelsize=tick_size, pad=10)
    ax1.tick_params(axis='both', which='minor', labelsize=tick_size-2, pad=8)
    
    # Residual plot
    ax2 = fig.add_subplot(gs[2, 0])
    residuals_scatter = ax2.scatter(T_data, residuals, alpha=0.7, color='#FF7043', s=50)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add residual statistics
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    ax2.axhline(y=residual_mean, color='blue', linestyle=':', alpha=0.5, linewidth=1.5,
               label=f'Mean: {residual_mean:.2f}')
    ax2.axhline(y=residual_mean + 2*residual_std, color='green', linestyle=':', 
               alpha=0.5, linewidth=1, label='±2σ')
    ax2.axhline(y=residual_mean - 2*residual_std, color='green', linestyle=':', 
               alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('Temperature (K)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
    ax2.set_ylabel('Residuals (J/mol)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
    ax2.set_title(f'Residual Plot\nσ = {residual_std:.2f}', 
                 fontsize=title_font_size, fontweight='bold', pad=15)
    ax2.grid(True, alpha=grid_alpha, linestyle='--')
    ax2.legend(fontsize=legend_font_size-2)
    ax2.tick_params(axis='both', labelsize=tick_size, pad=8)
    
    # Specific enthalpy fit
    ax3 = fig.add_subplot(gs[2, 1])
    H_specific_fit = H_fit / (molar_weight / 1000.0)
    ax3.plot(T_fit, H_specific_fit, 'purple', linewidth=curve_thickness, alpha=0.9)
    ax3.axvline(Tm_fit, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Add specific enthalpy statistics
    delta_H_specific = H_specific_fit.max() - H_specific_fit.min()
    ax3.annotate(f'ΔH = {delta_H_specific:,.0f} J/kg', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=font_size, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_xlabel('Temperature (K)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
    ax3.set_ylabel('Enthalpy (J/kg)', fontsize=label_fontsize, fontweight='bold', labelpad=15)
    ax3.set_title('Fitted Specific Enthalpy', fontsize=title_font_size, fontweight='bold', pad=15)
    ax3.grid(True, alpha=grid_alpha, linestyle='--')
    ax3.tick_params(axis='both', labelsize=tick_size, pad=8)
    
    # Coefficient summary with enhanced formatting
    ax4 = fig.add_subplot(gs[2, 2])
    ax4.axis('off')
    
    # Format coefficients with proper alignment and spacing
    coeff_text = (
        f"🔧 **Fitted Parameters:**\n\n"
        f"• A₁ = {A1_fit:.4f} J/(mol·K)\n"
        f"• A₂ = {A2_fit:.4f} J/(mol·K)\n"
        f"• Tₘ = {Tm_fit:.2f} K\n"
        f"• ΔHf = {DeltaHf_fit:,.0f} J/mol\n"
        f"• k = {k_fit:.6f} 1/K\n"
        f"• H₂₉₈ = {H298_fit:,.0f} J/mol\n"
        f"• M = {molar_weight:.2f} g/mol\n\n"
        f"📊 **Goodness of Fit:**\n"
        f"• R² = {r_squared:.6f}\n"
        f"• RMSE = {rmse:.2f} J/mol\n"
        f"• Data Points = {len(T_data)}\n"
        f"• Residual σ = {residual_std:.2f}"
    )
    
    # Add text with enhanced padding and formatting
    ax4.text(0.1, 0.95, coeff_text, transform=ax4.transAxes,
            fontsize=label_fontsize, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', 
                     alpha=0.9, pad=15, linewidth=2, edgecolor='navy'))
    
    # Set box thickness for all axes
    for ax in [ax1, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_linewidth(box_thickness)
    
    # Add composition info at the bottom
    comp_text = 'Composition: ' + ', '.join([f'{e}={f:.4f}' for e, f in list(composition.items())[:4]])
    if len(composition) > 4:
        comp_text += f" ... (+{len(composition)-4} more)"
    
    fig.text(0.5, 0.02, comp_text, ha='center', fontsize=font_size-1, 
            style='italic', alpha=0.9, fontweight='medium',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=5))
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.02, 0.98, f"Generated: {timestamp}", fontsize=8, alpha=0.6)
    
    return fig

def create_phase_diagram_visualization(analyzer, selected_indices, customizations=None):
    """Create phase diagram visualization for selected materials"""
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
    colormap = customizations.get('colormap', 'tab10')
    grid_alpha = customizations.get('grid_alpha', 0.3)
    figure_width = customizations.get('figure_width', 14)
    figure_height = customizations.get('figure_height', 10)
    tick_size = customizations.get('tick_size', 11)
    label_fontsize = customizations.get('label_fontsize', 12)
    
    fig, axes = plt.subplots(2, 2, figsize=(figure_width, figure_height), 
                            constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # Get colormap
    if colormap in plt.colormaps():
        cmap = plt.get_cmap(colormap)
    else:
        cmap = plt.get_cmap('tab10')
    
    colors = cmap(np.linspace(0, 1, min(10, len(selected_indices))))
    
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'D']
    
    for i, idx in enumerate(selected_indices):
        if idx >= len(analyzer.results_history):
            continue
            
        result = analyzer.results_history[idx]
        data = result['data']
        name = result['name']
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Plot 1: Enthalpy vs Temperature
        ax1.plot(data['Temperature_K'], data['Enthalpy_J_mol'],
                color=color, linewidth=curve_thickness,
                marker=marker, markersize=4,
                markevery=max(1, len(data)//20),
                label=name)
        
        # Plot 2: Specific Enthalpy vs Temperature
        ax2.plot(data['Temperature_K'], data['Enthalpy_J_kg'],
                color=color, linewidth=curve_thickness,
                marker=marker, markersize=4,
                markevery=max(1, len(data)//20),
                label=name)
        
        # Plot 3: Heat Capacity
        if len(data) > 1:
            dT = data['Temperature_K'].diff()
            dH = data['Enthalpy_J_mol'].diff()
            heat_capacity = dH / dT
            
            ax3.plot(data['Temperature_K'].iloc[1:], heat_capacity.iloc[1:],
                    color=color, linewidth=curve_thickness-1,
                    marker=marker, markersize=3,
                    markevery=max(1, len(data)//30),
                    label=name, alpha=0.8)
        
        # Plot 4: Cumulative enthalpy
        cumulative_enthalpy = data['Enthalpy_J_mol'].cumsum()
        ax4.plot(data['Temperature_K'], cumulative_enthalpy,
                color=color, linewidth=curve_thickness,
                marker=marker, markersize=4,
                markevery=max(1, len(data)//20),
                label=name)
    
    # Format plots
    ax1.set_xlabel('Temperature (K)', fontsize=label_fontsize)
    ax1.set_ylabel('Enthalpy (J/mol)', fontsize=label_fontsize)
    ax1.set_title('Molar Enthalpy', fontsize=title_font_size)
    ax1.grid(True, alpha=grid_alpha)
    ax1.legend(fontsize=legend_font_size-2)
    
    ax2.set_xlabel('Temperature (K)', fontsize=label_fontsize)
    ax2.set_ylabel('Specific Enthalpy (J/kg)', fontsize=label_fontsize)
    ax2.set_title('Specific Enthalpy', fontsize=title_font_size)
    ax2.grid(True, alpha=grid_alpha)
    ax2.legend(fontsize=legend_font_size-2)
    
    ax3.set_xlabel('Temperature (K)', fontsize=label_fontsize)
    ax3.set_ylabel('Heat Capacity (J/(mol·K))', fontsize=label_fontsize)
    ax3.set_title('Heat Capacity', fontsize=title_font_size)
    ax3.grid(True, alpha=grid_alpha)
    ax3.legend(fontsize=legend_font_size-2)
    
    ax4.set_xlabel('Temperature (K)', fontsize=label_fontsize)
    ax4.set_ylabel('Cumulative Enthalpy (J/mol)', fontsize=label_fontsize)
    ax4.set_title('Cumulative Enthalpy', fontsize=title_font_size)
    ax4.grid(True, alpha=grid_alpha)
    ax4.legend(fontsize=legend_font_size-2)
    
    # Set box thickness
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_linewidth(box_thickness)
    
    plt.suptitle('Phase Diagram Analysis', fontsize=title_font_size+2, fontweight='bold')
    
    return fig

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
        col_sess1, col_sess2, col_sess3 = st.columns(3)
        with col_sess1:
            st.metric("Computed Results", len(analyzer.results_history))
        with col_sess2:
            st.metric("Fitting Results", len(analyzer.fitting_results))
        with col_sess3:
            st.metric("Available TDB Files", len(analyzer.get_available_tdb_files()))
    
    # Create tabs with enhanced icons
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔬 Enthalpy Computation",
        "📊 Curve Fitting & Analysis", 
        "🔄 Multi-Material Comparison",
        "📈 Phase Diagram Analysis",
        "🎨 Visualization Customization",
        "ℹ️ Help & Settings"
    ])
    
    # ==================== TAB 1: Enthalpy Computation ====================
    with tab1:
        st.markdown('<div class="sub-header">🔬 Enthalpy Computation from TDB Files</div>', unsafe_allow_html=True)
        
        # File selection section
        st.markdown("#### 📁 TDB File Selection")
        col_file1, col_file2 = st.columns([1.2, 1])
        
        with col_file1:
            available_tdb_files = analyzer.get_available_tdb_files()
            
            if available_tdb_files:
                tdb_source = st.radio(
                    "Source:",
                    ["Select from database directory", "Upload new TDB file"],
                    horizontal=True,
                    key="tab1_tdb_source"
                )
            else:
                st.info(f"No TDB files found in '{analyzer.database_dir}'. Please upload a file.")
                tdb_source = "Upload new TDB file"
            
            tdb_path = None
            
            if tdb_source == "Select from database directory" and available_tdb_files:
                selected_file = st.selectbox(
                    f"Available TDB files in '{analyzer.database_dir}' directory:",
                    available_tdb_files,
                    help="Select a thermodynamic database file",
                    key="tab1_tdb_select"
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
                    key="tab1_uploader"
                )
                
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.tdb') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tdb_path = tmp_file.name
                    
                    st.success(f"✓ Uploaded: **{uploaded_file.name}**")
                    
                    if st.checkbox("💾 Save to 'databases' directory for future use", value=True, key="tab1_save_tdb"):
                        saved_path = analyzer.save_uploaded_tdb(uploaded_file)
                        if saved_path:
                            st.info(f"Saved to: `{saved_path}`")
        
        with col_file2:
            if tdb_path and os.path.exists(tdb_path):
                try:
                    with st.spinner("Loading thermodynamic database..."):
                        dbf = Database(tdb_path)
                    
                    st.markdown("#### 🗄️ Database Information")
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
        st.markdown("#### ⚙️ Calculation Settings")
        
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
                help="Select elements to include in your alloy composition",
                key="tab1_elements"
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
                key="tab1_comp_type"
            )
            
            # Initialize session state for composition
            composition_key = f"tab1_composition_{'_'.join(sorted(selected_elements))}"
            if composition_key not in st.session_state:
                n_elements = len(selected_elements)
                initial_values = {element: 1.0/n_elements for element in selected_elements}
                st.session_state[composition_key] = initial_values
            
            composition = {}
            total_entered = 0.0
            
            # Create input widgets for all but last element
            for i, element in enumerate(selected_elements[:-1]):
                col_slider, col_number = st.columns([3, 1])
                
                with col_slider:
                    current_value = st.session_state[composition_key].get(element, 1.0/len(selected_elements))
                    
                    # Use widget state management
                    widget_key_slider = f"tab1_slider_{element}"
                    widget_key_number = f"tab1_num_{element}"
                    
                    # Initialize widget state if not exists
                    if widget_key_slider not in st.session_state.widget_state:
                        st.session_state.widget_state[widget_key_slider] = current_value
                    if widget_key_number not in st.session_state.widget_state:
                        st.session_state.widget_state[widget_key_number] = current_value
                    
                    # Create slider
                    slider_value = st.slider(
                        f"{element} {fraction_type}",
                        0.0, 1.0,
                        float(st.session_state.widget_state[widget_key_slider]),
                        0.001,
                        key=widget_key_slider,
                        format="%.3f",
                        on_change=lambda el=element, key_s=widget_key_slider, key_n=widget_key_number: 
                            st.session_state.widget_state.update({key_n: st.session_state[key_s]})
                    )
                    
                    # Update widget state
                    st.session_state.widget_state[widget_key_slider] = slider_value
                
                with col_number:
                    # Create number input that syncs with slider
                    num_value = st.number_input(
                        "Value",
                        0.0, 1.0,
                        float(st.session_state.widget_state[widget_key_number]),
                        0.001,
                        key=widget_key_number,
                        format="%.3f",
                        label_visibility="collapsed",
                        on_change=lambda el=element, key_s=widget_key_slider, key_n=widget_key_number: 
                            st.session_state.widget_state.update({key_s: st.session_state[key_n]})
                    )
                    
                    # Update widget state
                    st.session_state.widget_state[widget_key_number] = num_value
                
                # Use the slider value (which is always in sync)
                composition[element] = slider_value
                total_entered += slider_value
            
            # Last element calculation
            last_element = selected_elements[-1]
            last_value = max(0.0, 1.0 - total_entered)
            composition[last_element] = last_value
            
            # Update session state
            st.session_state[composition_key] = composition
            
            # Display summary
            total = sum(composition.values())
            st.markdown(f"""
            <div class="info-box">
            <strong>Auto-calculated:</strong> {last_element} = {last_value:.4f}<br>
            <strong>Total:</strong> {total:.4f}<br>
            <strong>Status:</strong> <span style="color:{'green' if abs(total - 1.0) < 0.001 else 'red'}">
            {"✅ Valid" if abs(total - 1.0) < 0.001 else "❌ Invalid - Please adjust values"}
            </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Reset button
            if st.button("🔄 Reset Composition", key="tab1_reset_comp"):
                if composition_key in st.session_state:
                    del st.session_state[composition_key]
                for element in selected_elements:
                    widget_key_slider = f"tab1_slider_{element}"
                    widget_key_number = f"tab1_num_{element}"
                    if widget_key_slider in st.session_state.widget_state:
                        del st.session_state.widget_state[widget_key_slider]
                    if widget_key_number in st.session_state.widget_state:
                        del st.session_state.widget_state[widget_key_number]
                st.rerun()
            
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
                T_start = st.number_input("Start (K)", 100, 5000, 300, 10, key="tab1_T_start")
            with col_temp2:
                T_end = st.number_input("End (K)", T_start+10, 6000, 1500, 10, key="tab1_T_end")
            with col_temp3:
                T_step = st.number_input("Step (K)", 1, 200, 10, key="tab1_T_step")
            
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
                help="Select phases to consider in equilibrium calculation",
                key="tab1_phases"
            )
            
            if not selected_phases:
                st.warning("⚠️ Please select at least one phase")
                st.stop()
            
            # Pressure setting
            P = st.number_input("Pressure (Pa)", 1000, 10000000, 101325, 1000, key="tab1_pressure")
            
            # Additional options
            st.markdown("**Additional Options:**")
            show_heat_capacity = st.checkbox("Calculate heat capacity", value=True, key="tab1_heat_cap")
            smooth_data = st.checkbox("Smooth enthalpy data", value=False, key="tab1_smooth")
        
        # Compute button with session state preservation
        st.markdown("---")
        
        # Create a unique cache key for this simulation
        cache_key = f"simulation_{hash(str(composition_mole))}_{T_start}_{T_end}_{T_step}_{P}_{str(sorted(selected_phases))}"
        
        compute_button = st.button("🚀 Compute Enthalpy", type="primary", use_container_width=True, key="tab1_compute")
        
        if compute_button:
            with st.spinner("🔄 Performing equilibrium calculation... This may take a moment"):
                try:
                    # Check if result is already cached
                    if cache_key in st.session_state.simulation_cache:
                        st.info("📊 Using cached results from previous computation")
                        result_df = st.session_state.simulation_cache[cache_key]['data']
                        material_name = st.session_state.simulation_cache[cache_key]['name']
                    else:
                        conditions = {
                            v.T: (T_start, T_end, T_step),
                            v.P: P
                        }
                        
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
                            output='HM',
                            verbose=False,
                            broadcast=False
                        )
                        
                        # Extract results
                        T_values = eq_result.T.values.flatten()
                        result_values = eq_result['HM'].values.flatten()
                        
                        result_df = pd.DataFrame({
                            'Temperature_K': T_values,
                            'Enthalpy_J_mol': result_values
                        })
                        
                        result_df = result_df.dropna().sort_values('Temperature_K').reset_index(drop=True)
                        
                        if len(result_df) == 0:
                            st.error("❌ Calculation returned no valid results. Try adjusting parameters.")
                            st.stop()
                        
                        # Smooth data if requested
                        if smooth_data and len(result_df) > 10:
                            window_size = min(5, len(result_df) // 10)
                            result_df['Enthalpy_J_mol'] = result_df['Enthalpy_J_mol'].rolling(
                                window=window_size, center=True, min_periods=1
                            ).mean()
                        
                        # Cache the result
                        material_name = "-".join([f"{e}{composition_mole[e]:.2f}" for e in selected_elements[:3]])
                        if len(selected_elements) > 3:
                            material_name += f"-{len(selected_elements)-3}more"
                        
                        st.session_state.simulation_cache[cache_key] = {
                            'data': result_df,
                            'name': material_name,
                            'composition': composition_mole,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    
                    # Add specific enthalpy
                    result_df = analyzer.convert_to_specific_enthalpy(result_df, composition_mole)
                    
                    # Add heat capacity if requested
                    if show_heat_capacity and len(result_df) > 1:
                        dT = result_df['Temperature_K'].diff()
                        dH = result_df['Enthalpy_J_mol'].diff()
                        result_df['Heat_Capacity_J_per_mol_K'] = dH / dT
                    
                    # Store results in session state
                    result_info = {
                        'name': material_name,
                        'composition': composition_mole,
                        'composition_type': fraction_type,
                        'data': result_df,
                        'phases': selected_phases,
                        'tdb_file': os.path.basename(tdb_path),
                        'temperature_range': (T_start, T_end, T_step),
                        'output_quantity': 'HM',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'pressure': P,
                        'show_heat_capacity': show_heat_capacity
                    }
                    
                    # Check if this result already exists
                    existing_idx = -1
                    for i, res in enumerate(analyzer.results_history):
                        if (res['name'] == material_name and 
                            res['temperature_range'] == (T_start, T_end, T_step) and
                            res['tdb_file'] == os.path.basename(tdb_path)):
                            existing_idx = i
                            break
                    
                    if existing_idx >= 0:
                        analyzer.results_history[existing_idx] = result_info
                        st.info("↻ Updated existing calculation")
                    else:
                        analyzer.results_history.append(result_info)
                    
                    # Create visualization
                    fig = create_enhanced_visualization(
                        result_df, 
                        composition_mole, 
                        material_name,
                        analyzer.plot_customizations,
                        None,
                        show_heat_capacity
                    )
                    
                    # Create thumbnail
                    thumbnail = analyzer.create_thumbnail(fig)
                    if thumbnail:
                        thumb_exists = False
                        for i, thumb in enumerate(analyzer.history_thumbnails):
                            if thumb['name'] == material_name:
                                analyzer.history_thumbnails[i] = {
                                    'name': material_name,
                                    'thumbnail': thumbnail,
                                    'timestamp': result_info['timestamp']
                                }
                                thumb_exists = True
                                break
                        
                        if not thumb_exists:
                            analyzer.history_thumbnails.append({
                                'name': material_name,
                                'thumbnail': thumbnail,
                                'timestamp': result_info['timestamp']
                            })
                    
                    st.success(f"✅ Calculation completed! Generated {len(result_df)} data points.")
                    
                    # Display the plot in a container
                    with st.container():
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        st.pyplot(fig)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
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
                    
                    # Enhanced download section
                    st.markdown('<div class="download-section">', unsafe_allow_html=True)
                    st.markdown("#### 📥 Download Results")
                    
                    st.markdown('<div class="column-selector">', unsafe_allow_html=True)
                    st.markdown("**Select columns to include:**")
                    
                    available_columns = ['Temperature_K', 'Enthalpy_J_mol', 'Enthalpy_J_kg']
                    if show_heat_capacity and 'Heat_Capacity_J_per_mol_K' in result_df.columns:
                        available_columns.append('Heat_Capacity_J_per_mol_K')
                    
                    col_sel1, col_sel2 = st.columns(2)
                    
                    with col_sel1:
                        csv_columns = st.multiselect(
                            "CSV Download Columns:",
                            options=available_columns,
                            default=available_columns,
                            key="tab1_csv_columns"
                        )
                        
                        col_p1, col_p2, col_p3 = st.columns(3)
                        with col_p1:
                            if st.button("Basic", key="tab1_preset_basic"):
                                st.session_state.tab1_csv_columns = ['Temperature_K', 'Enthalpy_J_mol']
                                st.rerun()
                        with col_p2:
                            if st.button("All", key="tab1_preset_all"):
                                st.session_state.tab1_csv_columns = available_columns
                                st.rerun()
                        with col_p3:
                            if st.button("Clear", key="tab1_preset_clear"):
                                st.session_state.tab1_csv_columns = []
                                st.rerun()
                    
                    with col_sel2:
                        dat_columns = st.multiselect(
                            "DAT Download Columns:",
                            options=available_columns,
                            default=available_columns,
                            key="tab1_dat_columns"
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        if csv_columns:
                            csv_df = result_df[csv_columns].copy()
                            csv_full = csv_df.to_csv(index=False)
                            
                            col_suffix = "_".join([c.split('_')[0] for c in csv_columns])
                            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                            
                            st.download_button(
                                f"📄 Download CSV ({col_suffix})",
                                data=csv_full,
                                file_name=f"enthalpy_{material_name.replace(' ', '_')}_{col_suffix}_{timestamp_str}.csv",
                                mime="text/csv",
                                key=f"tab1_csv_download_{timestamp_str}"
                            )
                        else:
                            st.warning("Please select at least one column for CSV download")
                    
                    with col_dl2:
                        if dat_columns:
                            metadata = {
                                'TDB File': os.path.basename(tdb_path),
                                'Phases': ', '.join(selected_phases),
                                'Pressure (Pa)': P,
                                'Temperature Range': f"{T_start}-{T_end} K",
                                'Composition Type': fraction_type,
                                'Columns': ', '.join(dat_columns),
                                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            dat_df = result_df[dat_columns].copy()
                            dat_content = analyzer.format_dat_file(dat_df, composition_mole, metadata, dat_columns)
                            
                            dat_suffix = "_".join([c.split('_')[0] for c in dat_columns])
                            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                            
                            st.download_button(
                                f"📄 Download DAT ({dat_suffix})",
                                data=dat_content,
                                file_name=f"enthalpy_{material_name.replace(' ', '_')}_{dat_suffix}_{timestamp_str}.dat",
                                mime="text/plain",
                                key=f"tab1_dat_download_{timestamp_str}"
                            )
                        else:
                            st.warning("Please select at least one column for DAT download")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Data preview
                    with st.expander("🔍 View Complete Data Table", expanded=False):
                        st.markdown('<div class="data-table">', unsafe_allow_html=True)
                        st.dataframe(result_df, use_container_width=True, height=300)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ Calculation error: {str(e)}")
                    st.exception(e)
    
    # ==================== TAB 2: Curve Fitting ====================
    with tab2:
        st.markdown('<div class="sub-header">📊 Curve Fitting & Analysis</div>', unsafe_allow_html=True)
        
        if not analyzer.results_history:
            st.info("💡 No computed data available. Please perform calculations in the 'Enthalpy Computation' tab first.")
            st.stop()
        
        # Data source selection
        data_source = st.radio(
            "Select data source:",
            ["Use computed results from Tab 1", "Upload external CSV file"],
            horizontal=True,
            key="tab2_data_source"
        )
        
        if data_source == "Use computed results from Tab 1":
            result_options = [
                f"{i+1}. {res['name']} | {res['tdb_file']} | {res['temperature_range'][0]}-{res['temperature_range'][1]}K"
                for i, res in enumerate(analyzer.results_history)
            ]
            
            selected_result_idx = st.selectbox(
                "Select computed result:",
                range(len(result_options)),
                format_func=lambda x: result_options[x],
                key="tab2_result_select"
            )
            
            result_data = analyzer.results_history[selected_result_idx]['data']
            T_data = result_data['Temperature_K'].values
            H_data = result_data['Enthalpy_J_mol'].values
            material_name = analyzer.results_history[selected_result_idx]['name']
            composition_ref = analyzer.results_history[selected_result_idx]['composition']
            composition_type_ref = analyzer.results_history[selected_result_idx].get('composition_type', 'Mole Fraction')
            
            # Display composition info
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("#### 📝 Composition Reference")
            
            col_comp1, col_comp2 = st.columns(2)
            with col_comp1:
                display_type = st.radio(
                    "Display composition as:",
                    ["Mole Fraction", "Weight Fraction"],
                    horizontal=True,
                    key="tab2_display_comp_type"
                )
            
            with col_comp2:
                if display_type == "Weight Fraction" and composition_type_ref == "Mole Fraction":
                    display_comp = analyzer.convert_composition(composition_ref, 'mole', 'weight')
                elif display_type == "Mole Fraction" and composition_type_ref == "Weight Fraction":
                    display_comp = analyzer.convert_composition(composition_ref, 'weight', 'mole')
                else:
                    display_comp = composition_ref
                
                st.write("**Composition:**")
                comp_text = ""
                for element, fraction in display_comp.items():
                    comp_text += f"• {element}: {fraction:.6f}\n"
                st.text(comp_text)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            composition_for_fitting = composition_ref
            composition_type_for_fitting = composition_type_ref
            
        else:
            uploaded_csv = st.file_uploader(
                "Upload CSV with Temperature and Enthalpy columns",
                type=['csv'],
                key="tab2_uploader_csv"
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
                    temp_col = st.selectbox("Temperature column", df_upload.columns, key="tab2_temp_col")
                with col2:
                    enthalpy_col = st.selectbox("Enthalpy column (J/mol)", df_upload.columns, key="tab2_enthalpy_col")
                
                T_data = df_upload[temp_col].values
                H_data = df_upload[enthalpy_col].values
                material_name = "Uploaded Data"
                
                # Manual composition input
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("#### 📝 Composition Reference (Manual Input)")
                
                col_man1, col_man2 = st.columns(2)
                with col_man1:
                    comp_type = st.radio(
                        "Input composition as:",
                        ["Mole Fraction", "Weight Fraction"],
                        horizontal=True,
                        key="tab2_manual_comp_type"
                    )
                
                with col_man2:
                    elements = st.multiselect(
                        "Select elements:",
                        sorted(PERIODIC_TABLE.keys()),
                        default=['AL', 'CU', 'NI'],
                        key="tab2_manual_elements"
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
                                key=f"tab2_manual_comp_{element}"
                            )
                            composition_for_fitting[element] = fraction
                    
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
        st.markdown("#### ⚙️ Fitting Parameters")
        
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
                help="Sensible heat coefficient for solid phase",
                key="tab2_A1_guess"
            )
            A2_guess = st.number_input(
                "A₂ initial guess (J/mol·K)", 
                -100.0, 100.0, 
                float(min(30.0, max(1.0, H_slope * 0.3))), 
                0.1,
                help="Additional sensible heat coefficient for liquid phase",
                key="tab2_A2_guess"
            )
            Tm_guess = st.number_input(
                "Tₘ initial guess (K)", 
                float(T_data.min()), float(T_data.max()), 
                float(T_mid), 
                1.0,
                help="Melting temperature",
                key="tab2_Tm_guess"
            )
        
        with col_p2:
            DeltaHf_guess = st.number_input(
                "ΔHf initial guess (J/mol)", 
                -50000.0, 50000.0, 
                float(min(30000.0, max(5000.0, H_range * 0.6))), 
                100.0,
                help="Heat of fusion",
                key="tab2_DeltaHf_guess"
            )
            k_guess = st.number_input(
                "k initial guess (1/K)", 
                0.0001, 1.0, 
                0.01, 
                0.001,
                help="Sigmoid steepness parameter",
                key="tab2_k_guess"
            )
            H298_guess = st.number_input(
                "H₂₉₈ initial guess (J/mol)", 
                -50000.0, 50000.0, 
                float(H_data.min() * 0.9), 
                100.0,
                help="Reference enthalpy at 298 K",
                key="tab2_H298_guess"
            )
        
        # Advanced fitting options
        with st.expander("⚙️ Advanced Fitting Options", expanded=False):
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                max_iterations = st.number_input("Maximum iterations", 100, 10000, 5000, 100, key="tab2_max_iter")
                fit_method = st.selectbox("Fitting method", ["trf", "lm", "dogbox"], index=0, key="tab2_fit_method")
            with col_adv2:
                confidence_level = st.slider("Confidence level (%)", 90, 99, 95, 1, key="tab2_conf_level")
                generate_report = st.checkbox("Generate detailed report", value=True, key="tab2_gen_report")
        
        # Fit button
        st.markdown("---")
        fit_button = st.button("🎯 Perform Curve Fit", type="primary", use_container_width=True, key="tab2_fit_button")
        
        if fit_button:
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
                        method=fit_method,
                        maxfev=max_iterations
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
                    
                    # Calculate confidence intervals
                    perr = np.sqrt(np.diag(pcov))
                    confidence_intervals = []
                    for i, param in enumerate(fit_params):
                        ci = 1.96 * perr[i]
                        confidence_intervals.append((param - ci, param + ci))
                    
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
                        'confidence_intervals': {
                            'A1': confidence_intervals[0],
                            'A2': confidence_intervals[1],
                            'Tm': confidence_intervals[2],
                            'DeltaHf': confidence_intervals[3],
                            'k': confidence_intervals[4],
                            'H298': confidence_intervals[5]
                        },
                        'statistics': {
                            'r_squared': r_squared,
                            'rmse': rmse,
                            'data_points': len(T_data),
                            'molar_weight_g_per_mol': molar_weight,
                            'residual_std': np.std(residuals),
                            'confidence_level': confidence_level
                        },
                        'composition': composition_for_fitting,
                        'composition_type': composition_type_for_fitting,
                        'temperature_range': [float(T_data.min()), float(T_data.max())],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'fitting_method': fit_method,
                        'max_iterations': max_iterations
                    }
                    
                    # Check if this fit already exists
                    existing_idx = -1
                    for i, fit in enumerate(analyzer.fitting_results):
                        if (fit['material_name'] == material_name and 
                            fit['temperature_range'] == [float(T_data.min()), float(T_data.max())]):
                            existing_idx = i
                            break
                    
                    if existing_idx >= 0:
                        analyzer.fitting_results[existing_idx] = fit_result
                        st.info("↻ Updated existing fitting result")
                    else:
                        analyzer.fitting_results.append(fit_result)
                    
                    st.success(f"✅ Fitting completed! R² = {r_squared:.6f}, RMSE = {rmse:.2f} J/mol")
                    
                    # Create enhanced visualization
                    fig = create_curve_fitting_visualization(
                        T_data, H_data, T_fit, H_fit, fit_params, residuals,
                        r_squared, rmse, material_name, composition_for_fitting,
                        molar_weight, analyzer.plot_customizations
                    )
                    
                    with st.container():
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        st.pyplot(fig)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display equations
                    st.markdown("---")
                    col_eq1, col_eq2 = st.columns(2)
                    
                    with col_eq1:
                        st.markdown("#### 🧮 Molar Enthalpy Equation (J/mol)")
                        st.latex(rf"""
                        H_{{molar}}(T) = {A1_fit:.4f} \cdot T + {A2_fit:.4f} \cdot \max(T - {Tm_fit:.2f}, 0) + 
                        {DeltaHf_fit:,.0f} \cdot \frac{{1}}{{1 + e^{{-{k_fit:.6f}(T - {Tm_fit:.2f})}}}} + {H298_fit:,.0f}
                        """)
                    
                    with col_eq2:
                        st.markdown("#### 🧮 Specific Enthalpy Equation (J/kg)")
                        st.latex(rf"""
                        H_{{specific}}(T) = \frac{{1}}{{{molar_weight:.4f} \times 10^{{-3}}}} \times \left[{A1_fit:.4f} \cdot T + {A2_fit:.4f} \cdot \max(T - {Tm_fit:.2f}, 0) + 
                        {DeltaHf_fit:,.0f} \cdot \frac{{1}}{{1 + e^{{-{k_fit:.6f}(T - {Tm_fit:.2f})}}}} + {H298_fit:,.0f}\right]
                        """)
                    
                    # Display confidence intervals
                    st.markdown("#### 📐 Parameter Confidence Intervals (95%)")
                    ci_df = pd.DataFrame({
                        'Parameter': ['A₁', 'A₂', 'Tₘ', 'ΔHf', 'k', 'H₂₉₈'],
                        'Value': [f"{A1_fit:.4f}", f"{A2_fit:.4f}", f"{Tm_fit:.2f}", 
                                 f"{DeltaHf_fit:,.0f}", f"{k_fit:.6f}", f"{H298_fit:,.0f}"],
                        'Lower Bound': [f"{ci[0]:.4f}" for ci in confidence_intervals],
                        'Upper Bound': [f"{ci[1]:.4f}" for ci in confidence_intervals]
                    })
                    st.dataframe(ci_df, use_container_width=True, hide_index=True)
                    
                    # Download section
                    st.markdown('<div class="download-section">', unsafe_allow_html=True)
                    st.markdown("#### 📥 Download Fitting Results")
                    
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
                        
                        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button(
                            "📄 Coefficients (CSV)",
                            data=coeff_df.to_csv(index=False),
                            file_name=f"fitting_coeffs_{material_name.replace(' ', '_')}_{timestamp_str}.csv",
                            mime="text/csv",
                            key=f"tab2_coeff_csv_{timestamp_str}"
                        )
                    
                    with col_f2:
                        fitted_df = pd.DataFrame({
                            'Temperature_K': T_fit,
                            'Enthalpy_Fitted_J_mol': H_fit,
                            'Enthalpy_Fitted_J_kg': analyzer.specific_enthalpy_equation(T_fit, *fit_params, molar_weight)
                        })
                        
                        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button(
                            "📄 Fitted Curve (CSV)",
                            data=fitted_df.to_csv(index=False),
                            file_name=f"fitted_curve_{material_name.replace(' ', '_')}_{timestamp_str}.csv",
                            mime="text/csv",
                            key=f"tab2_curve_csv_{timestamp_str}"
                        )
                    
                    with col_f3:
                        json_data = json.dumps(fit_result, indent=4, default=str)
                        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button(
                            "📄 Full Results (JSON)",
                            data=json_data,
                            file_name=f"fitting_results_{material_name.replace(' ', '_')}_{timestamp_str}.json",
                            mime="application/json",
                            key=f"tab2_json_results_{timestamp_str}"
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Generate detailed report
                    if generate_report:
                        with st.expander("📋 Detailed Fitting Report", expanded=False):
                            st.markdown(f"""
                            ### Fitting Report for {material_name}
                            
                            **Dataset Information:**
                            - Data points: {len(T_data)}
                            - Temperature range: {T_data.min():.1f} K to {T_data.max():.1f} K
                            - Enthalpy range: {H_data.min():,.0f} to {H_data.max():,.0f} J/mol
                            
                            **Fitting Parameters:**
                            - Method: {fit_method}
                            - Maximum iterations: {max_iterations}
                            - Initial guess: {initial_guess}
                            
                            **Residual Analysis:**
                            - Residual standard deviation: {np.std(residuals):.2f} J/mol
                            - Residual mean: {np.mean(residuals):.2f} J/mol
                            - Maximum residual: {np.max(np.abs(residuals)):.2f} J/mol
                            
                            **Goodness of Fit:**
                            - R²: {r_squared:.6f}
                            - Adjusted R²: {1 - (1 - r_squared) * (len(T_data) - 1) / (len(T_data) - len(fit_params) - 1):.6f}
                            - RMSE: {rmse:.2f} J/mol
                            - MAE: {np.mean(np.abs(residuals)):.2f} J/mol
                            
                            **Physical Interpretation:**
                            - Melting temperature (Tₘ): {Tm_fit:.1f} K
                            - Heat of fusion (ΔHf): {DeltaHf_fit:,.0f} J/mol
                            - Sensible heat coefficients: A₁ = {A1_fit:.3f}, A₂ = {A2_fit:.3f} J/(mol·K)
                            - Transition sharpness (k): {k_fit:.6f} 1/K
                            """)
                            
                except Exception as e:
                    st.error(f"❌ Fitting error: {str(e)}")
                    st.exception(e)

    # ==================== TAB 3: Multi-Material Comparison ====================
    with tab3:
        st.markdown('<div class="sub-header">🔄 Multi-Material Comparison</div>', unsafe_allow_html=True)
        
        if not analyzer.results_history:
            st.info("💡 No computed data available for comparison. Please perform calculations in the 'Enthalpy Computation' tab first.")
            st.stop()
        
        # Display history with thumbnails
        st.markdown("#### 📚 Calculation History")
        
        if analyzer.history_thumbnails:
            n_thumbnails = len(analyzer.history_thumbnails)
            n_cols = 4
            n_rows = (n_thumbnails + n_cols - 1) // n_cols
            
            for row in range(n_rows):
                cols = st.columns(n_cols)
                for col in range(n_cols):
                    idx = row * n_cols + col
                    if idx < n_thumbnails:
                        thumb_info = analyzer.history_thumbnails[idx]
                        with cols[col]:
                            st.markdown(f"**{thumb_info['name']}**")
                            st.markdown(f"<small>{thumb_info['timestamp']}</small>", unsafe_allow_html=True)
                            if thumb_info['thumbnail']:
                                st.image(f"data:image/png;base64,{thumb_info['thumbnail']}", use_column_width=True)
        
        # Selection interface
        st.markdown("#### ✅ Select Materials to Compare")
        
        selection_options = []
        for i, res in enumerate(analyzer.results_history):
            label = f"{res['name']} | {res['tdb_file']} | {len(res['data'])} pts"
            selection_options.append((i, label))
        
        selected_labels = st.multiselect(
            "Select up to 8 materials for comparison:",
            [label for _, label in selection_options],
            default=[selection_options[i][1] for i in range(min(3, len(selection_options)))],
            max_selections=8,
            key="tab3_material_select"
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
        
        # Comparison options
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            show_plotly = st.checkbox("Show interactive plot (Plotly)", value=True, key="tab3_plotly")
            normalize_data = st.checkbox("Normalize enthalpy to zero at start", value=False, key="tab3_normalize")
        with col_opt2:
            compare_heat_capacity = st.checkbox("Compare heat capacities", value=True, key="tab3_heat_cap")
            show_statistics = st.checkbox("Show detailed statistics", value=True, key="tab3_stats")
        
        # Apply customizations
        customizations = analyzer.plot_customizations
        
        # Create and display comparison visualization
        if st.button("📊 Generate Comparison", type="primary", key="tab3_gen_comparison"):
            with st.spinner("Generating comparison visualization..."):
                fig = create_comparison_visualization(analyzer, selected_indices, customizations)
                if fig:
                    with st.container():
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        st.pyplot(fig)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive Plotly visualization
        if show_plotly and selected_indices:
            st.markdown("#### 📈 Interactive Visualization")
            result = analyzer.results_history[selected_indices[0]]
            data = result['data']
            composition = result['composition']
            material_name = result['name']
            
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
        if show_statistics:
            st.markdown("---")
            st.markdown("#### 📊 Comparison Summary Table")
            
            summary_data = []
            for idx in selected_indices:
                res = analyzer.results_history[idx]
                data = res['data']
                
                min_h = data['Enthalpy_J_mol'].min()
                max_h = data['Enthalpy_J_mol'].max()
                delta_h = max_h - min_h
                avg_slope = delta_h / (data['Temperature_K'].max() - data['Temperature_K'].min())
                
                if len(data) > 1 and 'Heat_Capacity_J_per_mol_K' in data.columns:
                    heat_capacity = data['Heat_Capacity_J_per_mol_K']
                    avg_cp = heat_capacity.mean()
                    max_cp = heat_capacity.max()
                else:
                    avg_cp = 0
                    max_cp = 0
                
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
                    'Avg Cₚ (J/mol·K)': f"{avg_cp:.2f}",
                    'Max Cₚ (J/mol·K)': f"{max_cp:.2f}",
                    'Tm (K)': f"{Tm:.1f}" if Tm else "N/A"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.markdown('<div class="data-table">', unsafe_allow_html=True)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Download options
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
        st.markdown("#### 📥 Download Comparison Data")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            combined_data = {'Temperature_K': analyzer.results_history[selected_indices[0]]['data']['Temperature_K'].values}
            
            for idx in selected_indices:
                res = analyzer.results_history[idx]
                name_clean = res['name'].replace(' ', '_').replace('-', '_')
                combined_data[f"{name_clean}_H_molar"] = res['data']['Enthalpy_J_mol'].values
                combined_data[f"{name_clean}_H_specific"] = res['data']['Enthalpy_J_kg'].values
                
                if 'Heat_Capacity_J_per_mol_K' in res['data'].columns:
                    combined_data[f"{name_clean}_Heat_Capacity"] = res['data']['Heat_Capacity_J_per_mol_K'].values
            
            combined_df = pd.DataFrame(combined_data)
            
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                "📄 Combined Data (CSV)",
                data=combined_df.to_csv(index=False),
                file_name=f"multi_material_comparison_{timestamp_str}.csv",
                mime="text/csv",
                key=f"tab3_combined_csv_{timestamp_str}"
            )
        
        with col_c2:
            if show_statistics:
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    "📄 Summary Table (CSV)",
                    data=summary_df.to_csv(index=False),
                    file_name=f"comparison_summary_{timestamp_str}.csv",
                    mime="text/csv",
                    key=f"tab3_summary_csv_{timestamp_str}"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ==================== TAB 4: Phase Diagram Analysis ====================
    with tab4:
        st.markdown('<div class="sub-header">📈 Phase Diagram Analysis</div>', unsafe_allow_html=True)
        
        if not analyzer.results_history:
            st.info("💡 No computed data available. Please perform calculations in the 'Enthalpy Computation' tab first.")
            st.stop()
        
        # Material selection
        st.markdown("#### 🔍 Select Materials for Phase Analysis")
        
        selection_options = []
        for i, res in enumerate(analyzer.results_history):
            label = f"{res['name']} | {res['tdb_file']} | {len(res['data'])} pts"
            selection_options.append((i, label))
        
        selected_labels = st.multiselect(
            "Select materials for phase analysis:",
            [label for _, label in selection_options],
            default=[selection_options[i][1] for i in range(min(3, len(selection_options)))],
            max_selections=6,
            key="tab4_material_select"
        )
        
        if not selected_labels:
            st.warning("⚠️ Please select at least one material for analysis")
            st.stop()
        
        selected_indices = []
        for label in selected_labels:
            for idx, opt_label in selection_options:
                if opt_label == label:
                    selected_indices.append(idx)
                    break
        
        # Analysis options
        col_ana1, col_ana2 = st.columns(2)
        with col_ana1:
            show_derivatives = st.checkbox("Show derivatives", value=True, key="tab4_derivatives")
            show_cumulative = st.checkbox("Show cumulative enthalpy", value=False, key="tab4_cumulative")
        with col_ana2:
            normalize_plots = st.checkbox("Normalize temperature range", value=False, key="tab4_normalize")
            highlight_transitions = st.checkbox("Highlight phase transitions", value=True, key="tab4_transitions")
        
        # Generate phase diagram
        if st.button("📈 Generate Phase Diagram", type="primary", key="tab4_gen_phase"):
            with st.spinner("Generating phase diagram..."):
                fig = create_phase_diagram_visualization(analyzer, selected_indices, analyzer.plot_customizations)
                if fig:
                    with st.container():
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        st.pyplot(fig)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # Phase transition analysis
        st.markdown("---")
        st.markdown("#### 🔬 Phase Transition Analysis")
        
        transition_data = []
        for idx in selected_indices:
            res = analyzer.results_history[idx]
            data = res['data']
            name = res['name']
            
            if len(data) > 1:
                dT = data['Temperature_K'].diff()
                dH = data['Enthalpy_J_mol'].diff()
                heat_capacity = dH / dT
                
                if len(heat_capacity) > 10:
                    heat_capacity_smooth = heat_capacity.rolling(window=5, center=True, min_periods=1).mean()
                    
                    from scipy.signal import find_peaks
                    peaks, properties = find_peaks(heat_capacity_smooth.iloc[1:].values, 
                                                  height=heat_capacity_smooth.mean(),
                                                  distance=len(heat_capacity_smooth)//10)
                    
                    if len(peaks) > 0:
                        for peak_idx in peaks:
                            T_transition = data['Temperature_K'].iloc[peak_idx + 1]
                            Cp_peak = heat_capacity_smooth.iloc[peak_idx + 1]
                            
                            transition_data.append({
                                'Material': name,
                                'Transition Temp (K)': f"{T_transition:.1f}",
                                'Peak Cₚ (J/mol·K)': f"{Cp_peak:.2f}",
                                'Strength': 'Strong' if Cp_peak > 2*heat_capacity_smooth.mean() else 'Weak'
                            })
        
        if transition_data:
            transition_df = pd.DataFrame(transition_data)
            st.markdown('<div class="data-table">', unsafe_allow_html=True)
            st.dataframe(transition_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No clear phase transitions detected in the selected materials.")

    # ==================== TAB 5: Visualization Customization ====================
    with tab5:
        st.markdown('<div class="sub-header">🎨 Visualization Customization</div>', unsafe_allow_html=True)
        st.markdown("Customize the appearance of all plots in the application.")
        
        # Create tabs within customization
        cust_tab1, cust_tab2, cust_tab3 = st.tabs(["📊 Plot Settings", "🎨 Color & Style", "📐 Layout & Export"])
        
        with cust_tab1:
            st.markdown("#### 📊 Line & Marker Settings")
            col_cust1, col_cust2 = st.columns(2)
            
            with col_cust1:
                # Use widget state management for sliders
                curve_thickness_key = "tab5_curve_thickness"
                if curve_thickness_key not in st.session_state.widget_state:
                    st.session_state.widget_state[curve_thickness_key] = analyzer.plot_customizations['curve_thickness']
                
                analyzer.plot_customizations['curve_thickness'] = st.slider(
                    "Curve Thickness",
                    0.5, 5.0, 
                    st.session_state.widget_state[curve_thickness_key], 
                    0.1,
                    help="Thickness of plot lines",
                    key=curve_thickness_key
                )
                st.session_state.widget_state[curve_thickness_key] = analyzer.plot_customizations['curve_thickness']
                
                box_thickness_key = "tab5_box_thickness"
                if box_thickness_key not in st.session_state.widget_state:
                    st.session_state.widget_state[box_thickness_key] = analyzer.plot_customizations['box_thickness']
                
                analyzer.plot_customizations['box_thickness'] = st.slider(
                    "Box/Spine Thickness",
                    0.5, 5.0, 
                    st.session_state.widget_state[box_thickness_key], 
                    0.1,
                    help="Thickness of plot borders",
                    key=box_thickness_key
                )
                st.session_state.widget_state[box_thickness_key] = analyzer.plot_customizations['box_thickness']
                
                marker_size_key = "tab5_marker_size"
                if marker_size_key not in st.session_state.widget_state:
                    st.session_state.widget_state[marker_size_key] = analyzer.plot_customizations['marker_size']
                
                analyzer.plot_customizations['marker_size'] = st.slider(
                    "Marker Size",
                    2, 15, 
                    st.session_state.widget_state[marker_size_key], 
                    1,
                    help="Size of data point markers",
                    key=marker_size_key
                )
                st.session_state.widget_state[marker_size_key] = analyzer.plot_customizations['marker_size']
                
                grid_alpha_key = "tab5_grid_alpha"
                if grid_alpha_key not in st.session_state.widget_state:
                    st.session_state.widget_state[grid_alpha_key] = analyzer.plot_customizations['grid_alpha']
                
                analyzer.plot_customizations['grid_alpha'] = st.slider(
                    "Grid Transparency",
                    0.0, 1.0, 
                    st.session_state.widget_state[grid_alpha_key], 
                    0.05,
                    help="Transparency of grid lines",
                    key=grid_alpha_key
                )
                st.session_state.widget_state[grid_alpha_key] = analyzer.plot_customizations['grid_alpha']
            
            with col_cust2:
                font_size_key = "tab5_font_size"
                if font_size_key not in st.session_state.widget_state:
                    st.session_state.widget_state[font_size_key] = analyzer.plot_customizations['font_size']
                
                analyzer.plot_customizations['font_size'] = st.slider(
                    "Font Size",
                    8, 20, 
                    st.session_state.widget_state[font_size_key], 
                    1,
                    help="Base font size for labels",
                    key=font_size_key
                )
                st.session_state.widget_state[font_size_key] = analyzer.plot_customizations['font_size']
                
                title_font_size_key = "tab5_title_font_size"
                if title_font_size_key not in st.session_state.widget_state:
                    st.session_state.widget_state[title_font_size_key] = analyzer.plot_customizations['title_font_size']
                
                analyzer.plot_customizations['title_font_size'] = st.slider(
                    "Title Font Size",
                    10, 24, 
                    st.session_state.widget_state[title_font_size_key], 
                    1,
                    help="Font size for titles",
                    key=title_font_size_key
                )
                st.session_state.widget_state[title_font_size_key] = analyzer.plot_customizations['title_font_size']
                
                legend_font_size_key = "tab5_legend_font_size"
                if legend_font_size_key not in st.session_state.widget_state:
                    st.session_state.widget_state[legend_font_size_key] = analyzer.plot_customizations['legend_font_size']
                
                analyzer.plot_customizations['legend_font_size'] = st.slider(
                    "Legend Font Size",
                    8, 16, 
                    st.session_state.widget_state[legend_font_size_key], 
                    1,
                    help="Font size for legend text",
                    key=legend_font_size_key
                )
                st.session_state.widget_state[legend_font_size_key] = analyzer.plot_customizations['legend_font_size']
                
                analyzer.plot_customizations['legend_location'] = st.selectbox(
                    "Legend Location",
                    ['best', 'upper right', 'upper left', 'lower left', 'lower right',
                     'right', 'center left', 'center right', 'lower center', 
                     'upper center', 'center'],
                    index=['best', 'upper right', 'upper left', 'lower left', 'lower right',
                          'right', 'center left', 'center right', 'lower center', 
                          'upper center', 'center'].index(analyzer.plot_customizations['legend_location']),
                    help="Position of the legend",
                    key="tab5_legend_location"
                )
        
        with cust_tab2:
            st.markdown("#### 🎨 Color & Style Settings")
            col_style1, col_style2 = st.columns(2)
            
            with col_style1:
                analyzer.plot_customizations['colormap'] = st.selectbox(
                    "Select Colormap",
                    COLORMAPS,
                    index=COLORMAPS.index(analyzer.plot_customizations['colormap']),
                    help="Color scheme for plots",
                    key="tab5_colormap"
                )
                
                # Display colormap preview
                fig_cmap, ax_cmap = plt.subplots(figsize=(8, 1))
                try:
                    cmap = plt.get_cmap(analyzer.plot_customizations['colormap'])
                    gradient = np.linspace(0, 1, 256).reshape(1, -1)
                    ax_cmap.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, 0, 1])
                except:
                    cmap = plt.get_cmap('viridis')
                    gradient = np.linspace(0, 1, 256).reshape(1, -1)
                    ax_cmap.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, 0, 1])
                
                ax_cmap.set_xticks([])
                ax_cmap.set_yticks([])
                ax_cmap.set_title(f"Preview: {analyzer.plot_customizations['colormap']}", fontsize=10)
                st.pyplot(fig_cmap)
                plt.close(fig_cmap)
                
                available_styles = plt.style.available
                analyzer.plot_customizations['plot_style'] = st.selectbox(
                    "Plot Style",
                    ['default'] + available_styles,
                    index=0,
                    help="Matplotlib style for plots",
                    key="tab5_plot_style"
                )
            
            with col_style2:
                tick_size_key = "tab5_tick_size"
                if tick_size_key not in st.session_state.widget_state:
                    st.session_state.widget_state[tick_size_key] = analyzer.plot_customizations['tick_size']
                
                analyzer.plot_customizations['tick_size'] = st.slider(
                    "Tick Label Size",
                    8, 16, 
                    st.session_state.widget_state[tick_size_key], 
                    1,
                    help="Size of tick labels",
                    key=tick_size_key
                )
                st.session_state.widget_state[tick_size_key] = analyzer.plot_customizations['tick_size']
                
                label_fontsize_key = "tab5_label_fontsize"
                if label_fontsize_key not in st.session_state.widget_state:
                    st.session_state.widget_state[label_fontsize_key] = analyzer.plot_customizations['label_fontsize']
                
                analyzer.plot_customizations['label_fontsize'] = st.slider(
                    "Axis Label Size",
                    8, 16, 
                    st.session_state.widget_state[label_fontsize_key], 
                    1,
                    help="Size of axis labels",
                    key=label_fontsize_key
                )
                st.session_state.widget_state[label_fontsize_key] = analyzer.plot_customizations['label_fontsize']
                
                annotation_fontsize_key = "tab5_annotation_fontsize"
                if annotation_fontsize_key not in st.session_state.widget_state:
                    st.session_state.widget_state[annotation_fontsize_key] = analyzer.plot_customizations.get('annotation_fontsize', 9)
                
                analyzer.plot_customizations['annotation_fontsize'] = st.slider(
                    "Annotation Font Size",
                    6, 14, 
                    st.session_state.widget_state[annotation_fontsize_key], 
                    1,
                    help="Size of annotation text",
                    key=annotation_fontsize_key
                )
                st.session_state.widget_state[annotation_fontsize_key] = analyzer.plot_customizations['annotation_fontsize']
                
                analyzer.plot_customizations['transparent_bg'] = st.checkbox(
                    "Transparent Background",
                    value=analyzer.plot_customizations.get('transparent_bg', False),
                    help="Use transparent background for plots",
                    key="tab5_transparent_bg"
                )
        
        with cust_tab3:
            st.markdown("#### 📐 Layout & Export Settings")
            col_layout1, col_layout2 = st.columns(2)
            
            with col_layout1:
                # FIXED: Using unique keys for sliders with widget state management
                figure_width_key = "tab5_figure_width"
                if figure_width_key not in st.session_state.widget_state:
                    st.session_state.widget_state[figure_width_key] = analyzer.plot_customizations['figure_width']
                
                analyzer.plot_customizations['figure_width'] = st.slider(
                    "Figure Width (inches)",
                    8.0, 20.0, 
                    st.session_state.widget_state[figure_width_key], 
                    0.5,
                    help="Width of the figure in inches",
                    key=figure_width_key
                )
                st.session_state.widget_state[figure_width_key] = analyzer.plot_customizations['figure_width']
                
                figure_height_key = "tab5_figure_height"
                if figure_height_key not in st.session_state.widget_state:
                    st.session_state.widget_state[figure_height_key] = analyzer.plot_customizations['figure_height']
                
                analyzer.plot_customizations['figure_height'] = st.slider(
                    "Figure Height (inches)",
                    6.0, 16.0, 
                    st.session_state.widget_state[figure_height_key], 
                    0.5,
                    help="Height of the figure in inches",
                    key=figure_height_key
                )
                st.session_state.widget_state[figure_height_key] = analyzer.plot_customizations['figure_height']
            
            with col_layout2:
                dpi_key = "tab5_dpi"
                if dpi_key not in st.session_state.widget_state:
                    st.session_state.widget_state[dpi_key] = analyzer.plot_customizations.get('dpi', 150)
                
                analyzer.plot_customizations['dpi'] = st.slider(
                    "Figure DPI",
                    72, 300, 
                    st.session_state.widget_state[dpi_key], 
                    10,
                    help="Resolution of exported figures",
                    key=dpi_key
                )
                st.session_state.widget_state[dpi_key] = analyzer.plot_customizations['dpi']
                
                export_format = st.selectbox(
                    "Default Export Format",
                    ['PNG', 'PDF', 'SVG', 'EPS'],
                    index=0,
                    help="Default format for figure exports",
                    key="tab5_export_format"
                )
        
        # Save and reset buttons
        st.markdown("---")
        col_save1, col_save2, col_save3 = st.columns(3)
        
        with col_save1:
            if st.button("💾 Save Customizations", use_container_width=True, key="tab5_save_cust"):
                st.success("✅ Customizations saved!")
                st.rerun()
        
        with col_save2:
            if st.button("🔄 Reset to Defaults", use_container_width=True, key="tab5_reset_cust"):
                analyzer.plot_customizations = {
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
                # Clear widget states
                for key in list(st.session_state.widget_state.keys()):
                    if key.startswith("tab5_"):
                        del st.session_state.widget_state[key]
                st.success("✅ Customizations reset to defaults!")
                st.rerun()
        
        with col_save3:
            if st.button("🎨 Apply to All Plots", use_container_width=True, key="tab5_apply_cust"):
                st.info("Customizations will be applied to all new plots.")
                st.rerun()
        
        # Export/Import customizations
        st.markdown("---")
        st.markdown("#### 📤 Export/Import Customizations")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            json_custom = json.dumps(analyzer.plot_customizations, indent=4)
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                "📄 Export Settings (JSON)",
                data=json_custom,
                file_name=f"plot_customizations_{timestamp_str}.json",
                mime="application/json",
                key=f"tab5_export_settings_{timestamp_str}"
            )
        
        with col_exp2:
            uploaded_custom = st.file_uploader(
                "Import settings from JSON",
                type=['json'],
                key="tab5_upload_custom"
            )
            
            if uploaded_custom is not None:
                try:
                    imported = json.load(uploaded_custom)
                    analyzer.plot_customizations.update(imported)
                    st.success("✅ Customizations imported successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error importing customizations: {str(e)}")

    # ==================== TAB 6: Help & Settings ====================
    with tab6:
        st.markdown('<div class="sub-header">ℹ️ Help & Application Settings</div>', unsafe_allow_html=True)
        
        col_h1, col_h2 = st.columns([1, 1.2])
        
        with col_h1:
            st.markdown("#### 📖 User Guide")
            
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
                
                4. **Phase Diagram Analysis**
                   - Compare phase transitions
                   - Analyze heat capacity peaks
                
                5. **Visualization Customization**
                   - Customize plot appearance
                   - Choose colormaps
                   - Adjust figure sizes and fonts
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
            st.markdown("#### ⚙️ Application Management")
            
            # Session data status
            st.markdown("##### 📈 Current Session Status")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Computed Results", len(analyzer.results_history))
            with col_stat2:
                st.metric("Fitting Results", len(analyzer.fitting_results))
            with col_stat3:
                st.metric("TDB Files", len(analyzer.get_available_tdb_files()))
            
            # Data management
            st.markdown("##### 💾 Data Management")
            col_data1, col_data2 = st.columns(2)
            
            with col_data1:
                if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True, key="tab6_clear_data"):
                    analyzer.results_history = []
                    analyzer.fitting_results = []
                    analyzer.history_thumbnails = []
                    st.session_state.simulation_cache = {}
                    st.session_state.widget_state = {}
                    st.success("✅ All session data cleared!")
                    st.rerun()
            
            with col_data2:
                if st.button("💾 Export Session", type="secondary", use_container_width=True, key="tab6_export_session"):
                    session_data = {
                        'results_history': analyzer.results_history,
                        'fitting_results': analyzer.fitting_results,
                        'simulation_cache': st.session_state.simulation_cache,
                        'customizations': analyzer.plot_customizations,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'version': 'Thermodynamic Enthalpy Analyzer Pro v3.0'
                    }
                    json_data = json.dumps(session_data, indent=4, default=str)
                    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        "📥 Download Session",
                        data=json_data,
                        file_name=f"enthalpy_analyzer_session_{timestamp_str}.json",
                        mime="application/json",
                        key=f"tab6_session_download_{timestamp_str}"
                    )
            
            # Database management
            st.markdown("##### 🗄️ Database Management")
            tdb_files = analyzer.get_available_tdb_files()
            
            if tdb_files:
                st.write(f"**Found {len(tdb_files)} TDB files:**")
                
                for tdb in tdb_files:
                    col_db1, col_db2, col_db3 = st.columns([3, 1, 1])
                    with col_db1:
                        file_path = analyzer.database_dir / tdb
                        size_kb = os.path.getsize(file_path) / 1024
                        st.caption(f"`{tdb}` ({size_kb:.1f} KB)")
                    with col_db2:
                        if st.button("ℹ️", key=f"tab6_info_{tdb}"):
                            st.info(f"**{tdb}**\nSize: {size_kb:.1f} KB\nPath: {file_path}")
                    with col_db3:
                        if st.button("🗑️", key=f"tab6_delete_{tdb}"):
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
            st.markdown("##### 🧪 Periodic Table Explorer")
            
            selected_element = st.selectbox(
                "Explore element properties:",
                sorted(PERIODIC_TABLE.keys()),
                format_func=lambda x: f"{PERIODIC_TABLE[x][0]} - {PERIODIC_TABLE[x][1]}",
                key="tab6_element_explorer"
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
            st.markdown("##### ℹ️ About")
            st.markdown("""
            **Thermodynamic Enthalpy Analyzer Pro v3.0**
            
            A comprehensive tool for thermodynamic calculations using CALPHAD method.
            
            **Key Features:**
            - **Session State Preservation**: Data persists across downloads and page interactions
            - **Enhanced Visualizations**: Publication-quality plots with extensive customization
            - **Advanced Analysis**: Curve fitting, multi-material comparison, phase analysis
            - **Interactive Plots**: Plotly integration for dynamic exploration
            
            **Core Libraries:**
            - pycalphad, scipy, xarray
            - matplotlib, plotly, streamlit
            
            **Session State Management:**
            - Results are cached to prevent re-computation
            - Widget keys are properly managed to avoid conflicts
            - All data persists across tab switches and downloads
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div class="session-info">'
        f'🔥 Thermodynamic Enthalpy Analyzer Pro v3.0 | '
        f'Session: {len(analyzer.results_history)} results, {len(analyzer.fitting_results)} fits | '
        f'Cache: {len(st.session_state.simulation_cache)} simulations | '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
