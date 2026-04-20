#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Elmer FEM .sif Generator for Dissimilar Material Welding - ENHANCED PERSISTENCE EDITION
- ALL HEAT SOURCE TYPES: Travelling Gaussian, Fixed Gaussian, Flat-Top, Double Ellipsoidal, Custom Gaussian
- FIXED GEOMETRY: Only specified faces/solids from Mesh_1_dimensions
- LINKED UDF SYSTEM: Expressions auto-update Fortran UDFs including Specific Enthalpy
- BULLETPROOF MULTISELECTS: Defensive filtering prevents invalid default errors
- COMPLETE DOWNLOAD: ZIP bundling + persistent session state for all generated files
- INTELLIGENT CACHING: @st.cache_data prevents redundant re-computation
- STATE PRESERVATION: Downloads never break session state; all files always accessible
- EQUATION-BASED ENTHALPY: Optional analytical H(T) model replaces .dat lookup tables
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re
import zipfile
import io
import math

# Page config
st.set_page_config(
    page_title="Elmer Weld Generator - Full",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stCodeBlock { background-color: #f8f9fa; }
    .stTextArea textarea { font-family: monospace; font-size: 0.85em; }
    .metric-card { background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
    .download-section { background: #e8f4fd; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .warning-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 0.25rem; }
    .success-box { background: #d4edda; border-left: 4px solid #28a745; padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 0.25rem; }
</style>
""", unsafe_allow_html=True)

st.title("🔥 Elmer FEM Generator for Dissimilar Welding")
st.markdown("""
**Generate complete `.sif` input files + Fortran UDFs + lookup tables for Al-Cu welding simulations.**
_Mesh file supplied externally with fixed geometry entities. Equation-based enthalpy UDF now available._
""")

# ====================== HELPER: UNIQUE KEY GENERATOR ======================
def uk(section: str, var: str, suffix: str = "") -> str:
    """Generate unique key for Streamlit widgets: section_var_suffix"""
    return f"{section}_{var}_{suffix}".strip("_")

# ====================== HELPER: DEFENSIVE MULTISELECT ======================
def safe_multiselect(label, options, default=None, key=None, **kwargs):
    """
    Bulletproof multiselect that filters invalid defaults.
    Prevents StreamlitAPIException when default values aren't in options.
    """
    if default is None:
        default = []
    valid_defaults = [d for d in default if d in options]
    if len(valid_defaults) < len(default):
        invalid = set(default) - set(options)
        st.warning(f"⚠️ Removed invalid defaults for '{label}': {invalid}")
    return st.multiselect(label, options, default=valid_defaults, key=key, **kwargs)

# ====================== FIXED GEOMETRY ENTITIES ======================
SOLID_NAMES = ["Solid_1front", "Solid_2back"]
FACE_NAMES = [
    "Face_1leftfront", "Face_2leftback", "Face_3frontfront", 
    "Face_4bottomfront", "Face_5topfront", "Face_6interfacefront",
    "Face_7bottomback", "Face_8topback", "Face_9backback",
    "Face_10rightfront", "Face_11rightback"
]

# ====================== SESSION STATE INITIALIZATION (COMPREHENSIVE) ======================
# Initialize ALL session state keys needed for persistence
if "generated_content" not in st.session_state:
    st.session_state.generated_content = {}
if "generation_timestamp" not in st.session_state:
    st.session_state.generation_timestamp = None
if "table_data_visc_a" not in st.session_state:
    st.session_state.table_data_visc_a = None
if "table_data_visc_b" not in st.session_state:
    st.session_state.table_data_visc_b = None
if "table_data_enth_a" not in st.session_state:
    st.session_state.table_data_enth_a = None
if "table_data_enth_b" not in st.session_state:
    st.session_state.table_data_enth_b = None
if "last_generation_params" not in st.session_state:
    st.session_state.last_generation_params = {}
if "use_enthalpy_udf_a" not in st.session_state:
    st.session_state.use_enthalpy_udf_a = False
if "use_enthalpy_udf_b" not in st.session_state:
    st.session_state.use_enthalpy_udf_b = False

# ====================== CACHED HELPER FUNCTIONS ======================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def parse_expression_cached(expr: str, temp_var: str = "temp") -> str:
    """Cached version of expression parser for Fortran code generation"""
    expr_fortran = expr.replace("T", temp_var)
    expr_fortran = re.sub(r'(\d+\.\d+)', r'\1_dp', expr_fortran)
    expr_fortran = re.sub(r'(?<!\.)(\b\d+\b)(?!\.)', r'\1.0_dp', expr_fortran)
    return expr_fortran

@st.cache_data(ttl=3600)
def extract_udf_code_cached(code_block: str) -> str:
    """Cached UDF code extractor - removes markdown formatting"""
    lines = code_block.split('\n')
    while lines and lines[0].strip() == '':
        lines.pop(0)
    while lines and lines[-1].strip() == '':
        lines.pop()
    return '\n'.join(lines)

@st.cache_data(ttl=3600)
def face_names_to_indices_cached(face_list: list) -> str:
    """Cached face name to index converter"""
    indices = []
    for face in face_list:
        match = re.search(r'Face_(\d+)', face)
        if match:
            indices.append(match.group(1))
    return " ".join(indices) if indices else "1"

@st.cache_data(ttl=3600)
def compute_enthalpy_preview_cached(T: float, alpha: float, beta1: float, beta2: float, 
                                    beta3: float, gamma: float, T0: float, C: float) -> float:
    """Cached enthalpy preview calculator for UI display"""
    linear_term = beta1 * T
    max_val = T - T0
    phase_term = beta2 * max_val if max_val > 0.0 else 0.0
    exp_arg = -gamma * (T - T0)
    if exp_arg > 700.0:
        sigmoid_term = 0.0
    elif exp_arg < -700.0:
        sigmoid_term = beta3
    else:
        exp_val = math.exp(exp_arg)
        denom = 1.0 + exp_val
        sigmoid_term = beta3 / denom if denom > 1.0e-300 else beta3
    bracket_sum = linear_term + phase_term + sigmoid_term + C
    if abs(alpha) < 1.0e-300:
        return 0.0
    return bracket_sum / alpha

# ====================== SIDEBAR: GLOBAL SETTINGS ======================
st.sidebar.header("⚙️ Global Settings")

project_name = st.sidebar.text_input("Project Name", value="Al_Cu_Weld", key=uk("global", "project"))
author = st.sidebar.text_input("Author", value="Your Name", key=uk("global", "author"))
date_str = datetime.now().strftime("%Y-%m-%d")

st.sidebar.subheader("📁 Output Files")
sif_filename = st.sidebar.text_input(".sif Filename", value=f"{project_name.lower()}.sif", key=uk("out", "sif"))
fortran_dir = st.sidebar.text_input("Fortran UDF Directory", value="./udfs/", key=uk("out", "f90dir"))
table_dir_visc = st.sidebar.text_input("Viscosity Table Dir", value="./viscosity/", key=uk("out", "viscdir"))
table_dir_enth = st.sidebar.text_input("Enthalpy Table Dir", value="./specific_enthalpy/", key=uk("out", "enthdir"))

# Sidebar: Quick access to downloads if content exists
if st.session_state.generated_content:
    st.sidebar.success("✅ Files generated!")
    if st.sidebar.button("📥 Go to Downloads", key="sidebar_goto_downloads"):
        st.session_state.active_tab = "generate"
else:
    st.sidebar.info("🔄 Configure parameters and click 'Generate All Files'")

# ====================== TABS NAVIGATION WITH STATE TRACKING ======================
tab_materials, tab_heat, tab_tables, tab_physics, tab_generate = st.tabs([
    "🧪 Materials & UDFs",
    "🔦 Heat Source", 
    "📊 Lookup Tables",
    "⚙️ Physics & BCs",
    "📥 Generate Files"
])

# Track active tab for better UX
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "materials"

# ====================== TAB 1: MATERIALS & FORTRAN UDFs (LINKED SYSTEM) ======================
with tab_materials:
    st.header("🧪 Material Properties & Linked Fortran UDFs")
    st.info("🔹 **Expressions automatically update UDFs** - edit property expressions below to see UDF code change in real-time!")
    
    mat_col1, mat_col2 = st.columns(2)
    
    # ----- MATERIAL A -----
    with mat_col1:
        st.subheader("Material A (Solid_1front)")
        mat_a_name = st.text_input("Material Name", value="AA6061_Al", key=uk("matA", "name"))
        mat_a_melting = st.number_input("Melting Point [K]", value=933.5, step=0.1, key=uk("matA", "tmelt"))
        
        st.markdown("**Temperature-Dependent Property Expressions** (T in Kelvin):")
        
        st.markdown("🔹 Density ρ(T) [kg/m³]")
        dens_a_s_expr = st.text_input("Solid phase: ρ = ", value="2700.0 - 0.11*(T - 298.0)", key=uk("matA", "dens_s_expr"))
        dens_a_l_expr = st.text_input("Liquid phase: ρ = ", value="2380.0 - 0.28*(T - 933.5)", key=uk("matA", "dens_l_expr"))
        
        st.markdown("🔹 Thermal Conductivity k(T) [W/(m·K)]")
        cond_a_s_expr = st.text_input("Solid: k = ", value="167.0 + 0.12*(T - 298.0)", key=uk("matA", "cond_s_expr"))
        cond_a_l_expr = st.text_input("Liquid: k = ", value="90.0 - 0.012*(T - 933.5)", key=uk("matA", "cond_l_expr"))
        
        st.markdown("🔹 CTE α(T) [1/K]")
        cte_a_s_expr = st.text_input("Solid: α = ", value="23.0e-6 + 2.1e-8*(T - 298.0)", key=uk("matA", "cte_s_expr"))
        cte_a_l_expr = st.text_input("Liquid: α = ", value="0.0", key=uk("matA", "cte_l_expr"))
        
        latent_a = st.number_input("Latent Heat of Fusion L_f [J/kg]", value=3.97e5, step=1e3, format="%.0f", key=uk("matA", "latent"))
        
        st.divider()
        st.markdown("### 🔧 Auto-Generated Fortran UDF – Material A")
        st.caption("This UDF is automatically generated from your expressions above")
        
        # Use cached parser
        dens_udf_a = f"""FUNCTION getDensity_{mat_a_name}(model, n, temp) RESULT(denst)
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t) :: model
  INTEGER :: n
  REAL(KIND=dp) :: temp, denst, tscaler
  REAL(KIND=dp) :: refSolDenst, refLiqDenst, refTemp, alphas, alphal
  LOGICAL :: GotIt
  TYPE(ValueList_t), POINTER :: material

  material => GetMaterial()
  IF (.NOT. ASSOCIATED(material)) CALL Fatal('getDensity', 'No material found')

  refSolDenst = GetConstReal(material, 'Reference Density Solid {mat_a_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getDensity', 'Ref density solid not found')
  alphas = GetConstReal(material, 'Density Coeff Solid {mat_a_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getDensity', 'Density coeff solid not found')
  
  refLiqDenst = GetConstReal(material, 'Reference Density Liquid {mat_a_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getDensity', 'Ref density liquid not found')
  alphal = GetConstReal(material, 'Density Coefficient Liquid {mat_a_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getDensity', 'Density coeff liquid not found')

  refTemp = GetConstReal(material, 'Melting Point Temperature of {mat_a_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getDensity', 'Melting point not found')
  tscaler = GetConstReal(material, 'Tscaler', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getDensity', 'Tscaler not found')

  ! Auto-generated from expression: {dens_a_s_expr} (solid), {dens_a_l_expr} (liquid)
  IF (refTemp <= temp) THEN
      CALL Warn('getDensity', 'Material A in liquid state.')
      denst = {parse_expression_cached(dens_a_l_expr)}
  ELSE
      denst = {parse_expression_cached(dens_a_s_expr)}
  END IF
END FUNCTION getDensity_{mat_a_name}"""
        
        st.code(dens_udf_a, language="fortran")
        
        cond_udf_a = f"""FUNCTION getThermalConductivity_{mat_a_name}(model, n, temp) RESULT(thcondt)
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t) :: model; INTEGER :: n; REAL(KIND=dp) :: temp, thcondt, tscaler
  REAL(KIND=dp) :: refSolThCond, refLiqThCond, refTemp, alphas, betas, alphal
  LOGICAL :: GotIt
  TYPE(ValueList_t), POINTER :: material

  material => GetMaterial()
  IF (.NOT. ASSOCIATED(material)) CALL Fatal('getThermalConductivity', 'No material found')

  refSolThCond = GetConstReal(material, 'Reference Thermal Conductivity Solid {mat_a_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Ref cond solid not found')
  alphas = GetConstReal(material, 'Cond Coeff As Solid {mat_a_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Coeff A solid not found')
  betas = GetConstReal(material, 'Cond Coeff Bs Solid {mat_a_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Coeff B solid not found')
  
  refLiqThCond = GetConstReal(material, 'Reference Thermal Conductivity Liquid {mat_a_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Ref cond liquid not found')
  alphal = GetConstReal(material, 'Cond Coeff Liquid {mat_a_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Coeff liquid not found')

  refTemp = GetConstReal(material, 'Melting Point Temperature of {mat_a_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Melting point not found')
  tscaler = GetConstReal(material, 'Tscaler', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Tscaler not found')

  ! Auto-generated from expression: {cond_a_s_expr} (solid), {cond_a_l_expr} (liquid)
  IF (refTemp <= temp) THEN
      CALL Warn('getThermalConductivity', 'Material A in liquid state.')
      thcondt = {parse_expression_cached(cond_a_l_expr)}
  ELSE
      thcondt = {parse_expression_cached(cond_a_s_expr)}
  END IF
END FUNCTION getThermalConductivity_{mat_a_name}"""
        
        st.code(cond_udf_a, language="fortran")
        
        cte_udf_a = f"""FUNCTION getThermalExpansivity_{mat_a_name}(model, n, temp) RESULT(expansivity)
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t) :: model; INTEGER :: n; REAL(KIND=dp) :: temp, expansivity, tscaler
  REAL(KIND=dp) :: refSolExp, refTemp, alphas, betas
  LOGICAL :: GotIt
  TYPE(ValueList_t), POINTER :: material

  material => GetMaterial()
  IF (.NOT. ASSOCIATED(material)) CALL Fatal('getThermalExpansivity', 'No material found')

  refSolExp = GetConstReal(material, 'Reference Thermal Expansivity Solid {mat_a_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalExpansivity', 'Ref expansivity solid not found')
  alphas = GetConstReal(material, 'Thermal Expansivity Coeff As Solid {mat_a_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalExpansivity', 'Coeff A expansivity not found')
  betas = GetConstReal(material, 'Thermal Expansivity Coeff Bs Solid {mat_a_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalExpansivity', 'Coeff B expansivity not found')

  refTemp = GetConstReal(material, 'Melting Point Temperature of {mat_a_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalExpansivity', 'Melting point not found')
  tscaler = GetConstReal(material, 'Tscaler', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalExpansivity', 'Tscaler not found')

  ! Auto-generated from expression: {cte_a_s_expr} (solid), {cte_a_l_expr} (liquid)
  IF (refTemp <= temp) THEN
      CALL Warn('getThermalExpansivity', 'Material A in liquid state.')
      expansivity = {parse_expression_cached(cte_a_l_expr)}
  ELSE
      expansivity = {parse_expression_cached(cte_a_s_expr)}
  END IF
END FUNCTION getThermalExpansivity_{mat_a_name}"""
        
        st.code(cte_udf_a, language="fortran")
    
    # ----- MATERIAL B -----
    with mat_col2:
        st.subheader("Material B (Solid_2back)")
        mat_b_name = st.text_input("Material Name", value="T2_Cu", key=uk("matB", "name"))
        mat_b_melting = st.number_input("Melting Point [K]", value=1356.6, step=0.1, key=uk("matB", "tmelt"))
        
        st.markdown("**Temperature-Dependent Property Expressions** (T in Kelvin):")
        
        st.markdown("🔹 Density ρ(T) [kg/m³]")
        dens_b_s_expr = st.text_input("Solid phase: ρ = ", value="8940.0 - 0.52*(T - 298.0)", key=uk("matB", "dens_s_expr"))
        dens_b_l_expr = st.text_input("Liquid phase: ρ = ", value="7992.0 - 0.44*(T - 1356.6)", key=uk("matB", "dens_l_expr"))
        
        st.markdown("🔹 Thermal Conductivity k(T) [W/(m·K)]")
        cond_b_s_expr = st.text_input("Solid: k = ", value="391.0 - 0.052*(T - 298.0)", key=uk("matB", "cond_s_expr"))
        cond_b_l_expr = st.text_input("Liquid: k = ", value="170.0 - 0.025*(T - 1356.6)", key=uk("matB", "cond_l_expr"))
        
        st.markdown("🔹 CTE α(T) [1/K]")
        cte_b_s_expr = st.text_input("Solid: α = ", value="16.4e-6 + 2.5e-8*(T - 298.0)", key=uk("matB", "cte_s_expr"))
        cte_b_l_expr = st.text_input("Liquid: α = ", value="0.0", key=uk("matB", "cte_l_expr"))
        
        latent_b = st.number_input("Latent Heat of Fusion L_f [J/kg]", value=2.05e5, step=1e3, format="%.0f", key=uk("matB", "latent"))
        
        st.divider()
        st.markdown("### 🔧 Auto-Generated Fortran UDF – Material B")
        st.caption("This UDF is automatically generated from your expressions above")
        
        dens_udf_b = f"""FUNCTION getDensity_{mat_b_name}(model, n, temp) RESULT(denst)
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t) :: model; INTEGER :: n; REAL(KIND=dp) :: temp, denst, tscaler
  REAL(KIND=dp) :: refSolDenst, refLiqDenst, refTemp, alphas, alphal
  LOGICAL :: GotIt
  TYPE(ValueList_t), POINTER :: material

  material => GetMaterial()
  IF (.NOT. ASSOCIATED(material)) CALL Fatal('getDensity', 'No material found')

  refSolDenst = GetConstReal(material, 'Reference Density Solid {mat_b_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getDensity', 'Ref density solid not found')
  alphas = GetConstReal(material, 'Density Coeff Solid {mat_b_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getDensity', 'Density coeff solid not found')
  
  refLiqDenst = GetConstReal(material, 'Reference Density Liquid {mat_b_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getDensity', 'Ref density liquid not found')
  alphal = GetConstReal(material, 'Density Coefficient Liquid {mat_b_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getDensity', 'Density coeff liquid not found')

  refTemp = GetConstReal(material, 'Melting Point Temperature of {mat_b_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getDensity', 'Melting point not found')
  tscaler = GetConstReal(material, 'Tscaler', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getDensity', 'Tscaler not found')

  ! Auto-generated from expression: {dens_b_s_expr} (solid), {dens_b_l_expr} (liquid)
  IF (refTemp <= temp) THEN
      CALL Warn('getDensity', 'Material B in liquid state.')
      denst = {parse_expression_cached(dens_b_l_expr)}
  ELSE
      denst = {parse_expression_cached(dens_b_s_expr)}
  END IF
END FUNCTION getDensity_{mat_b_name}"""
        
        st.code(dens_udf_b, language="fortran")
        
        cond_udf_b = f"""FUNCTION getThermalConductivity_{mat_b_name}(model, n, temp) RESULT(thcondt)
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t) :: model; INTEGER :: n; REAL(KIND=dp) :: temp, thcondt, tscaler
  REAL(KIND=dp) :: refSolThCond, refLiqThCond, refTemp, alphas, betas, alphal
  LOGICAL :: GotIt
  TYPE(ValueList_t), POINTER :: material

  material => GetMaterial()
  IF (.NOT. ASSOCIATED(material)) CALL Fatal('getThermalConductivity', 'No material found')

  refSolThCond = GetConstReal(material, 'Reference Thermal Conductivity Solid {mat_b_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Ref cond solid not found')
  alphas = GetConstReal(material, 'Cond Coeff As Solid {mat_b_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Coeff A solid not found')
  betas = GetConstReal(material, 'Cond Coeff Bs Solid {mat_b_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Coeff B solid not found')
  
  refLiqThCond = GetConstReal(material, 'Reference Thermal Conductivity Liquid {mat_b_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Ref cond liquid not found')
  alphal = GetConstReal(material, 'Cond Coeff Liquid {mat_b_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Coeff liquid not found')

  refTemp = GetConstReal(material, 'Melting Point Temperature of {mat_b_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Melting point not found')
  tscaler = GetConstReal(material, 'Tscaler', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Tscaler not found')

  ! Auto-generated from expression: {cond_b_s_expr} (solid), {cond_b_l_expr} (liquid)
  IF (refTemp <= temp) THEN
      CALL Warn('getThermalConductivity', 'Material B in liquid state.')
      thcondt = {parse_expression_cached(cond_b_l_expr)}
  ELSE
      thcondt = {parse_expression_cached(cond_b_s_expr)}
  END IF
END FUNCTION getThermalConductivity_{mat_b_name}"""
        
        st.code(cond_udf_b, language="fortran")
        
        cte_udf_b = f"""FUNCTION getThermalExpansivity_{mat_b_name}(model, n, temp) RESULT(expansivity)
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t) :: model; INTEGER :: n; REAL(KIND=dp) :: temp, expansivity, tscaler
  REAL(KIND=dp) :: refSolExp, refTemp, alphas, betas
  LOGICAL :: GotIt
  TYPE(ValueList_t), POINTER :: material

  material => GetMaterial()
  IF (.NOT. ASSOCIATED(material)) CALL Fatal('getThermalExpansivity', 'No material found')

  refSolExp = GetConstReal(material, 'Reference Thermal Expansivity Solid {mat_b_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalExpansivity', 'Ref expansivity solid not found')
  alphas = GetConstReal(material, 'Thermal Expansivity Coeff As Solid {mat_b_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalExpansivity', 'Coeff A expansivity not found')
  betas = GetConstReal(material, 'Thermal Expansivity Coeff Bs Solid {mat_b_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalExpansivity', 'Coeff B expansivity not found')

  refTemp = GetConstReal(material, 'Melting Point Temperature of {mat_b_name}', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalExpansivity', 'Melting point not found')
  tscaler = GetConstReal(material, 'Tscaler', GotIt)
  IF(.NOT. GotIt) CALL Fatal('getThermalExpansivity', 'Tscaler not found')

  ! Auto-generated from expression: {cte_b_s_expr} (solid), {cte_b_l_expr} (liquid)
  IF (refTemp <= temp) THEN
      CALL Warn('getThermalExpansivity', 'Material B in liquid state.')
      expansivity = {parse_expression_cached(cte_b_l_expr)}
  ELSE
      expansivity = {parse_expression_cached(cte_b_s_expr)}
  END IF
END FUNCTION getThermalExpansivity_{mat_b_name}"""
        
        st.code(cte_udf_b, language="fortran")
    
    # ====================== SPECIFIC ENTHALPY UDF SECTION ======================
    st.divider()
    with st.expander("🔥 Specific Enthalpy UDF Configuration", expanded=False):
        st.info("🔹 **Equation-based enthalpy**: Replace .dat lookup tables with analytical expression")
        st.markdown("""
        **Model**: `H(T) = (1/α) × [β₁·T + β₂·max(T-T₀,0) + β₃/(1+exp(-γ·(T-T₀))) + C]`
        
        - `α`: Scaling factor [kg/J] — inverse of overall multiplier
        - `β₁`: Linear sensible heat coefficient [J/(kg·K)]
        - `β₂`: Phase change contribution [J/(kg·K)]
        - `β₃`: Sigmoid amplitude for smooth latent heat [J/kg]
        - `γ`: Transition sharpness [1/K]
        - `T₀`: Reference/melting temperature [K]
        - `C`: Constant offset [J/kg]
        """)
        
        enthalpy_col1, enthalpy_col2 = st.columns(2)
        
        with enthalpy_col1:
            st.markdown("### Material A: Enthalpy Coefficients")
            use_enthalpy_udf_a = st.checkbox("Use equation-based enthalpy (Material A)", 
                                            value=st.session_state.use_enthalpy_udf_a, key=uk("enthA", "use_udf"))
            st.session_state.use_enthalpy_udf_a = use_enthalpy_udf_a
            
            if use_enthalpy_udf_a:
                alpha_a = st.number_input("α: Scaling Factor [kg/J]", 
                                         value=5.13580e-2, format="%.3e", key=uk("enthA", "alpha"))
                beta1_a = st.number_input("β₁: Linear Coeff [J/(kg·K)]", 
                                         value=27.9467, format="%.4f", key=uk("enthA", "beta1"))
                beta2_a = st.number_input("β₂: Phase Coeff [J/(kg·K)]", 
                                         value=3.5064, format="%.4f", key=uk("enthA", "beta2"))
                beta3_a = st.number_input("β₃: Sigmoid Amp [J/kg]", 
                                         value=9995.0, format="%.1f", key=uk("enthA", "beta3"))
                gamma_a = st.number_input("γ: Transition Sharpness [1/K]", 
                                         value=0.086717, format="%.6f", key=uk("enthA", "gamma"))
                T0_a = st.number_input("T₀: Reference Temp [K]", 
                                      value=1152.03, step=0.01, key=uk("enthA", "T0"))
                C_a = st.number_input("C: Constant Offset [J/kg]", 
                                     value=-23485.0, format="%.1f", key=uk("enthA", "C"))
                
                # Preview computed enthalpy at key temperatures
                st.markdown("**Preview Values**:")
                preview_temps = [298.0, 933.5, 1152.03, 1200.0, 1500.0]
                preview_data = []
                for T in preview_temps:
                    H = compute_enthalpy_preview_cached(T, alpha_a, beta1_a, beta2_a, beta3_a, gamma_a, T0_a, C_a)
                    preview_data.append({"T [K]": T, "H [J/kg]": f"{H:.2e}"})
                st.table(pd.DataFrame(preview_data))
        
        with enthalpy_col2:
            st.markdown("### Material B: Enthalpy Coefficients")
            use_enthalpy_udf_b = st.checkbox("Use equation-based enthalpy (Material B)", 
                                            value=st.session_state.use_enthalpy_udf_b, key=uk("enthB", "use_udf"))
            st.session_state.use_enthalpy_udf_b = use_enthalpy_udf_b
            
            if use_enthalpy_udf_b:
                alpha_b = st.number_input("α: Scaling Factor [kg/J]", 
                                         value=5.13580e-2, format="%.3e", key=uk("enthB", "alpha"))
                beta1_b = st.number_input("β₁: Linear Coeff [J/(kg·K)]", 
                                         value=27.9467, format="%.4f", key=uk("enthB", "beta1"))
                beta2_b = st.number_input("β₂: Phase Coeff [J/(kg·K)]", 
                                         value=3.5064, format="%.4f", key=uk("enthB", "beta2"))
                beta3_b = st.number_input("β₃: Sigmoid Amp [J/kg]", 
                                         value=9995.0, format="%.1f", key=uk("enthB", "beta3"))
                gamma_b = st.number_input("γ: Transition Sharpness [1/K]", 
                                         value=0.086717, format="%.6f", key=uk("enthB", "gamma"))
                T0_b = st.number_input("T₀: Reference Temp [K]", 
                                      value=1356.6, step=0.01, key=uk("enthB", "T0"))
                C_b = st.number_input("C: Constant Offset [J/kg]", 
                                     value=-23485.0, format="%.1f", key=uk("enthB", "C"))
                
                # Preview computed enthalpy at key temperatures
                st.markdown("**Preview Values**:")
                preview_temps = [298.0, 1000.0, 1356.6, 1400.0, 1800.0]
                preview_data = []
                for T in preview_temps:
                    H = compute_enthalpy_preview_cached(T, alpha_b, beta1_b, beta2_b, beta3_b, gamma_b, T0_b, C_b)
                    preview_data.append({"T [K]": T, "H [J/kg]": f"{H:.2e}"})
                st.table(pd.DataFrame(preview_data))
        
        # Generate and display the Fortran UDF code
        st.divider()
        st.markdown("### 🔧 Auto-Generated Fortran UDF: Specific Enthalpy")
        
        enthalpy_udf = f"""!===============================================================================
! getSpecificEnthalpy.F90 - Equation-based Specific Enthalpy UDF
! Generated for project: {project_name}
! Equation: H(T) = (1/alpha) * [beta1*T + beta2*max(T-T0,0) + 
!          beta3/(1+exp(-gamma*(T-T0))) + const_offset]
!===============================================================================
FUNCTION getSpecificEnthalpy(model, n, temp) RESULT(enthalpy)
  USE DefUtils
  IMPLICIT NONE

  !-----------------------------------------------------------------------------
  ! Input/Output arguments
  !-----------------------------------------------------------------------------
  TYPE(Model_t) :: model          ! Elmer model structure
  INTEGER :: n                     ! Node index
  REAL(KIND=dp) :: temp            ! Current temperature [K]
  REAL(KIND=dp) :: enthalpy        ! Specific enthalpy result [J/kg]

  !-----------------------------------------------------------------------------
  ! Local variables
  !-----------------------------------------------------------------------------
  INTEGER :: timestep, prevtimestep = -1
  REAL(KIND=dp) :: alpha, beta1, beta2, beta3, gamma, T0, const_offset
  REAL(KIND=dp) :: linear_term, phase_term, sigmoid_term, bracket_sum
  REAL(KIND=dp) :: exp_arg, exp_val, denom, max_val
  TYPE(ValueList_t), POINTER :: material
  LOGICAL :: GotIt

  !-----------------------------------------------------------------------------
  ! Persistent storage (retained between calls for efficiency)
  !-----------------------------------------------------------------------------
  SAVE prevtimestep, alpha, beta1, beta2, beta3, gamma, T0, const_offset

  !-----------------------------------------------------------------------------
  ! Get material parameter handle
  !-----------------------------------------------------------------------------
  material => GetMaterial()
  IF (.NOT. ASSOCIATED(material)) THEN
    CALL Fatal('getSpecificEnthalpy', 'No material associated with current element')
  END IF

  !-----------------------------------------------------------------------------
  ! Check if timestep changed - only re-read parameters when needed
  !-----------------------------------------------------------------------------
  timestep = GetTimestep()
  IF (timestep /= prevtimestep) THEN
    
    ! Read enthalpy coefficients from .sif file WITH ERROR HANDLING
    alpha = GetConstReal(material, 'Enthalpy Scaling Factor alpha', GotIt)
    IF (.NOT. GotIt) THEN
      CALL Fatal('getSpecificEnthalpy', 'Enthalpy Scaling Factor alpha not defined in .sif')
    END IF
    
    beta1 = GetConstReal(material, 'Enthalpy Linear Coeff beta1', GotIt)
    IF (.NOT. GotIt) THEN
      CALL Fatal('getSpecificEnthalpy', 'Enthalpy Linear Coeff beta1 not defined in .sif')
    END IF
    
    beta2 = GetConstReal(material, 'Enthalpy Phase Coeff beta2', GotIt)
    IF (.NOT. GotIt) THEN
      CALL Fatal('getSpecificEnthalpy', 'Enthalpy Phase Coeff beta2 not defined in .sif')
    END IF
    
    beta3 = GetConstReal(material, 'Enthalpy Sigmoid Amp beta3', GotIt)
    IF (.NOT. GotIt) THEN
      CALL Fatal('getSpecificEnthalpy', 'Enthalpy Sigmoid Amp beta3 not defined in .sif')
    END IF
    
    gamma = GetConstReal(material, 'Enthalpy Transition Gamma', GotIt)
    IF (.NOT. GotIt) THEN
      CALL Fatal('getSpecificEnthalpy', 'Enthalpy Transition Gamma not defined in .sif')
    END IF
    
    T0 = GetConstReal(material, 'Enthalpy Reference Temperature T0', GotIt)
    IF (.NOT. GotIt) THEN
      CALL Fatal('getSpecificEnthalpy', 'Enthalpy Reference Temperature T0 not defined in .sif')
    END IF
    
    const_offset = GetConstReal(material, 'Enthalpy Constant Offset C', GotIt)
    IF (.NOT. GotIt) THEN
      CALL Fatal('getSpecificEnthalpy', 'Enthalpy Constant Offset C not defined in .sif')
    END IF
    
    prevtimestep = timestep
    
  END IF

  !-----------------------------------------------------------------------------
  ! Compute enthalpy components
  !-----------------------------------------------------------------------------
  linear_term = beta1 * temp
  
  max_val = temp - T0
  IF (max_val < 0.0_dp) THEN
    phase_term = 0.0_dp
  ELSE
    phase_term = beta2 * max_val
  END IF
  
  exp_arg = -gamma * (temp - T0)
  
  IF (exp_arg > 700.0_dp) THEN
    sigmoid_term = 0.0_dp
  ELSE IF (exp_arg < -700.0_dp) THEN
    sigmoid_term = beta3
  ELSE
    exp_val = EXP(exp_arg)
    denom = 1.0_dp + exp_val
    IF (denom < 1.0e-300_dp) THEN
      sigmoid_term = beta3
    ELSE
      sigmoid_term = beta3 / denom
    END IF
  END IF
  
  bracket_sum = linear_term + phase_term + sigmoid_term + const_offset
  
  IF (ABS(alpha) < 1.0e-300_dp) THEN
    CALL Fatal('getSpecificEnthalpy', 'Enthalpy scaling factor alpha is zero or near-zero')
  END IF
  enthalpy = bracket_sum / alpha
  
  ! Optional sanity check
  IF (enthalpy < -1.0e10_dp .OR. enthalpy > 1.0e10_dp) THEN
    CALL Warn('getSpecificEnthalpy', 'Enthalpy value out of expected range')
  END IF
  
END FUNCTION getSpecificEnthalpy"""
        
        st.code(enthalpy_udf, language="fortran")
        
        # Download button for the UDF
        st.download_button(
            label="⬇️ Download getSpecificEnthalpy.F90",
            data=enthalpy_udf,
            file_name="getSpecificEnthalpy.F90",
            mime="text/plain",
            key=uk("enth", "download_udf")
        )

# ====================== TAB 2: HEAT SOURCE (ALL TYPES + CUSTOM GAUSSIAN) ======================
with tab_heat:
    st.header("🔦 Laser Heat Source Function")
    st.info("✅ **Select from multiple heat source types including Custom Gaussian**")
    
    heat_type = st.selectbox(
        "Heat Source Type",
        ["Travelling Gaussian", "Fixed Gaussian", "Flat-Top (Super-Gaussian)", 
         "Double Ellipsoidal", "Custom Gaussian (User Provided)"],
        key=uk("heat", "type")
    )
    
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        beam_radius = st.number_input("Beam Radius r₀ [m]", value=35.0e-6, format="%.2e", key=uk("heat", "radius"))
        heat_coeff = st.number_input("Heat Coefficient [W/m²]", value=8.68e9, format="%.2e", key=uk("heat", "coeff"))
        speed_x = st.number_input("Scan Speed X [m/s]", value=1.0, step=0.1, key=uk("heat", "speedx"))
        speed_y = st.number_input("Scan Speed Y [m/s]", value=0.0, step=0.1, key=uk("heat", "speedy"))
    with col_h2:
        scan_dist = st.number_input("Scan Distance [m]", value=600.0e-6, format="%.2e", key=uk("heat", "dist"))
        init_x = st.number_input("Initial X Position [m]", value=0.0, format="%.2e", key=uk("heat", "initx"))
        init_y = st.number_input("Initial Y Position [m]", value=0.0, format="%.2e", key=uk("heat", "inity"))
        absorptance = st.number_input("Absorptance Ω [1/m]", value=8.5e7, format="%.2e", key=uk("heat", "absorp"))
    
    if heat_type == "Flat-Top (Super-Gaussian)":
        st.subheader("🔷 Super-Gaussian Parameters")
        col_sg1, col_sg2 = st.columns(2)
        with col_sg1:
            sgo = st.number_input("Super-Gaussian Order n", value=3.0, step=0.1, key=uk("heat", "sgo"))
            m1 = st.number_input("Amplitude Prefactor m₁", value=2.0, step=0.1, key=uk("heat", "m1"))
        with col_sg2:
            m2 = st.number_input("Exponential Prefactor m₂", value=2.0, step=0.1, key=uk("heat", "m2"))
            rsgo = 1.0 / sgo if sgo > 0 else 0.3333
            st.info(f"Reciprocal 1/n = {rsgo:.4f} (auto-computed)")
    
    elif heat_type == "Double Ellipsoidal":
        st.subheader("🔷 Double Ellipsoidal Parameters")
        col_de1, col_de2 = st.columns(2)
        with col_de1:
            a_front = st.number_input("Front Semi-Axis a_f [m]", value=50.0e-6, format="%.2e", key=uk("heat", "a_front"))
            b_axis = st.number_input("Transverse Semi-Axis b [m]", value=35.0e-6, format="%.2e", key=uk("heat", "b_axis"))
        with col_de2:
            a_rear = st.number_input("Rear Semi-Axis a_r [m]", value=75.0e-6, format="%.2e", key=uk("heat", "a_rear"))
            f_factor = st.number_input("Front Fraction f_f", value=0.6, step=0.1, min_value=0.0, max_value=1.0, key=uk("heat", "f_factor"))
    
    # Generate heat source function based on type
    if heat_type == "Travelling Gaussian":
        heat_udf = f"""! Travelling Gaussian Heat Source for {project_name}
FUNCTION TravellingHeatSource(Model, n, t) RESULT(f)
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t) :: Model
  INTEGER :: n
  REAL(KIND=dp) :: t, f
  INTEGER :: timestep, prevtimestep = -1
  REAL(KIND=dp) :: Alpha, Coeff, xspeed, yspeed, Dist, Time, x, y, z, s1, s2, r, xzero, yzero, Omega
  TYPE(Mesh_t), POINTER :: Mesh
  TYPE(ValueList_t), POINTER :: Params
  LOGICAL :: Found, NewTimestep
  SAVE Mesh, Params, prevtimestep, time, Alpha, Coeff, xspeed, yspeed, Dist, Omega
  
  timestep = GetTimestep()
  NewTimestep = (timestep /= prevtimestep)
  IF(NewTimestep) THEN
    Mesh => GetMesh()
    Params => Model % Simulation
    time = GetTime()
    Alpha = GetCReal(Params, 'Heat source width')
    Coeff = GetCReal(Params, 'Heat source coefficient')
    xspeed = GetCReal(Params, 'Heat source speed x')
    yspeed = GetCReal(Params, 'Heat source speed y')
    Dist = GetCReal(Params, 'Heat source distance')
    xzero = GetCReal(Params, 'Heat source initial position x', Found)
    yzero = GetCReal(Params, 'Heat source initial position y', Found)
    Omega = GetCReal(Params, 'Absorptance of Surface Material')
    prevtimestep = timestep
  END IF
  x = Mesh % Nodes % x(n); y = Mesh % Nodes % y(n); z = Mesh % Nodes % z(n)
  s1 = xzero + time * xspeed; s2 = yzero + time * yspeed
  r = SQRT((x - s1)**2 + (y - s2)**2)
  f = Coeff * EXP(-2.0_dp * r**2 / Alpha**2 - Omega * ABS(z))
END FUNCTION TravellingHeatSource"""
        heat_proc_name = "TravellingHeatSource"
    
    elif heat_type == "Fixed Gaussian":
        heat_udf = f"""! Fixed Gaussian Heat Source for {project_name}
FUNCTION FixedHeatSource(Model, n, t) RESULT(f)
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t) :: Model; INTEGER :: n; REAL(KIND=dp) :: t, f
  INTEGER :: timestep, prevtimestep = -1
  REAL(KIND=dp) :: Alpha, Coeff, Dist0, Time, x, y, z, r
  TYPE(Mesh_t), POINTER :: Mesh; TYPE(ValueList_t), POINTER :: Params
  LOGICAL :: Found, NewTimestep
  SAVE Mesh, Params, prevtimestep, time, Alpha, Coeff, Dist0
  
  timestep = GetTimestep(); NewTimestep = (timestep /= prevtimestep)
  IF(NewTimestep) THEN
    Mesh => GetMesh(); Params => Model % Simulation; time = GetTime()
    Alpha = GetCReal(Params, 'Heat source width')
    Coeff = GetCReal(Params, 'Heat source coefficient')
    Dist0 = GetCReal(Params, 'Heat source initial position x', Found)
    prevtimestep = timestep
  END IF
  x = Mesh % Nodes % x(n); y = Mesh % Nodes % y(n); z = Mesh % Nodes % z(n)
  r = x - Dist0
  f = Coeff * EXP(-2.0_dp * r**2 / Alpha**2)
END FUNCTION FixedHeatSource"""
        heat_proc_name = "FixedHeatSource"
    
    elif heat_type == "Flat-Top (Super-Gaussian)":
        heat_udf = f"""! Super-Gaussian Travelling Heat Source (Flat-Top) for {project_name}
FUNCTION FlatTopHeatSource(Model, n, t) RESULT(f)
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t) :: Model; INTEGER :: n; REAL(KIND=dp) :: t, f
  INTEGER :: timestep, prevtimestep = -1
  REAL(KIND=dp) :: Alpha, Coeff, xspeed, yspeed, Dist, Time, x, y, z, s1, s2, r
  REAL(KIND=dp) :: xzero, yzero, sgo, m1, m2, rsgo
  TYPE(Mesh_t), POINTER :: Mesh; TYPE(ValueList_t), POINTER :: Params
  LOGICAL :: Found, NewTimestep
  SAVE Mesh, Params, prevtimestep, time, Alpha, Coeff, xspeed, yspeed, Dist
  SAVE xzero, yzero, sgo, m1, m2, rsgo
  
  timestep = GetTimestep(); NewTimestep = (timestep /= prevtimestep)
  IF(NewTimestep) THEN
    Mesh => GetMesh(); Params => Model % Simulation; time = GetTime()
    Alpha = GetCReal(Params, 'Heat source width')
    Coeff = GetCReal(Params, 'Heat source coefficient')
    xspeed = GetCReal(Params, 'Heat source speed x')
    yspeed = GetCReal(Params, 'Heat source speed y')
    Dist = GetCReal(Params, 'Heat source distance')
    xzero = GetCReal(Params, 'Heat source initial position x', Found)
    yzero = GetCReal(Params, 'Heat source initial position y', Found)
    sgo = GetCReal(Params, 'Super gaussian order n')
    rsgo = GetCReal(Params, 'reciproccal of Super gaussian order 1/n')
    m1 = GetCReal(Params, 'prefactor within amplitude term')
    m2 = GetCReal(Params, 'prefactor within exponential term')
    prevtimestep = timestep
  END IF
  x = Mesh % Nodes % x(n); y = Mesh % Nodes % y(n); z = Mesh % Nodes % z(n)
  s1 = xzero + time * xspeed; s2 = yzero + time * yspeed
  r = SQRT((x - s1)**2 + (y - s2)**2)
  f = m1**rsgo * sgo * Coeff * EXP(-m2 * r**sgo / Alpha**sgo) / gamma(rsgo)
END FUNCTION FlatTopHeatSource"""
        heat_proc_name = "FlatTopHeatSource"
    
    elif heat_type == "Double Ellipsoidal":
        heat_udf = f"""! Double Ellipsoidal Heat Source for {project_name}
FUNCTION DoubleEllipsoidalHeatSource(Model, n, t) RESULT(f)
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t) :: Model; INTEGER :: n; REAL(KIND=dp) :: t, f
  INTEGER :: timestep, prevtimestep = -1
  REAL(KIND=dp) :: Af, Ar, B, Cf, Cr, Ff, Fr, Q, xspeed, yspeed, Dist, Time
  REAL(KIND=dp) :: x, y, z, s1, s2, xf, xr, yb, zb, ff, fr
  TYPE(Mesh_t), POINTER :: Mesh; TYPE(ValueList_t), POINTER :: Params
  LOGICAL :: Found, NewTimestep
  SAVE Mesh, Params, prevtimestep, time, Af, Ar, B, Cf, Cr, Ff, Fr, Q, xspeed, yspeed, Dist
  
  timestep = GetTimestep(); NewTimestep = (timestep /= prevtimestep)
  IF(NewTimestep) THEN
    Mesh => GetMesh(); Params => Model % Simulation; time = GetTime()
    Af = GetCReal(Params, 'Front semi-axis a_f')
    Ar = GetCReal(Params, 'Rear semi-axis a_r')
    B = GetCReal(Params, 'Transverse semi-axis b')
    Cf = 2.0_dp * Af; Cr = 2.0_dp * Ar
    Ff = GetCReal(Params, 'Front fraction f_f')
    Fr = 1.0_dp - Ff
    Q = GetCReal(Params, 'Heat source coefficient')
    xspeed = GetCReal(Params, 'Heat source speed x')
    yspeed = GetCReal(Params, 'Heat source speed y')
    Dist = GetCReal(Params, 'Heat source distance')
    prevtimestep = timestep
  END IF
  x = Mesh % Nodes % x(n); y = Mesh % Nodes % y(n); z = Mesh % Nodes % z(n)
  s1 = time * xspeed; s2 = time * yspeed
  xf = x - s1; xr = s1 - x; yb = y - s2; zb = ABS(z)
  ff = Ff * 6.0_dp * SQRT(3.0_dp) / (PI * Af * B * Cf)
  fr = Fr * 6.0_dp * SQRT(3.0_dp) / (PI * Ar * B * Cr)
  IF (xf >= 0.0_dp) THEN
    f = Q * ff * EXP(-3.0_dp * (xf**2/Af**2 + yb**2/B**2 + zb**2/Cf**2))
  ELSE
    f = Q * fr * EXP(-3.0_dp * (xr**2/Ar**2 + yb**2/B**2 + zb**2/Cr**2))
  END IF
END FUNCTION DoubleEllipsoidalHeatSource"""
        heat_proc_name = "DoubleEllipsoidalHeatSource"
    
    else:  # Custom Gaussian (User Provided)
        st.info("🔹 Using expanded GaussianHeatSource with full error handling and optional absorption term")
        heat_udf = f"""!===============================================================================
! GaussianHeatSource.F90 - Custom Travelling Gaussian for {project_name}
! Reference: Kunwar et al., J. Mater. Sci. Technol. 50 (2020) 115-127
!===============================================================================
FUNCTION GaussianHeatSource(Model, n, t) RESULT(f)
  USE DefUtils
  IMPLICIT NONE

  !-----------------------------------------------------------------------------
  ! Input/Output arguments
  !-----------------------------------------------------------------------------
  TYPE(Model_t) :: Model          ! Elmer model structure
  INTEGER :: n                     ! Node index
  REAL(KIND=dp) :: t               ! Current time [s]
  REAL(KIND=dp) :: f               ! Heat flux result [W/m^2]

  !-----------------------------------------------------------------------------
  ! Local variables
  !-----------------------------------------------------------------------------
  INTEGER :: timestep, prevtimestep = -1
  REAL(KIND=dp) :: Alpha, Coeff, xspeed, yspeed, Dist, Time
  REAL(KIND=dp) :: x, y, z, s1, s2, r, xzero, yzero, Omega
  REAL(KIND=dp) :: dx, dy, exponent
  TYPE(Mesh_t), POINTER :: Mesh
  TYPE(ValueList_t), POINTER :: Params
  LOGICAL :: Found, NewTimestep
  
  !-----------------------------------------------------------------------------
  ! Persistent storage (retained between calls)
  !-----------------------------------------------------------------------------
  SAVE Mesh, Params, prevtimestep, Time
  SAVE Alpha, Coeff, xspeed, yspeed, Dist
  SAVE xzero, yzero, Omega

  !-----------------------------------------------------------------------------
  ! Check if timestep changed - only re-read parameters when needed
  !-----------------------------------------------------------------------------
  timestep = GetTimestep()
  NewTimestep = (timestep /= prevtimestep)

  IF (NewTimestep) THEN
    ! Get mesh and simulation parameter handles
    Mesh => GetMesh()
    Params => Model % Simulation
    
    ! Read current simulation time
    Time = GetTime()
    
    ! Read heat source parameters from .sif file WITH ERROR HANDLING
    Alpha  = GetCReal(Params, 'Heat source width', Found)
    IF (.NOT. Found) CALL Fatal('GaussianHeatSource', 'Heat source width not defined')
    
    Coeff  = GetCReal(Params, 'Heat source coefficient', Found)
    IF (.NOT. Found) CALL Fatal('GaussianHeatSource', 'Heat source coefficient not defined')
    
    xspeed = GetCReal(Params, 'Heat source speed x', Found)
    IF (.NOT. Found) xspeed = 0.0_dp
    
    yspeed = GetCReal(Params, 'Heat source speed y', Found)
    IF (.NOT. Found) yspeed = 0.0_dp
    
    Dist   = GetCReal(Params, 'Heat source distance', Found)
    IF (.NOT. Found) Dist = 0.0_dp
    
    xzero  = GetCReal(Params, 'Heat source initial position x', Found)
    IF (.NOT. Found) xzero = 0.0_dp
    
    yzero  = GetCReal(Params, 'Heat source initial position y', Found)
    IF (.NOT. Found) yzero = 0.0_dp
    
    Omega  = GetCReal(Params, 'Absorptance of Surface Material', Found)
    IF (.NOT. Found) Omega = 0.0_dp
    
    ! Store current timestep to avoid redundant reads
    prevtimestep = timestep
  END IF

  !-----------------------------------------------------------------------------
  ! Get current node coordinates
  !-----------------------------------------------------------------------------
  x = Mesh % Nodes % x(n)
  y = Mesh % Nodes % y(n)
  z = Mesh % Nodes % z(n)

  !-----------------------------------------------------------------------------
  ! Compute travelling heat source center position
  !-----------------------------------------------------------------------------
  s1 = xzero - Time * xspeed
  s2 = yzero + Time * yspeed

  !-----------------------------------------------------------------------------
  ! Compute radial distance from heat source center to current node
  !-----------------------------------------------------------------------------
  dx = x - s1
  dy = y - s2
  r = SQRT(dx*dx + dy*dy)

  !-----------------------------------------------------------------------------
  ! Compute Gaussian heat flux distribution
  !-----------------------------------------------------------------------------
  exponent = -2.0_dp * r*r / (Alpha*Alpha)
  
  ! Uncomment the line below to include depth-dependent absorption:
  ! exponent = exponent - Omega * ABS(z)
  
  f = Coeff * EXP(exponent)
  
END FUNCTION GaussianHeatSource"""
        heat_proc_name = "GaussianHeatSource"
    
    st.code(heat_udf, language="fortran")

# ====================== TAB 3: LOOKUP TABLES (WITH PERSISTENT SESSION STATE) ======================
with tab_tables:
    st.header("📊 Lookup Tables (.dat files)")
    st.markdown("Tab-separated files for viscosity and specific enthalpy vs. temperature.")
    
    table_choice = st.selectbox(
        "Select Table to Edit",
        ["Viscosity – Material A", "Viscosity – Material B", 
         "Specific Enthalpy – Material A", "Specific Enthalpy – Material B"],
        key=uk("table", "select")
    )
    
    # Pre-loaded sample data with persistent keys
    table_configs = {
        "Viscosity – Material A": {
            "default": pd.DataFrame({
                "Temperature_K": [300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 933.5, 1000.0],
                "Viscosity_Pas": [1.2e-3, 1.0e-3, 0.85e-3, 0.7e-3, 0.6e-3, 0.5e-3, 0.4e-3, 0.35e-3, 0.3e-3]
            }),
            "fname_prefix": "mu",
            "col1": "Temperature_K",
            "col2": "Viscosity_Pas",
            "key": "visc_a",
            "mat_selector": lambda: mat_a_name
        },
        "Viscosity – Material B": {
            "default": pd.DataFrame({
                "Temperature_K": [300.0, 500.0, 800.0, 1000.0, 1356.6, 1400.0, 1600.0, 1800.0, 2000.0],
                "Viscosity_Pas": [4.0e-3, 3.2e-3, 2.5e-3, 2.0e-3, 1.5e-3, 1.4e-3, 1.2e-3, 1.0e-3, 0.9e-3]
            }),
            "fname_prefix": "mu",
            "col1": "Temperature_K",
            "col2": "Viscosity_Pas",
            "key": "visc_b",
            "mat_selector": lambda: mat_b_name
        },
        "Specific Enthalpy – Material A": {
            "default": pd.DataFrame({
                "Temperature_K": [300.0, 310.0, 320.0, 400.0, 500.0, 600.0, 700.0, 800.0, 842.71, 850.0, 900.0, 1000.0],
                "Enthalpy_Jkg": [0.0, 8.98e3, 1.80e4, 9.05e4, 1.81e5, 2.72e5, 3.63e5, 4.54e5, 5.16e5, 5.24e5, 6.02e5, 7.50e5]
            }),
            "fname_prefix": "h",
            "col1": "Temperature_K",
            "col2": "Enthalpy_Jkg",
            "key": "enth_a",
            "mat_selector": lambda: mat_a_name
        },
        "Specific Enthalpy – Material B": {
            "default": pd.DataFrame({
                "Temperature_K": [300.0, 500.0, 800.0, 1000.0, 1356.6, 1400.0, 1600.0, 1800.0, 2000.0],
                "Enthalpy_Jkg": [0.0, 1.54e5, 3.08e5, 4.62e5, 6.67e5, 6.88e5, 7.90e5, 8.92e5, 9.94e5]
            }),
            "fname_prefix": "h",
            "col1": "Temperature_K",
            "col2": "Enthalpy_Jkg",
            "key": "enth_b",
            "mat_selector": lambda: mat_b_name
        }
    }
    
    config = table_configs[table_choice]
    session_key = f"table_data_{config['key']}"
    
    # Initialize session state for this table if not exists
    if st.session_state[session_key] is None:
        st.session_state[session_key] = config["default"].copy()
    
    edited_df = st.data_editor(
        st.session_state[session_key],
        num_rows="dynamic",
        key=uk("table", f"editor_{table_choice.replace(' ', '_')}"),
        column_config={
            config["col1"]: st.column_config.NumberColumn("T [K]", min_value=0, format="%.2f"),
            config["col2"]: st.column_config.NumberColumn(
                "Viscosity [Pa·s]" if "Viscosity" in table_choice else "Enthalpy [J/kg]",
                format="%.2e" if "Viscosity" in table_choice else "%.0f"
            )
        },
        hide_index=True
    )
    
    # ALWAYS update session state - this is critical for persistence
    st.session_state[session_key] = edited_df.copy()
    
    # Generate filename with current material name
    mat_name = config["mat_selector"]()
    fname = f"{config['fname_prefix']}_{mat_name.lower().replace('-', '_')}.dat"
    
    csv_content = edited_df.to_csv(sep='\t', index=False, float_format='%.6f')
    st.download_button(
        label=f"⬇️ Download {fname}",
        data=csv_content,
        file_name=fname,
        mime="text/tab-separated-values",
        key=uk("table", f"dl_{table_choice.replace(' ', '_')}")
    )
    
    # Show note about equation-based enthalpy option
    if "Specific Enthalpy" in table_choice:
        mat_is_a = "A" in table_choice
        use_udf = st.session_state.use_enthalpy_udf_a if mat_is_a else st.session_state.use_enthalpy_udf_b
        if use_udf:
            st.markdown('<div class="success-box">✅ Equation-based enthalpy UDF is enabled for this material. The .dat table below is for reference only and will not be used in the generated .sif file.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠️ Using .dat lookup table for enthalpy. Enable equation-based UDF in Materials tab for analytical H(T) model.</div>', unsafe_allow_html=True)

# ====================== TAB 4: PHYSICS & BOUNDARY CONDITIONS ======================
with tab_physics:
    st.header("⚙️ Physics Settings & Boundary Conditions")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.subheader("🕐 Time Stepping")
        coord_scaling = st.selectbox(
            "Coordinate Scaling",
            ["1.0e-6 (µm→m)", "1.0e-5 (10µm→m)", "1.0e-3 (mm→m)"],
            index=0, key=uk("phys", "coordscale")
        )
        dt_initial = st.number_input("Initial Δt [s]", value=1.0e-7, format="%.1e", key=uk("phys", "dtinit"))
        dt_main = st.number_input("Main Δt [s]", value=1.0e-5, format="%.1e", key=uk("phys", "dtmain"))
        n_steps_initial = st.number_input("Steps at initial Δt", value=1, min_value=1, key=uk("phys", "ninit"))
        n_steps_main = st.number_input("Steps at main Δt", value=60, min_value=1, key=uk("phys", "nmain"))
        bdf_order = st.selectbox("BDF Order", [1, 2, 3], index=1, key=uk("phys", "bdf"))
    
    with col_p2:
        st.subheader("📁 Output Settings")
        results_dir = st.text_input("Results Directory", value="./results/", key=uk("phys", "resdir"))
        output_file = st.text_input("Output File Base", value="old.result", key=uk("phys", "outfile"))
        post_file = st.text_input("Post-Processing File", value="a.vtu", key=uk("phys", "postfile"))
        mesh_name = st.text_input("Mesh Database Name", value="Mesh_weld_external", key=uk("phys", "meshname"))
    
    st.divider()
    st.subheader("🔗 Boundary Conditions (Fixed Geometry Entities)")
    
    st.markdown("🔹 **Solids** (Body assignments):")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        body1_solid = st.selectbox("Body 1 → Solid", SOLID_NAMES, index=0, key=uk("phys", "body1"))
    with col_s2:
        body2_solid = st.selectbox("Body 2 → Solid", SOLID_NAMES, index=1, key=uk("phys", "body2"))
    
    st.markdown("🔹 **Boundary Conditions** (select from fixed face list):")
    
    bc_fixed_defaults = ["Face_1leftfront", "Face_3frontfront", "Face_4bottomfront", "Face_7bottomback"]
    bc_fixed = safe_multiselect(
        "Fixed Displacement Faces (Zero Velocity)",
        FACE_NAMES,
        default=bc_fixed_defaults,
        key=uk("phys", "bcfixed")
    )
    
    bc_conv_defaults = ["Face_2leftback", "Face_5topfront"]
    bc_conv = safe_multiselect(
        "Convective Cooling Faces (h=15 W/m²K, T∞=298K)",
        FACE_NAMES,
        default=bc_conv_defaults,
        key=uk("phys", "bcconv")
    )
    htc_value = st.number_input("Heat Transfer Coefficient h [W/m²K]", value=15.0, step=1.0, key=uk("phys", "htc"))
    
    bc_temp_defaults = ["Face_4bottomfront"]
    bc_temp = safe_multiselect(
        "Fixed Temperature Faces (T = 298 K)",
        FACE_NAMES,
        default=bc_temp_defaults,
        key=uk("phys", "bctemp")
    )
    
    heat_face = st.selectbox(
        "Laser Heat Flux Boundary",
        FACE_NAMES,
        index=4,
        key=uk("phys", "heatface")
    )
    
    st.divider()
    st.subheader("🌡️ Phase Change Settings")
    col_pc1, col_pc2 = st.columns(2)
    with col_pc1:
        mushy_width = st.number_input("Mushy Zone Width ±ΔT [K]", value=10.0, step=1.0, key=uk("phys", "mushy"))
        latent_release = st.checkbox("Check Latent Heat Release", value=True, key=uk("phys", "latentcheck"))
    with col_pc2:
        phase_model = st.selectbox("Phase Change Model", ["Spatial 2", "Spatial 1", "None"], index=0, key=uk("phys", "phasemodel"))

# ====================== TAB 5: GENERATE FILES (ENHANCED PERSISTENCE & CACHING) ======================
with tab_generate:
    st.header("📥 Generate Complete Elmer Input Files")
    
    # Show persistent download section if content exists
    if st.session_state.generated_content and st.session_state.generation_timestamp:
        st.success(f"✅ Files generated at {st.session_state.generation_timestamp}!")
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
        st.markdown("### 📥 Download Generated Files (Always Available)")
        st.markdown("*Downloads remain functional even after downloading other files*")
        
        # Regenerate button - separate from downloads
        if st.button("🔄 Regenerate Files with Current Settings", key="regenerate_btn", type="secondary"):
            # Force re-generation by clearing timestamp
            st.session_state.generation_timestamp = None
            st.rerun()
        
        # Download buttons - ALWAYS read from session_state
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        gc = st.session_state.generated_content  # Shortcut
        
        with col_dl1:
            st.download_button(
                label="📄 Download case.sif",
                data=gc['sif_content'],
                file_name=gc['sif_filename'],
                mime="text/plain",
                key="dl_sif_persistent"
            )
            st.download_button(
                label="🔧 getDensity_A.F90",
                data=gc['dens_f90_a'],
                file_name=f"getDensity_{gc['mat_a_name']}.F90",
                mime="text/plain",
                key="dl_dens_a_persistent"
            )
            st.download_button(
                label="🔧 getDensity_B.F90",
                data=gc['dens_f90_b'],
                file_name=f"getDensity_{gc['mat_b_name']}.F90",
                mime="text/plain",
                key="dl_dens_b_persistent"
            )
        
        with col_dl2:
            st.download_button(
                label="🔧 getThermalConductivity_A.F90",
                data=gc['cond_f90_a'],
                file_name=f"getThermalConductivity_{gc['mat_a_name']}.F90",
                mime="text/plain",
                key="dl_cond_a_persistent"
            )
            st.download_button(
                label="🔧 getThermalConductivity_B.F90",
                data=gc['cond_f90_b'],
                file_name=f"getThermalConductivity_{gc['mat_b_name']}.F90",
                mime="text/plain",
                key="dl_cond_b_persistent"
            )
            st.download_button(
                label="🔧 HeatSource.F90",
                data=gc['heat_f90'],
                file_name="DifferentTypeHeatSource.F90",
                mime="text/plain",
                key="dl_heat_persistent"
            )
        
        with col_dl3:
            st.download_button(
                label="🔧 getThermalExpansivity_A.F90",
                data=gc['cte_f90_a'],
                file_name=f"getThermalExpansivity_{gc['mat_a_name']}.F90",
                mime="text/plain",
                key="dl_cte_a_persistent"
            )
            st.download_button(
                label="🔧 getThermalExpansivity_B.F90",
                data=gc['cte_f90_b'],
                file_name=f"getThermalExpansivity_{gc['mat_b_name']}.F90",
                mime="text/plain",
                key="dl_cte_b_persistent"
            )
            # Add enthalpy UDF download if enabled
            if gc.get('use_enthalpy_udf_a', False) or gc.get('use_enthalpy_udf_b', False):
                st.download_button(
                    label="🔧 getSpecificEnthalpy.F90",
                    data=gc['enth_udf'],
                    file_name="getSpecificEnthalpy.F90",
                    mime="text/plain",
                    key="dl_enth_udf_persistent"
                )
            # Lookup tables from persistent session state
            for table_key, label_prefix, fname_prefix in [
                ('visc_a_df', 'Viscosity – Material A', 'mu'),
                ('visc_b_df', 'Viscosity – Material B', 'mu'),
                ('enth_a_df', 'Specific Enthalpy – Material A', 'h'),
                ('enth_b_df', 'Specific Enthalpy – Material B', 'h')
            ]:
                df = gc.get(table_key)
                mat = gc['mat_a_name'] if 'a' in table_key else gc['mat_b_name']
                fname = f"{fname_prefix}_{mat.lower().replace('-', '_')}.dat"
                
                if df is not None and not df.empty:
                    content = df.to_csv(sep='\t', index=False, float_format='%.6f')
                else:
                    # Fallback defaults
                    if 'Viscosity' in label_prefix:
                        content = "Temperature_K\tViscosity_Pas\n300.0\t1.0e-3"
                    else:
                        content = "Temperature_K\tEnthalpy_Jkg\n300.0\t0.0"
                
                st.download_button(
                    label=f"📊 {fname}",
                    data=content,
                    file_name=fname,
                    mime="text/tab-separated-values",
                    key=f"dl_table_{table_key}_persistent"
                )
        
        # ZIP bundling - persistent version
        st.divider()
        st.subheader("📦 One-Click Download: All Files as ZIP")
        
        # Create ZIP from session state (always available)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(gc['sif_filename'], gc['sif_content'])
            zip_file.writestr(f"getDensity_{gc['mat_a_name']}.F90", gc['dens_f90_a'])
            zip_file.writestr(f"getDensity_{gc['mat_b_name']}.F90", gc['dens_f90_b'])
            zip_file.writestr(f"getThermalConductivity_{gc['mat_a_name']}.F90", gc['cond_f90_a'])
            zip_file.writestr(f"getThermalConductivity_{gc['mat_b_name']}.F90", gc['cond_f90_b'])
            zip_file.writestr(f"getThermalExpansivity_{gc['mat_a_name']}.F90", gc['cte_f90_a'])
            zip_file.writestr(f"getThermalExpansivity_{gc['mat_b_name']}.F90", gc['cte_f90_b'])
            zip_file.writestr("DifferentTypeHeatSource.F90", gc['heat_f90'])
            
            # Add enthalpy UDF if enabled
            if gc.get('use_enthalpy_udf_a', False) or gc.get('use_enthalpy_udf_b', False):
                zip_file.writestr("getSpecificEnthalpy.F90", gc['enth_udf'])
            
            for table_key, prefix in [
                ('visc_a_df', 'mu'), ('visc_b_df', 'mu'),
                ('enth_a_df', 'h'), ('enth_b_df', 'h')
            ]:
                df = gc.get(table_key)
                if df is not None and not df.empty:
                    mat = gc['mat_a_name'] if 'a' in table_key else gc['mat_b_name']
                    fname = f"{prefix}_{mat.lower().replace('-', '_')}.dat"
                    content = df.to_csv(sep='\t', index=False, float_format='%.6f')
                    zip_file.writestr(fname, content)
        
        st.download_button(
            label="📦 Download ALL files as ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"{gc.get('project_name', project_name)}_elmer_files.zip",
            mime="application/zip",
            key="download_all_zip_persistent"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate button - only shown if no content exists or user wants to regenerate
    if not st.session_state.generated_content or st.button("🔄 Generate All Files (New)", type="primary", use_container_width=True, key="generate_new_btn"):
        # === PRE-COMPUTE ALL SUBSTITUTION VALUES INTO DICTIONARY ===
        coord_val = coord_scaling.split()[0]
        timestep_intervals = f"{n_steps_initial} {n_steps_main}"
        timestep_sizes = f"{dt_initial:.1e} {dt_main:.1e}"
        output_intervals = "1 1"
        heat_proc = heat_proc_name
        
        # Convert face names to indices (cached)
        bc_fixed_idx = face_names_to_indices_cached(bc_fixed)
        bc_conv_idx = face_names_to_indices_cached(bc_conv)
        bc_temp_idx = face_names_to_indices_cached(bc_temp)
        heat_face_idx = re.search(r'Face_(\d+)', heat_face).group(1) if re.search(r'Face_(\d+)', heat_face) else "5"
        
        # Pre-compute material name variations
        mat_a_name_lower = mat_a_name.lower().replace('-', '_')
        mat_b_name_lower = mat_b_name.lower().replace('-', '_')
        
        # Pre-compute phase change intervals
        mat_a_melting_minus = mat_a_melting - mushy_width
        mat_a_melting_plus = mat_a_melting + mushy_width
        mat_b_melting_minus = mat_b_melting - mushy_width
        mat_b_melting_plus = mat_b_melting + mushy_width
        
        # Build comprehensive substitution dictionary
        substitutions = {
            'project_name': project_name, 'author': author, 'date_str': date_str,
            'mesh_name': mesh_name, 'results_dir': results_dir,
            'post_file': post_file, 'output_file': output_file,
            'table_dir_visc': table_dir_visc, 'table_dir_enth': table_dir_enth,
            'coord_val': coord_val, 'output_intervals': output_intervals,
            'timestep_intervals': timestep_intervals, 'timestep_sizes': timestep_sizes,
            'bdf_order': bdf_order,
            'mat_a_name': mat_a_name, 'mat_b_name': mat_b_name,
            'body1_solid': body1_solid, 'body2_solid': body2_solid,
            'mat_a_melting': mat_a_melting, 'mat_b_melting': mat_b_melting,
            'mat_a_name_lower': mat_a_name_lower, 'mat_b_name_lower': mat_b_name_lower,
            'mat_a_melting_minus': mat_a_melting_minus, 'mat_a_melting_plus': mat_a_melting_plus,
            'mat_b_melting_minus': mat_b_melting_minus, 'mat_b_melting_plus': mat_b_melting_plus,
            'heat_type': heat_type, 'heat_proc': heat_proc,
            'beam_radius': beam_radius, 'heat_coeff': heat_coeff,
            'speed_x': speed_x, 'speed_y': speed_y, 'scan_dist': scan_dist,
            'init_x': init_x, 'init_y': init_y, 'absorptance': absorptance,
            'bc_fixed_len': len(bc_fixed), 'bc_conv_len': len(bc_conv),
            'bc_temp_len': len(bc_temp),
            'bc_fixed_idx': bc_fixed_idx, 'bc_conv_idx': bc_conv_idx,
            'bc_temp_idx': bc_temp_idx, 'heat_face_idx': heat_face_idx,
            'htc_value': htc_value,
            'phase_model': phase_model,
            'latent_check': 'True' if latent_release else 'False',
        }
        
        if heat_type == "Flat-Top (Super-Gaussian)":
            substitutions.update({'sgo': sgo, 'rsgo': rsgo, 'm1': m1, 'm2': m2})
        elif heat_type == "Double Ellipsoidal":
            substitutions.update({'a_front': a_front, 'a_rear': a_rear, 'b_axis': b_axis, 'f_factor': f_factor})
        
        # === BUILD .sif FILE ===
        sif_content = """    !Phase change solid-liquid
    !Elmer solver input file for transient solid-liquid phase change with enthalpy formulation
    !Bilayer: {mat_a_name} ({body1_solid}) / {mat_b_name} ({body2_solid})
    !Project: {project_name} | Author: {author} | Date: {date_str}
    !Mesh supplied externally: {mesh_name}

    Header
      CHECK KEYWORDS Warn
      Mesh DB "." "{mesh_name}"
      Include Path ""
      Results Directory "{results_dir}"
    End

    Simulation
      Max Output Level = 5
      Coordinate System = Cartesian 3D
      Coordinate Mapping(3) = 1 2 3
      Coordinate Scaling = {coord_val}
      Simulation Type = Transient
      Steady State Max Iterations = 5
      Output Intervals (2) = {output_intervals}
      Timestep intervals (2) = {timestep_intervals}
      Timestep Sizes (2) = {timestep_sizes}
      Timestepping Method = BDF
      BDF Order = {bdf_order}
      Solver Input File = mesh1phasechange_lsenthalpy.sif
      Post File = "{post_file}"
      Output File = "{output_file}"
      Binary Output = Logical True
      Use Mesh Names = True
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Coefficients for input into the user defined subroutine {heat_proc}
      ! Parameters for the {heat_type}
      ! Terms related to Eq. 8 of Kunwar et al, Journal of Materials Science & Technology, 2020 (50), pp. 115-127
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      Heat Source Width = Real {beam_radius}
      Heat Source Coefficient = Real {heat_coeff}
      Heat Source Speed x = Real {speed_x}
      Heat Source Speed y = Real {speed_y}
      Heat Source Distance = Real {scan_dist}
      Heat source initial position x = Real {init_x}
      Heat source initial position y = Real {init_y}
      Absorptance of Top Surface Material = Real {absorptance}
      Absorptance of Bottom Surface Material = Real {absorptance}
      """.format(**substitutions)
        
        if heat_type == "Flat-Top (Super-Gaussian)":
            sif_content += """      Super gaussian order n = Real {sgo}
      reciproccal of Super gaussian order 1/n = Real {rsgo}
      prefactor within amplitude term = Real {m1}
      prefactor within exponential term = Real {m2}
""".format(**substitutions)
        elif heat_type == "Double Ellipsoidal":
            sif_content += """      Front semi-axis a_f = Real {a_front}
      Rear semi-axis a_r = Real {a_rear}
      Transverse semi-axis b = Real {b_axis}
      Front fraction f_f = Real {f_factor}
""".format(**substitutions)
        
        sif_content += """      Mesh Levels = 1
    End

    Constants
      Gravity(4) = 0 -1 0 9.82
      Stefan Boltzmann = 5.67e-08
      Permittivity of Vacuum = 8.8542e-12
      Boltzmann Constant = 1.3807e-23
      Unit Charge = 1.602e-19
    End

   Body 1
      Target Bodies(1) = 1
      Name = "{body1_solid}"
      Equation = 1
      Material = 1
      Body Force = 1
      Initial condition = 1
    End

    Body 2
      Target Bodies(1) = 2
      Name = "{body2_solid}"
      Equation = 1
      Material = 2
      Body Force = 1
      Initial condition = 1
    End
    
    ! Direct solver to be used to prevent numerical divergence errors
    Solver 1
      Equation = Heat Equation
      Procedure = "HeatSolve" "HeatSolver"
      Calculate Loads = True
      Variable = Temperature
      Exec Solver = Always
      Stabilize = True
      Bubbles = True
      Lumped Mass Matrix = False
      Optimize Bandwidth = True
      Steady State Convergence Tolerance = 1.0e-6
      Nonlinear System Convergence Tolerance = 1.0e-7
      Nonlinear System Max Iterations = 10
      Nonlinear System Newton After Iterations = 3
      Nonlinear System Newton After Tolerance = 1.0e-3
      Nonlinear System Relaxation Factor = 0.6
      Linear System Solver = Iterative
      Linear System Iterative Method = BiCGStab
      Linear System Max Iterations = 500
      Linear System Convergence Tolerance = 1.0e-10
      Linear System Preconditioning = ILU0
      Linear System ILUT Tolerance = 1.0e-3
      Linear System Abort Not Converged = False
      Linear System Residual Output = 1
      Linear System Precondition Recompute = 1
    End

    Solver 2
      Equation = Navier-Stokes
      Variable = Flow Solution[Velocity:3 Pressure:1]
      Procedure = "FlowSolve" "FlowSolver"
      Calculate Loads = True
      Exec Solver = Always
      Stabilize = True
      Bubbles = False
      Lumped Mass Matrix = False
      Optimize Bandwidth = True
      Steady State Convergence Tolerance = 1.0e-4
      Nonlinear System Convergence Tolerance = 1.0e-7
      Nonlinear System Max Iterations = 5
      Nonlinear System Newton After Iterations = 3
      Nonlinear System Newton After Tolerance = 1.0e-3
      Nonlinear System Relaxation Factor = 0.6
      Linear System Solver = Iterative
      Linear System Iterative Method = BiCGStab
      Linear System Max Iterations = 500
      Linear System Convergence Tolerance = 1.0e-10
      Linear System Preconditioning = ILU0
      Linear System ILUT Tolerance = 1.0e-3
      Linear System Abort Not Converged = False
      Linear System Residual Output = 1
      Linear System Precondition Recompute = 1
    End
    
    Solver 3
      Equation = "LinearDisp"
      Procedure = "StressSolve" "StressSolver"
      Variable = "Displacement"
      Variable DOFs = Integer 3
      Calculate Stresses = TRUE
      Calculate Strains = TRUE
      Calculate Principal = Logical TRUE
      Linear System Solver = Direct
      Linear System Symmetric = Logical True
      Linear System Scaling = Logical False
      Linear System Iterative Method = BiCGStab
      Linear System Direct Method = UMFPACK
      Linear System Convergence Tolerance = 1.0e-8
      Linear System Max Iterations = 200
      Linear System Preconditioning = ILU2
      Nonlinear System Convergence Tolerance = Real 1.0e-7
      Nonlinear System Max Iterations = Integer 1
      Nonlinear System Relaxation Factor = Real 1
      Steady State Convergence Tolerance = 1.0e-6
      Optimize Bandwidth = True
    End
    
    Solver 4
      Exec Solver = never
      Equation = SaveLine
      Procedure = "SaveData" "SaveLine"
      Filename = f.dat
    End

    Equation 1
      Name = "Equation 1"
      Phase Change Model = {phase_model}
      Check Latent Heat Release = {latent_check}
      Convection = Computed
      Navier-Stokes = True
      NS Convect = True
      Active Solvers(3) = 1 2 3
    End

    Material 1
      Name = "{mat_a_name}"
      !!!!!!!!!For elasticity + plasticity computation purpose!!!!!!!!!!!!
      Youngs Modulus = Variable vonMises
      Procedure "plasticityMaterialModel" "getPlasticity"
      Isotropic elastic modulus in elastic regime in Pa = Real 68.9e9
      Yield strength of the alloy materials in Pa = Real 249.0e6
      Strength coefficient in Ramberg-Osgood equation = Real 381.08e6
      Reciprocal of strain hardening coefficient = Real 9.7087
      Poisson Ratio = Real 0.33
      Reference Temperature = 298.0
      Heat Expansion Coefficient = Variable Temperature
      Procedure "getThermalExpansivity" "getThermalExpansivity"
      Reference Thermal Expansivity Solid {mat_a_name} = Real -8.371435292934836e-05
      Thermal Expansivity Coeff As Solid {mat_a_name} = Real -3.7262790630140875e-10
      Thermal Expansivity Coeff Bs Solid {mat_a_name} = Real 4.2255653792479394e-07
      Viscosity = Variable Temperature
    Real
      include {table_dir_visc}mu_{mat_a_name_lower}.dat
    End
""".format(**substitutions)
        
        # === SPECIFIC ENTHALPY SECTION FOR MATERIAL A ===
        if use_enthalpy_udf_a:
            sif_content += f"""      Specific Enthalpy = Variable Temperature
    Procedure "getSpecificEnthalpy" "getSpecificEnthalpy"
    Enthalpy Scaling Factor alpha = Real {alpha_a}
    Enthalpy Linear Coeff beta1 = Real {beta1_a}
    Enthalpy Phase Coeff beta2 = Real {beta2_a}
    Enthalpy Sigmoid Amp beta3 = Real {beta3_a}
    Enthalpy Transition Gamma = Real {gamma_a}
    Enthalpy Reference Temperature T0 = Real {T0_a}
    Enthalpy Constant Offset C = Real {C_a}
  End
"""
        else:
            sif_content += f"""      Specific Enthalpy = Variable Temperature
    Real
      include {table_dir_enth}h_{mat_a_name_lower}.dat
    End
"""
        
        # Continue Material 1 block
        sif_content += """      Phase Change Intervals(2,1) = {mat_a_melting_minus} {mat_a_melting_plus}
      Compressibility Model = Incompressible
      Reference Pressure = 0
      Specific Heat Ratio = 1.4
      Heat Conductivity = Variable Temperature
      Procedure "getFilmThermalConductivity" "getThermalConductivity"
      Reference Thermal Conductivity Solid {mat_a_name} = Real 167.0
      Cond Coeff As Solid {mat_a_name} = Real -1.17E-04
      Cond Coeff Bs Solid {mat_a_name} = Real 9.29E-03
      Reference Thermal Conductivity Liquid {mat_a_name} = Real 90.0
      Cond Coeff Liquid {mat_a_name} = Real 1.83E-02
      Melting Point Temperature of {mat_a_name} = Real {mat_a_melting}
      Density = Variable Temperature
      Procedure "getFilmDensity" "getDensity"
      Reference Density Solid {mat_a_name} = Real 2700.0
      Density Coeff Solid {mat_a_name} = Real -0.1898
      Reference Density Liquid {mat_a_name} = Real 2380.0
      Density Coefficient Liquid {mat_a_name} = Real -0.3153
      Tscaler = Real 1.0
    End

    Material 2
      Name = "{mat_b_name}"
      Youngs Modulus = Variable vonMises
      Procedure "plasticityMaterialModel" "getPlasticity"
      Isotropic elastic modulus in elastic regime in Pa = Real 115.0e9
      Yield strength of the alloy materials in Pa = Real 249.0e6
      Strength coefficient in Ramberg-Osgood equation = Real 381.08e6
      Reciprocal of strain hardening coefficient = Real 9.7087
      Poisson Ratio = Real 0.31
      Reference Temperature = 298.0
      Heat Expansion Coefficient = Variable Temperature
      Procedure "getThermalExpansivity" "getThermalExpansivity"
      Reference Thermal Expansivity Solid {mat_b_name} = Real -8.371435292934836e-05
      Thermal Expansivity Coeff As Solid {mat_b_name} = Real -3.7262790630140875e-10
      Thermal Expansivity Coeff Bs Solid {mat_b_name} = Real 4.2255653792479394e-07
      Viscosity = Variable Temperature
    Real
      include {table_dir_visc}mu_{mat_b_name_lower}.dat
    End
""".format(**substitutions)
        
        # === SPECIFIC ENTHALPY SECTION FOR MATERIAL B ===
        if use_enthalpy_udf_b:
            sif_content += f"""      Specific Enthalpy = Variable Temperature
    Procedure "getSpecificEnthalpy" "getSpecificEnthalpy"
    Enthalpy Scaling Factor alpha = Real {alpha_b}
    Enthalpy Linear Coeff beta1 = Real {beta1_b}
    Enthalpy Phase Coeff beta2 = Real {beta2_b}
    Enthalpy Sigmoid Amp beta3 = Real {beta3_b}
    Enthalpy Transition Gamma = Real {gamma_b}
    Enthalpy Reference Temperature T0 = Real {T0_b}
    Enthalpy Constant Offset C = Real {C_b}
  End
"""
        else:
            sif_content += f"""      Specific Enthalpy = Variable Temperature
    Real
      include {table_dir_enth}h_{mat_b_name_lower}.dat
    End
"""
        
        # Finish Material 2 block
        sif_content += """      Phase Change Intervals(2,1) = {mat_b_melting_minus} {mat_b_melting_plus}
      Compressibility Model = Incompressible
      Reference Pressure = 0
      Specific Heat Ratio = 1.4
      Heat Conductivity = Variable Temperature
      Procedure "getFilmThermalConductivity" "getThermalConductivity"
      Reference Thermal Conductivity Solid {mat_b_name} = Real 391.0
      Cond Coeff As Solid {mat_b_name} = Real -0.052
      Cond Coeff Bs Solid {mat_b_name} = Real 0.0
      Reference Thermal Conductivity Liquid {mat_b_name} = Real 170.0
      Cond Coeff Liquid {mat_b_name} = Real -0.025
      Melting Point Temperature of {mat_b_name} = Real {mat_b_melting}
      Density = Variable Temperature
      Procedure "getFilmDensity" "getDensity"
      Reference Density Solid {mat_b_name} = Real 8940.0
      Density Coeff Solid {mat_b_name} = Real -0.52
      Reference Density Liquid {mat_b_name} = Real 7992.0
      Density Coefficient Liquid {mat_b_name} = Real -0.44
      Tscaler = Real 1.0
    End
    
    Body Force 1
      Name = "Natural convection"
      Boussinesq = True
    End

    Initial Condition 1
      Name = "InitialCondition 1"
      Velocity 2 = 0
      Pressure = 0
      Velocity 1 = 0
      Temperature = 298.0
      Displacement 1 = 0
      Displacement 2 = 0
      Displacement 3 = 0
    End

    Boundary Condition 1
      Name = "Fixed Displacement Faces"
      Target Boundaries({bc_fixed_len}) = {bc_fixed_idx}
      Displacement 1 = 0
      Displacement 2 = 0
      Displacement 3 = 0
      Noslip wall BC = True
      Save Scalars = Logical True
    End
    
    Boundary Condition 2
      Name = "Convective Cooling Faces"
      Target Boundaries({bc_conv_len}) = {bc_conv_idx}
      External Temperature = 298.0
      Heat Transfer Coefficient = {htc_value}
      Noslip wall BC = True
      Save Scalars = Logical True
    End

    Boundary Condition 3
      Name = "Bottom Fixed Temperature"
      Target Boundaries({bc_temp_len}) = {bc_temp_idx}
      External Temperature = 298.0
      Noslip wall BC = True
      Displacement 1 = 0
      Displacement 2 = 0
      Displacement 3 = 0
      Save Scalars = Logical True
      Temperature = 298.0
    End

    Boundary Condition 4
      Name = "Top Laser Heat Flux"
      Target Boundaries(1) = {heat_face_idx}
      Heat Flux = Variable time
      Real Procedure "DifferentTypeHeatSource" "{heat_proc}"
      Save Line = True
    End
""".format(**substitutions)
        
        # === PREPARE FORTRAN UDF FILES (using cached extractor) ===
        dens_f90_a = extract_udf_code_cached(dens_udf_a)
        cond_f90_a = extract_udf_code_cached(cond_udf_a)
        cte_f90_a = extract_udf_code_cached(cte_udf_a)
        dens_f90_b = extract_udf_code_cached(dens_udf_b)
        cond_f90_b = extract_udf_code_cached(cond_udf_b)
        cte_f90_b = extract_udf_code_cached(cte_udf_b)
        heat_f90 = heat_udf  # Already clean
        
        # === PREPARE ENTHALPY UDF ===
        enth_udf = enthalpy_udf  # Already generated in Materials tab
        
        # === STORE ALL GENERATED CONTENT IN SESSION STATE (CRITICAL FOR PERSISTENCE) ===
        st.session_state.generated_content = {
            'sif_content': sif_content,
            'sif_filename': sif_filename,
            'dens_f90_a': dens_f90_a, 'dens_f90_b': dens_f90_b,
            'cond_f90_a': cond_f90_a, 'cond_f90_b': cond_f90_b,
            'cte_f90_a': cte_f90_a, 'cte_f90_b': cte_f90_b,
            'heat_f90': heat_f90,
            'enth_udf': enth_udf,
            'use_enthalpy_udf_a': use_enthalpy_udf_a,
            'use_enthalpy_udf_b': use_enthalpy_udf_b,
            'mat_a_name': mat_a_name, 'mat_b_name': mat_b_name,
            'table_dir_visc': table_dir_visc, 'table_dir_enth': table_dir_enth,
            'project_name': project_name,
            # Store ACTUAL edited DataFrames from session state
            'visc_a_df': st.session_state.table_data_visc_a.copy() if st.session_state.table_data_visc_a is not None else pd.DataFrame(),
            'visc_b_df': st.session_state.table_data_visc_b.copy() if st.session_state.table_data_visc_b is not None else pd.DataFrame(),
            'enth_a_df': st.session_state.table_data_enth_a.copy() if st.session_state.table_data_enth_a is not None else pd.DataFrame(),
            'enth_b_df': st.session_state.table_data_enth_b.copy() if st.session_state.table_data_enth_b is not None else pd.DataFrame(),
            # Store enthalpy coefficients for reference
            'alpha_a': alpha_a if use_enthalpy_udf_a else None,
            'beta1_a': beta1_a if use_enthalpy_udf_a else None,
            'beta2_a': beta2_a if use_enthalpy_udf_a else None,
            'beta3_a': beta3_a if use_enthalpy_udf_a else None,
            'gamma_a': gamma_a if use_enthalpy_udf_a else None,
            'T0_a': T0_a if use_enthalpy_udf_a else None,
            'C_a': C_a if use_enthalpy_udf_a else None,
            'alpha_b': alpha_b if use_enthalpy_udf_b else None,
            'beta1_b': beta1_b if use_enthalpy_udf_b else None,
            'beta2_b': beta2_b if use_enthalpy_udf_b else None,
            'beta3_b': beta3_b if use_enthalpy_udf_b else None,
            'gamma_b': gamma_b if use_enthalpy_udf_b else None,
            'T0_b': T0_b if use_enthalpy_udf_b else None,
            'C_b': C_b if use_enthalpy_udf_b else None,
        }
        
        # Store generation timestamp and params for reference
        st.session_state.generation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.last_generation_params = {
            'heat_type': heat_type, 'mat_a_name': mat_a_name, 'mat_b_name': mat_b_name,
            'beam_radius': beam_radius, 'heat_coeff': heat_coeff,
            'use_enthalpy_udf_a': use_enthalpy_udf_a, 'use_enthalpy_udf_b': use_enthalpy_udf_b,
        }
        
        st.success(f"✅ Files generated at {st.session_state.generation_timestamp}!")
        st.rerun()  # Re-run to show persistent download section
    
    # Instructions expander (always visible)
    with st.expander("📋 How to Compile & Run", expanded=True):
        gc = st.session_state.generated_content
        mat_a_lower = gc.get('mat_a_name', mat_a_name).lower().replace('-', '_') if gc else mat_a_name.lower().replace('-', '_')
        mat_b_lower = gc.get('mat_b_name', mat_b_name).lower().replace('-', '_') if gc else mat_b_name.lower().replace('-', '_')
        
        enth_note_a = "✅ Equation-based UDF" if gc.get('use_enthalpy_udf_a', False) else "📊 Lookup table (.dat)"
        enth_note_b = "✅ Equation-based UDF" if gc.get('use_enthalpy_udf_b', False) else "📊 Lookup table (.dat)"
        
        st.markdown(f"""
        **Directory Structure:**
        ```
        {project_name}/
        ├── {sif_filename}                 # Main Elmer input file
        ├── {mesh_name}.mesh              # Your externally-supplied mesh
        ├── {fortran_dir}
        │   ├── getDensity_{mat_a_name}.F90
        │   ├── getDensity_{mat_b_name}.F90
        │   ├── getThermalConductivity_{mat_a_name}.F90
        │   ├── getThermalConductivity_{mat_b_name}.F90
        │   ├── getThermalExpansivity_{mat_a_name}.F90
        │   ├── getThermalExpansivity_{mat_b_name}.F90
        │   ├── DifferentTypeHeatSource.F90
        │   └── getSpecificEnthalpy.F90    # [If equation-based enthalpy enabled]
        ├── {table_dir_visc}
        │   ├── mu_{mat_a_lower}.dat
        │   └── mu_{mat_b_lower}.dat
        └── {table_dir_enth}
            ├── h_{mat_a_lower}.dat        # [{enth_note_a}]
            └── h_{mat_b_lower}.dat        # [{enth_note_b}]
        ```
        
        **Compilation Steps:**
        ```bash
        # 1. Compile Fortran UDFs
        cd {fortran_dir}
        elmerfem -c getDensity_{mat_a_name}.F90
        elmerfem -c getDensity_{mat_b_name}.F90
        elmerfem -c getThermalConductivity_{mat_a_name}.F90
        elmerfem -c getThermalConductivity_{mat_b_name}.F90
        elmerfem -c getThermalExpansivity_{mat_a_name}.F90
        elmerfem -c getThermalExpansivity_{mat_b_name}.F90
        elmerfem -c DifferentTypeHeatSource.F90
        # If using equation-based enthalpy:
        elmerfem -c getSpecificEnthalpy.F90
        
        # 2. Link and run Elmer
        cd ..
        ElmerSolver {sif_filename}
        ```
        
        **Key Features:**
        - ✅ **Fixed geometry entities**: Only your specified faces/solids
        - ✅ **5 heat source types**: Including Custom Gaussian with full error handling
        - ✅ **Linked UDF system**: Expressions auto-update Fortran code
        - ✅ **Equation-based enthalpy**: Optional analytical H(T) model with 7 coefficients
        - ✅ **Bulletproof multiselects**: Defensive filtering prevents invalid default errors
        - ✅ **Persistent session state**: Downloads NEVER break; all files always accessible
        - ✅ **Intelligent caching**: @st.cache_data prevents redundant re-computation
        - ✅ **ZIP bundling**: One-click download of all project files
        - ✅ **KeyError-proof formatting**: Pre-computed substitution dictionary
        - ✅ **Complete .sif generation**: Ready-to-run Elmer input file
        
        **Reference:** Kunwar et al., *J. Mater. Sci. Technol.* 50 (2020) 115-127
        """)

# ====================== FOOTER ======================
st.markdown("---")
st.markdown("""
**💡 Pro Tips:**
- 🔄 **Session state persistence**: Download any file without losing access to others
- 💾 **Intelligent caching**: Expensive computations cached for 1 hour
- 📦 **ZIP bundling**: Cleanest UX for distributing complete project files
- 🔧 **Regenerate anytime**: Click "Regenerate" to update files with new parameters
- 🎯 **Always visible downloads**: Download section stays accessible in Generate tab
- 🔥 **Enthalpy UDF**: Enable equation-based model for smoother solver convergence

**🔗 Resources:**
- [Elmer FEM Documentation](https://www.elmerfem.org)
- [Fortran UDF Guide](https://github.com/ElmerCSC/elmerfem/blob/devel/fem/src/modules/DefUtils.F90)
- [Heat Source Models Reference](https://www.elmerfem.org/forum/viewtopic.php?t=7330)
""")

# ====================== DEBUG INFO (OPTIONAL) ======================
if st.checkbox("Show Debug Info", key=uk("debug", "show")):
    debug_info = {
        "project": project_name,
        "materials": {"A": mat_a_name, "B": mat_b_name},
        "mesh": mesh_name,
        "heat_source": heat_type,
        "timesteps": {"initial": dt_initial, "main": dt_main, "total": n_steps_initial + n_steps_main},
        "geometry": {"solids": SOLID_NAMES, "faces": FACE_NAMES},
        "boundary_conditions": {
            "fixed": bc_fixed,
            "convective": bc_conv,
            "fixed_temp": bc_temp,
            "heat_flux": heat_face
        },
        "enthalpy_udf": {
            "material_a_enabled": use_enthalpy_udf_a,
            "material_b_enabled": use_enthalpy_udf_b,
            "coefficients_a": {"alpha": alpha_a, "beta1": beta1_a, "beta2": beta2_a, "beta3": beta3_a, "gamma": gamma_a, "T0": T0_a, "C": C_a} if use_enthalpy_udf_a else None,
            "coefficients_b": {"alpha": alpha_b, "beta1": beta1_b, "beta2": beta2_b, "beta3": beta3_b, "gamma": gamma_b, "T0": T0_b, "C": C_b} if use_enthalpy_udf_b else None,
        },
        "session_state": {
            "has_generated_content": bool(st.session_state.generated_content),
            "generation_timestamp": st.session_state.generation_timestamp,
            "table_data_keys": [k for k in st.session_state.keys() if "table_data" in k]
        }
    }
    st.json(debug_info)
