#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Elmer FEM .sif Generator for Dissimilar Material Welding
- ALL HEAT SOURCE TYPES: Travelling Gaussian, Fixed Gaussian, Flat-Top,
  Double Ellipsoidal, **Custom Gaussian (user provided)**
- FIXED GEOMETRY: Only specified faces/solids from Mesh_1_dimensions
- LINKED UDF SYSTEM: Expressions auto-update Fortran UDFs
- BULLETPROOF MULTISELECTS: Defensive filtering prevents invalid default errors
- **ZIP DOWNLOAD**: All files in one click – no session state crash**
"""

import streamlit as st
import pandas as pd
from string import Template
from datetime import datetime
import re
import zipfile
import io

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
</style>
""", unsafe_allow_html=True)

st.title("🔥 Elmer FEM Generator for Dissimilar Welding")
st.markdown("""
**Generate complete `.sif` input files + Fortran UDFs + lookup tables for Al-Cu welding simulations.**
_Mesh file supplied externally with fixed geometry entities._
""")

# ====================== HELPER: UNIQUE KEY GENERATOR ======================
def uk(section: str, var: str, suffix: str = "") -> str:
    """Generate unique key for Streamlit widgets: section_var_suffix"""
    return f"{section}_{var}_{suffix}".strip("_")

# ====================== HELPER: DEFENSIVE MULTISELECT ======================
def safe_multiselect(label, options, default=None, key=None, **kwargs):
    """Bulletproof multiselect that filters invalid defaults."""
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

# ====================== TABS NAVIGATION ======================
tab_materials, tab_heat, tab_tables, tab_physics, tab_generate = st.tabs([
    "🧪 Materials & UDFs",
    "🔦 Heat Source",
    "📊 Lookup Tables",
    "⚙️ Physics & BCs",
    "📥 Generate Files"
])

# ====================== TAB 1: MATERIALS & FORTRAN UDFs ======================
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

        def parse_expression(expr, temp_var="temp"):
            expr_fortran = expr.replace("T", temp_var)
            expr_fortran = re.sub(r'(\d+\.\d+)', r'\1_dp', expr_fortran)
            expr_fortran = re.sub(r'(?<!\.)(\b\d+\b)(?!\.)', r'\1.0_dp', expr_fortran)
            return expr_fortran

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
      denst = {parse_expression(dens_a_l_expr)}
  ELSE
      denst = {parse_expression(dens_a_s_expr)}
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
      thcondt = {parse_expression(cond_a_l_expr)}
  ELSE
      thcondt = {parse_expression(cond_a_s_expr)}
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
      expansivity = {parse_expression(cte_a_l_expr)}
  ELSE
      expansivity = {parse_expression(cte_a_s_expr)}
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
      denst = {parse_expression(dens_b_l_expr)}
  ELSE
      denst = {parse_expression(dens_b_s_expr)}
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
      thcondt = {parse_expression(cond_b_l_expr)}
  ELSE
      thcondt = {parse_expression(cond_b_s_expr)}
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
      expansivity = {parse_expression(cte_b_l_expr)}
  ELSE
      expansivity = {parse_expression(cte_b_s_expr)}
  END IF
END FUNCTION getThermalExpansivity_{mat_b_name}"""
        st.code(cte_udf_b, language="fortran")

# ====================== TAB 2: HEAT SOURCE (WITH CUSTOM GAUSSIAN) ======================
with tab_heat:
    st.header("🔦 Laser Heat Source Function")
    st.info("✅ **Select from multiple heat source types**")

    # NEW: Added "Custom Gaussian (user provided)" option
    heat_type = st.selectbox(
        "Heat Source Type",
        ["Travelling Gaussian", "Fixed Gaussian", "Flat-Top (Super-Gaussian)",
         "Double Ellipsoidal", "Custom Gaussian (user provided)"],
        key=uk("heat", "type")
    )

    # Common parameters
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

    # Type-specific parameters
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

    else:  # NEW: Custom Gaussian (user provided)
        heat_udf = f"""! Custom Gaussian Heat Source (user provided) for {project_name}
FUNCTION GaussianHeatSource(Model, n, t) RESULT(f)
  USE DefUtils
  IMPLICIT NONE

  TYPE(Model_t) :: Model
  INTEGER :: n
  REAL(KIND=dp) :: t, f

  INTEGER :: timestep, prevtimestep = -1
  REAL(KIND=dp) :: Alpha, Coeff, xspeed, yspeed,  Dist, &
      Time, x, y, z, s1, s2,  sper, r, xzero, yzero, Omega
  TYPE(Mesh_t), POINTER :: Mesh
  TYPE(ValueList_t), POINTER :: Params
  LOGICAL :: Found, NewTimestep
  
  SAVE Mesh, Params, prevtimestep, time, Alpha, Coeff, xspeed, Dist, &
        yspeed, xzero, yzero
  
  timestep = GetTimestep()
  NewTimestep = ( timestep /= prevtimestep )

  IF( NewTimestep ) THEN
    Mesh => GetMesh()
    Params => Model % Simulation
    time = GetTime()
    Alpha = GetCReal(Params,'Heat source width')
    Coeff = GetCReal(Params,'Heat source coefficient')
    xspeed = GetCReal(Params,'Heat source speed x')
    yspeed = GetCReal(Params,'Heat source speed y')
    Dist = GetCReal(Params,'Heat source distance')
    xzero = GetCReal(Params,'Heat source initial position x', Found)
    yzero = GetCReal(Params,'Heat source initial position y', Found)
    Omega = GetCReal(Params,'Absorptance of Surface Material')
    prevtimestep = timestep
  END IF

  x = Mesh % Nodes % x(n)   
  y = Mesh % Nodes % y(n)   
  z = Mesh % Nodes % z(n)   

  s1 = xzero - time * xspeed
  s2 = yzero + time * yspeed 
  r = SQRT((x-s1)**2 + (y-s2)**2)   
  
  f = Coeff * EXP( -2*r**2 / Alpha**2 )
  ! Uncomment the line below to include absorption in z-direction:
  ! f = Coeff * EXP( -2*r**2 / Alpha**2 - Omega * ABS(z))
    
END FUNCTION GaussianHeatSource"""
        heat_proc_name = "GaussianHeatSource"

    st.code(heat_udf, language="fortran")

# ====================== TAB 3: LOOKUP TABLES (FIXED: store in session state) ======================
with tab_tables:
    st.header("📊 Lookup Tables (.dat files)")
    st.markdown("Tab-separated files for viscosity and specific enthalpy vs. temperature.")

    # We'll define all four tables explicitly, storing each in session_state.
    # This guarantees they are available during generation.

    # Table 1: Viscosity A
    with st.expander("Viscosity – Material A", expanded=True):
        default_visc_a = pd.DataFrame({
            "Temperature_K": [300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 933.5, 1000.0],
            "Viscosity_Pas": [1.2e-3, 1.0e-3, 0.85e-3, 0.7e-3, 0.6e-3, 0.5e-3, 0.4e-3, 0.35e-3, 0.3e-3]
        })
        visc_a_df = st.data_editor(
            default_visc_a,
            num_rows="dynamic",
            key=uk("table", "visc_a"),
            column_config={
                "Temperature_K": st.column_config.NumberColumn("T [K]", min_value=0, format="%.2f"),
                "Viscosity_Pas": st.column_config.NumberColumn("Viscosity [Pa·s]", format="%.2e")
            },
            hide_index=True
        )
        st.session_state["visc_a_df"] = visc_a_df

    # Table 2: Viscosity B
    with st.expander("Viscosity – Material B", expanded=True):
        default_visc_b = pd.DataFrame({
            "Temperature_K": [300.0, 500.0, 800.0, 1000.0, 1356.6, 1400.0, 1600.0, 1800.0, 2000.0],
            "Viscosity_Pas": [4.0e-3, 3.2e-3, 2.5e-3, 2.0e-3, 1.5e-3, 1.4e-3, 1.2e-3, 1.0e-3, 0.9e-3]
        })
        visc_b_df = st.data_editor(
            default_visc_b,
            num_rows="dynamic",
            key=uk("table", "visc_b"),
            column_config={
                "Temperature_K": st.column_config.NumberColumn("T [K]", min_value=0, format="%.2f"),
                "Viscosity_Pas": st.column_config.NumberColumn("Viscosity [Pa·s]", format="%.2e")
            },
            hide_index=True
        )
        st.session_state["visc_b_df"] = visc_b_df

    # Table 3: Enthalpy A
    with st.expander("Specific Enthalpy – Material A", expanded=True):
        default_enth_a = pd.DataFrame({
            "Temperature_K": [300.0, 310.0, 320.0, 400.0, 500.0, 600.0, 700.0, 800.0, 842.71, 850.0, 900.0, 1000.0],
            "Enthalpy_Jkg": [0.0, 8.98e3, 1.80e4, 9.05e4, 1.81e5, 2.72e5, 3.63e5, 4.54e5, 5.16e5, 5.24e5, 6.02e5, 7.50e5]
        })
        enth_a_df = st.data_editor(
            default_enth_a,
            num_rows="dynamic",
            key=uk("table", "enth_a"),
            column_config={
                "Temperature_K": st.column_config.NumberColumn("T [K]", min_value=0, format="%.2f"),
                "Enthalpy_Jkg": st.column_config.NumberColumn("Enthalpy [J/kg]", format="%.0f")
            },
            hide_index=True
        )
        st.session_state["enth_a_df"] = enth_a_df

    # Table 4: Enthalpy B
    with st.expander("Specific Enthalpy – Material B", expanded=True):
        default_enth_b = pd.DataFrame({
            "Temperature_K": [300.0, 500.0, 800.0, 1000.0, 1356.6, 1400.0, 1600.0, 1800.0, 2000.0],
            "Enthalpy_Jkg": [0.0, 1.54e5, 3.08e5, 4.62e5, 6.67e5, 6.88e5, 7.90e5, 8.92e5, 9.94e5]
        })
        enth_b_df = st.data_editor(
            default_enth_b,
            num_rows="dynamic",
            key=uk("table", "enth_b"),
            column_config={
                "Temperature_K": st.column_config.NumberColumn("T [K]", min_value=0, format="%.2f"),
                "Enthalpy_Jkg": st.column_config.NumberColumn("Enthalpy [J/kg]", format="%.0f")
            },
            hide_index=True
        )
        st.session_state["enth_b_df"] = enth_b_df

    st.info("💾 All tables are automatically saved in session state and will be included in the ZIP download.")

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
        index=4,  # Face_5topfront
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

# ====================== TAB 5: GENERATE FILES (ZIP DOWNLOAD) ======================
with tab_generate:
    st.header("📥 Generate Complete Elmer Input Files")

    if st.button("🔄 Generate All Files", type="primary", use_container_width=True, key=uk("gen", "btn")):
        # Retrieve all data from session state or current widgets
        # (the button reruns the script, so widgets are still available)

        # Retrieve table DataFrames from session state (fallback to defaults if missing)
        visc_a_df = st.session_state.get("visc_a_df", pd.DataFrame({"Temperature_K": [300], "Viscosity_Pas": [1.2e-3]}))
        visc_b_df = st.session_state.get("visc_b_df", pd.DataFrame({"Temperature_K": [300], "Viscosity_Pas": [4.0e-3]}))
        enth_a_df = st.session_state.get("enth_a_df", pd.DataFrame({"Temperature_K": [300], "Enthalpy_Jkg": [0]}))
        enth_b_df = st.session_state.get("enth_b_df", pd.DataFrame({"Temperature_K": [300], "Enthalpy_Jkg": [0]}))

        # Prepare substitution dictionary for .sif
        coord_val = coord_scaling.split()[0]
        timestep_intervals = f"{n_steps_initial} {n_steps_main}"
        timestep_sizes = f"{dt_initial:.1e} {dt_main:.1e}"
        output_intervals = "1 1"

        # Face name to index conversion
        def face_names_to_indices(face_list):
            indices = []
            for face in face_list:
                match = re.search(r'Face_(\d+)', face)
                if match:
                    indices.append(match.group(1))
            return " ".join(indices) if indices else "1"

        bc_fixed_idx = face_names_to_indices(bc_fixed)
        bc_conv_idx = face_names_to_indices(bc_conv)
        bc_temp_idx = face_names_to_indices(bc_temp)
        heat_face_match = re.search(r'Face_(\d+)', heat_face)
        heat_face_idx = heat_face_match.group(1) if heat_face_match else "5"

        # Build .sif content using Template
        sif_template = Template(f"""    !Phase change solid-liquid
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
      ! Coefficients for input into the user defined subroutine {heat_proc_name}
      ! Parameters for the {heat_type}
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
""")
        # Add type-specific parameters
        if heat_type == "Flat-Top (Super-Gaussian)":
            sif_template.template += f"""      Super gaussian order n = Real {sgo}
      reciproccal of Super gaussian order 1/n = Real {rsgo}
      prefactor within amplitude term = Real {m1}
      prefactor within exponential term = Real {m2}
"""
        elif heat_type == "Double Ellipsoidal":
            sif_template.template += f"""      Front semi-axis a_f = Real {a_front}
      Rear semi-axis a_r = Real {a_rear}
      Transverse semi-axis b = Real {b_axis}
      Front fraction f_f = Real {f_factor}
"""
        sif_template.template += f"""      Mesh Levels = 1
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
      Check Latent Heat Release = {'True' if latent_release else 'False'}
      Convection = Computed
      Navier-Stokes = True
      NS Convect = True
      Active Solvers(3) = 1 2 3
    End

    Material 1
      Name = "{mat_a_name}"
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
      include {table_dir_visc}mu_{mat_a_name.lower().replace('-', '_')}.dat
    End
      Specific Enthalpy = Variable Temperature
    Real
      include {table_dir_enth}h_{mat_a_name.lower().replace('-', '_')}.dat
    End
      Phase Change Intervals(2,1) = {mat_a_melting - mushy_width} {mat_a_melting + mushy_width}
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
      include {table_dir_visc}mu_{mat_b_name.lower().replace('-', '_')}.dat
    End
      Specific Enthalpy = Variable Temperature
    Real
      include {table_dir_enth}h_{mat_b_name.lower().replace('-', '_')}.dat
    End
      Phase Change Intervals(2,1) = {mat_b_melting - mushy_width} {mat_b_melting + mushy_width}
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
      Target Boundaries({len(bc_fixed)}) = {bc_fixed_idx}
      Displacement 1 = 0
      Displacement 2 = 0
      Displacement 3 = 0
      Noslip wall BC = True
      Save Scalars = Logical True
    End
    
    Boundary Condition 2
      Name = "Convective Cooling Faces"
      Target Boundaries({len(bc_conv)}) = {bc_conv_idx}
      External Temperature = 298.0
      Heat Transfer Coefficient = {htc_value}
      Noslip wall BC = True
      Save Scalars = Logical True
    End

    Boundary Condition 3
      Name = "Bottom Fixed Temperature"
      Target Boundaries({len(bc_temp)}) = {bc_temp_idx}
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
      Real Procedure "DifferentTypeHeatSource" "{heat_proc_name}"
      Save Line = True
    End
""")
        sif_content = sif_template.substitute()

        # Build ZIP archive
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # .sif file
            zip_file.writestr(sif_filename, sif_content)

            # Fortran UDFs
            zip_file.writestr(f"getDensity_{mat_a_name}.F90", dens_udf_a)
            zip_file.writestr(f"getDensity_{mat_b_name}.F90", dens_udf_b)
            zip_file.writestr(f"getThermalConductivity_{mat_a_name}.F90", cond_udf_a)
            zip_file.writestr(f"getThermalConductivity_{mat_b_name}.F90", cond_udf_b)
            zip_file.writestr(f"getThermalExpansivity_{mat_a_name}.F90", cte_udf_a)
            zip_file.writestr(f"getThermalExpansivity_{mat_b_name}.F90", cte_udf_b)
            zip_file.writestr("DifferentTypeHeatSource.F90", heat_udf)

            # Lookup tables
            zip_file.writestr(
                f"mu_{mat_a_name.lower().replace('-', '_')}.dat",
                visc_a_df.to_csv(sep='\t', index=False, float_format='%.6f')
            )
            zip_file.writestr(
                f"mu_{mat_b_name.lower().replace('-', '_')}.dat",
                visc_b_df.to_csv(sep='\t', index=False, float_format='%.6f')
            )
            zip_file.writestr(
                f"h_{mat_a_name.lower().replace('-', '_')}.dat",
                enth_a_df.to_csv(sep='\t', index=False, float_format='%.6f')
            )
            zip_file.writestr(
                f"h_{mat_b_name.lower().replace('-', '_')}.dat",
                enth_b_df.to_csv(sep='\t', index=False, float_format='%.6f')
            )

        # Provide the ZIP download button
        st.success(f"✅ All files generated for project `{project_name}`! Click below to download the complete package.")
        st.download_button(
            label="📦 Download ALL files as ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"{project_name}_elmer_files.zip",
            mime="application/zip",
            key="download_all_zip"
        )

        # Show contents of the ZIP
        with st.expander("📋 Contents of ZIP file", expanded=False):
            st.markdown(f"""
            **`{project_name}_elmer_files.zip`** contains:
            - `{sif_filename}`
            - `getDensity_{mat_a_name}.F90`, `getDensity_{mat_b_name}.F90`
            - `getThermalConductivity_{mat_a_name}.F90`, `getThermalConductivity_{mat_b_name}.F90`
            - `getThermalExpansivity_{mat_a_name}.F90`, `getThermalExpansivity_{mat_b_name}.F90`
            - `DifferentTypeHeatSource.F90`
            - `mu_{mat_a_name.lower().replace('-', '_')}.dat`
            - `mu_{mat_b_name.lower().replace('-', '_')}.dat`
            - `h_{mat_a_name.lower().replace('-', '_')}.dat`
            - `h_{mat_b_name.lower().replace('-', '_')}.dat`
            """)

        # Compilation instructions
        with st.expander("📋 How to Compile & Run", expanded=True):
            st.markdown(f"""
            **Directory Structure (extract ZIP):**
