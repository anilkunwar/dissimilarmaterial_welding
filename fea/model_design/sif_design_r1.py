#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Elmer FEM .sif Generator for Dissimilar Material Welding
- No geometry tab (mesh supplied externally)
- All widgets have unique keys to prevent StreamlitDuplicateElementId errors
- Generates complete .sif + Fortran UDFs + lookup tables
"""

import streamlit as st
import pandas as pd
from string import Template
from datetime import datetime
import re

# Page config
st.set_page_config(
    page_title="Elmer Weld Generator",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better readability
st.markdown("""
<style>
    .stCodeBlock { background-color: #f8f9fa; }
    .stTextArea textarea { font-family: monospace; font-size: 0.85em; }
    .metric-card { background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("🔥 Elmer FEM Generator for Dissimilar Welding")
st.markdown("""
**Generate complete `.sif` input files + Fortran UDFs + lookup tables for Al-Cu (or custom) welding simulations.**
_Mesh file supplied externally – focus on materials, physics, and boundary conditions._
""")

# ====================== HELPER: UNIQUE KEY GENERATOR ======================
def uk(section: str, var: str, suffix: str = "") -> str:
    """Generate unique key for Streamlit widgets: section_var_suffix"""
    return f"{section}_{var}_{suffix}".strip("_")

# ====================== SIDEBAR: GLOBAL SETTINGS ======================
st.sidebar.header("⚙️ Global Settings")

# Project metadata
project_name = st.sidebar.text_input("Project Name", value="Al_Cu_Weld", key="uk_global_project")
author = st.sidebar.text_input("Author", value="Your Name", key="uk_global_author")
date_str = datetime.now().strftime("%Y-%m-%d")

# File output settings
st.sidebar.subheader("📁 Output Files")
sif_filename = st.sidebar.text_input(".sif Filename", value=f"{project_name.lower()}.sif", key="uk_out_sif")
fortran_dir = st.sidebar.text_input("Fortran UDF Directory", value="./udfs/", key="uk_out_f90dir")
table_dir_visc = st.sidebar.text_input("Viscosity Table Dir", value="./viscosity/", key="uk_out_viscdir")
table_dir_enth = st.sidebar.text_input("Enthalpy Table Dir", value="./specific_enthalpy/", key="uk_out_enthdir")

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
    st.header("🧪 Material Properties & Fortran UDFs")
    
    mat_col1, mat_col2 = st.columns(2)
    
    # ----- MATERIAL A -----
    with mat_col1:
        st.subheader("Material A (Front/Body 1)")
        mat_a_name = st.text_input("Material Name", value="AA6061_Al", key=uk("matA", "name"))
        mat_a_melting = st.number_input("Melting Point [K]", value=933.5, step=0.1, key=uk("matA", "tmelt"))
        
        st.markdown("**Temperature-Dependent Property Expressions** (used in .sif Material block):")
        
        # Density expressions
        st.markdown("🔹 Density ρ(T) [kg/m³]")
        dens_a_s = st.text_input("Solid phase: ρ = ", value="2700.0 - 0.11*(T - 298.0)", key=uk("matA", "dens_s"))
        dens_a_l = st.text_input("Liquid phase: ρ = ", value="2380.0 - 0.28*(T - 933.5)", key=uk("matA", "dens_l"))
        
        # Thermal conductivity
        st.markdown("🔹 Thermal Conductivity k(T) [W/(m·K)]")
        cond_a_s = st.text_input("Solid: k = ", value="167.0 + 0.12*(T - 298.0)", key=uk("matA", "cond_s"))
        cond_a_l = st.text_input("Liquid: k = ", value="90.0 - 0.012*(T - 933.5)", key=uk("matA", "cond_l"))
        
        # Specific heat
        st.markdown("🔹 Specific Heat cₚ(T) [J/(kg·K)]")
        cp_a_s = st.text_input("Solid: cₚ = ", value="904.0 + 0.32*(T - 298.0)", key=uk("matA", "cp_s"))
        cp_a_l = st.text_input("Liquid: cₚ = ", value="1180.0", key=uk("matA", "cp_l"))
        
        # Young's modulus
        st.markdown("🔹 Young's Modulus E(T) [Pa]")
        young_a_s = st.text_input("Solid: E = ", value="68.9e9 - 4.5e7*(T - 298.0)", key=uk("matA", "young_s"))
        young_a_l = st.text_input("Liquid: E = ", value="0.0", key=uk("matA", "young_l"))
        
        # Poisson, CTE
        poisson_a = st.number_input("Poisson's Ratio ν", value=0.33, step=0.01, key=uk("matA", "poisson"))
        cte_a_s = st.text_input("CTE α (solid) [1/K]", value="23.0e-6 + 2.1e-8*(T - 298.0)", key=uk("matA", "cte_s"))
        cte_a_l = st.text_input("CTE α (liquid) [1/K]", value="0.0", key=uk("matA", "cte_l"))
        
        # Latent heat
        latent_a = st.number_input("Latent Heat of Fusion L_f [J/kg]", value=3.97e5, step=1e3, format="%.0f", key=uk("matA", "latent"))
        
        st.divider()
        st.markdown("### 🔧 Fortran UDF Editor – Material A")
        st.info("Edit the Fortran functions that Elmer will call. Variables: `temp` (K), `ref*` from .sif, return SI units.")
        
        # Density UDF
        dens_udf_a = st.text_area(
            "getDensity.F90 – Material A",
            value=f"""FUNCTION getDensity_{mat_a_name}(model, n, temp) RESULT(denst)
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

  IF (refTemp <= temp) THEN
      CALL Warn('getDensity', 'Material A in liquid state.')
      denst = refLiqDenst + alphal * (tscaler * (temp - 910.0_dp))
  ELSE
      denst = refSolDenst + alphas * (tscaler * (temp - 298.0_dp))
  END IF
END FUNCTION getDensity_{mat_a_name}""",
            height=350,
            key=uk("matA", "udf_dens")
        )
        
        # Conductivity UDF
        cond_udf_a = st.text_area(
            "getThermalConductivity.F90 – Material A",
            value=f"""FUNCTION getThermalConductivity_{mat_a_name}(model, n, temp) RESULT(thcondt)
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

  IF (refTemp <= temp) THEN
      CALL Warn('getThermalConductivity', 'Material A in liquid state.')
      thcondt = refLiqThCond + alphal * (tscaler * (temp - 900.0_dp))
  ELSE
      thcondt = refSolThCond + betas * (tscaler * (temp - 298.0_dp)) + &
                alphas * (tscaler * (temp - 298.0_dp))**2
  END IF
END FUNCTION getThermalConductivity_{mat_a_name}""",
            height=350,
            key=uk("matA", "udf_cond")
        )
    
    # ----- MATERIAL B -----
    with mat_col2:
        st.subheader("Material B (Back/Body 2)")
        mat_b_name = st.text_input("Material Name", value="T2_Cu", key=uk("matB", "name"))
        mat_b_melting = st.number_input("Melting Point [K]", value=1356.6, step=0.1, key=uk("matB", "tmelt"))
        
        st.markdown("**Temperature-Dependent Property Expressions** (used in .sif Material block):")
        
        # Density
        st.markdown("🔹 Density ρ(T) [kg/m³]")
        dens_b_s = st.text_input("Solid phase: ρ = ", value="8940.0 - 0.52*(T - 298.0)", key=uk("matB", "dens_s"))
        dens_b_l = st.text_input("Liquid phase: ρ = ", value="7992.0 - 0.44*(T - 1356.6)", key=uk("matB", "dens_l"))
        
        # Conductivity
        st.markdown("🔹 Thermal Conductivity k(T) [W/(m·K)]")
        cond_b_s = st.text_input("Solid: k = ", value="391.0 - 0.052*(T - 298.0)", key=uk("matB", "cond_s"))
        cond_b_l = st.text_input("Liquid: k = ", value="170.0 - 0.025*(T - 1356.6)", key=uk("matB", "cond_l"))
        
        # Specific heat
        st.markdown("🔹 Specific Heat cₚ(T) [J/(kg·K)]")
        cp_b_s = st.text_input("Solid: cₚ = ", value="385.0 + 0.10*(T - 298.0)", key=uk("matB", "cp_s"))
        cp_b_l = st.text_input("Liquid: cₚ = ", value="502.0", key=uk("matB", "cp_l"))
        
        # Young's modulus
        st.markdown("🔹 Young's Modulus E(T) [Pa]")
        young_b_s = st.text_input("Solid: E = ", value="115.0e9 - 4.0e7*(T - 298.0)", key=uk("matB", "young_s"))
        young_b_l = st.text_input("Liquid: E = ", value="0.0", key=uk("matB", "young_l"))
        
        # Poisson, CTE
        poisson_b = st.number_input("Poisson's Ratio ν", value=0.31, step=0.01, key=uk("matB", "poisson"))
        cte_b_s = st.text_input("CTE α (solid) [1/K]", value="16.4e-6 + 2.5e-8*(T - 298.0)", key=uk("matB", "cte_s"))
        cte_b_l = st.text_input("CTE α (liquid) [1/K]", value="0.0", key=uk("matB", "cte_l"))
        
        # Latent heat
        latent_b = st.number_input("Latent Heat of Fusion L_f [J/kg]", value=2.05e5, step=1e3, format="%.0f", key=uk("matB", "latent"))
        
        st.divider()
        st.markdown("### 🔧 Fortran UDF Editor – Material B")
        
        dens_udf_b = st.text_area(
            "getDensity.F90 – Material B",
            value=f"""FUNCTION getDensity_{mat_b_name}(model, n, temp) RESULT(denst)
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

  IF (refTemp <= temp) THEN
      CALL Warn('getDensity', 'Material B in liquid state.')
      denst = refLiqDenst + alphal * (tscaler * (temp - 910.0_dp))
  ELSE
      denst = refSolDenst + alphas * (tscaler * (temp - 298.0_dp))
  END IF
END FUNCTION getDensity_{mat_b_name}""",
            height=350,
            key=uk("matB", "udf_dens")
        )
        
        cond_udf_b = st.text_area(
            "getThermalConductivity.F90 – Material B",
            value=f"""FUNCTION getThermalConductivity_{mat_b_name}(model, n, temp) RESULT(thcondt)
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

  IF (refTemp <= temp) THEN
      CALL Warn('getThermalConductivity', 'Material B in liquid state.')
      thcondt = refLiqThCond + alphal * (tscaler * (temp - 900.0_dp))
  ELSE
      thcondt = refSolThCond + betas * (tscaler * (temp - 298.0_dp)) + &
                alphas * (tscaler * (temp - 298.0_dp))**2
  END IF
END FUNCTION getThermalConductivity_{mat_b_name}""",
            height=350,
            key=uk("matB", "udf_cond")
        )

# ====================== TAB 2: HEAT SOURCE ======================
with tab_heat:
    st.header("🔦 Laser Heat Source Function")
    
    heat_type = st.selectbox(
        "Heat Source Type",
        ["Travelling Gaussian", "Fixed Gaussian", "Super-Gaussian (Flat-Top)", "Custom"],
        key=uk("heat", "type")
    )
    
    # Default templates
    if heat_type == "Travelling Gaussian":
        default_heat = f"""! Travelling Gaussian Heat Source for {project_name}
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
    elif heat_type == "Super-Gaussian (Flat-Top)":
        default_heat = f"""! Super-Gaussian Travelling Heat Source (Flat-Top) for {project_name}
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
    else:
        default_heat = f"""! Custom Heat Source for {project_name} - Edit as needed
FUNCTION CustomHeatSource(Model, n, t) RESULT(f)
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t) :: Model; INTEGER :: n; REAL(KIND=dp) :: t, f
  ! Add your custom heat source logic here
  f = 0.0_dp  ! Placeholder
END FUNCTION CustomHeatSource"""
    
    heat_udf = st.text_area(
        "Edit Heat Source Fortran Function",
        value=default_heat,
        height=500,
        key=uk("heat", "udf")
    )
    
    # Heat source parameters (read by UDF via GetCReal)
    st.subheader("📋 Parameters Read by Heat Source UDF")
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
    
    if heat_type == "Super-Gaussian (Flat-Top)":
        st.subheader("🔷 Super-Gaussian Parameters")
        col_sg1, col_sg2 = st.columns(2)
        with col_sg1:
            sgo = st.number_input("Super-Gaussian Order n", value=3.0, step=0.1, key=uk("heat", "sgo"))
            m1 = st.number_input("Amplitude Prefactor m₁", value=2.0, step=0.1, key=uk("heat", "m1"))
        with col_sg2:
            m2 = st.number_input("Exponential Prefactor m₂", value=2.0, step=0.1, key=uk("heat", "m2"))
            rsgo = 1.0 / sgo if sgo > 0 else 0.3333
            st.info(f"Reciprocal 1/n = {rsgo:.4f} (auto-computed)")

# ====================== TAB 3: LOOKUP TABLES ======================
with tab_tables:
    st.header("📊 Lookup Tables (.dat files)")
    st.markdown("Tab-separated files for viscosity and specific enthalpy vs. temperature. Pre-loaded with sample data.")
    
    table_choice = st.selectbox(
        "Select Table to Edit",
        ["Viscosity – Material A", "Viscosity – Material B", 
         "Specific Enthalpy – Material A", "Specific Enthalpy – Material B"],
        key=uk("table", "select")
    )
    
    # Pre-loaded sample data matching your earlier values
    if "Viscosity – Material A" in table_choice:
        default_df = pd.DataFrame({
            "Temperature_K": [300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 933.5, 1000.0],
            "Viscosity_Pas": [1.2e-3, 1.0e-3, 0.85e-3, 0.7e-3, 0.6e-3, 0.5e-3, 0.4e-3, 0.35e-3, 0.3e-3]
        })
        fname = f"mu_{mat_a_name.lower().replace('-', '_')}.dat"
        col1_name, col2_name = "Temperature_K", "Viscosity_Pas"
    elif "Viscosity – Material B" in table_choice:
        default_df = pd.DataFrame({
            "Temperature_K": [300.0, 500.0, 800.0, 1000.0, 1356.6, 1400.0, 1600.0, 1800.0, 2000.0],
            "Viscosity_Pas": [4.0e-3, 3.2e-3, 2.5e-3, 2.0e-3, 1.5e-3, 1.4e-3, 1.2e-3, 1.0e-3, 0.9e-3]
        })
        fname = f"mu_{mat_b_name.lower().replace('-', '_')}.dat"
        col1_name, col2_name = "Temperature_K", "Viscosity_Pas"
    elif "Specific Enthalpy – Material A" in table_choice:
        default_df = pd.DataFrame({
            "Temperature_K": [300.0, 310.0, 320.0, 400.0, 500.0, 600.0, 700.0, 800.0, 842.71, 850.0, 900.0, 1000.0],
            "Enthalpy_Jkg": [0.0, 8.98e3, 1.80e4, 9.05e4, 1.81e5, 2.72e5, 3.63e5, 4.54e5, 5.16e5, 5.24e5, 6.02e5, 7.50e5]
        })
        fname = f"h_{mat_a_name.lower().replace('-', '_')}.dat"
        col1_name, col2_name = "Temperature_K", "Enthalpy_Jkg"
    else:  # Enthalpy B
        default_df = pd.DataFrame({
            "Temperature_K": [300.0, 500.0, 800.0, 1000.0, 1356.6, 1400.0, 1600.0, 1800.0, 2000.0],
            "Enthalpy_Jkg": [0.0, 1.54e5, 3.08e5, 4.62e5, 6.67e5, 6.88e5, 7.90e5, 8.92e5, 9.94e5]
        })
        fname = f"h_{mat_b_name.lower().replace('-', '_')}.dat"
        col1_name, col2_name = "Temperature_K", "Enthalpy_Jkg"
    
    # Editable dataframe with unique key
    edited_df = st.data_editor(
        default_df,
        num_rows="dynamic",
        key=uk("table", f"editor_{table_choice.replace(' ', '_')}"),
        column_config={
            col1_name: st.column_config.NumberColumn("T [K]", min_value=0, format="%.2f"),
            col2_name: st.column_config.NumberColumn(
                "Viscosity [Pa·s]" if "Viscosity" in table_choice else "Enthalpy [J/kg]",
                format="%.2e" if "Viscosity" in table_choice else "%.0f"
            )
        },
        hide_index=True
    )
    
    # Download button with unique key
    csv_content = edited_df.to_csv(sep='\t', index=False, float_format='%.6f')
    st.download_button(
        label=f"⬇️ Download {fname}",
        data=csv_content,
        file_name=fname,
        mime="text/tab-separated-values",
        key=uk("table", f"dl_{table_choice.replace(' ', '_')}")
    )
    
    # File upload option
    st.subheader("📤 Upload Existing .dat File")
    uploaded_file = st.file_uploader(
        "Choose TSV file", type=["dat", "tsv", "txt"], 
        key=uk("table", f"upload_{table_choice.replace(' ', '_')}")
    )
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file, sep=r'\s+|\t', engine='python', header=0)
            if df_upload.shape[1] >= 2:
                df_upload.columns = [col1_name, col2_name]
                st.session_state[f"table_data_{table_choice}"] = df_upload
                st.success(f"✅ Loaded {len(df_upload)} rows from `{uploaded_file.name}`")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

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
    st.subheader("🔗 Boundary Conditions")
    
    # Fixed displacement faces
    st.markdown("🔹 Fixed Displacement (Zero Velocity)")
    bc_fixed = st.multiselect(
        "Select Face IDs for Fixed BC",
        ["Face_1", "Face_2", "Face_3", "Face_4", "Face_5", "Face_6", "Face_7", "Face_8", "Face_9", "Face_10", "Face_11"],
        default=["Face_1", "Face_3", "Face_4", "Face_8"],
        key=uk("phys", "bcfixed")
    )
    
    # Thermal BCs
    st.markdown("🔹 Convective Cooling")
    bc_conv = st.multiselect(
        "Faces with Convective BC (h=15 W/m²K, T∞=298K)",
        ["Face_1", "Face_2", "Face_3", "Face_4", "Face_5", "Face_6", "Face_7", "Face_8", "Face_9", "Face_10", "Face_11"],
        default=["Face_2", "Face_5"],
        key=uk("phys", "bcconv")
    )
    htc_value = st.number_input("Heat Transfer Coefficient h [W/m²K]", value=15.0, step=1.0, key=uk("phys", "htc"))
    
    # Fixed temperature
    st.markdown("🔹 Fixed Temperature (Dirichlet)")
    bc_temp = st.multiselect(
        "Faces with Fixed T = 298 K",
        ["Face_1", "Face_2", "Face_3", "Face_4", "Face_5", "Face_6", "Face_7", "Face_8", "Face_9", "Face_10", "Face_11"],
        default=["Face_5"],
        key=uk("phys", "bctemp")
    )
    
    # Heat flux face
    st.markdown("🔹 Laser Heat Flux Boundary")
    heat_face = st.selectbox(
        "Face ID for Laser Heat Flux",
        ["Face_1", "Face_2", "Face_3", "Face_4", "Face_5", "Face_6", "Face_7", "Face_8", "Face_9", "Face_10", "Face_11"],
        index=6,  # Default to Face_7 (top)
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

# ====================== TAB 5: GENERATE FILES ======================
with tab_generate:
    st.header("📥 Generate Complete Elmer Input Files")
    
    if st.button("🔄 Generate All Files", type="primary", use_container_width=True, key="uk_gen_btn"):
        # === PREPARE SUBSTITUTION DICTIONARY ===
        coord_val = coord_scaling.split()[0]
        timestep_intervals = f"{n_steps_initial} {n_steps_main}"
        timestep_sizes = f"{dt_initial:.1e} {dt_main:.1e}"
        output_intervals = "1 1"
        
        # Heat source procedure name
        if heat_type == "Super-Gaussian (Flat-Top)":
            heat_proc = "FlatTopHeatSource"
        elif heat_type == "Fixed Gaussian":
            heat_proc = "FixedHeatSource"
        else:
            heat_proc = "TravellingHeatSource"
        
        # BC face indices (convert Face_N to integer)
        def face_to_idx(face_list):
            return " ".join(str(int(re.search(r'Face_(\d+)', f).group(1))) for f in face_list if re.search(r'Face_(\d+)', f))
        
        bc_fixed_idx = face_to_idx(bc_fixed) or "1"
        bc_conv_idx = face_to_idx(bc_conv) or "2"
        bc_temp_idx = face_to_idx(bc_temp) or "5"
        
        # === BUILD COMPLETE .sif FILE ===
        sif_template = Template(f"""    !Phase change solid-liquid
    !Elmer solver input file for transient solid-liquid phase change with enthalpy formulation
    !Bilayer: {mat_a_name} (Body 1) / {mat_b_name} (Body 2)
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
      ! Parameters for the TravellingHeatSource / FlatTopHeatSource
      ! Terms related to Eq. 8 of Kunwar et al, Journal of Materials Science & Technology, 2020 (50), pp. 115-127
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      Heat Source Width = Real {beam_radius}
      Heat Source Coefficient = Real {heat_coeff}
      Heat Source Speed x = Real {speed_x}
      Heat Source Speed y = Real {speed_y}
      Heat Source Distance = Real {scan_dist}
      Heat source initial position x = Real {init_x}
      Heat source initial position y = Real {init_y}
      Super gaussian order n = Real {sgo if heat_type == "Super-Gaussian (Flat-Top)" else 3.0}
      reciproccal of Super gaussian order 1/n = Real {1.0/sgo if heat_type == "Super-Gaussian (Flat-Top)" and sgo>0 else 0.3333}
      prefactor within amplitude term = Real {m1 if heat_type == "Super-Gaussian (Flat-Top)" else 2.0}
      prefactor within exponential term = Real {m2 if heat_type == "Super-Gaussian (Flat-Top)" else 2.0}
      Absorptance of Top Surface Material = Real {absorptance}
      Absorptance of Bottom Surface Material = Real {absorptance}
      Mesh Levels = 1
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
      Name = "Solid_1front"
      Equation = 1
      Material = 1
      Body Force = 1
      Initial condition = 1
    End

    Body 2
      Target Bodies(1) = 2
      Name = "Solid_2back"
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
      Check Latent Heat Release = {'True' if latent_release else 'False'}
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
      Isotropic elastic modulus in elastic regime in Pa = Real {young_a_s.split('=')[1].strip() if '=' in young_a_s else '68.9e9'}
      Yield strength of the alloy materials in Pa = Real 249.0e6
      Strength coefficient in Ramberg-Osgood equation = Real 381.08e6
      Reciprocal of strain hardening coefficient = Real 9.7087
      Poisson Ratio = Real {poisson_a}
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
      Reference Thermal Conductivity Solid {mat_a_name} = Real {cond_a_s.split('=')[1].strip().split('+')[0].strip() if '=' in cond_a_s and '+' in cond_a_s else '167.0'}
      Cond Coeff As Solid {mat_a_name} = Real -1.17E-04
      Cond Coeff Bs Solid {mat_a_name} = Real 9.29E-03
      Reference Thermal Conductivity Liquid {mat_a_name} = Real {cond_a_l.split('=')[1].strip().split('-')[0].strip() if '=' in cond_a_l and '-' in cond_a_l else '90.0'}
      Cond Coeff Liquid {mat_a_name} = Real 1.83E-02
      Melting Point Temperature of {mat_a_name} = Real {mat_a_melting}
      Density = Variable Temperature
      Procedure "getFilmDensity" "getDensity"
      Reference Density Solid {mat_a_name} = Real {dens_a_s.split('=')[1].strip().split('-')[0].strip() if '=' in dens_a_s else '2700.0'}
      Density Coeff Solid {mat_a_name} = Real -0.1898
      Reference Density Liquid {mat_a_name} = Real {dens_a_l.split('=')[1].strip().split('-')[0].strip() if '=' in dens_a_l else '2380.0'}
      Density Coefficient Liquid {mat_a_name} = Real -0.3153
      Tscaler = Real 1.0
    End

    Material 2
      Name = "{mat_b_name}"
      Youngs Modulus = Variable vonMises
      Procedure "plasticityMaterialModel" "getPlasticity"
      Isotropic elastic modulus in elastic regime in Pa = Real {young_b_s.split('=')[1].strip() if '=' in young_b_s else '115.0e9'}
      Yield strength of the alloy materials in Pa = Real 249.0e6
      Strength coefficient in Ramberg-Osgood equation = Real 381.08e6
      Reciprocal of strain hardening coefficient = Real 9.7087
      Poisson Ratio = Real {poisson_b}
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
      Reference Thermal Conductivity Solid {mat_b_name} = Real {cond_b_s.split('=')[1].strip().split('-')[0].strip() if '=' in cond_b_s and '-' in cond_b_s else '391.0'}
      Cond Coeff As Solid {mat_b_name} = Real -0.052
      Cond Coeff Bs Solid {mat_b_name} = Real 0.0
      Reference Thermal Conductivity Liquid {mat_b_name} = Real {cond_b_l.split('=')[1].strip().split('-')[0].strip() if '=' in cond_b_l and '-' in cond_b_l else '170.0'}
      Cond Coeff Liquid {mat_b_name} = Real -0.025
      Melting Point Temperature of {mat_b_name} = Real {mat_b_melting}
      Density = Variable Temperature
      Procedure "getFilmDensity" "getDensity"
      Reference Density Solid {mat_b_name} = Real {dens_b_s.split('=')[1].strip().split('-')[0].strip() if '=' in dens_b_s else '8940.0'}
      Density Coeff Solid {mat_b_name} = Real -0.52
      Reference Density Liquid {mat_b_name} = Real {dens_b_l.split('=')[1].strip().split('-')[0].strip() if '=' in dens_b_l else '7992.0'}
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
      Target Boundaries(1) = {bc_temp_idx.split()[0] if bc_temp_idx else '5'}
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
      Target Boundaries(1) = {re.search(r'Face_(\d+)', heat_face).group(1) if re.search(r'Face_(\d+)', heat_face) else '7'}
      Heat Flux = Variable time
      Real Procedure "DifferentTypeHeatSource" "{heat_proc}"
      Save Line = True
    End
""")
        
        sif_content = sif_template.substitute()
        
        # === PREPARE FORTRAN UDF FILES ===
        # Replace ${MAT_NAME} placeholders in UDFs
        def finalize_udf(code, mat_name):
            return code.replace(f"getDensity_{{mat_name}}", f"getDensity_{mat_name}") \
                      .replace(f"getThermalConductivity_{{mat_name}}", f"getThermalConductivity_{mat_name}")
        
        dens_f90_a = finalize_udf(dens_udf_a, mat_a_name)
        cond_f90_a = finalize_udf(cond_udf_a, mat_a_name)
        dens_f90_b = finalize_udf(dens_udf_b, mat_b_name)
        cond_f90_b = finalize_udf(cond_udf_b, mat_b_name)
        heat_f90 = heat_udf
        
        # === DISPLAY DOWNLOAD BUTTONS WITH UNIQUE KEYS ===
        st.success(f"✅ Files generated for project `{project_name}`!")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            st.download_button(
                label="📄 Download case.sif",
                data=sif_content,
                file_name=sif_filename,
                mime="text/plain",
                key=uk("dl", "sif")
            )
            st.download_button(
                label="🔧 getDensity_A.F90",
                data=dens_f90_a,
                file_name=f"getDensity_{mat_a_name}.F90",
                mime="text/plain",
                key=uk("dl", "dens_a")
            )
            st.download_button(
                label="🔧 getDensity_B.F90",
                data=dens_f90_b,
                file_name=f"getDensity_{mat_b_name}.F90",
                mime="text/plain",
                key=uk("dl", "dens_b")
            )
        
        with col_dl2:
            st.download_button(
                label="🔧 getThermalConductivity_A.F90",
                data=cond_f90_a,
                file_name=f"getThermalConductivity_{mat_a_name}.F90",
                mime="text/plain",
                key=uk("dl", "cond_a")
            )
            st.download_button(
                label="🔧 getThermalConductivity_B.F90",
                data=cond_f90_b,
                file_name=f"getThermalConductivity_{mat_b_name}.F90",
                mime="text/plain",
                key=uk("dl", "cond_b")
            )
            st.download_button(
                label="🔧 HeatSource.F90",
                data=heat_f90,
                file_name="DifferentTypeHeatSource.F90",
                mime="text/plain",
                key=uk("dl", "heat")
            )
        
        with col_dl3:
            # Lookup tables
            for choice, df in [
                ("Viscosity – Material A", pd.DataFrame({"Temperature_K": [300], "Viscosity_Pas": [1.2e-3]})),
                ("Viscosity – Material B", pd.DataFrame({"Temperature_K": [300], "Viscosity_Pas": [4.0e-3]})),
                ("Specific Enthalpy – Material A", pd.DataFrame({"Temperature_K": [300], "Enthalpy_Jkg": [0]})),
                ("Specific Enthalpy – Material B", pd.DataFrame({"Temperature_K": [300], "Enthalpy_Jkg": [0]}))
            ]:
                if "Viscosity" in choice:
                    mat = mat_a_name if "A" in choice else mat_b_name
                    fname = f"mu_{mat.lower().replace('-', '_')}.dat"
                    content = edited_df.to_csv(sep='\t', index=False, float_format='%.6f') if choice == table_choice else df.to_csv(sep='\t', index=False)
                else:
                    mat = mat_a_name if "A" in choice else mat_b_name
                    fname = f"h_{mat.lower().replace('-', '_')}.dat"
                    content = edited_df.to_csv(sep='\t', index=False, float_format='%.6f') if choice == table_choice else df.to_csv(sep='\t', index=False)
                
                st.download_button(
                    label=f"📊 {fname}",
                    data=content,
                    file_name=fname,
                    mime="text/tab-separated-values",
                    key=uk("dl", f"table_{choice.replace(' ', '_')}")
                )
        
        # === INSTRUCTIONS ===
        with st.expander("📋 How to Compile & Run", expanded=True):
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
            │   └── DifferentTypeHeatSource.F90
            ├── {table_dir_visc}
            │   ├── mu_{mat_a_name.lower().replace('-', '_')}.dat
            │   └── mu_{mat_b_name.lower().replace('-', '_')}.dat
            └── {table_dir_enth}
                ├── h_{mat_a_name.lower().replace('-', '_')}.dat
                └── h_{mat_b_name.lower().replace('-', '_')}.dat
            ```
            
            **Compilation Steps:**
            ```bash
            # 1. Compile Fortran UDFs
            cd {fortran_dir}
            elmerfem -c getDensity_{mat_a_name}.F90
            elmerfem -c getDensity_{mat_b_name}.F90
            elmerfem -c getThermalConductivity_{mat_a_name}.F90
            elmerfem -c getThermalConductivity_{mat_b_name}.F90
            elmerfem -c DifferentTypeHeatSource.F90
            
            # 2. Link and run Elmer
            cd ..
            ElmerSolver {sif_filename}
            ```
            
            **Troubleshooting:**
            - Ensure all `GetConstReal` parameter names in Fortran match exactly with .sif Material block keys
            - Use `REAL(KIND=dp)` consistently for double precision
            - For phase change: mushy zone ±{mushy_width} K around melting point ensures numerical stability
            - Monitor `Nonlinear System Relaxation Factor` (0.6 recommended) if convergence issues arise
            
            **Reference:** Kunwar et al., *J. Mater. Sci. Technol.* 50 (2020) 115-127
            """)

# ====================== FOOTER ======================
st.markdown("---")
st.markdown("""
**💡 Pro Tips for Large Simulations:**
- Use `Coordinate Scaling = 1.0e-6` for µm-scale geometries to avoid floating-point precision issues
- Start with coarse mesh + large Δt for testing, then refine
- For Al-Cu welding: IMC layer formation may require a third material with distinct properties
- Enable `Binary Output = Logical True` for faster I/O with large datasets

**🔗 Resources:**
- [Elmer FEM Documentation](https://www.elmerfem.org)
- [Fortran UDF Guide](https://github.com/ElmerCSC/elmerfem/blob/devel/fem/src/modules/DefUtils.F90)
- [Al-Cu Property Database](https://materials.springer.com)
""")

# Debug info (remove in production)
if st.checkbox("Show Debug Info", key="uk_debug"):
    st.json({
        "project": project_name,
        "materials": {"A": mat_a_name, "B": mat_b_name},
        "mesh": mesh_name,
        "heat_source": heat_type,
        "timesteps": {"initial": dt_initial, "main": dt_main, "total": n_steps_initial + n_steps_main}
    })
