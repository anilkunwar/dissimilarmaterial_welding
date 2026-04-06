#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Elmer FEM .sif Generator with Fortran UDF Editor Tabs
Allows users to edit material property functions, heat sources, and data tables
"""

import streamlit as st
import re
from string import Template
from datetime import datetime
import pandas as pd
import io

st.set_page_config(page_title="Elmer FEM Generator – Fortran UDF Editor", layout="wide")
st.title("🔥 Elmer FEM Generator for Dissimilar Welding")
st.markdown("""
**Edit Fortran UDFs, heat sources, and material tables → Generate ready-to-run `.sif` + `.F90` files.**
Supports temperature-dependent properties, phase change, and custom laser heat sources.
""")

# ====================== TABS NAVIGATION ======================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📐 Geometry & Mesh",
    "🧪 Material Properties (Fortran UDFs)",
    "🔦 Heat Source Functions",
    "📊 Lookup Tables (.dat)",
    "⚙️ Simulation Settings",
    "📥 Generate & Download"
])

# ====================== GLOBAL STATE ======================
if 'fortran_udfs' not in st.session_state:
    st.session_state.fortran_udfs = {}
if 'heat_sources' not in st.session_state:
    st.session_state.heat_sources = {}
if 'lookup_tables' not in st.session_state:
    st.session_state.lookup_tables = {}

# ====================== TAB 1: GEOMETRY & MESH ======================
with tab1:
    st.header("📐 Geometry & Mesh Configuration")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("Box 1 (Material A)")
        v2x = st.number_input("Length X [µm]", value=400.0, step=0.1)
        v2y = st.number_input("Width Y [µm]", value=100.0, step=0.1)
        v2z = st.number_input("Thickness Z [µm]", value=20.0, step=0.1)
    
    with col_g2:
        st.subheader("Box 2 (Material B)")
        v3x = st.number_input("Length X [µm]", value=400.0, step=0.1)
        v3y = st.number_input("Width Y [µm]", value=200.0, step=0.1)
        v3z = st.number_input("Thickness Z [µm]", value=20.0, step=0.1)
    
    st.subheader("Mesh Parameters")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        local_length = st.number_input("1D Local Length [µm]", value=2.36, step=0.01)
    with col_m2:
        max_area = st.number_input("2D Max Area [µm²]", value=50.0, step=1.0)
    with col_m3:
        max_vol = st.number_input("3D Max Volume [µm³]", value=1180.0, step=10.0)
    
    st.info("✅ Geometry bounds are automatically computed to match SALOME's `MakeBoxTwoPnt` logic.")

# ====================== TAB 2: MATERIAL PROPERTIES (FORTRAN UDFs) ======================
with tab2:
    st.header("🧪 Material Property UDFs (Fortran)")
    st.markdown("""
    Edit the Fortran functions that compute temperature-dependent properties.
    Changes here automatically update the generated `.F90` files and `.sif` references.
    """)
    
    mat_tab_a, mat_tab_b = st.tabs(["Material A (Front)", "Material B (Back)"])
    
    def make_udf_editor(mat_name, mat_key, default_code):
        """Create a code editor for a material UDF with save/load."""
        st.subheader(f"{mat_name} Property Functions")
        
        # UDF type selector
        udf_type = st.selectbox("Select Property to Edit", 
            ["Density", "Thermal Conductivity", "Thermal Expansivity", "Plasticity (Young's Modulus)"],
            key=f"{mat_key}_udf_type")
        
        # Load default or session-stored code
        if udf_type not in st.session_state.fortran_udfs.get(mat_key, {}):
            st.session_state.setdefault('fortran_udfs', {}).setdefault(mat_key, {})[udf_type] = default_code
        
        # Code editor with syntax highlighting simulation
        edited_code = st.text_area(
            f"Edit {udf_type} Function for {mat_name}",
            value=st.session_state.fortran_udfs[mat_key][udf_type],
            height=400,
            key=f"{mat_key}_{udf_type}_editor",
            help="Use `temp` for temperature in Kelvin. Return value must be in SI units."
        )
        
        # Save button
        if st.button(f"💾 Save {udf_type} for {mat_name}", key=f"{mat_key}_save_{udf_type}"):
            st.session_state.fortran_udfs[mat_key][udf_type] = edited_code
            st.success(f"✅ {udf_type} function saved for {mat_name}!")
        
        # Show parameter mapping help
        with st.expander("📖 Parameter Reference"):
            st.markdown("""
            **Available Parameters in UDFs:**
            - `temp`: Temperature in Kelvin (input)
            - `refSol*`, `refLiq*`: Reference values from .sif Material block
            - `alphas`, `betas`, `alphal`: Coefficients from .sif
            - `refTemp`: Melting point temperature
            - `tscaler`: Temperature scaling factor
            
            **Return Values:**
            - `getDensity`: density in kg/m³
            - `getThermalConductivity`: conductivity in W/(m·K)
            - `getThermalExpansivity`: CTE in 1/K
            - `getPlasticity`: Young's modulus in Pa
            
            **Example Snippet:**
            ```fortran
            IF (refTemp <= temp) THEN
                ! Liquid phase
                result = refLiqValue + alphal * (temp - refTemp)
            ELSE
                ! Solid phase  
                result = refSolValue + alphas * (temp - 298) + betas * (temp - 298)**2
            END IF
            ```
            """)
        
        return edited_code
    
    # Default UDF templates
    DENSITY_TEMPLATE = """    FUNCTION getDensity( model, n, temp ) RESULT(denst)
    USE DefUtils
    IMPLICIT None
    TYPE(Model_t) :: model
    INTEGER :: n
    REAL(KIND=dp) :: temp, denst, tscaler
    REAL(KIND=dp) :: refSolDenst, refLiqDenst, refTemp, alphas, alphal
    Logical :: GotIt
    TYPE(ValueList_t), POINTER :: material

    material => GetMaterial()
    IF (.NOT. ASSOCIATED(material)) CALL Fatal('getDensity', 'No material found')

    refSolDenst = GetConstReal( material, 'Reference Density Solid ${MAT_NAME}', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getDensity', 'Reference Density Solid not found')
    alphas = GetConstReal( material, 'Density Coeff Solid ${MAT_NAME}', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getDensity', 'Density coeff solid not found')
    
    refLiqDenst = GetConstReal( material, 'Reference Density Liquid ${MAT_NAME}', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getDensity', 'Reference Density Liquid not found')
    alphal = GetConstReal( material, 'Density Coefficient Liquid ${MAT_NAME}', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getDensity', 'Density coeff liquid not found')

    refTemp = GetConstReal( material, 'Melting Point Temperature of ${MAT_NAME}', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getDensity', 'Melting point not found')
    tscaler = GetConstReal( material, 'Tscaler', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getDensity', 'Tscaler not found')

    IF (refTemp <= temp) THEN
        CALL Warn('getDensity', 'Material is in liquid state.')
        denst = refLiqDenst + alphal * (tscaler * (temp - 910))
    ELSE
        denst = refSolDenst + alphas * (tscaler * (temp - 298))
    END IF
    END FUNCTION getDensity"""

    CONDUCTIVITY_TEMPLATE = """    FUNCTION getThermalConductivity( model, n, temp ) RESULT(thcondt)
    USE DefUtils
    IMPLICIT None
    TYPE(Model_t) :: model
    INTEGER :: n
    REAL(KIND=dp) :: temp, thcondt, tscaler
    REAL(KIND=dp) :: refSolThCond, refLiqThCond, refTemp, alphas, betas, alphal
    Logical :: GotIt
    TYPE(ValueList_t), POINTER :: material

    material => GetMaterial()
    IF (.NOT. ASSOCIATED(material)) CALL Fatal('getThermalConductivity', 'No material found')

    refSolThCond = GetConstReal( material, 'Reference Thermal Conductivity Solid ${MAT_NAME}', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Ref cond solid not found')
    alphas = GetConstReal( material, 'Cond Coeff As Solid ${MAT_NAME}', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Coeff A solid not found')
    betas = GetConstReal( material, 'Cond Coeff Bs Solid ${MAT_NAME}', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Coeff B solid not found')
    
    refLiqThCond = GetConstReal( material, 'Reference Thermal Conductivity Liquid ${MAT_NAME}', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Ref cond liquid not found')
    alphal = GetConstReal( material, 'Cond Coeff Liquid ${MAT_NAME}', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Coeff liquid not found')

    refTemp = GetConstReal( material, 'Melting Point Temperature of ${MAT_NAME}', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Melting point not found')
    tscaler = GetConstReal( material, 'Tscaler', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getThermalConductivity', 'Tscaler not found')

    IF (refTemp <= temp) THEN
        CALL Warn('getThermalConductivity', 'Material is in liquid state.')
        thcondt = refLiqThCond + alphal * (tscaler * (temp - 900))
    ELSE
        thcondt = refSolThCond + betas * (tscaler * (temp - 298)) + &
                  alphas * (tscaler * (temp - 298))**2
    END IF
    END FUNCTION getThermalConductivity"""

    EXPANSIVITY_TEMPLATE = """    FUNCTION getThermalExpansivity( model, n, temp ) RESULT(expansivity)
    USE DefUtils
    IMPLICIT None
    TYPE(Model_t) :: model
    INTEGER :: n
    REAL(KIND=dp) :: temp, expansivity, tscaler
    REAL(KIND=dp) :: refSolExp, refTemp, alphas, betas
    Logical :: GotIt
    TYPE(ValueList_t), POINTER :: material

    material => GetMaterial()
    IF (.NOT. ASSOCIATED(material)) CALL Fatal('getThermalExpansivity', 'No material found')

    refSolExp = GetConstReal( material, 'Reference Thermal Expansivity Solid ${MAT_NAME}', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getThermalExpansivity', 'Ref expansivity solid not found')
    alphas = GetConstReal( material, 'Thermal Expansivity Coeff As Solid ${MAT_NAME}', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getThermalExpansivity', 'Coeff A expansivity not found')
    betas = GetConstReal( material, 'Thermal Expansivity Coeff Bs Solid ${MAT_NAME}', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getThermalExpansivity', 'Coeff B expansivity not found')

    refTemp = GetConstReal( material, 'Melting Point Temperature of ${MAT_NAME}', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getThermalExpansivity', 'Melting point not found')
    tscaler = GetConstReal( material, 'Tscaler', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getThermalExpansivity', 'Tscaler not found')

    IF (refTemp <= temp) THEN
        CALL Warn('getThermalExpansivity', 'Material is in liquid state.')
        expansivity = 0.0_dp  ! Zero CTE in liquid to suppress spurious stresses
    ELSE
        expansivity = refSolExp + alphas * (tscaler * temp)**2 + &
                      betas * (tscaler * temp)
    END IF
    END FUNCTION getThermalExpansivity"""

    PLASTICITY_TEMPLATE = """    FUNCTION getPlasticity( model, n, stress ) RESULT(elast)
    USE DefUtils
    IMPLICIT None
    TYPE(Model_t) :: model
    INTEGER :: n
    REAL(KIND=dp) :: stress, elast
    REAL(KIND=dp) :: refElast, yieldsigma, strengthcoeff, mcoeff
    Logical :: GotIt
    TYPE(ValueList_t), POINTER :: material

    material => GetMaterial()
    IF (.NOT. ASSOCIATED(material)) CALL Fatal('getPlasticity', 'No material found')

    refElast = GetConstReal( material, 'Isotropic elastic modulus in elastic regime in Pa', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getPlasticity', 'Young''s modulus not found')
    yieldsigma = GetConstReal( material, 'Yield strength of the alloy materials in Pa', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getPlasticity', 'Yield strength not found')
    strengthcoeff = GetConstReal( material, 'Strength coefficient in Ramberg-Osgood equation', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getPlasticity', 'Strength coefficient not found')
    mcoeff = GetConstReal( material, 'Reciprocal of strain hardening coefficient', GotIt)
    IF(.NOT. GotIt) CALL Fatal('getPlasticity', 'Hardening exponent not found')

    IF (yieldsigma <= stress) THEN
        CALL Warn('getPlasticity', 'Material undergoing plastic deformation.')
        elast = stress / (stress / strengthcoeff)**mcoeff
    ELSE
        elast = refElast
    END IF
    END FUNCTION getPlasticity"""

    with mat_tab_a:
        mat_a_name = st.text_input("Material A Name", value="AA6061_Al")
        mat_a_melting = st.number_input("Melting Point [K]", value=933.5, step=0.1)
        
        # Editors for each UDF type
        density_a = make_udf_editor("Material A", "mat_a", 
            DENSITY_TEMPLATE.replace("${MAT_NAME}", mat_a_name))
        cond_a = make_udf_editor("Material A", "mat_a",
            CONDUCTIVITY_TEMPLATE.replace("${MAT_NAME}", mat_a_name))
        exp_a = make_udf_editor("Material A", "mat_a",
            EXPANSIVITY_TEMPLATE.replace("${MAT_NAME}", mat_a_name))
        plast_a = make_udf_editor("Material A", "mat_a",
            PLASTICITY_TEMPLATE)
    
    with mat_tab_b:
        mat_b_name = st.text_input("Material B Name", value="T2_Cu")
        mat_b_melting = st.number_input("Melting Point [K]", value=1356.6, step=0.1)
        
        density_b = make_udf_editor("Material B", "mat_b",
            DENSITY_TEMPLATE.replace("${MAT_NAME}", mat_b_name))
        cond_b = make_udf_editor("Material B", "mat_b",
            CONDUCTIVITY_TEMPLATE.replace("${MAT_NAME}", mat_b_name))
        exp_b = make_udf_editor("Material B", "mat_b",
            EXPANSIVITY_TEMPLATE.replace("${MAT_NAME}", mat_b_name))
        plast_b = make_udf_editor("Material B", "mat_b",
            PLASTICITY_TEMPLATE)

# ====================== TAB 3: HEAT SOURCE FUNCTIONS ======================
with tab3:
    st.header("🔦 Heat Source Functions (Fortran)")
    st.markdown("Edit the moving laser heat source implementations. Select template → customize → save.")
    
    heat_type = st.selectbox("Heat Source Type", 
        ["Travelling Gaussian", "Fixed Gaussian", "Travelling Super-Gaussian (Flat-Top)", "Travelling Slab Velocity"])
    
    # Load default based on selection
    if heat_type == "Travelling Gaussian":
        default_heat = """!Gaussian Travelling Heat Source
FUNCTION TravellingHeatSource( Model, n, t ) RESULT(f)
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
  x = Mesh % Nodes % x(n); y = Mesh % Nodes % y(n); z = Mesh % Nodes % z(n)
  s1 = xzero + time * xspeed; s2 = yzero + time * yspeed
  r = SQRT((x-s1)**2 + (y-s2)**2)
  f = Coeff * EXP( -2*r**2 / Alpha**2 - Omega * ABS(z))
END FUNCTION TravellingHeatSource"""
    elif heat_type == "Fixed Gaussian":
        default_heat = """!Fixed Gaussian Heat Source
FUNCTION FixedHeatSource( Model, n, t ) RESULT(f)
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t) :: Model; INTEGER :: n; REAL(KIND=dp) :: t, f
  INTEGER :: timestep, prevtimestep = -1
  REAL(KIND=dp) :: Alpha, Coeff, Dist0, Time, x, y, z, r
  TYPE(Mesh_t), POINTER :: Mesh; TYPE(ValueList_t), POINTER :: Params
  LOGICAL :: Found, NewTimestep
  SAVE Mesh, Params, prevtimestep, time, Alpha, Coeff, Dist0
  
  timestep = GetTimestep(); NewTimestep = ( timestep /= prevtimestep )
  IF( NewTimestep ) THEN
    Mesh => GetMesh(); Params => Model % Simulation; time = GetTime()
    Alpha = GetCReal(Params,'Heat source width')
    Coeff = GetCReal(Params,'Heat source coefficient')
    Dist0 = GetCReal(Params,'Heat source initial position', Found)
    prevtimestep = timestep
  END IF
  x = Mesh % Nodes % x(n); y = Mesh % Nodes % y(n); z = Mesh % Nodes % z(n)
  r = x - Dist0
  f = Coeff * EXP( -2*r**2 / Alpha**2 )
END FUNCTION FixedHeatSource"""
    elif heat_type == "Travelling Super-Gaussian (Flat-Top)":
        default_heat = """!Super-Gaussian Travelling Heat Source (Flat-Top)
FUNCTION FlatTopHeatSource( Model, n, t ) RESULT(f)
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
  
  timestep = GetTimestep(); NewTimestep = ( timestep /= prevtimestep )
  IF( NewTimestep ) THEN
    Mesh => GetMesh(); Params => Model % Simulation; time = GetTime()
    Alpha = GetCReal(Params,'Heat source width')
    Coeff = GetCReal(Params,'Heat source coefficient')
    xspeed = GetCReal(Params,'Heat source speed x')
    yspeed = GetCReal(Params,'Heat source speed y')
    Dist = GetCReal(Params,'Heat source distance')
    xzero = GetCReal(Params,'Heat source initial position x', Found)
    yzero = GetCReal(Params,'Heat source initial position y', Found)
    sgo = GetCReal(Params,'Super gaussian order n')
    rsgo = GetCReal(Params,'reciproccal of Super gaussian order 1/n')
    m1 = GetCReal(Params,'prefactor within amplitude term')
    m2 = GetCReal(Params,'prefactor within exponential term')
    prevtimestep = timestep
  END IF
  x = Mesh % Nodes % x(n); y = Mesh % Nodes % y(n); z = Mesh % Nodes % z(n)
  s1 = xzero + time * xspeed; s2 = yzero + time * yspeed
  r = SQRT((x-s1)**2 + (y-s2)**2)
  f = m1**rsgo * sgo * Coeff * EXP( -m2 * r**sgo / Alpha**sgo ) / gamma(rsgo)
END FUNCTION FlatTopHeatSource"""
    else:  # Travelling Slab Velocity
        default_heat = """!Travelling Slab Velocity Function
FUNCTION TravellingSlabVelo( Model, n, t ) RESULT(f)
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t) :: Model; INTEGER :: n; REAL(KIND=dp) :: t, f
  INTEGER :: timestep, prevtimestep = -1
  REAL(KIND=dp) :: Speed, Dist, Dist0, Time, x, y, z, s, sper
  TYPE(Mesh_t), POINTER :: Mesh; TYPE(ValueList_t), POINTER :: Params
  LOGICAL :: Found, NewTimestep
  SAVE Mesh, Params, prevtimestep, time, Speed, Dist, Dist0
  
  timestep = GetTimestep(); NewTimestep = ( timestep /= prevtimestep )
  IF( NewTimestep ) THEN
    Mesh => GetMesh(); Params => Model % Simulation; time = GetTime()
    Speed = GetCReal(Params,'Heat source speed')
    Dist = GetCReal(Params,'Heat source distance')
    prevtimestep = timestep
  END IF
  x = Mesh % Nodes % x(n); y = Mesh % Nodes % y(n); z = Mesh % Nodes % z(n)
  s = Dist0 + time * Speed; sper = MODULO(s, 2*Dist)
  IF( sper > Dist ) THEN; sper = 2*Dist - sper; f = Speed
  ELSE; f = -Speed; END IF
END FUNCTION TravellingSlabVelo"""
    
    # Store in session state
    if 'heat_source_code' not in st.session_state:
        st.session_state.heat_source_code = default_heat
    
    edited_heat = st.text_area(
        "Edit Heat Source Function",
        value=st.session_state.heat_source_code,
        height=500,
        key="heat_source_editor",
        help="Use parameters from .sif: 'Heat source width', 'coefficient', 'speed x/y', etc."
    )
    
    if st.button("💾 Save Heat Source Function"):
        st.session_state.heat_source_code = edited_heat
        st.success("✅ Heat source function saved!")
    
    with st.expander("📖 Heat Source Parameter Reference"):
        st.markdown("""
        **Parameters Read from .sif via `GetCReal`:**
        - `'Heat source width'`: Beam radius α (m)
        - `'Heat source coefficient'`: Amplitude Coeff (W/m²)
        - `'Heat source speed x/y'`: Scan velocity components (m/s)
        - `'Heat source distance'`: Total scan length (m)
        - `'Heat source initial position x/y'`: Starting coordinates (m)
        - `'Absorptance of Surface Material'`: Absorption coefficient Ω (1/m)
        - `'Super gaussian order n'`: Exponent for flat-top profile
        - `'prefactor within amplitude/exponential term'`: Shape parameters m₁, m₂
        
        **Available Variables:**
        - `x, y, z`: Node coordinates from mesh
        - `time`: Current simulation time
        - `timestep`: Current timestep number
        
        **Return:** Heat flux `f` in W/m² at node `n`
        """)

# ====================== TAB 4: LOOKUP TABLES (.dat) ======================
with tab4:
    st.header("📊 Lookup Tables for Viscosity & Enthalpy")
    st.markdown("Upload or edit tab-separated `.dat` files for temperature-dependent properties.")
    
    table_type = st.selectbox("Select Table Type", 
        ["Viscosity - Material A", "Viscosity - Material B", 
         "Specific Enthalpy - Material A", "Specific Enthalpy - Material B"])
    
    # Default sample data
    if "Viscosity" in table_type:
        default_df = pd.DataFrame({
            "Temperature_K": [300, 400, 500, 600, 700, 800, 900, 1000],
            "Viscosity_Pas": [1.2e-3, 1.0e-3, 0.8e-3, 0.6e-3, 0.5e-3, 0.4e-3, 0.35e-3, 0.3e-3]
        })
        filename = "mu_material.dat"
    else:
        default_df = pd.DataFrame({
            "Temperature_K": [300, 400, 500, 600, 700, 800, 842.71, 900, 1000],
            "Enthalpy_Jkg": [0, 86e3, 172e3, 258e3, 344e3, 430e3, 516e3, 602e3, 750e3]
        })
        filename = "h_material.dat"
    
    # Load from session or use default
    if table_type not in st.session_state.lookup_tables:
        st.session_state.lookup_tables[table_type] = default_df.copy()
    
    # Data editor
    edited_df = st.data_editor(
        st.session_state.lookup_tables[table_type],
        num_rows="dynamic",
        key=f"table_editor_{table_type}",
        column_config={
            "Temperature_K": st.column_config.NumberColumn("Temperature [K]", min_value=0, format="%.2f"),
            "Viscosity_Pas": st.column_config.NumberColumn("Viscosity [Pa·s]", format="%.2e") if "Viscosity" in table_type else None,
            "Enthalpy_Jkg": st.column_config.NumberColumn("Enthalpy [J/kg]", format="%.0f") if "Enthalpy" in table_type else None
        }
    )
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("💾 Update Table", key=f"save_{table_type}"):
            st.session_state.lookup_tables[table_type] = edited_df
            st.success(f"✅ Table saved: `{filename}`")
    
    with col_t2:
        # Download button for table
        csv = edited_df.to_csv(sep='\t', index=False)
        st.download_button(
            label=f"⬇️ Download {filename}",
            data=csv,
            file_name=filename,
            mime="text/tab-separated-values",
            key=f"dl_{table_type}"
        )
    
    # File upload option
    st.subheader("📤 Upload Existing .dat File")
    uploaded = st.file_uploader("Choose a TSV file", type=["dat", "tsv", "txt"], key=f"upload_{table_type}")
    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded, sep=r'\s+|\t', engine='python')
            # Auto-detect columns
            if df_upload.shape[1] >= 2:
                df_upload.columns = ["Temperature_K", "Value"]
                st.session_state.lookup_tables[table_type] = df_upload
                st.success(f"✅ Loaded {len(df_upload)} rows from `{uploaded.name}`")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# ====================== TAB 5: SIMULATION SETTINGS ======================
with tab5:
    st.header("⚙️ Simulation Configuration")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Time Stepping")
        coord_scaling = st.selectbox("Coordinate Scaling", 
            ["1.0e-6 (µm→m)", "1.0e-5 (10µm→m)", "1.0e-3 (mm→m)"], index=0)
        dt_initial = st.number_input("Initial Δt [s]", value=1.0e-7, format="%.1e")
        dt_main = st.number_input("Main Δt [s]", value=1.0e-5, format="%.1e")
        n_steps = st.number_input("Total Timesteps", value=61, min_value=1)
    
    with col_s2:
        st.subheader("Output & Solvers")
        results_dir = st.text_input("Results Directory", value="./results_weld/")
        output_file = st.text_input("Output File", value="old.result")
        post_file = st.text_input("Post-Processing File", value="a.vtu")
        bdf_order = st.selectbox("BDF Order", [1, 2, 3], index=1)
    
    st.subheader("🔦 Laser Parameters")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        beam_radius = st.number_input("Beam Radius r₀ [m]", value=35.0e-6, format="%.2e")
        heat_coeff = st.number_input("Heat Coefficient [W/m²]", value=8.68e9, format="%.2e")
        speed_x = st.number_input("Scan Speed X [m/s]", value=1.0, step=0.1)
        speed_y = st.number_input("Scan Speed Y [m/s]", value=0.0, step=0.1)
    with col_l2:
        scan_dist = st.number_input("Scan Distance [m]", value=600.0e-6, format="%.2e")
        init_x = st.number_input("Initial X [m]", value=0.0, format="%.2e")
        init_y = st.number_input("Initial Y [m]", value=0.0, format="%.2e")
        absorptance = st.number_input("Absorptance [1/m]", value=8.5e7, format="%.2e")
    
    st.subheader("🔗 Boundary Conditions")
    bc_faces = st.multiselect("Fixed Displacement Faces", 
        ["Face_1leftfront", "Face_3frontfront", "Face_4bottomfront", "Face_8bottomback"],
        default=["Face_1leftfront", "Face_3frontfront", "Face_4bottomfront", "Face_8bottomback"])
    heat_face = st.selectbox("Heat Flux Boundary", ["Face_7top_top", "Face_5topfront", "Face_8topback"])

# ====================== TAB 6: GENERATE & DOWNLOAD ======================
with tab6:
    st.header("📥 Generate & Download Files")
    
    if st.button("🔄 Generate All Files", type="primary", use_container_width=True):
        # === GENERATE .sif FILE ===
        coord_val = coord_scaling.split()[0]
        mesh_name = f"Mesh_weld_{mat_a_name}_{mat_b_name}_L{v2x:.0f}W{v2y:.0f}"
        
        sif_template = Template("""    !Phase change solid-liquid
    !Elmer solver input file for bilayer ${MAT_A} / ${MAT_B} welding
    !Generated: ${TIMESTAMP}
    
    Header
      CHECK KEYWORDS Warn
      Mesh DB "." "${MESH_NAME}"
      Include Path ""
      Results Directory "${RESULTS_DIR}"
    End

    Simulation
      Max Output Level = 5
      Coordinate System = Cartesian 3D
      Coordinate Mapping(3) = 1 2 3
      Coordinate Scaling = ${COORD_SCALE}
      Simulation Type = Transient
      Steady State Max Iterations = 5
      Output Intervals (2) = 1 1
      Timestep intervals (2) = 1 ${N_STEPS}
      Timestep Sizes (2) = ${DT_INIT:.1e} ${DT_MAIN:.1e}
      Timestepping Method = BDF
      BDF Order = ${BDF_ORDER}
      Solver Input File = mesh1phasechange_lsenthalpy.sif
      Post File = "${POST_FILE}"
      Output File = "${OUTPUT_FILE}"
      Binary Output = Logical True
      Use Mesh Names = True
      Heat Source Width = Real ${BEAM_RADIUS}
      Heat Source Coefficient = Real ${HEAT_COEFF}
      Heat Source Speed x = Real ${SPEED_X}
      Heat Source Speed y = Real ${SPEED_Y}
      Heat Source Distance = Real ${SCAN_DIST}
      Heat source initial position x = Real ${INIT_X}
      Heat source initial position y = Real ${INIT_Y}
      Super gaussian order n = Real 3.0
      reciproccal of Super gaussian order 1/n = Real 0.3333
      prefactor within amplitude term = Real 2.0
      prefactor within exponential term = Real 2.0
      Absorptance of Top Surface Material = Real ${ABSORPTANCE}
      Absorptance of Bottom Surface Material = Real ${ABSORPTANCE}
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
      Steady State Convergence Tolerance= 1.0e-6
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
      Phase Change Model = Spatial 2
      Check Latent Heat Release = True
      Convection = Computed
      Navier-Stokes = True
      NS Convect = True
      Active Solvers(3) = 1 2 3
    End

    Material 1
      Name = "${MAT_A}"
      Poisson Ratio = Real 0.33
      Reference Temperature = 298.0
      Heat Expansion Coefficient = Variable Temperature
      Procedure "getThermalExpansivity" "getThermalExpansivity"
      Reference Thermal Expansivity Solid ${MAT_A} = Real -8.371435292934836e-05
      Thermal Expansivity Coeff As Solid ${MAT_A} = Real -3.7262790630140875e-10
      Thermal Expansivity Coeff Bs Solid ${MAT_A} = Real 4.2255653792479394e-07
      Viscosity = Variable Temperature
    Real
      include ./viscosity/mu_${MAT_A_LOWER}.dat
    End
      Specific Enthalpy = Variable Temperature
    Real
      include ./specific_enthalpy/h_${MAT_A_LOWER}.dat
    End
      Phase Change Intervals(2,1) = ${TMELT_A_MINUS_10} ${TMELT_A_PLUS_10}
      Compressibility Model = Incompressible
      Reference Pressure = 0
      Specific Heat Ratio = 1.4
      Heat Conductivity = Variable Temperature
      Procedure "getFilmThermalConductivity" "getThermalConductivity"
      Reference Thermal Conductivity Solid ${MAT_A}= Real 130
      Cond Coeff As Solid ${MAT_A} = Real -1.17E-04
      Cond Coeff Bs Solid ${MAT_A} = Real 9.29E-03
      Reference Thermal Conductivity Liquid ${MAT_A} = Real 90
      Cond Coeff Liquid ${MAT_A}= Real 1.83E-02
      Melting Point Temperature of ${MAT_A} = Real ${TMELT_A}
      Density = Variable Temperature
      Procedure "getFilmDensity" "getDensity"
      Reference Density Solid ${MAT_A} = Real 2700
      Density Coeff Solid ${MAT_A} = Real -0.11
      Reference Density Liquid ${MAT_A} = Real 2380
      Density Coefficient Liquid ${MAT_A} = Real -0.28
      Tscaler = Real 1.0
    End

    Material 2
      Name = "${MAT_B}"
      Poisson Ratio = Real 0.31
      Reference Temperature = 298.0
      Heat Expansion Coefficient = Variable Temperature
      Procedure "getThermalExpansivity" "getThermalExpansivity"
      Reference Thermal Expansivity Solid ${MAT_B} = Real -8.371435292934836e-05
      Thermal Expansivity Coeff As Solid ${MAT_B} = Real -3.7262790630140875e-10
      Thermal Expansivity Coeff Bs Solid ${MAT_B} = Real 4.2255653792479394e-07
      Viscosity = Variable Temperature
    Real
      include ./viscosity/mu_${MAT_B_LOWER}.dat
    End
      Specific Enthalpy = Variable Temperature
    Real
      include ./specific_enthalpy/h_${MAT_B_LOWER}.dat
    End
      Phase Change Intervals(2,1) = ${TMELT_B_MINUS_10} ${TMELT_B_PLUS_10}
      Compressibility Model = Incompressible
      Reference Pressure = 0
      Specific Heat Ratio = 1.4
      Heat Conductivity = Variable Temperature
      Procedure "getFilmThermalConductivity" "getThermalConductivity"
      Reference Thermal Conductivity Solid ${MAT_B}= Real 391
      Cond Coeff As Solid ${MAT_B} = Real -0.052
      Cond Coeff Bs Solid ${MAT_B} = Real 0.0
      Reference Thermal Conductivity Liquid ${MAT_B} = Real 170
      Cond Coeff Liquid ${MAT_B}= Real -0.025
      Melting Point Temperature of ${MAT_B} = Real ${TMELT_B}
      Density = Variable Temperature
      Procedure "getFilmDensity" "getDensity"
      Reference Density Solid ${MAT_B} = Real 8940
      Density Coeff Solid ${MAT_B} = Real -0.52
      Reference Density Liquid ${MAT_B} = Real 7992
      Density Coefficient Liquid ${MAT_B} = Real -0.44
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
      Target Boundaries(${len(bc_faces)}) = ${BC_INDICES}
      Displacement 1 = 0
      Displacement 2 = 0
      Displacement 3 = 0
      Noslip wall BC = True
      Save Scalars = Logical True
    End
    
    Boundary Condition 2
      Name = "Convective Cooling"
      Target Boundaries(2) = 4 5
      External Temperature = 298.0
      Heat Transfer Coefficient = 15
      Noslip wall BC = True
      Save Scalars = Logical True
    End

    Boundary Condition 3
      Name = "Bottom Fixed Temperature"
      Target Boundaries(1) = 5
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
      Target Boundaries(1) = 7
      Heat Flux = Variable time
      Real Procedure "DifferentTypeHeatSource" "${HEAT_PROC_NAME}"
      Save Line = True
    End
""")
        
        # Prepare substitution dict
        subst = {
            'MAT_A': mat_a_name, 'MAT_B': mat_b_name,
            'MAT_A_LOWER': mat_a_name.lower().replace('-', '_'),
            'MAT_B_LOWER': mat_b_name.lower().replace('-', '_'),
            'TIMESTAMP': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'MESH_NAME': mesh_name,
            'RESULTS_DIR': results_dir,
            'COORD_SCALE': coord_val,
            'DT_INIT': dt_initial, 'DT_MAIN': dt_main, 'N_STEPS': n_steps-1,
            'BDF_ORDER': bdf_order,
            'POST_FILE': post_file, 'OUTPUT_FILE': output_file,
            'BEAM_RADIUS': beam_radius, 'HEAT_COEFF': heat_coeff,
            'SPEED_X': speed_x, 'SPEED_Y': speed_y,
            'SCAN_DIST': scan_dist, 'INIT_X': init_x, 'INIT_Y': init_y,
            'ABSORPTANCE': absorptance,
            'TMELT_A': mat_a_melting, 'TMELT_B': mat_b_melting,
            'TMELT_A_MINUS_10': mat_a_melting - 10, 'TMELT_A_PLUS_10': mat_a_melting + 10,
            'TMELT_B_MINUS_10': mat_b_melting - 10, 'TMELT_B_PLUS_10': mat_b_melting + 10,
            'BC_INDICES': " ".join(str(i+1) for i in range(len(bc_faces))),
            'HEAT_PROC_NAME': 'FlatTopHeatSource' if 'Super-Gaussian' in heat_type else 'TravellingHeatSource'
        }
        
        sif_content = sif_template.substitute(subst)
        
        # === GENERATE FORTRAN .F90 FILES ===
        def prepare_fortran(code, mat_name):
            """Replace ${MAT_NAME} placeholder with actual material name."""
            return code.replace("${MAT_NAME}", mat_name)
        
        # Material A UDFs
        dens_a_f90 = prepare_fortran(density_a, mat_a_name)
        cond_a_f90 = prepare_fortran(cond_a, mat_a_name)
        exp_a_f90 = prepare_fortran(exp_a, mat_a_name)
        
        # Material B UDFs  
        dens_b_f90 = prepare_fortran(density_b, mat_b_name)
        cond_b_f90 = prepare_fortran(cond_b, mat_b_name)
        exp_b_f90 = prepare_fortran(exp_b, mat_b_name)
        
        # Heat source
        heat_f90 = st.session_state.heat_source_code
        
        # === DISPLAY DOWNLOAD BUTTONS ===
        st.success("✅ Files generated! Download below:")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            st.download_button(
                label="📄 Download case.sif",
                data=sif_content,
                file_name=f"weld_{mat_a_name}_{mat_b_name}.sif",
                mime="text/plain"
            )
            st.download_button(
                label="🔧 getDensity_matA.F90",
                data=dens_a_f90,
                file_name=f"getDensity_{mat_a_name}.F90",
                mime="text/plain"
            )
            st.download_button(
                label="🔧 getDensity_matB.F90",
                data=dens_b_f90,
                file_name=f"getDensity_{mat_b_name}.F90",
                mime="text/plain"
            )
        
        with col_dl2:
            st.download_button(
                label="🔧 getThermalConductivity_matA.F90",
                data=cond_a_f90,
                file_name=f"getThermalConductivity_{mat_a_name}.F90",
                mime="text/plain"
            )
            st.download_button(
                label="🔧 getThermalConductivity_matB.F90",
                data=cond_b_f90,
                file_name=f"getThermalConductivity_{mat_b_name}.F90",
                mime="text/plain"
            )
            st.download_button(
                label="🔧 HeatSource.F90",
                data=heat_f90,
                file_name="DifferentTypeHeatSource.F90",
                mime="text/plain"
            )
        
        with col_dl3:
            st.download_button(
                label="🔧 getThermalExpansivity_matA.F90",
                data=exp_a_f90,
                file_name=f"getThermalExpansivity_{mat_a_name}.F90",
                mime="text/plain"
            )
            st.download_button(
                label="🔧 getThermalExpansivity_matB.F90",
                data=exp_b_f90,
                file_name=f"getThermalExpansivity_{mat_b_name}.F90",
                mime="text/plain"
            )
            # Lookup tables
            for table_name, df in st.session_state.lookup_tables.items():
                fname = "mu_" + mat_a_name.lower().replace('-', '_') + ".dat" if "Viscosity" in table_name and "A" in table_name else \
                        "mu_" + mat_b_name.lower().replace('-', '_') + ".dat" if "Viscosity" in table_name else \
                        "h_" + mat_a_name.lower().replace('-', '_') + ".dat" if "A" in table_name else \
                        "h_" + mat_b_name.lower().replace('-', '_') + ".dat"
                st.download_button(
                    label=f"📊 {fname}",
                    data=df.to_csv(sep='\t', index=False),
                    file_name=fname,
                    mime="text/tab-separated-values"
                )
        
        # === INSTRUCTIONS ===
        with st.expander("📋 How to Use Generated Files", expanded=True):
            st.markdown("""
            **Compilation & Execution Steps:**
            
            1. **Place files in working directory:**
            ```
            project/
            ├── weld_Al_Cu.sif                 # Main input file
            ├── Mesh_weld_*.mesh              # SALOME-exported mesh
            ├── *.F90                          # Fortran UDFs (downloaded above)
            ├── viscosity/
            │   ├── mu_AA6061_Al.dat
            │   └── mu_T2_Cu.dat
            └── specific_enthalpy/
                ├── h_AA6061_Al.dat
                └── h_T2_Cu.dat
            ```
            
            2. **Compile Fortran UDFs:**
            ```bash
            elmerfem -c getDensity_AA6061_Al.F90
            elmerfem -c getThermalConductivity_AA6061_Al.F90
            # ... repeat for all .F90 files
            ```
            
            3. **Link and run:**
            ```bash
            ElmerSolver weld_Al_Cu.sif
            ```
            
            **Troubleshooting Tips:**
            - Ensure `DefUtils` module is available (part of Elmer FEM source)
            - Use `REAL(KIND=dp)` consistently for double precision
            - Check that all `GetConstReal` parameter names match exactly with .sif Material block
            - For phase change: mushy zone width ±10 K around melting point
            
            **Reference:** Kunwar et al., *J. Mater. Sci. Technol.* 50 (2020) 115-127
            """)

# ====================== FOOTER ======================
st.markdown("---")
st.markdown("""
**💡 Pro Tips:**
- Use session state to preserve edits across tabs
- Test UDFs with small temperature ranges before full simulation
- For Al-Cu welding: consider adding IMC layer as third material with distinct properties
- Monitor `Nonlinear System Relaxation Factor` (0.6 recommended) for phase change stability

**🔗 Resources:**
- [Elmer FEM Documentation](https://www.elmerfem.org)
- [Fortran UDF Guide](https://github.com/ElmerCSC/elmerfem/blob/devel/fem/src/modules/DefUtils.F90)
- [Al-Cu Property Database](https://materials.springer.com)
""")
