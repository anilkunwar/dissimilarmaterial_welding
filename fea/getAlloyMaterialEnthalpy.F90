!-----------------------------------------------------
! User-defined function for ELMER: Specific enthalpy of arbitrary multicomponent alloy
! 
! Features:
!   - Complete database of all 118 elements with IUPAC 2021 atomic masses
!   - Dynamic detection of elements present in the alloy
!   - Flexible property naming: "Mole Fraction <Symbol>" (e.g., "Mole Fraction Ti")
!   - Optional custom atomic masses: "Atomic Mass <Symbol>" (overrides database)
!   - Composition validation with detailed error reporting
!   - Numerically stable sigmoid implementation
!
! Usage:
!   1. Define mole fractions for elements in your alloy:
!        Mole Fraction Fe = 0.7
!        Mole Fraction Cr = 0.18
!        Mole Fraction Ni = 0.1
!        Mole Fraction C  = 0.02
!   2. (Optional) Override atomic masses:
!        Atomic Mass Fe = 55.845
!   3. Define enthalpy coefficients as before
!
! Written By: Anil Kunwar
!-----------------------------------------------------
FUNCTION getEnthalpyFlexible(Model, n, Temperature) RESULT(SpecificEnthalpy)
  USE DefUtils
  IMPLICIT NONE
  
  ! Function interface
  TYPE(Model_t) :: Model
  INTEGER :: n  ! Node index (unused but required by Elmer API)
  REAL(KIND=dp) :: Temperature, SpecificEnthalpy
  
  ! Full periodic table database (IUPAC 2021 standard atomic weights)
  CHARACTER(LEN=2), PARAMETER, DIMENSION(118) :: element_symbols = [ &
    'H ', 'He', 'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', 'Ne', &
    'Na', 'Mg', 'Al', 'Si', 'P ', 'S ', 'Cl', 'Ar', 'K ', 'Ca', &
    'Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', &
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y ', 'Zr', &
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', &
    'Sb', 'Te', 'I ', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', &
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', &
    'Lu', 'Hf', 'Ta', 'W ', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', &
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', &
    'Pa', 'U ', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', &
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', &
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og' ]
    
  REAL(KIND=dp), PARAMETER, DIMENSION(118) :: standard_atomic_masses = [ &
    1.008_dp, 4.0026_dp, 6.94_dp, 9.0122_dp, 10.81_dp, 12.011_dp, 14.007_dp, 15.999_dp, &
    18.998_dp, 20.180_dp, 22.990_dp, 24.305_dp, 26.982_dp, 28.085_dp, 30.974_dp, 32.06_dp, &
    35.45_dp, 39.948_dp, 39.098_dp, 40.078_dp, 44.956_dp, 47.867_dp, 50.942_dp, 51.996_dp, &
    54.938_dp, 55.845_dp, 58.933_dp, 58.693_dp, 63.546_dp, 65.38_dp, 69.723_dp, 72.630_dp, &
    74.922_dp, 78.971_dp, 79.904_dp, 83.798_dp, 85.468_dp, 87.62_dp, 88.906_dp, 91.224_dp, &
    92.906_dp, 95.95_dp, 98.0_dp, 101.07_dp, 102.91_dp, 106.42_dp, 107.87_dp, 112.41_dp, &
    114.82_dp, 118.71_dp, 121.76_dp, 127.60_dp, 126.90_dp, 131.29_dp, 132.91_dp, 137.33_dp, &
    138.91_dp, 140.12_dp, 140.91_dp, 144.24_dp, 145.0_dp, 150.36_dp, 151.96_dp, 157.25_dp, &
    158.93_dp, 162.50_dp, 164.93_dp, 167.26_dp, 168.93_dp, 173.04_dp, 174.97_dp, 178.49_dp, &
    180.95_dp, 183.84_dp, 186.21_dp, 190.23_dp, 192.22_dp, 195.08_dp, 196.97_dp, 200.59_dp, &
    204.38_dp, 207.2_dp, 208.98_dp, 209.0_dp, 210.0_dp, 222.0_dp, 223.0_dp, 226.0_dp, &
    227.0_dp, 232.04_dp, 231.04_dp, 238.03_dp, 237.0_dp, 244.0_dp, 243.0_dp, 247.0_dp, &
    247.0_dp, 251.0_dp, 252.0_dp, 257.0_dp, 258.0_dp, 259.0_dp, 262.0_dp, 267.0_dp, &
    268.0_dp, 271.0_dp, 272.0_dp, 285.0_dp, 286.0_dp, 289.0_dp, 290.0_dp, 293.0_dp, &
    294.0_dp, 294.0_dp ]
  
  ! Local variables
  INTEGER, PARAMETER :: MAX_ELEMENTS = 20  ! Practical limit for alloys
  CHARACTER(LEN=MAX_NAME_LEN) :: prop_name, elem_name
  REAL(KIND=dp) :: mole_fractions(MAX_ELEMENTS), atomic_masses(MAX_ELEMENTS)
  CHARACTER(LEN=2) :: elem_symbols(MAX_ELEMENTS)
  INTEGER :: i, j, n_elements, n_found
  REAL(KIND=dp) :: total_x, M_alloy_g_mol, M_alloy
  REAL(KIND=dp) :: A1, A2, Tm, DeltaHf, k_param, H298
  REAL(KIND=dp) :: x, exp_arg, sigmoid_val, H_molar
  LOGICAL :: GotIt, found
  TYPE(ValueList_t), POINTER :: MaterialProps
  
  ! Get material properties pointer
  MaterialProps => GetMaterial()
  IF (.NOT. ASSOCIATED(MaterialProps)) &
    CALL Fatal("getEnthalpyFlexible", "No material properties found!")
  
  ! Step 1: Detect which elements are present in the material definition
  n_elements = 0
  DO i = 1, 118
    ! Check for "Mole Fraction <Symbol>" property
    WRITE(prop_name, '("Mole Fraction ",A2)') TRIM(element_symbols(i))
    CALL GetConstReal(MaterialProps, TRIM(prop_name), GotIt)
    
    IF (GotIt) THEN
      n_elements = n_elements + 1
      IF (n_elements > MAX_ELEMENTS) &
        CALL Fatal("getEnthalpyFlexible", "Too many elements defined. Maximum is "//TRIM(ToString(MAX_ELEMENTS)))
      
      elem_symbols(n_elements) = element_symbols(i)
    END IF
  END DO
  
  ! Validate at least one element is defined
  IF (n_elements == 0) THEN
    CALL Fatal("getEnthalpyFlexible", "No elements found. Define properties like 'Mole Fraction Fe = 0.7'")
  END IF
  
  ! Step 2: Read mole fractions and atomic masses for detected elements
  total_x = 0.0_dp
  DO i = 1, n_elements
    ! Read mole fraction
    WRITE(prop_name, '("Mole Fraction ",A2)') TRIM(elem_symbols(i))
    mole_fractions(i) = GetConstReal(MaterialProps, TRIM(prop_name), GotIt)
    IF (.NOT. GotIt) &
      CALL Fatal("getEnthalpyFlexible", "Missing '"//TRIM(prop_name)//"' after detection")
    
    ! Read atomic mass (custom or from database)
    WRITE(prop_name, '("Atomic Mass ",A2)') TRIM(elem_symbols(i))
    CALL GetConstReal(MaterialProps, TRIM(prop_name), GotIt)
    
    IF (GotIt) THEN
      ! User provided custom atomic mass
      atomic_masses(i) = GetConstReal(MaterialProps, TRIM(prop_name), GotIt)
    ELSE
      ! Use database value - find matching symbol
      found = .FALSE.
      DO j = 1, 118
        IF (TRIM(element_symbols(j)) == TRIM(elem_symbols(i))) THEN
          atomic_masses(i) = standard_atomic_masses(j)
          found = .TRUE.
          EXIT
        END IF
      END DO
      IF (.NOT. found) &
        CALL Fatal("getEnthalpyFlexible", "Element symbol '"//TRIM(elem_symbols(i))//"' not found in database")
    END IF
    
    ! Validate values
    IF (mole_fractions(i) < 0.0_dp .OR. mole_fractions(i) > 1.0_dp) &
      CALL Fatal("getEnthalpyFlexible", "Invalid mole fraction for "//TRIM(elem_symbols(i))// &
                 ": "//TRIM(ToString(mole_fractions(i))))
    IF (atomic_masses(i) <= 0.0_dp) &
      CALL Fatal("getEnthalpyFlexible", "Invalid atomic mass for "//TRIM(elem_symbols(i))// &
                 ": "//TRIM(ToString(atomic_masses(i))))
    
    total_x = total_x + mole_fractions(i)
  END DO
  
  ! Step 3: Validate composition
  IF (ABS(total_x - 1.0_dp) > 1.0e-4_dp) THEN
    CALL Info("getEnthalpyFlexible", "Sum of mole fractions = "//TRIM(ToString(total_x)))
    CALL Fatal("getEnthalpyFlexible", &
      "Sum of mole fractions deviates from 1.0 by >0.01%. Check composition definition.")
  END IF
  
  ! Step 4: Calculate alloy molar mass (g/mol -> kg/mol)
  M_alloy_g_mol = 0.0_dp
  DO i = 1, n_elements
    M_alloy_g_mol = M_alloy_g_mol + mole_fractions(i) * atomic_masses(i)
  END DO
  M_alloy = M_alloy_g_mol * 1.0e-3_dp  ! Convert to kg/mol
  
  IF (M_alloy <= 0.0_dp) &
    CALL Fatal("getEnthalpyFlexible", "Invalid calculated molar mass: "//TRIM(ToString(M_alloy)))
  
  ! Step 5: Read enthalpy coefficients (same as before)
  A1 = GetConstReal(MaterialProps, "Enthalpy A1", GotIt)
  IF (.NOT. GotIt) CALL Fatal("getEnthalpyFlexible", "Missing 'Enthalpy A1' in material definition")
  
  A2 = GetConstReal(MaterialProps, "Enthalpy A2", GotIt)
  IF (.NOT. GotIt) CALL Fatal("getEnthalpyFlexible", "Missing 'Enthalpy A2' in material definition")
  
  Tm = GetConstReal(MaterialProps, "Melting Point Temperature", GotIt)
  IF (.NOT. GotIt) CALL Fatal("getEnthalpyFlexible", "Missing 'Melting Point Temperature' in material definition")
  
  DeltaHf = GetConstReal(MaterialProps, "Latent Heat of Fusion", GotIt)
  IF (.NOT. GotIt) CALL Fatal("getEnthalpyFlexible", "Missing 'Latent Heat of Fusion' in material definition")
  
  k_param = GetConstReal(MaterialProps, "Enthalpy Sigmoid Sharpness", GotIt)
  IF (.NOT. GotIt) CALL Fatal("getEnthalpyFlexible", "Missing 'Enthalpy Sigmoid Sharpness' in material definition")
  
  H298 = GetConstReal(MaterialProps, "Reference Enthalpy at 298K", GotIt)
  IF (.NOT. GotIt) CALL Fatal("getEnthalpyFlexible", "Missing 'Reference Enthalpy at 298K' in material definition")
  
  ! Step 6: Compute enthalpy (same stable implementation)
  x = Temperature - Tm
  exp_arg = -k_param * x
  
  ! Safeguard against exp() overflow
  IF (exp_arg > 50.0_dp) THEN
    sigmoid_val = 0.0_dp
  ELSE IF (exp_arg < -50.0_dp) THEN
    sigmoid_val = 1.0_dp
  ELSE
    sigmoid_val = 1.0_dp / (1.0_dp + EXP(exp_arg))
  END IF
  
  ! Calculate molar enthalpy (J/mol)
  H_molar = A1 * Temperature + &
            A2 * (Temperature - Tm) + &
            DeltaHf * sigmoid_val + &
            H298
  
  ! Convert to specific enthalpy (J/kg)
  SpecificEnthalpy = H_molar / M_alloy
  
  ! Optional: Debug output (remove in production)
  !CALL Info("getEnthalpyFlexible", "Alloy molar mass = "//TRIM(ToString(M_alloy_g_mol))//" g/mol")
  !CALL Info("getEnthalpyFlexible", "Specific enthalpy at "//TRIM(ToString(Temperature))//" K = "// &
  !          TRIM(ToString(SpecificEnthalpy))//" J/kg")
  
END FUNCTION getEnthalpyFlexible
