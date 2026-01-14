!-----------------------------------------------------
! User-defined function for ELMER: Specific enthalpy of a pure material
! 
! Molar enthalpy model (J/mol):
!   H_molar(T) = A1*T + A2*(T - Tm) + DeltaHf * sigmoid(T - Tm, k) + H298
!   where sigmoid(x, k) = 1 / (1 + exp(-k*x))
!
! Specific enthalpy (J/kg):
!   H_specific = H_molar / M_molar
!   (M_molar = molar mass in kg/mol)
!
! Parameters required in material definition:
!   - Enthalpy A1             [J/(mol·K)]
!   - Enthalpy A2             [J/(mol·K)]
!   - Melting Point Temperature [K]
!   - Latent Heat of Fusion   [J/mol]
!   - Enthalpy Sigmoid Sharpness [1/K]
!   - Reference Enthalpy at 298K [J/mol]
!   - Molar Mass              [kg/mol]  (NOT g/mol!)
!
! Written By: Anil Kunwar
! Date: 2026-01-14
!-----------------------------------------------------
FUNCTION getEnthalpy(Model, n, Temperature) RESULT(SpecificEnthalpy)
  USE DefUtils
  IMPLICIT NONE
  
  ! Function interface
  TYPE(Model_t) :: Model
  INTEGER :: n  ! Node index (unused but required by Elmer API)
  REAL(KIND=dp) :: Temperature, SpecificEnthalpy
  
  ! Local variables
  REAL(KIND=dp) :: A1, A2, Tm, DeltaHf, k_param, H298, M_molar
  REAL(KIND=dp) :: x, exp_arg, sigmoid_val, H_molar
  LOGICAL :: GotIt
  TYPE(ValueList_t), POINTER :: MaterialProps
  
  ! Get material properties pointer
  MaterialProps => GetMaterial()
  IF (.NOT. ASSOCIATED(MaterialProps)) &
    CALL Fatal("getEnthalpy", "No material properties found!")
  
  ! Read parameters with explicit error handling
  A1 = GetConstReal(MaterialProps, "Enthalpy A1", GotIt)
  IF (.NOT. GotIt) CALL Fatal("getEnthalpy", "Missing 'Enthalpy A1' in material definition")
  
  A2 = GetConstReal(MaterialProps, "Enthalpy A2", GotIt)
  IF (.NOT. GotIt) CALL Fatal("getEnthalpy", "Missing 'Enthalpy A2' in material definition")
  
  Tm = GetConstReal(MaterialProps, "Melting Point Temperature", GotIt)
  IF (.NOT. GotIt) CALL Fatal("getEnthalpy", "Missing 'Melting Point Temperature' in material definition")
  
  DeltaHf = GetConstReal(MaterialProps, "Latent Heat of Fusion", GotIt)
  IF (.NOT. GotIt) CALL Fatal("getEnthalpy", "Missing 'Latent Heat of Fusion' in material definition")
  
  k_param = GetConstReal(MaterialProps, "Enthalpy Sigmoid Sharpness", GotIt)
  IF (.NOT. GotIt) CALL Fatal("getEnthalpy", "Missing 'Enthalpy Sigmoid Sharpness' in material definition")
  
  H298 = GetConstReal(MaterialProps, "Reference Enthalpy at 298K", GotIt)
  IF (.NOT. GotIt) CALL Fatal("getEnthalpy", "Missing 'Reference Enthalpy at 298K' in material definition")
  
  M_molar = GetConstReal(MaterialProps, "Molar Mass", GotIt)
  IF (.NOT. GotIt) CALL Fatal("getEnthalpy", "Missing 'Molar Mass' in material definition")
  IF (M_molar <= 0.0_dp) CALL Fatal("getEnthalpy", "Molar mass must be > 0 kg/mol")
  
  ! Compute sigmoid argument with overflow protection
  x = Temperature - Tm
  exp_arg = -k_param * x
  
  ! Safeguard against exp() overflow (|arg| > 50 is numerically 0 or 1)
  IF (exp_arg > 50.0_dp) THEN
    sigmoid_val = 0.0_dp  ! exp(exp_arg) -> inf, so 1/(1+inf) = 0
  ELSE IF (exp_arg < -50.0_dp) THEN
    sigmoid_val = 1.0_dp  ! exp(exp_arg) -> 0, so 1/(1+0) = 1
  ELSE
    sigmoid_val = 1.0_dp / (1.0_dp + EXP(exp_arg))
  END IF
  
  ! Calculate molar enthalpy (J/mol)
  H_molar = A1 * Temperature + &
            A2 * (Temperature - Tm) + &
            DeltaHf * sigmoid_val + &
            H298
  
  ! Convert to specific enthalpy (J/kg)
  SpecificEnthalpy = H_molar / M_molar
  
END FUNCTION getEnthalpy
