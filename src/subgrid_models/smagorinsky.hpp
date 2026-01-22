#ifndef SMAGO_SG_HPP_
#define SMAGO_SG_HPP_

#include <string>
#include "../athena.hpp"
#include "../athena_aliases.hpp"
#include "../utils/linear_algebra.hpp"


//Forward declarations
class ParameterInput;
class GRDynamical;
class MeshBlock;
class Z4c;
class Hydro;
class PassiveScalars;
class EquationOfState;

// Class SmagoSG : Implements Smagorinsky 

class SmagoSG {
 private:
  
  //Private members
  std::string visc_type;
  Real nu_turb;
  MeshBlock *pmb; 
  Z4c *pz4c;

  //Matter and Lorentz factor
  Hydro * ph;
  PassiveScalars *ps;
  EquationOfState * peos;

  AT_N_VS2 Gamma_ddd;     
  AT_N_VS2 Gamma_udd;
  AT_N_VS2 dg_ddd;
  AT_N_sca detg;
  AT_N_sym g_uu;
  AT_N_sym adm_gamma_dd;
  AT_N_vec w_util_d;
  AT_N_T2 Dv_dd;

  Real lmix;
  Real kiuchi_stretch, kiuchi_ampl;
  Real kiuchi2_lrho0, kiuchi2_lrho1;
  Real kiuchi2_a, kiuchi2_b, kiuchi2_c;

  int nx1, nx2, nx3;
  int il, iu, jl, ju, kl, ku;

  
  // Private methods
  void Christoffel_calc(int k, int j);

 public:
  // Public members
  AT_N_sym TurbStressTensor_dd;
  
  Real Get_TurbStressTensor_uu(int a, int b, int k, int j, int i);
  Real Get_TurbStressTensor_du(int m, int a, int k, int j, int i);
  Real Get_d_TurbStressTensor_d(int a, int k, int j, int i);

  // Constructor
  SmagoSG(MeshBlock *pmb_in,
          ParameterInput *pin);

  Real compute_turb_visc(Real rho, Real csound);

  void TurTensorCalculator();
};

#endif // SMAGO_SG_HPP_
