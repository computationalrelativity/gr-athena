#include "weak_opacity.hpp"
#include "fermi.hpp"
#include "error_codes.hpp"
#include <cmath>

using namespace std;
using namespace M1::Opacities::WeakRates::WeakRatesFermi;
using namespace M1::Opacities::WeakRates::WeakRates_Opacity;

int WeakOpacityMod::NeutrinoAbsorptionOpacityImpl(Real rho, Real temp, Real ye, Real& abs_n_nue, Real& abs_n_nua, Real& abs_n_nux, Real& abs_e_nue, Real& abs_e_nua, Real& abs_e_nux) {
  int iout = 0;

  Real rho0 = rho;
  Real temp0 = temp;
  Real ye0   = ye;

  if ((rho0<rho_min) || (temp0<temp_min)) {
    abs_n_nue = 0.0;
    abs_n_nua = 0.0;
    abs_n_nux = 0.0;
    abs_e_nue = 0.0;
    abs_e_nua = 0.0;
    abs_e_nux = 0.0;
    return iout;
  } //end if

  // boundsErr = enforceTableBounds(rho_cgs,temp0,ye0)
  bool boundsErr = EoS->ApplyTableLimits(rho0, temp0, ye0);

  if (boundsErr) {
    iout = WR_OPAC_BNDS_ERR;
    return iout;
  } // end if

  iout = Absorption_cgs(rho, temp, ye,
                        abs_n_nue, abs_n_nua, abs_n_nux,
                        abs_e_nue, abs_e_nua, abs_e_nux);

  return iout;
}

int WeakOpacityMod::NeutrinoScatteringOpacityImpl(Real rho, Real temp, Real ye, Real& sct_n_nue, Real& sct_n_nua, Real& sct_n_nux, Real& sct_e_nue, Real& sct_e_nua, Real& sct_e_nux) {
  int iout = 0;

  Real rho0 = rho;
  Real temp0 = temp;
  Real ye0   = ye;

  if ((rho0<rho_min) || (temp0<temp_min)) {
    sct_n_nue = 0.0;
    sct_n_nua = 0.0;
    sct_n_nux = 0.0;
    sct_e_nue = 0.0;
    sct_e_nua = 0.0;
    sct_e_nux = 0.0;
    return iout;
  } //end if

  // boundsErr = enforceTableBounds(rho_cgs,temp0,ye0)
  bool boundsErr = EoS->ApplyTableLimits(rho0, temp0, ye0);

  if (boundsErr) {
    iout = WR_OPAC_BNDS_ERR;
    return iout;
  } // end if

  iout = Scattering_cgs(rho, temp, ye,
                        sct_n_nue, sct_n_nua, sct_n_nux,
                        sct_e_nue, sct_e_nua, sct_e_nux);

  return iout;
}

int WeakOpacityMod::Absorption_cgs(Real rho, Real temp, Real ye, Real& abs_n_nue, Real& abs_n_nua, Real& abs_n_nux, Real& abs_e_nue, Real& abs_e_nua, Real& abs_e_nux) {
  int iout = 0;

// Broken out into separate function
/*
#define WEAK_RATES_ITS_ME
#include "inc/weak_rates_guts.inc"
#undef WEAK_RATES_ITS_ME
*/

Real eta_nue, eta_nua, eta_nux, eta_e, eta_np, eta_pn;
EoS->GetEtas(rho, temp, ye, eta_nue, eta_nua, eta_nux, eta_e, eta_np, eta_pn);

//----------------------------------------------------------------------
// ABSOPTION
//----------------------------------------------------------------------

  Real abs_zeta = (1.0 + 3.0 * (alpha*alpha)) * 0.25 * sigma_0/(me_erg*me_erg);

  Real block_factor = 1.0 + exp(eta_e - fermi5O4(eta_nue));
  Real zeta_nue_abs_n = eta_np * abs_zeta / block_factor;
  Real zeta_nua_abs_n = 0.0; // no absorption of e-anti-nu on neutrons
  Real zeta_nux_abs_n = 0.0; // no absorption of heavy-nu on neutrons

  block_factor = 1.0 + exp(-eta_e - fermi5O4(eta_nua));
  Real zeta_nue_abs_p = 0.0; // no absorption of e-nu on protons
  Real zeta_nua_abs_p = eta_pn * abs_zeta/block_factor;
  Real zeta_nux_abs_p = 0.0; // no absorption of heavy-nu on neutrons

  Real zeta_nue_abs_h = 0.0; // no absorption on nuclei
  Real zeta_nua_abs_h = 0.0; // no absorption on nuclei
  Real zeta_nux_abs_h = 0.0; // no absorption on nuclei

  Real zeta_nue = zeta_nue_abs_n + zeta_nue_abs_p + zeta_nue_abs_h;
  Real zeta_nua = zeta_nua_abs_n + zeta_nua_abs_p + zeta_nua_abs_h;
  Real zeta_nux = zeta_nux_abs_n + zeta_nux_abs_p + zeta_nux_abs_h;

  Real temp_erg_sq = (temp * temp * mev_to_erg * mev_to_erg);
  abs_n_nue = zeta_nue * temp_erg_sq * fermi4O2(eta_nue);
  abs_n_nua = zeta_nua * temp_erg_sq * fermi4O2(eta_nua);
  abs_n_nux = zeta_nux * temp_erg_sq * fermi4O2(eta_nux);

  abs_e_nue = zeta_nue * temp_erg_sq * fermi5O3(eta_nue);
  abs_e_nua = zeta_nua * temp_erg_sq * fermi5O3(eta_nua);
  abs_e_nux = zeta_nux * temp_erg_sq * fermi5O3(eta_nux);

  if (!std::isfinite(abs_n_nue) ||
      !std::isfinite(abs_n_nua) ||
      !std::isfinite(abs_n_nux) ||
      !std::isfinite(abs_e_nue) ||
      !std::isfinite(abs_e_nua) ||
      !std::isfinite(abs_e_nux))
  {
    iout = WR_OPAC_ABS_NONFINITE;
  }

  return iout;
} // END FUNCTION Absorption_cgs

int WeakOpacityMod::Scattering_cgs(Real rho, Real temp, Real ye, Real& sct_n_nue, Real& sct_n_nua, Real& sct_n_nux, Real& sct_e_nue, Real& sct_e_nua, Real& sct_e_nux) {
  int iout = 0;

// Broken out into separate function
/*
#define WEAK_RATES_ITS_ME
#include "inc/weak_rates_guts.inc"
#undef WEAK_RATES_ITS_ME
*/

Real eta_nue, eta_nua, eta_nux, eta_e, eta_np, eta_pn;
EoS->GetEtas(rho, temp, ye, eta_nue, eta_nua, eta_nux, eta_e, eta_np, eta_pn);

  /* TODO? Our EoS does not currently implement nuclei fractions, and we
           don't intend to use this opacity lib anyway, so I'm not 
           bothering to implement them, and just setting the heavy 
           nuclei to zero and the proton/neutron fractions with ye.
  */
  
  Real xp = ye;
  Real xn = 1-ye;
  Real abar = 1.0;
  Real zbar = 1.0;
  Real xh = 0.0;

// ---------------------------------------------------------------------
// SCATTERING
// ---------------------------------------------------------------------

  // Eqs (A17) with different species plus the A21 which multiplies
  // the mass fraction
  // The formula reported in Rosswog paper is wrong. A term is missing
  // (1+3g_A^2)/4 that is roughly ~1.4.
  // Corrected version from Ruffert et al. paper (TODO update using Barrows 2006)

  Real nb = rho/atomic_mass;

  Real scttr_cff_p = nb * ((1.0 + 5.0 * (alpha*alpha)) / 24.0) * sigma_0/(me_erg*me_erg);
  Real scttr_cff_n = nb * ((4.0 * (Cv-1.0)*(Cv-1.0) + 5.0 * (alpha*alpha)) / 24.0) * sigma_0/(me_erg*me_erg);

  // Neutrino nucleon scattering coefficients
  // electron neutrinos
  Real zeta_nue_sct_n = xn*scttr_cff_n;
  Real zeta_nue_sct_p = xp*scttr_cff_p;
  // electron antineutrinos
  Real zeta_nua_sct_n = xn*scttr_cff_n;
  Real zeta_nua_sct_p = xp*scttr_cff_p;
  // tau and mu neutrinos
  Real zeta_nux_sct_n = xn*scttr_cff_n;
  Real zeta_nux_sct_p = xp*scttr_cff_p;

  // Coherent neutrinos nucleus scattering (Shapiro & Teukolsky 1983,
  // sin^2Theta_w has been approx by 0.25)
  // XXX it has only 1 factor of A because zeta multiples the number fraction,
  // not the mass fractions
  Real scttr_cff = nb * 0.0625 * sigma_0 / (me_erg*me_erg) * abar * (1.0 - zbar / abar) * (1.0 - zbar / abar);

  // On heavy nuclei
  Real zeta_nue_sct_h = xh*scttr_cff;
  Real zeta_nua_sct_h = xh*scttr_cff;
  Real zeta_nux_sct_h = xh*scttr_cff;


  Real zeta_nue = zeta_nue_sct_n + zeta_nue_sct_p + zeta_nue_sct_h;
  Real zeta_nua = zeta_nua_sct_n + zeta_nua_sct_p + zeta_nua_sct_h;
  Real zeta_nux = zeta_nux_sct_n + zeta_nux_sct_p + zeta_nux_sct_h;

  Real temp_erg_sq = (temp * temp * mev_to_erg * mev_to_erg);
  sct_n_nue = zeta_nue * temp_erg_sq * fermi4O2(eta_nue);
  sct_n_nua = zeta_nua * temp_erg_sq * fermi4O2(eta_nua);
  sct_n_nux = zeta_nux * temp_erg_sq * fermi4O2(eta_nux);

  sct_e_nue = zeta_nue * temp_erg_sq * fermi5O3(eta_nue);
  sct_e_nua = zeta_nua * temp_erg_sq * fermi5O3(eta_nua);
  sct_e_nux = zeta_nux * temp_erg_sq * fermi5O3(eta_nux);

  if (!std::isfinite(sct_n_nue) ||
      !std::isfinite(sct_n_nua) ||
      !std::isfinite(sct_n_nux) ||
      !std::isfinite(sct_e_nue) ||
      !std::isfinite(sct_e_nua) ||
      !std::isfinite(sct_e_nux))
  {
    iout = WR_OPAC_SCA_NONFINITE;
  }

  return iout;
} // END FUNCTION Scattering_cgs

/*
### Old opacity Impl functions ###
*/

int WeakOpacityMod::NeutrinoOpacityImpl(Real rho, Real temp, Real ye, Real& kappa_0_nue, Real& kappa_0_nua, Real& kappa_0_nux, Real& kappa_1_nue, Real& kappa_1_nua, Real& kappa_1_nux) {
  // NeutrinoOpacityImpl = 0
  int iout = 0;

  Real rho0 = rho;
  Real temp0 = temp;
  Real ye0   = ye;

  if ((rho0<rho_min) || (temp0<temp_min)) {
    kappa_0_nue = 0.0;
    kappa_0_nua = 0.0;
    kappa_0_nux = 0.0;
    kappa_1_nue = 0.0;
    kappa_1_nua = 0.0;
    kappa_1_nux = 0.0;
    return iout;
  } //end if

  // boundsErr = enforceTableBounds(rho_cgs,temp0,ye0)
  bool boundsErr = EoS->ApplyTableLimits(rho0, temp0, ye0);

  if (boundsErr) {
    iout = WR_OPAC_BNDS_ERR;
    return iout;
  } // end if

  iout = Opacities_cgs(rho0, temp0, ye0,
                       kappa_0_nue, kappa_0_nua,
                       kappa_0_nux,
                       kappa_1_nue, kappa_1_nua,
                       kappa_1_nux);

  // Done elsewhere
  /*
  !Unit conversion
  kappa_cgs2cactus = 1. / (cgs2cactusLength)
  kappa_0_nue = kappa_cgs2cactus * kappa_0_nue_cgs
  kappa_0_nua = kappa_cgs2cactus * kappa_0_nua_cgs
  kappa_0_nux = kappa_cgs2cactus * kappa_0_nux_cgs

  kappa_1_nue = kappa_cgs2cactus * kappa_1_nue_cgs
  kappa_1_nua = kappa_cgs2cactus * kappa_1_nua_cgs
  kappa_1_nux = kappa_cgs2cactus * kappa_1_nux_cgs
  */
  return iout;
} // END FUNCTION NeutrinoOpacityImpl


int WeakOpacityMod::NeutrinoAbsorptionRateImpl(Real rho, Real temp, Real ye, Real& abs_0_nue, Real& abs_0_nua, Real& abs_0_nux, Real& abs_1_nue, Real& abs_1_nua, Real& abs_1_nux) {
  std::cout<<"NeutrinoAbsorptionRateImpl not implemented"<<std::endl;
  return -1;
}

int WeakOpacityMod::Opacities_cgs(Real rho, Real temp, Real ye, Real& kappa_0_nue, Real& kappa_0_nua, Real& kappa_0_nux, Real& kappa_1_nue, Real& kappa_1_nua, Real& kappa_1_nux) {
  int iout = 0;
  int err = 0;

  Real kappa_0_nue_abs = 0.0;
  Real kappa_0_nua_abs = 0.0;
  Real kappa_0_nux_abs = 0.0;
  Real kappa_1_nue_abs = 0.0;
  Real kappa_1_nua_abs = 0.0;
  Real kappa_1_nux_abs = 0.0;

  Real kappa_0_nue_sct = 0.0;
  Real kappa_0_nua_sct = 0.0;
  Real kappa_0_nux_sct = 0.0;
  Real kappa_1_nue_sct = 0.0;
  Real kappa_1_nua_sct = 0.0;
  Real kappa_1_nux_sct = 0.0;


  err = Absorption_cgs(rho, temp, ye, kappa_0_nue_abs, kappa_0_nua_abs, kappa_0_nux_abs, kappa_1_nue_abs, kappa_1_nua_abs, kappa_1_nux_abs);

  if (err != 0) {
    iout = err;
  }

  err = Scattering_cgs(rho, temp, ye, kappa_0_nue_sct, kappa_0_nua_sct, kappa_0_nux_sct, kappa_1_nue_sct, kappa_1_nua_sct, kappa_1_nux_sct);

  if (err != 0) {
    // write(*,*) "Opacities_cgs: Problem in Scattering_cgs"
    iout = err;
  }

  kappa_0_nue = kappa_0_nue_abs + kappa_0_nue_sct;
  kappa_0_nua = kappa_0_nua_abs + kappa_0_nua_sct;
  kappa_0_nux = kappa_0_nux_abs + kappa_0_nux_sct;

  kappa_1_nue = kappa_1_nue_abs + kappa_1_nue_sct;
  kappa_1_nua = kappa_1_nua_abs + kappa_1_nua_sct;
  kappa_1_nux = kappa_1_nux_abs + kappa_1_nux_sct;

  return iout;
} // END FUNCTION Opacities_cgs