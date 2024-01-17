#include "weak_emission.hpp"
#include "fermi.hpp"

#define WR_SQR(x)  ((x)*(x))
#define WR_CUBE(x) ((x)*(x)*(x))
#define WR_QUAD(x) ((x)*(x)*(x)*(x))

using namespace std;
using namespace WeakRatesFermi;
using namespace WeakRates_Emission;

int WeakEmissionMod::NeutrinoEmissionImpl(Real rho, Real temp, Real ye, Real& emi_n_nue, Real& emi_n_nua, Real& emi_n_nux, Real& emi_e_nue, Real& emi_e_nua, Real& emi_e_nux){
  int iout = 0;

  Real rho0  = rho;
  Real temp0 = temp;
  Real ye0   = ye;

  if ((rho0<rho_min) || (temp0<temp_min)) {
      emi_n_nue = 0.0;
      emi_n_nua = 0.0;
      emi_n_nux = 0.0;
      emi_e_nue = 0.0;
      emi_e_nua = 0.0;
      emi_e_nux = 0.0;
    return iout;
  } // end if

  // boundsErr = enforceTableBounds(rho_cgs,temp0,ye0)
  bool boundsErr = EoS->ApplyTableLimits(rho0, temp0, ye0);

  if (boundsErr) {
    iout = -1;
    return iout;
  } // end if

  int err = Emissions_cgs(rho0, temp0, ye0,
                          emi_n_nue, emi_n_nua, emi_n_nux,
                          emi_e_nue, emi_e_nua, emi_e_nux);

  if (err != 0) {
    iout = -1;
  } //  end if

  // Done elsewhere
  // TODO: normfact?
  /*
  !Unit conversion for the number rates:
  !number of neutrinos * cm^-3 * s^-1 to cactus units
  !1.586234651026439e+10
  r_cgs2cactus = 1. / (cgs2cactusTime*cgs2cactusLength**3)

  emissionRatesRloc_nue = R_nue_cgs * (r_cgs2cactus/normfact)
  emissionRatesRloc_nua = R_nua_cgs * (r_cgs2cactus/normfact)
  emissionRatesRloc_nux = R_nux_cgs * (r_cgs2cactus/normfact)

  !Unit conversion for the energy rates:
  !From MeV * cm^-3 * s^-1 to erg * cm^-3 * s^-1
  !and finally to cactus units
  !1.421737093266046e-50
  q_cgs2cactus = mev_to_erg*cgs2cactusenergy/ &
                 (cgs2cactusTime * cgs2cactusLength**3)

  emissionRatesQloc_nue = Q_nue_cgs * q_cgs2cactus
  emissionRatesQloc_nua = Q_nua_cgs * q_cgs2cactus
  emissionRatesQloc_nux = Q_nux_cgs * q_cgs2cactus
  */

  return iout;
} // END FUNCTION NeutrinoEmissionImpl

int WeakEmissionMod::Emissions_cgs(Real rho, Real temp, Real ye, Real& emi_n_nue, Real& emi_n_nua, Real& emi_n_nux, Real& emi_e_nue, Real& emi_e_nua, Real& emi_e_nux) {
  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // All the emission rates are expressed per unit of volume
  // The number rates (R) are in  1.0 / (sec cm^3)
  // The energy rates (Q) are in  MeV/ (sec cm^3)

  int iout = 0;

  // Broken out into separate function
  /*
  #define WEAK_RATES_ITS_ME
  #include "inc/weak_rates_guts.inc"
  #undef WEAK_RATES_ITS_ME
  */

  Real eta_nue, eta_nua, eta_nux, eta_e, eta_np, eta_pn;
  EoS->GetEtas(rho, temp, ye, eta_nue, eta_nua, eta_nux, eta_e, eta_np, eta_pn);

  Real xp = ye;
  Real xn = 1-ye;
  Real abar = 1.0;
  Real zbar = 1.0;
  Real xh = 0.0;

  // B5 B6 B7 Energy moments of electron and positrons
  Real hc_3 = WR_CUBE(hc_mevcm);
  Real temp_4 = WR_QUAD(temp);
  Real enr_p = 8 * pi / hc_3 * temp_4 * fermi3(eta_e);
  Real enr_m = 8 * pi / hc_3 * temp_4 * fermi3(-eta_e);

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Emission of nue by the beta-process (electron and positron capture
  // see eq.(27) (B1) (B2) see also Tubbs&Schramm, 1975; Bruenn, 1985

  // Coefficient
  Real me_mev = me_erg / mev_to_erg;
  Real beta = pi * clight * (1.0 + 3.0 * (alpha*alpha)) * sigma_0 / (hc_3 * (me_mev*me_mev));

  // Blocking factors (removed them to match Ott's leakage)
  Real block_factor_e = 1.0 + exp(eta_nue - fermi5O4(eta_e));
  Real block_factor_a = 1.0 + exp(eta_nua - fermi5O4(-eta_e));

  // neu electron capture rate
  Real temp_5 = WR_QUAD(temp) * temp;
  Real Rbeta_nue = beta * eta_pn * temp_5 *  fermi4(eta_e)/block_factor_e;

  // neu electron capture energy rate
  Real Qbeta_nue = Rbeta_nue * temp * fermi5O4(eta_e);

  // neu positron capture rate
  Real Rbeta_nua = beta * eta_np * temp_5 * fermi4(-eta_e)/block_factor_a;

  // neu positron capture energy rate
  Real Qbeta_nua = Rbeta_nua * temp * fermi5O4(-eta_e);

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // e-e+ pair processes from Ruffert et al. Ruffert 1998 formula (B9)
  // nu of the electron

  block_factor_e = 1.0 + exp(eta_nue - 0.5 * (fermi4O3(eta_e) + fermi4O3(-eta_e)));

  // anti-nu of the electron
  block_factor_a = 1.0 + exp(eta_nua - 0.5 * (fermi4O3(eta_e) + fermi4O3(-eta_e)));

  // nu of tau and mu
  Real block_factor_x = 1.0 + exp(eta_nux - 0.5 * (fermi4O3(eta_e) + fermi4O3(-eta_e)));

  // B8 electron-positron pair annihilation
  Real pair_const = ((sigma_0 * clight) / (me_mev*me_mev)) * enr_m * enr_p;

  Real enr_tilde_m = 8.0 * pi / hc_3 * temp_5 * fermi4(eta_e);
  Real enr_tilde_p = 8.0 * pi / hc_3 * temp_5 * fermi4(-eta_e);
  
  // B8
  Real Rpair = pair_const / (36.0 * block_factor_e * block_factor_a) * (WR_SQR(Cv-Ca) + WR_SQR(Cv+Ca));

  Real Rpair_nue = Rpair;
  Real Rpair_nua = Rpair;

  Real Qpair_Factor = 0.5 * (temp * (fermi4O3(-eta_e) + fermi4O3(eta_e)));

  // Matching the factor in Ott's leakage
  // Real Qpair_Factor = 0.5*(enr_tilde_m*enr_p+enr_m*enr_tilde_p)/(enr_m*enr_p);

  Real Qpair_nue = Rpair * Qpair_Factor;
  Real Qpair_nua = Rpair * Qpair_Factor;

  // B10
  Rpair =  pair_const/(9.0 * (block_factor_x*block_factor_x)) * (WR_SQR(Cv-Ca) + WR_SQR(Cv + Ca - 2.0));

  Real Rpair_nux = Rpair;
  Real Qpair_nux = Rpair * Qpair_Factor;

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // plasmon decay from Ruffert et al.
  // Definition of gamma
  Real gamma = gamma_0 * sqrt(((pi*pi) + 3.0 * (eta_e*eta_e))/3.0);

  // Blocking factor for plasmon decay
  block_factor_e = 1.0 + exp(eta_nue -(1.0 + 0.5 * (gamma*gamma)/(1.0 + gamma)));
  block_factor_a = 1.0 + exp(eta_nua -(1.0 + 0.5 * (gamma*gamma)/(1.0 + gamma)));
  block_factor_x = 1.0 + exp(eta_nux -(1.0 + 0.5 * (gamma*gamma)/(1.0 + gamma)));

  Real gamma_const = WR_CUBE(pi) * sigma_0 * clight * WR_SQR(WR_QUAD(temp)) / ((me_mev*me_mev) * 3.0 * fsc * (hc_3*hc_3)) * (WR_CUBE(gamma)*WR_CUBE(gamma)) * exp(-gamma) * (1.0+gamma);

  // B11
  Real Rgamma = Cv*Cv * gamma_const / (block_factor_e * block_factor_a);
  Real Qgamma_Factor = 0.5 * temp * (2.0 + (gamma*gamma) / (1.0+gamma));

  Real Rplasm_nue = Rgamma;
  Real Qplasm_nue = Rgamma * Qgamma_Factor;

  Real Rplasm_nua = Rgamma;
  Real Qplasm_nua = Rgamma * Qgamma_Factor;

  //B12
  Rgamma = (Cv-1.0)*(Cv-1.0) * 4.0 * gamma_const / (block_factor_x*block_factor_x);
  Real Rplasm_nux = Rgamma;
  Real Qplasm_nux = Rgamma * Qgamma_Factor;

  // Bremsstrahlung fitting formula described in
  // A. Burrows et al. Nuclear Physics A 777 (2006) 356-394
  // The factor 1/2 is to convert from emissivity of the pair to
  // emissivity for a single neutrino species
  Real Qbrem = 0.5 * 1.04e2 * 0.5 * (1.0/mev_to_erg) * ((xn*xn) + (xp*xp) + 28.0/3.0 * xn * xp) * (rho*rho) * pow(temp,5.5);
  Real Rbrem = 2.0 * Qbrem / (4.364 * temp);

  // Remeber the rates are in (MeV / ) sec / cm^3
  emi_n_nue = Rbeta_nue + Rpair_nue + Rplasm_nue + Rbrem;
  emi_n_nua = Rbeta_nua + Rpair_nua + Rplasm_nua + Rbrem;
  emi_n_nux = Rpair_nux + Rplasm_nux + 4.0 * Rbrem;

  emi_e_nue = Qbeta_nue + Qpair_nue + Qplasm_nue + Qbrem;
  emi_e_nua = Qbeta_nua + Qpair_nua + Qplasm_nua + Qbrem;
  emi_e_nux = Qpair_nux + Qplasm_nux + 4.0 * Qbrem;

  // Convert energy rate to erg / sec / cm^3
  emi_e_nue = mev_to_erg * emi_e_nue;
  emi_e_nua = mev_to_erg * emi_e_nua;
  emi_e_nux = mev_to_erg * emi_e_nux;

/*
#ifndef FORTRAN_DISABLE_IEEE_ARITHMETIC
    if (.not.ieee_is_finite(R_nue)) then
       write(*,*) "Emissions_cgs: NaN/Inf in R_nue", rho, temp, ye
       Emissions_cgs = -1
    endif

    if (.not.ieee_is_finite(R_nua)) then
       write(*,*) "Emissions_cgs: NaN/Inf in R_nua", rho, temp, ye
       Emissions_cgs = -1
    endif

    if (.not.ieee_is_finite(R_nux)) then
       write(*,*) "Emissions_cgs: NaN/Inf in R_nux", rho, temp, ye
       Emissions_cgs = -1
    endif

    if (.not.ieee_is_finite(Q_nue)) then
       write(*,*) "Emissions_cgs: NaN/Inf in Q_nue", rho, temp, ye
       Emissions_cgs = -1
    endif

    if (.not.ieee_is_finite(Q_nua)) then
       write(*,*) "Emissions_cgs: NaN/Inf in Q_nua", rho, temp, ye
       Emissions_cgs = -1
    endif

    if (.not.ieee_is_finite(Q_nux)) then
       write(*,*) "Emissions_cgs: NaN/Inf in Q_nux", rho, temp, ye
       Emissions_cgs = -1
    endif
#endif
*/

  return iout;
} // END FUNCTION Emissions_cgs

#undef WR_SQR
#undef WR_CUBE
#undef WR_QUAD