#ifndef WEAKRATES_H
#define WEAKRATES_H

#include <cassert>
#include <cmath>

// Athena++ headers
#include "../../../athena.hpp"

// Weakrates headers
#include "weak_rates_constants.hpp"
#include "../common/eos.hpp"

// Common headers
#include "../common/fermi.hpp"
#include "../common/error_codes.hpp"
#include "../common/weak_equilibrium.hpp"
#include "../common/units.hpp"
#include "../common/utils.hpp"

namespace M1::Opacities::WeakRates::WeakRatesNeutrinos{

namespace C = ::M1::Opacities::WeakRates::Constants;
namespace F = ::M1::Opacities::Common::Fermi;
namespace Units = ::M1::Opacities::Common::Units;

class WeakRates {
  public:
  // Type alias for the shared options struct
  using Opt = Common::OpacityUtils::Opt;

  // Constructor - reads M1_opacities settings from the shared opt struct.
  //
  // Flooring logic for rho_min / temp_min:
  //   1. Start from opt.min_*/max_* default values (code units -> CGS+MeV)
  //   2. if (opt.limits_from_table)  -> override from PS_EoS table
  //   3. if (opt.min_rho_usefloor)   -> override rho_min from EOS floor
  //      if (opt.min_t_usefloor)     -> override temp_min from EOS floor
  WeakRates(
    const Opt& opt,
    Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>* PS_EoS)
      : PS_EoS(PS_EoS), WR_EoS{ opt, PS_EoS }, opt_(opt)
  {
    // my_units (of WeakRates) vs code_units (of GR(M)HD)
    my_units   = &Units::WeakRatesUnits;
    code_units = &Units::GeometricSolar;

    // Unit conversion factors (code units -> CGS + MeV)
    Real conv_rho_mass = code_units->MassDensityConversion(*my_units);
    Real conv_rho      = PS_EoS->GetBaryonMass() * conv_rho_mass;
    Real conv_temp     = code_units->TemperatureConversion(*my_units);

    // Step 1: start from opt default values (code units -> CGS+MeV)
    // opt.min_rho etc. are mass densities in code units
    rho_min  = opt.min_rho * conv_rho_mass;
    temp_min = opt.min_t * conv_temp;

    // Step 2: override from EOS table if requested
    if (opt.limits_from_table)
    {
      rho_min  = conv_rho * PS_EoS->GetMinimumDensity();
      temp_min = conv_temp * PS_EoS->GetMinimumTemperature();
    }

    // Step 3: override minimums from EOS atmosphere floors if requested
    if (opt.min_rho_usefloor)
      rho_min = conv_rho * PS_EoS->GetDensityFloor();

    if (opt.min_t_usefloor)
      temp_min = conv_temp * PS_EoS->GetTemperatureFloor();

    atomic_mass = WR_EoS.AtomicMassImpl();

    // Initialize common equilibrium solver
    solver_.Initialize(&WR_EoS, opt);
  }

  // Destructor
  ~WeakRates()
  {
  }

  /*
  I have adopted the following naming convention for the transport
  coefficient variables: <variable>_<type>_<species>

  <variable> = [emi, abs, sct] for emission, absorption opacity, and
  scattering opacty respectively <type> = [n, e] for number and energy
  transport respectively <species> = [nue, nua, nux] for electron-,
  electron-anti-, and heavy-lepton neutrinos respectively


  For the densities and energies themselves I have used:
  <type>_<species>(_eq)

  for the same type and species above, and the (_eq) suffix denotes an
  equilibriated variable.
  */

  // =========================================================================
  // NeutrinoEmission
  //
  // Input:  rho, temp, ye in code units
  // Output: number rates [code units], energy rates [code units]
  // =========================================================================
  int NeutrinoEmission(Real rho,
                       Real temp,
                       Real ye,
                       Real& emi_n_nue,
                       Real& emi_n_nua,
                       Real& emi_n_nux,
                       Real& emi_e_nue,
                       Real& emi_e_nua,
                       Real& emi_e_nux)
  {
    // --- Convert code units -> CGS/MeV ---
    Real rho_cgs  = rho * code_units->MassDensityConversion(*my_units);
    Real temp_mev = temp * code_units->TemperatureConversion(*my_units);
    Real ye_loc   = ye;  // dimensionless

    // --- Bounds check (CGS/MeV) ---
    int bc = C::CheckAndApplyBounds(
      rho_cgs, temp_mev, ye_loc, rho_min, temp_min, &WR_EoS);
    if (bc == -2)
    {
      emi_n_nue = 0.0;
      emi_n_nua = 0.0;
      emi_n_nux = 0.0;
      emi_e_nue = 0.0;
      emi_e_nua = 0.0;
      emi_e_nux = 0.0;
      return 0;
    }
    if (bc == -1)
      return -1;

    // --- Core physics (CGS/MeV) ---
    int err = Emissions_cgs(rho_cgs,
                            temp_mev,
                            ye_loc,
                            emi_n_nue,
                            emi_n_nua,
                            emi_n_nux,
                            emi_e_nue,
                            emi_e_nua,
                            emi_e_nux);

    // --- Convert CGS -> code units ---
    Real n_rate_conv = my_units->NumberRateConversion(*code_units);
    Real e_rate_conv = my_units->EnergyRateConversion(*code_units);
    emi_n_nue *= n_rate_conv;
    emi_n_nua *= n_rate_conv;
    emi_n_nux *= n_rate_conv;
    emi_e_nue *= e_rate_conv;
    emi_e_nua *= e_rate_conv;
    emi_e_nux *= e_rate_conv;

    // Clamp to non-negative
    if (opt_.clamp_nonzero)
    {
      emi_n_nue = (emi_n_nue < 0.0) ? 0.0 : emi_n_nue;
      emi_n_nua = (emi_n_nua < 0.0) ? 0.0 : emi_n_nua;
      emi_n_nux = (emi_n_nux < 0.0) ? 0.0 : emi_n_nux;
      emi_e_nue = (emi_e_nue < 0.0) ? 0.0 : emi_e_nue;
      emi_e_nua = (emi_e_nua < 0.0) ? 0.0 : emi_e_nua;
      emi_e_nux = (emi_e_nux < 0.0) ? 0.0 : emi_e_nux;
    }

    return (err != 0) ? -1 : 0;
  }

  // =========================================================================
  // NeutrinoAbsorptionOpacity
  //
  // Input:  rho, temp, ye in code units
  // Output: absorption opacities [code units, 1/length]
  // =========================================================================
  int NeutrinoAbsorptionOpacity(Real rho,
                                Real temp,
                                Real ye,
                                Real& abs_n_nue,
                                Real& abs_n_nua,
                                Real& abs_n_nux,
                                Real& abs_e_nue,
                                Real& abs_e_nua,
                                Real& abs_e_nux)
  {
    // --- Convert code units -> CGS/MeV ---
    Real rho_cgs  = rho * code_units->MassDensityConversion(*my_units);
    Real temp_mev = temp * code_units->TemperatureConversion(*my_units);
    Real ye_loc   = ye;  // dimensionless

    // --- Bounds check (CGS/MeV) ---
    int bc = C::CheckAndApplyBounds(
      rho_cgs, temp_mev, ye_loc, rho_min, temp_min, &WR_EoS);
    if (bc == -2)
    {
      abs_n_nue = 0.0;
      abs_n_nua = 0.0;
      abs_n_nux = 0.0;
      abs_e_nue = 0.0;
      abs_e_nua = 0.0;
      abs_e_nux = 0.0;
      return 0;
    }
    if (bc == -1)
      return OPAC_BNDS_ERR;

    // --- Core physics (CGS/MeV) ---
    // NOTE: Uses bounds-checked rho_cgs/temp_mev/ye_loc (fixed from
    // original code which incorrectly passed pre-bounds-check values).
    int err = Absorption_cgs(rho_cgs,
                             temp_mev,
                             ye_loc,
                             abs_n_nue,
                             abs_n_nua,
                             abs_n_nux,
                             abs_e_nue,
                             abs_e_nua,
                             abs_e_nux);

    // --- Convert CGS -> code units ---
    Real k_conv = my_units->OpacityConversion(*code_units);
    abs_n_nue *= k_conv;
    abs_n_nua *= k_conv;
    abs_n_nux *= k_conv;
    abs_e_nue *= k_conv;
    abs_e_nua *= k_conv;
    abs_e_nux *= k_conv;

    // Clamp to non-negative
    if (opt_.clamp_nonzero)
    {
      abs_n_nue = (abs_n_nue < 0.0) ? 0.0 : abs_n_nue;
      abs_n_nua = (abs_n_nua < 0.0) ? 0.0 : abs_n_nua;
      abs_n_nux = (abs_n_nux < 0.0) ? 0.0 : abs_n_nux;
      abs_e_nue = (abs_e_nue < 0.0) ? 0.0 : abs_e_nue;
      abs_e_nua = (abs_e_nua < 0.0) ? 0.0 : abs_e_nua;
      abs_e_nux = (abs_e_nux < 0.0) ? 0.0 : abs_e_nux;
    }

    return err;
  }

  // =========================================================================
  // NeutrinoScatteringOpacity
  //
  // Input:  rho, temp, ye in code units
  // Output: scattering opacities [code units, 1/length]
  // =========================================================================
  int NeutrinoScatteringOpacity(Real rho,
                                Real temp,
                                Real ye,
                                Real& sct_n_nue,
                                Real& sct_n_nua,
                                Real& sct_n_nux,
                                Real& sct_e_nue,
                                Real& sct_e_nua,
                                Real& sct_e_nux)
  {
    // --- Convert code units -> CGS/MeV ---
    Real rho_cgs  = rho * code_units->MassDensityConversion(*my_units);
    Real temp_mev = temp * code_units->TemperatureConversion(*my_units);
    Real ye_loc   = ye;  // dimensionless

    // --- Bounds check (CGS/MeV) ---
    int bc = C::CheckAndApplyBounds(
      rho_cgs, temp_mev, ye_loc, rho_min, temp_min, &WR_EoS);
    if (bc == -2)
    {
      sct_n_nue = 0.0;
      sct_n_nua = 0.0;
      sct_n_nux = 0.0;
      sct_e_nue = 0.0;
      sct_e_nua = 0.0;
      sct_e_nux = 0.0;
      return 0;
    }
    if (bc == -1)
      return OPAC_BNDS_ERR;

    // --- Core physics (CGS/MeV) ---
    // NOTE: Uses bounds-checked rho_cgs/temp_mev/ye_loc (fixed from
    // original code which incorrectly passed pre-bounds-check values).
    int err = Scattering_cgs(rho_cgs,
                             temp_mev,
                             ye_loc,
                             sct_n_nue,
                             sct_n_nua,
                             sct_n_nux,
                             sct_e_nue,
                             sct_e_nua,
                             sct_e_nux);

    // --- Convert CGS -> code units ---
    Real k_conv = my_units->OpacityConversion(*code_units);
    sct_n_nue *= k_conv;
    sct_n_nua *= k_conv;
    sct_n_nux *= k_conv;
    sct_e_nue *= k_conv;
    sct_e_nua *= k_conv;
    sct_e_nux *= k_conv;

    // Clamp to non-negative
    if (opt_.clamp_nonzero)
    {
      sct_n_nue = (sct_n_nue < 0.0) ? 0.0 : sct_n_nue;
      sct_n_nua = (sct_n_nua < 0.0) ? 0.0 : sct_n_nua;
      sct_n_nux = (sct_n_nux < 0.0) ? 0.0 : sct_n_nux;
      sct_e_nue = (sct_e_nue < 0.0) ? 0.0 : sct_e_nue;
      sct_e_nua = (sct_e_nua < 0.0) ? 0.0 : sct_e_nua;
      sct_e_nux = (sct_e_nux < 0.0) ? 0.0 : sct_e_nux;
    }

    return err;
  }

  // =========================================================================
  // NeutrinoDensity
  //
  // Equilibrium neutrino number and energy densities (mu_nue = -mu_n + mu_p +
  // mu_e). Input:  rho, temp, ye in code units Output: number densities [code
  // units], energy densities [code units]
  // =========================================================================
  int NeutrinoDensity(Real rho,
                      Real temp,
                      Real ye,
                      Real& n_nue,
                      Real& n_nua,
                      Real& n_nux,
                      Real& e_nue,
                      Real& e_nua,
                      Real& e_nux)
  {
    // Guard: below density/temperature floor -> zero outputs
    Real rho_wr  = rho * code_units->MassDensityConversion(*my_units);
    Real temp_wr = temp * code_units->TemperatureConversion(*my_units);
    if (rho_wr < rho_min || temp_wr < temp_min)
    {
      n_nue = 0.0;
      n_nua = 0.0;
      n_nux = 0.0;
      e_nue = 0.0;
      e_nua = 0.0;
      e_nux = 0.0;
      return 0;
    }

    int ierr = solver_.NeutrinoDensity_cgs_erg(
      rho_wr, temp_wr, ye, n_nue, n_nua, n_nux, e_nue, e_nua, e_nux);

    Real n_conv = my_units->NumberDensityConversion(*code_units);
    Real e_conv = my_units->EnergyDensityConversion(*code_units);
    n_nue *= n_conv;
    n_nua *= n_conv;
    n_nux *= n_conv;
    e_nue *= e_conv;
    e_nua *= e_conv;
    e_nux *= e_conv;

    return ierr;
  }

  // =========================================================================
  // WeakEquilibrium
  //
  // Equilibrium T*, Ye* and neutrino densities from energy/lepton
  // conservation. Input:  rho, temp, ye, neutrino densities in code units
  // Output: temp_eq, ye_eq in code units; equilibrium densities in code units
  // =========================================================================
  int WeakEquilibrium(Real rho,
                      Real temp,
                      Real ye,
                      Real n_nue,
                      Real n_nua,
                      Real n_nux,
                      Real e_nue,
                      Real e_nua,
                      Real e_nux,
                      Real& temp_eq,
                      Real& ye_eq,
                      Real& n_nue_eq,
                      Real& n_nua_eq,
                      Real& n_nux_eq,
                      Real& e_nue_eq,
                      Real& e_nua_eq,
                      Real& e_nux_eq)
  {
    // Guard: below density/temperature floor -> zero outputs
    Real rho_wr  = rho * code_units->MassDensityConversion(*my_units);
    Real temp_wr = temp * code_units->TemperatureConversion(*my_units);
    if (rho_wr < rho_min || temp_wr < temp_min)
    {
      n_nue_eq = 0.0;
      n_nua_eq = 0.0;
      n_nux_eq = 0.0;
      e_nue_eq = 0.0;
      e_nua_eq = 0.0;
      e_nux_eq = 0.0;
      return 0;
    }

    Real n_conv = code_units->NumberDensityConversion(*my_units);
    Real e_conv = code_units->EnergyDensityConversion(*my_units);

    int ierr = solver_.WeakEquilibrium_cgs(rho_wr,
                                           temp_wr,
                                           ye,
                                           n_nue * n_conv,
                                           n_nua * n_conv,
                                           n_nux * n_conv,
                                           e_nue * e_conv,
                                           e_nua * e_conv,
                                           e_nux * e_conv,
                                           temp_eq,
                                           ye_eq,
                                           n_nue_eq,
                                           n_nua_eq,
                                           n_nux_eq,
                                           e_nue_eq,
                                           e_nua_eq,
                                           e_nux_eq);

    temp_eq *= my_units->TemperatureConversion(*code_units);
    // ye_eq is dimensionless
    n_conv = my_units->NumberDensityConversion(*code_units);
    e_conv = my_units->EnergyDensityConversion(*code_units);
    n_nue_eq *= n_conv;
    n_nua_eq *= n_conv;
    n_nux_eq *= n_conv;
    e_nue_eq *= e_conv;
    e_nua_eq *= e_conv;
    e_nux_eq *= e_conv;

    return ierr;
  }

  Real AverageBaryonMass()
  {
    Real am = WR_EoS.AtomicMassImpl();
    am *= my_units->MassConversion(*code_units);
    return am;
  }

  private:
  Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>* PS_EoS;
  const Opt& opt_;

  Units::UnitSystem* my_units;
  Units::UnitSystem* code_units;

  Common::EoS::EoSWrapper WR_EoS;

  // Equilibrium solver
  Common::WeakEquilibrium::WeakEquilibriumSolver<Common::EoS::EoSWrapper>
    solver_;

  // =========================================================================
  // Core physics routines - all operate in CGS / MeV units
  // =========================================================================

  // -----------------------------------------------------------------
  // Emissions_cgs
  //
  // Input:  rho [g/cm^3], temp [MeV], ye [dimensionless]
  // Output: number rates [1/(s cm^3)], energy rates [erg/(s cm^3)]
  //
  // Processes:
  //   1. Beta-process emission (Bruenn 1985)
  //   2. e-e+ pair annihilation (Ruffert 1998, B9)
  //   3. Plasmon decay (Ruffert et al.)
  //   4. Bremsstrahlung (Burrows 2006)
  // -----------------------------------------------------------------
  inline int Emissions_cgs(Real rho,
                           Real temp,
                           Real ye,
                           Real& emi_n_nue,
                           Real& emi_n_nua,
                           Real& emi_n_nux,
                           Real& emi_e_nue,
                           Real& emi_e_nua,
                           Real& emi_e_nux)
  {
    int iout = 0;

    Real eta_nue, eta_nua, eta_nux, eta_e, eta_np, eta_pn;
    WR_EoS.GetEtas(
      rho, temp, ye, eta_nue, eta_nua, eta_nux, eta_e, eta_np, eta_pn);

    Real xn, xp, xh, abar, zbar;
    WR_EoS.GetFracs(rho, temp, ye, xn, xp, xh, abar, zbar);

    // B5 B6 B7 Energy moments of electron and positrons
    const Real hc_3   = C::WR_POW3(C::hc_mevcm);
    const Real temp_4 = C::WR_POW4(temp);
    const Real enr_p  = 8.0 * C::pi / hc_3 * temp_4 * F::fermi3(eta_e);
    const Real enr_m  = 8.0 * C::pi / hc_3 * temp_4 * F::fermi3(-eta_e);

    // ----------------------------------------------------------------
    // Beta-process emission (Bruenn 1985)
    // ----------------------------------------------------------------
    const Real me_mev = C::me_erg / C::mev_to_erg;
    const Real beta = C::pi * C::clight * (1.0 + 3.0 * (C::alpha * C::alpha)) *
                      C::sigma_0 / (hc_3 * (me_mev * me_mev));

    Real block_factor_e = 1.0 + std::exp(eta_nue - F::fermi5O4(eta_e));
    Real block_factor_a = 1.0 + std::exp(eta_nua - F::fermi5O4(-eta_e));

    const Real temp_5 = C::WR_POW4(temp) * temp;
    const Real Rbeta_nue =
      beta * eta_pn * temp_5 * F::fermi4(eta_e) / block_factor_e;
    const Real Qbeta_nue = Rbeta_nue * temp * F::fermi5O4(eta_e);

    const Real Rbeta_nua =
      beta * eta_np * temp_5 * F::fermi4(-eta_e) / block_factor_a;
    const Real Qbeta_nua = Rbeta_nua * temp * F::fermi5O4(-eta_e);

    // ----------------------------------------------------------------
    // e-e+ pair annihilation (Ruffert 1998, B9)
    // ----------------------------------------------------------------
    block_factor_e =
      1.0 +
      std::exp(eta_nue - 0.5 * (F::fermi4O3(eta_e) + F::fermi4O3(-eta_e)));
    block_factor_a =
      1.0 +
      std::exp(eta_nua - 0.5 * (F::fermi4O3(eta_e) + F::fermi4O3(-eta_e)));
    Real block_factor_x =
      1.0 +
      std::exp(eta_nux - 0.5 * (F::fermi4O3(eta_e) + F::fermi4O3(-eta_e)));

    // B8 pair constant
    const Real pair_const =
      ((C::sigma_0 * C::clight) / (me_mev * me_mev)) * enr_m * enr_p;

    // B8
    Real Rpair = pair_const / (36.0 * block_factor_e * block_factor_a) *
                 (C::WR_POW2(C::Cv - C::Ca) + C::WR_POW2(C::Cv + C::Ca));

    const Real Rpair_nue = Rpair;
    const Real Rpair_nua = Rpair;

    const Real Qpair_Factor =
      0.5 * (temp * (F::fermi4O3(-eta_e) + F::fermi4O3(eta_e)));

    const Real Qpair_nue = Rpair * Qpair_Factor;
    const Real Qpair_nua = Rpair * Qpair_Factor;

    // B10
    Rpair = pair_const / (9.0 * (block_factor_x * block_factor_x)) *
            (C::WR_POW2(C::Cv - C::Ca) + C::WR_POW2(C::Cv + C::Ca - 2.0));

    const Real Rpair_nux = Rpair;
    const Real Qpair_nux = Rpair * Qpair_Factor;

    // ----------------------------------------------------------------
    // Plasmon decay (Ruffert et al.)
    // ----------------------------------------------------------------
    const Real gamma =
      C::gamma_0 * std::sqrt(((C::pi * C::pi) + 3.0 * (eta_e * eta_e)) / 3.0);

    block_factor_e =
      1.0 + std::exp(eta_nue - (1.0 + 0.5 * (gamma * gamma) / (1.0 + gamma)));
    block_factor_a =
      1.0 + std::exp(eta_nua - (1.0 + 0.5 * (gamma * gamma) / (1.0 + gamma)));
    block_factor_x =
      1.0 + std::exp(eta_nux - (1.0 + 0.5 * (gamma * gamma) / (1.0 + gamma)));

    const Real gamma_const =
      C::WR_POW3(C::pi) * C::sigma_0 * C::clight *
      C::WR_POW2(C::WR_POW4(temp)) /
      ((me_mev * me_mev) * 3.0 * C::fsc * (hc_3 * hc_3)) *
      (C::WR_POW3(gamma) * C::WR_POW3(gamma)) * std::exp(-gamma) *
      (1.0 + gamma);

    // B11
    Real Rgamma =
      C::Cv * C::Cv * gamma_const / (block_factor_e * block_factor_a);
    const Real Qgamma_Factor =
      0.5 * temp * (2.0 + (gamma * gamma) / (1.0 + gamma));

    const Real Rplasm_nue = Rgamma;
    const Real Qplasm_nue = Rgamma * Qgamma_Factor;
    const Real Rplasm_nua = Rgamma;
    const Real Qplasm_nua = Rgamma * Qgamma_Factor;

    // B12
    Rgamma = (C::Cv - 1.0) * (C::Cv - 1.0) * 4.0 * gamma_const /
             (block_factor_x * block_factor_x);
    const Real Rplasm_nux = Rgamma;
    const Real Qplasm_nux = Rgamma * Qgamma_Factor;

    // ----------------------------------------------------------------
    // Bremsstrahlung (Burrows 2006)
    // ----------------------------------------------------------------
    const Real Qbrem = 0.5 * 2.0778e2 * 0.5 * (1.0 / C::mev_to_erg) *
                       ((xn * xn) + (xp * xp) + 28.0 / 3.0 * xn * xp) *
                       (rho * rho) * std::pow(temp, 5.5);
    const Real Rbrem = 2.0 * Qbrem / (4.364 * temp);

    // ----------------------------------------------------------------
    // Sum contributions
    // ----------------------------------------------------------------
    emi_n_nue = Rbeta_nue + Rpair_nue + Rplasm_nue + Rbrem;
    emi_n_nua = Rbeta_nua + Rpair_nua + Rplasm_nua + Rbrem;
    emi_n_nux = Rpair_nux + Rplasm_nux + 4.0 * Rbrem;

    emi_e_nue = Qbeta_nue + Qpair_nue + Qplasm_nue + Qbrem;
    emi_e_nua = Qbeta_nua + Qpair_nua + Qplasm_nua + Qbrem;
    emi_e_nux = Qpair_nux + Qplasm_nux + 4.0 * Qbrem;

    // Convert energy rate from MeV/(s cm^3) to erg/(s cm^3)
    emi_e_nue *= C::mev_to_erg;
    emi_e_nua *= C::mev_to_erg;
    emi_e_nux *= C::mev_to_erg;

    return iout;
  }

  // -----------------------------------------------------------------
  // Absorption_cgs - charged-current absorption on nucleons (Bruenn 1985)
  //
  // Input:  rho [g/cm^3], temp [MeV], ye [dimensionless]
  // Output: absorption opacities [1/cm]
  // -----------------------------------------------------------------
  inline int Absorption_cgs(Real rho,
                            Real temp,
                            Real ye,
                            Real& abs_n_nue,
                            Real& abs_n_nua,
                            Real& abs_n_nux,
                            Real& abs_e_nue,
                            Real& abs_e_nua,
                            Real& abs_e_nux)
  {
    int iout = 0;

    Real eta_nue, eta_nua, eta_nux, eta_e, eta_np, eta_pn;
    WR_EoS.GetEtas(
      rho, temp, ye, eta_nue, eta_nua, eta_nux, eta_e, eta_np, eta_pn);

    // ABSORPTION
    const Real abs_zeta = (1.0 + 3.0 * (C::alpha * C::alpha)) * 0.25 *
                          C::sigma_0 / (C::me_erg * C::me_erg);

    Real block_factor         = 1.0 + std::exp(eta_e - F::fermi5O4(eta_nue));
    const Real zeta_nue_abs_n = eta_np * abs_zeta / block_factor;

    block_factor              = 1.0 + std::exp(-eta_e - F::fermi5O4(eta_nua));
    const Real zeta_nua_abs_p = eta_pn * abs_zeta / block_factor;

    const Real zeta_nue = zeta_nue_abs_n;  // + 0 (proton) + 0 (heavy)
    const Real zeta_nua = zeta_nua_abs_p;  // + 0 (neutron) + 0 (heavy)
    const Real zeta_nux = 0.0;  // no absorption on nucleons for heavy flavours

    const Real temp_erg_sq = temp * temp * C::mev_to_erg * C::mev_to_erg;
    abs_n_nue              = zeta_nue * temp_erg_sq * F::fermi4O2(eta_nue);
    abs_n_nua              = zeta_nua * temp_erg_sq * F::fermi4O2(eta_nua);
    abs_n_nux              = zeta_nux * temp_erg_sq * F::fermi4O2(eta_nux);

    abs_e_nue = zeta_nue * temp_erg_sq * F::fermi5O3(eta_nue);
    abs_e_nua = zeta_nua * temp_erg_sq * F::fermi5O3(eta_nua);
    abs_e_nux = zeta_nux * temp_erg_sq * F::fermi5O3(eta_nux);

    if (!std::isfinite(abs_n_nue) || !std::isfinite(abs_n_nua) ||
        !std::isfinite(abs_n_nux) || !std::isfinite(abs_e_nue) ||
        !std::isfinite(abs_e_nua) || !std::isfinite(abs_e_nux))
    {
      iout = OPAC_ABS_NONFINITE;
    }

    return iout;
  }

  // -----------------------------------------------------------------
  // Scattering_cgs - neutrino-nucleon + coherent neutrino-nucleus
  //                  scattering (Shapiro & Teukolsky 1983)
  //
  // Input:  rho [g/cm^3], temp [MeV], ye [dimensionless]
  // Output: scattering opacities [1/cm]
  // -----------------------------------------------------------------
  inline int Scattering_cgs(Real rho,
                            Real temp,
                            Real ye,
                            Real& sct_n_nue,
                            Real& sct_n_nua,
                            Real& sct_n_nux,
                            Real& sct_e_nue,
                            Real& sct_e_nua,
                            Real& sct_e_nux)
  {
    int iout = 0;

    Real eta_nue, eta_nua, eta_nux, eta_e, eta_np, eta_pn;
    WR_EoS.GetEtas(
      rho, temp, ye, eta_nue, eta_nua, eta_nux, eta_e, eta_np, eta_pn);

    Real xn, xp, xh, abar, zbar;
    WR_EoS.GetFracs(rho, temp, ye, xn, xp, xh, abar, zbar);

    // SCATTERING
    const Real nb = rho / atomic_mass;

    const Real scttr_cff_n = nb *
                             ((1.0 + 5.0 * (C::alpha * C::alpha)) / 24.0) *
                             C::sigma_0 / (C::me_erg * C::me_erg);
    const Real scttr_cff_p =
      nb *
      ((4.0 * (C::Cv - 1.0) * (C::Cv - 1.0) + 5.0 * (C::alpha * C::alpha)) /
       24.0) *
      C::sigma_0 / (C::me_erg * C::me_erg);

    // Neutrino-nucleon scattering (species-independent coefficients)
    const Real zeta_nue_sct_n = xn * scttr_cff_n;
    const Real zeta_nue_sct_p = xp * scttr_cff_p;
    const Real zeta_nua_sct_n = xn * scttr_cff_n;
    const Real zeta_nua_sct_p = xp * scttr_cff_p;
    const Real zeta_nux_sct_n = xn * scttr_cff_n;
    const Real zeta_nux_sct_p = xp * scttr_cff_p;

    // Coherent neutrino-nucleus scattering (Shapiro & Teukolsky 1983)
    const Real scttr_cff = nb * 0.0625 * C::sigma_0 / (C::me_erg * C::me_erg) *
                           abar * (1.0 - zbar / abar) * (1.0 - zbar / abar);

    const Real zeta_nue_sct_h = xh * scttr_cff;
    const Real zeta_nua_sct_h = xh * scttr_cff;
    const Real zeta_nux_sct_h = xh * scttr_cff;

    const Real zeta_nue = zeta_nue_sct_n + zeta_nue_sct_p + zeta_nue_sct_h;
    const Real zeta_nua = zeta_nua_sct_n + zeta_nua_sct_p + zeta_nua_sct_h;
    const Real zeta_nux = zeta_nux_sct_n + zeta_nux_sct_p + zeta_nux_sct_h;

    const Real temp_erg_sq = temp * temp * C::mev_to_erg * C::mev_to_erg;
    sct_n_nue              = zeta_nue * temp_erg_sq * F::fermi4O2(eta_nue);
    sct_n_nua              = zeta_nua * temp_erg_sq * F::fermi4O2(eta_nua);
    sct_n_nux              = zeta_nux * temp_erg_sq * F::fermi4O2(eta_nux);

    sct_e_nue = zeta_nue * temp_erg_sq * F::fermi5O3(eta_nue);
    sct_e_nua = zeta_nua * temp_erg_sq * F::fermi5O3(eta_nua);
    sct_e_nux = zeta_nux * temp_erg_sq * F::fermi5O3(eta_nux);

    if (!std::isfinite(sct_n_nue) || !std::isfinite(sct_n_nua) ||
        !std::isfinite(sct_n_nux) || !std::isfinite(sct_e_nue) ||
        !std::isfinite(sct_e_nua) || !std::isfinite(sct_e_nux))
    {
      iout = OPAC_SCA_NONFINITE;
    }

    return iout;
  }

  public:

  Real rho_min;
  Real temp_min;
  Real atomic_mass;
};

} // namespace M1::Opacities::WeakRates::WeakRatesNeutrinos

#endif // WEAKRATES_H
