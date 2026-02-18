#ifndef M1_OPACITIES_BNS_NURATES_WRAPPER_HPP_
#define M1_OPACITIES_BNS_NURATES_WRAPPER_HPP_

// Thin wrapper around the external bns_nurates c++ header-only library.
// Converts code_units <-> nurates_units, calls ComputeM1Opacities(),
// and returns per-species results in code_units.

#include <cassert>
#include <cmath>
#include <cstdio>

#include "../../../athena.hpp"
#include "../common/eos.hpp"
#include "../common/units.hpp"

// ---- Shield POW macros to avoid compilation warnings ----
#ifdef POW2
#pragma push_macro("POW2")
#undef POW2
#endif
#ifdef POW3
#pragma push_macro("POW3")
#undef POW3
#endif
#ifdef POW4
#pragma push_macro("POW4")
#undef POW4
#endif

// bns_nurates external library headers
#include "bns_nurates/include/bns_nurates.hpp"
#include "bns_nurates/include/constants.hpp"
#include "bns_nurates/include/distribution.hpp"
#include "bns_nurates/include/functions.hpp"
#include "bns_nurates/include/integration.hpp"
#include "bns_nurates/include/m1_opacities.hpp"

// ---- Restore shielded macros ----
#ifdef POW2
#pragma pop_macro("POW2")
#endif
#ifdef POW3
#pragma pop_macro("POW3")
#endif
#ifdef POW4
#pragma pop_macro("POW4")
#endif

namespace M1::Opacities::BNS_NuRates::BNSNu_Wrapper
{

namespace Units = ::M1::Opacities::Common::Units;

using Common::EoS::EoSWrapper;

// -----------------------------------------------------------------------
// BNSNuRatesParams  -  reaction flags, quadrature settings
// -----------------------------------------------------------------------
struct BNSNuRatesParams
{
  Real dm_eff;
  Real dU;

  bool use_abs_em;
  bool use_pair;
  bool use_brem;
  bool use_iso;
  bool use_inelastic_scatt;
  bool use_WM_ab;
  bool use_WM_sc;
  bool use_dU;
  bool use_dm_eff;
  bool use_equilibrium_distribution;
  bool use_NN_medium_corr;
  bool neglect_blocking;
  bool use_decay;
  bool use_BRT_brem;

  // no. of quadrature points in bns_nurates
  int quad_nx_1;  // beta_nucleon_scat
  int quad_nx_2;  // pair_bremsstrahlung_lepton_scat

  MyQuadrature quadrature_1;
  MyQuadrature quadrature_2;
};

// -----------------------------------------------------------------------
// BNSNuRatesWrapper - thin wrapper around the external library
// -----------------------------------------------------------------------
class BNSNuRatesWrapper
{
  public:

  BNSNuRatesWrapper(EoSWrapper* eos, ParameterInput* pin) : pmy_eos(eos)
  {
    code_units    = eos->GetCodeUnits();
    nurates_units = eos->GetNuratesUnits();
    wr_units      = eos->GetWrUnits();

    // Parameters for bns_nurates library
    // N.B. max quadrature points is BS_N_MAX/2 = 10 due to 2*n indexing
    params.quad_nx_1 = pin->GetOrAddInteger(
      "bns_nurates", "n_quad_points_beta_nucleon_scat", 10);
    params.quad_nx_2 = pin->GetOrAddInteger(
      "bns_nurates", "n_quad_points_pair_bremsstrahlung_lepton_scat", -1);

    params.use_abs_em =
      pin->GetOrAddBoolean("bns_nurates", "use_abs_em", true);
    params.use_pair = pin->GetOrAddBoolean("bns_nurates", "use_pair", true);
    params.use_brem = pin->GetOrAddBoolean("bns_nurates", "use_brem", true);
    params.use_iso  = pin->GetOrAddBoolean("bns_nurates", "use_iso", true);
    params.use_inelastic_scatt =
      pin->GetOrAddBoolean("bns_nurates", "use_inelastic_scatt", false);
    params.use_WM_ab = pin->GetOrAddBoolean("bns_nurates", "use_WM_ab", false);
    params.use_WM_sc = pin->GetOrAddBoolean("bns_nurates", "use_WM_sc", false);
    params.use_NN_medium_corr =
      pin->GetOrAddBoolean("bns_nurates", "use_NN_medium_corr", false);
    params.neglect_blocking =
      pin->GetOrAddBoolean("bns_nurates", "neglect_blocking", false);
    params.use_decay = pin->GetOrAddBoolean("bns_nurates", "use_decay", false);
    params.use_BRT_brem =
      pin->GetOrAddBoolean("bns_nurates", "use_BRT_brem", true);

    params.use_dU = pin->GetOrAddBoolean("bns_nurates", "use_dU", false);
    params.dU     = pin->GetOrAddReal("bns_nurates", "dU", 0.0);
    params.use_dm_eff =
      pin->GetOrAddBoolean("bns_nurates", "use_dm_eff", false);
    params.dm_eff =
      pin->GetOrAddReal("bns_nurates", "effective_mass_diff", 0.0);

    // Cutoff checks are handled by OpacityUtils::AboveCutoff (utils.hpp);
    // params.rho_min_cgs  = pin->GetOrAddReal("bns_nurates", "rho_min_cgs",
    // 0.); params.temp_min_mev = pin->GetOrAddReal("bns_nurates",
    // "temp_min_mev", 0.);

    params.use_equilibrium_distribution = pin->GetOrAddBoolean(
      "bns_nurates", "use_equilibrium_distribution", true);

    // Set quadratures
    params.quadrature_1.nx   = params.quad_nx_1;
    params.quadrature_1.dim  = 1;
    params.quadrature_1.type = kGauleg;
    params.quadrature_1.x1   = 0.;
    params.quadrature_1.x2   = 1.;
    GaussLegendre(&params.quadrature_1);

    if (params.quad_nx_2 == -1)
    {
      params.quadrature_2 = params.quadrature_1;
    }
    else
    {
      params.quadrature_2.nx   = params.quad_nx_2;
      params.quadrature_2.dim  = 1;
      params.quadrature_2.type = kGauleg;
      params.quadrature_2.x1   = 0.;
      params.quadrature_2.x2   = 1.;
      GaussLegendre(&params.quadrature_2);
    }
  }

  ~BNSNuRatesWrapper()
  {
  }

  // -------------------------------------------------------------------
  // ComputeOpacities
  //   All inputs/outputs in code_units (GeometricSolar).
  //   Returns the number of non-finite output values (0 = success).
  //   Output arrays are indexed by the external library species IDs
  //   (id_nue=0, id_anue=1, id_nux=2, id_anux=3) - 4 species.
  //
  // Input:
  //   nb               [code_units number density]
  //   temp             [MeV]
  //   ye               [-]
  //   mu_n, mu_p, mu_e [MeV]          (chemical potentials)
  //   n_nue .. n_anux  [code_units number density]   per single species
  //   j_nue .. j_anux  [code_units energy density]   per single species
  //   chi_nue .. chi_anux [-]          (Eddington factor, dimensionless)
  //
  // Species / NUX convention (input):
  //   NUX and ANUX are PER SINGLE heavy-lepton species (the 3->4 split
  //   has already been performed by the caller).
  //
  // Output (4-species arrays, code_units):
  //   eta_0[4]     [code_units number rate density]   (1/time/volume)
  //   eta[4]       [code_units energy rate density]   (energy/time/volume)
  //   kappa_0_a[4] [1/code_units length]              (number absorption
  //   opacity) kappa_a[4]   [1/code_units length]              (energy
  //   absorption opacity) kappa_s[4]   [1/code_units length] (scattering
  //   opacity)
  //
  // Species / NUX convention (output):
  //   Each of the 4 species (nue, anue, nux, anux) is returned
  //   independently (PER SINGLE species).  The caller is responsible
  //   for collapsing back to the 3-species M1 convention.
  //
  // Internal unit conversion:
  //   code_units -> nurates_units (NGS: nm, g, s, MeV) for the external
  //   library call, then nurates_units -> code_units on output.
  // -------------------------------------------------------------------
  int ComputeOpacities(Real nb,
                       Real temp,
                       Real ye,
                       Real mu_n,
                       Real mu_p,
                       Real mu_e,
                       Real n_nue,
                       Real j_nue,
                       Real chi_nue,
                       Real n_anue,
                       Real j_anue,
                       Real chi_anue,
                       Real n_nux,
                       Real j_nux,
                       Real chi_nux,
                       Real n_anux,
                       Real j_anux,
                       Real chi_anux,
                       // outputs: 4-species arrays
                       Real eta_0[4],
                       Real eta[4],
                       Real kappa_0_a[4],
                       Real kappa_a[4],
                       Real kappa_s[4]) const
  {
    // Unit conversion factors: code_units -> nurates_units (NGS: nm, g, s,
    // MeV)
    Real const unit_length =
      code_units->LengthConversion(*nurates_units);  // [cm -> nm]
    Real const unit_time =
      code_units->TimeConversion(*nurates_units);  // [Msun -> s]
    Real const unit_num_dens = code_units->NumberDensityConversion(
      *nurates_units);  // [Msun^-3 -> nm^-3]
    Real const unit_ene_dens = code_units->EnergyDensityConversion(
      *nurates_units);  // [Msun^-2 -> MeV/nm^3]
    Real const unit_num_dens_dot =
      unit_num_dens / unit_time;  // [number rate density conversion]
    Real const unit_ene_dens_dot =
      unit_ene_dens / unit_time;  // [energy rate density conversion]

    // Populate GreyOpacityParams for the external library
    GreyOpacityParams grey_op_params = { 0 };

    // reaction flags
    grey_op_params.opacity_flags.use_abs_em = params.use_abs_em;
    grey_op_params.opacity_flags.use_brem   = params.use_brem;
    grey_op_params.opacity_flags.use_pair   = params.use_pair;
    grey_op_params.opacity_flags.use_iso    = params.use_iso;
    grey_op_params.opacity_flags.use_inelastic_scatt =
      params.use_inelastic_scatt;

    // other flags
    grey_op_params.opacity_pars.use_WM_ab          = params.use_WM_ab;
    grey_op_params.opacity_pars.use_WM_sc          = params.use_WM_sc;
    grey_op_params.opacity_pars.use_dU             = params.use_dU;
    grey_op_params.opacity_pars.use_dm_eff         = params.use_dm_eff;
    grey_op_params.opacity_pars.use_NN_medium_corr = params.use_NN_medium_corr;
    grey_op_params.opacity_pars.neglect_blocking   = params.neglect_blocking;
    grey_op_params.opacity_pars.use_decay          = params.use_decay;
    grey_op_params.opacity_pars.use_BRT_brem       = params.use_BRT_brem;

    // EOS quantities in nurates_units (NGS)
    grey_op_params.eos_pars.nb   = nb * unit_num_dens;  // [nm^-3]
    grey_op_params.eos_pars.temp = temp;  // [MeV]  (same in both systems)
    grey_op_params.eos_pars.yp =
      pmy_eos->GetProtonFraction(nb, temp, ye);  // [-]
    grey_op_params.eos_pars.yn =
      pmy_eos->GetNeutronFraction(nb, temp, ye);  // [-]
    grey_op_params.eos_pars.mu_e = mu_e;  // [MeV]  (same in both systems)
    grey_op_params.eos_pars.mu_p = mu_p;  // [MeV]  (same in both systems)
    grey_op_params.eos_pars.mu_n = mu_n;  // [MeV]  (same in both systems)
    // NB logic for use_dU not implemented - see THC/WeakRates2
    grey_op_params.eos_pars.dU     = 0;  // [MeV]
    grey_op_params.eos_pars.dm_eff = 0;  // [MeV]

    // Reconstruct distribution function
    if (!params.use_equilibrium_distribution)
    {
      // M1 quantities in nurates_units
      // Note: no factor-of-2 here; nux/anux are already single heavy species
      grey_op_params.m1_pars.n[id_nue]  = n_nue * unit_num_dens;
      grey_op_params.m1_pars.n[id_anue] = n_anue * unit_num_dens;
      grey_op_params.m1_pars.n[id_nux]  = n_nux * unit_num_dens;
      grey_op_params.m1_pars.n[id_anux] = n_anux * unit_num_dens;

      grey_op_params.m1_pars.J[id_nue]  = j_nue * unit_ene_dens;
      grey_op_params.m1_pars.J[id_anue] = j_anue * unit_ene_dens;
      grey_op_params.m1_pars.J[id_nux]  = j_nux * unit_ene_dens;
      grey_op_params.m1_pars.J[id_anux] = j_anux * unit_ene_dens;

      grey_op_params.m1_pars.chi[id_nue]  = chi_nue;
      grey_op_params.m1_pars.chi[id_anue] = chi_anue;
      grey_op_params.m1_pars.chi[id_nux]  = chi_nux;
      grey_op_params.m1_pars.chi[id_anux] = chi_anux;

      grey_op_params.distr_pars = CalculateDistrParamsFromM1(
        &grey_op_params.m1_pars, &grey_op_params.eos_pars);
    }
    else
    {
      grey_op_params.distr_pars =
        NuEquilibriumParams(&grey_op_params.eos_pars);

      // compute neutrino number and energy densities at equilibrium
      ComputeM1DensitiesEq(&grey_op_params.eos_pars,
                           &grey_op_params.distr_pars,
                           &grey_op_params.m1_pars);

      grey_op_params.m1_pars.chi[id_nue]  = 0.333333333333333333333333333;
      grey_op_params.m1_pars.chi[id_anue] = 0.333333333333333333333333333;
      grey_op_params.m1_pars.chi[id_nux]  = 0.333333333333333333333333333;
      grey_op_params.m1_pars.chi[id_anux] = 0.333333333333333333333333333;
    }

    // Quadrature bounds check
    if (2 * params.quadrature_1.nx > BS_N_MAX ||
        2 * params.quadrature_2.nx > BS_N_MAX)
    {
      std::printf(
        "[bns_nurates]: quadrature nx (%d, %d) exceeds BS_N_MAX/2 = %d. "
        "Reduce n_quad_points_* to at most %d to avoid buffer overflow.\n",
        params.quadrature_1.nx,
        params.quadrature_2.nx,
        BS_N_MAX / 2,
        BS_N_MAX / 2);
      std::exit(0);
    }

    // ---- Call the external library ----
    M1Opacities opacities = ComputeM1Opacities(
      &params.quadrature_1, &params.quadrature_2, &grey_op_params);

    // Convert back to code_units
    for (int s = 0; s < 4; ++s)
    {
      eta_0[s]     = opacities.eta_0[s] / unit_num_dens_dot;
      eta[s]       = opacities.eta[s] / unit_ene_dens_dot;
      kappa_0_a[s] = opacities.kappa_0_a[s] * unit_length;
      kappa_a[s]   = opacities.kappa_a[s] * unit_length;
      kappa_s[s]   = opacities.kappa_s[s] * unit_length;
    }

    // Count non-finite values
    int ierr = 0;
    for (int s = 0; s < 4; ++s)
    {
      if (!std::isfinite(eta_0[s]))
        ierr++;
      if (!std::isfinite(eta[s]))
        ierr++;
      if (!std::isfinite(kappa_0_a[s]))
        ierr++;
      if (!std::isfinite(kappa_a[s]))
        ierr++;
      if (!std::isfinite(kappa_s[s]))
        ierr++;
    }

    return ierr;
  }

  // -------------------------------------------------------------------
  // Accessors
  // -------------------------------------------------------------------
  const BNSNuRatesParams& GetParams() const
  {
    return params;
  }
  bool UseEquilibriumDistribution() const
  {
    return params.use_equilibrium_distribution;
  }

  private:
  EoSWrapper* pmy_eos;

  Units::UnitSystem* code_units;
  Units::UnitSystem* nurates_units;
  Units::UnitSystem* wr_units;

  // mutable because GaussLegendre modifies the quadrature structs at init,
  // and ComputeM1Opacities takes non-const pointers
  mutable BNSNuRatesParams params;
};

}  // namespace M1::Opacities::BNS_NuRates::BNSNu_Wrapper

#endif  // M1_OPACITIES_BNS_NURATES_WRAPPER_HPP_
