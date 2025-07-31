#ifndef M1_OPACITIES_WEAKRATES_HPP
#define M1_OPACITIES_WEAKRATES_HPP

// Athena++ classes headers
#include "../../../athena.hpp"
#include "../../m1.hpp"
#include "../../m1_sources.hpp"
#include "../../m1_calc_closure.hpp"
#include "../../../hydro/hydro.hpp"
#include "../../../eos/eos.hpp"

#include "../../../coordinates/coordinates.hpp"
#include "../../../field/field.hpp"

// Weakrates header
#include "error_codes.hpp"
#include "weak_rates.hpp"
#include <cmath>
#include <iomanip>
#include <limits>

namespace M1::Opacities::WeakRates {

class WeakRates {

friend class ::M1::Opacities::Opacities;

public:
  WeakRates(MeshBlock *pmb, M1 * pm1, ParameterInput *pin) :
    pm1(pm1),
    pmy_mesh(pmb->pmy_mesh),
    pmy_block(pmb),
    pmy_coord(pmy_block->pcoord),
    N_GRPS(pm1->N_GRPS),
    N_SPCS(pm1->N_SPCS),
    revert_thick_limit_equilibrium(
      pin->GetOrAddBoolean("M1_opacities",
                           "revert_thick_limit_equilibrium",
                           false)
    ),
    propagate_hydro_equilibrium(
      pin->GetOrAddBoolean("M1_opacities",
                           "propagate_hydro_equilibrium",
                           false)
    ),
    wr_impose_equilibrium(
      pin->GetOrAddBoolean("M1_opacities",
                           "wr_impose_equilibrium",
                           false)
    ),
    wr_flag_equilibrium(
      pin->GetOrAddBoolean("M1_opacities",
                           "wr_flag_equilibrium",
                           false)
    ),
    wr_flag_equilibrium_species(
      pin->GetOrAddBoolean("M1_opacities",
                           "wr_flag_equilibrium_species",
                           false)
    ),
    wr_flag_equilibrium_dt_factor(
      pin->GetOrAddReal("M1_opacities",
                        "wr_flag_equilibrium_dt_factor",
                        1.0)
    ),
    wr_flag_equilibrium_recompute_fiducial(
      pin->GetOrAddBoolean("M1_opacities",
                           "wr_flag_equilibrium_recompute_fiducial",
                           false)
    ),
    correction_adjust_upward(
      pin->GetOrAddBoolean("M1_opacities",
                           "correction_adjust_upward",
                           false)
    )
  {

#if !USETM
    #pragma omp critical
    {
      std::cout << "Warning: ";
      std::cout << "M1::Opacities::WeakRates needs TEOS to work properly \n";
    }
#endif

#if !(NSCALARS>0)
    #pragma omp critical
    {
      std::cout << "Warning: ";
      std::cout << "M1::Opacities::WeakRates needs NSCALARS>0 to work function \n";
    }
#endif

    // Weakrates only works for nu_e + nu_ae + nu_x
    assert(N_SPCS==3);

    // Weakrates only works for 1 group
    assert(N_GRPS==1);

    // Create instance of WeakRatesNeutrinos::WeakRates
    // Set EoS from PS
    pmy_weakrates = new WeakRatesNeutrinos::WeakRates(pin,
                                                      &pmy_block->peos->GetEOS());

    // These are the defaults in THC, TODO convert to pin parameters
    opacity_tau_trap = pin->GetOrAddReal("M1_opacities", "tau_trap", 1.0);
    opacity_tau_delta = pin->GetOrAddReal("M1_opacities", "tau_delta", 1.0);
    opacity_corr_fac_max = pin->GetOrAddReal("M1_opacities", "max_correction_factor", 3.0);

    verbose_warn_weak =
        pin->GetOrAddBoolean("M1_opacities", "verbose_warn_weak", true);
    validate_opacities =
        pin->GetOrAddBoolean("M1_opacities", "validate_opacities", false);
    zero_invalid_radmat =
        pin->GetOrAddBoolean("M1_opacities", "zero_invalid_radmat", false);
    use_averages =
        pin->GetOrAddBoolean("M1_opacities", "use_averages", false);
    use_averaging_fix =
        pin->GetOrAddBoolean("M1_opacities", "use_averaging_fix", false);
  };

  ~WeakRates() {
    if (pmy_weakrates!= nullptr) {
      delete pmy_weakrates;
    }
  };

  inline void GetNearestNeighborAverages(int k, int j, int i,
                                         Real &avg_rho,
                                         Real &avg_T,
                                         Real &avg_Y_e,
                                         bool exclude_first_extrema)
  {
    // Calculate average of nearest neighbor densities, temperatures, and Y_e
    avg_rho = 0.0;
    avg_T = 0.0;
    avg_Y_e = 0.0;
    int count = 0;

    // If we need to exclude extrema, we need to track min/max values
    Real min_rho = std::numeric_limits<Real>::max();
    Real max_rho = -std::numeric_limits<Real>::max();
    Real min_T = std::numeric_limits<Real>::max();
    Real max_T = -std::numeric_limits<Real>::max();
    Real min_Y_e = std::numeric_limits<Real>::max();
    Real max_Y_e = -std::numeric_limits<Real>::max();

    for (int kk = -1; kk <= 1; ++kk)
    for (int jj = -1; jj <= 1; ++jj)
    for (int ii = -1; ii <= 1; ++ii)
    {
      if (ii == 0 && jj == 0 && kk == 0) continue;

      const Real rho = pm1->hydro.sc_w_rho(k+kk, j+jj, i+ii);
      const Real T = pm1->hydro.sc_T(k+kk, j+jj, i+ii);
      const Real Y_e = pm1->hydro.sc_w_Ye(k+kk, j+jj, i+ii);

      avg_rho += rho;
      avg_T   += T;
      avg_Y_e += Y_e;

      if (exclude_first_extrema)
      {
        // Track min/max values
        min_rho = std::min(min_rho, rho);
        max_rho = std::max(max_rho, rho);
        min_T = std::min(min_T, T);
        max_T = std::max(max_T, T);
        min_Y_e = std::min(min_Y_e, Y_e);
        max_Y_e = std::max(max_Y_e, Y_e);
      }

      count++;
    }

    if (exclude_first_extrema)
    {
      avg_rho -= min_rho;
      avg_rho -= max_rho;
      avg_T -= min_T;
      avg_T -= max_T;
      avg_Y_e -= min_Y_e;
      avg_Y_e -= max_Y_e;

      count -= 2;
    }

    avg_rho /= count;
    avg_T /= count;
    avg_Y_e /= count;

    // do not double-average temperature
    if (pmy_block->peos->smooth_temperature)
    {
      avg_T = pm1->hydro.sc_T(k, j, i);
    }

  }

  inline int ValidateRadMatQuantities(int k, int j, int i)
  {
    int ierr_invalid = 0;

    // Check all radiation-matter quantities for the current cell
    for (int s_idx = 0; s_idx < N_SPCS; ++s_idx)
    {
      // Check finiteness and non-negativity of all quantities
      if (!std::isfinite(pm1->radmat.sc_kap_a_0(0, s_idx)(k, j, i)) ||
          pm1->radmat.sc_kap_a_0(0, s_idx)(k, j, i) < 0.0 ||
          !std::isfinite(pm1->radmat.sc_kap_a(0, s_idx)(k, j, i)) ||
          pm1->radmat.sc_kap_a(0, s_idx)(k, j, i) < 0.0 ||
          !std::isfinite(pm1->radmat.sc_kap_s(0, s_idx)(k, j, i)) ||
          pm1->radmat.sc_kap_s(0, s_idx)(k, j, i) < 0.0 ||
          !std::isfinite(pm1->radmat.sc_eta_0(0, s_idx)(k, j, i)) ||
          pm1->radmat.sc_eta_0(0, s_idx)(k, j, i) < 0.0 ||
          !std::isfinite(pm1->radmat.sc_eta(0, s_idx)(k, j, i)) ||
          pm1->radmat.sc_eta(0, s_idx)(k, j, i) < 0.0 ||
          !std::isfinite(pm1->radmat.sc_avg_nrg(0, s_idx)(k, j, i)) ||
          pm1->radmat.sc_avg_nrg(0, s_idx)(k, j, i) < 0.0)
      {
        ierr_invalid = WR_RADMAT_INVALID;
      }
    }

    return ierr_invalid;
  }

  inline void SetZeroRadMatAtPoint(int k, int j, int i)
  {
    for (int s_idx = 0; s_idx < N_SPCS; ++s_idx)
    {
      pm1->radmat.sc_kap_a_0(0, s_idx)(k, j, i) = 0.0;
      pm1->radmat.sc_kap_a(0, s_idx)(k, j, i) = 0.0;
      pm1->radmat.sc_kap_s(0, s_idx)(k, j, i) = 0.0;
      pm1->radmat.sc_eta_0(0, s_idx)(k, j, i) = 0.0;
      pm1->radmat.sc_eta(0, s_idx)(k, j, i) = 0.0;
      pm1->radmat.sc_avg_nrg(0, s_idx)(k, j, i) = 0.0;
    }
  }

  // --------------------------------------------------------------------------
  // Logic for:
  // 1) Opacity calculation
  // 2) Equilibrium reversion
  // 3) Correction factor application
  inline int CalculateOpacityCoefficients(int k, int j, int i,
                                          Real rho, Real T, Real Y_e,
                                          Real* kap_a_n, Real* kap_a_e,
                                          Real* kap_s_n, Real* kap_s_e,
                                          Real* eta_n, Real* eta_e)
  {
    int iem, iab, isc;

    iem = pmy_weakrates->NeutrinoEmission(
      rho, T, Y_e,
      eta_n[NUE], eta_n[NUA], eta_n[NUX],
      eta_e[NUE], eta_e[NUA], eta_e[NUX]
    );

    iab = pmy_weakrates->NeutrinoAbsorptionOpacity(
      rho, T, Y_e,
      kap_a_n[NUE], kap_a_n[NUA], kap_a_n[NUX],
      kap_a_e[NUE], kap_a_e[NUA], kap_a_e[NUX]
    );

    isc = pmy_weakrates->NeutrinoScatteringOpacity(
      rho, T, Y_e,
      kap_s_n[NUE], kap_s_n[NUA], kap_s_n[NUX],
      kap_s_e[NUE], kap_s_e[NUA], kap_s_e[NUX]
    );

    int res = iem || iab || isc;
    return res;
  }

  // Time-scale for equilibrium regime
  inline Real CalculateTau(Real * kap_a_e, Real* kap_s_e)
  {
    /*
    tau = std::min(
      std::sqrt(kap_a_e[NUE] * (kap_a_e[NUE] + kap_s_e[NUE])),
      std::sqrt(kap_a_e[NUA] * (kap_a_e[NUA] + kap_s_e[NUA]))
    ) * dt;
    */
    /*
    auto min3 = [&](Real a, Real b, Real c)
    {
      return std::min(a, std::min(b, c));
    };

    auto max3 = [&](Real a, Real b, Real c)
    {
      return std::max(a, std::max(b, c));
    };

    const Real tau_fac = 1.0 / max3(
      std::sqrt(kap_a_e[NUE] * (kap_a_e[NUE] + kap_s_e[NUE])),
      std::sqrt(kap_a_e[NUA] * (kap_a_e[NUA] + kap_s_e[NUA])),
      std::sqrt(kap_a_e[NUX] * (kap_a_e[NUX] + kap_s_e[NUX]))
    );
    */

    Real tau_fac = std::numeric_limits<Real>::infinity();

    const int ix_g = 0;

    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      if (ix_s == NUX)
        continue;

      const Real rat = 1.0 / std::sqrt(
        kap_a_e[ix_s] * (kap_a_e[ix_s] + kap_s_e[ix_s])
      );

      tau_fac = (std::isfinite(rat))
        ? std::min(tau_fac, rat)
        : tau_fac;
    }
    return tau_fac;
  }

  // based on internal (corrected) opacity
  inline Real CalculateTau(const int s_idx,
                           const int k, const int j, const int i)
  {
    typedef M1::vars_RadMat RM;
    RM & rm = pm1->radmat;

    const int ix_g = 0;
    const Real kap_s_e = rm.sc_kap_s(ix_g, s_idx)(k, j, i);
    const Real kap_a_e = rm.sc_kap_a(ix_g, s_idx)(k, j, i);

    const Real rat = 1.0 / std::sqrt(
      kap_a_e * (kap_a_e + kap_s_e)
    );

    return (std::isfinite(rat))
      ? rat
      : std::numeric_limits<Real>::infinity();
  }

  inline int ComputeEquilibriumDensities(
    const int k, const int j, const int i,
    const Real dt,
    const Real rho, const Real T, const Real Y_e,
    Real & tau,
    Real & T_star, Real & Y_e_star,
    Real* kap_a_n, Real* kap_a_e,
    Real* kap_s_n, Real* kap_s_e,
    Real* dens_n, Real* dens_e,
    bool ignore_current_data,
    bool & TY_adjusted,
    bool & calculate_trapped
  )
  {
    int ierr_we = 0;
    int ierr_nd = 0;

    TY_adjusted = false;

    Real dens_n_trap[3];
    Real dens_e_trap[3];
    Real dens_n_thin[3];
    Real dens_e_thin[3];

    calculate_trapped = (
      (opacity_tau_trap >= 0.0) &&
      (tau < dt / opacity_tau_trap) &&
      (rho >= pm1->opt_solver.tra_rho_min)
    );

    // Calculate equilibrium blackbody functions with trapped neutrinos
    if (calculate_trapped)
    {
      // Ensure evolution method delegated based on here detected equilibrium
      const static int ix_g = 0;
      // typedef M1::evolution_strategy::opt_solution_regime osr_r;
      // typedef M1::evolution_strategy::opt_source_treatment ost_r;
      // AthenaArray<osr_r> & sol_r = pm1->ev_strat.masks.solution_regime;
      // AthenaArray<ost_r> & src_r = pm1->ev_strat.masks.source_treatment;

      // BD: use instead wr_flag_equilibrium_corrected
      // for (int ix_s=0; ix_s<3; ++ix_s)
      // {
      //   pm1->SetMaskSolutionRegime(
      //     M1::M1::t_sln_r::equilibrium,ix_g,ix_s,k,j,i
      //   );
      // }

      // set thick limit
      if (revert_thick_limit_equilibrium)
      {

        for (int ix_s=0; ix_s<3; ++ix_s)
        {
          Closures::EddingtonFactors::ThickLimit(
            pm1->lab_aux.sc_xi(ix_g,ix_s)(k,j,i),
            pm1->lab_aux.sc_chi(ix_g,ix_s)(k,j,i)
          );
        }
      }

      // ----------------------------------------------------------------------
      Real invsdetg = pm1->geom.sc_oo_sqrt_det_g(k, j, i);

      // FF number density
      dens_n[0] = pm1->rad.sc_n(0, 0)(k, j, i) * invsdetg;
      dens_n[1] = pm1->rad.sc_n(0, 1)(k, j, i) * invsdetg;
      dens_n[2] = pm1->rad.sc_n(0, 2)(k, j, i) * invsdetg;

      // FF energy density
      dens_e[0] = pm1->rad.sc_J(0, 0)(k, j, i) * invsdetg;
      dens_e[1] = pm1->rad.sc_J(0, 1)(k, j, i) * invsdetg;
      dens_e[2] = pm1->rad.sc_J(0, 2)(k, j, i) * invsdetg;

      // Calculate equilibriated state
      ierr_we = pmy_weakrates->WeakEquilibrium(
        rho, T, Y_e,
        (ignore_current_data) ? 0.0 : dens_n[0],
        (ignore_current_data) ? 0.0 : dens_n[1],
        (ignore_current_data) ? 0.0 : dens_n[2],
        (ignore_current_data) ? 0.0 : dens_e[0],
        (ignore_current_data) ? 0.0 : dens_e[1],
        (ignore_current_data) ? 0.0 : dens_e[2],
        T_star, Y_e_star,
        dens_n_trap[0],
        dens_n_trap[1],
        dens_n_trap[2],
        dens_e_trap[0],
        dens_e_trap[1],
        dens_e_trap[2]
      );

      if (ierr_we)
      {
        return ierr_we;
      }

      // immediately break on error or non-finite values ----------------------
      Real val = 0;
      for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
      {
        val += dens_n_trap[ix_s] + dens_e_trap[ix_s];
      }
      if (!std::isfinite(val))
      {
        return WE_FAIL_NONFINITE;
      }

      TY_adjusted = true;
      // ----------------------------------------------------------------------
    }
    else
    {
      // No trapped calculation, so leave T,Y_e unmodified if later propagated
      T_star = T;
      Y_e_star = Y_e;
    }

    const bool need_thin = !(
      calculate_trapped &&
      (tau < dt / (opacity_tau_trap + opacity_tau_delta))
    );

    // only calculate if actually needed
    if (need_thin)
    {
      ierr_nd = pmy_weakrates->NeutrinoDensity(
          rho, T, Y_e,
          dens_n_thin[0], dens_n_thin[1], dens_n_thin[2],
          dens_e_thin[0], dens_e_thin[1], dens_e_thin[2]
      );

      if (ierr_nd)
      {
        // immediately break on error
        return ierr_nd;
      }
    }

    // Set the black body function
    if (calculate_trapped)
    {
      // proceed further ------------------------------------------------------
      if (tau < dt / (opacity_tau_trap + opacity_tau_delta))
      {
        for (int s_idx = 0; s_idx < N_SPCS; ++s_idx)
        {
          dens_n[s_idx] = dens_n_trap[s_idx];
          dens_e[s_idx] = dens_e_trap[s_idx];
        }
      }
      else
      {
        // tau in dt * [1 / (opacity_tau_trap + opacity_tau_delta),
        //              1 / opacity_tau_trap)

        // const Real lam = (tau - opacity_tau_trap) / opacity_tau_delta;
        const Real I_tau_min = dt / (opacity_tau_trap + opacity_tau_delta);
        const Real I_tau_max = dt / opacity_tau_trap;

        // tau_map in [0, 1]
        const Real tau_map = (tau - I_tau_min) / (I_tau_max - I_tau_min);
        const Real lam = 1 - tau_map;

        for (int s_idx = 0; s_idx < N_SPCS; ++s_idx)
        {
          dens_n[s_idx] = (
            lam * dens_n_trap[s_idx] + (1.0 - lam) * dens_n_thin[s_idx]
          );
          dens_e[s_idx] = (
            lam * dens_e_trap[s_idx] + (1.0 - lam) * dens_e_thin[s_idx]
          );
        }

        // Intermediate regime, so linearly interpolate T,Y_e for later propagation
        T_star = lam * T_star + (1.0 - lam) * T;
        Y_e_star = lam * Y_e_star + (1.0 - lam) * Y_e;
      }
    }
    else
    {
      // free-streaming
      for (int s_idx = 0; s_idx < N_SPCS; ++s_idx)
      {
        dens_n[s_idx] = dens_n_thin[s_idx];
        dens_e[s_idx] = dens_e_thin[s_idx];
      }
    }

    // finally, it may be useful to retain the equilibrium --------------------
    if (pm1->opt.retain_equilibrium)
    {
      const int ix_g = 0;
      const Real sc_sqrt_det_g = pm1->geom.sc_sqrt_det_g(k,j,i);

      for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
      {
        pm1->eql.sc_J(ix_g,ix_s)(k,j,i) = sc_sqrt_det_g * dens_e[ix_s];
        pm1->eql.sc_n(ix_g,ix_s)(k,j,i) = sc_sqrt_det_g * dens_n[ix_s];
      }

    }

    return 0;
  }

  inline void ApplyOpacityCorrections(
    int k, int j, int i,
    const Real dt,
    const Real T, const Real T_star,
    const Real* dens_n,  const Real* dens_e,
    const Real* kap_a_n, const Real* kap_a_e,
    const Real* kap_s_n, const Real* kap_s_e,
    const Real* eta_n,   const Real* eta_e
    )
  {
    /*
    // Calculate correction factors
    Real corr_fac[3];
    for (int s_idx = 0; s_idx < N_SPCS; ++s_idx)
    {
      pm1->radmat.sc_avg_nrg(0, s_idx)(k, j, i) = ((dens_e[s_idx] > 0) &&
                                                   (dens_n[s_idx] > 0))
          ? dens_e[s_idx] / dens_n[s_idx]
          : 0.0;

      corr_fac[s_idx] = pm1->rad.sc_J(0, s_idx)(k, j, i) /
        (pm1->rad.sc_n(0, s_idx)(k, j, i) *
         pm1->radmat.sc_avg_nrg(0, s_idx)(k, j, i));

      if (!std::isfinite(corr_fac[s_idx]))
      {
        corr_fac[s_idx] = 1.0;
        if (!std::isfinite(pm1->radmat.sc_avg_nrg(0, s_idx)(k, j, i)))
        {
          pm1->radmat.sc_avg_nrg(0, s_idx)(k, j, i) = 0;
        }
        // should never land here (due to flooring prior to call of opac.)
        // assert(false);
      }

      corr_fac[s_idx] *= corr_fac[s_idx];
      corr_fac[s_idx] =
          std::max(1.0 / opacity_corr_fac_max,
                    std::min(corr_fac[s_idx], opacity_corr_fac_max));
    }

    // Energy scattering
    pm1->radmat.sc_kap_s(0, 0)(k, j, i) = corr_fac[0] * kap_s_e[NUE];
    pm1->radmat.sc_kap_s(0, 1)(k, j, i) = corr_fac[1] * kap_s_e[NUA];
    pm1->radmat.sc_kap_s(0, 2)(k, j, i) = corr_fac[2] * kap_s_e[NUX];

    // Enforce Kirchhoff's law
    // For electron lepton neutrinos we change the opacity
    // For heavy lepton neutrinos we change the emissivity

    // Electron neutrinos
    pm1->radmat.sc_kap_a_0(0, 0)(k, j, i) = corr_fac[0] * kap_a_n[NUE];
    pm1->radmat.sc_kap_a(0, 0)(k, j, i)   = corr_fac[0] * kap_a_e[NUE];

    pm1->radmat.sc_eta_0(0, 0)(k, j, i) =
        pm1->radmat.sc_kap_a_0(0, 0)(k, j, i) * dens_n[0];
    pm1->radmat.sc_eta(0, 0)(k, j, i) =
        pm1->radmat.sc_kap_a(0, 0)(k, j, i) * dens_e[0];

    // Electron anti-neutrinos
    pm1->radmat.sc_kap_a_0(0, 1)(k, j, i) = corr_fac[1] * kap_a_n[NUA];
    pm1->radmat.sc_kap_a(0, 1)(k, j, i)   = corr_fac[1] * kap_a_e[NUA];

    pm1->radmat.sc_eta_0(0, 1)(k, j, i) =
        pm1->radmat.sc_kap_a_0(0, 1)(k, j, i) * dens_n[1];
    pm1->radmat.sc_eta(0, 1)(k, j, i) =
        pm1->radmat.sc_kap_a(0, 1)(k, j, i) * dens_e[1];

    // Heavy lepton neutrinos
    pm1->radmat.sc_eta_0(0, 2)(k, j, i) = corr_fac[2] * eta_n[NUX];
    pm1->radmat.sc_eta(0, 2)(k, j, i)   = corr_fac[2] * eta_e[NUX];

    pm1->radmat.sc_kap_a_0(0, 2)(k, j, i) =
        (dens_n[2] > 1e-20 ? pm1->radmat.sc_eta_0(0, 2)(k, j, i) / dens_n[2]
                        : 0.0);
    pm1->radmat.sc_kap_a(0, 2)(k, j, i) =
        (dens_e[2] > 1e-20 ? pm1->radmat.sc_eta(0, 2)(k, j, i) / dens_e[2] : 0.0);
    */

    // Do computations, check validity afterwards
    typedef M1::vars_RadMat RM;
    typedef M1::vars_Rad R;
    RM & rm = pm1->radmat;
    R  & r  = pm1->rad;

    Real corr_fac[3];
    const Real ix_g = 0;  // only 1 group

    const Real W = pm1->fidu.sc_W(k,j,i);

    for (int s_idx = 0; s_idx < N_SPCS; ++s_idx)
    {
      // equilibrium and incoming energies
      Real avg_nrg_eql = ((dens_n[s_idx] > 0) &&
                          (dens_e[s_idx] > 0))
        ? dens_e[s_idx] / dens_n[s_idx]
        : 0.0;

      // Compute directly if not computed elsewhere: --------------------------
      AT_C_sca & sc_E = pm1->lab.sc_E(ix_g,s_idx);
      AT_N_vec & sp_F_d = pm1->lab.sp_F_d(ix_g,s_idx);
      AT_C_sca & sc_nG = pm1->lab.sc_nG(ix_g,s_idx);

      Real dotFv (0.0);
      for (int a=0; a<N; ++a)
      {
        dotFv += sp_F_d(a,k,j,i) * pm1->fidu.sp_v_u(a,k,j,i);
      }
      rm.sc_avg_nrg(ix_g,s_idx)(k,j,i) = (
        W / sc_nG(k,j,i) * (sc_E(k,j,i) - dotFv)
      );
      rm.sc_avg_nrg(ix_g,s_idx)(k,j,i) = std::max(
        rm.sc_avg_nrg(ix_g,s_idx)(k,j,i), 0.0
      );
      // ----------------------------------------------------------------------

      Real avg_nrg_inc = rm.sc_avg_nrg(0, s_idx)(k, j, i);
      corr_fac[s_idx] = avg_nrg_inc / avg_nrg_eql;

      if (!std::isfinite(corr_fac[s_idx]))
        corr_fac[s_idx] = 1.0;

      if (!std::isfinite(avg_nrg_eql))
        avg_nrg_eql = 0.0;

      rm.sc_avg_nrg(0, s_idx)(k, j, i) = avg_nrg_eql;

      // T_star should not be used for this.
      // corr_fac[s_idx] = std::min(
      //   std::max(1.0, T_star / T), opacity_corr_fac_max
      // );

      if (correction_adjust_upward)
      {
        corr_fac[s_idx] = std::max(
          std::min(corr_fac[s_idx], opacity_corr_fac_max), 1.0
        );
      }
      else
      {
        corr_fac[s_idx] =
            std::max(1.0 / opacity_corr_fac_max,
                      std::min(corr_fac[s_idx], opacity_corr_fac_max));
      }

      corr_fac[s_idx] *= corr_fac[s_idx];
    }

    // Energy scattering
    rm.sc_kap_s(0, 0)(k, j, i) = corr_fac[0] * kap_s_e[NUE];
    rm.sc_kap_s(0, 1)(k, j, i) = corr_fac[1] * kap_s_e[NUA];
    rm.sc_kap_s(0, 2)(k, j, i) = corr_fac[2] * kap_s_e[NUX];

    // Enforce Kirchhoff's law
    // For electron lepton neutrinos we change the opacity
    // For heavy lepton neutrinos we change the emissivity

    // Electron neutrinos
    rm.sc_kap_a_0(0, 0)(k, j, i) = corr_fac[0] * kap_a_n[NUE];
    rm.sc_kap_a(0, 0)(k, j, i)   = corr_fac[0] * kap_a_e[NUE];

    rm.sc_eta_0(0, 0)(k, j, i) = rm.sc_kap_a_0(0, 0)(k, j, i) * dens_n[0];
    rm.sc_eta(0, 0)(k, j, i)   = rm.sc_kap_a(0, 0)(k, j, i) * dens_e[0];

    // Electron anti-neutrinos
    rm.sc_kap_a_0(0, 1)(k, j, i) = corr_fac[1] * kap_a_n[NUA];
    rm.sc_kap_a(0, 1)(k, j, i)   = corr_fac[1] * kap_a_e[NUA];

    rm.sc_eta_0(0, 1)(k, j, i) = rm.sc_kap_a_0(0, 1)(k, j, i) * dens_n[1];
    rm.sc_eta(0, 1)(k, j, i) = rm.sc_kap_a(0, 1)(k, j, i) * dens_e[1];

    // Heavy lepton neutrinos
    rm.sc_eta_0(0, 2)(k, j, i) = corr_fac[2] * eta_n[NUX];
    rm.sc_eta(0, 2)(k, j, i)   = corr_fac[2] * eta_e[NUX];

    rm.sc_kap_a_0(0, 2)(k, j, i) = (dens_n[2] > 0)
      ? rm.sc_eta_0(0, 2)(k, j, i) / dens_n[2]
      : 0.0;
    rm.sc_kap_a(0, 2)(k, j, i) = (dens_e[2] > 0)
      ? rm.sc_eta(0, 2)(k, j, i) / dens_e[2]
      : 0.0;

    // Check validity. Zero quantities if any not valid -----------------------
    bool valid = true;
    for (int s_idx = 0; s_idx < N_SPCS; ++s_idx)
    {
      valid = valid && std::isfinite(corr_fac[s_idx]);
      valid = valid && std::isfinite(rm.sc_avg_nrg(0, s_idx)(k, j, i));
    }
    valid = (valid &&
             std::isfinite(rm.sc_kap_a_0(0, 2)(k, j, i)) &&
             std::isfinite(rm.sc_kap_a(0, 2)(k, j, i)));

    if (!valid)
    {
      if (verbose_warn_weak)
      {
        const Real rho = pm1->hydro.sc_w_rho(k, j, i);
        const Real Y_e = pm1->hydro.sc_w_Ye(k, j, i);

        const Real tau = 0;
        const Real Y_e_star = 0;

        PrintOpacityDiagnostics(
          "ApplyOpacityCorrections failure",
          WR_OPAC_CORRECTION_ERROR,
          k, j, i, dt, rho, T, Y_e, tau, T_star, Y_e_star,
          kap_a_n, kap_a_e, kap_s_n, kap_s_e, eta_n, eta_e,
          dens_n, dens_e
        );
      }
      SetZeroRadMatAtPoint(k,j,i);
    }
  }

  // Flag equilibrium treatment
  inline void FlagEquilibrium(
    const Real dt,
    const Real tau,
    const Real rho,
    const int k, const int j, const int i)
  {
    const int ix_g = 0;
    const Real dt_fac = dt * wr_flag_equilibrium_dt_factor;

    bool flag_eql = ((opacity_tau_trap >= 0.0) &&
                     (tau < dt_fac / opacity_tau_trap) &&
                     (rho >= pm1->opt_solver.eql_rho_min));

    M1::M1::t_sln_r sln_flag = (flag_eql)
      ? M1::M1::t_sln_r::equilibrium
      : M1::M1::t_sln_r::noop;

    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      pm1->SetMaskSolutionRegime(sln_flag,ix_g,ix_s,k,j,i);
    }

    // optionally assume thick limit and recompute fiducial frame
    if (wr_flag_equilibrium_recompute_fiducial &&
        (sln_flag == M1::M1::t_sln_r::equilibrium))
    {

      // access previous state vector (opacities only computed _once_)
      M1::M1::vars_Lab U_P { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
      pm1->SetVarAliasesLab(pm1->storage.u1, U_P);

      for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
      {
        AT_C_sca & sc_E    = U_P.sc_E(  ix_g,ix_s);
        AT_N_vec & sp_F_d  = U_P.sp_F_d(ix_g,ix_s);
        AT_C_sca & sc_xi   = pm1->lab_aux.sc_xi( ix_g,ix_s);
        AT_C_sca & sc_chi  = pm1->lab_aux.sc_chi(ix_g,ix_s);

        AT_C_sca & sc_nG   = U_P.sc_nG(ix_g,ix_s);

        AT_C_sca & sc_J   = pm1->rad.sc_J(  ix_g,ix_s);
        AT_D_vec & st_H_u = pm1->rad.st_H_u(ix_g, ix_s);
        AT_C_sca & sc_n   = pm1->rad.sc_n(  ix_g,ix_s);

        AT_C_sca & sc_avg_nrg = pm1->radmat.sc_avg_nrg(ix_g,ix_s);

        Closures::EddingtonFactors::ThickLimit(
          pm1->lab_aux.sc_xi(ix_g,ix_s)(k,j,i),
          pm1->lab_aux.sc_chi(ix_g,ix_s)(k,j,i)
        );

        const Real Gamma__ = Assemble::Frames::ToFiducial(
          *pm1,
          sc_J, st_H_u, sc_n,
          sc_chi,
          sc_E, sp_F_d, sc_nG,
          k, j, i
        );

        // Average energy (Eulerian frame)
        /*
        Real dotFv (0.0);
        for (int a=0; a<N; ++a)
        {
          dotFv += sp_F_d(a,k,j,i) * pm1->fidu.sp_v_u(a,k,j,i);
        }
        const Real W = pm1->fidu.sc_W(k,j,i);
        sc_avg_nrg(k,j,i) = W / sc_nG(k,j,i) * (sc_E(k,j,i) - dotFv);
        */
      }

    }
  }

  // Flag eql; compute per-species
  inline void FlagEquilibrium(
    const Real dt,
    const Real rho,
    const int k, const int j, const int i)
  {
    const int ix_g = 0;  // fix 1 group
    const Real dt_fac = dt * wr_flag_equilibrium_dt_factor;

    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      const Real tau = CalculateTau(ix_s, k, j, i);

      bool flag_eql = ((opacity_tau_trap >= 0.0) &&
                      (tau < dt_fac / opacity_tau_trap) &&
                      (rho >= pm1->opt_solver.eql_rho_min));

      M1::M1::t_sln_r sln_flag = (flag_eql)
        ? M1::M1::t_sln_r::equilibrium
        : M1::M1::t_sln_r::noop;

      pm1->SetMaskSolutionRegime(sln_flag,ix_g,ix_s,k,j,i);
    }
  }

  inline void ImposeEquilibriumDensities(
    const int k, const int j, const int i,
    Real* dens_n, Real* dens_e
  )
  {
    // Set equilibrium in fiducial frame:
    // (sc_J, st_H_d={sc_H_t, sp_H_d}, sc_n)
    // Reconstruct Eulerian: (sc_E, sp_F_d, sc_nG)
    const Real sc_sqrt_det_g = pm1->geom.sc_sqrt_det_g(k,j,i);
    const Real W = pm1->fidu.sc_W(k,j,i);
    const Real W2 = SQR(W);

    const int ix_g = 0;

    // access previous state vector
    M1::M1::vars_Lab U_P { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
    pm1->SetVarAliasesLab(pm1->storage.u1, U_P);

    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      pm1->SetMaskSolutionRegime(
        M1::M1::t_sln_r::equilibrium_wr,ix_g,ix_s,k,j,i
      );

      AT_C_sca & sc_J   = pm1->rad.sc_J(ix_g,ix_s);
      AT_D_vec & st_H_u = pm1->rad.st_H_u(ix_g,ix_s);
      AT_C_sca & sc_n   = pm1->rad.sc_n(ix_g,ix_s);

      AT_C_sca & sc_E    = pm1->lab.sc_E(ix_g,ix_s);
      AT_N_vec & sp_F_d  = pm1->lab.sp_F_d(ix_g,ix_s);
      AT_C_sca & sc_nG   = pm1->lab.sc_nG(ix_g,ix_s);

      AT_C_sca & P_sc_E    = U_P.sc_E(ix_g,ix_s);
      AT_N_vec & P_sp_F_d  = U_P.sp_F_d(ix_g,ix_s);
      AT_C_sca & P_sc_nG   = U_P.sc_nG(ix_g,ix_s);

      AT_C_sca & S_sc_E    = pm1->sources.sc_E(ix_g,ix_s);
      AT_N_vec & S_sp_F_d  = pm1->sources.sp_F_d(ix_g,ix_s);
      AT_C_sca & S_sc_nG   = pm1->sources.sc_nG(ix_g,ix_s);

      AT_C_sca & sc_chi  = pm1->lab_aux.sc_chi(ix_g,ix_s);
      AT_C_sca & sc_xi   = pm1->lab_aux.sc_xi(ix_g,ix_s);

      // sc_J(k,j,i) = nudens(1,ix_s) * sc_sqrt_det_g;
      sc_J(k,j,i) = dens_e[ix_s] * sc_sqrt_det_g;

      for (int a=0; a<D; ++a)
      {
        st_H_u(a,k,j,i) = 0;
      }

      // (sc_J, st_H_d) -> (sc_E, sp_F_d) reduces to:
      sc_E(k,j,i) = ONE_3RD * sc_J(k,j,i) * (
        4.0 * W2 - 1.0
      );

      for (int a=0; a<N; ++a)
      {
        sp_F_d(a,k,j,i) = 4.0 * ONE_3RD * W2 *
                          pm1->fidu.sp_v_d(a,k,j,i) * sc_J(k,j,i);
      }

      Real dotFv__ (0.0);
      for (int a=0; a<N; ++a)
      {
        dotFv__ += sp_F_d(a,k,j,i) * pm1->fidu.sp_v_u(a,k,j,i);
      }
      const Real Gamma__ = W / sc_J(k,j,i) * (
        sc_E(k,j,i) - dotFv__
      );

      // Prepare neutrino number density
      // sc_n(k,j,i) = nudens(0, ix_s) * sc_sqrt_det_g;
      sc_n(k,j,i) = dens_n[ix_s] * sc_sqrt_det_g;

      sc_nG(k,j,i) = std::max(Gamma__ * sc_n(k,j,i), pm1->opt.fl_nG);
      sc_n(k,j,i) = sc_nG(k,j,i) / Gamma__;  // propagate back

      // Flooring -----------------------------------------------------------
      const bool floor_applied = sc_E(k,j,i) < pm1->opt.fl_E;
      sc_E(k,j,i) = std::max(sc_E(k,j,i), pm1->opt.fl_E);

      if (floor_applied)
      {
        for (int a=0; a<N; ++a)
        {
          sp_F_d(a,k,j,i) = 0;
        }
      }

      sc_nG(k,j,i) = std::max(sc_nG(k,j,i), pm1->opt.fl_nG);

      // Sources --------------------------------------------------------------
      /*
      if (pm1->opt_solver.equilibrium_sources)
      {
        Assemble::Frames::sources_sc_E_sp_F_d(
          *pm1,
          S_sc_E,
          S_sp_F_d,
          sc_chi,
          sc_E,
          sp_F_d,
          pm1->radmat.sc_eta(ix_g, ix_s), // V.sc_eta,
          pm1->radmat.sc_kap_a(ix_g, ix_s), // V.sc_kap_a,
          pm1->radmat.sc_kap_s(ix_g, ix_s), // V.sc_kap_s,
          k, j, i
        );

        Assemble::Frames::sources_sc_nG(
          *pm1,
          S_sc_nG,
          sc_n,
          pm1->radmat.sc_eta_0(ix_g, ix_s), // V.sc_eta_0,
          pm1->radmat.sc_kap_a_0(ix_g, ix_s), // V.sc_kap_a_0,
          k, j, i
        );
      }
      else
      {
        S_sc_E(k,j,i) = 0.0;
        for (int a=0; a<D; ++a)
        {
          S_sp_F_d(a,k,j,i) = 0.0;
        }
        S_sc_nG(k,j,i) = 0.0;
      }
      */

      if (pm1->opt_solver.equilibrium_src_E_F_d)
      {
        if (pm1->opt_solver.equilibrium_use_diff_src)
        {
          // new - star
          S_sc_E(k,j,i) = sc_E(k,j,i) - P_sc_E(k,j,i);
          for (int n=0; n<N; ++n)
          {
            S_sp_F_d(n,k,j,i) = sp_F_d(n,k,j,i) - P_sp_F_d(n,k,j,i);
          }
        }
        else
        {
          Assemble::Frames::sources_sc_E_sp_F_d(
            *pm1,
            S_sc_E,
            S_sp_F_d,
            sc_chi,
            sc_E,
            sp_F_d,
            pm1->radmat.sc_eta(ix_g, ix_s),   // V.sc_eta,
            pm1->radmat.sc_kap_a(ix_g, ix_s), // V.sc_kap_a,
            pm1->radmat.sc_kap_s(ix_g, ix_s), // V.sc_kap_s,
            k, j, i
          );
        }
      }
      else
      {
        S_sc_E(k,j,i) = 0.0;
        for (int a=0; a<D; ++a)
        {
          S_sp_F_d(a,k,j,i) = 0.0;
        }
      }

      if (pm1->opt_solver.equilibrium_src_nG)
      {
        if (pm1->opt_solver.equilibrium_use_diff_src)
        {
          // new - star
          S_sc_nG(k,j,i) = sc_nG(k,j,i) - P_sc_nG(k,j,i);
        }
        else
        {
          Assemble::Frames::sources_sc_nG(
            *pm1,
            S_sc_nG,
            sc_n,
            pm1->radmat.sc_eta_0(ix_g, ix_s),   // V.sc_eta_0,
            pm1->radmat.sc_kap_a_0(ix_g, ix_s), // V.sc_kap_a_0,
            k, j, i
          );
        }
      }
      else
      {
        S_sc_nG(k,j,i) = 0.0;
      }
    }
  }

  inline void PrintOpacityDiagnostics(
    const std::string & msg, const int error_code,
    int k, int j, int i,
    Real dt,
    Real rho, Real T, Real Y_e,
    Real tau, Real T_star, Real Y_e_star,
    const Real* kap_a_n, const Real* kap_a_e,
    const Real* kap_s_n, const Real* kap_s_e,
    const Real* eta_n,   const Real* eta_e,
    const Real* dens_n,  const Real* dens_e)
  {
    std::cout << "M1: Opacity diagnostics";
    if (!msg.empty())
    {
      std::cout << " [" << msg << "; error_code: " << error_code << "]";
    }
    std::cout << " at (i,j,k)=(" << i << "," << j << "," << k << ")\n";
    std::cout << std::setprecision(14);
    std::cout << "Physical conditions:\n"
              << "  dt    = " << dt << "\n"
              << "  rho   = " << rho << "\n"
              << "  T     = " << T << "\n"
              << "  Y_e   = " << Y_e << "\n"
              << "  tau   = " << tau << "\n"
              << "  T*    = " << T_star << "\n"
              << "  Y_e*  = " << Y_e_star << "\n"
              << " rad.sc_n(0,NUE)(k,j,i)   = "
              << pm1->rad.sc_n(0, NUE)(k, j, i) << "\n"
              << " rad.sc_n(0,NUA)(k,j,i)   = "
              << pm1->rad.sc_n(0, NUA)(k, j, i) << "\n"
              << " rad.sc_n(0,NUX)(k,j,i)   = "
              << pm1->rad.sc_n(0, NUX)(k, j, i) << "\n"
              << " rad.sc_J(0,NUE)(k,j,i)   = "
              << pm1->rad.sc_J(0, NUE)(k, j, i) << "\n"
              << " rad.sc_J(0,NUA)(k,j,i)   = "
              << pm1->rad.sc_J(0, NUA)(k, j, i) << "\n"
              << " rad.sc_J(0,NUX)(k,j,i)   = "
              << pm1->rad.sc_J(0, NUX)(k, j, i) << "\n"
              << " OO(det_gamma)            = "
              << pm1->geom.sc_oo_sqrt_det_g(k, j, i) << "\n";

    std::string regime;

    // infer regime using same logic as ComputeEquilibriumDensities -----------

    // BD: would be better to unify this detection logic more cleanly...

    const bool calculate_trapped = (
      (opacity_tau_trap >= 0.0) &&
      (tau < dt / opacity_tau_trap) &&
      (rho >= pm1->opt_solver.tra_rho_min)
    );
    const bool need_thin = !(
      calculate_trapped &&
      (tau < dt / (opacity_tau_trap + opacity_tau_delta))
    );

    if (calculate_trapped)
    {
      if (tau < dt / (opacity_tau_trap + opacity_tau_delta))
      {
        regime = "trapped";
      }
      else
      {
        regime = "interpolated";
      }
    }
    else
    {
      regime = "thin";
    }
    // ------------------------------------------------------------------------
    std::cout << "Opacity regime: " << regime << "\n";

    std::cout << "Opacity coefficients:\n";
    for (int s_idx = 0; s_idx < N_SPCS; ++s_idx)
    {
      std::string species;
      switch (s_idx) {
        case NUE: species = "nu_e "; break;
        case NUA: species = "nu_a "; break;
        case NUX: species = "nu_x "; break;
      }

      std::cout << "  " << species << ":\n"
                << "    kap_a_n = " << kap_a_n[s_idx] << "\n"
                << "    kap_a_e = " << kap_a_e[s_idx] << "\n"
                << "    kap_s_n = " << kap_s_n[s_idx] << "\n"
                << "    kap_s_e = " << kap_s_e[s_idx] << "\n"
                << "    dens_n  = " << dens_n[s_idx] << "\n"
                << "    dens_e  = " << dens_e[s_idx] << "\n";
    }

    Real avg_rho = 0.0;
    Real avg_T = 0.0;
    Real avg_Y_e = 0.0;
    GetNearestNeighborAverages(k, j, i, avg_rho, avg_T, avg_Y_e, false);

    std::cout << "Neighbor averages (excluding current point):\n"
              << "  avg_rho = " << avg_rho << "\n"
              << "  avg_T   = " << avg_T << "\n"
              << "  avg_Y_e = " << avg_Y_e << "\n";

    std::cout << "Current radmat values:\n";
    for (int s_idx = 0; s_idx < N_SPCS; ++s_idx) {
      std::string species;
      switch (s_idx) {
        case NUE: species = "nu_e "; break;
        case NUA: species = "nu_a "; break;
        case NUX: species = "nu_x "; break;
      }

      std::cout << "  " << species << ":\n"
                << "    kap_a_0 = " << pm1->radmat.sc_kap_a_0(0, s_idx)(k, j, i) << "\n"
                << "    kap_a   = " << pm1->radmat.sc_kap_a(0, s_idx)(k, j, i) << "\n"
                << "    kap_s   = " << pm1->radmat.sc_kap_s(0, s_idx)(k, j, i) << "\n"
                << "    eta_0   = " << pm1->radmat.sc_eta_0(0, s_idx)(k, j, i) << "\n"
                << "    eta     = " << pm1->radmat.sc_eta(0, s_idx)(k, j, i) << "\n"
                << "    avg_nrg = " << pm1->radmat.sc_avg_nrg(0, s_idx)(k, j, i) << "\n";
    }

    std::cout << std::endl;
  }

  // Do calculation here ------------------------------------------------------
  inline int CalculateOpacityWeakRates(Real const dt, AA &u)
  {
    auto& status = pm1->ev_strat.status;

    M1_FLOOP3(k, j, i)
    if (pm1->MaskGet(k, j, i))
    {
      Real rho = pm1->hydro.sc_w_rho(k, j, i);

      // check whether we need to do anything
      if (rho <= pmy_weakrates->rho_min_code_units)
      {
        SetZeroRadMatAtPoint(k, j, i);
        continue;
      }

      Real T = pm1->hydro.sc_T(k,j,i);
      Real Y_e = pm1->hydro.sc_w_Ye(k, j, i);

      // Order in the arrays is: e, a, x
      Real kap_a_n[N_SPCS];
      Real kap_a_e[N_SPCS];
      Real kap_s_n[N_SPCS];
      Real kap_s_e[N_SPCS];
      Real eta_n[N_SPCS];
      Real eta_e[N_SPCS];

      Real dens_n[N_SPCS];
      Real dens_e[N_SPCS];

      // For equilibrium
      Real Y_e_star;
      Real T_star;
      Real tau;
      bool TY_adjusted = false;
      bool calculate_trapped = false;

      if (use_averages)
      {
        // Smoothed data (nn avg) for the opacity calculation
        //
        // Probably not a great idea...
        const bool exclude_first_extrema = true;
        GetNearestNeighborAverages(k, j, i, rho, T, Y_e, exclude_first_extrema);
      }

      int ierr_opac = CalculateOpacityCoefficients(
        k, j, i, rho, T, Y_e,
        kap_a_n, kap_a_e,
        kap_s_n, kap_s_e,
        eta_n, eta_e
      );

      if (ierr_opac)
      {
        status.num_opac_failures++;

        if (verbose_warn_weak)
        {
          PrintOpacityDiagnostics(
            "CalculateOpacityCoefficients failure", ierr_opac,
            k, j, i, dt, rho, T, Y_e, tau, T_star, Y_e_star,
            kap_a_n, kap_a_e, kap_s_n, kap_s_e, eta_n, eta_e,
            dens_n, dens_e
          );
        }

        if (use_averaging_fix)
        {
          const bool exclude_first_extrema = true;
          GetNearestNeighborAverages(k, j, i, rho, T, Y_e, exclude_first_extrema);

          ierr_opac = CalculateOpacityCoefficients(
            k, j, i, rho, T, Y_e,
            kap_a_n, kap_a_e,
            kap_s_n, kap_s_e,
            eta_n, eta_e
          );

          // Now revert the point:
          rho = pm1->hydro.sc_w_rho(k, j, i);
          T = pm1->hydro.sc_T(k,j,i);
          Y_e = pm1->hydro.sc_w_Ye(k, j, i);

          if (ierr_opac)
          {
            if (verbose_warn_weak)
            {
              PrintOpacityDiagnostics(
                "CalculateOpacityCoefficients (averaged) failure", ierr_opac,
                k, j, i, dt, rho, T, Y_e, tau, T_star, Y_e_star,
                kap_a_n, kap_a_e, kap_s_n, kap_s_e, eta_n, eta_e,
                dens_n, dens_e
              );
            }
          }
          else
          {
            status.num_opac_fixes++;
          }
        }
      }

      // Compute characteristic equilibrium time-scale
      tau = CalculateTau(kap_a_e, kap_s_e);

      bool ignore_current_data = false;


      int ierr_we = ComputeEquilibriumDensities(
        k, j, i,
        dt,
        rho, T, Y_e,
        tau,
        T_star, Y_e_star,
        kap_a_n, kap_a_e,
        kap_s_n, kap_s_e,
        dens_n, dens_e,
        ignore_current_data,
        TY_adjusted,
        calculate_trapped
      );

      if (ierr_we)
      {
        status.num_equi_failures++;

        if (verbose_warn_weak)
        {
          PrintOpacityDiagnostics(
            "ComputeEquilibriumDensities failure", ierr_we,
            k, j, i, dt, rho, T, Y_e, tau, T_star, Y_e_star,
            kap_a_n, kap_a_e, kap_s_n, kap_s_e, eta_n, eta_e,
            dens_n, dens_e);
        }

        if (use_averaging_fix)
        {
          const bool exclude_first_extrema = true;
          GetNearestNeighborAverages(k, j, i, rho, T, Y_e, exclude_first_extrema);

          ierr_we = ComputeEquilibriumDensities(
            k, j, i,
            dt,
            rho, T, Y_e,
            tau,
            T_star, Y_e_star,
            kap_a_n, kap_a_e,
            kap_s_n, kap_s_e,
            dens_n, dens_e,
            ignore_current_data,
            TY_adjusted,
            calculate_trapped
          );

          // Now revert the point:
          rho = pm1->hydro.sc_w_rho(k, j, i);
          T = pm1->hydro.sc_T(k,j,i);
          Y_e = pm1->hydro.sc_w_Ye(k, j, i);

          if (ierr_we)
          {
            if (verbose_warn_weak)
            {
              PrintOpacityDiagnostics(
                "ComputeEquilibriumDensities (averaged) failure", ierr_we,
                k, j, i, dt, rho, T, Y_e, tau, T_star, Y_e_star,
                kap_a_n, kap_a_e, kap_s_n, kap_s_e, eta_n, eta_e,
                dens_n, dens_e);
            }
          }
          else
          {
            status.num_equi_fixes++;
          }
        }
      }

      // retry ignoring current neutrino data
      if (ierr_we)
      {
        ignore_current_data = true;
        ierr_we = ComputeEquilibriumDensities(
          k, j, i,
          dt,
          rho, T, Y_e,
          tau,
          T_star, Y_e_star,
          kap_a_n, kap_a_e,
          kap_s_n, kap_s_e,
          dens_n, dens_e,
          ignore_current_data,
          TY_adjusted,
          calculate_trapped
        );

        if (ierr_we)
        {
          if (verbose_warn_weak)
          {
            PrintOpacityDiagnostics(
              "ComputeEquilibriumDensities - ignored neutrino data - failure",
              ierr_we,
              k, j, i, dt, rho, T, Y_e, tau, T_star, Y_e_star,
              kap_a_n, kap_a_e, kap_s_n, kap_s_e, eta_n, eta_e,
              dens_n, dens_e);
          }
        }
        else
        {
          status.num_equi_ignored++;
        }
      }

      ApplyOpacityCorrections(
        k, j, i,
        dt,
        T, T_star,
        dens_n, dens_e,
        kap_a_n, kap_a_e,
        kap_s_n, kap_s_e,
        eta_n, eta_e
      );

      if (wr_flag_equilibrium)
      {
        if (wr_flag_equilibrium_species)
        {
          FlagEquilibrium(dt, rho, k, j, i);
        }
        else
        {
          FlagEquilibrium(dt, tau, rho, k, j, i);
        }
      }

      if (propagate_hydro_equilibrium && TY_adjusted)
      {
        // N.B.:
        // pm1->hydro.X slice into phydro
        // similarly Y_e slices into pscalars->r
        pm1->hydro.sc_w_Ye(k, j, i) = Y_e_star;
        pm1->hydro.sc_T(k,j,i) = T_star;

        // update pressure keeping density fixed
        Real const nb = rho / (pmy_block->peos->GetEOS().GetBaryonMass());
        Real Y_e_star_vec[MAX_SPECIES] = {Y_e_star};
        pm1->hydro.sc_w_p(k,j,i) = pmy_block->peos->GetEOS().GetPressure(
          nb, T_star, Y_e_star_vec
        );

        pmy_block->peos->PrimitiveToConserved(
          pmy_block->phydro->w,
          pmy_block->pscalars->r,
          pmy_block->pfield->bcc,
          pmy_block->phydro->u,
          pmy_block->pscalars->s,
          pmy_block->pcoord,
          i, i,
          j, j,
          k, k
        );
      }

      if (calculate_trapped && wr_impose_equilibrium)
      {
        ImposeEquilibriumDensities(k, j, i, dens_n, dens_e);
      }

      if (validate_opacities)
      {
        int ierr_vrm = ValidateRadMatQuantities(k,j,i);
        if (ierr_vrm && verbose_warn_weak)
        {
          PrintOpacityDiagnostics(
            "ValidateRadMatQuantities failure", ierr_vrm,
            k, j, i, dt, rho, T, Y_e, tau, T_star, Y_e_star,
            kap_a_n, kap_a_e, kap_s_n, kap_s_e, eta_n, eta_e,
            dens_n, dens_e);
        }

        if (ierr_vrm && zero_invalid_radmat)
        {
          SetZeroRadMatAtPoint(k,j,i);
          status.num_radmat_zero++;
        }
      }
    }

    return 0;
  };

private:
  M1 *pm1;
  Mesh *pmy_mesh;
  MeshBlock *pmy_block;
  Coordinates *pmy_coord;
  WeakRatesNeutrinos::WeakRates *pmy_weakrates = nullptr;

  const int N_GRPS;
  const int N_SPCS;

  enum {NUE=0, NUA=1, NUX=2};

  const bool revert_thick_limit_equilibrium;
  const bool propagate_hydro_equilibrium;
  const bool wr_impose_equilibrium;
  const bool wr_flag_equilibrium;
  const Real wr_flag_equilibrium_dt_factor;
  const bool wr_flag_equilibrium_recompute_fiducial;
  const bool wr_flag_equilibrium_species;
  const bool correction_adjust_upward;

  // Options for controlling weakrates opacities
  Real opacity_tau_trap;
  Real opacity_tau_delta;
  Real opacity_corr_fac_max;

  bool verbose_warn_weak;
  bool validate_opacities;
  bool zero_invalid_radmat;
  bool use_averages;
  bool use_averaging_fix;
};

} // namespace M1::Opacities::WeakRates

#endif //M1_OPACITIES_WEAKRATES_HPP
