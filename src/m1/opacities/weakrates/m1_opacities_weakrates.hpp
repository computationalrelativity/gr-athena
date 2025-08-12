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

private:
  enum class cstate {none, need, failed};
  enum class cmp_eql_dens_ini {current_sv,
                               prev_eql,
                               thin,
                               zero};


public:
  WeakRates(MeshBlock *pmb, M1 * pm1, ParameterInput *pin) :
    pm1(pm1),
    pmy_mesh(pmb->pmy_mesh),
    pmy_block(pmb),
    pmy_coord(pmy_block->pcoord),
    N_GRPS(pm1->N_GRPS),
    N_SPCS(pm1->N_SPCS),
    wr_dfloor(
      pin->GetOrAddReal("M1_opacities",
                        "wr_dfloor",
                        0.0)
    ),
    wr_equilibrium_fallback_thin(
      pin->GetOrAddBoolean("M1_opacities",
                           "wr_equilibrium_fallback_thin",
                           false)
    ),
    wr_equilibrium_fallback_zero(
      pin->GetOrAddBoolean("M1_opacities",
                           "wr_equilibrium_fallback_zero",
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
    wr_flag_equilibrium_nue_equals_nua(
      pin->GetOrAddBoolean("M1_opacities",
                           "wr_flag_equilibrium_nue_equals_nua",
                           false)
    ),
    wr_flag_equilibrium_no_nux(
      pin->GetOrAddBoolean("M1_opacities",
                           "wr_flag_equilibrium_no_nux",
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

    opacity_tau_trap = pin->GetOrAddReal("M1_opacities", "tau_trap", 1.0);
    opacity_tau_delta = pin->GetOrAddReal("M1_opacities", "tau_delta", 1.0);
    opacity_corr_fac_max = pin->GetOrAddReal("M1_opacities", "max_correction_factor", 1.75);

    verbose_warn_weak =
        pin->GetOrAddBoolean("M1_opacities", "verbose_warn_weak", true);
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

  inline void GetNearestNeighborAverage(int k, int j, int i,
                                        Real &avg,
                                        AA &arr,
                                        bool exclude_first_extrema)
  {
    // Calculate average of nearest neighbors
    avg = 0.0;
    int count = 0;

    // If we need to exclude extrema, we need to track min/max values
    Real min_val = std::numeric_limits<Real>::max();
    Real max_val = -std::numeric_limits<Real>::max();

    // fix offsets for FLOOP
    const int ii_il = (i==M1_IX_IL-M1_FSIZEI) ? 0 : -1;
    const int ii_iu = (i==M1_IX_IU+M1_FSIZEI) ? 0 : +1;

    const int jj_jl = (j==M1_IX_JL-M1_FSIZEJ) ? 0 : -1;
    const int jj_ju = (j==M1_IX_JU+M1_FSIZEJ) ? 0 : +1;

    const int kk_kl = (k==M1_IX_KL-M1_FSIZEK) ? 0 : -1;
    const int kk_ku = (k==M1_IX_KU+M1_FSIZEK) ? 0 : +1;

    for (int kk = kk_kl; kk <= kk_ku; ++kk)
    for (int jj = jj_jl; jj <= jj_ju; ++jj)
    for (int ii = ii_il; ii <= ii_iu; ++ii)
    {
      if (ii == 0 && jj == 0 && kk == 0) continue;

      const Real val = arr(k+kk, j+jj, i+ii);

      avg += val;

      if (exclude_first_extrema)
      {
        // Track min/max values
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
      }

      count++;
    }

    if (exclude_first_extrema)
    {
      avg -= min_val;
      avg -= max_val;

      count -= 2;
    }

    avg /= count;
  }

  inline int GetNearestNeighborAverageMasked(int k, int j, int i,
                                             Real &avg,
                                             AA &arr,
                                             AthenaArray<cstate> &mask)
  {
    // Calculate average of nearest neighbors
    avg = 0.0;
    int count = 0;

    // fix offsets for FLOOP
    const int ii_il = (i==M1_IX_IL-M1_FSIZEI) ? 0 : -1;
    const int ii_iu = (i==M1_IX_IU+M1_FSIZEI) ? 0 : +1;

    const int jj_jl = (j==M1_IX_JL-M1_FSIZEJ) ? 0 : -1;
    const int jj_ju = (j==M1_IX_JU+M1_FSIZEJ) ? 0 : +1;

    const int kk_kl = (k==M1_IX_KL-M1_FSIZEK) ? 0 : -1;
    const int kk_ku = (k==M1_IX_KU+M1_FSIZEK) ? 0 : +1;

    for (int kk = kk_kl; kk <= kk_ku; ++kk)
    for (int jj = jj_jl; jj <= jj_ju; ++jj)
    for (int ii = ii_il; ii <= ii_iu; ++ii)
    {
      if (ii == 0 && jj == 0 && kk == 0) continue;

      // Skip if mask is not cstate::need
      if (mask(k+kk, j+jj, i+ii) != cstate::need)
        continue;

      avg += arr(k+kk, j+jj, i+ii);
      count++;
    }

    if (count > 0)
      avg /= static_cast<Real>(count);
    else
      avg = 0.0; // Or NaN if signaling no neighbors is preferred

    if (count == 0)
      return -1;
    return 0;
  }


  // Tailored average for hydro quantities
  inline void GetNearestNeighborAverageHydro(int k, int j, int i,
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

  // Check whether calculation is required based on e.g. density considerations
  inline bool AboveCutoff(const int k, const int j, const int i)
  {
    bool need_calc = true;

    Real rho = pm1->hydro.sc_w_rho(k,j,i);
    need_calc = need_calc and (rho > pmy_weakrates->rho_min_code_units);

#if FLUID_ENABLED
    Real D = pmy_block->phydro->u(IDN,k,j,i);
    need_calc = need_calc and (pmy_block->phydro->u(IDN,k,j,i) > wr_dfloor);
#endif // FLUID_ENABLED

    return need_calc;
  }

  inline void GetHydroAveraged(const int k, const int j, const int i,
                               Real &rho, Real &T, Real &Y_e)
  {
    const bool exclude_first_extrema = true;
    GetNearestNeighborAverageHydro(k, j, i, rho, T, Y_e,
                                   exclude_first_extrema);
  }

  inline void GetHydro(const int k, const int j, const int i,
                       Real &rho, Real &T, Real &Y_e)
  {
    if (use_averages)
    {
      GetHydroAveraged(k, j, i, rho, T, Y_e);
    }
    else
    {
      rho = pm1->hydro.sc_w_rho(k,j,i);
      T = pm1->hydro.sc_T(k,j,i);
      Y_e = pm1->hydro.sc_w_Ye(k,j,i);
    }
  }

  inline int ValidateRadMatQuantities(int k, int j, int i)
  {
    int ierr_invalid = 0;

    // Check all radiation-matter quantities for the current cell
    const int ix_g = 0;
    for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
    {
      // Check finiteness and non-negativity of all quantities
      if (!std::isfinite(pm1->radmat.sc_kap_a_0(ix_g, ix_s)(k, j, i)) ||
          pm1->radmat.sc_kap_a_0(ix_g, ix_s)(k, j, i) < 0.0 ||
          !std::isfinite(pm1->radmat.sc_kap_a(ix_g, ix_s)(k, j, i)) ||
          pm1->radmat.sc_kap_a(ix_g, ix_s)(k, j, i) < 0.0 ||
          !std::isfinite(pm1->radmat.sc_kap_s(ix_g, ix_s)(k, j, i)) ||
          pm1->radmat.sc_kap_s(ix_g, ix_s)(k, j, i) < 0.0 ||
          !std::isfinite(pm1->radmat.sc_eta_0(ix_g, ix_s)(k, j, i)) ||
          pm1->radmat.sc_eta_0(ix_g, ix_s)(k, j, i) < 0.0 ||
          !std::isfinite(pm1->radmat.sc_eta(ix_g, ix_s)(k, j, i)) ||
          pm1->radmat.sc_eta(ix_g, ix_s)(k, j, i) < 0.0 ||
          !std::isfinite(pm1->radmat.sc_avg_nrg(ix_g, ix_s)(k, j, i)) ||
          pm1->radmat.sc_avg_nrg(ix_g, ix_s)(k, j, i) < 0.0)
      {
        ierr_invalid = WR_RADMAT_INVALID;
      }
    }

    return ierr_invalid;
  }

  inline void SetZeroRadMatAtPoint(int k, int j, int i)
  {
    const int ix_g = 0;
    for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
    {
      pm1->radmat.sc_kap_a_0(ix_g, ix_s)(k, j, i) = 0.0;
      pm1->radmat.sc_kap_a(ix_g, ix_s)(k, j, i) = 0.0;
      pm1->radmat.sc_kap_s(ix_g, ix_s)(k, j, i) = 0.0;
      pm1->radmat.sc_eta_0(ix_g, ix_s)(k, j, i) = 0.0;
      pm1->radmat.sc_eta(ix_g, ix_s)(k, j, i) = 0.0;
      pm1->radmat.sc_avg_nrg(ix_g, ix_s)(k, j, i) = 0.0;
    }
  }

  // --------------------------------------------------------------------------
  // Logic for:
  // 1) Opacity calculation
  // 2) Equilibrium reversion
  // 3) Correction factor application

  inline int CalculateOpacityCoefficients(
    int k, int j, int i,
    Real rho, Real T, Real Y_e
  )
  {
    int iem, iab, isc;

    typedef M1::vars_RadMat RM;
    RM & rm = pm1->radmat;

    // Order in the arrays is: e, a, x
    const int ix_g = 0;
    iem = pmy_weakrates->NeutrinoEmission(
      rho, T, Y_e,
      rm.sc_eta_0(ix_g,NUE)(k,j,i), // eta_n[NUE]
      rm.sc_eta_0(ix_g,NUA)(k,j,i), // eta_n[NUA]
      rm.sc_eta_0(ix_g,NUX)(k,j,i), // eta_n[NUX]
      rm.sc_eta(ix_g,NUE)(k,j,i),   // eta_e[NUE]
      rm.sc_eta(ix_g,NUA)(k,j,i),   // eta_e[NUA]
      rm.sc_eta(ix_g,NUX)(k,j,i)    // eta_e[NUX]
    );

    iab = pmy_weakrates->NeutrinoAbsorptionOpacity(
      rho, T, Y_e,
      rm.sc_kap_a_0(ix_g,NUE)(k,j,i), // kap_a_n[NUE]
      rm.sc_kap_a_0(ix_g,NUA)(k,j,i), // kap_a_n[NUA]
      rm.sc_kap_a_0(ix_g,NUX)(k,j,i), // kap_a_n[NUX]
      rm.sc_kap_a(ix_g,NUE)(k,j,i),   // kap_a_e[NUE]
      rm.sc_kap_a(ix_g,NUA)(k,j,i),   // kap_a_e[NUA]
      rm.sc_kap_a(ix_g,NUX)(k,j,i)    // kap_a_e[NUX]
    );

    Real kap_s_n[N_SPCS];
    isc = pmy_weakrates->NeutrinoScatteringOpacity(
      rho, T, Y_e,
      kap_s_n[NUE], kap_s_n[NUA], kap_s_n[NUX],
      rm.sc_kap_s(ix_g,NUE)(k,j,i),   // kap_s_e[NUE]
      rm.sc_kap_s(ix_g,NUA)(k,j,i),   // kap_s_e[NUA]
      rm.sc_kap_s(ix_g,NUX)(k,j,i)    // kap_s_e[NUX]
    );

    int res = iem || iab || isc;
    return res;
  }

  // time-scale for equilibriation based on internal opacity
  inline Real CalculateTau(const int s_idx,
                           const int k, const int j, const int i)
  {
    // short-circuit if NUX excluded from eql.
    if ((s_idx == NUX) && wr_flag_equilibrium_no_nux)
    {
      return std::numeric_limits<Real>::infinity();
    }

    typedef M1::vars_RadMat RM;
    RM & rm = pm1->radmat;

    const int ix_g = 0;
    const Real kap_s_e = rm.sc_kap_s(ix_g, s_idx)(k, j, i);
    const Real kap_a_e = rm.sc_kap_a(ix_g, s_idx)(k, j, i);

    Real rat = 1.0 / std::sqrt(
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
    const Real tau,
    const cmp_eql_dens_ini initial_guess
  )
  {
    int ierr_we = 0;
    int ierr_nd = 0;

    bool TY_adjusted = false;
    bool calculate_trapped = true;

    Real T_star = T;
    Real Y_e_star = Y_e;

    Real dens_n[3];
    Real dens_e[3];
    Real dens_n_trap[3];
    Real dens_e_trap[3];
    Real dens_n_thin[3];
    Real dens_e_thin[3];

    const bool need_thin = !(
      calculate_trapped &&
      (tau < dt / (opacity_tau_trap + opacity_tau_delta))
    ) || initial_guess == cmp_eql_dens_ini::thin;

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

    calculate_trapped = (
      (opacity_tau_trap >= 0.0) &&
      (tau < dt / opacity_tau_trap) &&
      (rho >= pm1->opt_solver.tra_rho_min)
    );

    // Calculate equilibrium blackbody functions with trapped neutrinos
    if (calculate_trapped)
    {
      // ----------------------------------------------------------------------
      Real invsdetg = pm1->geom.sc_oo_sqrt_det_g(k, j, i);

      if (initial_guess == cmp_eql_dens_ini::current_sv)
      {
        // FF number density
        dens_n[0] = pm1->rad.sc_n(0, 0)(k, j, i) * invsdetg;
        dens_n[1] = pm1->rad.sc_n(0, 1)(k, j, i) * invsdetg;
        dens_n[2] = pm1->rad.sc_n(0, 2)(k, j, i) * invsdetg;

        // FF energy density
        dens_e[0] = pm1->rad.sc_J(0, 0)(k, j, i) * invsdetg;
        dens_e[1] = pm1->rad.sc_J(0, 1)(k, j, i) * invsdetg;
        dens_e[2] = pm1->rad.sc_J(0, 2)(k, j, i) * invsdetg;
      }
      else if (initial_guess == cmp_eql_dens_ini::prev_eql)
      {
        const int ix_g = 0;
        const Real sc_sqrt_det_g = pm1->geom.sc_sqrt_det_g(k,j,i);

        for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
        {
          dens_n[ix_s] = pm1->eql.sc_n(ix_g,ix_s)(k,j,i) * invsdetg;
          dens_e[ix_s] = pm1->eql.sc_J(ix_g,ix_s)(k,j,i) * invsdetg;
        }
      }
      else if (initial_guess == cmp_eql_dens_ini::thin)
      {
        // FF number density
        dens_n[0] = dens_n_thin[0];
        dens_n[1] = dens_n_thin[1];
        dens_n[2] = dens_n_thin[2];

        // FF energy density
        dens_e[0] = dens_e_thin[0];
        dens_e[1] = dens_e_thin[1];
        dens_e[2] = dens_e_thin[2];
      }

      // Calculate equilibriated state
      const bool ignore_current_data = initial_guess == cmp_eql_dens_ini::zero;

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
        // Try averaging fix if applicable
        if (!use_averages && use_averaging_fix)
        {
          Real rho, T, Y_e;
          GetHydroAveraged(k, j, i, rho, T, Y_e);

          return ComputeEquilibriumDensities(
            k, j, i,
            dt,
            rho, T, Y_e,
            tau,
            initial_guess
          );
        }
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

    // finally, retain the equilibrium ----------------------------------------
    const int ix_g = 0;
    const Real sc_sqrt_det_g = pm1->geom.sc_sqrt_det_g(k,j,i);

    for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
    {
      pm1->eql.sc_n(ix_g,ix_s)(k,j,i) = sc_sqrt_det_g * dens_n[ix_s];
      pm1->eql.sc_J(ix_g,ix_s)(k,j,i) = sc_sqrt_det_g * dens_e[ix_s];
    }

    return 0;
  }

  inline void ApplyOpacityCorrections(int k, int j, int i)
  {
    // numerical shenanigans --------------------------------------------------
    // BD: TODO- probably better shifted to numerical utils.
    auto compute_rel_eps = [](Real val) -> Real
    {
      constexpr Real machine_eps = std::numeric_limits<Real>::epsilon();
      constexpr Real base_scale = 10.0;
      return std::max(base_scale * machine_eps * val,
                      std::pow(machine_eps, 0.75));
    };

    auto normalize_denominator = [&](Real val) -> Real
    {
      return val;
      Real rel_eps = compute_rel_eps(val);
      return (val < rel_eps) ? rel_eps : val;
    };
    // ------------------------------------------------------------------------

    // Do computations, check validity afterwards
    typedef M1::vars_RadMat RM;
    typedef M1::vars_Rad R;
    RM & rm = pm1->radmat;
    R  & r  = pm1->rad;

    Real corr_fac[3];

    const Real W = pm1->fidu.sc_W(k,j,i);
    const Real oo_sc_sqrt_det_g = OO(pm1->geom.sc_sqrt_det_g(k,j,i));

    const Real ix_g = 0;  // only 1 group
    for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
    {
      const Real sc_n = pm1->eql.sc_n(ix_g,ix_s)(k,j,i) * oo_sc_sqrt_det_g;
      const Real sc_J = pm1->eql.sc_J(ix_g,ix_s)(k,j,i) * oo_sc_sqrt_det_g;

      // equilibrium and incoming energies
      Real avg_nrg_eql = sc_J / sc_n;
      avg_nrg_eql = (!std::isfinite(avg_nrg_eql) || avg_nrg_eql < 0.0)
        ? 0.0
        : avg_nrg_eql;

      // Compute directly  ----------------------------------------------------
      AT_C_sca & sc_E = pm1->lab.sc_E(ix_g,ix_s);
      AT_N_vec & sp_F_d = pm1->lab.sp_F_d(ix_g,ix_s);
      AT_C_sca & sc_nG = pm1->lab.sc_nG(ix_g,ix_s);

      Real dotFv (0.0);
      for (int a=0; a<N; ++a)
      {
        dotFv += sp_F_d(a,k,j,i) * pm1->fidu.sp_v_u(a,k,j,i);
      }
      Real avg_nrg_inc = std::max(
        W / sc_nG(k,j,i) * (sc_E(k,j,i) - dotFv),
        0.0
      );
      // ----------------------------------------------------------------------

      // Prepare correction factors
      corr_fac[ix_s] = avg_nrg_inc / avg_nrg_eql;

      if (!std::isfinite(corr_fac[ix_s]))
        corr_fac[ix_s] = 1.0;

      rm.sc_avg_nrg(ix_g, ix_s)(k, j, i) = avg_nrg_eql;

      if (correction_adjust_upward)
      {
        corr_fac[ix_s] = std::max(
          std::min(corr_fac[ix_s], opacity_corr_fac_max), 1.0
        );
      }
      else
      {
        corr_fac[ix_s] =
            std::max(1.0 / opacity_corr_fac_max,
                     std::min(corr_fac[ix_s], opacity_corr_fac_max));
      }

      // need to correct by square
      corr_fac[ix_s] = SQR(corr_fac[ix_s]);
    }

    // apply the correction factors
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      const Real cf = corr_fac[ix_s];

      // undensitize
      const Real n = pm1->eql.sc_n(ix_g,ix_s)(k,j,i) * oo_sc_sqrt_det_g;
      const Real J = pm1->eql.sc_J(ix_g,ix_s)(k,j,i) * oo_sc_sqrt_det_g;

      Real & kap_s   = rm.sc_kap_s(  ix_g,ix_s)(k,j,i);
      Real & kap_a   = rm.sc_kap_a(  ix_g,ix_s)(k,j,i);
      Real & kap_a_0 = rm.sc_kap_a_0(ix_g,ix_s)(k,j,i);
      Real & eta   = rm.sc_eta(  ix_g,ix_s)(k,j,i);
      Real & eta_0 = rm.sc_eta_0(ix_g,ix_s)(k,j,i);

      // Energy scattering
      kap_s = cf * kap_s;

      // Enforce Kirchhoff's law:
      // For electron lepton neutrinos we change the opacity
      // For heavy lepton neutrinos we change the emissivity

      if (ix_s == NUX)
      {
        // Heavy lepton
        eta_0 = cf * eta_0;
        eta = cf * eta;

        kap_a_0 = (n > 0) ? eta_0 / n : 0.0;
        kap_a   = (J > 0) ? eta / J : 0.0;

        kap_a_0 = (!std::isfinite(kap_a_0) || kap_a_0 < 0)
          ? 0.0 : kap_a_0;
        kap_a = (!std::isfinite(kap_a) || kap_a < 0)
          ? 0.0 : kap_a;
      }
      else
      {
        // Electron neutrinos & Electron anti-neutrinos
        kap_a_0 = cf * kap_a_0;
        kap_a   = cf * kap_a;

        eta_0 = kap_a_0 * n;
        eta   = kap_a * J;
      }
    }
  }

  // set solution mask for equilibrium points;
  inline void FlagEquilibrium(
    const Real dt,
    const Real rho,
    const int k, const int j, const int i,
    const bool per_species)
  {
    const int ix_g = 0;  // fix 1 group
    const Real dt_fac = dt * wr_flag_equilibrium_dt_factor;

    Real tau_[N_SPCS];
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      tau_[ix_s] = CalculateTau(ix_s,k,j,i);
    }

    if (!per_species)
    {
      // N.B. this will equilibriate nu_x with this and can be spurious!
      const Real tau = std::min(tau_[NUE], tau_[NUA]);
      for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
      {
          tau_[ix_s] = tau;
      }
    }
    else
    {
      if (wr_flag_equilibrium_nue_equals_nua)
      {
        const Real tau = std::min(tau_[NUE], tau_[NUA]);
        tau_[NUE] = tau;
        tau_[NUA] = tau;
      }
    }

    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      const Real tau = tau_[ix_s];
      bool flag_eql = ((opacity_tau_trap >= 0.0) &&
                      (tau < dt_fac / opacity_tau_trap) &&
                      (rho >= pm1->opt_solver.eql_rho_min));

      M1::M1::t_sln_r sln_flag = (flag_eql)
        ? M1::M1::t_sln_r::equilibrium
        : M1::M1::t_sln_r::noop;

      pm1->SetMaskSolutionRegime(sln_flag,ix_g,ix_s,k,j,i);
    }
  }

  inline void PrintOpacityDiagnostics(
    const std::string & msg, const int error_code,
    Real dt, int k, int j, int i)
  {
    // Prepare tau and hydro
    Real rho, T, Y_e;
    GetHydro(k, j, i, rho, T, Y_e);
    Real tau = std::min(
      CalculateTau(NUE,k,j,i), CalculateTau(NUA,k,j,i)
    );

    std::cout << "M1: Opacity diagnostics";
    if (!msg.empty())
    {
      std::cout << " [" << msg << "; error_code: " << error_code << "]";
    }
    std::cout << " at (i,j,k)=(" << i << "," << j << "," << k << ")\n";
    std::cout << std::setprecision(14);
    std::cout << "Physical conditions:\n"
              << "  Real  dt    = " << dt << ";\n"
              << "  rho         = " << rho << ";\n"
              << "  T           = " << T << ";\n"
              << "  Y_e         = " << Y_e << ";\n"
              << "  tau         = " << tau << ";\n"
              << "  pm1->rad.sc_n(0,NUE)(k,j,i) = "
              << pm1->rad.sc_n(0, NUE)(k, j, i) << ";\n"
              << "  pm1->rad.sc_n(0,NUA)(k,j,i) = "
              << pm1->rad.sc_n(0, NUA)(k, j, i) << ";\n"
              << "  pm1->rad.sc_n(0,NUX)(k,j,i) = "
              << pm1->rad.sc_n(0, NUX)(k, j, i) << ";\n"
              << "  pm1->rad.sc_J(0,NUE)(k,j,i) = "
              << pm1->rad.sc_J(0, NUE)(k, j, i) << ";\n"
              << "  pm1->rad.sc_J(0,NUA)(k,j,i) = "
              << pm1->rad.sc_J(0, NUA)(k, j, i) << ";\n"
              << "  pm1->rad.sc_J(0,NUX)(k,j,i) = "
              << pm1->rad.sc_J(0, NUX)(k, j, i) << ";\n"
              << "  pm1->geom.sc_oo_sqrt_det_g(k, j, i) = "
              << pm1->geom.sc_oo_sqrt_det_g(k, j, i) << ";\n";

    std::string regime;

    // infer regime using same logic as ComputeEquilibriumDensities -----------

    // BD: would be better to unify this detection logic more cleanly...

    const bool calculate_trapped = (
      (opacity_tau_trap >= 0.0) &&
      (tau < dt / opacity_tau_trap) &&
      (rho >= pm1->opt_solver.tra_rho_min)
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

    Real avg_rho = 0.0;
    Real avg_T = 0.0;
    Real avg_Y_e = 0.0;
    GetNearestNeighborAverageHydro(k, j, i, avg_rho, avg_T, avg_Y_e, false);

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

      // Per-species [from corrected]
      const Real tau_kapeff = CalculateTau(s_idx, k, j, i);
      const bool eql_flag = tau_kapeff < dt * wr_flag_equilibrium_dt_factor;

      std::cout << "  " << species << ":\n"
                << "    kap_a_0    = " << pm1->radmat.sc_kap_a_0(0, s_idx)(k, j, i) << "\n"
                << "    kap_a      = " << pm1->radmat.sc_kap_a(0, s_idx)(k, j, i) << "\n"
                << "    kap_s      = " << pm1->radmat.sc_kap_s(0, s_idx)(k, j, i) << "\n"
                << "    eta_0      = " << pm1->radmat.sc_eta_0(0, s_idx)(k, j, i) << "\n"
                << "    eta        = " << pm1->radmat.sc_eta(0, s_idx)(k, j, i) << "\n"
                << "    avg_nrg    = " << pm1->radmat.sc_avg_nrg(0, s_idx)(k, j, i) << "\n"
                << "    tau_kapeff = " << tau_kapeff << "\n"
                << "    eql_flag   = " << eql_flag << "\n";
    }

    std::cout << std::endl;
  }

  // Do calculation here ------------------------------------------------------
  inline int CalculateOpacityWeakRates(Real const dt, AA &u)
  {
    auto& status = pm1->ev_strat.status;

    // Identify points where we need to perform a calculation =================
    AthenaArray<cstate> calc_state(pm1->mbi.nn3,
                                   pm1->mbi.nn2,
                                   pm1->mbi.nn1);
    calc_state.Fill(cstate::need);

    AthenaArray<int> ierr(pm1->mbi.nn3, pm1->mbi.nn2, pm1->mbi.nn1);
    ierr.Fill(0);

    M1_FLOOP3(k, j, i)
    {
      if (!(pm1->MaskGet(k, j, i) && AboveCutoff(k,j,i)))
      {
        calc_state(k,j,i) = cstate::none;
        SetZeroRadMatAtPoint(k, j, i);
        continue;
      }
    }
    // ========================================================================

    // compute (uncorrected) opacities ========================================
    M1_FLOOP3(k, j, i)
    if (calc_state(k,j,i) == cstate::need)
    {
      Real rho, T, Y_e;
      GetHydro(k, j, i, rho, T, Y_e);
      ierr(k,j,i) = CalculateOpacityCoefficients(k,j,i,rho,T,Y_e);

      if (ierr(k,j,i))
      {
        calc_state(k,j,i) = cstate::failed;
      }
    }

    // deal with failures -----------------------------------------------------
    M1_FLOOP3(k, j, i)
    if (calc_state(k,j,i) == cstate::failed)
    {
      // BD: TODO- if needed extend (could switch with averages etc)
      PrintOpacityDiagnostics(
        "CalculateOpacityCoefficients - general failure",
        ierr(k,j,i), dt, k, j, i
      );
      assert(false);

      // assume all is well - reset to "need" for next step of calculation
      calc_state(k,j,i) = cstate::need;
      ierr(k,j,i) = 0;
    }
    // ========================================================================

    // compute species densities ==============================================
    {
      int num_failing = 0;

      M1_FLOOP3(k, j, i)
      if (calc_state(k,j,i) == cstate::need)
      {
        Real rho, T, Y_e;
        GetHydro(k, j, i, rho, T, Y_e);

        const Real tau = std::min(
          CalculateTau(NUE,k,j,i), CalculateTau(NUA,k,j,i)
        );

        ierr(k,j,i) = ComputeEquilibriumDensities(
          k, j, i,
          dt,
          rho, T, Y_e,
          tau,
          cmp_eql_dens_ini::current_sv
        );

        // BD: TODO- DEBUG
        // if ((rho > 1e-7) && (k == 7) && (j == 6) && (i == 5))
        //   ierr(k,j,i) = 23;

        if (ierr(k,j,i))
        {
          calc_state(k,j,i) = cstate::failed;
          num_failing++;

          if (verbose_warn_weak)
          {
            PrintOpacityDiagnostics("ComputeEquilibriumDensities: failing",
                                    ierr(k,j,i), dt, k, j, i);
          }
        }
      }

      // (optional) smoothing -------------------------------------------------
      // BD: TODO- if needed

      // deal with failures ---------------------------------------------------
      if (num_failing > 0)
      {
        std::vector<cmp_eql_dens_ini> ced_fallback;

        ced_fallback.push_back(cmp_eql_dens_ini::prev_eql);
        if (wr_equilibrium_fallback_thin)
        {
          ced_fallback.push_back(cmp_eql_dens_ini::thin);
        }
        if (wr_equilibrium_fallback_zero)
        {
          ced_fallback.push_back(cmp_eql_dens_ini::zero);
        }

        M1_FLOOP3(k, j, i)
        if (calc_state(k,j,i) == cstate::failed)
        {
          if (num_failing == 0) continue;

          Real rho, T, Y_e;
          GetHydro(k, j, i, rho, T, Y_e);

          const Real tau = std::min(
            CalculateTau(NUE,k,j,i), CalculateTau(NUA,k,j,i)
          );

          for (const auto& ini_method : ced_fallback)
          {
            ierr(k,j,i) = ComputeEquilibriumDensities(
              k, j, i,
              dt,
              rho, T, Y_e,
              tau,
              ini_method
            );

            // BD: TODO- DEBUG
            // if ((rho > 1e-7) && (k == 7) && (j == 6) && (i == 5))
            //   ierr(k,j,i) = -1;

            if (ierr(k,j,i) && verbose_warn_weak)
            {
              std::string msg = (
                "ComputeEquilibriumDensities [" +
                std::to_string(static_cast<int>(ini_method)) + "]"
              );
              PrintOpacityDiagnostics(msg, ierr(k,j,i),
                                      dt, k, j, i);
            }

            if (!ierr(k,j,i))
            {
              calc_state(k,j,i) = cstate::need;
              num_failing--;
              break;
            }
          }

          /*
          if (ierr(k,j,i))
          {
            // all methods exhausted - fail
            PrintOpacityDiagnostics(
              "ComputeEquilibriumDensities: general failure",
              ierr(k,j,i), dt, k, j, i
            );
            assert(false);
          }
          */

          // calc_state(k,j,i) = cstate::need;
          // ierr(k,j,i) = 0;
        }
      }

      // second pass (replace with averages) ----------------------------------
      // if some (isolated) points remain, we can replace them with nn avg
      if (num_failing > 0)
      {
        M1_FLOOP3(k, j, i)
        if (calc_state(k,j,i) == cstate::failed)
        {
          if (num_failing == 0) continue;

          // Averaging logic to build eql.sc_n, eql.sc_J
          const int ix_g = 0;
          int ierr_avg = 0;

          for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
          {
            Real avg_n, avg_J;

            GetNearestNeighborAverageMasked(
              k, j, i, avg_n, pm1->eql.sc_n(ix_g,ix_s).array(), calc_state
            );

            if (ierr_avg) // no viable nn
            {
              break;
            }

            GetNearestNeighborAverageMasked(
              k, j, i, avg_J, pm1->eql.sc_J(ix_g,ix_s).array(), calc_state
            );

            // This really assumes isolated point of failure to avoid arr.
            // copies needed for a proper, non-overwriting average.
            pm1->eql.sc_n(ix_g,ix_s)(k,j,i) = avg_n;
            pm1->eql.sc_J(ix_g,ix_s)(k,j,i) = avg_J;
          }

          if (ierr_avg)
          {
            if (verbose_warn_weak)
            {
              PrintOpacityDiagnostics(
                "ComputeEquilibriumDensities: nn_avg replacement",
                0, dt, k, j, i
              );
            }
          }
          else
          {
            calc_state(k,j,i) = cstate::need;
            ierr(k,j,i) = 0;
            num_failing--;
          }
        }
      }

      if (num_failing > 0)
      {
        // all methods exhausted - fail
        std::printf("ComputeEquilibriumDensities: general failure\n");
        assert(false);
      }
    }
    // ========================================================================

    // correct opacities ======================================================
    M1_FLOOP3(k, j, i)
    if (calc_state(k,j,i) == cstate::need)
    {
      ApplyOpacityCorrections(k, j, i);
    }
    // ========================================================================

    // equilibrium flag =======================================================
    if (wr_flag_equilibrium)
    {
      M1_FLOOP3(k, j, i)
      if (calc_state(k,j,i) == cstate::need)
      {
        Real rho, T, Y_e;
        GetHydro(k, j, i, rho, T, Y_e);

        FlagEquilibrium(
          dt, rho, k, j, i,
          wr_flag_equilibrium_species
        );
      }
    }
    // ========================================================================

    // final sanity checks ====================================================
    M1_FLOOP3(k, j, i)
    if (calc_state(k,j,i) == cstate::need)
    {
      ierr(k,j,i) = ValidateRadMatQuantities(k,j,i);

      if (ierr(k,j,i))
      {
        if (zero_invalid_radmat)
        {
          SetZeroRadMatAtPoint(k,j,i);
        }

        if (verbose_warn_weak)
        {
          PrintOpacityDiagnostics("ValidateRadMatQuantities", ierr(k,j,i),
                                  dt, k, j, i);
        }
      }
    }
    // ========================================================================

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

  const Real wr_dfloor;
  const bool wr_equilibrium_fallback_thin;
  const bool wr_equilibrium_fallback_zero;
  const bool wr_flag_equilibrium;
  const Real wr_flag_equilibrium_dt_factor;
  const bool wr_flag_equilibrium_species;
  const bool wr_flag_equilibrium_nue_equals_nua;
  const bool wr_flag_equilibrium_no_nux;
  const bool correction_adjust_upward;

  // Options for controlling weakrates opacities
  Real opacity_tau_trap;
  Real opacity_tau_delta;
  Real opacity_corr_fac_max;

  bool verbose_warn_weak;
  bool zero_invalid_radmat;
  bool use_averages;
  bool use_averaging_fix;
};

} // namespace M1::Opacities::WeakRates

#endif //M1_OPACITIES_WEAKRATES_HPP
