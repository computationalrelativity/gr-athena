#ifndef M1_OPACITIES_WEAKRATES_HPP
#define M1_OPACITIES_WEAKRATES_HPP

// Athena++ classes headers
#include "../../../athena.hpp"
#include "../../m1.hpp"
#include "../../../hydro/hydro.hpp"

// Weakrates header
#include "weak_rates.hpp"
#include <cmath>
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
        ierr_invalid = 1;
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

    return iem || iab || isc;
  }

  inline int ComputeEquilibriumDensities(
    const int k, const int j, const int i,
    const int dt,
    const Real rho, const Real T, const Real Y_e,
    Real & tau,
    Real & T_star, Real & Y_e_star,
    Real* kap_a_n, Real* kap_a_e,
    Real* kap_s_n, Real* kap_s_e,
    Real* dens_n, Real* dens_e,
    bool ignore_current_data
  )
  {
    int ierr_we = 0;
    int ierr_nd = 0;

    Real dens_n_trap[3];
    Real dens_e_trap[3];
    Real dens_n_thin[3];
    Real dens_e_thin[3];

    // Time-scale for equilibrium regime
    tau = std::min(
      std::sqrt(kap_a_e[NUE] * (kap_a_e[NUE] + kap_s_e[NUE])),
      std::sqrt(kap_a_e[NUA] * (kap_a_e[NUA] + kap_s_e[NUA]))
    ) * dt;

    // Calculate equilibrium blackbody functions with trapped neutrinos
    if (opacity_tau_trap >= 0.0 &&
        tau > opacity_tau_trap)
    {
      // Ensure evolution method delegated based on here detected equilibrium
      const static int ix_g = 0;
      typedef M1::evolution_strategy::opt_solution_regime osr_r;
      typedef M1::evolution_strategy::opt_source_treatment ost_r;
      AthenaArray<osr_r> & sol_r = pm1->ev_strat.masks.solution_regime;
      AthenaArray<ost_r> & src_r = pm1->ev_strat.masks.source_treatment;

      for (int ix_s=0; ix_s<3; ++ix_s)
      {
        sol_r(ix_g,ix_s,k,j,i) = osr_r::equilibrium;
        src_r(ix_g,ix_s,k,j,i) = ost_r::set_zero;
      }

      // set thick limit
      if (revert_thick_limit_equilibrium)
      {
        for (int ix_s=0; ix_s<3; ++ix_s)
        {
          pm1->lab_aux.sc_chi(ix_g,ix_s)(k,j,i) = ONE_3RD;
          pm1->lab_aux.sc_xi(ix_g,ix_s)(k,j,i) = 0;
        }
      }

      // ----------------------------------------------------------------------
      if (ignore_current_data)
      {
        for (int s_idx = 0; s_idx < N_SPCS; ++s_idx)
        {
          dens_n[s_idx] = 0;
          dens_e[s_idx] = 0;
        }
      }
      else
      {
        Real invsdetg = pm1->geom.sc_oo_sqrt_det_g(k, j, i);

        // FF number density
        dens_n[0] = pm1->rad.sc_n(0, 0)(k, j, i) * invsdetg;
        dens_n[1] = pm1->rad.sc_n(0, 1)(k, j, i) * invsdetg;
        dens_n[2] = pm1->rad.sc_n(0, 2)(k, j, i) * invsdetg;

        // FF energy density
        dens_e[0] = pm1->rad.sc_J(0, 0)(k, j, i) * invsdetg;
        dens_e[1] = pm1->rad.sc_J(0, 1)(k, j, i) * invsdetg;
        dens_e[2] = pm1->rad.sc_J(0, 2)(k, j, i) * invsdetg;
      }

      // Calculate equilibriated state
      ierr_we = pmy_weakrates->WeakEquilibrium(rho, T, Y_e,
                                               dens_n[0],
                                               dens_n[1],
                                               dens_n[2],
                                               dens_e[0],
                                               dens_e[1],
                                               dens_e[2],
                                               T_star, Y_e_star,
                                               dens_n_trap[0],
                                               dens_n_trap[1],
                                               dens_n_trap[2],
                                               dens_e_trap[0],
                                               dens_e_trap[1],
                                               dens_e_trap[2]);
      // ----------------------------------------------------------------------
    }

    ierr_nd = pmy_weakrates->NeutrinoDensity(
        rho, T, Y_e,
        dens_n_thin[0], dens_n_thin[1], dens_n_thin[2],
        dens_e_thin[0], dens_e_thin[1], dens_e_thin[2]
    );

    // Set the black body function
    if (opacity_tau_trap < 0 || tau <= opacity_tau_trap)
    {
      dens_n[0] = dens_n_thin[0];
      dens_n[1] = dens_n_thin[1];
      dens_n[2] = dens_n_thin[2];

      dens_e[0] = dens_e_thin[0];
      dens_e[1] = dens_e_thin[1];
      dens_e[2] = dens_e_thin[2];
    }
    else if (tau > opacity_tau_trap + opacity_tau_delta)
    {
      dens_n[0] = dens_n_trap[0];
      dens_n[1] = dens_n_trap[1];
      dens_n[2] = dens_n_trap[2];

      dens_e[0] = dens_e_trap[0];
      dens_e[1] = dens_e_trap[1];
      dens_e[2] = dens_e_trap[2];
    }
    else
    {
      Real lam = (tau - opacity_tau_trap) / opacity_tau_delta;

      dens_n[0] = lam * dens_n_trap[0] + (1 - lam) * dens_n_thin[0];
      dens_n[1] = lam * dens_n_trap[1] + (1 - lam) * dens_n_thin[1];
      dens_n[2] = lam * dens_n_trap[2] + (1 - lam) * dens_n_thin[2];

      dens_e[0] = lam * dens_e_trap[0] + (1 - lam) * dens_e_thin[0];
      dens_e[1] = lam * dens_e_trap[1] + (1 - lam) * dens_e_thin[1];
      dens_e[2] = lam * dens_e_trap[2] + (1 - lam) * dens_e_thin[2];
    }

    // ----------------------------------------------------------------------
    Real val = 0;
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      val += dens_n_trap[ix_s] + dens_e_trap[ix_s];
    }

    return (ierr_we || ierr_nd || !std::isfinite(val));
  }

  inline void ApplyOpacityCorrections(
    int k, int j, int i,
    const Real* dens_n,  const Real* dens_e,
    const Real* kap_a_n, const Real* kap_a_e,
    const Real* kap_s_n, const Real* kap_s_e,
    const Real* eta_n,   const Real* eta_e
    )
  {
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
  }

  inline void PrintOpacityDiagnostics(
    const std::string & msg,
    int k, int j, int i,
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
      std::cout << " [" << msg << "]";
    }
    std::cout << " at (i,j,k)=(" << i << "," << j << "," << k << ")\n";

    std::cout << "Physical conditions:\n"
              << "  rho   = " << rho << "\n"
              << "  T     = " << T << "\n"
              << "  Y_e   = " << Y_e << "\n"
              << "  tau   = " << tau << "\n"
              << "  T*    = " << T_star << "\n"
              << "  Y_e*  = " << Y_e_star << "\n";

    std::string regime;
    if (opacity_tau_trap < 0 || tau <= opacity_tau_trap) {
      regime = "thin";
    } else if (tau > opacity_tau_trap + opacity_tau_delta) {
      regime = "trapped";
    } else {
      regime = "interpolated";
    }

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
            "ComputeEquilibriumDensities failure",
            k, j, i, rho, T, Y_e, tau, T_star, Y_e_star,
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
                "CalculateOpacityCoefficients (averaged) failure",
                k, j, i, rho, T, Y_e, tau, T_star, Y_e_star,
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
        ignore_current_data
      );

      if (ierr_we)
      {
        status.num_equi_failures++;

        if (verbose_warn_weak)
        {
          PrintOpacityDiagnostics(
            "ComputeEquilibriumDensities failure",
            k, j, i, rho, T, Y_e, tau, T_star, Y_e_star,
            kap_a_n, kap_a_e, kap_s_n, kap_s_e, eta_n, eta_e,
            dens_n, dens_e);
        }

        if (use_averaging_fix)
        {
          const bool exclude_first_extrema = true;
          GetNearestNeighborAverages(k, j, i, rho, T, Y_e, exclude_first_extrema);

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
            ignore_current_data
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
                "ComputeEquilibriumDensities (averaged) failure",
                k, j, i, rho, T, Y_e, tau, T_star, Y_e_star,
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
          ignore_current_data
        );

        if (ierr_we)
        {
          if (verbose_warn_weak)
          {
            PrintOpacityDiagnostics(
              "ComputeEquilibriumDensities - ignored neutrino data - failure",
              k, j, i, rho, T, Y_e, tau, T_star, Y_e_star,
              kap_a_n, kap_a_e, kap_s_n, kap_s_e, eta_n, eta_e,
              dens_n, dens_e);
          }
        }
        else
        {
          status.num_equi_ignored++;
        }
      }

      if (propagate_hydro_equilibrium)
      {
        pm1->hydro.sc_w_Ye(k, j, i) = Y_e_star;
        pm1->hydro.sc_T(k,j,i) = T_star;
      }

      ApplyOpacityCorrections(
        k, j, i,
        dens_n, dens_e,
        kap_a_n, kap_a_e,
        kap_s_n, kap_s_e,
        eta_n, eta_e
      );

      if (validate_opacities)
      {
        int ierr_vrm = ValidateRadMatQuantities(k,j,i);
        if (ierr_vrm && verbose_warn_weak)
        {
          PrintOpacityDiagnostics(
            "ValidateRadMatQuantities failure",
            k, j, i, rho, T, Y_e, tau, T_star, Y_e_star,
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

  // Options for controlling weakrates opacities
  Real opacity_tau_trap;
  Real opacity_tau_delta;
  Real opacity_corr_fac_max;

  bool verbose_warn_weak;
  bool validate_opacities;
  bool zero_invalid_radmat;
  bool use_averaging_fix;
};

} // namespace M1::Opacities::WeakRates

#endif //M1_OPACITIES_WEAKRATES_HPP
