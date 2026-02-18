#ifndef M1_OPACITIES_COMMON_RATES_PIPELINE_HPP_
#define M1_OPACITIES_COMMON_RATES_PIPELINE_HPP_

// Shared opacity utilities
#include "utils.hpp"

namespace M1::Opacities::Common
{

// ============================================================================
// RatesPipeline - unified opacity pipeline shared by all rate backends.
//
// Species convention (3-species M1 transport):
//   NUE = 0 (electron neutrino)
//   NUA = 1 (electron anti-neutrino)
//   NUX = 2 (heavy-lepton neutrinos - TOTAL for all 4 species)
//
// Units: all quantities are in code_units (GeometricSolar) unless noted.
//   Opacities   (kap_a, kap_a_0, kap_s):  [1/code_length]
//   Emissivities (eta, eta_0):             [1/code_length *
//   code_energy_density] Densities    (eql.sc_n, eql.sc_J):     [densitized
//   code_units] (sqrt_det_g * n)
//
// Pipeline stages:
//   1. CalculateOpacityCoefficients  - raw opacities from the backend
//   2. ComputeEquilibriumDensities   - thin/trapped/interpolated eql densities
//   3. OpacityKirchoffCorrected      - Kirchhoff enforcement (no correction)
//   4. NN density corrections        - optional median/extrema fixes
//   5. FlagEquilibrium (raw)         - optional equilibrium mask
//   (pre-correction)
//   6. OpacityKirchoffCorrected      - Kirchhoff + energy-ratio correction
//   7. NN opacity corrections        - optional median/extrema fixes
//   8. FlagEquilibrium (corrected)   - optional equilibrium mask
//   (post-correction)
//   9. ValidateRadMatQuantities      - sanity checks on final values
//
// The Backend type must expose the following public interface:
//   - M1 *pm1
//   - Common::OpacityUtils opu
//   - int CalculateOpacityCoefficients(int k, int j, int i,
//                                      Real rho, Real T, Real Y_e)
//   - int ComputeEquilibriumDensities(int k, int j, int i,
//                                     Real dt, Real rho, Real T, Real Y_e,
//                                     Real tau,
//                                     OpacityUtils::cmp_eql_dens_ini ini,
//                                     bool using_averaging_fix)
// ============================================================================
template <typename Backend>
inline int RatesPipeline(Real const dt, AA& u, Backend& b)
{
  using cstate           = OpacityUtils::cstate;
  using cmp_eql_dens_ini = OpacityUtils::cmp_eql_dens_ini;

  auto& opu = b.opu;
  auto* pm1 = b.pm1;

  constexpr int NUE = OpacityUtils::NUE;
  constexpr int NUA = OpacityUtils::NUA;

  // Identify points where we need to perform a calculation =================
  AthenaArray<cstate> calc_state(pm1->mbi.nn3, pm1->mbi.nn2, pm1->mbi.nn1);
  calc_state.Fill(cstate::need);

  AthenaArray<int> ierr(pm1->mbi.nn3, pm1->mbi.nn2, pm1->mbi.nn1);
  ierr.Fill(0);

  M1_FLOOP3(k, j, i)
  {
#if defined(Z4C_WITH_HYDRO_ENABLED)
    bool ahf_cut = (pm1->opt_excision.m1_disable_ahf_opac &&
                    (pm1->pmy_block->phydro->excision_mask(k, j, i) < 1));

    if (!(pm1->MaskGet(k, j, i) && opu.AboveCutoff(k, j, i)) || ahf_cut)
#else
    if (!(pm1->MaskGet(k, j, i) && opu.AboveCutoff(k, j, i)))
#endif
    {
      calc_state(k, j, i) = cstate::none;
      opu.SetZeroRadMatAtPoint(k, j, i);
      continue;
    }
  }
  // ========================================================================

  // compute (uncorrected) opacities ========================================
  M1_FLOOP3(k, j, i)
  if (calc_state(k, j, i) == cstate::need)
  {
    Real rho, T, Y_e;
    opu.GetHydro(k, j, i, rho, T, Y_e);

    ierr(k, j, i) = b.CalculateOpacityCoefficients(k, j, i, rho, T, Y_e);

    if (ierr(k, j, i))
    {
      calc_state(k, j, i) = cstate::failed;
    }
  }

  // deal with failures -----------------------------------------------------
  M1_FLOOP3(k, j, i)
  if (calc_state(k, j, i) == cstate::failed)
  {
    opu.PrintOpacityDiagnostics(
      "CalculateOpacityCoefficients - general failure",
      ierr(k, j, i),
      dt,
      k,
      j,
      i);
    assert(false);

    calc_state(k, j, i) = cstate::need;
    ierr(k, j, i)       = 0;
  }
  // ========================================================================

  // compute species densities ==============================================
  {
    int num_failing = 0;

    M1_FLOOP3(k, j, i)
    if (calc_state(k, j, i) == cstate::need)
    {
      Real rho, T, Y_e;
      opu.GetHydro(k, j, i, rho, T, Y_e);

      const Real tau = std::min(opu.CalculateTau(NUE, k, j, i),
                                opu.CalculateTau(NUA, k, j, i));

      ierr(k, j, i) = b.ComputeEquilibriumDensities(
        k, j, i, dt, rho, T, Y_e, tau, cmp_eql_dens_ini::current_sv, false);

      if (ierr(k, j, i))
      {
        calc_state(k, j, i) = cstate::failed;
        num_failing++;

        if (opu.opt.verbose_warn_weak)
        {
          opu.PrintOpacityDiagnostics("ComputeEquilibriumDensities: failing",
                                      ierr(k, j, i),
                                      dt,
                                      k,
                                      j,
                                      i);
        }
      }
    }

    // deal with failures ---------------------------------------------------
    if (num_failing > 0)
    {
      std::vector<cmp_eql_dens_ini> ced_fallback;

      ced_fallback.push_back(cmp_eql_dens_ini::prev_eql);
      if (opu.opt.equilibrium_fallback_thin)
      {
        ced_fallback.push_back(cmp_eql_dens_ini::thin);
      }
      if (opu.opt.equilibrium_fallback_zero)
      {
        ced_fallback.push_back(cmp_eql_dens_ini::zero);
      }

      M1_FLOOP3(k, j, i)
      if (calc_state(k, j, i) == cstate::failed)
      {
        if (num_failing == 0)
          continue;

        Real rho, T, Y_e;
        opu.GetHydro(k, j, i, rho, T, Y_e);

        const Real tau = std::min(opu.CalculateTau(NUE, k, j, i),
                                  opu.CalculateTau(NUA, k, j, i));

        for (const auto& ini_method : ced_fallback)
        {
          ierr(k, j, i) = b.ComputeEquilibriumDensities(
            k, j, i, dt, rho, T, Y_e, tau, ini_method, false);

          if (ierr(k, j, i) && opu.opt.verbose_warn_weak)
          {
            std::string msg =
              ("ComputeEquilibriumDensities [" +
               std::to_string(static_cast<int>(ini_method)) + "]");
            opu.PrintOpacityDiagnostics(msg, ierr(k, j, i), dt, k, j, i);
          }

          if (!ierr(k, j, i))
          {
            calc_state(k, j, i) = cstate::need;
            num_failing--;
            break;
          }
        }
      }
    }

    // NN average fallback for remaining failures ---------------------------
    if (num_failing > 0)
    {
      M1_FLOOP3(k, j, i)
      if (calc_state(k, j, i) == cstate::failed)
      {
        if (num_failing == 0)
          continue;

        const int ix_g = 0;
        int ierr_avg   = 0;

        for (int ix_s = 0; ix_s < opu.N_SPCS; ++ix_s)
        {
          Real avg_n, avg_J;

          ierr_avg = opu.GetNearestNeighborAverageMasked(
            k, j, i, avg_n, pm1->eql.sc_n(ix_g, ix_s).array(), calc_state);

          if (ierr_avg)
          {
            break;
          }

          ierr_avg = opu.GetNearestNeighborAverageMasked(
            k, j, i, avg_J, pm1->eql.sc_J(ix_g, ix_s).array(), calc_state);

          if (ierr_avg)
          {
            break;
          }

          pm1->eql.sc_n(ix_g, ix_s)(k, j, i) = avg_n;
          pm1->eql.sc_J(ix_g, ix_s)(k, j, i) = avg_J;
        }

        if (ierr_avg)
        {
          if (opu.opt.verbose_warn_weak)
          {
            opu.PrintOpacityDiagnostics(
              "ComputeEquilibriumDensities: nn_avg replacement",
              0,
              dt,
              k,
              j,
              i);
          }
        }
        else
        {
          calc_state(k, j, i) = cstate::need;
          ierr(k, j, i)       = 0;
          num_failing--;
        }
      }
    }

    if (num_failing > 0)
    {
      std::printf("ComputeEquilibriumDensities: general failure\n");
      assert(false);
    }
  }
  // ========================================================================

  // enforce Kirchhoff on raw opacities =====================================
  {
    M1_FLOOP3(k, j, i)
    if (calc_state(k, j, i) == cstate::need)
    {
      const bool use_correction = false;
      const Real tau            = std::min(opu.CalculateTau(NUE, k, j, i),
                                opu.CalculateTau(NUA, k, j, i));
      opu.OpacityKirchoffCorrected(dt, tau, k, j, i, use_correction);
    }
  }
  // ========================================================================

  // NN density corrections =================================================
  if (opu.opt.fix_nn_densities)
  {
    std::vector<GroupSpeciesContainer<AT_C_sca>*> data;
    data.push_back(&pm1->eql.sc_n);
    data.push_back(&pm1->eql.sc_J);

    for (int p = 0; p < opu.opt.fix_num_passes; ++p)
      M1_FLOOP3(k, j, i)
    if (calc_state(k, j, i) == cstate::need)
    {
      opu.ApplyValueFixes(data, calc_state, k, j, i);
    }
  }
  // ========================================================================

  // equilibrium flag (raw opacities) =======================================
  if (opu.opt.flag_equilibrium_raw)
  {
    M1_FLOOP3(k, j, i)
    if (calc_state(k, j, i) == cstate::need)
    {
      Real rho, T, Y_e;
      opu.GetHydro(k, j, i, rho, T, Y_e);

      opu.FlagEquilibrium(
        dt, rho, T, k, j, i, opu.opt.flag_equilibrium_species);
    }

    opu.FlagEquilibriumSmear(calc_state);
  }
  // ========================================================================

  // correct opacities by energy ratio ======================================
  M1_FLOOP3(k, j, i)
  if (calc_state(k, j, i) == cstate::need)
  {
    const bool use_correction = true;
    const Real tau =
      std::min(opu.CalculateTau(NUE, k, j, i), opu.CalculateTau(NUA, k, j, i));
    opu.OpacityKirchoffCorrected(dt, tau, k, j, i, use_correction);
  }
  // ========================================================================

  // NN opacity corrections =================================================
  if (opu.opt.fix_nn_opacities)
  {
    typedef M1::vars_RadMat RM;
    RM& rm = pm1->radmat;

    std::vector<GroupSpeciesContainer<AT_C_sca>*> data;
    data.push_back(&rm.sc_kap_s);
    data.push_back(&rm.sc_kap_a);
    data.push_back(&rm.sc_eta);

    data.push_back(&rm.sc_kap_a_0);
    data.push_back(&rm.sc_eta_0);

    for (int p = 0; p < opu.opt.fix_num_passes; ++p)
      M1_FLOOP3(k, j, i)
    if (calc_state(k, j, i) == cstate::need)
    {
      opu.ApplyValueFixes(data, calc_state, k, j, i);
    }
  }
  // ========================================================================

  // equilibrium flag =======================================================
  if (opu.opt.flag_equilibrium)
  {
    M1_FLOOP3(k, j, i)
    if (calc_state(k, j, i) == cstate::need)
    {
      Real rho, T, Y_e;
      opu.GetHydro(k, j, i, rho, T, Y_e);

      opu.FlagEquilibrium(
        dt, rho, T, k, j, i, opu.opt.flag_equilibrium_species);
    }

    opu.FlagEquilibriumSmear(calc_state);
  }
  // ========================================================================

  // final sanity checks ====================================================
  M1_FLOOP3(k, j, i)
  if (calc_state(k, j, i) == cstate::need)
  {
    ierr(k, j, i) = opu.ValidateRadMatQuantities(k, j, i);

    if (ierr(k, j, i))
    {
      if (opu.opt.zero_invalid_radmat)
      {
        opu.SetZeroRadMatAtPoint(k, j, i);
      }

      if (opu.opt.verbose_warn_weak)
      {
        opu.PrintOpacityDiagnostics(
          "ValidateRadMatQuantities", ierr(k, j, i), dt, k, j, i);
      }
    }
  }
  // ========================================================================

  return 0;
}

}  // namespace M1::Opacities::Common

#endif  // M1_OPACITIES_COMMON_RATES_PIPELINE_HPP_
