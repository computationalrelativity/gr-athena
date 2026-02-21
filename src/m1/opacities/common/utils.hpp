#ifndef M1_OPACITIES_COMMON_UTILS_HPP
#define M1_OPACITIES_COMMON_UTILS_HPP

// Athena++ classes headers
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <string>
#include <vector>

#include "../../../athena.hpp"
#include "../../../coordinates/coordinates.hpp"
#include "../../../eos/eos.hpp"
#include "../../../hydro/hydro.hpp"
#include "../../m1.hpp"
#include "error_codes.hpp"

// ============================================================================
namespace M1::Opacities::Common
{
// ============================================================================

/// No-op callable for the default RecomputeOpacFn template parameter in
/// OpacityUtils::ComputeEquilibriumDensities.  Backends that want to
/// recompute opacities with adjusted (T_star, Y_e_star) pass their own
/// callable instead.
struct NoOpRecompute
{
  int operator()(int /*k*/,
                 int /*j*/,
                 int /*i*/,
                 Real /*rho*/,
                 Real /*T*/,
                 Real /*Y_e*/) const
  {
    return 0;
  }
};

/// Shared utility class for opacity backends (WeakRates, BNSNuRates).
///
/// Holds common configuration parameters and provides identical helper
/// methods used by both backends.  Each backend owns an instance of this
/// class as a member (composition, not inheritance).
class OpacityUtils
{
  // -- public types
  // -----------------------------------------------------------
  public:
  enum class cstate
  {
    none,
    need,
    failed
  };
  enum class cmp_eql_dens_ini
  {
    current_sv,
    prev_eql,
    thin,
    zero
  };
  enum
  {
    NUE = 0,
    NUA = 1,
    NUX = 2
  };

  // -- public data (accessed by owning backend class)
  // -------------------------
  public:
  M1* pm1;
  MeshBlock* pmy_block;
  Coordinates* pmy_coord;

  const int N_GRPS;
  const int N_SPCS;

  // label used in PrintOpacityDiagnostics header
  std::string diagnostics_label;

  // --- all pin-read configuration parameters --------------------------------
  struct Opt
  {
    // floors
    const Real cut_dfloor;
    const Real cut_rho_floor;
    const Real cut_tfloor;

    // equilibrium flags
    const bool flag_equilibrium;
    const bool flag_equilibrium_raw;
    const bool flag_equilibrium_species;
    const Real flag_equilibrium_dt_factor;
    const bool flag_equilibrium_nue_equals_nua;
    const bool flag_equilibrium_no_nux;
    const int flag_equilibrium_nn;

    // trapped-neutrino correction
    const bool correct_trapped;
    const bool correction_adjust_upward;
    const bool correction_uses_fiducial_frame;
    const bool correct_emissivity_nux;
    const bool correct_opacity_nux;

    // regime thresholds
    const Real opacity_tau_trap;
    const Real opacity_tau_delta;
    const Real opacity_corr_fac_max;

    // opacity recompute after equilibrium
    const bool recompute_opacities_trapped;
    const bool recompute_opacities_interpolated;

    // diagnostics / robustness
    const bool verbose_warn_weak;
    const bool zero_invalid_radmat;
    const bool use_averages;
    const bool use_averaging_fix;
    const bool clamp_nonzero;

    // nearest-neighbor nn fixes
    const bool fix_nn_densities;
    const bool fix_nn_opacities;

    // nearest-neighbor fix parameters
    const int fix_num_passes;
    const int fix_num_neighbors;
    const Real fix_fac_median;
    const bool fix_exclude_first_extrema;
    const bool fix_keep_base_point;
    const Real fix_fac_min;
    const Real fix_fac_max;

    // equilibrium fallback
    const bool equilibrium_fallback_thin;
    const bool equilibrium_fallback_zero;

    // ----- EOS / table-limit parameters (weakrates backend) ----------------
    const bool apply_table_limits_internally;
    const bool enforced_limits_fail;
    const bool tabulated_particle_fractions;
    const bool tabulated_degeneracy_parameter;
    const Real min_eql_yl;

    // ----- EOS / table-limit parameters (shared) ---------------------------
    // These are in code units (by default, geometricsolar).
    const Real min_rho;
    const Real max_rho;
    const Real min_t;
    const Real max_t;
    const Real min_ye;
    const Real max_ye;
    const bool limits_from_table;
    const bool min_rho_usefloor;
    const bool min_t_usefloor;

    // ----- Equilibration / trapped density & temperature limits -------------
    // Populated post-construction from pm1->opt_solver; code units
    // (geometricsolar).
    Real eql_rho_min = 0.0;
    Real tra_rho_min = 0.0;
    Real eql_t_min   = 0.0;

    // Constructor: reads all from pin
    explicit Opt(ParameterInput* pin)
        : cut_dfloor(pin->GetOrAddReal("M1_opacities", "cut_dfloor", 0.0)),
          cut_rho_floor(
            pin->GetOrAddReal("M1_opacities", "cut_rho_floor", 0.0)),
          cut_tfloor(pin->GetOrAddReal("M1_opacities", "cut_tfloor", 0.0)),
          flag_equilibrium(
            pin->GetOrAddBoolean("M1_opacities", "flag_equilibrium", false)),
          flag_equilibrium_raw(pin->GetOrAddBoolean("M1_opacities",
                                                    "flag_equilibrium_raw",
                                                    false)),
          flag_equilibrium_species(
            pin->GetOrAddBoolean("M1_opacities",
                                 "flag_equilibrium_species",
                                 false)),
          flag_equilibrium_dt_factor(
            pin->GetOrAddReal("M1_opacities",
                              "flag_equilibrium_dt_factor",
                              1.0)),
          flag_equilibrium_nue_equals_nua(
            pin->GetOrAddBoolean("M1_opacities",
                                 "flag_equilibrium_nue_equals_nua",
                                 false)),
          flag_equilibrium_no_nux(
            pin->GetOrAddBoolean("M1_opacities",
                                 "flag_equilibrium_no_nux",
                                 false)),
          flag_equilibrium_nn(
            pin->GetOrAddInteger("M1_opacities", "flag_equilibrium_nn", 0)),
          correct_trapped(
            pin->GetOrAddBoolean("M1_opacities", "correct_trapped", false)),
          correction_adjust_upward(
            pin->GetOrAddBoolean("M1_opacities",
                                 "correction_adjust_upward",
                                 false)),
          correction_uses_fiducial_frame(
            pin->GetOrAddBoolean("M1_opacities",
                                 "correction_uses_fiducial_frame",
                                 false)),
          correct_emissivity_nux(pin->GetOrAddBoolean("M1_opacities",
                                                      "correct_emissivity_nux",
                                                      false)),
          correct_opacity_nux(pin->GetOrAddBoolean("M1_opacities",
                                                   "correct_opacity_nux",
                                                   false)),
          opacity_tau_trap(pin->GetOrAddReal("M1_opacities", "tau_trap", 1.0)),
          opacity_tau_delta(
            pin->GetOrAddReal("M1_opacities", "tau_delta", 1.0)),
          opacity_corr_fac_max(
            pin->GetOrAddReal("M1_opacities", "max_correction_factor", 1.75)),
          recompute_opacities_trapped(
            pin->GetOrAddBoolean("M1_opacities",
                                 "recompute_opacities_trapped",
                                 false)),
          recompute_opacities_interpolated(
            pin->GetOrAddBoolean("M1_opacities",
                                 "recompute_opacities_interpolated",
                                 false)),
          verbose_warn_weak(
            pin->GetOrAddBoolean("M1_opacities", "verbose_warn_weak", true)),
          zero_invalid_radmat(pin->GetOrAddBoolean("M1_opacities",
                                                   "zero_invalid_radmat",
                                                   false)),
          use_averages(
            pin->GetOrAddBoolean("M1_opacities", "use_averages", false)),
          use_averaging_fix(
            pin->GetOrAddBoolean("M1_opacities", "use_averaging_fix", false)),
          clamp_nonzero(
            pin->GetOrAddBoolean("M1_opacities", "clamp_nonzero", true)),
          fix_nn_densities(
            pin->GetOrAddBoolean("M1_opacities", "fix_nn_densities", false)),
          fix_nn_opacities(
            pin->GetOrAddBoolean("M1_opacities", "fix_nn_opacities", false)),
          fix_num_passes(
            pin->GetOrAddInteger("M1_opacities", "fix_num_passes", 3)),
          fix_num_neighbors(
            pin->GetOrAddInteger("M1_opacities", "fix_num_neighbors", 1)),
          fix_fac_median(
            pin->GetOrAddReal("M1_opacities", "fix_fac_median", 2.0)),
          fix_exclude_first_extrema(
            pin->GetOrAddBoolean("M1_opacities",
                                 "fix_exclude_first_extrema",
                                 true)),
          fix_keep_base_point(pin->GetOrAddBoolean("M1_opacities",
                                                   "fix_keep_base_point",
                                                   false)),
          fix_fac_min(pin->GetOrAddReal("M1_opacities", "fix_fac_min", 0.5)),
          fix_fac_max(pin->GetOrAddReal("M1_opacities", "fix_fac_max", 1.5)),
          equilibrium_fallback_thin(
            pin->GetOrAddBoolean("M1_opacities",
                                 "equilibrium_fallback_thin",
                                 false)),
          equilibrium_fallback_zero(
            pin->GetOrAddBoolean("M1_opacities",
                                 "equilibrium_fallback_zero",
                                 false)),
          // ----- EOS / table-limit parameters ----------
          apply_table_limits_internally(
            pin->GetOrAddBoolean("M1_opacities",
                                 "apply_table_limits_internally",
                                 true)),
          enforced_limits_fail(pin->GetOrAddBoolean("M1_opacities",
                                                    "enforced_limits_fail",
                                                    true)),
          tabulated_particle_fractions(
            pin->GetOrAddBoolean("M1_opacities",
                                 "tabulated_particle_fractions",
                                 true)),
          tabulated_degeneracy_parameter(
            pin->GetOrAddBoolean("M1_opacities",
                                 "tabulated_degeneracy_parameter",
                                 true)),
          min_eql_yl(
            pin->GetOrAddReal("M1_opacities",
                              "min_eql_yl",
                              -std::numeric_limits<Real>::infinity())),
          min_rho(pin->GetOrAddReal("M1_opacities", "min_rho", 0.0)),
          max_rho(pin->GetOrAddReal("M1_opacities",
                                    "max_rho",
                                    std::numeric_limits<Real>::infinity())),
          min_t(pin->GetOrAddReal("M1_opacities", "min_t", 0.0)),
          max_t(pin->GetOrAddReal("M1_opacities",
                                  "max_t",
                                  std::numeric_limits<Real>::infinity())),
          min_ye(pin->GetOrAddReal("M1_opacities", "min_ye", 0.0)),
          max_ye(pin->GetOrAddReal("M1_opacities", "max_ye", 1.0)),
          limits_from_table(
            pin->GetOrAddBoolean("M1_opacities", "limits_from_table", true)),
          min_rho_usefloor(
            pin->GetOrAddBoolean("M1_opacities", "min_rho_usefloor", false)),
          min_t_usefloor(
            pin->GetOrAddBoolean("M1_opacities", "min_t_usefloor", false))
    {
    }
  };

  Opt opt;

  // -- construction
  // -----------------------------------------------------------
  public:
  OpacityUtils(M1* pm1_,
               MeshBlock* pmy_block_,
               ParameterInput* pin,
               const std::string& diagnostics_label_)
      : pm1(pm1_),
        pmy_block(pmy_block_),
        pmy_coord(pmy_block_->pcoord),
        N_GRPS(pm1_->N_GRPS),
        N_SPCS(pm1_->N_SPCS),
        diagnostics_label(diagnostics_label_),
        opt(pin)
  {
    // Populate fields that require pm1 (not available during Opt construction)
    opt.eql_rho_min = pm1_->opt_solver.eql_rho_min;
    opt.tra_rho_min = pm1_->opt_solver.tra_rho_min;
    opt.eql_t_min   = pm1_->opt_solver.eql_t_min;
  }

  // -- methods
  // ----------------------------------------------------------------
  public:
  // -------------------------------------------------------------------------
  // AboveCutoff
  // -------------------------------------------------------------------------
  inline bool AboveCutoff(const int k, const int j, const int i)
  {
    bool need_calc = true;

#if FLUID_ENABLED
    need_calc =
      need_calc && (pmy_block->phydro->u(IDN, k, j, i) > opt.cut_dfloor);
    need_calc =
      need_calc && (pm1->hydro.sc_w_rho(k, j, i) > opt.cut_rho_floor);
    need_calc = need_calc && (pm1->hydro.sc_T(k, j, i) >= opt.cut_tfloor);
#endif  // FLUID_ENABLED

    return need_calc;
  }

  // -------------------------------------------------------------------------
  // GetNearestNeighborAverageMasked
  // -------------------------------------------------------------------------
  inline int GetNearestNeighborAverageMasked(int k,
                                             int j,
                                             int i,
                                             Real& avg,
                                             AA& arr,
                                             AthenaArray<cstate>& mask)
  {
    avg       = 0.0;
    int count = 0;

    const int ii_il = (i == M1_IX_IL - M1_FSIZEI) ? 0 : -1;
    const int ii_iu = (i == M1_IX_IU + M1_FSIZEI) ? 0 : +1;

    const int jj_jl = (j == M1_IX_JL - M1_FSIZEJ) ? 0 : -1;
    const int jj_ju = (j == M1_IX_JU + M1_FSIZEJ) ? 0 : +1;

    const int kk_kl = (k == M1_IX_KL - M1_FSIZEK) ? 0 : -1;
    const int kk_ku = (k == M1_IX_KU + M1_FSIZEK) ? 0 : +1;

    for (int kk = kk_kl; kk <= kk_ku; ++kk)
      for (int jj = jj_jl; jj <= jj_ju; ++jj)
        for (int ii = ii_il; ii <= ii_iu; ++ii)
        {
          if (ii == 0 && jj == 0 && kk == 0)
            continue;

          if (mask(k + kk, j + jj, i + ii) != cstate::need)
            continue;

          avg += arr(k + kk, j + jj, i + ii);
          count++;
        }

    if (count > 0)
      avg /= static_cast<Real>(count);
    else
      avg = 0.0;

    if (count == 0)
      return -1;
    return 0;
  }

  // -------------------------------------------------------------------------
  // GetNearestNeighborAverageHydro
  // -------------------------------------------------------------------------
  inline void GetNearestNeighborAverageHydro(int k,
                                             int j,
                                             int i,
                                             Real& avg_rho,
                                             Real& avg_T,
                                             Real& avg_Y_e,
                                             bool exclude_first_extrema)
  {
    avg_rho   = 0.0;
    avg_T     = 0.0;
    avg_Y_e   = 0.0;
    int count = 0;

    Real min_rho = std::numeric_limits<Real>::max();
    Real max_rho = -std::numeric_limits<Real>::max();
    Real min_T   = std::numeric_limits<Real>::max();
    Real max_T   = -std::numeric_limits<Real>::max();
    Real min_Y_e = std::numeric_limits<Real>::max();
    Real max_Y_e = -std::numeric_limits<Real>::max();

    for (int kk = -1; kk <= 1; ++kk)
      for (int jj = -1; jj <= 1; ++jj)
        for (int ii = -1; ii <= 1; ++ii)
        {
          if (ii == 0 && jj == 0 && kk == 0)
            continue;

          const Real rho = pm1->hydro.sc_w_rho(k + kk, j + jj, i + ii);
          const Real T   = pm1->hydro.sc_T(k + kk, j + jj, i + ii);
          const Real Y_e = pm1->hydro.sc_w_Ye(k + kk, j + jj, i + ii);

          avg_rho += rho;
          avg_T += T;
          avg_Y_e += Y_e;

          if (exclude_first_extrema)
          {
            min_rho = std::min(min_rho, rho);
            max_rho = std::max(max_rho, rho);
            min_T   = std::min(min_T, T);
            max_T   = std::max(max_T, T);
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

  // -------------------------------------------------------------------------
  // GetHydroAveraged
  // -------------------------------------------------------------------------
  inline void GetHydroAveraged(const int k,
                               const int j,
                               const int i,
                               Real& rho,
                               Real& T,
                               Real& Y_e)
  {
    const bool exclude_first_extrema = true;
    GetNearestNeighborAverageHydro(
      k, j, i, rho, T, Y_e, exclude_first_extrema);
  }

  // -------------------------------------------------------------------------
  // GetHydro
  // -------------------------------------------------------------------------
  inline void GetHydro(const int k,
                       const int j,
                       const int i,
                       Real& rho,
                       Real& T,
                       Real& Y_e)
  {
    if (opt.use_averages)
    {
      GetHydroAveraged(k, j, i, rho, T, Y_e);
    }
    else
    {
      rho = pm1->hydro.sc_w_rho(k, j, i);
      T   = pm1->hydro.sc_T(k, j, i);
      Y_e = pm1->hydro.sc_w_Ye(k, j, i);
    }
  }

  // -------------------------------------------------------------------------
  // ValidateRadMatQuantities
  // -------------------------------------------------------------------------
  inline int ValidateRadMatQuantities(int k, int j, int i)
  {
    int ierr_invalid = 0;

    const int ix_g = 0;
    for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
    {
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
        ierr_invalid = RADMAT_INVALID;
      }
    }

    return ierr_invalid;
  }

  // -------------------------------------------------------------------------
  // SetZeroRadMatAtPoint
  // -------------------------------------------------------------------------
  inline void SetZeroRadMatAtPoint(int k, int j, int i)
  {
    const int ix_g = 0;
    for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
    {
      pm1->radmat.sc_kap_a_0(ix_g, ix_s)(k, j, i) = 0.0;
      pm1->radmat.sc_kap_a(ix_g, ix_s)(k, j, i)   = 0.0;
      pm1->radmat.sc_kap_s(ix_g, ix_s)(k, j, i)   = 0.0;
      pm1->radmat.sc_eta_0(ix_g, ix_s)(k, j, i)   = 0.0;
      pm1->radmat.sc_eta(ix_g, ix_s)(k, j, i)     = 0.0;
      pm1->radmat.sc_avg_nrg(ix_g, ix_s)(k, j, i) = 0.0;
    }
  }

  // -------------------------------------------------------------------------
  // CalculateTau
  //   time-scale for equilibriation based on internal opacity
  // -------------------------------------------------------------------------
  inline Real CalculateTau(const int s_idx,
                           const int k,
                           const int j,
                           const int i)
  {
    // short-circuit if NUX excluded from eql.
    if ((s_idx == NUX) && opt.flag_equilibrium_no_nux)
    {
      return std::numeric_limits<Real>::infinity();
    }

    typedef M1::vars_RadMat RM;
    RM& rm = pm1->radmat;

    const int ix_g   = 0;
    const Real kap_s = rm.sc_kap_s(ix_g, s_idx)(k, j, i);
    const Real kap_a = rm.sc_kap_a(ix_g, s_idx)(k, j, i);

    Real rat = 1.0 / std::sqrt(kap_a * (kap_a + kap_s));

    return (std::isfinite(rat)) ? rat : std::numeric_limits<Real>::infinity();
  }

  // -------------------------------------------------------------------------
  // PrintOpacityDiagnostics
  // -------------------------------------------------------------------------
  inline void PrintOpacityDiagnostics(const std::string& msg,
                                      const int error_code,
                                      Real dt,
                                      int k,
                                      int j,
                                      int i)
  {
    Real rho, T, Y_e;
    GetHydro(k, j, i, rho, T, Y_e);
    Real tau =
      std::min(CalculateTau(NUE, k, j, i), CalculateTau(NUA, k, j, i));

    std::cout << "M1: Opacity diagnostics";
    if (!diagnostics_label.empty())
    {
      std::cout << " " << diagnostics_label;
    }
    if (!msg.empty())
    {
      std::cout << " [" << msg << "; error_code: " << error_code << "]";
    }
    std::cout << " at (i,j,k)=(" << i << "," << j << "," << k << ")\n";
    std::cout << std::setprecision(14);
    std::cout
      << "Physical conditions:\n"
      << "  Real  dt    = " << dt << ";\n"
      << "  rho         = " << rho << ";\n"
      << "  T           = " << T << ";\n"
      << "  Y_e         = " << Y_e << ";\n"
      << "  tau         = " << tau << ";\n"
      << "  pm1->rad.sc_n(0,NUE)(k,j,i) = " << pm1->rad.sc_n(0, NUE)(k, j, i)
      << ";\n"
      << "  pm1->rad.sc_n(0,NUA)(k,j,i) = " << pm1->rad.sc_n(0, NUA)(k, j, i)
      << ";\n"
      << "  pm1->rad.sc_n(0,NUX)(k,j,i) = " << pm1->rad.sc_n(0, NUX)(k, j, i)
      << ";\n"
      << "  pm1->rad.sc_J(0,NUE)(k,j,i) = " << pm1->rad.sc_J(0, NUE)(k, j, i)
      << ";\n"
      << "  pm1->rad.sc_J(0,NUA)(k,j,i) = " << pm1->rad.sc_J(0, NUA)(k, j, i)
      << ";\n"
      << "  pm1->rad.sc_J(0,NUX)(k,j,i) = " << pm1->rad.sc_J(0, NUX)(k, j, i)
      << ";\n"
      << "  pm1->eql.sc_n(0,NUE)(k,j,i) = " << pm1->eql.sc_n(0, NUE)(k, j, i)
      << ";\n"
      << "  pm1->eql.sc_n(0,NUA)(k,j,i) = " << pm1->eql.sc_n(0, NUA)(k, j, i)
      << ";\n"
      << "  pm1->eql.sc_n(0,NUX)(k,j,i) = " << pm1->eql.sc_n(0, NUX)(k, j, i)
      << ";\n"
      << "  pm1->eql.sc_J(0,NUE)(k,j,i) = " << pm1->eql.sc_J(0, NUE)(k, j, i)
      << ";\n"
      << "  pm1->eql.sc_J(0,NUA)(k,j,i) = " << pm1->eql.sc_J(0, NUA)(k, j, i)
      << ";\n"
      << "  pm1->eql.sc_J(0,NUX)(k,j,i) = " << pm1->eql.sc_J(0, NUX)(k, j, i)
      << ";\n"
      << "  pm1->geom.sc_oo_sqrt_det_g(k, j, i) = "
      << pm1->geom.sc_oo_sqrt_det_g(k, j, i) << ";\n";

    std::string regime;

    const bool calculate_trapped =
      ((opt.opacity_tau_trap >= 0.0) && (tau < dt / opt.opacity_tau_trap) &&
       (rho >= opt.tra_rho_min));

    if (calculate_trapped)
    {
      if (tau < dt / (opt.opacity_tau_trap + opt.opacity_tau_delta))
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
    std::cout << "Opacity regime: " << regime << "\n";

    Real avg_rho = 0.0;
    Real avg_T   = 0.0;
    Real avg_Y_e = 0.0;
    GetNearestNeighborAverageHydro(k, j, i, avg_rho, avg_T, avg_Y_e, false);

    std::cout << "Neighbor averages (excluding current point):\n"
              << "  avg_rho = " << avg_rho << "\n"
              << "  avg_T   = " << avg_T << "\n"
              << "  avg_Y_e = " << avg_Y_e << "\n";

    std::cout << "Current radmat values:\n";
    for (int s_idx = 0; s_idx < N_SPCS; ++s_idx)
    {
      std::string species;
      switch (s_idx)
      {
        case NUE:
          species = "nu_e ";
          break;
        case NUA:
          species = "nu_a ";
          break;
        case NUX:
          species = "nu_x ";
          break;
      }

      const Real tau_kapeff = CalculateTau(s_idx, k, j, i);
      const bool eql_flag   = tau_kapeff < dt * opt.flag_equilibrium_dt_factor;

      std::cout << "  " << species << ":\n"
                << "    kap_a_0    = "
                << pm1->radmat.sc_kap_a_0(0, s_idx)(k, j, i) << "\n"
                << "    kap_a      = "
                << pm1->radmat.sc_kap_a(0, s_idx)(k, j, i) << "\n"
                << "    kap_s      = "
                << pm1->radmat.sc_kap_s(0, s_idx)(k, j, i) << "\n"
                << "    eta_0      = "
                << pm1->radmat.sc_eta_0(0, s_idx)(k, j, i) << "\n"
                << "    eta        = " << pm1->radmat.sc_eta(0, s_idx)(k, j, i)
                << "\n"
                << "    avg_nrg    = "
                << pm1->radmat.sc_avg_nrg(0, s_idx)(k, j, i) << "\n"
                << "    tau_kapeff = " << tau_kapeff << "\n"
                << "    eql_flag   = " << eql_flag << "\n";
    }

    std::cout << std::endl;
  }

  // -------------------------------------------------------------------------
  // FlagEquilibrium
  //   set solution mask for equilibrium points
  // -------------------------------------------------------------------------
  inline void FlagEquilibrium(const Real dt,
                              const Real rho,
                              const Real T,
                              const int k,
                              const int j,
                              const int i,
                              const bool per_species)
  {
    const int ix_g    = 0;
    const Real dt_fac = dt * opt.flag_equilibrium_dt_factor;

    Real tau_[N_SPCS];
    for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
    {
      tau_[ix_s] = CalculateTau(ix_s, k, j, i);
    }

    if (!per_species)
    {
      const Real tau = std::min(tau_[NUE], tau_[NUA]);
      for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
      {
        tau_[ix_s] = tau;
      }
    }
    else
    {
      if (opt.flag_equilibrium_nue_equals_nua)
      {
        const Real tau = std::min(tau_[NUE], tau_[NUA]);
        tau_[NUE]      = tau;
        tau_[NUA]      = tau;
      }
    }

    for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
    {
      const Real tau = tau_[ix_s];
      bool flag_eql  = ((opt.opacity_tau_trap >= 0.0) &&
                       (tau < dt_fac / opt.opacity_tau_trap) &&
                       (rho >= opt.eql_rho_min) && (T >= opt.eql_t_min));

      M1::M1::t_sln_r sln_flag =
        (flag_eql) ? M1::M1::t_sln_r::equilibrium : M1::M1::t_sln_r::noop;

      pm1->SetMaskSolutionRegime(sln_flag, ix_g, ix_s, k, j, i);
    }
  }

  // -------------------------------------------------------------------------
  // FlagEquilibriumSmear
  //   smear the mask by a number of nearest-neighbours
  // -------------------------------------------------------------------------
  inline void FlagEquilibriumSmear(AthenaArray<cstate>& calc_state)
  {
    const int nn = opt.flag_equilibrium_nn;
    if (nn <= 0)
    {
      return;
    }

    const int ix_g = 0;

    AthenaArray<M1::M1::t_sln_r> sln_r(
      pmy_block->ncells3, pmy_block->ncells2, pmy_block->ncells1);

    for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
    {
      sln_r.Fill(M1::M1::t_sln_r::noop);

      M1_FLOOP3(k, j, i)
      if (calc_state(k, j, i) == cstate::need)
      {
        bool found = false;
        [&]()
        {
          for (int k_ix = -nn; k_ix <= nn; ++k_ix)
            for (int j_ix = -nn; j_ix <= nn; ++j_ix)
              for (int i_ix = -nn; i_ix <= nn; ++i_ix)
              {
                const int ii =
                  std::min(std::max(i + i_ix, 0), pmy_block->ncells1 - 1);
                const int jj =
                  std::min(std::max(j + j_ix, 0), pmy_block->ncells2 - 1);
                const int kk =
                  std::min(std::max(k + k_ix, 0), pmy_block->ncells3 - 1);

                M1::M1::t_sln_r flag_cur =
                  pm1->GetMaskSolutionRegime(ix_g, ix_s, kk, jj, ii);

                if (flag_cur == M1::M1::t_sln_r::equilibrium)
                {
                  sln_r(k, j, i) = flag_cur;
                  found          = true;
                }
                if (found)
                  return;  // exit lambda
              }
        }();
      }

      M1_FLOOP3(k, j, i)
      if ((calc_state(k, j, i) == cstate::need) &&
          (sln_r(k, j, i) == M1::M1::t_sln_r::equilibrium))
      {
        pm1->SetMaskSolutionRegime(
          M1::M1::t_sln_r::equilibrium, ix_g, ix_s, k, j, i);
      }
    }
  }

  // -------------------------------------------------------------------------
  // OpacityKirchoffCorrected
  //   Kirchoff law together with corrections.
  //
  //   Enforces Kirchhoff's law (eta = kap_a * B) and optionally applies
  //   trapped-neutrino energy-ratio corrections.  The correction factor
  //   cf = (avg_nrg_inc / avg_nrg_eql)^2 rescales opacities/emissivities.
  //
  //   NUX convention (3-species M1 transport, NUX index = 2):
  //     NUX = TOTAL for all 4 heavy-lepton species throughout.
  //     The equilibrium densities eql.sc_n/sc_J for NUX are already TOTAL.
  //
  //   Species-dependent Kirchhoff enforcement:
  //     NUE, NUA: correct opacity first, then derive emissivity
  //       kap_a_0 *= cf,  kap_a *= cf
  //       eta_0 = kap_a_0 * n,  eta = kap_a * J
  //     NUX:      correct emissivity first (if enabled), then derive opacity
  //       eta_0 *= cf,  eta *= cf    (only if correct_emissivity_nux)
  //       kap_a_0 = eta_0 / n,  kap_a = eta / J
  //
  //   in:   dt   [code_units]  (timestep)
  //         tau  [code_units]  (effective opacity timescale, min of NUE/NUA)
  //         k,j,i              (grid indices)
  //         use_correction      (whether to apply energy-ratio correction)
  //   out:  modifies pm1->radmat in-place (kap_a_0, kap_a, kap_s, eta_0, eta)
  //         All opacity/emissivity values remain in code_units.
  // -------------------------------------------------------------------------
  inline void OpacityKirchoffCorrected(const Real dt,
                                       const Real tau,
                                       int k,
                                       int j,
                                       int i,
                                       const bool use_correction)
  {
    typedef M1::vars_RadMat RM;
    typedef M1::vars_Rad R;
    RM& rm = pm1->radmat;
    R& r   = pm1->rad;

    Real corr_fac[3];

    const Real W                = pm1->fidu.sc_W(k, j, i);
    const Real oo_sc_sqrt_det_g = OO(pm1->geom.sc_sqrt_det_g(k, j, i));

    const int ix_g = 0;  // only 1 group
    if (use_correction)
    {
      for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
      {
        if ((!opt.correct_trapped) &&
            (tau < dt / (opt.opacity_tau_trap + opt.opacity_tau_delta)))
        {
          corr_fac[ix_s] = 1.0;
          continue;
        }

        // equilibrium and incoming energies
        Real avg_nrg_eql = (pm1->eql.sc_n(ix_g, ix_s)(k, j, i) > 0)
                           ? (pm1->eql.sc_J(ix_g, ix_s)(k, j, i) /
                              pm1->eql.sc_n(ix_g, ix_s)(k, j, i))
                           : 0.0;

        Real avg_nrg_inc;

        if (opt.correction_uses_fiducial_frame)
        {
          avg_nrg_inc = std::max(pm1->rad.sc_J(ix_g, ix_s)(k, j, i) /
                                   pm1->rad.sc_n(ix_g, ix_s)(k, j, i),
                                 0.0);
        }
        else
        {
          AT_C_sca& sc_E   = pm1->lab.sc_E(ix_g, ix_s);
          AT_N_vec& sp_F_d = pm1->lab.sp_F_d(ix_g, ix_s);
          AT_C_sca& sc_nG  = pm1->lab.sc_nG(ix_g, ix_s);

          Real dotFv(0.0);
          for (int a = 0; a < M1_NDIM; ++a)
          {
            dotFv += sp_F_d(a, k, j, i) * pm1->fidu.sp_v_u(a, k, j, i);
          }
          avg_nrg_inc =
            std::max(W / sc_nG(k, j, i) * (sc_E(k, j, i) - dotFv), 0.0);
        }

        // Prepare correction factors
        corr_fac[ix_s] = avg_nrg_inc / avg_nrg_eql;

        if (!std::isfinite(corr_fac[ix_s]))
          corr_fac[ix_s] = 1.0;

        rm.sc_avg_nrg(ix_g, ix_s)(k, j, i) = avg_nrg_eql;

        if (opt.correction_adjust_upward)
        {
          corr_fac[ix_s] =
            std::max(std::min(corr_fac[ix_s], opt.opacity_corr_fac_max), 1.0);
        }
        else
        {
          corr_fac[ix_s] =
            std::max(1.0 / opt.opacity_corr_fac_max,
                     std::min(corr_fac[ix_s], opt.opacity_corr_fac_max));
        }

        // need to correct by square
        corr_fac[ix_s] = SQR(corr_fac[ix_s]);
      }
    }
    else
    {
      for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
      {
        corr_fac[ix_s] = 1.0;
      }
    }

    // apply the correction factors
    for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
    {
      const Real cf = corr_fac[ix_s];

      // undensitize
      const Real n = pm1->eql.sc_n(ix_g, ix_s)(k, j, i) * oo_sc_sqrt_det_g;
      const Real J = pm1->eql.sc_J(ix_g, ix_s)(k, j, i) * oo_sc_sqrt_det_g;

      Real& kap_s   = rm.sc_kap_s(ix_g, ix_s)(k, j, i);
      Real& kap_a   = rm.sc_kap_a(ix_g, ix_s)(k, j, i);
      Real& kap_a_0 = rm.sc_kap_a_0(ix_g, ix_s)(k, j, i);
      Real& eta     = rm.sc_eta(ix_g, ix_s)(k, j, i);
      Real& eta_0   = rm.sc_eta_0(ix_g, ix_s)(k, j, i);

      // Energy scattering
      kap_s = cf * kap_s;

      // Enforce Kirchhoff's law:
      // For electron lepton neutrinos we change the opacity
      // For heavy lepton neutrinos we change the emissivity

      if (ix_s == NUX)
      {
        // Heavy lepton
        if (opt.correct_opacity_nux)
        {
          // correct opacity first, derive emissivity
          // (same direction as NUE/NUA) - ATK style.
          kap_a_0 = cf * kap_a_0;
          kap_a   = cf * kap_a;

          eta_0 = kap_a_0 * n;
          eta   = kap_a * J;
        }
        else
        {
          // existing behavior: correct emissivity first, derive opacity
          if (opt.correct_emissivity_nux)
          {
            eta_0 = cf * eta_0;
            eta   = cf * eta;
          }

          kap_a_0 = (n > 0) ? eta_0 / n : 0.0;
          kap_a   = (J > 0) ? eta / J : 0.0;

          kap_a_0 = (!std::isfinite(kap_a_0) || kap_a_0 < 0) ? 0.0 : kap_a_0;
          kap_a   = (!std::isfinite(kap_a) || kap_a < 0) ? 0.0 : kap_a;
        }
      }
      else
      {
        // Electron neutrinos & Electron anti-neutrinos
        kap_a_0 = cf * kap_a_0;
        kap_a   = cf * kap_a;

        // Guard against negative opacities
        kap_a_0 = (!std::isfinite(kap_a_0) || kap_a_0 < 0) ? 0.0 : kap_a_0;
        kap_a   = (!std::isfinite(kap_a) || kap_a < 0) ? 0.0 : kap_a;

        eta_0 = kap_a_0 * n;
        eta   = kap_a * J;
      }
    }
  }

  // -------------------------------------------------------------------------
  // GetNearestNeighborAverage
  //   average of nearest neighbors with optional extrema exclusion
  //   (uses FLOOP boundary clamping)
  // -------------------------------------------------------------------------
  inline void GetNearestNeighborAverage(int k,
                                        int j,
                                        int i,
                                        Real& avg,
                                        AA& arr,
                                        bool exclude_first_extrema)
  {
    // Calculate average of nearest neighbors
    avg       = 0.0;
    int count = 0;

    // If we need to exclude extrema, we need to track min/max values
    Real min_val = std::numeric_limits<Real>::max();
    Real max_val = -std::numeric_limits<Real>::max();

    // fix offsets for FLOOP
    const int ii_il = (i == M1_IX_IL - M1_FSIZEI) ? 0 : -1;
    const int ii_iu = (i == M1_IX_IU + M1_FSIZEI) ? 0 : +1;

    const int jj_jl = (j == M1_IX_JL - M1_FSIZEJ) ? 0 : -1;
    const int jj_ju = (j == M1_IX_JU + M1_FSIZEJ) ? 0 : +1;

    const int kk_kl = (k == M1_IX_KL - M1_FSIZEK) ? 0 : -1;
    const int kk_ku = (k == M1_IX_KU + M1_FSIZEK) ? 0 : +1;

    for (int kk = kk_kl; kk <= kk_ku; ++kk)
      for (int jj = jj_jl; jj <= jj_ju; ++jj)
        for (int ii = ii_il; ii <= ii_iu; ++ii)
        {
          if (ii == 0 && jj == 0 && kk == 0)
            continue;

          const Real val = arr(k + kk, j + jj, i + ii);

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

  // -------------------------------------------------------------------------
  // ValueWithinNearestNeighbors
  //   check if a value is within bounds of its nearest neighbors
  // -------------------------------------------------------------------------
  inline void ValueWithinNearestNeighbors(
    bool& value_bounded,
    Real& value_average,
    const AA& src,
    const AthenaArray<cstate>& calc_state,
    const int n,
    const int k,
    const int j,
    const int i,
    const int num_neighbors,
    Real fac_min,
    Real fac_max,
    const bool exclude_first_extrema,
    const bool keep_base_point  // exclude base-point from average
  )
  {
    value_bounded = true;
    value_average = 0.0;

    const int nn = num_neighbors;
    int count    = 0;

    Real min_val = std::numeric_limits<Real>::max();
    Real max_val = -std::numeric_limits<Real>::max();

    for (int kk = -nn; kk <= nn; ++kk)
      for (int jj = -nn; jj <= nn; ++jj)
        for (int ii = -nn; ii <= nn; ++ii)
        {
          const int i_ix = ii + i;
          const int j_ix = jj + j;
          const int k_ix = kk + k;

          // ignore central point for extrema
          if ((ii == 0) && (jj == 0) && (kk == 0))
          {
            continue;
          }

          // from FLOOP macro
          if ((i_ix < M1_IX_IL - M1_FSIZEI) ||
              (i_ix > M1_IX_IU + M1_FSIZEI + 1))
            continue;

          if ((j_ix < M1_IX_JL - M1_FSIZEJ) ||
              (j_ix > M1_IX_JU + M1_FSIZEJ + 1))
            continue;

          if ((k_ix < M1_IX_KL - M1_FSIZEK) ||
              (k_ix > M1_IX_KU + M1_FSIZEK + 1))
            continue;

          // check whether data is computed at the point
          bool use_point = calc_state(k_ix, j_ix, i_ix) == cstate::need;

          const Real val = src(n, k_ix, j_ix, i_ix);

          if (use_point)
          {
            // Track min/max values
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);

            value_average += val;
            count++;
          }
        }

    const Real val = src(n, k, j, i);

    if (keep_base_point)
    {
      value_average += val;
      count += 1;
    }

    if (exclude_first_extrema && count > 2)
    {
      value_average -= min_val;
      value_average -= max_val;

      count -= 2;
    }

    value_average /= count;
    value_bounded = (fac_min * min_val < val) && (val < fac_max * max_val);
  }

  // -------------------------------------------------------------------------
  // ValueWithinNearestNeighborMedian
  //   median-based outlier detection and correction
  // -------------------------------------------------------------------------
  inline void ValueWithinNearestNeighborMedian(
    bool& value_bounded,
    Real& value_corrected,
    const AA& src,
    const AthenaArray<cstate>& calc_state,
    const int n,
    const int k,
    const int j,
    const int i,
    const int num_neighbors,
    const Real fac_median,
    const bool exclude_first_extrema,
    const bool keep_base_point  // exclude base-point from median
  )
  {
    value_bounded   = true;
    value_corrected = src(n, k, j, i);

    const int nn            = num_neighbors;
    const int max_neighbors = POW3(2 * nn + 1);
    Real neighbor_vals[max_neighbors];

    int count = 0;

    Real min_val = std::numeric_limits<Real>::max();
    Real max_val = -std::numeric_limits<Real>::max();

    for (int kk = -nn; kk <= nn; ++kk)
      for (int jj = -nn; jj <= nn; ++jj)
        for (int ii = -nn; ii <= nn; ++ii)
        {
          const int i_ix = ii + i;
          const int j_ix = jj + j;
          const int k_ix = kk + k;

          // ignore central point for extrema
          if ((ii == 0) && (jj == 0) && (kk == 0) && (!keep_base_point))
          {
            continue;
          }

          // from FLOOP macro
          if ((i_ix < M1_IX_IL - M1_FSIZEI) ||
              (i_ix > M1_IX_IU + M1_FSIZEI + 1))
            continue;

          if ((j_ix < M1_IX_JL - M1_FSIZEJ) ||
              (j_ix > M1_IX_JU + M1_FSIZEJ + 1))
            continue;

          if ((k_ix < M1_IX_KL - M1_FSIZEK) ||
              (k_ix > M1_IX_KU + M1_FSIZEK + 1))
            continue;

          // check whether data is computed at the point
          bool use_point = calc_state(k_ix, j_ix, i_ix) == cstate::need;

          if (use_point)
          {
            const Real val = src(n, k_ix, j_ix, i_ix);

            neighbor_vals[count++] = val;

            if (exclude_first_extrema)
            {
              min_val = std::min(min_val, val);
              max_val = std::max(max_val, val);
            }
          }
        }

    // nothing to do?
    if (count == 0)
    {
      return;
    }

    if (exclude_first_extrema && count > 2)
    {
      int remove_count = 0;

      // min_val to the end
      for (int m = 0; m < count; ++m)
      {
        if (neighbor_vals[m] == min_val)
        {
          std::swap(neighbor_vals[m], neighbor_vals[count - 1 - remove_count]);
          remove_count++;
          break;
        }
      }

      // max_val to the end
      for (int m = 0; m < count - remove_count; ++m)
      {
        if (neighbor_vals[m] == max_val)
        {
          std::swap(neighbor_vals[m], neighbor_vals[count - 1 - remove_count]);
          remove_count++;
          break;
        }
      }

      count -= remove_count;
    }

    // compute the median
    int ix_mid = count / 2;
    std::nth_element(
      neighbor_vals, neighbor_vals + ix_mid, neighbor_vals + count);
    Real median = neighbor_vals[ix_mid];

    if (count % 2 == 0)
    {
      // largest value in lower half for even count
      Real max_lower =
        *std::max_element(neighbor_vals, neighbor_vals + ix_mid);
      median = 0.5 * (median + max_lower);
    }

    // are we within a factor of the median?
    if ((median != 0) &&
        (std::abs(value_corrected - median) <= fac_median * median))
    {
      return;
    }

    value_bounded = false;

    Real sum    = 0.0;
    int n_valid = 0;

    for (int m = 0; m < count; ++m)
    {
      if (std::abs(neighbor_vals[m] - median) <= fac_median * median)
      {
        sum += neighbor_vals[m];
        n_valid++;
      }
    }

    if (n_valid > 0)
    {
      value_corrected = sum / n_valid;
    }
    else
    {
      // fallback to median if all points outside threshold
      value_corrected = median;
    }
  }

  // -------------------------------------------------------------------------
  // ApplyValueFixes
  //   orchestrates nearest-neighbor extrema / median correction passes
  // -------------------------------------------------------------------------
  inline void ApplyValueFixes(
    std::vector<GroupSpeciesContainer<AT_C_sca>*>& data,
    const AthenaArray<cstate>& calc_state,
    const int k,
    const int j,
    const int i)
  {
    const int num_neighbors          = opt.fix_num_neighbors;
    const Real fac_median            = opt.fix_fac_median;
    const bool exclude_first_extrema = opt.fix_exclude_first_extrema;
    const bool keep_base_point       = opt.fix_keep_base_point;

    const Real fac_min = opt.fix_fac_min;
    const Real fac_max = opt.fix_fac_max;

    bool bounded_extrema = true;
    Real val_cor_avg     = 0.0;

    for (size_t ix_d = 0; ix_d < data.size(); ++ix_d)
      for (int ix_g = 0; ix_g < N_GRPS; ++ix_g)
        for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
        {
          ValueWithinNearestNeighbors(bounded_extrema,
                                      val_cor_avg,
                                      (*data[ix_d])(ix_g, ix_s).array(),
                                      calc_state,
                                      0,
                                      k,
                                      j,
                                      i,
                                      num_neighbors,
                                      fac_min,
                                      fac_max,
                                      exclude_first_extrema,
                                      keep_base_point);

          if (!bounded_extrema)
          {
            bool bounded_median = true;
            Real val_cor_med    = 0.0;
            ValueWithinNearestNeighborMedian(bounded_median,
                                             val_cor_med,
                                             (*data[ix_d])(ix_g, ix_s).array(),
                                             calc_state,
                                             0,
                                             k,
                                             j,
                                             i,
                                             num_neighbors,
                                             fac_median,
                                             exclude_first_extrema,
                                             keep_base_point);

            if (bounded_median)
            {
              continue;
            }

            (*data[ix_d])(ix_g, ix_s)(k, j, i) = val_cor_med;
          }
        }
  }

  // -------------------------------------------------------------------------
  // ComputeEquilibriumDensities  (unified, templated)
  //
  //   Computes thin + trapped equilibrium densities, interpolates based on
  //   tau, and stores the result in pm1->eql.
  //
  //   Units:
  //     in:   rho, T, Y_e  [code_units]  (GeometricSolar; T in MeV)
  //           dt, tau       [code_units]
  //     out:  pm1->eql.sc_n  [densitized code_units]  (sqrt_det_g * n)
  //           pm1->eql.sc_J  [densitized code_units]  (sqrt_det_g * J)
  //
  //   NUX convention (3-species: NUE=0, NUA=1, NUX=2):
  //     All NUX values (dens_n[2], dens_e[2]) are TOTAL for all 4
  //     heavy-lepton species.  The EquilibriumProvider is expected to
  //     return NUX=TOTAL from both NeutrinoDensity and WeakEquilibrium.
  //     The nux_weight factor (applied in CalculateEquilibriumDensity
  //     in m1_opacities.hpp) is 1.0 for 3-species -> stays TOTAL.
  //
  //   EquilibriumProvider must expose:
  //     int NeutrinoDensity(Real rho, Real T, Real Y_e,
  //                         Real& n0, Real& n1, Real& n2,
  //                         Real& e0, Real& e1, Real& e2);
  //     int WeakEquilibrium(Real rho, Real T, Real Y_e,
  //                         Real n0, Real n1, Real n2,
  //                         Real e0, Real e1, Real e2,
  //                         Real& T_eq, Real& Ye_eq,
  //                         Real& n0_eq, Real& n1_eq, Real& n2_eq,
  //                         Real& e0_eq, Real& e1_eq, Real& e2_eq);
  //
  //   RecomputeOpacFn (optional, defaults to NoOpRecompute) must expose:
  //     int operator()(int k, int j, int i, Real rho, Real T, Real Y_e);
  // -------------------------------------------------------------------------
  template <typename EquilibriumProvider,
            typename RecomputeOpacFn = NoOpRecompute>
  inline int ComputeEquilibriumDensities(
    EquilibriumProvider* provider,
    const int k,
    const int j,
    const int i,
    const Real dt,
    const Real rho,
    const Real T,
    const Real Y_e,
    const Real tau,
    const cmp_eql_dens_ini initial_guess,
    const bool using_averaging_fix,
    RecomputeOpacFn recompute_opac = NoOpRecompute{})
  {
    // Guard: below density/temperature floor -> zero equilibrium densities
    if (rho < opt.eql_rho_min || T < opt.eql_t_min)
    {
      const int ix_g           = 0;
      const Real sc_sqrt_det_g = pm1->geom.sc_sqrt_det_g(k, j, i);
      for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
      {
        pm1->eql.sc_n(ix_g, ix_s)(k, j, i) = 0.0;
        pm1->eql.sc_J(ix_g, ix_s)(k, j, i) = 0.0;
      }
      return 0;
    }

    int ierr_we = 0;
    int ierr_nd = 0;

    bool TY_adjusted       = false;
    bool calculate_trapped = true;

    Real T_star   = T;
    Real Y_e_star = Y_e;

    Real dens_n[3];
    Real dens_e[3];
    Real dens_n_trap[3];
    Real dens_e_trap[3];
    Real dens_n_thin[3];
    Real dens_e_thin[3];

    const bool need_thin =
      !(calculate_trapped &&
        (tau < dt / (opt.opacity_tau_trap + opt.opacity_tau_delta))) ||
      initial_guess == cmp_eql_dens_ini::thin;

    // only calculate if actually needed
    if (need_thin)
    {
      ierr_nd = provider->NeutrinoDensity(rho,
                                          T,
                                          Y_e,
                                          dens_n_thin[0],
                                          dens_n_thin[1],
                                          dens_n_thin[2],
                                          dens_e_thin[0],
                                          dens_e_thin[1],
                                          dens_e_thin[2]);

      if (ierr_nd)
      {
        // immediately break on error
        return ierr_nd;
      }
    }

    calculate_trapped =
      ((opt.opacity_tau_trap >= 0.0) && (tau < dt / opt.opacity_tau_trap) &&
       (rho >= opt.tra_rho_min));

    // Calculate equilibrium blackbody functions with trapped neutrinos
    if (calculate_trapped)
    {
      // --------------------------------------------------------------------
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
        const int ix_g           = 0;
        const Real sc_sqrt_det_g = pm1->geom.sc_sqrt_det_g(k, j, i);

        for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
        {
          dens_n[ix_s] = pm1->eql.sc_n(ix_g, ix_s)(k, j, i) * invsdetg;
          dens_e[ix_s] = pm1->eql.sc_J(ix_g, ix_s)(k, j, i) * invsdetg;
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

      ierr_we =
        provider->WeakEquilibrium(rho,
                                  T,
                                  Y_e,
                                  (ignore_current_data) ? 0.0 : dens_n[0],
                                  (ignore_current_data) ? 0.0 : dens_n[1],
                                  (ignore_current_data) ? 0.0 : dens_n[2],
                                  (ignore_current_data) ? 0.0 : dens_e[0],
                                  (ignore_current_data) ? 0.0 : dens_e[1],
                                  (ignore_current_data) ? 0.0 : dens_e[2],
                                  T_star,
                                  Y_e_star,
                                  dens_n_trap[0],
                                  dens_n_trap[1],
                                  dens_n_trap[2],
                                  dens_e_trap[0],
                                  dens_e_trap[1],
                                  dens_e_trap[2]);

      if (ierr_we)
      {
        // Try averaging fix if applicable
        if (!opt.use_averages && opt.use_averaging_fix && !using_averaging_fix)
        {
          Real rho, T, Y_e;
          GetHydroAveraged(k, j, i, rho, T, Y_e);

          return ComputeEquilibriumDensities(provider,
                                             k,
                                             j,
                                             i,
                                             dt,
                                             rho,
                                             T,
                                             Y_e,
                                             tau,
                                             initial_guess,
                                             true,
                                             recompute_opac);
        }
        return ierr_we;
      }

      // immediately break on error or non-finite values --------------------
      Real val = 0;
      for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
      {
        val += dens_n_trap[ix_s] + dens_e_trap[ix_s];
      }
      if (!std::isfinite(val))
      {
        return WE_FAIL_NONFINITE;
      }

      TY_adjusted = true;
      // --------------------------------------------------------------------
    }
    else
    {
      // No trapped calculation, so leave T,Y_e unmodified if later propagated
      T_star   = T;
      Y_e_star = Y_e;
    }

    // Set the black body function
    const bool regime_trapped =
      calculate_trapped &&
      (tau < dt / (opt.opacity_tau_trap + opt.opacity_tau_delta));

    if (calculate_trapped)
    {
      // proceed further ----------------------------------------------------
      if (regime_trapped)
      {
        for (int s_idx = 0; s_idx < N_SPCS; ++s_idx)
        {
          dens_n[s_idx] = dens_n_trap[s_idx];
          dens_e[s_idx] = dens_e_trap[s_idx];
        }
      }
      else
      {
        // tau in dt * [1 / (opt.opacity_tau_trap + opt.opacity_tau_delta),
        //              1 / opt.opacity_tau_trap)

        // const Real lam = (tau - opt.opacity_tau_trap) /
        // opt.opacity_tau_delta;
        const Real I_tau_min =
          dt / (opt.opacity_tau_trap + opt.opacity_tau_delta);
        const Real I_tau_max = dt / opt.opacity_tau_trap;

        // tau_map in [0, 1]
        const Real tau_map = (tau - I_tau_min) / (I_tau_max - I_tau_min);
        const Real lam     = 1 - tau_map;

        for (int s_idx = 0; s_idx < N_SPCS; ++s_idx)
        {
          dens_n[s_idx] =
            (lam * dens_n_trap[s_idx] + (1.0 - lam) * dens_n_thin[s_idx]);
          dens_e[s_idx] =
            (lam * dens_e_trap[s_idx] + (1.0 - lam) * dens_e_thin[s_idx]);
        }

        // Intermediate regime: interpolate T,Y_e
        T_star   = lam * T_star + (1.0 - lam) * T;
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
    const int ix_g           = 0;
    const Real sc_sqrt_det_g = pm1->geom.sc_sqrt_det_g(k, j, i);

    for (int ix_s = 0; ix_s < N_SPCS; ++ix_s)
    {
      pm1->eql.sc_n(ix_g, ix_s)(k, j, i) = sc_sqrt_det_g * dens_n[ix_s];
      pm1->eql.sc_J(ix_g, ix_s)(k, j, i) = sc_sqrt_det_g * dens_e[ix_s];
    }

    if ((opt.recompute_opacities_trapped && regime_trapped) ||
        (opt.recompute_opacities_interpolated && calculate_trapped))
    {
      const int ierr_opac = recompute_opac(k, j, i, rho, T_star, Y_e_star);

      if (ierr_opac)
      {
        PrintOpacityDiagnostics("CalculateOpacityCoefficients - weql failure",
                                ierr_opac,
                                dt,
                                k,
                                j,
                                i);
        assert(false);
      }
    }
    return 0;
  }

};  // class OpacityUtils

// ============================================================================
}  // namespace M1::Opacities::Common
// ============================================================================

#endif  // M1_OPACITIES_COMMON_UTILS_HPP
