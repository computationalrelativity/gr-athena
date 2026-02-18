#ifndef M1_OPACITIES_COMMON_WEAK_EQUILIBRIUM_HPP_
#define M1_OPACITIES_COMMON_WEAK_EQUILIBRIUM_HPP_

//! \file weak_equilibrium.hpp
//  \brief Common weak equilibrium solver, templated on EoS provider.
//
//  Implements a 2D Newton-Raphson solver for computing equilibrium
//  temperature and electron fraction given energy and lepton number
//  conservation constraints.  Also provides routines for computing
//  equilibrium neutrino number and energy densities from chemical
//  potentials.
//
//  Template parameter EoSProvider must support:
//    eos->ChemicalPotentials_cgs(rho, temp, ye, mu_n, mu_p, mu_e)
//    eos->GetEnergyDensity(rho, temp, ye)
//    eos->GetMinimumEnergyDensity(rho, ye)
//    eos->ApplyTableLimits(rho, temp, ye)
//    eos->ApplyLeptonLimits(yl)
//    eos->GetTableLimits(rho_min, rho_max, temp_min, temp_max, ye_min, ye_max)
//    eos->AtomicMassImpl()

#include <algorithm>
#include <cmath>

#include "../../../athena.hpp"
#include "error_codes.hpp"
#include "fermi.hpp"

namespace M1::Opacities::Common::WeakEquilibrium
{

template <typename EoSProvider>
class WeakEquilibriumSolver
{
  public:
  WeakEquilibriumSolver()
      : eos_(nullptr),
        atomic_mass_(0.0),
        eos_rho_min_(0.0),
        eos_rho_max_(0.0),
        eos_temp_min_(0.0),
        eos_temp_max_(0.0),
        eos_ye_min_(0.0),
        eos_ye_max_(0.0)
  {
  }

  //! Initialize the solver with an EoS provider and cutoffs.
  //  Must be called before SolveWeakEquilibrium.
  void Initialize(EoSProvider* eos)
  {
    eos_         = eos;
    atomic_mass_ = eos_->AtomicMassImpl();
    eos_->GetTableLimits(eos_rho_min_,
                         eos_rho_max_,
                         eos_temp_min_,
                         eos_temp_max_,
                         eos_ye_min_,
                         eos_ye_max_);
  }

  //! Core solver interface.  All I/O in CGS + MeV.
  //  nux inputs/outputs are TOTAL (all 4 heavy-lepton species).
  //
  //  y_in[0] = Ye, y_in[1] = Y_nue, y_in[2] = Y_anue,
  //  y_in[3] = Y_nux (single-species, i.e. total/4)
  //  e_in[0] = fluid energy [erg/cm^3], e_in[1..3] = neutrino energies
  //            (e_in[3] is total nux energy)
  //
  //  Post-solve: y_eq[3] and e_eq[3] are TOTAL for all 4 heavy-lepton
  //  species.
  void SolveWeakEquilibrium(Real rho,
                            Real T,
                            Real y_in[4],
                            Real e_in[4],
                            Real& T_eq,
                            Real y_eq[4],
                            Real e_eq[4],
                            int& na,
                            int& ierr)
  {
    // Total lepton fraction and internal energy
    Real yl = y_in[0] + y_in[1] - y_in[2];  // [#/baryon]
    eos_->ApplyLeptonLimits(yl);
    Real u = e_in[0] + e_in[1] + e_in[2] + e_in[3];  // [erg/cm^3]

    // Guess coefficients for [T_fac, Ye_fac]
    Real vec_guess[n_at][2] = {
      { 1.00e0, 1.00e0 }, { 0.90e0, 1.25e0 }, { 0.90e0, 1.10e0 },
      { 0.90e0, 1.00e0 }, { 0.90e0, 0.90e0 }, { 0.90e0, 0.75e0 },
      { 0.75e0, 1.25e0 }, { 0.75e0, 1.10e0 }, { 0.75e0, 1.00e0 },
      { 0.75e0, 0.90e0 }, { 0.75e0, 0.75e0 }, { 0.50e0, 1.25e0 },
      { 0.50e0, 1.10e0 }, { 0.50e0, 1.00e0 }, { 0.50e0, 0.90e0 },
      { 0.50e0, 0.75e0 },
    };

    na   = 0;
    ierr = 1;

    Real x0[2];
    Real x1[2];

    while (ierr != 0 && na < n_at)
    {
      x0[0] = vec_guess[na][0] * T;        // T guess  [MeV]
      x0[1] = vec_guess[na][1] * y_in[0];  // Ye guess [#/baryon]

      new_raph_2dim(rho, u, yl, x0, x1, ierr);
      na += 1;
    }

    // Assign output
    if (ierr == 0)
    {
      T_eq    = x1[0];
      y_eq[0] = x1[1];
    }
    else
    {
      // Fallback: return input values
      T_eq = T;
      for (int i = 0; i < 4; i++)
      {
        y_eq[i] = y_in[i];
        e_eq[i] = e_in[i];
      }
      return;
    }

    // Compute equilibrated neutrino properties
    Real mu_n, mu_p, mu_e;
    eos_->ChemicalPotentials_cgs(rho, T_eq, y_eq[0], mu_n, mu_p, mu_e);

    Real mus[2]     = { 0.0 };
    Real eta[3]     = { 0.0 };
    Real nu_dens[3] = { 0.0 };
    Real e_dens[3]  = { 0.0 };

    mus[0] = mu_e;         // electron chem pot [MeV]
    mus[1] = mu_n - mu_p;  // n-p chem pot [MeV]

    nu_deg_param_trap(T_eq, mus, eta);
    dens_nu_trap(T_eq, eta, nu_dens);
    nu_dens[2] = 4.0 * nu_dens[2];  // total for all 4 heavy-lepton species

    Real nb = rho / atomic_mass_;

    y_eq[1] = nu_dens[0] / nb;         // electron neutrino
    y_eq[2] = nu_dens[1] / nb;         // electron anti-neutrino
    y_eq[3] = nu_dens[2] / nb;         // heavy-lepton total (all 4 species)
    y_eq[0] = yl - y_eq[1] + y_eq[2];  // fluid electron fraction

    edens_nu_trap(T_eq, eta, e_dens);

    e_eq[1] = e_dens[0] * mev_to_erg;  // electron neutrino [erg/cm^3]
    e_eq[2] = e_dens[1] * mev_to_erg;  // electron anti-neutrino [erg/cm^3]
    e_eq[3] = 4.0 * e_dens[2] * mev_to_erg;  // heavy-lepton total [erg/cm^3]
    e_eq[0] = u - e_eq[1] - e_eq[2] - e_eq[3];  // fluid energy [erg/cm^3]

    // Validate: energy above table minimum
    Real e_min = eos_->GetMinimumEnergyDensity(rho, y_eq[0]);

    if (e_eq[0] < e_min)
    {
      ierr = WE_FAIL_INI_ASSIGN_NRG;
      T_eq = T;
      for (int i = 0; i < 4; i++)
      {
        y_eq[i] = y_in[i];
        e_eq[i] = e_in[i];
      }
      return;
    }

    // Validate: Ye within table bounds
    if (y_eq[0] < eos_ye_min_ || y_eq[0] > eos_ye_max_)
    {
      ierr = WE_FAIL_INI_ASSIGN_Y_E;
      T_eq = T;
      for (int i = 0; i < 4; i++)
      {
        y_eq[i] = y_in[i];
        e_eq[i] = e_in[i];
      }
      return;
    }
  }

  //! Compute equilibrium neutrino number and energy densities from
  //  Fermi integrals.  All I/O in CGS + MeV.
  //
  //  Input:  rho [g/cm^3], temp [MeV], ye [dimensionless].
  //  Output: n_* [cm^-3], en_* [MeV/cm^3].
  //  nux outputs are TOTAL for all 4 heavy-lepton species.
  //  Returns 0 on success, WE_ND_NONFINITE if any output is non-finite.
  int NeutrinoDensity_cgs(Real rho,
                          Real temp,
                          Real ye,
                          Real& n_nue,
                          Real& n_anue,
                          Real& n_nux,
                          Real& en_nue,
                          Real& en_anue,
                          Real& en_nux)
  {
    int iout = 0;

    // Obtain chemical potentials from the EoS [MeV]
    Real mu_n, mu_p, mu_e;
    eos_->ChemicalPotentials_cgs(rho, temp, ye, mu_n, mu_p, mu_e);

    // Degeneracy parameters
    Real chem_pot[2] = { mu_e, mu_n - mu_p };
    Real eta[3];
    nu_deg_param_trap(temp, chem_pot, eta);

    // Number densities [cm^-3] and energy densities [MeV/cm^3]
    Real nu_dens[3], enu_dens[3];
    dens_nu_trap(temp, eta, nu_dens);
    edens_nu_trap(temp, eta, enu_dens);

    // nue, anue are single-species; nux is total (4 heavy-lepton species)
    n_nue  = nu_dens[0];
    n_anue = nu_dens[1];
    n_nux  = 4.0 * nu_dens[2];

    en_nue  = enu_dens[0];
    en_anue = enu_dens[1];
    en_nux  = 4.0 * enu_dens[2];

    // Validate outputs
    if (!std::isfinite(n_nue) || !std::isfinite(n_anue) ||
        !std::isfinite(n_nux) || !std::isfinite(en_nue) ||
        !std::isfinite(en_anue) || !std::isfinite(en_nux))
    {
      iout = WE_ND_NONFINITE;
    }

    return iout;
  }

  //! Compute equilibrium neutrino number and energy densities from
  //  Fermi integrals, with energy output in erg/cm^3.
  //
  //  Input:  rho [g/cm^3], temp [MeV], ye [dimensionless].
  //  Output: n_* [cm^-3], en_* [erg/cm^3].
  //  nux outputs are TOTAL for all 4 heavy-lepton species.
  //  Returns 0 on success, WE_ND_NONFINITE if any output is non-finite.
  int NeutrinoDensity_cgs_erg(Real rho,
                              Real temp,
                              Real ye,
                              Real& n_nue,
                              Real& n_anue,
                              Real& n_nux,
                              Real& en_nue,
                              Real& en_anue,
                              Real& en_nux)
  {
    int iout = NeutrinoDensity_cgs(
      rho, temp, ye, n_nue, n_anue, n_nux, en_nue, en_anue, en_nux);
    // Convert energy densities from MeV/cm^3 to erg/cm^3
    en_nue *= mev_to_erg;
    en_anue *= mev_to_erg;
    en_nux *= mev_to_erg;
    return iout;
  }

  //! Compute weak equilibrium: equilibrium T, Ye, and neutrino
  //  densities/energies assuming energy + lepton number conservation.
  //  All I/O in CGS (energy in erg/cm^3, temperature in MeV).
  //  nux inputs/outputs are TOTAL for all 4 heavy-lepton species.
  //  Returns 0 on success, raw WE_ error code on failure.
  int WeakEquilibrium_cgs(Real rho,
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
    Real nb = rho / atomic_mass_;

    // Pack number fractions [#/baryon]
    Real y_in[4] = { 0.0 };
    y_in[0]      = ye;
    y_in[1]      = n_nue / nb;
    y_in[2]      = n_nua / nb;
    y_in[3]      = 0.25 * n_nux / nb;  // single-species (total / 4)

    // Pack energies [erg/cm^3]
    Real e_in[4] = { 0.0 };
    e_in[0]      = eos_->GetEnergyDensity(rho, temp, ye);
    e_in[1]      = e_nue;
    e_in[2]      = e_nua;
    e_in[3]      = e_nux;

    // Solve
    Real y_eq[4] = { 0.0 };
    Real e_eq[4] = { 0.0 };
    int na       = 0;
    int ierr     = 0;
    SolveWeakEquilibrium(rho, temp, y_in, e_in, temp_eq, y_eq, e_eq, na, ierr);

    // Unpack
    ye_eq    = y_eq[0];
    n_nue_eq = nb * y_eq[1];
    n_nua_eq = nb * y_eq[2];
    n_nux_eq = nb * y_eq[3];  // total heavy-lepton (all 4 species)
    e_nue_eq = e_eq[1];
    e_nua_eq = e_eq[2];
    e_nux_eq = e_eq[3];  // total heavy-lepton (all 4 species)

    return ierr;
  }

  private:
  EoSProvider* eos_;
  Real atomic_mass_;

  // -----------------------------------------------------------------------
  // Newton-Raphson parameters
  // -----------------------------------------------------------------------
  static constexpr Real eps_lim   = 1.e-7;
  static constexpr int n_cut_max  = 8;
  static constexpr int n_max_iter = 100;
  static constexpr int n_at       = 16;

  static constexpr Real delta_ye = 0.005;
  static constexpr Real delta_t  = 0.01;

  // -----------------------------------------------------------------------
  // Physical constants (CGS + MeV)
  // -----------------------------------------------------------------------
  static constexpr Real pi           = M_PI;
  static constexpr Real pi2          = pi * pi;
  static constexpr Real pi4          = pi2 * pi2;
  static constexpr Real mev_to_erg   = 1.60217733e-6;
  static constexpr Real hc_mevcm     = 1.23984172e-10;
  static constexpr Real oo_hc_mevcm3 = 1.0 / (hc_mevcm * hc_mevcm * hc_mevcm);

  // Derived constants
  static constexpr Real pref1 = 4.0 / 3.0 * pi * oo_hc_mevcm3;
  static constexpr Real pref2 = 4.0 * pi * mev_to_erg * oo_hc_mevcm3;
  static constexpr Real cnst3 = 7.0 * pi4 / 15.0;
  static constexpr Real cnst4 = 14.0 * pi4 / 15.0;
  static constexpr Real cnst5 = 7.0 * pi4 / 60.0;
  static constexpr Real cnst6 = 7.0 * pi4 / 30.0;

  // EOS table limits (cached, CGS + MeV)
  Real eos_rho_min_;
  Real eos_rho_max_;
  Real eos_temp_min_;
  Real eos_temp_max_;
  Real eos_ye_min_;
  Real eos_ye_max_;

  // -----------------------------------------------------------------------
  // Internal solver routines
  // -----------------------------------------------------------------------

  //! 2D Newton-Raphson with KKT boundary handling and line-search bisection.
  void new_raph_2dim(Real rho,
                     Real u,
                     Real yl,
                     Real x0[2],
                     Real x1[2],
                     int& ierr)
  {
    x1[0] = x0[0];
    x1[1] = x0[1];

    bool KKT = false;

    Real y[2] = { 0.0 };
    func_eq_weak(rho, u, yl, x1, y);

    Real err = 0.0;
    error_func_eq_weak(yl, u, y, err);

    int n_iter      = 0;
    Real J[2][2]    = { { 0.0 } };
    Real invJ[2][2] = { { 0.0 } };
    Real dx1[2]     = { 0.0 };
    Real dxa[2]     = { 0.0 };
    Real norm[2]    = { 0.0 };
    Real x1_tmp[2]  = { 0.0 };

    while (err > eps_lim && n_iter <= n_max_iter && !KKT)
    {
      jacobi_eq_weak(rho, u, yl, x1, J, ierr);
      if (ierr != 0)
      {
        ierr = WE_FAIL_JACOBIAN;
        return;
      }

      Real det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
      if (det == 0.0)
      {
        ierr = WE_FAIL_DET_SINGULAR;
        return;
      }

      inv_jacobi(det, J, invJ);

      dx1[0] = -(invJ[0][0] * y[0] + invJ[0][1] * y[1]);
      dx1[1] = -(invJ[1][0] * y[0] + invJ[1][1] * y[1]);

      // KKT boundary check
      if (x1[0] == eos_temp_min_)
        norm[0] = -1.0;
      else if (x1[0] == eos_temp_max_)
        norm[0] = 1.0;
      else
        norm[0] = 0.0;

      if (x1[1] <= eos_ye_min_)
        norm[1] = -1.0;
      else if (x1[1] >= eos_ye_max_)
        norm[1] = 1.0;
      else
        norm[1] = 0.0;

      Real scal = norm[0] * norm[0] + norm[1] * norm[1];
      if (scal <= 0.5)
        scal = 1.0;

      dxa[0] = dx1[0] - (dx1[0] * norm[0] + dx1[1] * norm[1]) * norm[0] / scal;
      dxa[1] = dx1[1] - (dx1[0] * norm[0] + dx1[1] * norm[1]) * norm[1] / scal;

      bool on_bnd = (norm[0] != 0.0 || norm[1] != 0.0);

      if (on_bnd)
      {
        Real y_tan = y[0] * (-norm[1]) + y[1] * norm[0];

        if (std::fabs(y_tan) < eps_lim &&
            (dxa[0] * dxa[0] + dxa[1] * dxa[1]) <
              (eps_lim * eps_lim * (dx1[0] * dx1[0] + dx1[1] * dx1[1])))
        {
          KKT  = true;
          ierr = WE_FAIL_KKT;
          return;
        }
      }

      // Line-search bisection
      int n_cut    = 0;
      Real fac_cut = 1.0;
      Real err_old = err;

      while (n_cut <= n_cut_max && err >= err_old)
      {
        Real step0 = (on_bnd ? dxa[0] : dx1[0]);
        Real step1 = (on_bnd ? dxa[1] : dx1[1]);

        x1_tmp[0] = x1[0] + step0 * fac_cut;
        x1_tmp[1] = x1[1] + step1 * fac_cut;

        if (std::isnan(x1_tmp[0]))
        {
          ierr = WE_FAIL_NEXT_STEP;
          return;
        }

        eos_->ApplyTableLimits(rho, x1_tmp[0], x1_tmp[1]);

        x1[0] = x1_tmp[0];
        x1[1] = x1_tmp[1];

        func_eq_weak(rho, u, yl, x1, y);
        error_func_eq_weak(yl, u, y, err);

        n_cut += 1;
        fac_cut *= 0.5;
      }

      n_iter += 1;
    }

    ierr = (n_iter <= n_max_iter) ? 0 : WE_FAIL_STAGNATED;
  }

  //! Residuals for energy + lepton conservation.
  void func_eq_weak(Real rho, Real u, Real yl, Real x[2], Real y[2])
  {
    Real nb = rho / atomic_mass_;

    Real mu_n, mu_p, mu_e;
    eos_->ChemicalPotentials_cgs(rho, x[0], x[1], mu_n, mu_p, mu_e);
    Real mus[2] = { mu_e, mu_n - mu_p };

    Real e = eos_->GetEnergyDensity(rho, x[0], x[1]);

    Real eta_vec[2] = { 0.0 };
    nu_deg_param_trap(x[0], mus, eta_vec);
    Real eta  = eta_vec[0];
    Real eta2 = eta * eta;

    Real t3 = x[0] * x[0] * x[0];
    Real t4 = t3 * x[0];
    y[0]    = x[1] + pref1 * t3 * eta * (pi2 + eta2) / nb - yl;
    y[1] =
      (e + pref2 * t4 * ((cnst5 + 0.5 * eta2 * (pi2 + 0.5 * eta2)) + cnst6)) /
        u -
      1.0;
  }

  //! Error norm from residuals.
  void error_func_eq_weak(Real yl, Real u, Real y[2], Real& err)
  {
    err = std::abs(y[0] / yl) + std::abs(y[1] / 1.0);
  }

  //! Jacobian for the 2D Newton-Raphson.
  void jacobi_eq_weak(Real rho,
                      Real u,
                      Real yl,
                      Real x[2],
                      Real J[2][2],
                      int& ierr)
  {
    Real t  = x[0];
    Real ye = x[1];

    Real mu_n, mu_p, mu_e;
    eos_->ChemicalPotentials_cgs(rho, t, ye, mu_n, mu_p, mu_e);
    Real mus[2] = { mu_e, mu_n - mu_p };

    Real eta_vec[3] = { 0.0 };
    nu_deg_param_trap(t, mus, eta_vec);
    Real eta  = eta_vec[0];
    Real eta2 = eta * eta;

    Real detadt, detadye, dedt, dedye;
    eta_e_gradient(rho, t, ye, eta, detadt, detadye, dedt, dedye, ierr);
    if (ierr != 0)
      return;

    Real nb = rho / atomic_mass_;

    Real t2 = t * t;
    Real t3 = t2 * t;
    Real t4 = t3 * t;
    J[0][0] = pref1 / nb * t2 *
              (3.e0 * eta * (pi2 + eta2) + t * (pi2 + 3.e0 * eta2) * detadt);
    J[0][1] = 1.e0 + pref1 / nb * t3 * (pi2 + 3.e0 * eta2) * detadye;

    J[1][0] = (dedt + pref2 * t3 *
                        (cnst3 + cnst4 + 2.e0 * eta2 * (pi2 + 0.5 * eta2) +
                         eta * t * (pi2 + eta2) * detadt)) /
              u;
    J[1][1] = (dedye + pref2 * t4 * eta * (pi2 + eta2) * detadye) / u;

    if (std::isnan(eta))
    {
      ierr = 1;
      return;
    }
    if (std::isnan(detadt))
    {
      ierr = 1;
      return;
    }
    if (std::isnan(t))
    {
      ierr = 1;
      return;
    }

    ierr = 0;
  }

  //! Numerical gradients of eta and internal energy via 4 EOS calls.
  void eta_e_gradient(Real rho,
                      Real t,
                      Real ye,
                      Real eta,
                      Real& detadt,
                      Real& detadye,
                      Real& dedt,
                      Real& dedye,
                      int& ierr)
  {
    Real mu_n, mu_p, mu_e;

    // --- Vary ye ---
    Real ye1 = std::max(ye - delta_ye, eos_ye_min_);

    eos_->ChemicalPotentials_cgs(rho, t, ye1, mu_n, mu_p, mu_e);
    Real mus1[2] = { mu_e, mu_n - mu_p };
    Real e1      = eos_->GetEnergyDensity(rho, t, ye1);

    Real ye2 = std::min(ye + delta_ye, eos_ye_max_);

    eos_->ChemicalPotentials_cgs(rho, t, ye2, mu_n, mu_p, mu_e);
    Real mus2[2] = { mu_e, mu_n - mu_p };
    Real e2      = eos_->GetEnergyDensity(rho, t, ye2);

    Real dmuedye   = (mus2[0] - mus1[0]) / (ye2 - ye1);
    Real dmuhatdye = (mus2[1] - mus1[1]) / (ye2 - ye1);
    dedye          = (e2 - e1) / (ye2 - ye1);

    // --- Vary T ---
    Real t1 = std::max(t - delta_t, eos_temp_min_);
    Real t2 = std::min(t + delta_t, eos_temp_max_);

    eos_->ChemicalPotentials_cgs(rho, t1, ye, mu_n, mu_p, mu_e);
    mus1[0] = mu_e;
    mus1[1] = mu_n - mu_p;
    e1      = eos_->GetEnergyDensity(rho, t1, ye);

    eos_->ChemicalPotentials_cgs(rho, t2, ye, mu_n, mu_p, mu_e);
    mus2[0] = mu_e;
    mus2[1] = mu_n - mu_p;
    e2      = eos_->GetEnergyDensity(rho, t2, ye);

    Real dmuedt   = (mus2[0] - mus1[0]) / (t2 - t1);
    Real dmuhatdt = (mus2[1] - mus1[1]) / (t2 - t1);
    dedt          = (e2 - e1) / (t2 - t1);

    // Combine
    detadt  = (-eta + dmuedt - dmuhatdt) / t;
    detadye = (dmuedye - dmuhatdye) / t;

    ierr = std::isnan(detadt) ? 1 : 0;
  }

  //! Inverts a 2x2 Jacobian.
  void inv_jacobi(Real det, Real J[2][2], Real invJ[2][2])
  {
    Real inv_det = 1.0 / det;
    invJ[0][0]   = J[1][1] * inv_det;
    invJ[1][1]   = J[0][0] * inv_det;
    invJ[0][1]   = -J[0][1] * inv_det;
    invJ[1][0]   = -J[1][0] * inv_det;
  }

  //! Neutrino degeneracy parameters from chemical potentials.
  //
  //  in:   temp_m   [MeV]    (temperature)
  //        chem_pot [MeV]    chem_pot[0] = mu_e, chem_pot[1] = mu_hat = mu_n -
  //                                                                     mu_p
  //  out:  eta      [-]      degeneracy parameters (dimensionless)
  //        eta[0] = (mu_e - mu_hat)/T   (nue)
  //        eta[1] = -(mu_e - mu_hat)/T  (anue)
  //        eta[2] = 0                   (nux - zero chemical potential)
  //
  //  NUX convention: each nux species has eta=0 (no chemical potential).
  //  The caller must multiply by 4 to get the total for all heavy-lepton
  //  species.
  void nu_deg_param_trap(Real temp_m, Real chem_pot[2], Real eta[3])
  {
    if (temp_m > 0.0)
    {
      eta[0] = (chem_pot[0] - chem_pot[1]) / temp_m;
      eta[1] = -eta[0];
      eta[2] = 0.0;
    }
    else
    {
      eta[0] = 0.0;
      eta[1] = 0.0;
      eta[2] = 0.0;
    }
  }

  //! Equilibrium neutrino number densities [#/cm^3].
  //
  //  in:   temp_m    [MeV]    (temperature)
  //        eta_nu    [-]      (degeneracy parameters from nu_deg_param_trap)
  //  out:  nu_dens   [cm^-3]  (number densities)
  //        nu_dens[0] = nue, nu_dens[1] = anue, nu_dens[2] = single nux
  //        species
  //
  //  NUX convention: nu_dens[2] is per SINGLE heavy-lepton species.
  //  The caller must multiply by 4 for the total heavy-lepton number density.
  void dens_nu_trap(Real temp_m, Real eta_nu[3], Real nu_dens[3])
  {
    const Real pref = 4.0 * pi / (hc_mevcm * hc_mevcm * hc_mevcm);
    Real temp_m3    = temp_m * temp_m * temp_m;

    for (int it = 0; it < 3; it++)
    {
      Real f2     = Fermi::fermi2(eta_nu[it]);
      nu_dens[it] = pref * temp_m3 * f2;
    }
  }

  //! Equilibrium neutrino energy densities [MeV/cm^3].
  //
  //  in:   temp_m    [MeV]       (temperature)
  //        eta_nu    [-]         (degeneracy parameters from
  //        nu_deg_param_trap)
  //  out:  enu_dens  [MeV/cm^3]  (energy densities)
  //        enu_dens[0] = nue, enu_dens[1] = anue, enu_dens[2] = single nux
  //        species
  //
  //  NUX convention: enu_dens[2] is per SINGLE heavy-lepton species.
  //  The caller must multiply by 4 for the total heavy-lepton energy density.
  void edens_nu_trap(Real temp_m, Real eta_nu[3], Real enu_dens[3])
  {
    const Real pref = 4.0 * pi / (hc_mevcm * hc_mevcm * hc_mevcm);
    Real temp_m4    = temp_m * temp_m * temp_m * temp_m;

    for (int it = 0; it < 3; it++)
    {
      Real f3      = Fermi::fermi3(eta_nu[it]);
      enu_dens[it] = pref * temp_m4 * f3;
    }
  }
};

}  // namespace M1::Opacities::Common::WeakEquilibrium

#endif  // M1_OPACITIES_COMMON_WEAK_EQUILIBRIUM_HPP_
