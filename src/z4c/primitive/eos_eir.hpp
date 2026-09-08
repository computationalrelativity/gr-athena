#ifndef EOS_EIR_H
#define EOS_EIR_H

//! \file eos_eir.hpp
//  \brief Defines EOSEIR, a thermal EOS policy built on tabulated
//         EIR electron quantities plus analytic ion/radiation terms.

///  \warning This code assumes the table to be uniformly spaced in
///           log ne and log t

#include <cmath>
#include <cstddef>
#include <string>

#include "../../athena.hpp"
#include "eos_policy_interface.hpp"

namespace Primitive
{

class EOSEIR : public EOSPolicyInterface
{
  friend class EOSTransition;

  public:
  enum TableVariables
  {
    ECLOGP   = 0,  //! logo of pressure / 1 MeV fm^-3
    ECENT    = 1,  //! entropy per baryon [kb]
    ECLOGEPS = 2,  //! log of specific internal energy
    ECETA    = 3,  //! electron degeneracy parameter
    ECDEPSDT = 4,
    ECDPDN   = 5,
    ECDPDT   = 6,
    ECNVARS  = 7
  };

  protected:
  /// Constructor
  EOSEIR();

  /// Destructor
  ~EOSEIR();

  /// Temperature from energy density
  Real TemperatureFromE(Real n, Real e, Real* Y);

  /// Calculate the temperature from the pressure
  Real TemperatureFromP(Real n, Real p, Real* Y);
  //
  /// Temperature from specific internal energy. guess_it, when given,
  /// warm-starts the temperature bracket with the table index of a
  /// previous call (the c2p iteration hot path); a stale or foreign-grid
  /// index is validated and falls back to the full search, so any int is
  /// safe to pass.
  Real TemperatureFromEps(Real n, Real eps, Real* Y, int* guess_it = nullptr);

  /// Calculate the temperature from the entropy
  Real TemperatureFromEntropy(Real n, Real s, Real* Y);

  /// Calculate the energy density.
  Real Energy(Real n, Real T, Real* Y);

  /// Calculate the pressure.
  Real Pressure(Real n, Real T, Real* Y);

  /// Calculate the average baryon number per nucleus.
  Real Abar(Real n, Real T, Real* Y);

  /// Calculate the entropy per baryon.
  Real Entropy(Real n, Real T, Real* Y);

  /// Calculate the enthalpy per baryon.
  Real Enthalpy(Real n, Real T, Real* Y);

  /// Calculate the sound speed.
  Real SoundSpeed(Real n, Real T, Real* Y);

  /// Calculate the specific internal energy per unit mass
  Real SpecificInternalEnergy(Real n, Real T, Real* Y);

  /// Calculate the neutron chemical potential
  Real NeutronChemicalPotential(Real n, Real T, Real* Y);

  /// Calculate the proton chemical potential
  Real ProtonChemicalPotential(Real n, Real T, Real* Y);

  /// Calculate the baryon electron chemical potential
  Real ElectronChemicalPotential(Real n, Real T, Real* Y);

  /// Calculate the baryon chemical potential
  Real BaryonChemicalPotential(Real n, Real T, Real* Y);

  /// Calculate the charge chemical potential
  Real ChargeChemicalPotential(Real n, Real T, Real* Y);

  /// Calculate the electron-lepton chemical potential
  Real ElectronLeptonChemicalPotential(Real n, Real T, Real* Y);

  /// Get the minimum enthalpy per baryon.
  Real MinimumEnthalpy();

  /// Get the minimum pressure at a given density and composition
  Real MinimumPressure(Real n, Real* Y);

  /// Get the maximum pressure at a given density and composition
  Real MaximumPressure(Real n, Real* Y);

  /// Get the minimum energy at a given density and composition
  Real MinimumInternalEnergy(Real n, Real* Y);

  /// Get the maximum energy at a given density and composition
  Real MaximumInternalEnergy(Real n, Real* Y);

  /// Get the minimum entropy per baryon at a given density and composition
  Real MinimumEntropy(Real n, Real* Y);

  /// Get the maximum entropy per baryon at a given density and composition
  Real MaximumEntropy(Real n, Real* Y);

  public:
  /// Reads the table file.
  void ReadTableFromFile(std::string fname, Real min_Ye, Real max_Ye);

  /// Set the baryon mass.
  ///
  /// N.B. mb is a convention, not a physical mass. It is the reference mass
  /// per baryon that defines rho = mb * n and the rest-mass / eps split, it
  /// is set at runtime from hydro/bmass, and the RHINE glue converts mass
  /// excesses against it. It is unrelated to mn, mp below, and must not be
  /// touched by SetNucleonMasses.
  void SetBaryonMass(Real new_mb);

  /// Set the physical nucleon masses [MeV].
  ///
  /// Used by EOSTransition to adopt the values carried by the compose table,
  /// so that both halves of the blended EOS build their chemical potentials
  /// and rest-mass zero points from the same constants. Only evaluations use
  /// mn, mp (chemical potentials, Sackur-Tetrode entropy), never the stored
  /// table, so no rebuild is needed.
  void SetNucleonMasses(Real new_mn, Real new_mp);

  /// Enable/disable the ion Coulomb (OCP) correction to P, eps and s.
  inline void SetCoulomb(bool use)
  {
    use_coulomb = use;
  }
  inline bool GetCoulomb() const
  {
    return use_coulomb;
  }

  /// Get the raw number density
  Real const* GetRawLogNumberDensity() const
  {
    return m_log_ne;
  }
  /// Get the raw number density
  Real const* GetRawLogTemperature() const
  {
    return m_log_t;
  }
  /// Get the raw table data
  Real const* GetRawTable() const
  {
    return m_table;
  }

  // Indexing used to access the data
  inline ptrdiff_t index(int iv, int in, int it) const
  {
    return it + m_nt * (in + m_nn * iv);
  }

  /// Check if the EOS has been initialized properly.
  inline bool IsInitialized() const
  {
    return m_initialized;
  }

  /// Set the number of species. Throw an exception if
  /// the number of species is invalid.
  void SetNSpecies(int n);

  private:
  inline Real inverse_abar(Real* Y) const
  {
    Real abar = Y[SCXN] + Y[SCXP] + Y[SCXA] / 4 +
                ((Y[SCXH] > 0.0) ? Y[SCXH] / Y[SCAH] : 0.0);
    if (abar <= 0.0)
    {
      printf(
        "EOSEIR::inverse_abar: got invalid mass fractions, sum is "
        "%.5e\n",
        abar);
      return 1.0;
    }
    return abar;
  }

  // Classical one-component-plasma Coulomb correction for a single mean
  // ion averaged over the charged species (free neutrons keep their ideal
  // terms but carry no charge). Fit expressions and coefficients are the
  // Yakovlev & Shalybkov (1989) forms exactly as used in the Timmes
  // Helmholtz EOS, so helmeos serves as a reference implementation.
  // Returns the per-ion energy u = E_C/(N_i T), its Gamma derivative and
  // the per-ion entropy s [kB]; y_chg = 0 signals "no correction".
  // Deliberately NOT applied to the chemical potentials: those feed only
  // the neutrino transport, negligible for matter in the EIR regime.
  // ponytail: single mean ion, no linear mixing rule; refine if mixed-Z
  // compositions in the EIR regime ever matter.
  struct CoulombTerms
  {
    Real y_chg;  // charged ions per baryon
    Real gamma;  // plasma coupling parameter
    Real u;      // E_C per ion in units of T
    Real du;     // du/dGamma
    Real s;      // S_C per ion in kB
  };
  inline CoulombTerms coulomb_terms(Real n, Real T, Real* Y) const
  {
    CoulombTerms c = {0.0, 0.0, 0.0, 0.0, 0.0};
    Real y_chg = Y[SCXP] + Y[SCXA] / 4 +
                 ((Y[SCXH] > 0.0) ? Y[SCXH] / Y[SCAH] : 0.0);
    if (!(y_chg > 1e-30) || !(Y[SCYE] > 0.0) || !(T > 0.0))
    {
      return c;
    }
    Real zbar  = Y[SCYE] / y_chg;  // charge neutrality
    Real n_i   = n * y_chg;
    Real a_i   = cbrt(3.0 / (4.0 * M_PI * n_i));
    Real gamma = zbar * zbar * esqu / (a_i * T);
    constexpr Real a1 = -0.898004, b1 = 0.96786, c1 = 0.220703,
                   d1 = -0.86097, e1 = 2.5269;
    constexpr Real a2 = 0.29561, b2 = 1.9885, c2 = 0.288675;
    if (gamma >= 1.0)
    {
      Real x = sqrt(sqrt(gamma));  // gamma^(1/4)
      c.u    = a1 * gamma + b1 * x + c1 / x + d1;
      c.du   = a1 + 0.25 * (b1 * x - c1 / x) / gamma;
      c.s    = -(3.0 * b1 * x - 5.0 * c1 / x + d1 * (log(gamma) - 1.0) - e1);
    }
    else
    {
      Real x = gamma * sqrt(gamma);  // gamma^(3/2)
      Real y = pow(gamma, b2);
      c.u    = -3.0 * c2 * x + a2 * y;
      c.du   = (-4.5 * c2 * x + a2 * b2 * y) / gamma;
      c.s    = -(c2 * x - a2 * (b2 - 1.0) / b2 * y);
    }
    c.y_chg = y_chg;
    c.gamma = gamma;
    return c;
  }

  /// Low level function, not intended for outside use
  Real temperature_from_var(int vi, Real var, Real n, Real* Y,
                            int* guess_it = nullptr) const;
  /// Low level evaluation function, not intended for outside use
  Real eval_at_nty(int vi, Real n, Real T, Real* Y) const;
  /// Low level evaluation function, not intended for outside use
  Real eval_at_lnty(int vi, Real ln, Real lT) const;
  /// Low level function to add the analytic terms
  Real add_rad_ion(int vi, Real var, Real n, Real T, Real* Y) const;

  /// Evaluate interpolation weight for density
  void weight_idx_ln(Real* w0, Real* w1, int* in, Real log_n) const;
  /// Evaluate interpolation weight for temperature
  void weight_idx_lt(Real* w0, Real* w1, int* it, Real log_t) const;

  private:
  // Inverse of table spacing
  Real m_id_log_ne, m_id_log_t;
  // Table size
  int m_nn, m_nt;
  // Minimum enthalpy per baryon
  const Real m_min_h = 0.0;

  // Table storage, care should be made to store these data on the GPU later
  // Static pointers used to share access to single instance of table in memory
  // (per MPI process)
  static Real* m_log_ne;
  static Real* m_log_t;
  static Real* m_table;

  // bool to protect against access of uninitialised table, and prevent
  // repeated reading of table
  static bool m_initialized;

  // ion Coulomb (OCP) correction switch, per instance like mb
  bool use_coulomb = true;

  // Auxiliary static variables to share data only available when table is open
  // to those threads that do not open it variables from EOSEIR
  static Real sm_id_log_ne, sm_id_log_t;
  static int sm_nn, sm_nt;

  // variables from EOSPolicy
  static Real s_mb, s_max_n, s_min_n, s_max_T, s_min_T;
  // (the unused EOSPolicy bounds s_max_P/s_min_P/s_max_e/s_min_e are not
  // mirrored here)
  static constexpr Real hbarc = 197.3269804;  // MeV fm
  // const Real asol = 8.563456312967042e-08; // pi**2/(15*hbarc^3) (MeV fm)^-3
  static constexpr Real asol =
    M_PI * M_PI / (15.0 * hbarc * hbarc * hbarc);  // (MeV fm)^-3
  // const Real sac_const = 244654.27090035815; // h^2/(2*pi) in (MeV fm)^2
  static constexpr Real sac_const = hbarc * hbarc * 2.0 * M_PI;  // (MeV fm)^2
  static constexpr Real esqu      = 1.4399764;                   // e^2, MeV fm
  static constexpr Real me        = 0.5109989461;                // MeV
  // Physical nucleon masses [MeV]. The CODATA values below are the defaults;
  // when the EIR EOS is driven by EOSTransition these are replaced by
  // the values carried by the compose table (SetNucleonMasses), since the
  // table energies and chemical potentials were built with them.
  static constexpr Real mn_codata = 939.5654133;                 // MeV
  static constexpr Real mp_codata = 938.2720813;                 // MeV
  static Real mn;                                                // MeV
  static Real mp;                                                // MeV
  static constexpr Real ma        = 3727.379378;                 // MeV
  static constexpr int g_n        = 2;  // neutron spin degeneracy
  static constexpr int g_p        = 2;  // proton spin degeneracy
  static constexpr int g_a =
    1;  // alpha particle spin degeneracy set to 1 as in Just+ 2023
  static constexpr int g_h =
    1;  // heavy nuclei spin degeneracy set to 1 as in Just+ 2023
};

}  // namespace Primitive

#endif
