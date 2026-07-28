#ifndef EOS_TRANS_H
#define EOS_TRANS_H

//! \file eos_transition.hpp
//  \brief Defines EOSTransition, which is used to implement a transition
//  between EOSCompose and EOSHelmholtz.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "../../athena.hpp"
#include "../../globals.hpp"
#include "eos_compose.hpp"
#include "eos_helmholtz.hpp"
#include "eos_policy_interface.hpp"
#include "numtools_root.hpp"

namespace Primitive
{

class EOSTransition : public EOSPolicyInterface
{
  protected:
  /// Constructor
  EOSTransition();

  /// Destructor
  ~EOSTransition();

  /// Temperature from energy density
  Real TemperatureFromE(Real n, Real e, Real* Y);

  /// Temperature from specific internal energy
  Real TemperatureFromEps(Real n, Real eps, Real* Y);

  /// Temperature from entropy per baryon
  Real TemperatureFromEntropy(Real n, Real eps, Real* Y);

  /// Calculate the temperature using.
  Real TemperatureFromP(Real n, Real p, Real* Y);

  /// Calculate the energy density using.
  Real Energy(Real n, Real T, Real* Y);

  /// Calculate the pressure using.
  Real Pressure(Real n, Real T, Real* Y);

  /// Calculate the entropy per baryon using.
  Real Entropy(Real n, Real T, Real* Y);

  /// Calculate the enthalpy per baryon using.
  Real Enthalpy(Real n, Real T, Real* Y);

  /// Fused temperature + pressure + enthalpy from energy: avoiding redundant
  /// lookups
  void TemperaturePressureAndEnthalpyFromE(Real n,
                                           Real e,
                                           Real* Y,
                                           Real* T,
                                           Real* P,
                                           Real* h,
                                           int* guess_it = nullptr);

  void PressureAndEnthalpyFromE(Real n,
                                Real e,
                                Real* Y,
                                Real* P,
                                Real* h,
                                int* guess_it = nullptr);

  /// Fused pressure + enthalpy: single weight computation for both P and h.
  void PressureAndEnthalpy(Real n, Real T, Real* Y, Real* P, Real* h);

  /// Calculate the sound speed.
  Real SoundSpeed(Real n, Real T, Real* Y);

  /// Calculate the specific internal energy per unit mass
  Real SpecificInternalEnergy(Real n, Real T, Real* Y);

  /// Calculate the baryon chemical potential
  Real BaryonChemicalPotential(Real n, Real T, Real* Y);

  /// Calculate the charge chemical potential
  Real ChargeChemicalPotential(Real n, Real T, Real* Y);

  /// Calculate the electron-lepton chemical potential
  Real ElectronLeptonChemicalPotential(Real n, Real T, Real* Y);

  /// Calculate the interaction potential difference
  Real InteractionPotentialDifference(Real n, Real T, Real* Y);

  /// Species fractions
  Real FrYn(Real n, Real T, Real* Y);
  Real FrYp(Real n, Real T, Real* Y);
  Real FrXa(Real n, Real T, Real* Y);
  Real FrXh(Real n, Real T, Real* Y);
  Real AN(Real n, Real T, Real* Y);
  Real ZN(Real n, Real T, Real* Y);

  /// Get the minimum enthalpy per baryon.
  Real MinimumEnthalpy();

  /// The lowest valid temperature depends on density: inside the validity
  /// ramp toward the Helmholtz density cutoff the compose table
  /// participates for every T, so its grid minimum applies.
  Real MinimumValidTemperature(Real n) const
  {
    return (log(n) >= m_ln_n_h0) ? compose_eos->min_T : min_T;
  }

  /// Interior fast-path guard. For n above the Helmholtz density cutoff the
  /// transition weight is identically 1 (TransitionFactor's first branch), so
  /// callers may skip SanitizeMassFractions + TransitionFactor and delegate
  /// straight to the compose table. The compose accessors read only Y[0] = Ye;
  /// clamp it to the compose grid (== [min_Y[SCYE], max_Y[SCYE]]) exactly as
  /// SanitizeMassFractions would, so the delegated result is bit-identical.
  /// Returns true (with yq set) when the fast path applies.
  inline bool InteriorYq(Real n, const Real* Y, Real& yq) const
  {
    if (n <= m_helm_n_max)
      return false;
    yq = std::max(min_Y[SCYE], std::min(Y[SCYE], max_Y[SCYE]));
    return true;
  }

  /// Get the minimum pressure at a given density and composition
  Real MinimumPressure(Real n, Real* Y);

  /// Get the maximum pressure at a given density and composition
  Real MaximumPressure(Real n, Real* Y);

  /// Get the minimum entropy per baryon at a given density and composition
  Real MinimumEntropy(Real n, Real* Y);

  /// Get the maximum entropy per baryon at a given density and composition
  Real MaximumEntropy(Real n, Real* Y);

  /// Get the minimum specific internal energy at a given density and
  /// composition
  Real MinimumSpecificInternalEnergy(Real n, Real* Y);

  /// Get the maximum specific internal energy at a given density and
  /// composition
  Real MaximumSpecificInternalEnergy(Real n, Real* Y);

  /// Get the minimum energy at a given density and composition
  Real MinimumEnergy(Real n, Real* Y);

  /// Get the maximum energy at a given density and composition
  Real MaximumEnergy(Real n, Real* Y);

  public:
  /// Calls the individual initialization functions
  void InitializeTables(std::string fname,
                        std::string helm_fname,
                        Real baryon_mass);

  /// Some setters for parameters
  void SetTransition(Real n_start, Real n_end, Real T_start, Real T_end);

  /// Get the NSE value of the binding energy per baryon
  Real GetNSEBindingEnergy(Real n, Real T, Real* Y);

  /// Get the transition parameters
  void PrintParameters();

  /// Get the factor to transition between the two EOSs, as a function of
  /// density and temperature
  Real TransitionFactor(Real n, Real T) const;
  /// Log-space variant taking precomputed ln = log(n), lT = log(T).
  Real TransitionFactor(Real n, Real T, Real ln, Real lT) const;

  /// Normalize the mass fractions to 1 and return the difference normalization
  /// factor
  Real SanitizeMassFractions(Real* Y, Real* Y_norm) const;

  Real GetTempTransStart() const
  {
    return trans_T_start;
  }

  Real GetDensTransStart() const
  {
    return exp(trans_ln_start);
  }

  //! \brief Boundaries for the error policy (ResetFloorTransition).
  //
  //  ld_* are the *effective* upper limits of the low-density (Helmholtz)
  //  EOS -- the start of the validity ramps beyond which TransitionFactor
  //  blends toward (and at the cutoff, forces) the compose table -- not
  //  the raw Helmholtz table limits. Beyond them the state must respect
  //  the compose table's validity (hd_*): e.g. a state with n > ld_n and
  //  T < hd_t would evaluate the compose table extrapolated below its
  //  temperature grid.
  void GetTableBoundaries(Real& ld_n, Real& hd_n, Real& ld_t, Real& hd_t)
  {
    ld_n = exp(m_ln_n_h0);
    ld_t = exp(m_ln_T_h0);
    hd_n = compose_eos->min_n;
    hd_t = compose_eos->min_T;
  }

  /// Check if the EOS has been initialized properly.
  inline bool IsInitialized() const
  {
    return m_initialized;
  }

  /// Set the upper temperature for using the helmholtz eos at all
  void SetHelmholtzTMax(Real T_max)
  {
    m_helm_T_max = T_max;
    if (m_initialized)
      update_bounds();
  }

  /// Set the upper density for using the helmholtz eos at all
  void SetHelmholtzNMax(Real n_max)
  {
    m_helm_n_max = n_max;
    if (m_initialized)
      update_bounds();
  }

  /// Set the widths (in decades) of the validity ramps below the
  /// Helmholtz cutoffs, over which the weight blends toward the compose
  /// table instead of jumping there.
  void SetHelmholtzRampDecades(Real n_decades, Real T_decades)
  {
    m_helm_n_ramp_dec = n_decades;
    m_helm_T_ramp_dec = T_decades;
    if (m_initialized)
      update_bounds();
  }

  private:
  /// Set the baryon mass.
  void SetBaryonMass(Real new_mb);

  void update_bounds();

  /// Low level inversion function, not intended for outside use
  Real temperature_from_var_trans(int iv, Real var, Real n, Real* Y) const;
  int comp_it_trans_start, comp_it_trans_end, comp_it_helm_tmax;

  static constexpr Real mn  = EOSHelmholtz::mn;   // neutron mass in MeV
  static constexpr Real mp  = EOSHelmholtz::mp;   // proton mass in MeV
  static constexpr Real ma  = EOSHelmholtz::ma;   // alpha mass in MeV
  static constexpr Real me  = EOSHelmholtz::me;   // electron mass in MeV
  static constexpr Real mFe = 52103.06261020851;  // ATOMIC mass of 56Fe, MeV

  protected:
  EOSCompOSE* compose_eos;
  EOSHelmholtz* helmholtz_eos;

  Real trans_T_start, trans_T_end, trans_ln_start,
    trans_ln_end;  // Transition parameters
  // Transitions width
  Real m_trans_T_width, m_trans_ln_width;

  Real mb, max_n, min_n, max_T, min_T, max_Y[MAX_SPECIES], min_Y[MAX_SPECIES];

  // Minimum enthalpy per baryon
  Real m_min_h;

  // bool to protect against access of uninitialised table, and prevent
  // repeated reading of table
  bool m_initialized;
  static bool s_printed_parameters;

  // helmholtz upper bounds
  Real m_helm_n_max, m_helm_T_max;
  // widths (in decades) of the validity ramps below the helmholtz bounds
  Real m_helm_n_ramp_dec, m_helm_T_ramp_dec;
  // derived: ramp starts (log) and inverse log-widths
  Real m_ln_n_h0, m_ln_T_h0, m_id_ln_n_ramp, m_id_ln_T_ramp;

  class RootFunctor
  {
    public:
    Real operator()(Real wt,
                    int iv,
                    int it,
                    Real v0,
                    Real dv,
                    Real lt0,
                    Real dlt,
                    Real n,
                    Real ln,
                    Real* Y,
                    Real var,
                    const EOSTransition* peos) const
    {
      // lt = log(t) is known exactly, so hand it (and the loop-invariant
      // ln = log(n)) to the log-space TransitionFactor.
      Real lt      = lt0 + wt * dlt;
      Real t       = exp(lt);
      Real f_trans = peos->TransitionFactor(n, t, ln, lt);
      // iv is an EOSCompOSE table index; it must not be passed to the
      // Helmholtz table, whose variable indexing is different (and which
      // stores log(eps) rather than log(e)).
      Real var_helm;
      if (iv == EOSCompOSE::ECLOGP)
      {
        var_helm = peos->helmholtz_eos->Pressure(n, t, Y);
      }
      else if (iv == EOSCompOSE::ECLOGE)
      {
        var_helm = peos->helmholtz_eos->Energy(n, t, Y);
      }
      else
      {
        throw std::logic_error(
          "EOSTransition::RootFunctor only implemented for log(P) and "
          "log(e).");
      }
      Real var_comp = exp(v0 + wt * dv);
      Real var_pt   = log(var_comp * f_trans + var_helm * (1 - f_trans));

      return (var - var_pt) / var;  // N.B error is expected to be relative
    }
  };

  NumTools::Root root;
  RootFunctor RootFunction;
};
}  // namespace Primitive

#endif
