#ifndef EOS_TRANS_H
#define EOS_TRANS_H

//! \file eos_transition.hpp
//  \brief Defines EOSTransition, which is used to implement a transition between EOSCompose and EOSHelmholtz.

#include <cstddef>
#include <string>

#include <boost/math/tools/roots.hpp>

#include "../../athena.hpp"
#include "eos_policy_interface.hpp"
#include "eos_compose.hpp"
#include "eos_helmholtz.hpp"


namespace Primitive {

struct UnitSystem;

class EOSTransition : public EOSPolicyInterface {
  protected:
    /// Constructor
    EOSTransition();

    /// Destructor
    ~EOSTransition();

    /// Temperature from energy density
    Real TemperatureFromE(Real n, Real e, Real *Y);

    /// Temperature from specific internal energy
    Real TemperatureFromEps(Real n, Real eps, Real *Y);

    /// Temperature from entropy per baryon
    Real TemperatureFromEntropy(Real n, Real eps, Real *Y);

    /// Calculate the temperature using.
    Real TemperatureFromP(Real n, Real p, Real *Y);

    /// Calculate the energy density using.
    Real Energy(Real n, Real T, Real *Y);

    /// Calculate the pressure using.
    Real Pressure(Real n, Real T, Real *Y);

    /// Calculate the entropy per baryon using.
    Real Entropy(Real n, Real T, Real *Y);

    /// Calculate the enthalpy per baryon using.
    Real Enthalpy(Real n, Real T, Real *Y);

    /// Calculate the sound speed.
    Real SoundSpeed(Real n, Real T, Real *Y);

    /// Calculate the average baryon number per nucleon.
    Real Abar(Real n, Real T, Real *Y);

    /// Calculate the specific internal energy per unit mass
    Real SpecificInternalEnergy(Real n, Real T, Real *Y);

    /// Calculate the baryon chemical potential
    Real BaryonChemicalPotential(Real n, Real T, Real *Y);

    /// Calculate the charge chemical potential
    Real ChargeChemicalPotential(Real n, Real T, Real *Y);

    /// Calculate the electron-lepton chemical potential
    Real ElectronLeptonChemicalPotential(Real n, Real T, Real *Y);

    /// Species fractions
    Real FrYn(Real n, Real T, Real *Y);
    Real FrYp(Real n, Real T, Real *Y);
    Real FrYa(Real n, Real T, Real *Y);
    Real FrYh(Real n, Real T, Real *Y);
    Real AN(Real n, Real T, Real *Y);
    Real ZN(Real n, Real T, Real *Y);

    /// Get the minimum enthalpy per baryon.
    Real MinimumEnthalpy();

    /// Get the minimum pressure at a given density and composition
    Real MinimumPressure(Real n, Real *Y);

    /// Get the maximum pressure at a given density and composition
    Real MaximumPressure(Real n, Real *Y);

    /// Get the minimum specific internal energy at a given density and composition
    Real MinimumSpecificInternalEnergy(Real n, Real *Y);

    /// Get the maximum specific internal energy at a given density and composition
    Real MaximumSpecificInternalEnergy(Real n, Real *Y);

    /// Get the minimum energy at a given density and composition
    Real MinimumEnergy(Real n, Real *Y);

    /// Get the maximum energy at a given density and composition
    Real MaximumEnergy(Real n, Real *Y);

  public:
    /// Calls the individual initialization functions
    void InitializeTables(std::string fname, std::string helm_fname, std::string heating_fname, Real baryon_mass);

    /// Some setters for parameters
    void SetTransition(Real n_start, Real n_end, Real T_start, Real T_end);

    /// Get the NSE value of the binding energy per baryon
    Real GetNSEBindingEnergy(Real n, Real T, Real *Y);

    /// Get the transition parameters
    void PrintParameters();

    /// Get the factor to transition between the two EOSs, as a function of density and temperature
    Real TransitionFactor(Real n, Real T) const;

    Real const GetTempTransStart() const {
      return trans_T_start;
    }

    Real const GetDensTransStart() const {
      return exp(trans_ln_start);
    }

    void GetTableBoundaries(Real &ld_n, Real &hd_n, Real &ld_t, Real &hd_t) {
      ld_n = helmholtz_eos->MaximumDensity();
      ld_t = helmholtz_eos->MaximumTemperature();
      hd_n = compose_eos->MinimumDensity();
      hd_t = compose_eos->MinimumTemperature();
    }

    /// Check if the EOS has been initialized properly.
    inline bool IsInitialized() const {
      return m_initialized;
    }

  private:
    /// Set the baryon mass.
    void SetBaryonMass(Real new_mb);

    void update_bounds();

    /// Low level inversion function, not intended for outside use
    Real temperature_from_var_trans(int iv, Real var, Real n, Real *Y) const;
    int comp_it_trans_start, comp_it_trans_end;

  protected:
    EOSCompOSE * compose_eos;
    EOSHelmholtz * helmholtz_eos;


    Real trans_T_start, trans_T_end, trans_ln_start, trans_ln_end; // Transition parameters
    // Transitions width
    Real m_trans_T_width, m_trans_ln_width;

    Real mb, max_n, min_n, max_T, min_T, max_Y[MAX_SPECIES], min_Y[MAX_SPECIES];

    // Minimum enthalpy per baryon
    Real m_min_h;

    // bool to protect against access of uninitialised table, and prevent repeated reading of table
    bool m_initialized;
};
} // namespace Primitive

#endif
