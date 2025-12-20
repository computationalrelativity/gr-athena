#ifndef HYBRID_TABLE_H
#define HYBRID_TABLE_H

//! \file hybrid_table.hpp
//  \brief Defines EOSTable, which stores information from a 1D 
//         tabulated equation of state in CompOSE format.
//
//  Tables should be generated using
//  <a href="https://bitbucket.org/dradice/pycompose">PyCompOSE</a>

///  \warning This code assumes the table to be uniformly spaced in
///           log nb

#include <cstddef>
#include <string>

#include "../../athena.hpp"
#include "eos_policy_interface.hpp"

namespace Primitive {

class HybridTable : public EOSPolicyInterface {
  public:
    enum TableVariables {
      ECLOGP  = 0,  //! log (pressure / 1 MeV fm^-3)
      ECENT   = 1,  //! entropy per baryon [kb]
      ECMUB   = 2,  //! baryon chemical potential [MeV]
      ECMUQ   = 3,  //! charge chemical potential [MeV]
      ECMUL   = 4,  //! lepton chemical potential [MeV]
      ECLOGE  = 5,  //! log (total energy density / 1 MeV fm^-3)
      ECCS    = 6,  //! sound speed [c]
      ECNVARS = 7
    };

  protected:
    /// Constructor
    HybridTable();

    /// Destructor
    ~HybridTable();

    /// Temperature from energy density
    Real TemperatureFromE(Real n, Real e, Real *Y);

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

    /// Calculate the specific internal energy per unit mass
    Real SpecificInternalEnergy(Real n, Real T, Real *Y);
    
    /// Calculate the baryon chemical potential
    [[ noreturn ]]
    Real BaryonChemicalPotential(Real n, Real T, Real *Y);

    /// Calculate the charge chemical potential
    [[ noreturn ]]
    Real ChargeChemicalPotential(Real n, Real T, Real *Y);

    /// Calculate the electron-lepton chemical potential
    [[ noreturn ]]
    Real ElectronLeptonChemicalPotential(Real n, Real T, Real *Y);

    /// Species fractions
    [[ noreturn ]]
    Real FrYn(Real n, Real T, Real *Y);
    [[ noreturn ]]
    Real FrYp(Real n, Real T, Real *Y);
    [[ noreturn ]]
    Real FrYh(Real n, Real T, Real *Y);

    [[ noreturn ]]
    Real AN(Real n, Real T, Real *Y);
    [[ noreturn ]]
    Real ZN(Real n, Real T, Real *Y);

    /// Get the minimum enthalpy per baryon.
    Real MinimumEnthalpy();

    /// Get the minimum pressure at a given density and composition
    Real MinimumPressure(Real n, Real *Y);

    /// Get the maximum pressure at a given density and composition
    Real MaximumPressure(Real n, Real *Y);

    /// Get the minimum energy at a given density and composition
    Real MinimumEnergy(Real n, Real *Y);

    /// Get the maximum energy at a given density and composition
    Real MaximumEnergy(Real n, Real *Y);

  public:
    /// Reads the table file.
    void ReadTableFromFile(std::string fname);

    /// Get the raw number density
    Real const * GetRawLogNumberDensity() const {
      return m_log_nb;
    }

    /// Get the raw table data
    Real const * GetRawTable() const {
      return m_table;
    }

    // Indexing used to access the data
    inline ptrdiff_t index(int iv, int in) const {
      return in + m_nn*iv;
    }

    /// Check if the EOS has been initialized properly.
    inline bool IsInitialized() const {
      return m_initialized;
    }

    /// Set the number of species. Throw an exception if
    /// the number of species is invalid.
    void SetNSpecies(int n);

    inline void SetThermalGamma(Real g) {
      gamma_th = (g <= 1.0) ? 1.00001 : ((g >= 2.0) ? 2.0 : g);
      gamma_th_m1 = gamma_th - 1.0;
    }

    /// Get the adiabatic constant for the thermal part.
    inline Real GetThermalGamma() const {
      return gamma_th;
    }

  private:
    ///Table Energy Density
    Real ColdEnergy(Real n);

    /// Table Pressure
    Real ColdPressure(Real n);

    /// Table Enthalpy
    Real ColdEnthalpy(Real n);

    /// Table Cs2
    Real ColdSoundSpeed(Real n);

    /// Low level evaluation function, not intended for outside use
    Real eval_at_n(int vi, Real n) const;
    /// Low level evaluation function, not intended for outside use
    Real eval_at_ln(int vi, Real ln) const;

    /// Evaluate interpolation weight for density
    void weight_idx_ln(Real *w0, Real *w1, int *in, Real log_n) const;

  private:
    // Inverse of table spacing
    Real m_id_log_nb;
    // Table size
    int m_nn;
    // Minimum enthalpy per baryon
    Real m_min_h;

    // Table storage, care should be made to store these data on the GPU later
    // Static pointers used to share access to single instance of table in memory (per MPI process)
    static Real * m_log_nb;
    static Real * m_table;

    // bool to protect against access of uninitialised table, and prevent repeated reading of table
    static bool m_initialized;
    
    // Auxiliary static variables to share data only available when table is open to those threads that do not open it
    // variables from EOSCompOSE
    static Real sm_id_log_nb;
    static int sm_nn;
    static Real sm_min_h;
    
    // Thermal Gamma
    Real gamma_th;
    Real gamma_th_m1;

    // variables from EOSPolicy
    static Real s_mb, s_max_n, s_min_n;
};

} // namespace Primitive

#endif
