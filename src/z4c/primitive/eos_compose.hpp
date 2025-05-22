#ifndef EOS_COMPOSE_H
#define EOS_COMPOSE_H

//! \file eos_compose.hpp
//  \brief Defines EOSTable, which stores information from a tabulated
//         equation of state in CompOSE format.
//
//  Tables should be generated using
//  <a href="https://bitbucket.org/dradice/pycompose">PyCompOSE</a>

///  \warning This code assumes the table to be uniformly spaced in
///           log nb, log t, and yq

#include <cstddef>
#include <string>

#include "../../athena.hpp"
#include "eos_policy_interface.hpp"

namespace Primitive {

class EOSCompOSE : public EOSPolicyInterface {
  public:
    enum TableVariables {
      ECLOGP  = 0,  //! log (pressure / 1 MeV fm^-3)
      ECENT   = 1,  //! entropy per baryon [kb]
      ECMUB   = 2,  //! baryon chemical potential [MeV]
      ECMUQ   = 3,  //! charge chemical potential [MeV]
      ECMUL   = 4,  //! lepton chemical potential [MeV]
      ECLOGE  = 5,  //! log (total energy density / 1 MeV fm^-3)
      ECCS    = 6,  //! sound speed [c]
      ECABAR  = 7,  //! Average mass number
      ECNVARS = 8
    };

  protected:
    /// Constructor
    EOSCompOSE();

    /// Destructor
    ~EOSCompOSE();

    /// Temperature from energy density
    Real TemperatureFromE(Real n, Real e, Real *Y);

    /// Calculate the temperature from the pressure
    Real TemperatureFromP(Real n, Real p, Real *Y);

    /// Calculate the temperature from the entropy
    Real TemperatureFromEntropy(Real n, Real s, Real *Y);

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
    Real BaryonChemicalPotential(Real n, Real T, Real *Y);

    /// Calculate the charge chemical potential
    Real ChargeChemicalPotential(Real n, Real T, Real *Y);

    /// Calculate the electron-lepton chemical potential
    Real ElectronLeptonChemicalPotential(Real n, Real T, Real *Y);

    /// Calculate the average mass number
    Real Abar(Real n, Real T, Real *Y);

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

    /// Get the minimum entropy per baryon at a given density and composition
    Real MinimumEntropy(Real n, Real *Y);

    /// Get the maximum entropy per baryon at a given density and composition
    Real MaximumEntropy(Real n, Real *Y);

  public:
    /// Reads the table file.
    void ReadTableFromFile(std::string fname);

    /// Get the raw number density
    Real const * GetRawLogNumberDensity() const {
      return m_log_nb;
    }
    Real const * GetRawYq() const {
      return m_yq;
    }
    /// Get the raw number density
    Real const * GetRawLogTemperature() const {
      return m_log_t;
    }
    /// Get the raw table data
    Real const * GetRawTable() const {
      return m_table;
    }

    // Indexing used to access the data
    inline ptrdiff_t index(int iv, int in, int iy, int it) const {
      return it + m_nt*(iy + m_ny*(in + m_nn*iv));
    }

    /// Check if the EOS has been initialized properly.
    inline bool IsInitialized() const {
      return m_initialized;
    }

    /// Set the number of species. Throw an exception if
    /// the number of species is invalid.
    void SetNSpecies(int n);

    /// Set the maxium density.
    /// Values higher than the max of the table will lead to extrapolation
    void SetMaximumDensity(Real n_max) {
      max_n = n_max;
    }

    /// Set the maxium termperature.
    /// Values higher than the max of the table will lead to extrapolation
    void SetMaximumTemperature(Real T_max) {
      max_T = T_max;
    }

  private:
    /// Low level function, not intended for outside use
    Real temperature_from_var(int vi, Real var, Real n, Real Yq) const;
    /// Low level evaluation function, not intended for outside use
    Real eval_at_nty(int vi, Real n, Real T, Real Yq) const;
    /// Low level evaluation function, not intended for outside use
    Real eval_at_lnty(int vi, Real ln, Real lT, Real Yq) const;

    /// Evaluate interpolation weight for density
    void weight_idx_ln(Real *w0, Real *w1, int *in, Real log_n) const;
    /// Evaluate interpolation weight for composition
    void weight_idx_yq(Real *w0, Real *w1, int *iy, Real yq) const;
    /// Evaluate interpolation weight for temperature
    void weight_idx_lt(Real *w0, Real *w1, int *it, Real log_t) const;

  private:
    // Inverse of table spacing
    Real m_id_log_nb, m_id_log_t, m_id_yq;
    // Table size
    int m_nn, m_nt, m_ny;
    // Minimum enthalpy per baryon
    Real m_min_h;

    // Table storage, care should be made to store these data on the GPU later
    // Static pointers used to share access to single instance of table in memory (per MPI process)
    static Real * m_log_nb;
    static Real * m_log_t;
    static Real * m_yq;
    static Real * m_table;

    // bool to protect against access of uninitialised table, and prevent repeated reading of table
    static bool m_initialized;

    // Auxiliary static variables to share data only available when table is open to those threads that do not open it
    // variables from EOSCompOSE
    static Real sm_id_log_nb, sm_id_log_t, sm_id_yq;
    static int sm_nn, sm_nt, sm_ny;
    static Real sm_min_h;

    // variables from EOSPolicy
    static Real s_mb, s_max_n, s_min_n, s_max_T, s_min_T, s_max_Y[MAX_SPECIES], s_min_Y[MAX_SPECIES];
    // these correspond to defined but unused vars in EOSPolicy
    // static Real s_max_P, s_min_P, s_max_e, s_min_e;
};

} // namespace Primitive

#endif
