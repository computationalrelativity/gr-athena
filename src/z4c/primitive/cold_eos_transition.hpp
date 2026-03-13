#ifndef COLDEOS_TRANSITION_H
#define COLDEOS_TRANSITION_H

//! \file eos_transition.hpp
//  \brief Defines EOSTable, which stores information from a tabulated
//         equation of state in CompOSE format.
//         It is identical to cold_eos_compose except for the UpdateBaryonMass
//         call.
//
//  Tables should be generated using
//  <a href="https://bitbucket.org/dradice/pycompose">PyCompOSE</a>

///  \warning This code assumes the table to be uniformly spaced in log nb

#include <cstddef>
#include <string>

#include "../../athena.hpp"
#include "../../globals.hpp"
#include "unit_system.hpp"

namespace Primitive {

class ColdEOSTransition {
  public:
    enum TableVariables {
      ECLOGN  = 0,  //! log (number density / fm^-3)
      ECLOGP  = 1,  //! log (pressure / 1 MeV fm^-3)
      ECLOGE  = 2,  //! log (total energy density / 1 MeV fm^-3)
      ECDPDN  = 3,  //! Derivative of pressure wrt. number density
      ECENT   = 4,  //! entropy per baryon [kb]
      ECH     = 5,  //! enthapy per baryon [MeV]
      ECY     = 6,  //! Abundance of species
      ECABAR  = 7,  //! Average baryon number per nucleon
      ECNVARS = 8
    };

  protected:
    /// Constructor
    ColdEOSTransition();

    /// Destructor
    ~ColdEOSTransition();

    /// Calculate the pressure from the number density
    Real Pressure(Real n);

    /// Calculate the energy from the number density
    Real Energy(Real n);

    /// Calculate the derivative of the pressure wrt. the numberdensity from the number density
    Real dPdn(Real n);

    /// Calculate the specific internal energy from the number density
    Real SpecificInternalEnergy(Real n);

    /// Calculate the abundance of species iy from the number density
    Real Y(Real n, int iy);

    /// Calculate the specific entropy from the number density
    Real Entropy(Real n);

    /// Calculate the specific enthalpy from the number density
    Real Enthalpy(Real n);

    /// Calculate the number density from the pressure
    Real DensityFromPressure(Real P);

    /// Number of particle species
    int n_species;
    /// Baryon mass
    Real mb;
    /// maximum number density
    Real max_n;
    /// minimum number density
    Real min_n;
    /// temperature of the slice
    Real T;
    /// Code unit system
    UnitSystem* code_units;
    /// ColdEOS unit system
    UnitSystem* eos_units;

  public:
    /// Reads the cold slice table from file.
    void ReadColdSliceFromFile(std::string fname);

    /// Update the baryon mass (in MeV)
    void UpdateBaryonMass(Real new_mb) {
      mb = new_mb;
    }

    /// Dumps the eos_akmalpr.d file that lorene routines expect
    void DumpLoreneEOSFile(std::string fname);

    // Indexing used to access the data
    inline ptrdiff_t index(int iv, int ix) const {
      return ix + m_np*iv;
    }

    /// Get the raw table data
    Real const * GetRawTable() const {
      return m_table;
    }

    /// Check if the EOS has been initialized properly.
    inline bool IsInitialized() const {
      return m_initialized;
    }


  private:
    /// Internal evaluation functions
    template<int LIX_EXTRAPOLATE>
    void weight_idx_ln(Real *w0, Real *w1, int *in, Real log_n) const;

    template<int LIX_EXTRAPOLATE>
    Real eval_at_n(int iv, Real n) const;
    template<int LIX_EXTRAPOLATE>
    Real eval_at_ln(int iv, Real log_n) const;

    Real eval_at_general(int iv_in, int iv_out, Real v) const;
    int D0_x_2(double *f, double *x, int n, double *df);

    Real linterp1d(Real x, Real * xp, Real *fp) const;

  private:
    // number of points in the table
    int m_np;

    // Table storage
    Real * m_table;

    bool m_initialized;

    Real m_id_log_nb;

    int i_lorene_cut;

};

} // namespace Primitive

#endif
