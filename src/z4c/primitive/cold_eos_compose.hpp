#ifndef COLDEOS_COMPOSE_H
#define COLDEOS_COMPOSE_H

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
#include "../../globals.hpp"
#include "unit_system.hpp"

namespace Primitive {

class ColdEOSCompOSE {
  public:
    enum TableVariables {
      ECLOGN  = 0,  //! log (number density / fm^-3)
      ECLOGP  = 1,  //! log (pressure / 1 MeV fm^-3)
      ECLOGE  = 2,  //! log (total energy density / 1 MeV fm^-3)
      ECDPDN  = 3,  //! Derivative of pressure wrt. number density
      ECH     = 4,  //! enthapy per baryon [MeV]
      ECY     = 5,  //! Abundance of species
      ECNVARS = 5 + NSCALARS
    };

  protected:
    /// Constructor
    ColdEOSCompOSE();

    /// Destructor
    ~ColdEOSCompOSE();

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

    /// Calculate the specific enthalpy from the number density
    Real Enthalpy(Real n);

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
    void ReadColdSliceFromFile(std::string fname, std::string species_names[NSCALARS]);

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

    Real eval_at_general(int ii, int iv, Real h) const;
    int D0_x_2(double *f, double *x, int n, double *df);

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
