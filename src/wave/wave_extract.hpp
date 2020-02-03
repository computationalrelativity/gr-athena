#ifndef WAVE_EXTRACT_HPP
#define WAVE_EXTRACT_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wave_extract.hpp
//  \brief definitions for the WaveExtract class

#include <string>

#include "../athena.hpp"
#include "../athena_arrays.hpp"

// Forward declaration
class Mesh;
class MeshBlock;
class SphericalGrid;
class SphericalPatch;
class ParameterInput;

//! \class WaveExtract
//! \brief Extracts the l=0,m=0 component of the wave on a unit sphere
//! This class performs the global reduction
class WaveExtract {
  public:
    //! Creates the WaveExtract object
    WaveExtract(Mesh * pmesh, ParameterInput * pin);
    //! Destructor (will close output file)
    ~WaveExtract();
    //! Reduces the data from all of the SphericalPatches
    void ReduceMonopole();
    //! Write data to file
    void Write(int iter, Real time) const;
  public:
    //! Monopole term
    Real monopole;
    //! SphericalGrid for wave extraction
    SphericalGrid * psphere;
  private:
    int root;
    bool ioproc;
    std::string ofname;
    Mesh const * pmesh;
    FILE * pofile;
};

//! \class WaveExtractLocal
//! \brief Extracts the l=0,m=0 component of the wave on a unit sphere
//! This class performs the reduction on each SphericalPatch
class WaveExtractLocal {
  public:
    //! Creates the WaveExtractLocal object
    WaveExtractLocal(SphericalGrid * psphere, MeshBlock * pmb, ParameterInput * pin);
    ~WaveExtractLocal();
    //! Computes the l=0, m=0 of the given grid function
    void Decompose(AthenaArray<Real> const & u);
  public:
    //! Monopole term
    Real monopole;
    //! Patch of the spherical grid on which we are working
    SphericalPatch * ppatch;
  private:
    AthenaArray<Real> data;
    AthenaArray<Real> weight;
    Real rad;
};

#endif
