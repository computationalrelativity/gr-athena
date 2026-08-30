#ifndef WAVE_EXTRACT_HPP
#define WAVE_EXTRACT_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file wave_extract.hpp
//  \brief definitions for the WaveExtract class

#include <string>
#include <vector>

#include "../athena.hpp"
#include "../athena_arrays.hpp"

// Forward declaration
class Mesh;
class MeshBlock;
class SphericalGrid;
class SphericalPatch;
class ParameterInput;

//! \class WaveExtractHarmonics
//! \brief Fixed spin-weighted harmonics for one geodesic grid
class WaveExtractHarmonics
{
  public:
  WaveExtractHarmonics(SphericalGrid const* psphere, int lmax, bool bitant);

  inline Real const* Ylm(int l, int m, int vertex) const
  {
    const int mode = l*l - 4 + (m + l);
    return &ylm_[2 * (mode * num_vertices_ + vertex)];
  }

  inline Real BitantZFac(int vertex) const
  {
    return bitant_z_fac_[vertex];
  }

  private:
  int num_vertices_;
  std::vector<Real> ylm_;
  std::vector<Real> bitant_z_fac_;
};

//! \class WaveExtract
//! \brief Extracts the l m  components of the wave on a unit sphere
//! This class performs the global reduction
class WaveExtract
{
  public:
  //! Creates the WaveExtract object
  WaveExtract(Mesh* pmesh, ParameterInput* pin, int n);
  //! Destructor (will close output file)
  ~WaveExtract();
  //! Accumulates the local data from all SphericalPatches.
  void AccumulateMultipole();
  //! Reduces all extraction radii to rank zero.
  static void ReduceAll(std::vector<WaveExtract*>& wave_extractions);
  //! Write data to file
  void Write(int iter, Real time) const;

  public:
  //!  Array of lm modes
  AthenaArray<Real> psi;
  //! SphericalGrid for wave extraction
  int rad_id;
  SphericalGrid* psphere;

  private:
  int lmax;
  std::string ofname;
  Mesh const* pmesh;
  FILE* pofile;
};

//! \class WaveExtractLocal
//! \brief Extracts the l m components of the wave on a unit sphere
//! This class performs the reduction on each SphericalPatch
class WaveExtractLocal
{
  public:
  //! Creates the WaveExtractLocal object
  WaveExtractLocal(SphericalGrid* psphere,
                   MeshBlock* pmb,
                   ParameterInput* pin,
                   int n,
                   WaveExtractHarmonics const* pwave_harmonics);
  ~WaveExtractLocal();
  //! Computes the l m modes of the given grid function
  void Decompose_multipole(AthenaArray<Real> const& u_R,
                           AthenaArray<Real> const& u_I);

  public:
  //! lm projections
  AthenaArray<Real> psi;
  //! Patch of the spherical grid on which we are working
  SphericalPatch* ppatch;

  private:
  WaveExtractHarmonics const* pwave_harmonics;
  AthenaArray<Real> datareal;
  AthenaArray<Real> dataim;
  AthenaArray<Real> weight;
  Real rad;
  int lmax;
};

#endif
