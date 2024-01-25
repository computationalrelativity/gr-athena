#ifndef EJECTA_HPP
#define EJECTA_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ejecta.hpp
//  \brief definitions for the Ejecta class

// TODO
// * check public/private
// * check for missing definitions

#include <string>
#include <hdf5.h>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"
#include "../utils/lagrange_interp.hpp"
#include "z4c.hpp"

// Forward declaration
class Mesh;
class MeshBlock;
class ParameterInput;

#define SQ(X) ((X)*(X))
#define NDIM (3)
#define NOTHER (1)

//! \class Ejecta
//! \brief Ejecta extraction
class Ejecta {
  
public:
  //! Creates the AHF object
  Ejecta(Mesh * pmesh, ParameterInput * pin, int nrad);
  //! Destructor (will close output file)
  ~Ejecta();
  //!
  //!
  //void Write(int iter, Real time);
  //!

  Real radius;
  bool verbose;
  //! Grid points
  int ntheta, nphi;

  //! start and stop times for each surface
  Real start_time;
  Real stop_time;
  //! compute every n iterations
  int compute_every_iter;

  void Calculate(int iter, Real time);
  void Write(int iter, Real time);

private:
  int nr;
  bool bitant;
  //! Number of horizons
  int nstart, nrad;
  int fastflow_iter=0;
  AthenaArray<Real> prim[NHYDRO], cons[NHYDRO], Y[NSCALARS], T, Bcc[3];
  AthenaArray<Real> z4c[Z4c::N_Z4c], adm[Z4c::N_ADM];
  AthenaArray<Real> other[NOTHER];
  AthenaArray<Real> theta, phi;

  //! Flag points 
  AthenaArray<int> havepoint;

  void Interp(MeshBlock * pmb);
  int tpindex(const int i, const int j);
  Real th_grid(const int i);
  Real ph_grid(const int j);
  Real dth_grid();
  Real dph_grid();

  Mesh const * pmesh;

  int root;
  int ioproc;
  std::string ofname_summary;
  FILE * pofile_summary;

  // Functions taken from Z4c ...
  // ... compute spatial determinant of a 3x3  matrix
  Real SpatialDet(Real const gxx, Real const gxy, Real const gxz,
                  Real const gyy, Real const gyz, Real const gzz);

  // ... compute inverse of a 3x3 matrix
  void SpatialInv(Real const detginv,
                  Real const gxx, Real const gxy, Real const gxz,
                  Real const gyy, Real const gyz, Real const gzz,
                  Real * uxx, Real * uxy, Real * uxz,
                  Real * uyy, Real * uyz, Real * uzz);
  // ... compute trace of a rank 2 covariant spatial tensor
  Real Trace(Real const detginv,
             Real const gxx, Real const gxy, Real const gxz,
             Real const gyy, Real const gyz, Real const gzz,
             Real const Axx, Real const Axy, Real const Axz,
             Real const Ayy, Real const Ayz, Real const Azz);

};

#endif
