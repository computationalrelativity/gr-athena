#ifndef AHF_HPP
#define AHF_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ahf.hpp
//  \brief definitions for the AHF class

// TODO
// * check public/private
// * check for missing definitions

#include <string>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"
#include "../utils/lagrange_interp.hpp"

// Forward declaration
class Mesh;
class MeshBlock;
class ParameterInput;

// Max number of horizons 

#define SQ(X) ((X)*(X))
#define NDIM (3)

//! \class AHF
//! \brief Apparent Horizon Finder
class AHF {
  
public:
  //! Creates the AHF object
  AHF(Mesh * pmesh, ParameterInput * pin, int nh);
  //! Destructor (will close output file)
  ~AHF();
  //!
  void Find(int iter, Real time);
  //!
  void Write(int iter, Real time);

  //! Horizon found
  bool ah_found;
  //! Initial guess
  Real initial_radius;
  Real expand_guess;
  //! Center
  Real center[3];
  //! Fast flow parameters
  Real hmean_tol;
  Real mass_tol;
  int flow_iterations;
  bool verbose;
  //! Multipoles
  int lmax;
  //! Grid points
  int ntheta, nphi;

  //! n surface follows the puncture tracker if use_puncture[n] > 0 
  int use_puncture;
  //! n surface uses the punctures' mass-weighted center 
  //bool use_puncture_massweighted_center[NHORIZON];
  bool use_puncture_massweighted_center;
  //! Distance in M at which BHs are considered as merged
  Real merger_distance;

  //! start and stop times for each surface
  Real start_time;
  Real stop_time;
  //! compute every n iterations
  int compute_every_iter;
  
private:
  int npunct;
  int lmax1;
  int lmpoints;
  int nh;
  bool wait_until_punc_are_close;
  Real compute_start_time, compute_stop_time;
  //! Number of horizons
  int nstart, nhorizon;
  int fastflow_iter=0;
  //! Arrays of Legendre polys and drvts
  AthenaArray<Real> P, dPdth, dPdth2;
  //! Arrays of sphericasl harmonics and drvts
  AthenaArray<Real> Y0, Yc, Ys;
  AthenaArray<Real> dY0dth, dYcdth, dYsdth, dYcdph, dYsdph;
  AthenaArray<Real> dY0dth2, dYcdth2, dYcdthdph, dYsdth2, dYsdthdph, dYcdph2, dYsdph2;
  //! Arrays for spectral coefs 
  AthenaArray<Real> a0; 
  AthenaArray<Real> ac; 
  AthenaArray<Real> as; 
  Real last_a0;
  //! Arrays for the various fields on the sphere
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> g;
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> K;
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> dg; 
  AthenaArray<Real> rr;
  // Array computed in SurfaceIntegrals
  AthenaArray<Real> rho;
  //! Indexes of surface integrals
  enum {
    iarea,
    icoarea,
    ihrms,
    ihmean,
    iSx, iSy, iSz,
    invar
  };
  //! Array of surface integrals
  Real integrals[invar];
  //! Indexes of horizon quantities
  enum {
    harea,
    hcoarea,
    hhrms,
    hhmean,
    hSx, hSy, hSz, hS,
    hmass,
    hmeanradius,
    hnvar
  };
  //! Array of horizon quantities
  Real ah_prop[hnvar];

  //! Flag points 
  AthenaArray<int> havepoint;

  void MetricDerivatives(MeshBlock * pmb);
  void MetricInterp(MeshBlock * pmb);
  void SurfaceIntegrals();
  void FastFlowLoop();
  void UpdateFlowSpectralComponents();
  void RadiiFromSphericalHarmonics();
  void InitialGuess();
  void ComputeSphericalHarmonics();
  void ComputeLegendre(const Real theta);
  int lmindex(const int l, const int m);
  int tpindex(const int i, const int j);
  Real th_grid(const int i);
  Real ph_grid(const int j);
  Real dth_grid();
  Real dph_grid();
  void factorial_list(Real * fac, const int maxn); 

  Mesh const * pmesh;

  int root;
  int ioproc;
  std::string ofname_summary;
  std::string ofname_shape;
  FILE * pofile_summary;
  FILE * pofile_shape;

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

  // Functions to interface with puncture tracker
  Real PuncMaxDistance();
  Real PuncMaxDistance(const int pix);
  Real PuncSumMasses();
  void PuncWeightedMassCentralPoint(Real *xc, Real *yc, Real *zc);
  bool PuncAreClose();
};

#endif
