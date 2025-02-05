#ifndef AHF_HPP
#define AHF_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ahf.hpp
//  \brief definitions for the AHF class

#include <string>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"
#include "../utils/lagrange_interp.hpp"
#include "z4c.hpp"
#include "z4c_macro.hpp"

// Forward declaration
class Mesh;
class MeshBlock;
class ParameterInput;

#define INTERP_ORDER_2 (0) // Default: 2*NGHOST-1

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
  //!
  bool CalculateMetricDerivatives(int iter, Real time);
  //!
  bool DeleteMetricDerivatives(int iter, Real time);

  Real GetHorizonRadius() const
  { return ah_prop[hmeanradius];}
  //! Horizon found
  bool ah_found;
  //! Initial guess
  Real initial_radius;
  //! Minimum radius
  Real rr_min;
  Real expand_guess;
  //! Center
  Real center[3];
  //! Fast flow parameters
  Real hmean_tol;
  Real mass_tol;
  int flow_iterations;
  Real flow_alpha_beta_const;
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
  bool bitant;
  bool use_stored_metric_drvts;
  //! Number of horizons
  int nstart, nhorizon;
  //! Arrays for the grid and quadrature weights
  AthenaArray<Real> th_grid, ph_grid, weights;
#if (INTERP_ORDER_2)
  static const int metric_interp_order = 2;
#else
  static const int metric_interp_order = 2*NGHOST-1;
#endif
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
  AthenaArray<Real> rr, rr_dth, rr_dph;
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
    hminradius,
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
  void UpdateFlowSpectralComponents_old();
  void RadiiFromSphericalHarmonics();
  void InitialGuess();
  void ComputeSphericalHarmonics();
  void ComputeLegendre(const Real theta);
  int lmindex(const int l, const int m);
  int tpindex(const int i, const int j);
  void GLQuad_Nodes_Weights(const Real a, const Real b, Real * x, Real * w, const int n);
  void SetGridWeights(std::string method);
  void factorial_list(Real * fac, const int maxn);

  Mesh const * pmesh;

  int root;
  int ioproc;
  std::string ofname_summary;
  std::string ofname_shape;
  std::string ofname_verbose;
  FILE * pofile_summary;
  FILE * pofile_shape;
  FILE * pofile_verbose;

  // Functions to interface with puncture tracker
  Real PuncMaxDistance();
  Real PuncMaxDistance(const int pix);
  Real PuncSumMasses();
  void PuncWeightedMassCentralPoint(Real *xc, Real *yc, Real *zc);
  bool PuncAreClose();

};

#endif
