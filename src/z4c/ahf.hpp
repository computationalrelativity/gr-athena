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
#include "../utils/finite_differencing.hpp"
#include "../utils/lagrange_interp.hpp"
#include "z4c.hpp"
#include "z4c_macro.hpp"

// Forward declaration
class Mesh;
class MeshBlock;
class ParameterInput;

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
  //! Number of horizons
  int nhorizon;
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
  
  // Functions to interface with puncture tracker
  Real PuncMaxDistance();
  Real PuncMaxDistance(const int pix);
  Real PuncSumMasses();
  void PuncWeightedMassCentralPoint(Real *xc, Real *yc, Real *zc);
  bool PuncAreClose();
private:
  struct FD_ {
    int stride[3];
    //Real idx[3];
    //1st derivative
    // 1st derivative stecil
    typedef FDCenteredStencil<1, NGHOST-1> s1;

    typedef FDRightBiasedStencilBeyond<
        1, NGHOST, 1
      > sr_1B;
    typedef FDRightBiasedStencilBeyond<
        1, NGHOST, 2
      > sr_2B;
    typedef FDRightBiasedStencilBeyond<
        1, NGHOST, 3
      > sr_3B; 
    typedef FDLeftBiasedStencilBeyond<
        1, NGHOST, 1
      > sl_1B; 
    typedef FDLeftBiasedStencilBeyond<
        1, NGHOST, 2
      > sl_2B;
    typedef FDLeftBiasedStencilBeyond<
        1, NGHOST, 3
      > sl_3B; 

    // Generic first derivative
    inline Real Gx(int dir, int INIT, int END, int index, Real & u) {
      Real * pu = &u;
      int lopsided;
      if (index < INIT) {lopsided = INIT - index;}
      else if (index > END) {lopsided = END - index;}
      else {lopsided = 0;}
      
      Real out(0.);
      // Dx
      switch (lopsided) { 
        case 0:
          pu -= s1::offset*stride[dir];

          for(int n1 = 0; n1 < s1::nghost; ++n1) {
            int const n2  = s1::width - n1 - 1;
            Real const c1 = s1::coeff[n1] * pu[n1*stride[dir]];
            Real const c2 = s1::coeff[n2] * pu[n2*stride[dir]];
            out += (c1 + c2);
          }
          out += s1::coeff[s1::nghost] * pu[s1::nghost*stride[dir]];
          break;
        case 1:
          for(int n = 0; n < sr_1B::width; ++n) {
            out += sr_1B::coeff[n] * pu[(n - sr_1B::offset)*stride[dir]];
          }
          break;
        case 2:
          for(int n = 0; n < sr_2B::width; ++n) {
            out += sr_2B::coeff[n] * pu[(n - sr_2B::offset)*stride[dir]];
          }
          break;
        case 3:
          if (NGHOST == 3) {
            break;
          }
          for(int n = 0; n < sr_3B::width; ++n) {
            out += sr_3B::coeff[n] * pu[(n - sr_3B::offset)*stride[dir]];
          }
          break;
        case -1:
          for(int n = 0; n < sl_1B::width; ++n) {
            out += sl_1B::coeff[n] * pu[(n - sl_1B::offset)*stride[dir]];
          }
          break;
        case -2: 
          for(int n = 0; n < sl_2B::width; ++n) {
            out += sl_2B::coeff[n] * pu[(n - sl_2B::offset)*stride[dir]];
          }
          break;
        case -3:
          if (NGHOST == 3) {
            break;
          }
          for(int n = 0; n < sl_3B::width; ++n) {
            out += sl_3B::coeff[n] * pu[(n - sl_3B::offset)*stride[dir]];
          }
          break;
        default:
          break;
      }
      return out;
    }
  } FD;
};

#endif
