#ifndef WAVE_EXTRACT_RWZ_HPP
#define WAVE_EXTRACT_RWZ_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wave_extract_rwz.hpp
//  \brief Definitions for the WaveExtractRWZ class

#include <string>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"
#include "../utils/tensor.hpp"
//#include "../utils/lagrange_interp.hpp"
//#include "z4c.hpp" 
//#include "z4c_macro.hpp"

// Forward declaration
class Mesh;
class MeshBlock;
class ParameterInput;

//#define NDIM (3) // already defined
#define MDIM (2)

//! \class WaveExtractRWZ
//! \brief RWZ waveform extraction 
class WaveExtractRWZ {

public:
  //! Creates the WaveExtractRWZ object
  WaveExtractRWZ(Mesh * pmesh, ParameterInput * pin, int n);
  //! Destructor 
  ~WaveExtractRWZ();
    
  //! Step 1 ... 4 of the extraction:
  void FlagSpherePointsContainedMesh();
  void MetricToSphere();
  void BackgroundReduce();
  void MultipoleReduce();
  void Write(int iter, Real time);
  
  //! Sphere parameters
  Real Radius; // Coordinate radius
  Real center[3];
  Real Schwarzschild_radius;
  Real Schwarzschild_mass;
  
  //! Grid points
  int Ntheta, Nphi;

  //! Id of this extraction radius
  int Nrad;

  //! Multipoles
  int lmax;
  int lmpoints;
  
  //! Master functions & multipoles
  AthenaArray<Real> Psie,Psio;
  AthenaArray<Real> Psie_dyn,Psio_dyn;
  AthenaArray<Real> Psie_sch,Psio_sch;
  AthenaArray<Real> Qplus, Qstar;
  
private:
  
  static const int metric_interp_order = 2*NGHOST-1;

  //! Functions for grid on spheres
  Real coord_theta(const int i);
  Real coord_phi(const int j);
  Real dth_grid();
  Real dph_grid();
  int TPIndex(const int i, const int j);
  void FlagSpherePointsContained(MeshBlock * pmb);
  void SetWeightsIntegral(std::string method);
  void GLQuad_Nodes_Weights(const Real a, const Real b, Real * x, Real * w, const int n);

  //! Functions for spherical harmonics
  int MPoints(const int l);
  int MIndex(const int l, const int m);
  Real RWZnorm(const int l);  
  Real Factorial(const int n); 
  Real SphHarm_Plm(const int l, const int m, const Real x);
  void SphHarm_Ylm(const int l, const int m, const Real theta, const Real phi,
		   Real * YlmR, Real * YlmI);
  void SphHarm_Ylm_a(const int l, const int m, const Real theta, const Real phi,
		     Real * YthR, Real * YthI, Real * YphR, Real * YphI,
		     Real * XR, Real * XI, Real * WR, Real * WI);
  void ComputeSphericalHarmonics();

  //! Helper functions
  Real LeviCivitaSymbol(const int a, const int b, const int c);
  void InterpMetricToSphere(MeshBlock * pmb);
  void MasterFuns();
  void MultipolesGaugeInvariant();
  void SphOrthogonality();
  
  //! Functions for output
  std::string OutputFileName(std::string base);

  //! Schwarzschild radius, its time drvt, 
  //  the (dr_schwarzschild/dr_isotropic) Jacobians and its time drvt
  Real rsch, dot_rsch, dot2_rsch;
  Real drsch_dri, d2rsch_dri2, drsch_dri_dot;
  Real dri_drsch, d2ri_drsch2, dri_drsch_dot;
    
  //! Grid points in theta and phi
  AthenaArray<Real> th_grid, ph_grid;
  AthenaArray<Real> weights;
  
  //! Flag sphere points on this rank
  AthenaArray<int> havepoint;
  
  //! Arrays for the various fields on the sphere
  
  //! 3+1 metric on the sphere
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd;
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> dr_gamma_dd;
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> dr2_gamma_dd; //TODO 2nd dvtrs not yet implemented
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> dot_gamma_dd;
  
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> beta_d;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dr_beta_d;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dot_beta_d;
  
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> beta_u;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dr_beta_u;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dot_beta_u;

  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> beta2;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> dr_beta2;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> dot_beta2;
  
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> alpha;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> dr_alpha;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> dot_alpha;
  
  //! Spherical harmonics on the sphere (complex -> 2 components)
  AthenaArray<Real> Y, Yth, Yph, X, W;
  
  //! Spherical metric on M^2
  utils::tensor::TensorPointwise<Real, utils::tensor::Symmetries::SYM2, MDIM, 2> g_dd;
  utils::tensor::TensorPointwise<Real, utils::tensor::Symmetries::SYM2, MDIM, 2> g_uu;
  utils::tensor::TensorPointwise<Real, utils::tensor::Symmetries::SYM2, MDIM, 2> g_dr_dd; //d/dr g_AB
  utils::tensor::TensorPointwise<Real, utils::tensor::Symmetries::SYM2, MDIM, 2> g_dot_dd; //d/dt g_AB
  utils::tensor::TensorPointwise<Real, utils::tensor::Symmetries::SYM2, MDIM, 2> g_dr_uu; //d/dr g^AB
  utils::tensor::TensorPointwise<Real, utils::tensor::Symmetries::SYM2, MDIM, 2> g_dot_uu; //d/dt g^AB
  //TODO 2nd drvts
  utils::tensor::TensorPointwise<Real, utils::tensor::Symmetries::SYM2, MDIM, 3> Gamma_udd; // Christoffels (time-independent)
  utils::tensor::TensorPointwise<Real, utils::tensor::Symmetries::SYM2, MDIM, 3> Gamma_dyn_udd; // Christoffels (time-dep)
  Real norm_Delta_Gamma;
  
  //! Multipoles (complex)
  AthenaArray<Real> h00, h01, h11, h0, h1, G, K; // even
  AthenaArray<Real> h00_dr, h01_dr, h11_dr, h0_dr, h1_dr, G_dr, K_dr; // d/dr drvts
  AthenaArray<Real> h00_dot, h01_dot, h11_dot, h0_dot, h1_dot, G_dot, K_dot; // d/dt drvts
  AthenaArray<Real> H0, H01, H, H1; // odd
  AthenaArray<Real> H0_dr, H01_dr, H_dr, H1_dr; // d/dr drvts
  AthenaArray<Real> H0_dot, H01_dot, H_dot, H1_dot; // d/dt drvts

  //! Gauge-invariant multipoles
  AthenaArray<Real> kappa_00, kappa_01, kappa_11, kappa_0, kappa_1; // SB lets not use this!
  //utils::tensor::TensorPointwise<Real, utils::tensor::Symmetries::SYM2, MDIM, 2> kappa_dd;
  //utils::tensor::TensorPointwise<Real, utils::tensor::Symmetries::NONE, MDIM, 1> kappa_d;
  // SB The above variables are defined at a points, but the kappa's have a multipolar index. These var type was incorrect. They require 1 dimension for the multipolar index and 1 for the Re/Im part, the following should work:
  AthenaTensor<Real, TensorSymm::SYM2, 2, 2> kappa_dd;
  AthenaTensor<Real, TensorSymm::SYM2, 2, 1> kappa_d;
  AthenaArray<Real> kappa, Tr_kappa_dd;
  Real norm_Tr_kappa_dd;
  
  //! Indexes for real/imag (0/1) part of complex fields
  enum{Re, Im, RealImag};
  
  //! Arrays for sphere integrals
  enum {
    I_ADM_M,
    I_ADM_Px, I_ADM_Py, I_ADM_Pz,
    I_ADM_Jx, I_ADM_Jy, I_ADM_Jz, 
    NVADM,
  };
  Real integrals_adm[NVADM];

  enum {
    // ADM 
    I_adm_M,
    I_adm_Px, I_adm_Py, I_adm_Pz,
    I_adm_Jx, I_adm_Jy, I_adm_Jz, 
    // Schw. radius & co
    Irsch2, Idrsch_dri, Id2rsch_dri2, // Areal radius and drvts wrt R
    Idot_rsch2,  Idrsch_dri_dot,
    // Background 2-metric
    Ig00, Ig0R, IgRR, // g_AB on M^2
    IgRt, Igtt, Igpp,
    IdR_g00,  IdR_g0R, IdR_gRR, // dg_AB/dr 
    IdR_gtt,
    Idot_g00, Idot_g0R, Idot_gRR, // dg_AB/dt
    //TODO 2nd drvts
    NVBackground,
  };
  Real integrals_background[NVBackground];

  Real * integrals_multipoles;
  enum {
    Ih00, Ih01, Ih11, // even
    Ih0, Ih1,
    IG, IK,
    Ih00_dr, Ih01_dr, Ih11_dr, // even d/dr drvts
    Ih0_dr, Ih1_dr,
    IG_dr, IK_dr,
    Ih00_dot, Ih01_dot, Ih11_dot, // even d/dt drvts
    Ih0_dot, Ih1_dot,
    IG_dot, IK_dot,
    IH0, IH1, // odd
    IH,
    IH0_dr, IH1_dr, // odd d/dr drvts
    IH_dr,
    IH0_dot, IH1_dot, // odd d/dt drvts
    IH_dot,    
    NVMultipoles,
  };

  //! Options for areal radius
  enum{
    areal,
    areal_simple,
    average_schw,
    schw_gthth,
    schw_gphph,
    NOptRadius,
  };
  static char const * const ArealRadiusMethod[NOptRadius];
  int method_areal_radius;

  //! msc
  Mesh const * pmesh;
  bool verbose;    
  bool bitant;
  bool subtract_background;
  int root;
  int ioproc;
  int outprec;
  
  // Indexes for basenames of various output files
  enum {
    Iof_diagnostic,// This should be first!
    Iof_adm, // This should be second!
    Iof_Psie, 
    Iof_Psio, 
    Iof_Psie_dyn, 
    Iof_Psio_dyn,
    Iof_Qplus,
    Iof_Qstar,
    Iof_hlm, // This should be last!
    Iof_Num, 
  };
  std::string ofbname[Iof_Num];
  std::string ofname;
  FILE * pofile;
  std::ofstream outfile;

};

#endif
