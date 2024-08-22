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
//#include "../utils/lagrange_interp.hpp"
//#include "z4c.hpp"
//#include "z4c_macro.hpp"

// Forward declaration
class Mesh;
class MeshBlock;
class ParameterInput;

//! \class WaveExtractRWZ
//! \brief RWZ waveform extraction 
class WaveExtractRWZ {

public:
  //! Creates the WaveExtractRWZ object
  WaveExtractRWZ(Mesh * pmesh, ParameterInput * pin, int n);
  //! Destructor 
  ~WaveExtractRWZ();
    
  //! Step 1 ... 4 of the extraction:
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
  AthenaArray<Real> Psie,Psie;
  AthenaArray<Real> Psie_dyn,Psie_dyn;
  AthenaArray<Real> Psio_sch,Psio_sch;
  AthenaArray<Real> Qplus, Qstar;

private:

  //! Functions for grid on spheres
  Real coord_theta(const int i);
  Real coord_phi(const int j);
  Real dth_grid();
  Real dph_grid();
  int TPIndex(const int i, const int j);
  void FlagSpherePointsContained(MeshBlock * pmb);

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
  void InterpMetricToSphere(MeshBlock * pmb);
  void TransformMetricCarToSph();
  void MasterFuns();
  void MultipolesGaugeInvariant();
  
  //! Functions for output
  std::string OutputFileName(std::string base);

  //! Schwarzschild radius, its time drvt, 
  //  the (dr_schwarzschild/dr_isotropic) Jacobians and its time drvt
  Real rsch, dt_rsch;
  Real drsch_dri, d2rsch_dri2, drsch_dri_dot;
  Real dri_drsch, d2ri_drsch2, dri_drsch_dot;
    
  //! Grid points in theta and phi
  AthenaArray<Real> th_grid, ph_grid;
  
  //! Flag sphere points on this rank
  AthenaArray<int> havepoint;
  
  //! Arrays for the various fields on the sphere
  
  // 3+1 metric
  AthenaTensor<Real, TensorSymm::SYM2, 2, 2> gamma_dd;
  AthenaTensor<Real, TensorSymm::SYM2, 2, 3> dgamma_ddd;
  AthenaTensor<Real, TensorSymm::SYM2, 2, 2> dot_gamma_dd;  
  AthenaTensor<Real, TensorSymm::NONE, 2, 1> beta_u;  
  AthenaTensor<Real, TensorSymm::NONE, 2, 1> dot_beta_u;
  AthenaTensor<Real, TensorSymm::NONE, 2, 1> beta_d;
  AthenaTensor<Real, TensorSymm::NONE, 2, 0> alpha;
  AthenaTensor<Real, TensorSymm::NONE, 2, 0> dot_alpha;

  //! Spherical harmonics on the sphere (complex -> 2 components)
  AthenaArray<Real> Y, Yth, Yph, X, W;
  
  //! Spherical metric on M^2 (pointwise)
  AthenaTensor<Real, TensorSymm::SYM2, 0, 2> g_dd;
  AthenaTensor<Real, TensorSymm::SYM2, 0, 2> g_uu;
  AthenaTensor<Real, TensorSymm::SYM2, 0, 2> g_dr_dd; //d/dr g_AB
  AthenaTensor<Real, TensorSymm::SYM2, 0, 2> g_dt_dd; //d/dt g_AB
  AthenaTensor<Real, TensorSymm::SYM2, 0, 2> g_dr_uu; //d/dr g^AB
  AthenaTensor<Real, TensorSymm::SYM2, 0, 2> g_dt_uu; //d/dt g^AB
  AthenaTensor<Real, TensorSymm::SYM2, 0, 3> Gamma_udd; // Christoffels (time-independent)
  AthenaTensor<Real, TensorSymm::SYM2, 0, 3> Gamma_dyn_udd; // Christoffels (time-dep)
  Real norm_Delta_Gamma;
  
  //! Multipoles (complex)
  AthenaArray<Real> h00, h01, h11, h0, h1, G, K; // even
  AthenaArray<Real> h00_dr, h01_dr, h11_dr, h0_dr, h1_dr, G_dr, K_dr; // d/dr drvts
  AthenaArray<Real> h00_dt, h01_dt, h11_dt, h0_dt, h1_dt, G_dt, K_dt; // d/dt drvts
  AthenaArray<Real> H0, H01, H; // odd
  AthenaArray<Real> H0_dr, H01_dr, H_dr; // d/dr drvts
  AthenaArray<Real> H0_dt, H01_dt, H_dt; // d/dt drvts

  //! Gauge-invariant multipoles
  AthenaTensor<Real, TensorSymm::SYM2, 0, 2> kappa_dd;
  AthenaTensor<Real, TensorSymm::NONE, 0, 1> kappa_d;
  AthenaArray<Real> kappa, Tr_kappa_dd;
  Real norm_Tr_kappa_dd;
  
  //! Indexes for real/imag (0/1) part of complex fields
  enum{Re, Im, RealImag};
  
  //! Arrays for sphere integrals
  Real integrals_background[NVBackground];
  enum {
    Irsch2, Idrsch_dri, Id2rsch_dri2, // Areal radius and drvts wrt R
    Idt_rsch,  Idrsch_dri_dot,
    Ig00, Ig0r, Igrr, // g_AB on M^2
    Igrt, Igtt, Igpp,
    Idr_g00,  Idr_g0r, Idr_grr, // dg_AB/dr 
    Idr_gtt,
    Idt_g00, Idt_g0r, Idt_grr, // dg_AB/dt
    NVBackground,
  };

  Real * integrals_multipoles;
  enum {
    Ih00, Ih01, Ih11, // even
    Ih0, Ih1,
    IG, IK,
    Ih00_dr, Ih01_dr, Ih11_dr, // even d/dr drvts
    Ih0_dr, Ih1_dr,
    IG_dr, IK_dr,
    Ih00_dt, Ih01_dt, Ig11_dt, // even d/dt drvts
    Ih0_dt, Ih1_dt,
    IG_dt, IK_dt,
    IH0, IH1, // odd
    IH,
    IH0_dr, IH1_dr, // odd d/dr drvts
    IH_dr,
    IH0_dt, IH1_dt, // odd d/dt drvts
    IH_dt,    
    NVMultipoles,
  };

  //! msc
  Mesh const * pmesh;
  bool verbose;    
  bool bitant;
  int root;
  int ioproc;
  int outprec;
  
  // Indexes for basenames of various output files
  enum {
    Iof_diagnostic,// This should be first!
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
