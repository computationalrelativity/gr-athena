#ifndef WAVE_EXTRACT_RWZ_HPP
#define WAVE_EXTRACT_RWZ_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file wave_extract_rwz.hpp
//  \brief Definitions for the WaveExtractRWZ class

#include <string>

#include "../athena_aliases.hpp"
#include "../utils/grid_theta_phi.hpp"
#include "../utils/lagrange_interp.hpp"

// Forward declaration
class Mesh;
class MeshBlock;
class ParameterInput;

using namespace gra::aliases;

// output extra field (DEBUG)
#define WAVE_EXTRACT_RWZ_EXTRAOUTPUT (0)  // 0=false, 1=true

//! \class WaveExtractRWZ
//! \brief RWZ waveform extraction
class WaveExtractRWZ
{
  public:
  // -- Construction / destruction --------------------------------------------
  WaveExtractRWZ(Mesh* pmesh, ParameterInput* pin, int n);
  ~WaveExtractRWZ();

  // -- Main API --------------------------------------------------------------
  void MetricToSphere();
  void BackgroundReduce();
  void MultipoleReduce();
  void Write(int iter, Real time);

  // -- Grid infrastructure (public: needed by AMR teardown hook) -------------
  static const int metric_interp_order = 2 * NGHOST - 1;
  GridThetaPhi<LagrangeInterpND<metric_interp_order, 3, true>> grid_;

  private:
  // -- Configuration (set once from ParameterInput) --------------------------
  struct
  {
    int lmax;
    Real Radius;
    Real center[3];
    bool verbose;
    bool bitant;
    bool subtract_background;
    int mpi_root;
    int outprec;
    int method_areal_radius;
  } opt;

  // -- Derived constants -----------------------------------------------------
  int Nrad;
  int lmpoints;

  // -- Schwarzschild quantities ----------------------------------------------
  Real Schwarzschild_radius;
  Real Schwarzschild_mass;
  Real rsch, dot_rsch, dot2_rsch;
  Real drsch_dri, d2rsch_dri2, drsch_dri_dot, d3rsch_dri3;
  Real dri_drsch, d2ri_drsch2, dri_drsch_dot, d3ri_drsch3;

  // -- 3+1 metric on the sphere ----------------------------------------------
  AT_N_sym gamma_dd;
  AT_N_sym dr_gamma_dd;
  AT_N_sym dr2_gamma_dd;  // TODO 2nd dvtrs not yet implemented
  AT_N_sym dot_gamma_dd;
  AT_N_sym dr_dot_gamma_dd;

  AT_N_vec beta_d;
  AT_N_vec dr_beta_d;
  AT_N_vec dr2_beta_d;
  AT_N_vec dot_beta_d;

  AT_N_vec beta_u;
  AT_N_vec dr_beta_u;
  AT_N_vec dr2_beta_u;
  AT_N_vec dot_beta_u;

  AT_N_sca beta2;
  AT_N_sca dr_beta2;
  AT_N_sca dr2_beta2;
  AT_N_sca dot_beta2;

  AT_N_sca alpha;
  AT_N_sca dr_alpha;
  AT_N_sca dr2_alpha;
  AT_N_sca dot_alpha;

  // -- Spherical harmonics on the sphere (complex -> 2 components) -----------
  AA Y, Yth, Yph, X, W;

  // -- Spherical metric on M^2 -----------------------------------------------
  ATP_M_sym g_dd;
  ATP_M_sym g_uu;
  ATP_M_sym g_dr_dd;   // d/dr g_AB
  ATP_M_sym g_dr2_dd;  // d2/dr2 g_AB
  ATP_M_sym g_dot_dd;  // d/dt g_AB
  ATP_M_sym g_dr_uu;   // d/dr g^AB
  ATP_M_sym g_dr2_uu;  // d2/dr2 g^AB
  ATP_M_sym g_dot_uu;  // d/dt g^AB
  // TODO 2nd drvts
  ATP_M_VS2 Gamma_udd;      // Christoffels (time-independent)
  ATP_M_VS2 Gamma_dyn_udd;  // Christoffels (time-dep)
  Real norm_Delta_Gamma;

  // -- Multipoles (even) -----------------------------------------------------
  AA h00, h01, h11, h0, h1, G, K;
  AA h00_dr, h01_dr, h11_dr, h0_dr, h1_dr, G_dr, K_dr;  // d/dr drvts
  AA G_dr2, K_dr2;                                      // d2/dr2 drvts
  AA G_dr_dot, K_dr_dot;                                // d/dr d/dt drvts
  AA h00_dot, h01_dot, h11_dot, h0_dot, h1_dot, G_dot, K_dot;  // d/dt drvts

  // -- Multipoles (odd) ------------------------------------------------------
  AA H0, H01, H, H1;
  AA H0_dr, H01_dr, H_dr, H1_dr;      // d/dr drvts
  AA H0_dot, H01_dot, H_dot, H1_dot;  // d/dt drvts
  AA H0_dr2, H_dr2;                   // d2/dr2 drvts
  AA H1_dr_dot;                       // d/dr d/dt drvts

  // -- Gauge-invariant multipoles --------------------------------------------
  AA kappa_00, kappa_01, kappa_11, kappa_0,
    kappa_1;  // SB lets not use this!
  // SB The above variables are defined at a points, but the kappa's have a
  // multipolar index. These var type was incorrect. They require 1 dimension
  // for the multipolar index and 1 for the Re/Im part:
  AT_M_sym kappa_dd;
  AT_M_vec kappa_d;
  AA kappa, Tr_kappa_dd;
  Real norm_Tr_kappa_dd;

  // -- Master functions ------------------------------------------------------
  AA Psie, Psio;
  AA Psie_dyn, Psio_dyn;
  AA Psie_sch, Psio_sch;
  AA Qplus, Qstar;
  AA Psie_dr, Psio_dr;
  AA Qplus_dr, Qstar_dr;

  // -- Surface integral bookkeeping ------------------------------------------
  enum
  {
    Re,
    Im,
    RealImag
  };

  enum
  {
    I_ADM_M,
    I_ADM_Px,
    I_ADM_Py,
    I_ADM_Pz,
    I_ADM_Jx,
    I_ADM_Jy,
    I_ADM_Jz,
    NVADM,
  };
  Real integrals_adm[NVADM];

  enum
  {
    // ADM
    I_adm_M,
    I_adm_Px,
    I_adm_Py,
    I_adm_Pz,
    I_adm_Jx,
    I_adm_Jy,
    I_adm_Jz,
    // Schw. radius & co
    Irsch2,
    Idrsch_dri,
    Id2rsch_dri2,  // Areal radius and drvts wrt R
    Idot_rsch2,
    Idrsch_dri_dot,
    // Background 2-metric
    Ig00,
    Ig0R,
    IgRR,  // g_AB on M^2
    IgRt,
    Igtt,
    Igpp,
    IdR_g00,
    IdR_g0R,
    IdR_gRR,  // dg_AB/dr
    IdR2_g00,
    IdR2_g0R,
    IdR2_gRR,  // d2g_AB/dr2
    IdR_gtt,
    Idot_g00,
    Idot_g0R,
    Idot_gRR,  // dg_AB/dt
    // TODO 2nd drvts
    NVBackground,
  };
  Real integrals_background[NVBackground];

  Real* integrals_multipoles;
  enum
  {
    Ih00,
    Ih01,
    Ih11,  // even
    Ih0,
    Ih1,
    IG,
    IK,
    Ih00_dr,
    Ih01_dr,
    Ih11_dr,  // even d/dr drvts
    Ih0_dr,
    Ih1_dr,
    IG_dr,
    IK_dr,
    IG_dr2,
    IK_dr2,  // even d2/dr2 drvts
    IG_dr_dot,
    IK_dr_dot,  // even mixed drvts
    Ih00_dot,
    Ih01_dot,
    Ih11_dot,  // even d/dt drvts
    Ih0_dot,
    Ih1_dot,
    IG_dot,
    IK_dot,
    IH0,
    IH1,  // odd
    IH,
    IH0_dr,
    IH1_dr,  // odd d/dr drvts
    IH_dr,
    IH0_dot,
    IH1_dot,  // odd d/dt drvts
    IH_dot,
    IH0_dr2,
    IH_dr2,
    IH1_dr_dot,  // odd second drvts
    NVMultipoles,
  };

  // -- Areal radius options --------------------------------------------------
  enum
  {
    areal,
    areal_simple,
    average_schw,
    schw_gthth,
    schw_gphph,
    NOptRadius,
  };
  static char const* const ArealRadiusMethod[NOptRadius];

  // -- I/O -------------------------------------------------------------------
#if (WAVE_EXTRACT_RWZ_EXTRAOUTPUT)
  // includes indexes of various extra output files
  enum
  {
    Iof_diagnostic,  // This should be first!
    Iof_adm,         // This should be second!
    Iof_Psie,
    Iof_Psio,
    Iof_Psie_dyn,
    Iof_Psio_dyn,
    Iof_Qplus,
    Iof_Qstar,
    //
    Iof_H1_dot,
    Iof_H0_dr,
    Iof_H0,
    Iof_H1,
    Iof_H,
    Iof_H_dr,
    Iof_Psie_dr,
    Iof_Psio_dr,
    Iof_Qplus_dr,
    Iof_Qstar_dr,
    //
    Iof_hlm,  // This should be last!
    Iof_Num,
  };
#else
  enum
  {
    Iof_diagnostic,  // This should be first!
    Iof_adm,         // This should be second!
    Iof_Psie,
    Iof_Psio,
    Iof_Psie_dyn,
    Iof_Psio_dyn,
    Iof_Qplus,
    Iof_Qstar,
    Iof_hlm,  // This should be last!
    Iof_Num,
  };
#endif
  std::string ofbname[Iof_Num];
  std::string ofname;
  FILE* pofile;
  std::ofstream outfile;

  // -- Back-pointers ---------------------------------------------------------
  Mesh const* pmesh;

  // -- Private methods -------------------------------------------------------
  int MPoints(const int l);
  int MIndex(const int l, const int m);
  Real RWZnorm(const int l);
  void ComputeSphericalHarmonics();

  Real LeviCivitaSymbol(const int a, const int b, const int c);
  void InterpMetricToSphere();
  void MasterFuns();
  void MultipolesGaugeInvariant();
  void SphOrthogonality();

  std::string OutputFileName(std::string base);
};

#endif
