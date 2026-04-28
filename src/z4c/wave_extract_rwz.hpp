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
#include <vector>

#include "../athena_aliases.hpp"
#include "../utils/grid_theta_phi.hpp"
#include "../utils/grid_theta_phi_fields.hpp"
#include "../utils/lagrange_interp.hpp"
#include "../utils/spherical_harmonics.hpp"

// Forward declaration
class Mesh;
class MeshBlock;
class ParameterInput;

using namespace gra::aliases;

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

  // Split-phase background: local accumulation then finalization after MPI
  void BackgroundAccumulate();
  void BackgroundFinalize();

  // Split-phase multipoles: local accumulation then finalization after MPI
  void MultipoleAccumulate();
  void MultipoleFinalize();

  void Write(int iter, Real time);

  // -- Pipelined extraction across all radii ---------------------------------
  //! Run the full extraction pipeline for every radius in \p rwz_vec,
  //! batching the two MPI_Allreduce calls (background + multipoles) so that
  //! the total number of reductions is 2 regardless of how many radii exist.
  static void ExtractAll(std::vector<WaveExtractRWZ*>& rwz_vec,
                         int iter,
                         Real time);

  // -- Accessors for pipelined MPI reduce ------------------------------------
  Real* background_integrals()
  {
    return integrals_background;
  }
  int n_background_integrals() const
  {
    return NVBackground;
  }

  Real* multipole_integrals()
  {
    return integrals_multipoles;
  }
  int n_multipole_integrals() const
  {
    return NVMultipoles * 2 * lmpoints;
  }

  int n_lmpoints() const
  {
    return lmpoints;
  }

  // -- Grid infrastructure (public: needed by AMR teardown hook) -------------
  static const int metric_interp_order = 2 * NGHOST - 1;
  static constexpr Real eps_det        = 1e-12;
  gra::grids::theta_phi::Grid<LagrangeInterpND<metric_interp_order, 3, true>>
    grid_;

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
    bool extra_output;
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

  // -- 3+1 metric on the sphere (bundled with radial/time derivatives) -------
  //  Access: gamma(a,b,i,j) for value, gamma(D10,a,b,i,j) for d_r, etc.
  //  See ix_DRT for derivative index constants.
  gra::grids::theta_phi::DTensorField<Real, TensorSymm::SYM2, 3, 2> gamma;
  gra::grids::theta_phi::DTensorField<Real, TensorSymm::NONE, 3, 1> beta_d;
  gra::grids::theta_phi::DTensorField<Real, TensorSymm::NONE, 3, 1> beta_u;
  gra::grids::theta_phi::DTensorField<Real, TensorSymm::NONE, 3, 0> alpha;
  gra::grids::theta_phi::DTensorField<Real, TensorSymm::NONE, 3, 0> beta2;

  // -- Spherical harmonics on the sphere (complex, l=2..lmax) ----------------
  gra::sph_harm::ComplexHarmonicTable ylm_;

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
  // Each DMultipoleField bundles D00 (value), D10 (d_r), D20 (d^2_r),
  // D01 (d_t), D11 (d_r d_t) slots as AthenaArray<Real>(lmpoints, 2).
  gra::grids::theta_phi::DMultipoleField<Real> h00, h01, h11, h0, h1, G, K;

  // -- Multipoles (odd) ------------------------------------------------------
  gra::grids::theta_phi::DMultipoleField<Real> H0, H1, H;

  // -- Gauge-invariant multipoles --------------------------------------------
  // kappa_dd(A,B,lm,c): even-parity gauge-invariant tensor on M^2
  // kappa_d(A,lm,c):    odd-parity gauge-invariant vector on M^2
  // kappa(lm,c):        even-parity gauge-invariant scalar
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
  std::string ofbname[Iof_Num];
  FILE* ofile[Iof_Num]{};

  // -- Back-pointers ---------------------------------------------------------
  Mesh const* pmesh;

  // -- Private methods -------------------------------------------------------
  void ReadOptions(ParameterInput* pin, int n);
  void PrepareArrays();

  //! RWZ normalization factor: sqrt((l+2)!/(l-2)!)
  static inline Real RWZnorm(const int l)
  {
    return std::sqrt(gra::sph_harm::Factorial(l + 2) /
                     gra::sph_harm::Factorial(l - 2));
  }

  void InterpMetricToSphere();
  void MasterFuns();
  void MultipolesGaugeInvariant();

  std::string OutputFileName(std::string base);
  FILE* OpenOutputFile(int iof);
};

#endif
