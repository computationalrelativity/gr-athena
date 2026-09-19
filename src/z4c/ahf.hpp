#ifndef AHF_HPP
#define AHF_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file ahf.hpp
//  \brief definitions for the AHF class

#include <string>

#include "../athena_aliases.hpp"
#include "../utils/grid_theta_phi.hpp"
#include "../utils/lagrange_interp.hpp"
#include "../utils/spherical_harmonics.hpp"

// Forward declaration
class Mesh;
class MeshBlock;
class ParameterInput;

using namespace gra::aliases;

//! \class AHF
//! \brief Apparent Horizon Finder
class AHF
{
  public:
  // -- Types -----------------------------------------------------------------
  enum class ExpansionFix
  {
    do_nothing,
    cure_divu
  };

  enum class StepRule
  {
    fixed,
    monotone,
    bb1,
    bb2
  };

  // Fast-flow driving function: the array `rho` fed into the spectral
  // update is rho = weight(theta,phi) * H, i.e. the projected quantity is
  // always (weight * Theta) per Gundlach 1997 (gr-qc/9809004 eq. 8-9).
  //   H  : weight = 1            (pure mean-curvature/expansion flow)
  //   Hu : weight = u = |grad F| (default; regularizes near coordinate
  //                               poles / grazing incidence)
  //   F3 : weight = 2 r^2 |grad F| /
  //          [ (g^ij - s^i s^j)(gbar_ij - grad_i r grad_j r) ]
  //        with gbar the flat background metric of (r,theta,phi) -- the
  //        area/normalization-aware weight from Gundlach's original paper.
  enum class FlowFunction
  {
    H,
    Hu,
    F3
  };

  // FastFlowLoop termination status
  enum class ExitCode
  {
    success        = 0,
    not_finite     = 1,
    hmean_diverged = 2,
    meanradius_neg = 3,
    mass_collapse  = 4,
    max_iters      = 5,
    stagnated      = 6
  };

  // -- Construction / destruction --------------------------------------------
  AHF(Mesh* pmesh, ParameterInput* pin, int idx_ahf);
  ~AHF();

  // -- Main API --------------------------------------------------------------
  void Find(int iter, Real time);
  void Write(int iter, Real time);

  // -- Accessors (read externally by pgens / eos) ----------------------------
  bool IsFound() const
  {
    return ah_found;
  }
  Real TimeFound() const
  {
    return time_first_found;
  }
  Real GetHorizonMeanRadius() const
  {
    return ah_prop[hmeanradius];
  }
  Real GetGWFlux() const
  {
    return ah_prop[hgwflux];
  }
  Real GetHorizonMinRadius() const
  {
    return rr_min;
  }
  Real GetCenter(int i) const
  {
    return center[i];
  }

  private:
  // -- Configuration (set once from ParameterInput) --------------------------
  struct
  {
    Real initial_radius;
    Real expand_guess;
    bool propagate_iter_coefficients;
    Real hmean_tol;
    Real mass_tol;
    Real spec_tol;
    Real hrms_tol;
    Real hrms_rel_tol;
    bool stagnation_detect;
    int stagnation_window;
    Real stagnation_improvement_frac;
    int stagnation_warmup;
    int mode_ramp_lmin;
    int mode_ramp_iters_per_step;
    int mode_ramp_modes_per_step;
    bool auto_retry;
    int max_retries;
    Real retry_shrink;
    Real retry_grow;
    int flow_iterations;
    Real flow_alpha_beta_const;
    FlowFunction flow_function = FlowFunction::Hu;
    StepRule step_rule = StepRule::monotone;
    Real alpha_min;
    Real alpha_max;
    Real alpha_grow;
    Real alpha_shrink;
    bool verbose;
    int lmax;
    int use_puncture;
    bool use_puncture_massweighted_center;
    int use_extrema;
    Real merger_distance;
    Real start_time;
    Real stop_time;
    bool wait_until_punc_are_close;
    bool bitant;
    ExpansionFix expansion_fix = ExpansionFix::do_nothing;
    int mpi_root;
    std::string ofname_summary;
    std::string ofname_shape;
    std::string ofname_shear;
    std::string ofname_verbose;
  } opt;

  // -- Compile-time constants ------------------------------------------------
  static const int metric_interp_order = 2 * NGHOST - 1;
  static constexpr Real min_surface_radius =
    1e-10;  // floor on r(theta,phi) to avoid coordinate singularity
  static constexpr Real min_mass =
    1e-10;  // floor on irreducible mass to abort failed flow

  // -- Grid infrastructure ---------------------------------------------------
  gra::grids::theta_phi::Grid<LagrangeInterpND<metric_interp_order, 3>> grid_;

  // -- Spectral decomposition ------------------------------------------------
  gra::sph_harm::RealHarmonicTable ylm_;
  AA a0, ac, as;
  Real last_a0;
  AA last_a0_full;  // last-found l=0 modes    (size lmax+1)
  AA last_ac;       // last-found cosine modes (size lmpoints)
  AA last_as;       // last-found sine modes   (size lmpoints)

  // -- Fields on the sphere --------------------------------------------------
  AT_N_sym g;
  AT_N_sym K;
  AT_N_VS2 dg;
  AA rr, rr_dth, rr_dph;
  AA rho;

  // -- Shear tensor / spin-2 shear scalar -------------------------------------
  AT_N_sym sigma_dd;      // sigma_ij  (transverse-traceless part of B_ij)
  AT_N_sym sigma_uu;      // sigma^ij = g^{ik} g^{jl} sigma_kl
  AA shear2;              // sigma_ij sigma^ij           (real, on the grid)
  AA shear_re, shear_im;  // Re/Im[sigma_ab m^a m^b], m=(v-iw)/sqrt2 (on grid)

  // Precomputed spin-weight -2 harmonics, l = 2..lmax, m = -l..l, packed via
  // gra::sph_harm::lmindex_complex / lmpoints_complex (same packing as
  // ComplexHarmonicTable, so indices are directly comparable elsewhere).
  AA swsh2_re, swsh2_im;  // (ntheta, nphi, lmpoints_complex(lmax))
  AA c2_re, c2_im;        // accumulated/reduced coefficients,
                          // size lmpoints_complex(lmax)

  // -- Surface integral bookkeeping ------------------------------------------
  enum
  {
    iarea,
    icoarea,
    ihrms,
    ihmean,
    iSx,
    iSy,
    iSz,
    ishear2,  // area-weighted sum of sigma_ij sigma^ij
    invar
  };
  Real integrals[invar];
  enum
  {
    harea,
    hcoarea,
    hhrms,
    hhmean,
    hSx,
    hSy,
    hSz,
    hS,
    hmass,
    hmass_irr,
    hchi,
    hmeanradius,
    hminradius,
    hshearrms,   // sqrt(<sigma_ij sigma^ij>_area)
    hgwflux,     // instantaneous GW flux: (1/16pi) * oint sigma_ij sigma^ij dA
    hnvar
  };
  Real ah_prop[hnvar];

  // -- Internal state --------------------------------------------------------
  bool ah_found;
  Real time_first_found;
  Real rr_min;
  Real center[3];
  int idx_ahf;
  int fastflow_iter    = 0;
  Real spec_resid_last = -1.0;
  ExitCode last_exit   = ExitCode::max_iters;

  // -- I/O -------------------------------------------------------------------
  FILE* pofile_summary;
  FILE* pofile_shape;
  FILE* pofile_shear;
  FILE* pofile_verbose;

  // -- Back-pointers ---------------------------------------------------------
  Mesh const* pmesh;
  ParameterInput* pin;

  // -- Private methods -------------------------------------------------------
  void ReadOptions(ParameterInput* pin);
  void PrepareArrays();
  void SetupIO();
  void MetricInterp();
  void SurfaceIntegrals();
  void FastFlowLoop();
  void UpdateFlowSpectralComponents();
  void InitialGuess(bool cold = false);
  void RecomputeABfac(Real alpha, Real beta, int lmax, Real* ABfac) const;

  bool LevelSetGradient(int i,
                        int j,
                        ATP_N_vec& dFdi,
                        ATP_N_sym& dFdidj,
                        Real& xp,
                        Real& yp,
                        Real& zp);
  void ExpansionAndNormal(int i,
                          int j,
                          const ATP_N_vec& dFdi,
                          const ATP_N_sym& dFdidj,
                          ATP_N_vec& R,
                          Real& H,
                          Real& u,
                          ATP_N_sym& nnF_out,
                          ATP_N_sym& ginv_out,
                          ATP_N_vec& dFdi_u_out);
  void ShearTensor(int i,
                   int j,
                   const ATP_N_vec& dFdi,
                   const ATP_N_vec& dFdi_u,
                   const ATP_N_sym& nnF,
                   const ATP_N_sym& ginv,
                   Real u,
                   Real& shear2_out,
                   Real& sre,
                   Real& sim);
  void PrepareSWSH2Table();
  Real FlowFunctionRho(int i,
                       int j,
                       Real H,
                       Real u,
                       const ATP_N_vec& dFdi_u,
                       const ATP_N_sym& ginv);
  Real SurfaceElement(int i, int j);
  void SpinIntegrand(Real xp,
                     Real yp,
                     Real zp,
                     const ATP_N_vec& R,
                     int i,
                     int j,
                     Real& Sx,
                     Real& Sy,
                     Real& Sz);

  // Puncture tracker interface
  Real PuncMaxDistance();
  Real PuncMaxDistance(const int pix);
  Real PuncSumMasses();
  void PuncWeightedMassCentralPoint(Real* xc, Real* yc, Real* zc);
  bool PuncAreClose();
};

#endif
