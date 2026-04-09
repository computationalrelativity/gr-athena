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
    Real hmean_tol;
    Real mass_tol;
    int flow_iterations;
    Real flow_alpha_beta_const;
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
    std::string ofname_verbose;
  } opt;

  // -- Compile-time constants ------------------------------------------------
  static const int metric_interp_order = 2 * NGHOST - 1;
  static constexpr Real min_surface_radius =
    1e-10;  // floor on r(theta,phi) to avoid coordinate singularity
  static constexpr Real min_mass =
    1e-10;  // floor on irreducible mass to abort failed flow

  // -- Grid infrastructure ---------------------------------------------------
  GridThetaPhi<LagrangeInterpND<metric_interp_order, 3>> grid_;

  // -- Spectral decomposition ------------------------------------------------
  gra::sph_harm::RealHarmonicTable ylm_;
  AA a0, ac, as;
  Real last_a0;

  // -- Fields on the sphere --------------------------------------------------
  AT_N_sym g;
  AT_N_sym K;
  AT_N_VS2 dg;
  AA rr, rr_dth, rr_dph;
  AA rho;

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
    hmeanradius,
    hminradius,
    hnvar
  };
  Real ah_prop[hnvar];

  // -- Internal state --------------------------------------------------------
  bool ah_found;
  Real time_first_found;
  Real rr_min;
  Real center[3];
  int idx_ahf;
  int fastflow_iter = 0;

  // -- I/O -------------------------------------------------------------------
  FILE* pofile_summary;
  FILE* pofile_shape;
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
  void InitialGuess();

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
                          Real& u);
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
