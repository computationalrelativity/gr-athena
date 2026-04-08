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
  int lmpoints;
  AA a0, ac, as;
  Real last_a0;

  // Derivative multi-index for spherical harmonic tables
  enum
  {
    D00,
    D10,
    D01,
    D20,
    D11,
    D02,
    NDERIV
  };

  //! Legendre polynomials: P_all(3, lmax+1, lmax+1), sliced by d/dth order
  AA P_all;
  AA P, dPdth, dPdth2;

  //! Spherical harmonics: compact storage + named shallow-slice views
  AA Y0_all;  // (3, ntheta, nphi, lmax+1)        - m=0: value, d/dth, d2/dth2
  AA Yc_all;  // (NDERIV, ntheta, nphi, lmpoints)  - m>0 cosine
  AA Ys_all;  // (NDERIV, ntheta, nphi, lmpoints)  - m>0 sine
  AA Y0, Yc, Ys;
  AA dY0dth, dYcdth, dYsdth, dYcdph, dYsdph;
  AA dY0dth2, dYcdth2, dYcdthdph, dYsdth2, dYsdthdph, dYcdph2, dYsdph2;

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
  void RadiiFromSphericalHarmonics();
  void InitialGuess();
  void ComputeSphericalHarmonics();
  int lmindex(const int l, const int m) const;

  // Puncture tracker interface
  Real PuncMaxDistance();
  Real PuncMaxDistance(const int pix);
  Real PuncSumMasses();
  void PuncWeightedMassCentralPoint(Real* xc, Real* yc, Real* zc);
  bool PuncAreClose();
};

#endif
