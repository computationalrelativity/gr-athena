#ifndef EJECTA_HPP
#define EJECTA_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ejecta.hpp
//  \brief definitions for the Ejecta class

// TODO
// * check public/private
// * check for missing definitions

#include <string>
#include <hdf5.h>

#include "../athena_aliases.hpp"
#include "../utils/lagrange_interp.hpp"
#include "z4c.hpp"

// Forward declaration
class Mesh;
class MeshBlock;
class ParameterInput;

#define SQ(X) ((X)*(X))

//! \class Ejecta
//! \brief Ejecta extraction
class Ejecta {
  
public:

  enum {
    I_detg,
    I_Mdot,
    I_bernoulli,
    I_enthalpy,
    I_entropy,
    I_lorentz,
    I_u_t,
    I_fD_r,
    I_v_mag,
    I_poynting,
    NOTHER
  };
  //! Creates the AHF object
  Ejecta(Mesh * pmesh, ParameterInput * pin, int nrad);
  //! Destructor (will close output file)
  ~Ejecta();
  //!
  //!
  //void Write(int iter, Real time);
  //!

  Real radius;
  bool verbose;
  //! Grid points
  int ntheta, nphi;

  //! start and stop times for each surface
  Real start_time;
  Real stop_time;
  //! compute every n iterations
  int compute_every_iter;

  void Calculate(int iter, Real time);
  void Write(int iter, Real time);

private:
  int nr;
  bool bitant;
  //! Number of horizons
  int nstart, nrad;
  int fastflow_iter=0;
  AthenaArray<Real> prim[NHYDRO], cons[NHYDRO], Y[NSCALARS], T, Bcc[3];
  AthenaArray<Real> z4c[Z4c::N_Z4c], adm[Z4c::N_ADM];
  AthenaArray<Real> other[NOTHER];
  AthenaArray<Real> theta, phi;
  Real mass_contained;
  Real Mdot_total;


  // Unboundedness criteria
  enum{I_unbound_bernoulli,I_unbound_bernoulli_outflow,I_unbound_geodesic,I_unbound_geodesic_outflow,n_unbound};
  //integrated quantities
  enum{I_int_mass,I_int_entr,I_int_rho,I_int_temp,I_int_ye,I_int_vel,I_int_ber,I_int_velinf,n_int};
  enum{I_hist_entr,I_hist_logrho,I_hist_temp,I_hist_ye,I_hist_vel,I_hist_ber,I_hist_theta,I_hist_velinf,n_hist};
  AthenaArray<Real> hist[n_hist], hist_grid[n_hist];
  Real delta_hist[n_hist];
  int n_bins[n_hist];

  //! Flag points 
  AthenaArray<int> havepoint;
 AthenaArray<Real> integrals_unbound, az_integrals_unbound; 

  void Interp(MeshBlock * pmb);
  void Mass(MeshBlock * pmb);

  void SphericalIntegrals();

  int tpindex(const int i, const int j);
  Real th_grid(const int i);
  Real ph_grid(const int j);
  Real dth_grid();
  Real dph_grid();

  Mesh const * pmesh;

  int root;
  int ioproc;
  std::string ofname_summary, ofname_bernoulli, ofname_bernoulli_outflow, ofname_geodesic, ofname_geodesic_outflow;
  std::string ofname_unbound[n_unbound], ofname_az_unbound[n_unbound], ofname_hist_unbound[n_unbound][n_hist];
  FILE * pofile_summary, *pofile_bernoulli, *pofile_bernoulli_outflow, *pofile_geodesic, *pofile_geodesic_outflow;
  FILE * pofile_unbound[n_unbound], *pofile_az_unbound[n_unbound];
  FILE *pofile_hist_unbound[n_unbound][n_hist];

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
  Real MassLossRate(Real const fx, Real const fy, Real const fz,
                          Real const sinth, Real const costh, Real const sinph, Real const cosph);
  Real MassLossRate2(Real const D, Real const ux, Real const uy, Real const uz, Real const W,
                           Real const alpha, Real const betax, Real const betay, Real const betaz,
                           Real const sinth, Real const costh, Real const sinph, Real const cosph);
};

#endif
