#ifndef Z4C_EJECTA_HPP
#define Z4C_EJECTA_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ejecta.hpp
//  \brief definitions for the Ejecta class

// TODO
// * refactor / tidy-up:
//   - only dump things that are specified to be dumped
// * combine MPI reduction

// C++ standard headers
#include <string>

// Athena++ headers
#include "../athena_aliases.hpp"
#include "z4c.hpp"

// External libraries
#include <hdf5.h>

// Forward declaration
class Mesh;
class MeshBlock;
class ParameterInput;


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
  Ejecta(Mesh * pmesh, ParameterInput * pin, int nrad);
  ~Ejecta();

  Real radius;
  bool verbose;
  //! Grid points
  int ntheta, nphi;

  //! start and stop times for each surface
  Real start_time;
  Real stop_time;

  void Calculate(const Real time);
  void Write(const Real time);

private:
  int file_number;

  int nr;
  bool bitant;
  //! Number of horizons
  int nstart, nrad;

  AA prim[NHYDRO], cons[NHYDRO], Y[NSCALARS], T, Bcc[3];
  AA z4c[Z4c::N_Z4c], adm[Z4c::N_ADM];
  AA other[NOTHER];
  AA theta, phi;
  Real mass_contained;
  Real Mdot_total;

  // Unboundedness criteria
  enum {
    I_unbound_bernoulli,
    I_unbound_bernoulli_outflow,
    I_unbound_geodesic,
    I_unbound_geodesic_outflow,
    n_unbound
  };
  // integrated quantities
  enum {
    I_int_mass,
    I_int_entr,
    I_int_rho,
    I_int_temp,
    I_int_ye,
    I_int_vel,
    I_int_ber,
    I_int_velinf,
    n_int
  };
  enum {
    I_hist_entr,
    I_hist_logrho,
    I_hist_temp,
    I_hist_ye,
    I_hist_vel,
    I_hist_ber,
    I_hist_theta,
    I_hist_velinf,
    n_hist
  };
  AA hist[n_hist], hist_grid[n_hist];
  Real delta_hist[n_hist];
  int n_bins[n_hist];

  //! Flag points
  AthenaArray<int> havepoint;
  AA integrals_unbound, az_integrals_unbound;

  void Interp(MeshBlock *pmb);
  void Mass(MeshBlock *pmb);

  void SphericalIntegrals();

  int tpindex(const int i, const int j);
  Real th_grid(const int i);
  Real ph_grid(const int j);
  Real dth_grid();
  Real dph_grid();

  Mesh * pmesh;
  ParameterInput * pin;

  std::string ofname_summary, ofname_bernoulli, ofname_bernoulli_outflow,
      ofname_geodesic, ofname_geodesic_outflow;
  std::string ofname_unbound[n_unbound], ofname_az_unbound[n_unbound],
      ofname_hist_unbound[n_unbound][n_hist];
  FILE *pofile_summary, *pofile_bernoulli, *pofile_bernoulli_outflow,
      *pofile_geodesic, *pofile_geodesic_outflow;
  FILE *pofile_unbound[n_unbound], *pofile_az_unbound[n_unbound];
  FILE *pofile_hist_unbound[n_unbound][n_hist];

  Real MassLossRate(Real const fx, Real const fy, Real const fz,
                    Real const sinth, Real const costh, Real const sinph,
                    Real const cosph);
  Real MassLossRate2(Real const D, Real const ux, Real const uy, Real const uz,
                     Real const W, Real const alpha, Real const betax,
                     Real const betay, Real const betaz, Real const sinth,
                     Real const costh, Real const sinph, Real const cosph);


  // hdf5 mumbo-jumbo ---------------------------------------------------------
  public:
    // BD: TODO - might be useful to shift to outputs/... ?

    // infer next file-name based on internal state
    void hdf5_get_next_filename(std::string & filename);

    // Create / open a file for R/W; return handle
    hid_t hdf5_touch_file(const std::string & filename);

    // Implementation details for groups: -------------------------------------
    // pass in a full path; (nested) groups created automatically
    void _hdf5_prepare_path(hid_t & id_file, const std::string & full_path);
    // ------------------------------------------------------------------------

    // wrapped to write_arr_nd
    void hdf5_write_scalar(hid_t & id_file,
                           const std::string & full_path,
                           Real scalar);

    // write n-dimensional athena array to some path (groups gen. automatic)
    void hdf5_write_arr_nd(hid_t & id_file,
                           const std::string & full_path,
                           const AA & arr);

    // clean up of open handle
    void hdf5_close_file(hid_t & id_file);

};

#endif  // Z4C_EJECTA_HPP
