//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file gr_kadath_bns.cpp
//  \brief Initial conditions for binary neutron stars.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <streambuf>

// Athena++ headers
#include "../athena_aliases.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../field/seed_magnetic_field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "../parameter_input.hpp"
#include "../trackers/extrema_tracker.hpp"
#include "../utils/linear_algebra.hpp"
#include "../utils/utils.hpp"
#include "../z4c/ahf.hpp"
#include "../z4c/z4c.hpp"

#if M1_ENABLED
#include "../m1/m1.hpp"
#include "../m1/m1_set_equilibrium.hpp"
#endif

// Kadath FUKa headers
#include "kadath_bin_ns.hpp"
#include "EOS/EOS.hh"
#include "coord_fields.hpp"
#include "Configurator/config_enums.hpp"
#include "Configurator/config_binary.hpp"
#include "Configurator/configurator_boost.hpp"
#include "exporter_utilities.hpp"
#include "bco_utilities.hpp"

// Configuration checking
#if not FLUID_ENABLED
#error "This problem generator requires fluid (-f)"
#endif

//----------------------------------------------------------------------------------------
using namespace gra::aliases;
using namespace Primitive;
using export_utils::PSI;
using export_utils::ALP;
using export_utils::BETX;
using export_utils::BETY;
using export_utils::BETZ;
using export_utils::AXX;
using export_utils::AXY;
using export_utils::AXZ;
using export_utils::AYY;
using export_utils::AYZ;
using export_utils::AZZ;
using export_utils::H;
using export_utils::UX;
using export_utils::UY;
using export_utils::UZ;
using export_utils::NUM_QUANTS;
//----------------------------------------------------------------------------------------

namespace
{
// Global variables
ColdEOS<COLDEOS_POLICY>* ceos = NULL;

Real sep;
Real pgasmax_1;
Real pgasmax_2;
Real centre_m, centre_p;

// Kadath spectral fields (allocated in InitUserMeshData, freed in DeleteTemporaryUserMeshData)
std::vector<std::reference_wrapper<const Kadath::Scalar>>* g_quants = NULL;
Kadath::Space_bin_ns* g_space   = NULL;
Kadath::Scalar* g_conf           = NULL;
Kadath::Scalar* g_lapse          = NULL;
Kadath::Vector* g_shift          = NULL;
Kadath::Scalar* g_logh           = NULL;
Kadath::Scalar* g_phi            = NULL;
Kadath::Base_tensor* g_basis     = NULL;
Kadath::System_of_eqs* g_syst    = NULL;
Kadath::Metric_flat* g_fmet      = NULL;
Kadath::Tensor* g_A_tens         = NULL;
Kadath::Vector* g_vel_kad        = NULL;

// Kadath EOS type flags
bool g_use_cold_table  = false;
bool g_use_cold_pwpoly = false;
double g_com_offset     = 0.0;
double g_fuka_rho_rescale = 1.0;

// --------------------------------------------------------------------------
// Unit conversion constants
Real const c_light = 2.99792458e8;   // Speed of light [m/s]
Real const G_grav  = 6.67428e-11;    // Gravitational constant [m^3/kg/s^2]
Real const M_sun   = 1.98892e30;     // Solar mass [kg]

Real const athenaM  = M_sun;
Real const athenaL  = athenaM * G_grav / (c_light * c_light);
Real const athenaT  = athenaL / c_light;
// --------------------------------------------------------------------------

void SeedMagneticFields(MeshBlock* pmb, ParameterInput* pin)
{
  Real pcut_1 = pin->GetReal("problem", "pcut_1") * pgasmax_1;
  Real pcut_2 = pin->GetReal("problem", "pcut_2") * pgasmax_2;

  Real ns_1 = pin->GetReal("problem", "ns_1");
  Real ns_2 = pin->GetReal("problem", "ns_2");

  Real A_amp_1 = pin->GetReal("problem", "b_amp_1") * 0.5 /
                 std::pow(pgasmax_1 - pcut_1, ns_1);
  Real A_amp_2 = pin->GetReal("problem", "b_amp_2") * 0.5 /
                 std::pow(pgasmax_2 - pcut_2, ns_2);

  const Real cp = centre_p;
  const Real cm = centre_m;

  SeedFaceBFromEdgePotential(
    pmb,
    [=](Real x,
        Real y,
        Real z,
        Real p,
        Real /*rho*/,
        Real& Ax,
        Real& Ay,
        Real& Az)
    {
      if (x > 0)
      {
        Real amp = A_amp_2 * std::max(std::pow(p - pcut_2, ns_2), 0.0);
        Ax       = -y * amp;
        Ay       = (x - cp) * amp;
      }
      else
      {
        Real amp = A_amp_1 * std::max(std::pow(p - pcut_1, ns_1), 0.0);
        Ax       = -y * amp;
        Ay       = (x - cm) * amp;
      }
      Az = 0.0;
    });
}

}  // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput* pin)
{
  if (adaptive == true)
  {
    EnrollUserRefinementCondition(Mesh::StandardRefinementCondition);
  }

  EnrollUserStandardHydro(pin);
  EnrollUserStandardField(pin);
  EnrollUserStandardZ4c(pin);
  EnrollUserStandardM1(pin);

  if (resume_flag)
    return;

  ceos = new ColdEOS<COLDEOS_POLICY>();
  InitColdEOS(ceos, pin);

  // FUKA (Margherita) defines the rest-mass density as rho = nb*m_amu using the
  // atomic mass unit (Margherita_constants::mnuc_MeV), whereas the CompOSE tables
  // define rho = nb*mn using the neutron mass (ColdEOS::mb). Recovering nb from
  // FUKA's density via rho/mn undershoots nb by m_amu/mn (~0.86%), which the
  // con2prim inversion absorbs as a spurious ~15 MeV initial temperature.
  // When <problem>/rescale_fuka_density is enabled, multiply FUKA's density by
  // mn/m_amu so nb is recovered consistently with the evolution table.
  const bool rescale_fuka_density =
      pin->GetOrAddBoolean("problem", "rescale_fuka_density", false);
  if (rescale_fuka_density) {
    g_fuka_rho_rescale = ceos->mb / Margherita_constants::mnuc_MeV;
    if (Globals::my_rank == 0) {
      std::printf("FUKA density rescaling enabled: factor = %.6f\n",
                  g_fuka_rho_rescale);
    }
  }

  std::string fname = pin->GetOrAddString("problem", "initial_data_file", "bns.info");

  if (!file_exists(fname.c_str()))
  {
    std::stringstream msg;
    msg << "### FATAL ERROR problem/initial_data_file: " << fname
        << " could not be accessed.";
    ATHENA_ERROR(msg);
  }

  if (Globals::my_rank == 0)
  {
    std::cout << "Reading Kadath BNS config from " << fname << " ..." << std::endl;
  }

  kadath_config_boost<BIN_INFO> bconfig(fname);

  const double h_cut      = bconfig.eos<double>(HCUT, BCO1);
  const std::string eos_file = bconfig.eos<std::string>(EOSFILE, BCO1);
  const std::string eos_type = bconfig.eos<std::string>(EOSTYPE, BCO1);

  double& units = bconfig(QPIG);
  double& omega = bconfig(GOMEGA);
  double& ome1  = bconfig(OMEGA, BCO1);
  double& ome2  = bconfig(OMEGA, BCO2);
  double& axis  = bconfig(COM);
  g_com_offset   = bconfig(COM);
  sep            = bconfig(DIST);

  std::string kadath_filename = bconfig.space_filename();

  FILE* fin = fopen(kadath_filename.c_str(), "r");
  if (!fin)
  {
    std::stringstream msg;
    msg << "### FATAL ERROR: Kadath space file " << kadath_filename
        << " could not be opened.";
    ATHENA_ERROR(msg);
  }

  g_space  = new Kadath::Space_bin_ns(fin);
  g_conf   = new Kadath::Scalar(*g_space, fin);
  g_lapse  = new Kadath::Scalar(*g_space, fin);
  g_shift  = new Kadath::Vector(*g_space, fin);
  g_logh   = new Kadath::Scalar(*g_space, fin);
  g_phi    = new Kadath::Scalar(*g_space, fin);
  fclose(fin);

  g_quants = new std::vector<std::reference_wrapper<const Kadath::Scalar>>();
  g_quants->reserve(NUM_QUANTS);
  for (int i = 0; i < NUM_QUANTS; ++i)
    g_quants->push_back(std::cref(*g_conf));

  (*g_quants)[PSI]  = std::cref(*g_conf);
  (*g_quants)[ALP]  = std::cref(*g_lapse);
  (*g_quants)[BETX] = std::cref((*g_shift)(1));
  (*g_quants)[BETY] = std::cref((*g_shift)(2));
  (*g_quants)[BETZ] = std::cref((*g_shift)(3));

  g_basis = new Kadath::Base_tensor(g_shift->get_basis());
  int ndom = g_space->get_nbr_domains();

  double xc1 = bco_utils::get_center(*g_space, g_space->NS1);
  double xc2 = bco_utils::get_center(*g_space, g_space->NS2);
  double xo  = bco_utils::get_center(*g_space, ndom - 1);

  g_fmet = new Kadath::Metric_flat(*g_space, *g_basis);

  CoordFields<Kadath::Space_bin_ns> cfields(*g_space);
  vec_ary_t coord_vectors{ default_binary_vector_ary(*g_space) };
  update_fields(cfields, coord_vectors, {}, xo, xc1, xc2);

  g_syst = new Kadath::System_of_eqs(*g_space, 0, ndom - 1);
  g_fmet->set_system(*g_syst, "f");

  Kadath::Param p;
  if (eos_type == "Cold_Table")
  {
    g_use_cold_table = true;
    using eos_t = Kadath::Margherita::Cold_Table;
    const int interp_pts = (bconfig.eos<int>(INTERP_PTS, BCO1) == 0)
                             ? 2000 : bconfig.eos<int>(INTERP_PTS, BCO1);
    ::EOS<eos_t, ::PRESSURE>::init(eos_file, h_cut, interp_pts);
    g_syst->add_ope("eps", &::EOS<eos_t, ::EPSILON>::action, &p);
    g_syst->add_ope("press", &::EOS<eos_t, ::PRESSURE>::action, &p);
    g_syst->add_ope("rho", &::EOS<eos_t, ::DENSITY>::action, &p);
  }
  else if (eos_type == "Cold_PWPoly")
  {
    g_use_cold_pwpoly = true;
    using eos_t = Kadath::Margherita::Cold_PWPoly;
    ::EOS<eos_t, ::PRESSURE>::init(eos_file, h_cut);
    g_syst->add_ope("eps", &::EOS<eos_t, ::EPSILON>::action, &p);
    g_syst->add_ope("press", &::EOS<eos_t, ::PRESSURE>::action, &p);
    g_syst->add_ope("rho", &::EOS<eos_t, ::DENSITY>::action, &p);
  }
  else
  {
    std::stringstream msg;
    msg << "### FATAL ERROR: Unknown Kadath EOS type '" << eos_type << "'";
    ATHENA_ERROR(msg);
  }

  g_syst->add_cst("4piG", units);
  g_syst->add_cst("PI", M_PI);
  g_syst->add_cst("omes1", ome1);
  g_syst->add_cst("omes2", ome2);
  g_syst->add_cst("mg", *coord_vectors[GLOBAL_ROT]);
  g_syst->add_cst("mm", *coord_vectors[BCO1_ROT]);
  g_syst->add_cst("mp", *coord_vectors[BCO2_ROT]);
  g_syst->add_cst("ex", *coord_vectors[EX]);
  g_syst->add_cst("ey", *coord_vectors[EY]);
  g_syst->add_cst("ez", *coord_vectors[EZ]);
  g_syst->add_cst("sm", *coord_vectors[S_BCO1]);
  g_syst->add_cst("sp", *coord_vectors[S_BCO2]);
  g_syst->add_cst("einf", *coord_vectors[S_INF]);
  g_syst->add_cst("xaxis", axis);
  g_syst->add_cst("ome", omega);
  g_syst->add_cst("P", *g_conf);
  g_syst->add_cst("N", *g_lapse);
  g_syst->add_cst("bet", *g_shift);
  g_syst->add_cst("phi", *g_phi);
  g_syst->add_cst("H", *g_logh);

  g_syst->add_def("NP = P*N");
  g_syst->add_def("Ntilde = N / P^6");
  g_syst->add_def("Morb^i = mg^i + xaxis * ey^i");
  g_syst->add_def("omega^i = bet^i + ome * Morb^i");

  for (int d = g_space->NS1; d <= g_space->ADAPTED1; ++d)
    g_syst->add_def(d, "s^i  = omes1 * mm^i");
  for (int d = g_space->NS2; d <= g_space->ADAPTED2; ++d)
    g_syst->add_def(d, "s^i  = omes2 * mp^i");

  g_syst->add_def(
    "A_ij = (D_i bet_j + D_j bet_i - 2. / 3.* D^k bet_k * f_ij) /2. / N");
  g_syst->add_def("h = exp(H)");

  for (int d = 0; d < ndom; ++d)
  {
    if ((d <= g_space->ADAPTED1) || (d >= g_space->NS2 && d <= g_space->ADAPTED2))
      g_syst->add_def(d, "eta_i = D_i phi + P^4 * s_i");
    else
      g_syst->add_def(d, "eta_i = D_i phi");
  }

  g_syst->add_def("Wsquare = eta^i * eta_i / h^2 / P^4 + 1.");
  g_syst->add_def("W = sqrt(Wsquare)");
  g_syst->add_def("U^i = eta^i / P^4 / h / W");

  g_A_tens = new Kadath::Tensor(g_syst->give_val_def("A"));
  Kadath::Index ind(*g_A_tens);
  (*g_quants)[AXX] = std::cref((*g_A_tens)(ind));
  ind.inc();
  (*g_quants)[AXY] = std::cref((*g_A_tens)(ind));
  ind.inc();
  (*g_quants)[AXZ] = std::cref((*g_A_tens)(ind));
  ind.inc();
  ind.inc();
  (*g_quants)[AYY] = std::cref((*g_A_tens)(ind));
  ind.inc();
  (*g_quants)[AYZ] = std::cref((*g_A_tens)(ind));
  ind.inc();
  ind.inc();
  ind.inc();
  (*g_quants)[AZZ] = std::cref((*g_A_tens)(ind));

  (*g_quants)[H] = std::cref(*g_logh);

  g_vel_kad = new Kadath::Vector(g_syst->give_val_def("U"));
  (*g_quants)[UX] = std::cref((*g_vel_kad)(1));
  (*g_quants)[UY] = std::cref((*g_vel_kad)(2));
  (*g_quants)[UZ] = std::cref((*g_vel_kad)(3));

  // Force spectral coefficients to be evaluated for all fields
  for (int kq = 0; kq < NUM_QUANTS; ++kq)
    (*g_quants)[kq].get().coef();

  // Warmup val_point to init summation_1d dispatch table
  {
    Kadath::Point pt_warm(3);
    pt_warm.set(1) = xc1;
    pt_warm.set(2) = 0.0;
    pt_warm.set(3) = 0.0;
    (void)(*g_quants)[PSI].get().val_point(pt_warm);
  }

  if (Globals::my_rank == 0)
  {
    std::cout << "Kadath system assembled." << std::endl;
  }

  // --------------------------------------------------------------------------
  // Determine star centres and peak densities (for B-field seeding)
  Real dx_interp = pin->GetOrAddReal("problem", "dx_interp_maximum", 0.01);
  int npt_interp = static_cast<int>(std::round(sep / dx_interp));

  {
    std::vector<double> xp(npt_interp), xn(npt_interp), yz(npt_interp, 0.0);
    for (int i = 0; i < npt_interp; ++i)
      xp[i] = i * dx_interp;
    for (int i = 0; i < npt_interp; ++i)
      xn[i] = -(npt_interp - 1 - i) * dx_interp;

    auto logh_point = [&](double x, double y, double z) -> double {
      Kadath::Point pt(3);
      pt.set(1) = x - axis;
      pt.set(2) = y;
      pt.set(3) = z;
      return (*g_quants)[H].get().val_point(pt);
    };

    // Star 1 (x < 0)
    double max_h = -1.0;
    int imax     = 0;
    for (int i = 0; i < npt_interp; ++i)
    {
      double h = logh_point(xn[i], 0.0, 0.0);
      if (h > max_h)
      {
        max_h = h;
        imax  = i;
      }
    }
    centre_m = xn[imax];

    Real h_max = std::exp(max_h);
    if (g_use_cold_table)
    {
      pgasmax_1 = ::EOS<Kadath::Margherita::Cold_Table, ::PRESSURE>::get(h_max);
    }
    else
    {
      pgasmax_1 = ::EOS<Kadath::Margherita::Cold_PWPoly, ::PRESSURE>::get(h_max);
    }

    // Star 2 (x > 0)
    max_h = -1.0;
    imax  = 0;
    for (int i = 0; i < npt_interp; ++i)
    {
      double h = logh_point(xp[i], 0.0, 0.0);
      if (h > max_h)
      {
        max_h = h;
        imax  = i;
      }
    }
    centre_p = xp[imax];

    h_max     = std::exp(max_h);
    if (g_use_cold_table)
    {
      pgasmax_2 = ::EOS<Kadath::Margherita::Cold_Table, ::PRESSURE>::get(h_max);
    }
    else
    {
      pgasmax_2 = ::EOS<Kadath::Margherita::Cold_PWPoly, ::PRESSURE>::get(h_max);
    }
  }

  if (Globals::my_rank == 0)
  {
    std::cout << "  Star 1 centre: " << centre_m << "  peak " << pgasmax_1 << std::endl;
    std::cout << "  Star 2 centre: " << centre_p << "  peak " << pgasmax_2 << std::endl;
  }

#ifdef MPI_PARALLEL
  {
    const bool synchronize_ns_extrema =
      pin->GetOrAddBoolean("problem", "synchronize_ns_extrema", true);
    if (synchronize_ns_extrema)
    {
      Real buf[4] = { pgasmax_1, pgasmax_2, centre_m, centre_p };
      MPI_Bcast(buf, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      pgasmax_1 = buf[0];
      pgasmax_2 = buf[1];
      centre_m  = buf[2];
      centre_p  = buf[3];
    }
  }
#endif

  if (Globals::my_rank == 0) {
    std::printf("NS star centres from FUKA data (grid coordinates):\n");
    std::printf("  NS1 (x < 0): x = %.6f\n", centre_m);
    std::printf("  NS2 (x > 0): x = %.6f\n", centre_p);
    std::printf("Set bh_0_x = %.6f, bh_1_x = %.6f in input file.\n",
                centre_m, centre_p);
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
//========================================================================================

void MeshBlock::InitUserMeshBlockData(ParameterInput* pin)
{
  const bool use_fb = precon->xorder_use_fb;
  AllocateUserOutputVariables(use_fb + M1_ENABLED * 4);
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput* pin)
{
  MeshBlock* pmb = this;
  const bool use_fb = precon->xorder_use_fb;
  if (use_fb)
    CC_GLOOP3(k, j, i)
    {
      user_out_var(0, k, j, i) = phydro->fallback_mask(k, j, i);
    }
}

void MeshBlock::UserWorkAfterOutput(ParameterInput* pin)
{
  AA c2p_status;
  c2p_status.InitWithShallowSlice(phydro->derived_ms, IX_C2P, 1);
  c2p_status.Fill(0);
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Interpolates Kadath BNS initial data onto the grid.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput* pin)
{
  using namespace LinearAlgebra;

  Real const tol_det_zero =
    pin->GetOrAddReal("problem", "tolerance_det_zero", 1e-10);

  AthenaArray<Real> empty;
#if NSCALARS > 0
  AthenaArray<Real>& r_scalar = pscalars->r;
  AthenaArray<Real>& s_scalar = pscalars->s;
#else
  AthenaArray<Real>& r_scalar = empty;
  AthenaArray<Real>& s_scalar = empty;
#endif

  MB_info* mbi = &(pz4c->mbi);

  AT_N_sca alpha(pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec beta_u(pz4c->storage.adm, Z4c::I_ADM_betax);
  AT_N_sym g_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym K_dd(pz4c->storage.adm, Z4c::I_ADM_Kxx);

  const int il = 0, iu = ncells1 - 1;
  const int jl = 0, ju = ncells2 - 1;
  const int kl = 0, ku = ncells3 - 1;

  Real w_rho_atm = pin->GetReal("hydro", "dfloor");
  Real rho_cut =
    std::max(pin->GetOrAddReal("problem", "rho_cut", w_rho_atm), w_rho_atm);

  // --------------------------------------------------------------------------
  // Geometry pass: fill ADM variables on the Z4c geometry grid
#pragma omp critical
  {
    if (!g_quants)
    {
      std::cout << "### FATAL ERROR: Kadath fields not initialised." << std::endl;
      std::_Exit(EXIT_FAILURE);
    }

    for (int k = 0; k < mbi->nn3; ++k)
      for (int j = 0; j < mbi->nn2; ++j)
        for (int i = 0; i < mbi->nn1; ++i)
        {
          Kadath::Point pt(3);
          pt.set(1) = mbi->x1(i) - g_com_offset;
          pt.set(2) = mbi->x2(j);
          pt.set(3) = mbi->x3(k);

          double qv[NUM_QUANTS];
          for (int kq = 0; kq < NUM_QUANTS; ++kq)
            qv[kq] = (*g_quants)[kq].get().val_point(pt);

          const double psi  = qv[PSI];
          const double psi4 = psi * psi * psi * psi;

          alpha(k, j, i)        = qv[ALP];
          beta_u(0, k, j, i)    = qv[BETX];
          beta_u(1, k, j, i)    = qv[BETY];
          beta_u(2, k, j, i)    = qv[BETZ];

          g_dd(0, 0, k, j, i)   = psi4;
          g_dd(0, 1, k, j, i)   = 0.0;
          g_dd(0, 2, k, j, i)   = 0.0;
          g_dd(1, 1, k, j, i)   = psi4;
          g_dd(1, 2, k, j, i)   = 0.0;
          g_dd(2, 2, k, j, i)   = psi4;

          K_dd(0, 0, k, j, i)   = qv[AXX] * psi4;
          K_dd(0, 1, k, j, i)   = qv[AXY] * psi4;
          K_dd(0, 2, k, j, i)   = qv[AXZ] * psi4;
          K_dd(1, 1, k, j, i)   = qv[AYY] * psi4;
          K_dd(1, 2, k, j, i)   = qv[AYZ] * psi4;
          K_dd(2, 2, k, j, i)   = qv[AZZ] * psi4;

          const Real det = Det3Metric(g_dd, k, j, i);
          if (std::fabs(det) <= tol_det_zero)
          {
            std::cout << "### WARNING: det(g) = " << det
                      << " at (i,j,k)=(" << i << "," << j << "," << k << ")"
                      << std::endl;
          }
        }
  }

  // --------------------------------------------------------------------------
  // Matter pass: fill hydro primitives on the cell-centred grid
#pragma omp critical
  {
    AthenaArray<Real>& w = phydro->w;
#if NSCALARS > 0
    AthenaArray<Real>& r = pscalars->r;
    AthenaArray<Real>& s = pscalars->s;
    r.Fill(0.0);
    s.Fill(0.0);
#endif

    for (int k = kl; k <= ku; ++k)
      for (int j = jl; j <= ju; ++j)
        for (int i = il; i <= iu; ++i)
        {
          Kadath::Point pt(3);
          pt.set(1) = pcoord->x1v(i) - g_com_offset;
          pt.set(2) = pcoord->x2v(j);
          pt.set(3) = pcoord->x3v(k);

          double qv[NUM_QUANTS];
          for (int kq = 0; kq < NUM_QUANTS; ++kq)
            qv[kq] = (*g_quants)[kq].get().val_point(pt);

          const double h_enth = std::exp(qv[H]);

          Real w_rho, w_p;
          if (h_enth == 1.0)
          {
            w_rho = 0.0;
            w_p   = 0.0;
          }
          else
          {
            if (g_use_cold_table)
            {
              using eos_t = Kadath::Margherita::Cold_Table;
              w_rho = ::EOS<eos_t, ::DENSITY>::get(h_enth);
              w_p   = ::EOS<eos_t, ::PRESSURE>::get(h_enth);
            }
            else // g_use_cold_pwpoly
            {
              using eos_t = Kadath::Margherita::Cold_PWPoly;
              w_rho = ::EOS<eos_t, ::DENSITY>::get(h_enth);
              w_p   = ::EOS<eos_t, ::PRESSURE>::get(h_enth);
            }

            if (w_rho < rho_cut)
            {
              w_rho = 0.0;
              w_p   = 0.0;
            }
          }

          // Reconcile FUKA's atomic-mass-unit density with the table's
          // neutron-mass convention. No-op (factor 1) when disabled.
          w_rho *= g_fuka_rho_rescale;

          const double psi4 = qv[PSI] * qv[PSI] * qv[PSI] * qv[PSI];

          // Velocity U^i with full spatial metric g_ij = psi^4 delta_ij
          Real vu[3] = { static_cast<Real>(qv[UX]),
                         static_cast<Real>(qv[UY]),
                         static_cast<Real>(qv[UZ]) };

          Real vsq = psi4 * (vu[0] * vu[0] + vu[1] * vu[1] + vu[2] * vu[2]);
          if (1.0 - vsq <= 0.0)
          {
            Real fac = std::sqrt((1.0 - 1e-15) / vsq);
            vu[0] *= fac;
            vu[1] *= fac;
            vu[2] *= fac;
            vsq = 1.0 - 1.0e-15;
          }

          Real W_lorentz = 1.0 / std::sqrt(1.0 - vsq);

          w(IDN, k, j, i) = w_rho;
          w(IPR, k, j, i) = w_p;
          w(IVX, k, j, i) = W_lorentz * vu[0];
          w(IVY, k, j, i) = W_lorentz * vu[1];
          w(IVZ, k, j, i) = W_lorentz * vu[2];
        }
  }

  // --------------------------------------------------------------------------
  // Post-processing: populate composition scalars from ColdEOS.
  // Kadath EOS already set (rho, P) directly; only atmosphere points need reset.
  {
    AthenaArray<Real>& w = phydro->w;

#if NSCALARS > 0
    Real Y_atm[NSCALARS] = { 0.0 };
    for (int iy = 0; iy < NSCALARS; ++iy)
      Y_atm[iy] =
        pin->GetReal("hydro", "y" + std::to_string(iy) + "_atmosphere");
#endif

    for (int k = kl; k <= ku; ++k)
      for (int j = jl; j <= ju; ++j)
        for (int i = il; i <= iu; ++i)
        {
          if (w(IDN, k, j, i) > rho_cut)
          {
#if NSCALARS > 0
            for (int iy = 0; iy < NSCALARS; ++iy)
              r_scalar(iy, k, j, i) = ceos->GetY(w(IDN, k, j, i), iy);
#endif
          }
          else
          {
            w(IDN, k, j, i) = 0.0;
            w(IPR, k, j, i) = 0.0;

#if NSCALARS > 0
            for (int iy = 0; iy < NSCALARS; ++iy)
              r_scalar(iy, k, j, i) = Y_atm[iy];
#endif

            for (int ix = 0; ix < 3; ++ix)
              w(IVX + ix, k, j, i) = 0.0;
          }
        }
  }

  // --------------------------------------------------------------------------
#if MAGNETIC_FIELDS_ENABLED
  for (int k = 0; k < ncells3; ++k)
    for (int j = 0; j < ncells2; ++j)
      for (int i = 0; i < ncells1; ++i)
      {
        for (int n = 0; n < NHYDRO; ++n)
          if (!std::isfinite(phydro->w(n, k, j, i)))
          {
            PrimHelper::ApplyPrimitiveFloors(
              peos->GetEOS(), phydro->w, r_scalar, k, j, i);
            continue;
          }
      }

  SeedMagneticFields(this, pin);
#endif

  // --------------------------------------------------------------------------
  // Construct Z4c vars from ADM vars
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);

  bool fix_gauge_precollapsed =
    pin->GetOrAddBoolean("problem", "fix_gauge_precollapsed", false);
  if (fix_gauge_precollapsed)
  {
    pz4c->Z4cToADM(pz4c->storage.u, pz4c->storage.adm);
    pz4c->GaugePreCollapsedLapse(pz4c->storage.adm, pz4c->storage.u);
  }

  // --------------------------------------------------------------------------
  // Floor primitives if requested
  bool id_floor_primitives =
    pin->GetOrAddBoolean("problem", "id_floor_primitives", false);
  if (id_floor_primitives)
  {
    for (int k = 0; k <= ncells3 - 1; ++k)
      for (int j = 0; j <= ncells2 - 1; ++j)
        for (int i = 0; i <= ncells1 - 1; ++i)
        {
          PrimHelper::ApplyPrimitiveFloors(
            peos->GetEOS(), phydro->w, r_scalar, k, j, i);
        }
  }

  // Initialise conserved variables
  peos->PrimitiveToConserved(phydro->w,
                             r_scalar,
                             pfield->bcc,
                             phydro->u,
                             s_scalar,
                             pcoord,
                             0,
                             ncells1 - 1,
                             0,
                             ncells2 - 1,
                             0,
                             ncells3 - 1);

  return;
}

//========================================================================================
//! \fn void Mesh::DeleteTemporaryUserMeshData()
//========================================================================================

void Mesh::DeleteTemporaryUserMeshData()
{
  delete ceos;
  ceos = NULL;

  if (g_quants)
  {
    delete g_vel_kad;
    g_vel_kad = NULL;
    delete g_A_tens;
    g_A_tens = NULL;
    delete g_syst;
    g_syst = NULL;
    delete g_fmet;
    g_fmet = NULL;
    delete g_basis;
    g_basis = NULL;
    delete g_phi;
    g_phi = NULL;
    delete g_logh;
    g_logh = NULL;
    delete g_shift;
    g_shift = NULL;
    delete g_lapse;
    g_lapse = NULL;
    delete g_conf;
    g_conf = NULL;
    delete g_space;
    g_space = NULL;
    delete g_quants;
    g_quants = NULL;
  }

  return;
}
