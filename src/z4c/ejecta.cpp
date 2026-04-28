//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================

// C++ standard headers
#include <unistd.h>

#include <cmath>  // NAN
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

// Athena++ headers
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../outputs/outputs.hpp"
#include "../parameter_input.hpp"
#include "../utils/linear_algebra.hpp"
#include "../utils/utils.hpp"
#include "ejecta.hpp"
#include "z4c_macro.hpp"

// External libraries
// ..

// ============================================================================

namespace
{

std::string prim_names[NHYDRO] = {
  "rho", "vx", "vy", "vz", "press",
};

std::string cons_names[NHYDRO] = {
  "dens", "Mx", "My", "Mz", "tau",
};

// std::string adm_names[Z4c::N_ADM] = {
//     "gxx", "gxy", "gxz", "gyy", "gyz", "gzz",  "Kxx",
//     "Kxy", "Kxz", "Kyy", "Kyz", "Kzz", "psi4",
// };

// std::string z4c_names[Z4c::N_Z4c] = {
//     "chi",  "gxx",   "gxy",   "gxz",   "gyy",   "gyz",   "gzz",  "Khat",
//     "Axx",  "Axy",   "Axz",   "Ayy",   "Ayz",   "Azz",   "Gamx", "Gamy",
//     "Gamz", "Theta", "alpha", "betax", "betay", "betaz",
// };

std::string other_names[Ejecta::NOTHER] = { "detg",     "Mdot",    "bernoulli",
                                            "enthalpy", "entropy", "lorentz",
                                            "u_t",      "fD_r",    "v_mag",
                                            "poynting" };
}  // namespace

//-----------------------------------------------------------------------------
//! \fn Ejecta::Ejecta(Mesh * pmesh, ParameterInput * pin, int n)
//  \brief class for ejecta extraction class
Ejecta::Ejecta(Mesh* pmesh, ParameterInput* pin, int n)
    : pmesh(pmesh), pin(pin)
{
  file_number = pin->GetOrAddInteger("ejecta", "file_number", 0);

  std::string int_names[n_int] = { "mass_", "entr_", "rho_", "temp_",
                                   "ye_",   "vel_",  "ber_", "velinf_" };

  std::string hist_names[n_hist] = { "entr_", "logrho_", "temp_",  "ye_",
                                     "vel_",  "ber_",    "theta_", "velinf_" };

  std::string unbound_names[n_unbound] = {
    "bernoulli_", "bernoulli_outflow_", "geodesic_", "geodesic_outflow_"
  };
  nrad = pin->GetOrAddInteger("ejecta", "num_rad", 1);
  nr   = n;
  std::string parname;
  std::string n_str = std::to_string(nr);

  {
    const int ntheta_local = pin->GetOrAddInteger("ejecta", "ntheta", 10);
    const int nphi_local   = pin->GetOrAddInteger("ejecta", "nphi", 6);
    grid_.Initialize(ntheta_local, nphi_local, "midpoint");
  }

  bitant = pin->GetOrAddBoolean("mesh", "bitant", false);

  parname = "radius_";
  parname += n_str;
  radius = pin->GetOrAddReal("ejecta", parname, 300);

  start_time = pin->GetOrAddReal("ejecta", "start_time", 0.0);
  stop_time  = pin->GetOrAddReal("ejecta", "stop_time", 10000.0);

  mass_contained = 0.0;

  for (int n = 0; n < NHYDRO; ++n)
  {
    prim[n].NewAthenaArray(grid_.ntheta, grid_.nphi);
    cons[n].NewAthenaArray(grid_.ntheta, grid_.nphi);
  }
  for (int n = 0; n < NDIM; ++n)
  {
    Bcc[n].NewAthenaArray(grid_.ntheta, grid_.nphi);
  }
#if FLUID_ENABLED
  for (int n = 0; n < NSCALARS; ++n)
  {
    Y[n].NewAthenaArray(grid_.ntheta, grid_.nphi);
  }
  T.NewAthenaArray(grid_.ntheta, grid_.nphi);
#endif

  for (int n = 0; n < Z4c::N_ADM; ++n)
  {
    adm[n].NewAthenaArray(grid_.ntheta, grid_.nphi);
  }
  for (int n = 0; n < Z4c::N_Z4c; ++n)
  {
    z4c[n].NewAthenaArray(grid_.ntheta, grid_.nphi);
  }
  for (int n = 0; n < NOTHER; ++n)
  {
    other[n].NewAthenaArray(grid_.ntheta, grid_.nphi);
  }
  // n iterates over unboundedness criteria, m over variables, l over bins in
  // histogram i over theta j over phi

  //  {I_hist_entr,I_hist_logrho,I_hist_temp,I_hist_ye,I_hist_vel,I_hist_ber,
  //   I_hist_theta,I_hist_velinf}
  Real def_max[n_hist] = { 200.0, -2.5, 10.0, 0.55, 1.0, 1.0, M_PI, 1.0 };
  Real def_min[n_hist] = { 0.0, -15.0, 0.0, 0.035, 0.0, 0.0, 0.0, 0.0 };
  Real max_hist[n_hist], min_hist[n_hist];
  for (int m = 0; m < n_hist; ++m)
  {
    parname = "hist_n_";
    parname += int_names[m];
    n_bins[m] = pin->GetOrAddInteger("ejecta", parname, 50);
    parname   = "hist_max_";
    parname += int_names[m];
    max_hist[m] = pin->GetOrAddReal("ejecta", parname, def_max[m]);
    parname     = "hist_min_";
    parname += int_names[m];
    min_hist[m] = pin->GetOrAddReal("ejecta", parname, def_min[m]);
  }

  for (int m = 0; m < n_hist; ++m)
  {
    hist_grid[m].NewAthenaArray(n_bins[m]);
  }
  for (int m = 0; m < n_hist; ++m)
  {
    delta_hist[m] = (max_hist[m] - min_hist[m]) / n_bins[m];
    for (int l = 0; l < n_bins[m]; ++l)
    {
      hist_grid[m](l) = min_hist[m] + l * delta_hist[m];
    }
  }

  integrals_unbound.NewAthenaArray(n_unbound, n_int);
  az_integrals_unbound.NewAthenaArray(n_unbound, n_int, grid_.ntheta);
  for (int m = 0; m < n_hist; ++m)
  {
    hist[m].NewAthenaArray(n_unbound, n_bins[m]);
  }

  // Prepare output
  parname = "ejecta_file_summary_";
  parname += n_str;
  ofname_summary = pin->GetString("job", "problem_id") + ".";
  ofname_summary +=
    pin->GetOrAddString("ejecta", parname, "ejecta_summary_" + n_str);
  ofname_summary += ".txt";

  for (int n = 0; n < n_unbound; ++n)
  {
    parname = "ejecta_file_";
    parname += unbound_names[n];
    parname += n_str;
    ofname_unbound[n] = pin->GetString("job", "problem_id") + ".";
    ofname_unbound[n] += pin->GetOrAddString("ejecta", parname, parname);
    ofname_unbound[n] += ".txt";
  }
  for (int n = 0; n < n_unbound; ++n)
  {
    parname = "ejecta_file_az_";
    parname += unbound_names[n];
    parname += n_str;
    ofname_az_unbound[n] = pin->GetString("job", "problem_id") + ".";
    ofname_az_unbound[n] += pin->GetOrAddString("ejecta", parname, parname);
    ofname_az_unbound[n] += ".txt";
  }
  for (int n = 0; n < n_unbound; ++n)
  {
    for (int m = 0; m < n_hist; ++m)
    {
      parname = "ejecta_file_histogram_";
      parname += unbound_names[n];
      parname += hist_names[m];
      parname += n_str;
      ofname_hist_unbound[n][m] = pin->GetString("job", "problem_id") + ".";
      ofname_hist_unbound[n][m] +=
        pin->GetOrAddString("ejecta", parname, parname);
      ofname_hist_unbound[n][m] += ".txt";
    }
  }

  if (Globals::my_rank == 0)
  {
    // Summary file
    bool new_file = true;
    if (access(ofname_summary.c_str(), F_OK) == 0)
    {
      new_file = false;
    }
    pofile_summary = fopen(ofname_summary.c_str(), "a");
    if (NULL == pofile_summary)
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in AHF constructor" << std::endl;
      msg << "Could not open file '" << pofile_summary << "' for writing!";
      throw std::runtime_error(msg.str().c_str());
    }
    if (new_file)
    {
      fprintf(pofile_summary,
              "# 1:iter 2:time 3:rho 4:press 5:Ye 6:vx 7:vy "
              "8:vz 9:Bx 10:By 11:Bz\n");
      fflush(pofile_summary);
    }

    for (int n = 0; n < n_unbound; ++n)
    {
      new_file = true;
      if (access(ofname_unbound[n].c_str(), F_OK) == 0)
      {
        new_file = false;
      }
      pofile_unbound[n] = fopen(ofname_unbound[n].c_str(), "a");
      if (NULL == pofile_unbound[n])
      {
        std::stringstream msg;
        msg << "### FATAL ERROR in Ejecta constructor" << std::endl;
        msg << "Could not open file '" << pofile_unbound[n]
            << "' for writing!";
        throw std::runtime_error(msg.str().c_str());
      }
      if (new_file)
      {
        fprintf(pofile_unbound[n],
                "# 1:iter 2:time 3:mass 4:entropy 5:rho 6:temperature 7:Ye "
                "8:velocity 9:bernoulli 10:velocity_inf \n");
        fflush(pofile_unbound[n]);
      }
    }

    for (int n = 0; n < n_unbound; ++n)
    {
      new_file = true;
      if (access(ofname_az_unbound[n].c_str(), F_OK) == 0)
      {
        new_file = false;
      }
      pofile_az_unbound[n] = fopen(ofname_az_unbound[n].c_str(), "a");
      if (NULL == pofile_az_unbound[n])
      {
        std::stringstream msg;
        msg << "### FATAL ERROR in Ejecta constructor" << std::endl;
        msg << "Could not open file '" << pofile_az_unbound[n]
            << "' for writing!";
        throw std::runtime_error(msg.str().c_str());
      }
      if (new_file)
      {
        fprintf(pofile_az_unbound[n],
                "# 1:theta 2:mass 3:entropy 4:rho 5:temperature 6:Ye "
                "7:velocity 8:bernoulli 9:velocity_inf \n");
        fflush(pofile_az_unbound[n]);
      }
    }
    for (int n = 0; n < n_unbound; ++n)
    {
      for (int m = 0; m < n_hist; ++m)
      {
        new_file = true;
        if (access(ofname_hist_unbound[n][m].c_str(), F_OK) == 0)
        {
          new_file = false;
        }
        pofile_hist_unbound[n][m] =
          fopen(ofname_hist_unbound[n][m].c_str(), "a");
        if (NULL == pofile_hist_unbound[n][m])
        {
          std::stringstream msg;
          msg << "### FATAL ERROR in Ejecta constructor" << std::endl;
          msg << "Could not open file '" << pofile_hist_unbound[n][m]
              << "' for writing!";
          throw std::runtime_error(msg.str().c_str());
        }
        if (new_file)
        {
          fprintf(pofile_hist_unbound[n][m], "# 1:bin 2:weight \n");
          fflush(pofile_hist_unbound[n][m]);
        }
      }
    }
  }
}

Ejecta::~Ejecta()
{
  // Close files
  if (Globals::my_rank == 0)
  {
    fclose(pofile_summary);
    for (int n = 0; n < n_unbound; ++n)
    {
      fclose(pofile_az_unbound[n]);
      fclose(pofile_unbound[n]);
      for (int m = 0; m < n_hist; ++m)
      {
        fclose(pofile_hist_unbound[n][m]);
      }
    }
  }
}

void Ejecta::Mass(MeshBlock* pmb)
{
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks,
      ke = pmb->ke;
  for (int k = ks; k <= ke; k++)
  {
    Real z = pmb->pcoord->x3v(k);
    for (int j = js; j <= je; j++)
    {
      Real y = pmb->pcoord->x2v(j);
      for (int i = is; i <= ie; i++)
      {
        Real x   = pmb->pcoord->x1v(i);
        Real vol = (pmb->pcoord->dx1v(i)) * (pmb->pcoord->dx2v(j)) *
                   (pmb->pcoord->dx3v(k));
        Real rad2 = x * x + y * y + z * z;
        if (rad2 < SQR(radius))
        {
          mass_contained += vol * pmb->phydro->u(IDN, k, j, i);
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------
// \!fn void Ejecta::Interp()
// \brief interpolate quantities on the sphere using pre-built interpolator
// pools
void Ejecta::Interp()
{
  using namespace LinearAlgebra;
  using InterpType = LagrangeInterpND<ejecta_interp_order, 3>;

  // Pool references
  std::vector<InterpType>& pool_cc = grid_.interp_pool_cc;
  std::vector<InterpType>& pool_met =
    SW_CCX_VC(grid_.interp_pool_cc, grid_.interp_pool_vc);

  ATP_N_sym adm_g_dd;

  for (int i = 0; i < grid_.ntheta; ++i)
  {
    const Real sinth = std::sin(grid_.th_grid(i));
    const Real costh = std::cos(grid_.th_grid(i));

    for (int j = 0; j < grid_.nphi; ++j)
    {
      if (!grid_.IsOwned(i, j))
        continue;

      MeshBlock* pmb         = grid_.mask_mb(i, j);
      const int pidx         = grid_.mask_interp_idx(i, j);
      InterpType& interp_cc  = pool_cc[pidx];
      InterpType& interp_met = pool_met[pidx];

      // Shallow slices for this MeshBlock
      AthenaArray<Real> prim_[NHYDRO], cons_[NHYDRO];
      for (int n = 0; n < NHYDRO; ++n)
      {
        prim_[n].InitWithShallowSlice(pmb->phydro->w, IDN + n, 1);
        cons_[n].InitWithShallowSlice(pmb->phydro->u, IDN + n, 1);
      }
#if FLUID_ENABLED
      AthenaArray<Real> Y_[NSCALARS];
      for (int n = 0; n < NSCALARS; ++n)
      {
        Y_[n].InitWithShallowSlice(pmb->pscalars->r, IYF + n, 1);
      }
#endif
      AthenaArray<Real> Bcc_[NDIM];
      for (int n = 0; n < NDIM; ++n)
      {
        if (MAGNETIC_FIELDS_ENABLED)
          Bcc_[n].InitWithShallowSlice(pmb->pfield->bcc, IB1 + n, 1);
      }

      AthenaArray<Real> adm_[Z4c::N_ADM], z4c_[Z4c::N_Z4c];
      for (int n = 0; n < Z4c::N_ADM; ++n)
        adm_[n].InitWithShallowSlice(pmb->pz4c->storage.adm, n, 1);
      for (int n = 0; n < Z4c::N_Z4c; ++n)
        z4c_[n].InitWithShallowSlice(pmb->pz4c->storage.u, n, 1);

      const Real sinph = std::sin(grid_.ph_grid(j));
      const Real cosph = std::cos(grid_.ph_grid(j));

      // Interpolate hydro (CC pool)
      for (int n = 0; n < NHYDRO; ++n)
      {
        prim[n](i, j) = interp_cc.eval(&(prim_[n](0, 0, 0)));
        cons[n](i, j) = interp_cc.eval(&(cons_[n](0, 0, 0)));
      }
#if FLUID_ENABLED
      for (int n = 0; n < NSCALARS; ++n)
      {
        Y[n](i, j) = interp_cc.eval(&(Y_[n](0, 0, 0)));
      }
      Real Ypt[NSCALARS];
      for (int n = 0; n < NSCALARS; ++n)
      {
        Ypt[n] = Y[n](i, j);
      }
      Real npt    = prim[IDN](i, j) / pmb->peos->GetEOS().GetBaryonMass();
      Real Wvu[3] = { prim[IVX](i, j), prim[IVY](i, j), prim[IVZ](i, j) };
      Real prpt   = prim[IPR](i, j);
      pmb->peos->GetEOS().ApplyDensityLimits(npt);
      pmb->peos->GetEOS().ApplySpeciesLimits(Ypt);
      T(i, j) = pmb->peos->GetEOS().GetTemperatureFromP(npt, prpt, Ypt);
#endif
      if (MAGNETIC_FIELDS_ENABLED)
      {
        for (int n = 0; n < NDIM; ++n)
        {
          Bcc[n](i, j) = interp_cc.eval(&(Bcc_[n](0, 0, 0)));
        }
      }

      // Interpolate metric (CC or VC pool, depending on Z4c centering)
      for (int n = 0; n < Z4c::N_ADM; ++n)
      {
        adm[n](i, j) = interp_met.eval(&(adm_[n](0, 0, 0)));
      }
      for (int n = 0; n < Z4c::N_Z4c; ++n)
      {
        z4c[n](i, j) = interp_met.eval(&(z4c_[n](0, 0, 0)));
      }
      other[I_detg](i, j) = Det3Metric(adm[0](i, j),
                                       adm[1](i, j),
                                       adm[2](i, j),
                                       adm[3](i, j),
                                       adm[4](i, j),
                                       adm[5](i, j));

      adm_g_dd(0, 0) = adm[Z4c::I_ADM_gxx](i, j);
      adm_g_dd(0, 1) = adm[Z4c::I_ADM_gxy](i, j);
      adm_g_dd(0, 2) = adm[Z4c::I_ADM_gxz](i, j);
      adm_g_dd(1, 1) = adm[Z4c::I_ADM_gyy](i, j);
      adm_g_dd(1, 2) = adm[Z4c::I_ADM_gyz](i, j);
      adm_g_dd(2, 2) = adm[Z4c::I_ADM_gzz](i, j);
      Mdot_total += other[1](i, j);
      other[I_lorentz](i, j) = 1.0;
      for (int a = 0; a < NDIM; ++a)
      {
        for (int b = 0; b < NDIM; ++b)
        {
          other[I_lorentz](i, j) +=
            adm_g_dd(a, b) * prim[IVX + a](i, j) * prim[IVX + b](i, j);
        }
      }
      other[I_lorentz](i, j) = sqrt(other[I_lorentz](i, j));

      Real vx = prim[IVX](i, j) / other[I_lorentz](i, j);
      Real vy = prim[IVY](i, j) / other[I_lorentz](i, j);
      Real vz = prim[IVZ](i, j) / other[I_lorentz](i, j);

      Real fx = cons[IDN](i, j) * (z4c[Z4c::I_Z4c_alpha](i, j) * vx -
                                   z4c[Z4c::I_Z4c_betax](i, j));
      Real fy = cons[IDN](i, j) * (z4c[Z4c::I_Z4c_alpha](i, j) * vy -
                                   z4c[Z4c::I_Z4c_betay](i, j));
      Real fz = cons[IDN](i, j) * (z4c[Z4c::I_Z4c_alpha](i, j) * vz -
                                   z4c[Z4c::I_Z4c_betaz](i, j));

      other[I_fD_r](i, j) =
        fx * cosph * sinth + fy * sinph * sinth + fz * costh;
      other[I_Mdot](i, j) =
        MassLossRate(fx, fy, fz, sinth, costh, sinph, cosph);
      other[I_v_mag](i, j) = sqrt(1.0 - 1.0 / SQR(other[I_lorentz](i, j)));

      other[I_u_t](i, j) =
        -other[I_lorentz](i, j) * z4c[Z4c::I_Z4c_alpha](i, j);
      for (int a = 0; a < NDIM; ++a)
      {
        for (int b = 0; b < NDIM; ++b)
        {
          other[I_u_t](i, j) += adm_g_dd(a, b) *
                                z4c[Z4c::I_Z4c_betax + a](i, j) *
                                prim[IVX + b](i, j);
        }
      }
      Real temppt = T(i, j);
      pmb->peos->GetEOS().ApplyPrimitiveFloor(npt, Wvu, prpt, temppt, Ypt);
      other[I_enthalpy](i, j) =
        pmb->peos->GetEOS().GetEnthalpy(npt, temppt, Ypt);
      other[I_entropy](i, j) =
        pmb->peos->GetEOS().GetEntropyPerBaryon(npt, temppt, Ypt);
      other[I_bernoulli](i, j) =
        -other[I_enthalpy](i, j) * other[I_u_t](i, j) - 1.0;
      Real bi_u[3];
      Real b0_u = 0.0;
      for (int a = 0; a < NDIM; ++a)
      {
        for (int b = 0; b < NDIM; ++b)
        {
          b0_u += Bcc[a](i, j) * prim[IVX + b](i, j) * adm_g_dd(a, b) /
                  z4c[Z4c::I_Z4c_alpha](i, j);
        }
      }
      for (int a = 0; a < NDIM; ++a)
      {
        bi_u[a] = Bcc[a](i, j) / other[I_lorentz](i, j) +
                  z4c[Z4c::I_Z4c_alpha](i, j) * b0_u * prim[IVX + a](i, j) -
                  b0_u * z4c[Z4c::I_Z4c_betax + a](i, j);
      }

      Real bsq = SQR(z4c[Z4c::I_Z4c_alpha](i, j) * b0_u);
      for (int a = 0; a < NDIM; ++a)
      {
        for (int b = 0; b < NDIM; ++b)
        {
          bsq += Bcc[a](i, j) * Bcc[b](i, j) * adm_g_dd(a, b);
        }
      }
      bsq /= SQR(other[I_lorentz](i, j));

      Real ui_u[3];
      for (int a = 0; a < NDIM; ++a)
      {
        ui_u[a] = prim[IVX + a](i, j) - other[I_lorentz](i, j) *
                                          z4c[Z4c::I_Z4c_betax + a](i, j) /
                                          z4c[Z4c::I_Z4c_alpha](i, j);
      }

      Real ur_u = 0.0;
      for (int a = 0; a < NDIM; ++a)
      {
        ur_u += grid_.x_cart(a, i, j) * ui_u[a] / radius;
      }
      Real br_u = 0.0;
      for (int a = 0; a < NDIM; ++a)
      {
        br_u += grid_.x_cart(a, i, j) * bi_u[a] / radius;
      }

      Real betasq = 0.0;
      for (int a = 0; a < NDIM; ++a)
      {
        for (int b = 0; b < NDIM; ++b)
        {
          betasq += z4c[Z4c::I_Z4c_betax + a](i, j) *
                    z4c[Z4c::I_Z4c_betax + b](i, j) * adm_g_dd(a, b);
        }
      }

      Real b0_d = (-SQR(z4c[Z4c::I_Z4c_alpha](i, j)) + betasq) * b0_u;
      for (int a = 0; a < NDIM; ++a)
      {
        for (int b = 0; b < NDIM; ++b)
        {
          b0_d += z4c[Z4c::I_Z4c_betax + a](i, j) * bi_u[b] * adm_g_dd(a, b);
        }
      }

      other[I_poynting](i, j) = bsq * ur_u * other[I_u_t](i, j) -
                                br_u * b0_d;  // T^r_t = b^2 u^r u_t - b^r b_t
    }  // phi loop
  }  // theta loop
}

//-----------------------------------------------------------------------------
// \!fn void Ejecta::Calculate(const Real time)
// \brief Calculate ejecta quantities
void Ejecta::Calculate(const Real time)
{
  if ((time < start_time) || (time > stop_time))
    return;

  for (int n = 0; n < NHYDRO; ++n)
  {
    prim[n].ZeroClear();
    cons[n].ZeroClear();
  }
#if FLUID_ENABLED
  for (int n = 0; n < NSCALARS; ++n)
  {
    Y[n].ZeroClear();
  }
  T.ZeroClear();
#endif
  for (int n = 0; n < 3; ++n)
  {
    Bcc[n].ZeroClear();
  }
  for (int n = 0; n < Z4c::N_ADM; ++n)
  {
    adm[n].ZeroClear();
  }
  for (int n = 0; n < Z4c::N_Z4c; ++n)
  {
    z4c[n].ZeroClear();
  }
  for (int n = 0; n < NOTHER; ++n)
  {
    other[n].ZeroClear();
  }
  mass_contained = 0.0;
  Mdot_total     = 0.0;

  // Fill Cartesian coordinates for the fixed-radius sphere
  for (int i = 0; i < grid_.ntheta; ++i)
  {
    const Real sinth = std::sin(grid_.th_grid(i));
    const Real costh = std::cos(grid_.th_grid(i));
    for (int j = 0; j < grid_.nphi; ++j)
    {
      const Real sinph = std::sin(grid_.ph_grid(j));
      const Real cosph = std::cos(grid_.ph_grid(j));

      Real xc = radius * sinth * cosph;
      Real yc = radius * sinth * sinph;
      Real zc = radius * costh;

      // Fold bitant: reflect z to positive hemisphere
      if (bitant)
        zc = std::abs(zc);

      grid_.x_cart(0, i, j) = xc;
      grid_.x_cart(1, i, j) = yc;
      grid_.x_cart(2, i, j) = zc;
    }
  }

  // Lazy Prepare (fixed sphere, survives between calls unless TearDown by AMR)
  if (!grid_.prepared)
    grid_.Prepare(pmesh, true, SW_CCX_VC(false, true));

  // Interpolate on all owned points
  Interp();

  // Mass integral (still per-MeshBlock)
  const std::vector<MeshBlock*>& pmb_array = pmesh->GetMeshBlocksCached();
  for (MeshBlock* pmb : pmb_array)
  {
    Mass(pmb);
  }

  SphericalIntegrals();

// #ifdef MPI_PARALLEL
//   MPI_Allreduce(MPI_IN_PLACE, rho.data(), ntheta*nphi, MPI_ATHENA_REAL,
//   MPI_SUM, MPI_COMM_WORLD);
// #endif
#ifdef MPI_PARALLEL
  int rank;
  const int root = 0;  // always zero

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == root)
  {
    for (int n = 0; n < NHYDRO; ++n)
    {
      MPI_Reduce(MPI_IN_PLACE,
                 prim[n].data(),
                 grid_.ntheta * grid_.nphi,
                 MPI_ATHENA_REAL,
                 MPI_SUM,
                 root,
                 MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE,
                 cons[n].data(),
                 grid_.ntheta * grid_.nphi,
                 MPI_ATHENA_REAL,
                 MPI_SUM,
                 root,
                 MPI_COMM_WORLD);
    }
#if FLUID_ENABLED
    MPI_Reduce(MPI_IN_PLACE,
               T.data(),
               grid_.ntheta * grid_.nphi,
               MPI_ATHENA_REAL,
               MPI_SUM,
               root,
               MPI_COMM_WORLD);
    for (int n = 0; n < NSCALARS; ++n)
    {
      MPI_Reduce(MPI_IN_PLACE,
                 Y[n].data(),
                 grid_.ntheta * grid_.nphi,
                 MPI_ATHENA_REAL,
                 MPI_SUM,
                 root,
                 MPI_COMM_WORLD);
    }
#endif
    for (int n = 0; n < 3; ++n)
    {
      MPI_Reduce(MPI_IN_PLACE,
                 Bcc[n].data(),
                 grid_.ntheta * grid_.nphi,
                 MPI_ATHENA_REAL,
                 MPI_SUM,
                 root,
                 MPI_COMM_WORLD);
    }

    for (int n = 0; n < Z4c::N_ADM; ++n)
    {
      MPI_Reduce(MPI_IN_PLACE,
                 adm[n].data(),
                 grid_.ntheta * grid_.nphi,
                 MPI_ATHENA_REAL,
                 MPI_SUM,
                 root,
                 MPI_COMM_WORLD);
    }
    for (int n = 0; n < Z4c::N_Z4c; ++n)
    {
      MPI_Reduce(MPI_IN_PLACE,
                 z4c[n].data(),
                 grid_.ntheta * grid_.nphi,
                 MPI_ATHENA_REAL,
                 MPI_SUM,
                 root,
                 MPI_COMM_WORLD);
    }
    for (int n = 0; n < NOTHER; ++n)
    {
      MPI_Reduce(MPI_IN_PLACE,
                 other[n].data(),
                 grid_.ntheta * grid_.nphi,
                 MPI_ATHENA_REAL,
                 MPI_SUM,
                 root,
                 MPI_COMM_WORLD);
    }

    MPI_Reduce(MPI_IN_PLACE,
               integrals_unbound.data(),
               n_unbound * n_int,
               MPI_ATHENA_REAL,
               MPI_SUM,
               root,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE,
               az_integrals_unbound.data(),
               n_unbound * n_int * grid_.ntheta,
               MPI_ATHENA_REAL,
               MPI_SUM,
               root,
               MPI_COMM_WORLD);
    for (int m = 0; m < n_hist; ++m)
    {
      MPI_Reduce(MPI_IN_PLACE,
                 hist[m].data(),
                 n_unbound * n_bins[m],
                 MPI_ATHENA_REAL,
                 MPI_SUM,
                 root,
                 MPI_COMM_WORLD);
    }

    MPI_Reduce(MPI_IN_PLACE,
               &Mdot_total,
               1,
               MPI_ATHENA_REAL,
               MPI_SUM,
               root,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE,
               &mass_contained,
               1,
               MPI_ATHENA_REAL,
               MPI_SUM,
               root,
               MPI_COMM_WORLD);
  }
  else
  {
    for (int n = 0; n < NHYDRO; ++n)
    {
      MPI_Reduce(prim[n].data(),
                 prim[n].data(),
                 grid_.ntheta * grid_.nphi,
                 MPI_ATHENA_REAL,
                 MPI_SUM,
                 root,
                 MPI_COMM_WORLD);
      MPI_Reduce(cons[n].data(),
                 cons[n].data(),
                 grid_.ntheta * grid_.nphi,
                 MPI_ATHENA_REAL,
                 MPI_SUM,
                 root,
                 MPI_COMM_WORLD);
    }
#if FLUID_ENABLED
    MPI_Reduce(T.data(),
               T.data(),
               grid_.ntheta * grid_.nphi,
               MPI_ATHENA_REAL,
               MPI_SUM,
               root,
               MPI_COMM_WORLD);
    for (int n = 0; n < NSCALARS; ++n)
    {
      MPI_Reduce(Y[n].data(),
                 Y[n].data(),
                 grid_.ntheta * grid_.nphi,
                 MPI_ATHENA_REAL,
                 MPI_SUM,
                 root,
                 MPI_COMM_WORLD);
    }
#endif
    for (int n = 0; n < 3; ++n)
    {
      MPI_Reduce(Bcc[n].data(),
                 Bcc[n].data(),
                 grid_.ntheta * grid_.nphi,
                 MPI_ATHENA_REAL,
                 MPI_SUM,
                 root,
                 MPI_COMM_WORLD);
    }

    for (int n = 0; n < Z4c::N_ADM; ++n)
    {
      MPI_Reduce(adm[n].data(),
                 adm[n].data(),
                 grid_.ntheta * grid_.nphi,
                 MPI_ATHENA_REAL,
                 MPI_SUM,
                 root,
                 MPI_COMM_WORLD);
    }
    for (int n = 0; n < Z4c::N_Z4c; ++n)
    {
      MPI_Reduce(z4c[n].data(),
                 z4c[n].data(),
                 grid_.ntheta * grid_.nphi,
                 MPI_ATHENA_REAL,
                 MPI_SUM,
                 root,
                 MPI_COMM_WORLD);
    }
    for (int n = 0; n < NOTHER; ++n)
    {
      MPI_Reduce(other[n].data(),
                 other[n].data(),
                 grid_.ntheta * grid_.nphi,
                 MPI_ATHENA_REAL,
                 MPI_SUM,
                 root,
                 MPI_COMM_WORLD);
    }

    MPI_Reduce(integrals_unbound.data(),
               integrals_unbound.data(),
               n_unbound * n_int,
               MPI_ATHENA_REAL,
               MPI_SUM,
               root,
               MPI_COMM_WORLD);
    MPI_Reduce(az_integrals_unbound.data(),
               az_integrals_unbound.data(),
               n_unbound * n_int * grid_.ntheta,
               MPI_ATHENA_REAL,
               MPI_SUM,
               root,
               MPI_COMM_WORLD);
    for (int m = 0; m < n_hist; ++m)
    {
      MPI_Reduce(hist[m].data(),
                 hist[m].data(),
                 n_unbound * n_bins[m],
                 MPI_ATHENA_REAL,
                 MPI_SUM,
                 root,
                 MPI_COMM_WORLD);
    }
    MPI_Reduce(&Mdot_total,
               &Mdot_total,
               1,
               MPI_ATHENA_REAL,
               MPI_SUM,
               root,
               MPI_COMM_WORLD);
    MPI_Reduce(&mass_contained,
               &mass_contained,
               1,
               MPI_ATHENA_REAL,
               MPI_SUM,
               root,
               MPI_COMM_WORLD);
  }
#endif
  Write(time);
}

//-----------------------------------------------------------------------------

void Ejecta::SphericalIntegrals()
{
  Real integrals[n_int];
  Real histvals[n_hist];
  //  AthenaArray<int> count, countaz;
  AA_B unbound;
  //  count.NewAthenaArray(n_unbound);
  //  countaz.NewAthenaArray(n_unbound,ntheta);
  unbound.NewAthenaArray(n_unbound);
  unbound.Fill(false);
  //  count.ZeroClear();
  //  countaz.ZeroClear();

  integrals_unbound.ZeroClear();
  az_integrals_unbound.ZeroClear();
  for (int m = 0; m < n_hist; ++m)
  {
    hist[m].ZeroClear();
  }

  for (int i = 0; i < grid_.ntheta; i++)
  {
    Real sinth = std::sin(grid_.th_grid(i));
    Real costh = std::cos(grid_.th_grid(i));
    for (int j = 0; j < grid_.nphi; j++)
    {
      Real sinph = std::sin(grid_.ph_grid(j));
      Real cosph = std::cos(grid_.ph_grid(j));

      if (!grid_.IsOwned(i, j))
        continue;  // continue if this process doesnt have this point

      if (!(other[I_bernoulli](i, j) > 0.0) && !(other[I_u_t](i, j) < -1.0))
        continue;  // continue if point bound

      Real x = radius * cosph * sinth;
      Real y = radius * sinph * sinth;
      Real z = radius * costh;
      Real vrad =
        x * prim[IVX](i, j) + y * prim[IVY](i, j) + z * prim[IVZ](i, j);
      Real weight = other[I_fD_r](i, j) * SQR(radius) * sinth *
                    grid_.dtheta() *
                    grid_.dphi();  // radial mass flux * area elt

      // values for integrals over spheres
      integrals[I_int_mass] = weight;  // mass flux
      integrals[I_int_rho] =
        prim[IDN](i, j) * weight;  // mass weighted density
      integrals[I_int_entr] =
        other[I_entropy](i, j) * weight;  // mass weighted entropy per baryon
      integrals[I_int_temp] = T(i, j) * weight;     // mass weighted Temp
      integrals[I_int_ye]   = Y[0](i, j) * weight;  // Mass weighted Ye
      integrals[I_int_vel] =
        other[I_v_mag](i, j) * weight;  // mass weighted 3-velocity
      integrals[I_int_ber] =
        other[I_bernoulli](i, j) * weight;  // mass weighted bernoulli
      integrals[I_int_velinf] = sqrt(2.0 * (-other[I_u_t](i, j) - 1.0)) *
                                weight;  // mass weighted v_inf

      // values for histograms
      histvals[I_hist_entr]   = other[I_entropy](i, j);
      histvals[I_hist_logrho] = std::log10(prim[IDN](i, j));
      histvals[I_hist_temp]   = T(i, j);
      histvals[I_hist_ye]     = Y[0](i, j);
      histvals[I_hist_vel]    = other[I_v_mag](i, j);
      histvals[I_hist_ber]    = other[I_bernoulli](i, j);
      histvals[I_hist_velinf] = sqrt(2.0 * (-other[I_u_t](i, j) - 1.0));
      histvals[I_hist_theta]  = grid_.th_grid(i);

      // unboundedeness criteria
      unbound(I_unbound_bernoulli) =
        (other[I_bernoulli](i, j) > 0.0) ? true : false;
      unbound(I_unbound_bernoulli_outflow) =
        ((other[I_bernoulli](i, j) > 0.0) && (vrad > 0.0)) ? true : false;
      unbound(I_unbound_geodesic) = (other[I_u_t](i, j) < -1.0) ? true : false;
      unbound(I_unbound_geodesic_outflow) =
        ((other[I_u_t](i, j) < -1.0) && (vrad > 0.0)) ? true : false;

      for (int n = 0; n < n_unbound; ++n)
      {  // unboundedness criteria
        if (!unbound(n))
          continue;
        //          ++count(n);
        //       	  ++countaz(n,i);
        for (int m = 0; m < n_int; ++m)
        {  // integrated variables
          integrals_unbound(n, m) += integrals[m];
          az_integrals_unbound(n, m, i) += integrals[m];
        }
        for (int m = 0; m < n_hist; ++m)
        {  // loop over histogram variables
          if (histvals[m] < hist_grid[m](0))
          {  // if value is below lower limit
             // of histogram add to lowest bin
            hist[m](n, 0) += weight;
            continue;
          }
          for (int l = 1; l < n_bins[m]; l++)
          {  // loop over histogram bins for variable m
            if (histvals[m] < hist_grid[m](l))
            {
              hist[m](n, l - 1) += weight;
              break;
            }
            if (histvals[m] > hist_grid[m](n_bins[m] - 1))
            {  // if value is above largest bin add to largest bin
              hist[m](n, n_bins[m] - 1) += weight;
            }
          }  // bin loop l
        }  // variable loop m
      }  // unboundedness loop n

    }  // end phi loop
  }  // end theta loop
}

//-----------------------------------------------------------------------------
// \!fn void Ejecta::Write(const Real time)
// \brief Output ejecta quantities
void Ejecta::Write(const Real time)
{
  if (Globals::my_rank == 0)
  {
#ifdef HDF5OUTPUT
    Write_hdf5(time);
#endif
    Write_scalars(time);
  }
  file_number++;
  pin->OverwriteParameter("ejecta", "file_number", file_number);
}

#ifdef HDF5OUTPUT
void Ejecta::Write_hdf5(const Real time)
{
  const int iter = file_number;
  std::string filename;
  hdf5_get_next_filename(filename);
  static const bool use_existing = false;
  hid_t id_file                  = hdf5_touch_file(filename, use_existing);

  // scalars [grid] ---------------------------------------------------------
  hdf5_write_scalar(id_file, "time", time);
  hdf5_write_scalar(id_file, "radius", radius);

  // 1d arrays [grid] -------------------------------------------------------
  hdf5_write_arr_nd(id_file, "theta", grid_.th_grid);
  hdf5_write_arr_nd(id_file, "phi", grid_.ph_grid);

  // 2d arrays [matter] -----------------------------------------------------
  for (int n = 0; n < NHYDRO; ++n)
  {
    std::string full_path = "/prim/" + prim_names[n];
    hdf5_write_arr_nd(id_file, full_path, prim[n]);
  }

  for (int n = 0; n < NHYDRO; ++n)
  {
    std::string full_path = "/cons/" + cons_names[n];
    hdf5_write_arr_nd(id_file, full_path, cons[n]);
  }

#if FLUID_ENABLED
  {
    std::string full_path = "/prim/T";
    hdf5_write_arr_nd(id_file, full_path, T);

    for (int n = 0; n < NSCALARS; ++n)
    {
      std::string full_path = "/prim/Y_" + std::to_string(n);
      hdf5_write_arr_nd(id_file, full_path, Y[n]);
    }
  }
#endif  // FLUID_ENABLED

#if MAGNETIC_FIELDS_ENABLED
  for (int n = 0; n < NFIELD; ++n)
  {
    std::string full_path = "/Bcc/B_" + std::to_string(n + 1);
    hdf5_write_arr_nd(id_file, full_path, Bcc[n]);
  }
#endif  // MAGNETIC_FIELDS_ENABLED

  // 2d arrays [geometry] ---------------------------------------------------

  for (int n = 0; n < Z4c::N_ADM; ++n)
  {
    std::string var_name  = Z4c::ADM_names[n];
    std::string full_path = "/adm/" + var_name;
    hdf5_write_arr_nd(id_file, full_path, adm[n]);
  }

  for (int n = 0; n < Z4c::N_Z4c; ++n)
  {
    std::string var_name  = Z4c::Z4c_names[n];
    std::string full_path = "/z4c/" + var_name;
    hdf5_write_arr_nd(id_file, full_path, z4c[n]);
  }

  // 2d arrays [other] ------------------------------------------------------
  for (int n = 0; n < NOTHER; ++n)
  {
    std::string full_path = "/other/" + other_names[n];
    hdf5_write_arr_nd(id_file, full_path, other[n]);
  }

  // scalars [misc] ---------------------------------------------------------
  hdf5_write_scalar(id_file, "mass", mass_contained);
  hdf5_write_scalar(id_file, "Mdot_total", Mdot_total);

  // Finally close
  hdf5_close_file(id_file);
}
#endif  // HDF5OUTPUT

void Ejecta::Write_scalars(const Real time)
{
  const int iter = file_number;

  // Summary file
  for (int n = 0; n < n_unbound; ++n)
  {
    fprintf(pofile_unbound[n], "%d %g ", iter, time);
    fprintf(pofile_unbound[n], "%.15e ", integrals_unbound(n, I_int_mass));
    for (int m = I_int_entr; m < n_int; m++)
    {
      fprintf(
        pofile_unbound[n],
        "%.15e ",
        integrals_unbound(n, m) /
          integrals_unbound(n, I_int_mass));  // normalise by average mass
    }
    fprintf(pofile_unbound[n], "\n");
    fflush(pofile_unbound[n]);
  }
  for (int n = 0; n < n_unbound; ++n)
  {
    fprintf(pofile_az_unbound[n], "### Time = %g \n", time);
    for (int i = 0; i < grid_.ntheta; ++i)
    {
      fprintf(pofile_az_unbound[n], "%.15e ", grid_.th_grid(i));
      fprintf(pofile_az_unbound[n],
              "%.15e ",
              az_integrals_unbound(n, I_int_mass, i));
      for (int m = I_int_entr; m < n_int; m++)
      {
        fprintf(
          pofile_az_unbound[n],
          "%.15e ",
          az_integrals_unbound(n, m, i) /
            az_integrals_unbound(n, I_int_mass, i));  // normalise by mass
      }
      fprintf(pofile_az_unbound[n], "\n");
    }
    fprintf(pofile_az_unbound[n], "\n");
    fflush(pofile_az_unbound[n]);
  }
  for (int n = 0; n < n_unbound; ++n)
  {
    for (int m = 0; m < n_hist; ++m)
    {
      fprintf(pofile_hist_unbound[n][m], "### Time = %g \n", time);
      for (int l = 0; l < n_bins[m]; ++l)
      {
        fprintf(pofile_hist_unbound[n][m],
                "%.15e %.15e \n",
                hist_grid[m](l) + delta_hist[m] / 2.0,
                hist[m](n, l));  // bin value is centre of bin
      }
      fprintf(pofile_hist_unbound[n][m], "\n");
      fflush(pofile_hist_unbound[n][m]);
    }
  }
}

Real Ejecta::MassLossRate(Real const fx,
                          Real const fy,
                          Real const fz,
                          Real const sinth,
                          Real const costh,
                          Real const sinph,
                          Real const cosph)
{
  Real r_x = cosph * sinth;
  Real r_y = sinph * sinth;
  Real r_z = costh;
  return (fx * r_x + fy * r_y + fz * r_z) * SQR(radius) * sinth *
         grid_.dtheta() * grid_.dphi();
}

Real Ejecta::MassLossRate2(Real const D,
                           Real const ux,
                           Real const uy,
                           Real const uz,
                           Real const W,
                           Real const alpha,
                           Real const betax,
                           Real const betay,
                           Real const betaz,
                           Real const sinth,
                           Real const costh,
                           Real const sinph,
                           Real const cosph)
{
  Real r_x = cosph * sinth;
  Real r_y = sinph * sinth;
  Real r_z = costh;
  Real v_x = alpha * ux / W - betax;
  Real v_y = alpha * uy / W - betay;
  Real v_z = alpha * uz / W - betaz;
  return D * (v_x * r_x + v_y * r_y + v_z * r_z) * SQR(radius) * sinth *
         grid_.dtheta() * grid_.dphi();
}
