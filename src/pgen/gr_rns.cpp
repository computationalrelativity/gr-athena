//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file gr_rns.cpp
//  \brief Initial conditions for rotating neutron star from Stergioulas' RNS
//  code
//         Requires the library:
//         https://bitbucket.org/bernuzzi/rnsc/src/master/

#include <cassert>  // assert
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../field/seed_magnetic_field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../z4c/z4c.hpp"
#if M1_ENABLED
#include "../m1/m1.hpp"
#include "../m1/m1_set_equilibrium.hpp"
#endif  // M1_ENABLED

#include "RNS.h"  // https://bitbucket.org/bernuzzi/rnsc/src/master/

using namespace std;

#if not FLUID_ENABLED
#error "This problem generator requires fluid (-f)"
#endif

namespace
{
static ini_data* rns_data;
Primitive::ColdEOS<Primitive::COLDEOS_POLICY>* ceos = NULL;
Real mb_rnsc = 931.191715903434;  // RNSC uses this mass factor

void SeedMagneticFields(MeshBlock* pmb, ParameterInput* pin);
}  // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can
//  also be used to initialize variables which are global to (and therefore can
//  be passed to) other functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput* pin)
{
  EnrollUserStandardHydro(pin);
  EnrollUserStandardField(pin);
  EnrollUserStandardZ4c(pin);
  EnrollUserStandardM1(pin);

  /*
  // New outputs can now be specified with the form:
  EnrollUserHistoryOutput(
    [&](MeshBlock *pmb, int iout){ return 1.0; },
    "some_name",
    UserHistoryOperation::min
  );
  */

  if (!resume_flag)
  {
    string set_name = "problem";
    RNS_params_set_default();
    string inputfile =
      pin->GetOrAddString("problem", "filename", "tovgamma2.par");
    RNS_params_set_inputfile((char*)inputfile.c_str());
    rns_data = RNS_make_initial_data();
    ceos     = new Primitive::ColdEOS<Primitive::COLDEOS_POLICY>;
    InitColdEOS(ceos, pin);
  }

  if (adaptive == true)
  {
    // Default AMR driven by ExtremaTracker (and AHFs where available).
    // To use a custom criterion instead, define a local
    //   int MyRefinementCondition(MeshBlock*);
    // and call EnrollUserRefinementCondition(MyRefinementCondition);
    // which will override this default.
    EnrollUserRefinementCondition(Mesh::StandardRefinementCondition);
  }

  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput* pin)
{
  if (!resume_flag)
    RNS_finalise(rns_data);
  delete ceos;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn
// \brief Setup User work

void MeshBlock::InitUserMeshBlockData(ParameterInput* pin)
{
  // Allocate output arrays for fluxes
#if M1_ENABLED
  AllocateUserOutputVariables(4);
#endif  // M1_ENABLED
  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput* pin)
{
#if M1_ENABLED
  AA& fl = pm1->ev_strat.masks.flux_limiter;

  int il = pm1->mbi.nn1;
  int jl = pm1->mbi.nn2;
  int kl = pm1->mbi.nn3;

  if (pm1->opt.flux_limiter_use_mask)
    M1_ILOOP3(k, j, i)
    {
      user_out_var(0, k, j, i) = fl(0, k, j, i);
      user_out_var(1, k, j, i) = fl(1, k, j, i);
      user_out_var(2, k, j, i) = fl(2, k, j, i);
    }

  if (pm1->opt.flux_lo_fallback)
    M1_ILOOP3(k, j, i)
    {
      user_out_var(3, k, j, i) = pm1->ev_strat.masks.pp(k, j, i);
    }

  /*
  AthenaArray<Real> &x1flux = phydro->flux[X1DIR];
  AthenaArray<Real> &x2flux = phydro->flux[X2DIR];
  AthenaArray<Real> &x3flux = phydro->flux[X3DIR];

  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1)
  {
    jl -= NGHOST;
    ju += NGHOST;
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1)
  {
    kl -= NGHOST;
    ku += NGHOST;
  }

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  for (int i=il; i<=iu; ++i)
  {
    user_out_var(0,k,j,i) = x1flux(0,k,j,i);
    user_out_var(1,k,j,i) = x1flux(1,k,j,i);
    user_out_var(2,k,j,i) = x1flux(2,k,j,i);
    user_out_var(3,k,j,i) = x1flux(3,k,j,i);
    user_out_var(4,k,j,i) = x1flux(4,k,j,i);
    user_out_var(5,k,j,i) = x2flux(0,k,j,i);
    user_out_var(6,k,j,i) = x2flux(1,k,j,i);
    user_out_var(7,k,j,i) = x2flux(2,k,j,i);
    user_out_var(8,k,j,i) = x2flux(3,k,j,i);
    user_out_var(9,k,j,i) = x2flux(4,k,j,i);
    user_out_var(10,k,j,i) = x3flux(0,k,j,i);
    user_out_var(11,k,j,i) = x3flux(1,k,j,i);
    user_out_var(12,k,j,i) = x3flux(2,k,j,i);
    user_out_var(13,k,j,i) = x3flux(3,k,j,i);
    user_out_var(14,k,j,i) = x3flux(4,k,j,i);
  }
  */
#endif  // M1_ENABLED
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput* pin)
{
#ifdef Z4C_ASSERT_FINITE
  // as a sanity check (these should be over-written)
  pz4c->adm.psi4.Fill(NAN);
  pz4c->adm.g_dd.Fill(NAN);
  pz4c->adm.K_dd.Fill(NAN);

  pz4c->z4c.chi.Fill(NAN);
  pz4c->z4c.Khat.Fill(NAN);
  pz4c->z4c.Theta.Fill(NAN);
  pz4c->z4c.alpha.Fill(NAN);
  pz4c->z4c.Gam_u.Fill(NAN);
  pz4c->z4c.beta_u.Fill(NAN);
  pz4c->z4c.g_dd.Fill(NAN);
  pz4c->z4c.A_dd.Fill(NAN);

  /*
  pz4c->con.C.Fill(NAN);
  pz4c->con.H.Fill(NAN);
  pz4c->con.M.Fill(NAN);
  pz4c->con.Z.Fill(NAN);
  pz4c->con.M_d.Fill(NAN);
  */
#endif

  bool verbose = pin->GetOrAddBoolean("problem", "verbose", 0);

  Real pres_pert = pin->GetOrAddReal("problem", "pres_pert", 0);
  Real v_pert    = pin->GetOrAddReal("problem", "v_pert", 0);

  // Scalar arrays - safe even when NSCALARS == 0 (pscalars is nullptr)
  AthenaArray<Real> empty;
#if NSCALARS > 0
  AthenaArray<Real>& r_scalar = pscalars->r;
  AthenaArray<Real>& s_scalar = pscalars->s;
#else
  AthenaArray<Real>& r_scalar = empty;
  AthenaArray<Real>& s_scalar = empty;
#endif

  //  MeshBlock * pmb = pmy_block;
  //  Coordinates * pco = pcoord;
  //  Z4c * pz4c = pmb->pz4c;

  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pz4c->mbi);

  //---------------------------------------------------------------------------
  // Interpolate ADM metric

  if (verbose)
    std::cout << "Interpolating ADM metric on current MeshBlock." << std::endl;

  int imin[3] = { 0, 0, 0 };
  int n[3]    = { mbi->nn1, mbi->nn2, mbi->nn3 };
  int sz      = n[0] * n[1] * n[2];

  // temporary variables
  // this could be done instead by accessing and casting the Athena vars but
  // then it is coupled to implementation details etc.
  Real *gxx = new Real[sz], *gyy = new Real[sz], *gzz = new Real[sz];
  Real *gxy = new Real[sz], *gxz = new Real[sz], *gyz = new Real[sz];

  Real *Kxx = new Real[sz], *Kyy = new Real[sz], *Kzz = new Real[sz];
  Real *Kxy = new Real[sz], *Kxz = new Real[sz], *Kyz = new Real[sz];

  Real* alp   = new Real[sz];
  Real *betax = new Real[sz], *betay = new Real[sz], *betaz = new Real[sz];

  Real* x = new Real[n[0]];
  Real* y = new Real[n[1]];
  Real* z = new Real[n[2]];

  // Populate coordinates
  for (int i = 0; i < n[0]; ++i)
  {
    x[i] = mbi->x1(i);
  }
  for (int i = 0; i < n[1]; ++i)
  {
    y[i] = mbi->x2(i);
  }
  for (int i = 0; i < n[2]; ++i)
  {
    z[i] = mbi->x3(i);
  }

  // Interpolate geometry
  RNS_Cartesian_interpolation(
    rns_data,  // struct containing the previously calculated solution
    imin,      // min, max idxs of Cartesian Grid in the three directions
    n,         // TODO WC: check this!!!1
    n,         // total number of indices in each direction
    x,         // x,         // Cartesian coordinates
    y,         // y,
    z,         // z,
    alp,       // alp,       // lapse
    betax,     // betax,   // shift vector
    betay,     // betay,
    betaz,     // betaz,
    gxx,       // gxx,       // metric components
    gxy,       // gxy,
    gxz,       // gxz,
    gyy,       // gyy,
    gyz,       // gyz,
    gzz,       // gzz,
    Kxx,       // kxx,       // extrinsic curvature components
    Kxy,       // kxy,
    Kxz,       // kxz,
    Kyy,       // kyy,
    Kyz,       // kyz,
    Kzz,       // kzz
    NULL,      // rho
    NULL,      // epsl
    NULL,      // vx
    NULL,      // vy
    NULL,      // vz
    NULL,      // ux
    NULL,      // uy
    NULL,      // uz
    NULL       // pres
  );

  for (int k = 0; k < mbi->nn3; ++k)
    for (int j = 0; j < mbi->nn2; ++j)
      for (int i = 0; i < mbi->nn1; ++i)
      {
        int flat_ix = i + n[0] * (j + n[1] * k);

        pz4c->storage.adm(Z4c::I_ADM_gxx, k, j, i) = gxx[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_gxy, k, j, i) = gxy[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_gxz, k, j, i) = gxz[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_gyy, k, j, i) = gyy[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_gyz, k, j, i) = gyz[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_gzz, k, j, i) = gzz[flat_ix];

        pz4c->storage.adm(Z4c::I_ADM_Kxx, k, j, i) = Kxx[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_Kxy, k, j, i) = Kxy[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_Kxz, k, j, i) = Kxz[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_Kyy, k, j, i) = Kyy[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_Kyz, k, j, i) = Kyz[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_Kzz, k, j, i) = Kzz[flat_ix];

        pz4c->storage.adm(Z4c::I_ADM_alpha, k, j, i) = alp[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_betax, k, j, i) = betax[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_betay, k, j, i) = betay[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_betaz, k, j, i) = betaz[flat_ix];

        // TODO what to do with psi4 buffer?
        // pz4c->storage.adm(Z4c::I_ADM_psi4,k,j,i) = 0.0;
      }

  delete[] gxx;
  delete[] gxy;
  delete[] gxz;
  delete[] gyy;
  delete[] gyz;
  delete[] gzz;

  delete[] Kxx;
  delete[] Kxy;
  delete[] Kxz;
  delete[] Kyy;
  delete[] Kyz;
  delete[] Kzz;

  delete[] alp;
  delete[] betax;
  delete[] betay;
  delete[] betaz;

  delete[] x;
  delete[] y;
  delete[] z;

  //---------------------------------------------------------------------------
  // ADM-to-Z4c
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u1);

  // TODO Needed?
  //  pcoord->UpdateMetric();
  //  if(pmy_mesh->multilevel){
  //    pmr->pcoarsec->UpdateMetric();
  //  }

  //---------------------------------------------------------------------------
  // Interpolate primitives

  if (verbose)
    std::cout << "Interpolating primitives on current MeshBlock." << std::endl;

  n[0] = ncells1;
  n[1] = ncells2;
  n[2] = ncells3;
  sz   = n[0] * n[1] * n[2];

  Real *rho = new Real[sz], *pres = new Real[sz], *ye = new Real[sz];
  Real *ux = new Real[sz], *uy = new Real[sz], *uz = new Real[sz];

  x = new Real[n[0]];
  y = new Real[n[1]];
  z = new Real[n[2]];

  // Populate coordinates
  for (int i = 0; i < n[0]; ++i)
  {
    x[i] = pcoord->x1v(i);
  }
  for (int i = 0; i < n[1]; ++i)
  {
    y[i] = pcoord->x2v(i);
  }
  for (int i = 0; i < n[2]; ++i)
  {
    z[i] = pcoord->x3v(i);
  }

  // Interpolate primitives
  RNS_Cartesian_interpolation(
    rns_data,  // struct containing the previously calculated solution
    imin,      // min, max idxs of Cartesian Grid in the three directions
    n,         //
    n,         // total number of indices in each direction
    x,         // x,         // Cartesian coordinates
    y,         // y,
    z,         // z,
    NULL,      // alp,       // lapse
    NULL,      // betax,   // shift vector
    NULL,      // betay,
    NULL,      // betaz,
    NULL,      // gxx,       // metric components
    NULL,      // gxy,
    NULL,      // gxz,
    NULL,      // gyy,
    NULL,      // gyz,
    NULL,      // gzz,
    NULL,      // kxx,       // extrinsic curvature components
    NULL,      // kxy,
    NULL,      // kxz,
    NULL,      // kyy,
    NULL,      // kyz,
    NULL,      // kzz
    rho,       // rho
    NULL,      // epsl - maybe for new EOS we want to take this
    NULL,      // vx
    NULL,      // vy
    NULL,      // vz
    ux,        // vx
    uy,        // vy
    uz,        // vz
    pres       // pres
  );

  Real pres_diff = 0.0;

  Real rho_min = pin->GetReal("hydro", "dfloor");

  for (int k = 0; k < ncells3; ++k)
    for (int j = 0; j < ncells2; ++j)
      for (int i = 0; i < ncells1; ++i)
      {
        int flat_ix = i + n[0] * (j + n[1] * k);
        Real r      = std::sqrt(x[i] * x[i] + y[j] * y[j] + z[k] * z[k]);

#if defined(USE_COMPOSE_EOS) || defined(USE_HYBRID_EOS)
        rho[flat_ix] *= ceos->mb / mb_rnsc;  // adjust for rns baryon mass
#endif
        if (rho[flat_ix] > rho_min)
        {
          Real pres_eos = ceos->GetPressure(rho[flat_ix]);
          pres_diff     = max(abs(pres[flat_ix] / pres_eos - 1), pres_diff);
          pres[flat_ix] = pres_eos;
        }

#if NSCALARS > 0
        for (int l = 0; l < NSCALARS; ++l)
          pscalars->r(l, k, j, i) = ceos->GetY(rho[flat_ix], l);
#endif

        phydro->w(IDN, k, j, i) = rho[flat_ix];
        phydro->w(IPR, k, j, i) = pres[flat_ix];
        phydro->w(IVX, k, j, i) = ux[flat_ix];
        phydro->w(IVY, k, j, i) = uy[flat_ix];
        phydro->w(IVZ, k, j, i) = uz[flat_ix];

        // Add perturbations
        if (pres_pert and r < rns_data->r_e)
        {
          phydro->w(IPR, k, j, i) -= pres_pert * pres[flat_ix];
        }
        if (v_pert and r < rns_data->r_e)
        {
          phydro->w(IVX, k, j, i) -=
            v_pert * std::cos(M_PI * r / (2.0 * rns_data->r_e)) * x[i] / r;
          phydro->w(IVY, k, j, i) -=
            v_pert * std::cos(M_PI * r / (2.0 * rns_data->r_e)) * y[j] / r;
          phydro->w(IVZ, k, j, i) -=
            v_pert * std::cos(M_PI * r / (2.0 * rns_data->r_e)) * z[k] / r;
        }
      }

  if (pres_diff > 1e-3)
    std::cout
      << "WARNING: Interpolated pressure does not match eos. abs. rel. diff = "
      << pres_diff << std::endl;

  delete[] rho;
  delete[] pres;
  delete[] ye;
  delete[] ux;
  delete[] uy;
  delete[] uz;

  delete[] x;
  delete[] y;
  delete[] z;

  //---------------------------------------------------------------------------
  // Initialise conserved variables

  if (verbose)
    std::cout << "Initializing conservatives on current MeshBlock."
              << std::endl;

  // Prepare index bounds
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1)
  {
    jl -= NGHOST;
    ju += NGHOST;
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1)
  {
    kl -= NGHOST;
    ku += NGHOST;
  }

  peos->PrimitiveToConserved(phydro->w,
                             r_scalar,
                             pfield->bcc,
                             phydro->u,
                             s_scalar,
                             pcoord,
                             il,
                             iu,
                             jl,
                             ju,
                             kl,
                             ku);

  //---------------------------------------------------------------------------
  // Initialise matter & ADM constraints
  // TODO(WC) (don't strictly need this here, will be caught in task list
  // before used

  // if(verbose)
  //   std::cout << "Initializing matter and constraints on current MeshBlock."
  //   << std::endl;

  // --------------------------------------------------------------------------
  // The following is now done else-where and is redundant here
  /*
  // Set up ADM matter variables
  pz4c->GetMatter(pz4c->storage.mat,
                  pz4c->storage.adm,
                  phydro->w,
                  pscalars->r,
                  pfield->bcc);

  pz4c->ADMConstraints(pz4c->storage.con,
                       pz4c->storage.adm,
                       pz4c->storage.mat,
                       pz4c->storage.u);
  */
  // --------------------------------------------------------------------------

#ifdef Z4C_ASSERT_FINITE
  pz4c->assert_is_finite_adm();
  pz4c->assert_is_finite_con();
  pz4c->assert_is_finite_mat();
  pz4c->assert_is_finite_z4c();
#endif

#if MAGNETIC_FIELDS_ENABLED
  // Regularize prims (needed for some boosted data)
  for (int k = 0; k < ncells3; k++)
    for (int j = 0; j < ncells2; j++)
      for (int i = 0; i < ncells1; i++)
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

  return;
}

namespace
{

void SeedMagneticFields(MeshBlock* pmb, ParameterInput* pin)
{
  // B field ------------------------------------------------------------------
  // The vector potential is
  //   A_x = -y * b_amp * max(p - pcut, 0) * (1 - rho/rhomax)^magindex
  //   A_y =  x * b_amp * max(p - pcut, 0) * (1 - rho/rhomax)^magindex
  //   A_z =  0
  //
  // Face-centred B = curl(A) is computed via the discrete Stokes theorem on
  // cell edges, which gives div(B) = 0 to machine precision.

  // Get central density from RNS data
  int imin[3] = { 0, 0, 0 };
  int imax[3] = { 1, 1, 1 };
  double x[1] = { 0.0 };
  Real rhomax;
  Real prnsmax;

  RNS_Cartesian_interpolation(
    rns_data,  // struct containing the previously calculated solution
    imin,      // min, max idxs of Cartesian Grid in the three directions
    imax,      // TODO WC: check this!!!1
    imax,      // total number of indices in each direction
    x,         // x,         // Cartesian coordinates
    x,         // y,
    x,         // z,
    NULL,      // alp,       // lapse
    NULL,      // betax,   // shift vector
    NULL,      // betay,
    NULL,      // betaz,
    NULL,      // gxx,       // metric components
    NULL,      // gxy,
    NULL,      // gxz,
    NULL,      // gyy,
    NULL,      // gyz,
    NULL,      // gzz,
    NULL,      // kxx,       // extrinsic curvature components
    NULL,      // kxy,
    NULL,      // kxz,
    NULL,      // kyy,
    NULL,      // kyz,
    NULL,      // kzz
    &rhomax,   // rho
    NULL,      // epsl
    NULL,      // vx
    NULL,      // vy
    NULL,      // vz
    NULL,      // ux
    NULL,      // uy
    NULL,      // uz
    &prnsmax   // pres
  );
  //
  Real pgasmax = ceos->GetPressure(rhomax);
  printf("rhomax=%.5e prnsmax=%.5e pmax=%.5e\n", rhomax, prnsmax, pgasmax);

  Real pcut    = pin->GetReal("problem", "pcut") * pgasmax;
  int magindex = pin->GetInteger("problem", "magindex");

  Real b_amp =
    pin->GetReal("problem", "b_amp") * 0.5 / (pgasmax - pcut) / 8.351416e19;

  SeedFaceBFromEdgePotential(pmb,
                             [=](Real x,
                                 Real y,
                                 Real /*z*/,
                                 Real p,
                                 Real rho,
                                 Real& Ax,
                                 Real& Ay,
                                 Real& Az)
                             {
                               Real amp =
                                 b_amp * std::max(p - pcut, 0.0) *
                                 std::pow(1.0 - rho / rhomax, magindex);
                               Ax = -y * amp;
                               Ay = x * amp;
                               Az = 0.0;
                             });
}
}  // namespace

/*
Real L1rhodiff(MeshBlock *pmb, int iout) {
  Real L1rho = 0.0;
  Real vol,dx,dy,dz;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke
= pmb->ke; for (int k=ks; k<=ke; k++) { for (int j=js; j<=je; j++) { for (int
i=is; i<=ie; i++) { dx = pmb->pcoord->dx1v(i); dy = pmb->pcoord->dx2v(j); dz =
pmb->pcoord->dx3v(k); vol = dx*dy*dz; L1rho +=
std::abs(pmb->phydro->w(IDN,k,j,i) - pmb->phydro->w_init(IDN,k,j,i))*vol;
      }
    }
  }
  return L1rho;
}
*/
