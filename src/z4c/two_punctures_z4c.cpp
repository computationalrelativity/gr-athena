//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file puncture_z4c.cpp
//  \brief implementation of functions in the Z4c class for initializing puntures evolution

// C++ standard headers
#include <cmath> // pow

// Athena++ headers
#include "z4c.hpp"
#include "z4c_macro.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"

// twopuncturesc: Stand-alone library ripped from Cactus
#include "TwoPunctures.h"

// Logic for enforcement of symmetry in ID ------------------------------------
static const bool id_impose_symmetries = false;

static const bool id_scalars_symmetric_x = true;
static const bool id_scalars_symmetric_y = true;
static const bool id_scalars_symmetric_z = true;

static const int id_K_xx_symmetry_x = -1;
static const int id_K_xy_symmetry_x =  1;
static const int id_K_xz_symmetry_x =  1;
static const int id_K_yy_symmetry_x = -1;
static const int id_K_yz_symmetry_x = -1;
static const int id_K_zz_symmetry_x = -1;

static const int id_K_xx_symmetry_y = -1;
static const int id_K_xy_symmetry_y =  1;
static const int id_K_xz_symmetry_y = -1;
static const int id_K_yy_symmetry_y = -1;
static const int id_K_yz_symmetry_y =  1;
static const int id_K_zz_symmetry_y = -1;

static const int id_K_xx_symmetry_z =  1;
static const int id_K_xy_symmetry_z =  1;
static const int id_K_xz_symmetry_z = -1;
static const int id_K_yy_symmetry_z =  1;
static const int id_K_yz_symmetry_z = -1;
static const int id_K_zz_symmetry_z =  1;

static const int id_g_xx_symmetry_x =  1;
static const int id_g_xy_symmetry_x =  1;
static const int id_g_xz_symmetry_x =  1;
static const int id_g_yy_symmetry_x =  1;
static const int id_g_yz_symmetry_x =  1;
static const int id_g_zz_symmetry_x =  1;

static const int id_g_xx_symmetry_y =  1;
static const int id_g_xy_symmetry_y =  1;
static const int id_g_xz_symmetry_y =  1;
static const int id_g_yy_symmetry_y =  1;
static const int id_g_yz_symmetry_y =  1;
static const int id_g_zz_symmetry_y =  1;

static const int id_g_xx_symmetry_z =  1;
static const int id_g_xy_symmetry_z =  1;
static const int id_g_xz_symmetry_z =  1;
static const int id_g_yy_symmetry_z =  1;
static const int id_g_yz_symmetry_z =  1;
static const int id_g_zz_symmetry_z =  1;


// template<typename T>
// static bool simeq(T val, T val_B=0, T tol=std::numeric_limits<T>::epsilon())
// {
//   return std::abs(val - val_B) < tol;
// }

// template<typename T>
// static T sign(T x, T tol=std::numeric_limits<T>::epsilon())
// {
//   // set:
//   // 1 for x > 0
//   // 0 for x = 0  (within tol)
//   //-1 for x < 0

//   return ((x > 0) ? 1 : -1) * (
//     (simeq(x, static_cast<T>(0), tol)) ? 0 : 1
//   );
// }

template<typename T>
int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

template<typename T>
static void impose_ordinate_sym_pos(T * gr, const ssize_t sz)
{
  for(int ix=0; ix<sz; ++ix)
    gr[ix] = sgn(gr[ix]) * gr[ix];
}

template<typename T>
static void impose_bitant_phase_x_dir(
  T * gr_x,
  T * Fxy,
  T * Fxz,
  int * n,
  int * strides)
{
  for(int k=0; k<n[2]; ++k)
  for(int j=0; j<n[1]; ++j)
  for(int i=0; i<n[0]; ++i)
  {
    // set zero on axis and negate for (x < 0)
    const T ph = sgn(gr_x[i]);

    const int ix = i * strides[0] + j * strides[1] + k * strides[2];
    Fxy[ix] = ph * Fxy[ix];
    Fxz[ix] = ph * Fxz[ix];
  }

}

template<typename T>
static void impose_bitant_phase_y_dir(
  T * gr_y,
  T * Fxy,
  T * Fyz,
  int * n,
  int * strides)
{
  for(int k=0; k<n[2]; ++k)
  for(int j=0; j<n[1]; ++j)
  {
    // set zero on axis and negate for (y < 0)
    const T ph = sgn(gr_y[j]);

    for(int i=0; i<n[0]; ++i)
    {

      const int ix = i * strides[0] + j * strides[1] + k * strides[2];
      Fxy[ix] = ph * Fxy[ix];
      Fyz[ix] = ph * Fyz[ix];
    }
  }

}

template<typename T>
static void impose_bitant_phase_z_dir(
  T * gr_z,
  T * Fxz,
  T * Fyz,
  int * n,
  int * strides)
{
  for(int k=0; k<n[2]; ++k)
  {
    // set zero on axis and negate for (z < 0)
    const T ph = sgn(gr_z[k]);

    for(int j=0; j<n[1]; ++j)
    for(int i=0; i<n[0]; ++i)
    {
      const int ix = i * strides[0] + j * strides[1] + k * strides[2];
      Fxz[ix] = ph * Fxz[ix];
      Fyz[ix] = ph * Fyz[ix];
    }
  }

}

template<typename T>
static void impose_sym2_phase(
  const int ph_sym_x,
  const int ph_sym_y,
  const int ph_sym_z,
  T * gr_x,
  T * gr_y,
  T * gr_z,
  T * F_ij,
  int * n,
  int * strides)
{
  for(int k=0; k<n[2]; ++k)
  for(int j=0; j<n[1]; ++j)
  for(int i=0; i<n[0]; ++i)
  {
    // prepare phase based on passed symmetry
    int ph = (
      ((ph_sym_z == -1) ? sgn(gr_z[k]) : 1) *
      ((ph_sym_y == -1) ? sgn(gr_y[j]) : 1) *
      ((ph_sym_x == -1) ? sgn(gr_x[i]) : 1)
    );

    const int ix = i * strides[0] + j * strides[1] + k * strides[2];
    F_ij[ix] = ph * F_ij[ix];
  }

}

static void copy_ordinates(
  Real * x, Real * y, Real * z,
  AthenaArray<Real> & x_, AthenaArray<Real> & y_, AthenaArray<Real> & z_,
  int * n)
{
  for(int ix_I = 0; ix_I < n[0]; ix_I++)
    x[ix_I] = x_(ix_I);

  for(int ix_J = 0; ix_J < n[1]; ix_J++)
    y[ix_J] = y_(ix_J);

  for(int ix_K = 0; ix_K < n[2]; ix_K++)
    z[ix_K] = z_(ix_K);
}


//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMTwoPunctures(AthenaArray<Real> & u)
// \brief Initialize ADM vars to two punctures

void Z4c::ADMTwoPunctures(ParameterInput *pin, AthenaArray<Real> & u_adm, ini_data *data)
{
  bool verbose = pin->GetOrAddBoolean("problem", "verbose", 0);
  //if(verbose)
  //  Z4c::DebugInfoVars();

  ADM_vars adm;
  SetADMAliases(u_adm, adm);

  //MeshBlock * pmb = pmy_block;
  //Coordinates * pco = pmb->pcoord;

  // Flat spacetime
  ADMMinkowski(u_adm);

  //--

  // construct initial data set and interpolate ADM vars. to current MeshBlock
  if(verbose)
  {
    std::cout << "Generating two puncture data." << std::endl;
    std::cout << "Interpolating current MeshBlock." << std::endl;
  }

  int imin[3] = {0, 0, 0};

  // dimensions of block in each direction
  //int n[3] = {(*pmb).block_size.nx1 + 2 * GSIZEI,
  //            (*pmb).block_size.nx2 + 2 * GSIZEJ,
  //            (*pmb).block_size.nx3 + 2 * GSIZEK};
  int n[3] = {mbi.nn1, mbi.nn2, mbi.nn3};

  int sz = n[0] * n[1] * n[2];
  // this could be done instead by accessing and casting the Athena vars but
  // then it is coupled to implementation details etc.
  Real *gxx = new Real[sz], *gyy = new Real[sz], *gzz = new Real[sz];
  Real *gxy = new Real[sz], *gxz = new Real[sz], *gyz = new Real[sz];

  Real *Kxx = new Real[sz], *Kyy = new Real[sz], *Kzz = new Real[sz];
  Real *Kxy = new Real[sz], *Kxz = new Real[sz], *Kyz = new Real[sz];

  Real *psi = new Real[sz];
  Real *alp = new Real[sz];

  Real *x = new Real[n[0]];
  Real *y = new Real[n[1]];
  Real *z = new Real[n[2]];

  // need to populate coordinates
  copy_ordinates(x, y, z, mbi.x1, mbi.x2, mbi.x3, n);

  // this is called many times with exactly the same sgnature
  auto do_interp = [&]{
    TwoPunctures_Cartesian_interpolation
      (data, // struct containing the previously calculated solution
      imin, // min, max idxs of Cartesian Grid in the three directions
      n,    // <-imax, but this collapses
      n,    // total number of indices in each direction
      x,    // x,         // Cartesian coordinates
      y,    // y,
      z,    // z,
      alp,  // alp,       // lapse
      psi,  // psi,       // conformal factor and derivatives
      NULL, // psix,
      NULL, // psiy,
      NULL, // psiz,
      NULL, // psixx,
      NULL, // psixy,
      NULL, // psixz,
      NULL, // psiyy,
      NULL, // psiyz,
      NULL, // psizz,
      gxx,  // gxx,       // metric components
      gxy,  // gxy,
      gxz,  // gxz,
      gyy,  // gyy,
      gyz,  // gyz,
      gzz,  // gzz,
      Kxx,  // kxx,       // extrinsic curvature components
      Kxy,  // kxy,
      Kxz,  // kxz,
      Kyy,  // kyy,
      Kyz,  // kyz,
      Kzz   // kzz
      );
    };

  // interpolation without symmetries
  do_interp();

  int flat_ix;
  double psi4;

  GLOOP3(k,j,i){
    flat_ix = i + n[0]*(j + n[1]*k);

    psi4 = pow(psi[flat_ix], 4);
    adm.psi4(k, j, i) = psi4;

    // needs psi4 multiplication (done lower)
    adm.g_dd(0, 0, k, j, i) = gxx[flat_ix];
    adm.g_dd(1, 1, k, j, i) = gyy[flat_ix];
    adm.g_dd(2, 2, k, j, i) = gzz[flat_ix];
    adm.g_dd(0, 1, k, j, i) = gxy[flat_ix];
    adm.g_dd(0, 2, k, j, i) = gxz[flat_ix];
    adm.g_dd(1, 2, k, j, i) = gyz[flat_ix];

    adm.K_dd(0, 0, k, j, i) = Kxx[flat_ix];
    adm.K_dd(1, 1, k, j, i) = Kyy[flat_ix];
    adm.K_dd(2, 2, k, j, i) = Kzz[flat_ix];
    adm.K_dd(0, 1, k, j, i) = Kxy[flat_ix];
    adm.K_dd(0, 2, k, j, i) = Kxz[flat_ix];
    adm.K_dd(1, 2, k, j, i) = Kyz[flat_ix];
  }

  if (id_impose_symmetries)
  {
    int strides[3] = {1, n[0], n[1] * n[2]};

    /*
    // flip coordinate sgn to always use the same data
    // this will ensure symmetry preservation on the ID
    //
    // below we apply appropriate phases
    if (id_bitant_x)
      impose_ordinate_sym_pos(x, n[0]);  // mutates x

    if (id_bitant_y)
      impose_ordinate_sym_pos(y, n[1]);  // mutates y

    if (id_bitant_z)
      impose_ordinate_sym_pos(z, n[2]);  // mutates z
    */


    /*
    if (id_bitant_x)
    {
      int strides[3] = {1, n[0], n[1] * n[2]};
      impose_bitant_phase_x_dir(&mbi.x1(0), gxy, gxz, n, strides);
      impose_bitant_phase_x_dir(&mbi.x1(0), Kxy, Kxz, n, strides);
    }

    if (id_bitant_y)
    {
      int strides[3] = {1, n[0], n[1] * n[2]};
      impose_bitant_phase_y_dir(&mbi.x2(0), gxy, gyz, n, strides);
      impose_bitant_phase_y_dir(&mbi.x2(0), Kxy, Kyz, n, strides);
    }

    if (id_bitant_z)
    {
      int strides[3] = {1, n[0], n[1] * n[2]};
      impose_bitant_phase_z_dir(&mbi.x3(0), gxz, gyz, n, strides);
      impose_bitant_phase_z_dir(&mbi.x3(0), Kxz, Kyz, n, strides);
    }
    */

    // scalar field symmetries ------------------------------------------------
    if (id_scalars_symmetric_x)
      impose_ordinate_sym_pos(x, n[0]);  // mutates x

    if (id_scalars_symmetric_y)
      impose_ordinate_sym_pos(y, n[1]);  // mutates y

    if (id_scalars_symmetric_z)
      impose_ordinate_sym_pos(z, n[2]);  // mutates z

    if (id_scalars_symmetric_x or
        id_scalars_symmetric_y or
        id_scalars_symmetric_z)
    {
      do_interp();

      GLOOP3(k,j,i){
        flat_ix = i + n[0]*(j + n[1]*k);
        psi4 = pow(psi[flat_ix], 4);
        adm.psi4(k, j, i) = psi4;
      }

      // reset original variables
      copy_ordinates(x, y, z, mbi.x1, mbi.x2, mbi.x3, n);
    }
    // ------------------------------------------------------------------------

    // ------------------------------------------------------------------------
    // Deal with extrinsic curvature symmetries

    // K_xx -----------------------------------------------
    if (id_K_xx_symmetry_x != 0)
      impose_ordinate_sym_pos(x, n[0]);  // mutates x

    if (id_K_xx_symmetry_y != 0)
      impose_ordinate_sym_pos(y, n[1]);  // mutates y

    if (id_K_xx_symmetry_z != 0)
      impose_ordinate_sym_pos(z, n[2]);  // mutates z

    if ((id_K_xx_symmetry_x != 0) or
        (id_K_xx_symmetry_y != 0) or
        (id_K_xx_symmetry_z != 0))
    {

      do_interp();
      // reset original variables
      copy_ordinates(x, y, z, mbi.x1, mbi.x2, mbi.x3, n);

      // impose phases
      impose_sym2_phase(
        id_K_xx_symmetry_x,
        id_K_xx_symmetry_y,
        id_K_xx_symmetry_z,
        x, y, z, Kxx, n, strides
      );

      GLOOP3(k,j,i){
        flat_ix = i + n[0]*(j + n[1]*k);
        adm.K_dd(0, 0, k, j, i) = Kxx[flat_ix];
      }

    }
    // ----------------------------------------------------

    // K_xy -----------------------------------------------
    if (id_K_xy_symmetry_x != 0)
      impose_ordinate_sym_pos(x, n[0]);  // mutates x

    if (id_K_xy_symmetry_y != 0)
      impose_ordinate_sym_pos(y, n[1]);  // mutates y

    if (id_K_xy_symmetry_z != 0)
      impose_ordinate_sym_pos(z, n[2]);  // mutates z

    if ((id_K_xy_symmetry_x != 0) or
        (id_K_xy_symmetry_y != 0) or
        (id_K_xy_symmetry_z != 0))
    {

      do_interp();
      // reset original variables
      copy_ordinates(x, y, z, mbi.x1, mbi.x2, mbi.x3, n);

      // impose phases
      impose_sym2_phase(
        id_K_xy_symmetry_x,
        id_K_xy_symmetry_y,
        id_K_xy_symmetry_z,
        x, y, z, Kxy, n, strides
      );

      GLOOP3(k,j,i){
        flat_ix = i + n[0]*(j + n[1]*k);
        adm.K_dd(0, 1, k, j, i) = Kxy[flat_ix];
      }

    }
    // ----------------------------------------------------

    // K_xz -----------------------------------------------
    if (id_K_xz_symmetry_x != 0)
      impose_ordinate_sym_pos(x, n[0]);  // mutates x

    if (id_K_xz_symmetry_y != 0)
      impose_ordinate_sym_pos(y, n[1]);  // mutates y

    if (id_K_xz_symmetry_z != 0)
      impose_ordinate_sym_pos(z, n[2]);  // mutates z

    if ((id_K_xz_symmetry_x != 0) or
        (id_K_xz_symmetry_y != 0) or
        (id_K_xz_symmetry_z != 0))
    {

      do_interp();
      // reset original variables
      copy_ordinates(x, y, z, mbi.x1, mbi.x2, mbi.x3, n);

      // impose phases
      impose_sym2_phase(
        id_K_xz_symmetry_x,
        id_K_xz_symmetry_y,
        id_K_xz_symmetry_z,
        x, y, z, Kxz, n, strides
      );

      GLOOP3(k,j,i){
        flat_ix = i + n[0]*(j + n[1]*k);
        adm.K_dd(0, 2, k, j, i) = Kxz[flat_ix];
      }

    }
    // ----------------------------------------------------

    // K_yy -----------------------------------------------
    if (id_K_yy_symmetry_x != 0)
      impose_ordinate_sym_pos(x, n[0]);  // mutates x

    if (id_K_yy_symmetry_y != 0)
      impose_ordinate_sym_pos(y, n[1]);  // mutates y

    if (id_K_yy_symmetry_z != 0)
      impose_ordinate_sym_pos(z, n[2]);  // mutates z

    if ((id_K_yy_symmetry_x != 0) or
        (id_K_yy_symmetry_y != 0) or
        (id_K_yy_symmetry_z != 0))
    {

      do_interp();
      // reset original variables
      copy_ordinates(x, y, z, mbi.x1, mbi.x2, mbi.x3, n);

      // impose phases
      impose_sym2_phase(
        id_K_yy_symmetry_x,
        id_K_yy_symmetry_y,
        id_K_yy_symmetry_z,
        x, y, z, Kyy, n, strides
      );

      GLOOP3(k,j,i){
        flat_ix = i + n[0]*(j + n[1]*k);
        adm.K_dd(1, 1, k, j, i) = Kyy[flat_ix];
      }

    }
    // ----------------------------------------------------

    // K_yz -----------------------------------------------
    if (id_K_yz_symmetry_x != 0)
      impose_ordinate_sym_pos(x, n[0]);  // mutates x

    if (id_K_yz_symmetry_y != 0)
      impose_ordinate_sym_pos(y, n[1]);  // mutates y

    if (id_K_yz_symmetry_z != 0)
      impose_ordinate_sym_pos(z, n[2]);  // mutates z

    if ((id_K_yz_symmetry_x != 0) or
        (id_K_yz_symmetry_y != 0) or
        (id_K_yz_symmetry_z != 0))
    {

      do_interp();
      // reset original variables
      copy_ordinates(x, y, z, mbi.x1, mbi.x2, mbi.x3, n);

      // impose phases
      impose_sym2_phase(
        id_K_yz_symmetry_x,
        id_K_yz_symmetry_y,
        id_K_yz_symmetry_z,
        x, y, z, Kyz, n, strides
      );

      GLOOP3(k,j,i){
        flat_ix = i + n[0]*(j + n[1]*k);
        adm.K_dd(1, 2, k, j, i) = Kyz[flat_ix];
      }

    }
    // ----------------------------------------------------

    // K_zz -----------------------------------------------
    if (id_K_zz_symmetry_x != 0)
      impose_ordinate_sym_pos(x, n[0]);  // mutates x

    if (id_K_zz_symmetry_y != 0)
      impose_ordinate_sym_pos(y, n[1]);  // mutates y

    if (id_K_zz_symmetry_z != 0)
      impose_ordinate_sym_pos(z, n[2]);  // mutates z

    if ((id_K_zz_symmetry_x != 0) or
        (id_K_zz_symmetry_y != 0) or
        (id_K_zz_symmetry_z != 0))
    {

      do_interp();
      // reset original variables
      copy_ordinates(x, y, z, mbi.x1, mbi.x2, mbi.x3, n);

      // impose phases
      impose_sym2_phase(
        id_K_zz_symmetry_x,
        id_K_zz_symmetry_y,
        id_K_zz_symmetry_z,
        x, y, z, Kzz, n, strides
      );

      GLOOP3(k,j,i){
        flat_ix = i + n[0]*(j + n[1]*k);
        adm.K_dd(2, 2, k, j, i) = Kzz[flat_ix];
      }

    }
    // ----------------------------------------------------

    // ------------------------------------------------------------------------
    // Deal with metric symmetries

    // g_xx -----------------------------------------------
    if (id_g_xx_symmetry_x != 0)
      impose_ordinate_sym_pos(x, n[0]);  // mutates x

    if (id_g_xx_symmetry_y != 0)
      impose_ordinate_sym_pos(y, n[1]);  // mutates y

    if (id_g_xx_symmetry_z != 0)
      impose_ordinate_sym_pos(z, n[2]);  // mutates z

    if ((id_g_xx_symmetry_x != 0) or
        (id_g_xx_symmetry_y != 0) or
        (id_g_xx_symmetry_z != 0))
    {

      do_interp();
      // reset original variables
      copy_ordinates(x, y, z, mbi.x1, mbi.x2, mbi.x3, n);

      // impose phases
      impose_sym2_phase(
        id_g_xx_symmetry_x,
        id_g_xx_symmetry_y,
        id_g_xx_symmetry_z,
        x, y, z, gxx, n, strides
      );

      GLOOP3(k,j,i){
        flat_ix = i + n[0]*(j + n[1]*k);
        adm.g_dd(0, 0, k, j, i) = gxx[flat_ix];
      }

    }
    // ----------------------------------------------------

    // g_xy -----------------------------------------------
    if (id_g_xy_symmetry_x != 0)
      impose_ordinate_sym_pos(x, n[0]);  // mutates x

    if (id_g_xy_symmetry_y != 0)
      impose_ordinate_sym_pos(y, n[1]);  // mutates y

    if (id_g_xy_symmetry_z != 0)
      impose_ordinate_sym_pos(z, n[2]);  // mutates z

    if ((id_g_xy_symmetry_x != 0) or
        (id_g_xy_symmetry_y != 0) or
        (id_g_xy_symmetry_z != 0))
    {

      do_interp();
      // reset original variables
      copy_ordinates(x, y, z, mbi.x1, mbi.x2, mbi.x3, n);

      // impose phases
      impose_sym2_phase(
        id_g_xy_symmetry_x,
        id_g_xy_symmetry_y,
        id_g_xy_symmetry_z,
        x, y, z, gxy, n, strides
      );

      GLOOP3(k,j,i){
        flat_ix = i + n[0]*(j + n[1]*k);
        adm.g_dd(0, 1, k, j, i) = gxy[flat_ix];
      }

    }
    // ----------------------------------------------------

    // g_xz -----------------------------------------------
    if (id_g_xz_symmetry_x != 0)
      impose_ordinate_sym_pos(x, n[0]);  // mutates x

    if (id_g_xz_symmetry_y != 0)
      impose_ordinate_sym_pos(y, n[1]);  // mutates y

    if (id_g_xz_symmetry_z != 0)
      impose_ordinate_sym_pos(z, n[2]);  // mutates z

    if ((id_g_xz_symmetry_x != 0) or
        (id_g_xz_symmetry_y != 0) or
        (id_g_xz_symmetry_z != 0))
    {

      do_interp();
      // reset original variables
      copy_ordinates(x, y, z, mbi.x1, mbi.x2, mbi.x3, n);

      // impose phases
      impose_sym2_phase(
        id_g_xz_symmetry_x,
        id_g_xz_symmetry_y,
        id_g_xz_symmetry_z,
        x, y, z, gxz, n, strides
      );

      GLOOP3(k,j,i){
        flat_ix = i + n[0]*(j + n[1]*k);
        adm.g_dd(0, 2, k, j, i) = gxz[flat_ix];
      }

    }
    // ----------------------------------------------------

    // g_yy -----------------------------------------------
    if (id_g_yy_symmetry_x != 0)
      impose_ordinate_sym_pos(x, n[0]);  // mutates x

    if (id_g_yy_symmetry_y != 0)
      impose_ordinate_sym_pos(y, n[1]);  // mutates y

    if (id_g_yy_symmetry_z != 0)
      impose_ordinate_sym_pos(z, n[2]);  // mutates z

    if ((id_g_yy_symmetry_x != 0) or
        (id_g_yy_symmetry_y != 0) or
        (id_g_yy_symmetry_z != 0))
    {

      do_interp();
      // reset original variables
      copy_ordinates(x, y, z, mbi.x1, mbi.x2, mbi.x3, n);

      // impose phases
      impose_sym2_phase(
        id_g_yy_symmetry_x,
        id_g_yy_symmetry_y,
        id_g_yy_symmetry_z,
        x, y, z, gyy, n, strides
      );

      GLOOP3(k,j,i){
        flat_ix = i + n[0]*(j + n[1]*k);
        adm.g_dd(1, 1, k, j, i) = gyy[flat_ix];
      }

    }
    // ----------------------------------------------------

    // g_yz -----------------------------------------------
    if (id_g_yz_symmetry_x != 0)
      impose_ordinate_sym_pos(x, n[0]);  // mutates x

    if (id_g_yz_symmetry_y != 0)
      impose_ordinate_sym_pos(y, n[1]);  // mutates y

    if (id_g_yz_symmetry_z != 0)
      impose_ordinate_sym_pos(z, n[2]);  // mutates z

    if ((id_g_yz_symmetry_x != 0) or
        (id_g_yz_symmetry_y != 0) or
        (id_g_yz_symmetry_z != 0))
    {

      do_interp();
      // reset original variables
      copy_ordinates(x, y, z, mbi.x1, mbi.x2, mbi.x3, n);

      // impose phases
      impose_sym2_phase(
        id_g_yz_symmetry_x,
        id_g_yz_symmetry_y,
        id_g_yz_symmetry_z,
        x, y, z, gyz, n, strides
      );

      GLOOP3(k,j,i){
        flat_ix = i + n[0]*(j + n[1]*k);
        adm.g_dd(1, 2, k, j, i) = gyz[flat_ix];
      }

    }
    // ----------------------------------------------------

    // g_zz -----------------------------------------------
    if (id_g_zz_symmetry_x != 0)
      impose_ordinate_sym_pos(x, n[0]);  // mutates x

    if (id_g_zz_symmetry_y != 0)
      impose_ordinate_sym_pos(y, n[1]);  // mutates y

    if (id_g_zz_symmetry_z != 0)
      impose_ordinate_sym_pos(z, n[2]);  // mutates z

    if ((id_g_zz_symmetry_x != 0) or
        (id_g_zz_symmetry_y != 0) or
        (id_g_zz_symmetry_z != 0))
    {

      do_interp();
      // reset original variables
      copy_ordinates(x, y, z, mbi.x1, mbi.x2, mbi.x3, n);

      // impose phases
      impose_sym2_phase(
        id_g_zz_symmetry_x,
        id_g_zz_symmetry_y,
        id_g_zz_symmetry_z,
        x, y, z, gzz, n, strides
      );

      GLOOP3(k,j,i){
        flat_ix = i + n[0]*(j + n[1]*k);
        adm.g_dd(2, 2, k, j, i) = gzz[flat_ix];
      }

    }
    // ----------------------------------------------------

  }

  // finalize psi4 multiplication
  GLOOP3(k,j,i){
    adm.g_dd(0, 0, k, j, i) = adm.psi4(k, j, i) * adm.g_dd(0, 0, k, j, i);
    adm.g_dd(1, 1, k, j, i) = adm.psi4(k, j, i) * adm.g_dd(1, 1, k, j, i);
    adm.g_dd(2, 2, k, j, i) = adm.psi4(k, j, i) * adm.g_dd(2, 2, k, j, i);
    adm.g_dd(0, 1, k, j, i) = adm.psi4(k, j, i) * adm.g_dd(0, 1, k, j, i);
    adm.g_dd(0, 2, k, j, i) = adm.psi4(k, j, i) * adm.g_dd(0, 2, k, j, i);
    adm.g_dd(1, 2, k, j, i) = adm.psi4(k, j, i) * adm.g_dd(1, 2, k, j, i);
  }

  delete [] gxx; delete [] gyy; delete [] gzz;
  delete [] gxy; delete [] gxz; delete [] gyz;

  delete [] Kxx; delete [] Kyy; delete [] Kzz;
  delete [] Kxy; delete [] Kxz; delete [] Kyz;

  delete [] psi; delete [] alp;

  delete [] x; delete [] y; delete [] z;

  if(verbose)
    std::cout << "\n\n<-Z4c::ADMTwoPunctures\n\n";
}

//void Z4c::DebugInfoVars(){
//  // dump some basic info to term
//  printf("\n\n->Z4c::DebugInfoVars\n");
//
//  MeshBlock * pmb = pmy_block;
//  Coordinates * pco = pmb->pcoord;
//
//  printf("\n=Ghost node info:\n");
//  printf("(GSIZEI, GSIZEJ, GSIZEK)=(%d, %d, %d)\n",
//         GSIZEI, GSIZEJ, GSIZEK);
//
//  printf("\n=MeshBlock.block_size [physical nodes]");
//
//  int nxyz[3] = {(*pmb).block_size.nx1,
//                 (*pmb).block_size.nx2,
//                 (*pmb).block_size.nx3};
//
//  printf("(nx1, nx2, nx3)=(%d, %d, %d)\n", nxyz[0], nxyz[1], nxyz[2]);
//  printf("(x1min, x1max)=(%lf, %lf)\n",
//         (*pmb).block_size.x1min,
//         (*pmb).block_size.x1max);
//
//  printf("(x2min, x2max)=(%lf, %lf)\n",
//         (*pmb).block_size.x2min,
//         (*pmb).block_size.x2max);
//
//  printf("(x3min, x3max)=(%lf, %lf)\n",
//         (*pmb).block_size.x3min,
//         (*pmb).block_size.x3max);
//
//
//  printf("\n=Coordinates info [current block with ghosts]\n");
//  printf("(x1min, x1max)=(%lf, %lf)\n",
//         (*pco).x1v(0),
//         (*pco).x1v(nxyz[0] + GSIZEI * 2));
//
//  printf("(x2min, x2max)=(%lf, %lf)\n",
//         (*pco).x2v(0),
//         (*pco).x2v(nxyz[1] + GSIZEJ * 2));
//
//  printf("(x3min, x3max)=(%lf, %lf)\n",
//         (*pco).x3v(0),
//         (*pco).x3v(nxyz[2] + GSIZEK * 2));
//
//
//  printf("\n\n<-Z4c::DebugInfoVars\n");
//
//}
