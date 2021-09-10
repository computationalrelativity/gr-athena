// for testing util

// C, C++ headers
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>     // std::vector
#include <string>

// C++ headers
#include <cmath>      // sqrt()
// #include <csignal>    // ISO C/C++ signal() and sigset_t, sigemptyset() POSIX C extensions
// #include <cstdint>    // int64_t
// #include <cstdio>     // sscanf()
// #include <cstdlib>    // strtol
// #include <ctime>      // clock(), CLOCKS_PER_SEC, clock_t
// #include <exception>  // exception
// #include <iomanip>    // setprecision()
// #include <limits>     // max_digits10
// #include <new>        // bad_alloc
#include <time.h>

// #include <fenv.h>     // for exception tracking

// #include <benchmark/benchmark.h>

// Cut down Athena++ functionality --------------------------------------------
#include "athena.hpp"
#include "athena_arrays.hpp"

#include "defs.hpp"
// ----------------------------------------------------------------------------

#include "utils/interp_intergrid.hpp"

// ----------------------------------------------------------------------------
// for testing:
static AthenaArray<Real> make_ords(
  const Real a, const Real b, const int N, const int nghost)
{

  Real dx = (b - a) / N;

  AthenaArray<Real> ords;
  ords.NewAthenaArray(N+2*nghost+1);

  for(int ix=0; ix<N+2*nghost+1; ++ix) {
    ords(ix) = a + dx * (ix - nghost);
  }

  return ords;
}

static AthenaArray<Real> make_ords_cc(
  const Real a, const Real b, const int N, const int nghost)
{
  // cell-centered analogue of make_ords
  Real dx = (b - a) / N;

  AthenaArray<Real> ords;
  ords.NewAthenaArray(N+2*nghost);

  for(int ix=0; ix<N+2*nghost; ++ix) {
    ords(ix) = a + dx * (ix - nghost) + (dx / 2);
  }

  return ords;
}

static void test_1d(const bool stdout_dump)
{
  const int N_x = 6;
  const int sz_x_vc = 2*VC_NGHOST+N_x+1;
  const int sz_x_cc = 2*CC_NGHOST+N_x;

  Real x_a = 1.2;
  Real x_b = 2.4;

  AthenaArray<Real> x_vc = make_ords(x_a, x_b, N_x, VC_NGHOST);
  AthenaArray<Real> x_cc = make_ords_cc(x_a, x_b, N_x, CC_NGHOST);
  AthenaArray<Real> rds;
  rds.NewAthenaArray(1);
  rds(0) = 1./(x_cc(1) - x_cc(0));

  AthenaArray<Real> poly_x_vc, poly_x_cc, dpoly_x_cc;
  AthenaArray<Real> i_poly_x_vc, i_poly_x_cc, i_dpoly_x_cc;
  AthenaArray<Real> li_poly_x_vc, li_poly_x_cc, li_dpoly_x_cc;

  poly_x_vc.NewAthenaArray(sz_x_vc);
  poly_x_cc.NewAthenaArray(sz_x_cc);
  dpoly_x_cc.NewAthenaArray(sz_x_cc);

  i_poly_x_vc.NewAthenaArray(sz_x_vc);
  i_poly_x_cc.NewAthenaArray(sz_x_cc);

  i_dpoly_x_cc.NewAthenaArray(2, sz_x_cc);

  li_dpoly_x_cc.NewAthenaArray(sz_x_cc);


  li_poly_x_vc.NewAthenaArray(sz_x_vc);
  li_poly_x_cc.NewAthenaArray(sz_x_cc);

  // AthenaArray<Real> poly_x_vc = test_utils::zero_vec(sz_x_vc);
  // AthenaArray<Real> poly_x_cc = test_utils::zero_vec(sz_x_cc);

  // AthenaArray<Real> i_poly_x_cc = test_utils::zero_vec(sz_x_cc);
  // AthenaArray<Real> i_poly_x_vc = test_utils::zero_vec(sz_x_vc);


  for(int i=0; i<sz_x_vc; ++i)
    poly_x_vc(i) = -2*std::pow(x_vc(i), 2.0);

  for(int i=0; i<sz_x_cc; ++i)
  {
    poly_x_cc(i) = -2*std::pow(x_cc(i), 2.0);
    dpoly_x_cc(i) = -4.0*x_cc(i);
  }

  // for(int lix=0; lix<NGRCV_HSZ; ++lix)
  // {
  //   Real const lc = InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][lix];
  //   printf("lc: %1.4f\n", lc);
  // }
  const int dim = 1;
  const int cc_il = CC_NGHOST, cc_iu = sz_x_cc-CC_NGHOST;
  InterpIntergrid::var_map_VC2CC(poly_x_vc, i_poly_x_cc, dim,
    0, 0, cc_il, cc_iu-1, 0, 0, 0, 0);

  InterpIntergrid::var_map_VC2CC_Taylor1(
    rds, poly_x_vc, i_dpoly_x_cc, dim,
    cc_il, cc_iu-1, 0, 0, 0, 0);

  const int vc_il = VC_NGHOST, vc_iu = sz_x_vc-VC_NGHOST;
  InterpIntergrid::var_map_CC2VC(poly_x_cc, i_poly_x_vc, dim,
    0, 0, vc_il, vc_iu-1, 0, 0, 0, 0);

  // construct test for InterpIntergridLocal ----------------------------------
  int N[] = {N_x};
  Real rdx[] = {1./(x_vc(1)-x_vc(0))};

  InterpIntergridLocal * ig = new InterpIntergridLocal(dim, &N[0], &rdx[0]);

  #pragma omp simd
  for(int l=cc_il; l<=(cc_iu-1); ++l)
    li_poly_x_cc(l) = ig->map1d_VC2CC(poly_x_vc(l));

  #pragma omp simd
  for(int l=vc_il; l<=(vc_iu-1); ++l)
    li_poly_x_vc(l) = ig->map1d_CC2VC(poly_x_cc(l));

  #pragma omp simd
  for(int l=cc_il; l<=(cc_iu-1); ++l)
   li_dpoly_x_cc(l) = ig->map1d_VC2CC_der(0,poly_x_vc(l));

  // --------------------------------------------------------------------------

  // compute sum of abs error over all interpolated points
  Real err_vc = 0.0, err_cc = 0.0, err_cc_loc = 0.0, err_vc_loc = 0.0;
  Real err_dcc = 0.0, err_dcc_loc = 0.0;

  for(int i=cc_il; i<=cc_iu-1; ++i)
  {
    err_cc += std::abs(
      i_poly_x_cc(i) - poly_x_cc(i)
    );

    err_cc_loc += std::abs(
      i_poly_x_cc(i) - li_poly_x_cc(i)
    );

    err_dcc += std::abs(
      dpoly_x_cc(i) - i_dpoly_x_cc(1,i)
    );

    err_dcc_loc += std::abs(
      dpoly_x_cc(i) - li_dpoly_x_cc(i)
    );

  }

  for(int i=vc_il; i<=vc_iu-1; ++i)
  {
    err_vc += std::abs(
      i_poly_x_vc(i) - poly_x_vc(i)
    );

    err_vc_loc += std::abs(
      i_poly_x_vc(i) - li_poly_x_vc(i)
    );
  }

  coutBoldBlue("(err_cc, err_vc, err_cc_loc, err_vc_loc, ");
  coutBoldBlue("err_dcc, err_dcc_loc):\n");
  printf("(%1.4e,%1.4e,%1.4e,%1.4e,%1.4e,%1.4e)\n",
    err_cc, err_vc, err_cc_loc, err_vc_loc, err_dcc, err_dcc_loc);


  if(stdout_dump)
  {
    // dump to std out
    coutBoldRed("x_vc:\n");
    x_vc.print_all();
    coutBoldRed("x_cc:\n");
    x_cc.print_all();
    coutBoldRed("poly_x_vc:\n");
    poly_x_vc.print_all();
    coutBoldRed("i_poly_x_vc:\n");
    i_poly_x_vc.print_all();
    coutBoldRed("li_poly_x_vc:\n");
    li_poly_x_vc.print_all();
    coutBoldRed("poly_x_cc:\n");
    poly_x_cc.print_all();
    coutBoldRed("i_poly_x_cc:\n");
    i_poly_x_cc.print_all();
    coutBoldRed("li_poly_x_cc:\n");
    li_poly_x_cc.print_all();

    coutBoldRed("dpoly_x_cc:\n");
    dpoly_x_cc.print_all();

    coutBoldRed("i_dpoly_x_cc:\n");
    i_dpoly_x_cc.print_all();

    coutBoldRed("li_dpoly_x_cc:\n");
    li_dpoly_x_cc.print_all();
  }

  // clean-up
  x_vc.DeleteAthenaArray();
  x_cc.DeleteAthenaArray();
  rds.DeleteAthenaArray();

  poly_x_vc.DeleteAthenaArray();
  poly_x_cc.DeleteAthenaArray();
  dpoly_x_cc.DeleteAthenaArray();

  i_poly_x_vc.DeleteAthenaArray();
  i_poly_x_vc.DeleteAthenaArray();
  i_dpoly_x_cc.DeleteAthenaArray();

  li_poly_x_vc.DeleteAthenaArray();
  li_poly_x_vc.DeleteAthenaArray();

  li_dpoly_x_cc.DeleteAthenaArray();

  delete ig;
}

static void test_2d(const bool stdout_dump)
{
  const int N_x = 6, N_y = 8;

  const int sz_x_vc = 2*VC_NGHOST+N_x+1;
  const int sz_x_cc = 2*CC_NGHOST+N_x;

  const int sz_y_vc = 2*VC_NGHOST+N_y+1;
  const int sz_y_cc = 2*CC_NGHOST+N_y;

  const Real x_a = 1.2, x_b = 2.4;
  const Real y_a = 3.2, y_b = 4.4;

  AthenaArray<Real> x_vc = make_ords(x_a, x_b, N_x, VC_NGHOST);
  AthenaArray<Real> x_cc = make_ords_cc(x_a, x_b, N_x, CC_NGHOST);

  AthenaArray<Real> y_vc = make_ords(y_a, y_b, N_y, VC_NGHOST);
  AthenaArray<Real> y_cc = make_ords_cc(y_a, y_b, N_y, CC_NGHOST);

  AthenaArray<Real> rds;
  rds.NewAthenaArray(2);
  rds(0) = 1./(x_cc(1) - x_cc(0));
  rds(1) = 1./(y_cc(1) - y_cc(0));

  AthenaArray<Real> poly_xy_vc, poly_xy_cc, dpoly_xy_cc;
  AthenaArray<Real> i_poly_xy_vc, i_poly_xy_cc, i_dpoly_xy_cc;
  AthenaArray<Real> li_poly_xy_vc, li_poly_xy_cc, li_dpoly_xy_cc;

  poly_xy_vc.NewAthenaArray(sz_y_vc, sz_x_vc);
  poly_xy_cc.NewAthenaArray(sz_y_cc, sz_x_cc);
  dpoly_xy_cc.NewAthenaArray(4, sz_y_cc, sz_x_cc);

  i_poly_xy_vc.NewAthenaArray(sz_y_vc, sz_x_vc);
  i_poly_xy_cc.NewAthenaArray(sz_y_cc, sz_x_cc);

  i_dpoly_xy_cc.NewAthenaArray(4, sz_y_cc, sz_x_cc);

  li_dpoly_xy_cc.NewAthenaArray(2, sz_y_cc ,sz_x_cc);

  li_poly_xy_vc.NewAthenaArray(sz_y_vc, sz_x_vc);
  li_poly_xy_cc.NewAthenaArray(sz_y_cc, sz_x_cc);


  // poly.
  for(int j=0; j<sz_y_vc; ++j)
    for(int i=0; i<sz_x_vc; ++i)
      poly_xy_vc(j,i) = -2*std::pow(x_vc(i), 2.0)*std::pow(y_vc(j), 2.0);

  for(int j=0; j<sz_y_cc; ++j)
    for(int i=0; i<sz_x_cc; ++i)
    {
      poly_xy_cc(j,i) = -2*std::pow(x_cc(i), 2.0)*std::pow(y_cc(j), 2.0);

      dpoly_xy_cc(0,j,i) = poly_xy_cc(j,i);
      dpoly_xy_cc(1,j,i) = -4.0*x_cc(i)*std::pow(y_cc(j), 2.0);
      dpoly_xy_cc(2,j,i) = -4.0*std::pow(x_cc(i), 2.0)*y_cc(j);
      dpoly_xy_cc(3,j,i) = -8.0*x_cc(i)*y_cc(j);
    }

  // linear.
  for(int j=0; j<sz_y_vc; ++j)
    for(int i=0; i<sz_x_vc; ++i)
      poly_xy_vc(j,i) = -2*(x_vc(i)-3)*(y_vc(j)+2);

  for(int j=0; j<sz_y_cc; ++j)
    for(int i=0; i<sz_x_cc; ++i)
    {
      poly_xy_cc(j,i) = -2*(x_cc(i)-3)*(y_cc(j)+2);

      dpoly_xy_cc(0,j,i) = poly_xy_cc(j,i);
      dpoly_xy_cc(1,j,i) = -2*(y_cc(j)+2);
      dpoly_xy_cc(2,j,i) = -2*(x_cc(i)-3);
      dpoly_xy_cc(3,j,i) = -2.0;
    }


  const int dim = 2;
  const int cc_il = CC_NGHOST, cc_iu = sz_x_cc-CC_NGHOST;
  const int cc_jl = CC_NGHOST, cc_ju = sz_y_cc-CC_NGHOST;

  InterpIntergrid::var_map_VC2CC(poly_xy_vc, i_poly_xy_cc, dim,
    0, 0, cc_il, cc_iu-1, cc_jl, cc_ju-1, 0, 0);

  InterpIntergrid::var_map_VC2CC_Taylor1(
    rds, poly_xy_vc, i_dpoly_xy_cc, dim,
    cc_il, cc_iu-1, cc_jl, cc_ju-1, 0, 0);

  const int vc_il = VC_NGHOST, vc_iu = sz_x_vc-VC_NGHOST;
  const int vc_jl = VC_NGHOST, vc_ju = sz_y_vc-VC_NGHOST;
  InterpIntergrid::var_map_CC2VC(poly_xy_cc, i_poly_xy_vc, dim,
    0, 0, vc_il, vc_iu-1, vc_jl, vc_ju-1, 0, 0);

  // construct test for InterpIntergridLocal ----------------------------------
  int N[] = {N_x, N_y};
  Real rdx[] = {1./(x_vc(1)-x_vc(0)), 1./(y_vc(1)-y_vc(0))};

  InterpIntergridLocal * ig = new InterpIntergridLocal(dim, &N[0], &rdx[0]);

  for(int m=cc_jl; m<=(cc_ju-1); ++m)
  {
#pragma omp simd
    for(int l=cc_il; l<=(cc_iu-1); ++l)
      li_poly_xy_cc(m,l) = ig->map2d_VC2CC(poly_xy_vc(m,l));
  }

  for(int m=vc_jl; m<=(vc_ju-1); ++m)
  {
#pragma omp simd
    for(int l=vc_il; l<=(vc_iu-1); ++l)
      li_poly_xy_vc(m,l) = ig->map2d_CC2VC(poly_xy_cc(m,l));
  }

  for(int m=cc_jl; m<=(cc_ju-1); ++m)
  {
#pragma omp simd
    for(int l=cc_il; l<=(cc_iu-1); ++l)
    {
      li_dpoly_xy_cc(0,m,l) = ig->map2d_VC2CC_der(0,poly_xy_vc(m,l));
      li_dpoly_xy_cc(1,m,l) = ig->map2d_VC2CC_der(1,poly_xy_vc(m,l));
    }
  }

  // --------------------------------------------------------------------------


  // compute sum of abs error over all interpolated points
  Real err_vc = 0.0, err_cc = 0.0, err_cc_loc = 0.0, err_vc_loc = 0.0;
  Real err_dcc = 0.0, err_dcc_loc = 0.0;

  for(int j=cc_jl; j<=cc_ju-1; ++j)
    for(int i=cc_il; i<=cc_iu-1; ++i)
    {
      err_cc += std::abs(
        i_poly_xy_cc(j,i) - poly_xy_cc(j,i)
      );

      err_cc_loc += std::abs(
        i_poly_xy_cc(j,i) - li_poly_xy_cc(j,i)
      );

      for(int f=0; f<4; ++f)
      {
        err_dcc += std::abs(
          dpoly_xy_cc(f,j,i) - i_dpoly_xy_cc(f,j,i)
        );
      }

      err_dcc_loc += std::abs(
        dpoly_xy_cc(1,j,i) - li_dpoly_xy_cc(0,j,i)
      );

      err_dcc_loc += std::abs(
        dpoly_xy_cc(2,j,i) - li_dpoly_xy_cc(1,j,i)
      );

    }

  for(int j=vc_jl; j<=vc_ju-1; ++j)
    for(int i=vc_il; i<=vc_iu-1; ++i)
    {
      err_vc += std::abs(
        i_poly_xy_vc(j,i) - poly_xy_vc(j,i)
      );

      err_vc_loc += std::abs(
        i_poly_xy_vc(j,i) - li_poly_xy_vc(j,i)
      );

    }

  coutBoldBlue("(err_cc, err_vc, err_cc_loc, err_vc_loc, ");
  coutBoldBlue("err_dcc, err_dcc_loc):\n");
  printf("(%1.4e,%1.4e,%1.4e,%1.4e,%1.4e,%1.4e)\n",
    err_cc, err_vc, err_cc_loc, err_vc_loc, err_dcc, err_dcc_loc);

  if(stdout_dump)
  {
    coutBoldRed("poly_xy_vc:\n");
    poly_xy_vc.print_all();

    coutBoldRed("i_poly_xy_vc:\n");
    i_poly_xy_vc.print_all();

    coutBoldRed("li_poly_xy_vc:\n");
    li_poly_xy_vc.print_all();

    coutBoldRed("poly_xy_cc:\n");
    poly_xy_cc.print_all();

    coutBoldRed("i_poly_xy_cc:\n");
    i_poly_xy_cc.print_all();

    coutBoldRed("li_poly_xy_cc:\n");
    li_poly_xy_cc.print_all();

    coutBoldRed("dpoly_xy_cc:\n");
    dpoly_xy_cc.print_all();
    coutBoldRed("i_dpoly_xy_cc:\n");
    i_dpoly_xy_cc.print_all();

  }

  // clean-up
  x_vc.DeleteAthenaArray();
  x_cc.DeleteAthenaArray();

  y_vc.DeleteAthenaArray();
  y_cc.DeleteAthenaArray();

  rds.DeleteAthenaArray();

  poly_xy_vc.DeleteAthenaArray();
  poly_xy_cc.DeleteAthenaArray();
  dpoly_xy_cc.DeleteAthenaArray();

  i_poly_xy_vc.DeleteAthenaArray();
  i_poly_xy_vc.DeleteAthenaArray();
  i_dpoly_xy_cc.DeleteAthenaArray();

  li_dpoly_xy_cc.DeleteAthenaArray();
  li_poly_xy_vc.DeleteAthenaArray();
  li_poly_xy_vc.DeleteAthenaArray();

  delete ig;
}

static void test_3d(const bool stdout_dump)
{
  const int N_x = 6, N_y = 8, N_z = 4;

  const int sz_x_vc = 2*VC_NGHOST+N_x+1;
  const int sz_x_cc = 2*CC_NGHOST+N_x;

  const int sz_y_vc = 2*VC_NGHOST+N_y+1;
  const int sz_y_cc = 2*CC_NGHOST+N_y;

  const int sz_z_vc = 2*VC_NGHOST+N_z+1;
  const int sz_z_cc = 2*CC_NGHOST+N_z;

  const Real x_a = 1.2, x_b = 2.4;
  const Real y_a = 3.2, y_b = 4.4;
  const Real z_a = 2.2, z_b = 3.7;

  AthenaArray<Real> x_vc = make_ords(x_a, x_b, N_x, VC_NGHOST);
  AthenaArray<Real> x_cc = make_ords_cc(x_a, x_b, N_x, CC_NGHOST);

  AthenaArray<Real> y_vc = make_ords(y_a, y_b, N_y, VC_NGHOST);
  AthenaArray<Real> y_cc = make_ords_cc(y_a, y_b, N_y, CC_NGHOST);

  AthenaArray<Real> z_vc = make_ords(z_a, z_b, N_z, VC_NGHOST);
  AthenaArray<Real> z_cc = make_ords_cc(z_a, z_b, N_z, CC_NGHOST);

  AthenaArray<Real> rds;
  rds.NewAthenaArray(3);
  rds(0) = 1./(x_cc(1) - x_cc(0));
  rds(1) = 1./(y_cc(1) - y_cc(0));
  rds(2) = 1./(z_cc(1) - z_cc(0));

  AthenaArray<Real> poly_xyz_vc, poly_xyz_cc, dpoly_xyz_cc;
  AthenaArray<Real> i_poly_xyz_vc, i_poly_xyz_cc, i_dpoly_xyz_cc;
  AthenaArray<Real> li_poly_xyz_vc, li_poly_xyz_cc, li_dpoly_xyz_cc;

  poly_xyz_vc.NewAthenaArray(sz_z_vc, sz_y_vc, sz_x_vc);
  poly_xyz_cc.NewAthenaArray(sz_z_cc, sz_y_cc, sz_x_cc);
  dpoly_xyz_cc.NewAthenaArray(8, sz_z_cc, sz_y_cc, sz_x_cc);

  i_poly_xyz_vc.NewAthenaArray(sz_z_vc, sz_y_vc, sz_x_vc);
  i_poly_xyz_cc.NewAthenaArray(sz_z_cc, sz_y_cc, sz_x_cc);
  i_dpoly_xyz_cc.NewAthenaArray(8,sz_z_cc, sz_y_cc, sz_x_cc);

  li_dpoly_xyz_cc.NewAthenaArray(3, sz_z_cc, sz_y_cc ,sz_x_cc);


  li_poly_xyz_vc.NewAthenaArray(sz_z_vc, sz_y_vc, sz_x_vc);
  li_poly_xyz_cc.NewAthenaArray(sz_z_cc, sz_y_cc, sz_x_cc);


  // linear
  for(int k=0; k<sz_z_vc; ++k)
  for(int j=0; j<sz_y_vc; ++j)
  for(int i=0; i<sz_x_vc; ++i)
    poly_xyz_vc(k,j,i) = -2*(x_vc(i)-2)*(y_vc(j)+1)*(z_vc(k)-3);

  for(int k=0; k<sz_z_cc; ++k)
  for(int j=0; j<sz_y_cc; ++j)
  for(int i=0; i<sz_x_cc; ++i)
  {
    poly_xyz_cc(k,j,i) = -2*(x_cc(i)-2)*(y_cc(j)+1)*(z_cc(k)-3);

    dpoly_xyz_cc(0,k,j,i) = poly_xyz_cc(k,j,i);
    dpoly_xyz_cc(1,k,j,i) = -2*(y_cc(j)+1)*(z_cc(k)-3);
    dpoly_xyz_cc(2,k,j,i) = -2*(x_cc(i)-2)*(z_cc(k)-3);
    dpoly_xyz_cc(3,k,j,i) = -2*(x_cc(i)-2)*(y_cc(j)+1);
    dpoly_xyz_cc(4,k,j,i) = -2*(z_cc(k)-3);
    dpoly_xyz_cc(5,k,j,i) = -2*(x_cc(i)-2);
    dpoly_xyz_cc(6,k,j,i) = -2*(y_cc(j)+1);
    dpoly_xyz_cc(7,k,j,i) = -2.0;
  }


  const int dim = 3;
  const int cc_il = CC_NGHOST, cc_iu = sz_x_cc-CC_NGHOST;
  const int cc_jl = CC_NGHOST, cc_ju = sz_y_cc-CC_NGHOST;
  const int cc_kl = CC_NGHOST, cc_ku = sz_z_cc-CC_NGHOST;

  InterpIntergrid::var_map_VC2CC(poly_xyz_vc, i_poly_xyz_cc, dim,
    0, 0, cc_il, cc_iu-1, cc_jl, cc_ju-1, cc_kl, cc_ku-1);

  InterpIntergrid::var_map_VC2CC_Taylor1(
    rds, poly_xyz_vc, i_dpoly_xyz_cc, dim,
    cc_il, cc_iu-1, cc_jl, cc_ju-1, cc_kl, cc_ku-1);

  const int vc_il = VC_NGHOST, vc_iu = sz_x_vc-VC_NGHOST;
  const int vc_jl = VC_NGHOST, vc_ju = sz_y_vc-VC_NGHOST;
  const int vc_kl = VC_NGHOST, vc_ku = sz_z_vc-VC_NGHOST;
  InterpIntergrid::var_map_CC2VC(poly_xyz_cc, i_poly_xyz_vc, dim,
    0, 0, vc_il, vc_iu-1, vc_jl, vc_ju-1, vc_kl, vc_ku-1);

  // construct test for InterpIntergridLocal ----------------------------------
  int N[] = {N_x, N_y, N_z};
  Real rdx[] = {
    1./(x_vc(1)-x_vc(0)), 1./(y_vc(1)-y_vc(0)), 1./(z_vc(1)-z_vc(0))
  };

  InterpIntergridLocal * ig = new InterpIntergridLocal(dim, &N[0], &rdx[0]);

  for(int n=cc_kl; n<=(cc_ku-1); ++n)
    for(int m=cc_jl; m<=(cc_ju-1); ++m)
    {
#pragma omp simd
      for(int l=cc_il; l<=(cc_iu-1); ++l)
        li_poly_xyz_cc(n,m,l) = ig->map3d_VC2CC(poly_xyz_vc(n,m,l));
    }

  for(int n=vc_kl; n<=(vc_ku-1); ++n)
    for(int m=vc_jl; m<=(vc_ju-1); ++m)
    {
#pragma omp simd
      for(int l=vc_il; l<=(vc_iu-1); ++l)
        li_poly_xyz_vc(n,m,l) = ig->map3d_CC2VC(poly_xyz_cc(n,m,l));
    }

  for(int n=cc_kl; n<=(cc_ku-1); ++n)
    for(int m=cc_jl; m<=(cc_ju-1); ++m)
    {
#pragma omp simd
      for(int l=cc_il; l<=(cc_iu-1); ++l)
      {
        li_dpoly_xyz_cc(0,n,m,l) = ig->map3d_VC2CC_der(0, poly_xyz_vc(n,m,l));
        li_dpoly_xyz_cc(1,n,m,l) = ig->map3d_VC2CC_der(1, poly_xyz_vc(n,m,l));
        li_dpoly_xyz_cc(2,n,m,l) = ig->map3d_VC2CC_der(2, poly_xyz_vc(n,m,l));
      }
    }

  // --------------------------------------------------------------------------


  // compute sum of abs error over all interpolated points
  Real err_vc = 0.0, err_cc = 0.0, err_cc_loc = 0.0, err_vc_loc = 0.0;
  Real err_dcc = 0.0, err_dcc_loc = 0.0;

  for(int k=cc_kl; k<=cc_ku-1; ++k)
  for(int j=cc_jl; j<=cc_ju-1; ++j)
  for(int i=cc_il; i<=cc_iu-1; ++i)
  {
    err_cc += std::abs(
      i_poly_xyz_cc(k,j,i) - poly_xyz_cc(k,j,i)
    );

    err_cc_loc += std::abs(
      i_poly_xyz_cc(k,j,i) - li_poly_xyz_cc(k,j,i)
    );

    for(int f=0; f<8; ++f)
    {
      err_dcc += std::abs(
        dpoly_xyz_cc(f,k,j,i) - i_dpoly_xyz_cc(f,k,j,i)
      );
    }

    err_dcc_loc += std::abs(
      dpoly_xyz_cc(1,k,j,i) - li_dpoly_xyz_cc(0,k,j,i)
    );

    err_dcc_loc += std::abs(
      dpoly_xyz_cc(2,k,j,i) - li_dpoly_xyz_cc(1,k,j,i)
    );

    err_dcc_loc += std::abs(
      dpoly_xyz_cc(3,k,j,i) - li_dpoly_xyz_cc(2,k,j,i)
    );

  }

  for(int k=vc_kl; k<=vc_ku-1; ++k)
  for(int j=vc_jl; j<=vc_ju-1; ++j)
  for(int i=vc_il; i<=vc_iu-1; ++i)
  {
    err_vc += std::abs(
      i_poly_xyz_vc(k,j,i) - poly_xyz_vc(k,j,i)
    );

    err_vc_loc += std::abs(
      i_poly_xyz_vc(k,j,i) - li_poly_xyz_vc(k,j,i)
    );
  }

  coutBoldBlue("(err_cc, err_vc, err_cc_loc, err_vc_loc, ");
  coutBoldBlue("err_dcc, err_dcc_loc):\n");
  printf("(%1.4e,%1.4e,%1.4e,%1.4e,%1.4e,%1.4e)\n",
    err_cc, err_vc, err_cc_loc, err_vc_loc, err_dcc, err_dcc_loc);

  if(stdout_dump)
  {
    coutBoldRed("poly_xyz_vc:\n");
    poly_xyz_vc.print_all();

    coutBoldRed("i_poly_xyz_vc:\n");
    i_poly_xyz_vc.print_all();

    coutBoldRed("poly_xyz_cc:\n");
    poly_xyz_cc.print_all();

    coutBoldRed("i_poly_xyz_cc:\n");
    i_poly_xyz_cc.print_all();
  }

  // clean-up
  x_vc.DeleteAthenaArray();
  x_cc.DeleteAthenaArray();

  y_vc.DeleteAthenaArray();
  y_cc.DeleteAthenaArray();

  z_vc.DeleteAthenaArray();
  z_cc.DeleteAthenaArray();

  rds.DeleteAthenaArray();

  poly_xyz_vc.DeleteAthenaArray();
  poly_xyz_cc.DeleteAthenaArray();
  dpoly_xyz_cc.DeleteAthenaArray();

  i_poly_xyz_vc.DeleteAthenaArray();
  i_poly_xyz_vc.DeleteAthenaArray();
  i_dpoly_xyz_cc.DeleteAthenaArray();

  li_dpoly_xyz_cc.DeleteAthenaArray();
  li_poly_xyz_vc.DeleteAthenaArray();
  li_poly_xyz_vc.DeleteAthenaArray();

  delete ig;
}

static void test_FC_2d(const bool stdout_dump)
{
  const int N_x = 6, N_y = 8;

  const int sz_x_vc = 2*VC_NGHOST+N_x+1;
  const int sz_x_cc = 2*CC_NGHOST+N_x;

  const int sz_y_vc = 2*VC_NGHOST+N_y+1;
  const int sz_y_cc = 2*CC_NGHOST+N_y;

  const Real x_a = 1.2, x_b = 2.4;
  const Real y_a = 3.2, y_b = 4.4;

  AthenaArray<Real> x_vc = make_ords(x_a, x_b, N_x, VC_NGHOST);
  AthenaArray<Real> x_cc = make_ords_cc(x_a, x_b, N_x, CC_NGHOST);

  AthenaArray<Real> y_vc = make_ords(y_a, y_b, N_y, VC_NGHOST);
  AthenaArray<Real> y_cc = make_ords_cc(y_a, y_b, N_y, CC_NGHOST);

  AthenaArray<Real> rds;
  rds.NewAthenaArray(2);
  rds(0) = 1./(x_cc(1) - x_cc(0));
  rds(1) = 1./(y_cc(1) - y_cc(0));

  AthenaArray<Real> poly_xy_vc, poly_xy_cc, poly_xy_fc0, poly_xy_fc1;
  AthenaArray<Real> i_poly_xy_fc0, i_poly_xy_fc1;

  poly_xy_vc.NewAthenaArray(sz_y_vc, sz_x_vc);
  poly_xy_cc.NewAthenaArray(sz_y_cc, sz_x_cc);

  poly_xy_fc0.NewAthenaArray(sz_y_cc, sz_x_vc);
  poly_xy_fc1.NewAthenaArray(sz_y_vc, sz_x_cc);

  i_poly_xy_fc0.NewAthenaArray(sz_y_cc, sz_x_vc);
  i_poly_xy_fc1.NewAthenaArray(sz_y_vc, sz_x_cc);

  // poly.
  for(int j=0; j<sz_y_vc; ++j)
    for(int i=0; i<sz_x_vc; ++i)
      poly_xy_vc(j,i) = -2*std::pow(x_vc(i), 2.0)*std::pow(y_vc(j), 2.0);

  for(int j=0; j<sz_y_cc; ++j)
    for(int i=0; i<sz_x_cc; ++i)
    {
      poly_xy_cc(j,i) = -2*std::pow(x_cc(i), 2.0)*std::pow(y_cc(j), 2.0);
    }

  for(int j=0; j<sz_y_cc; ++j)
    for(int i=0; i<sz_x_vc; ++i)
    {
      poly_xy_fc0(j,i) = -2*std::pow(x_vc(i), 2.0)*std::pow(y_cc(j), 2.0);
    }

  for(int j=0; j<sz_y_vc; ++j)
    for(int i=0; i<sz_x_cc; ++i)
    {
      poly_xy_fc1(j,i) = -2*std::pow(x_cc(i), 2.0)*std::pow(y_vc(j), 2.0);
    }

  // linear.
  for(int j=0; j<sz_y_vc; ++j)
    for(int i=0; i<sz_x_vc; ++i)
      poly_xy_vc(j,i) = -2*(x_vc(i)-3)*(y_vc(j)+2);

  for(int j=0; j<sz_y_cc; ++j)
    for(int i=0; i<sz_x_cc; ++i)
    {
      poly_xy_cc(j,i) = -2*(x_cc(i)-3)*(y_cc(j)+2);
    }

  for(int j=0; j<sz_y_cc; ++j)
    for(int i=0; i<sz_x_vc; ++i)
    {
      poly_xy_fc0(j,i) = -2*(x_vc(i)-3)*(y_cc(j)+2);
    }

  for(int j=0; j<sz_y_vc; ++j)
    for(int i=0; i<sz_x_cc; ++i)
    {
      poly_xy_fc1(j,i) = -2*(x_cc(i)-3)*(y_vc(j)+2);
    }


  const int dim = 2;
  const int cc_il = CC_NGHOST, cc_iu = sz_x_cc-CC_NGHOST;
  const int cc_jl = CC_NGHOST, cc_ju = sz_y_cc-CC_NGHOST;

  const int vc_il = VC_NGHOST, vc_iu = sz_x_vc-VC_NGHOST;
  const int vc_jl = VC_NGHOST, vc_ju = sz_y_vc-VC_NGHOST;

  // construct test for InterpIntergridLocal ----------------------------------
  int N[] = {N_x, N_y};
  Real rdx[] = {1./(x_vc(1)-x_vc(0)), 1./(y_vc(1)-y_vc(0))};

  InterpIntergridLocal * ig = new InterpIntergridLocal(dim, &N[0], &rdx[0]);

  int dir = 0;

  // note split in iteration idx char
  for(int m=cc_jl; m<=(cc_ju-1); ++m)
  {
#pragma omp simd
    for(int l=vc_il; l<=(vc_iu-1); ++l)
      i_poly_xy_fc0(m,l) = ig->map2d_VC2FC(dir, poly_xy_vc(m,l));
  }

  dir = 1;

  // note split in iteration idx char
  for(int m=vc_jl; m<=(vc_ju-1); ++m)
  {
#pragma omp simd
    for(int l=cc_il; l<=(cc_iu-1); ++l)
      i_poly_xy_fc1(m,l) = ig->map2d_VC2FC(dir, poly_xy_vc(m,l));
  }


  // --------------------------------------------------------------------------


  // compute sum of abs error over all interpolated points
  Real err_fc = 0.0;

  for(int j=cc_jl; j<=cc_ju-1; ++j)
    for(int i=vc_il; i<=vc_iu-1; ++i)
    {
      err_fc += std::abs(
        i_poly_xy_fc0(j,i) - poly_xy_fc0(j,i)
      );

    }

  for(int j=vc_jl; j<=vc_ju-1; ++j)
    for(int i=cc_il; i<=cc_iu-1; ++i)
    {
      err_fc += std::abs(
        i_poly_xy_fc1(j,i) - poly_xy_fc1(j,i)
      );

    }

  coutBoldBlue("(err_fc):\n");
  printf("(%1.4e)\n", err_fc);

  if(stdout_dump)
  {
    coutBoldRed("poly_xy_vc:\n");
    poly_xy_vc.print_all();

    coutBoldRed("poly_xy_fc0:\n");
    poly_xy_fc0.print_all();

    coutBoldRed("i_poly_xy_fc0:\n");
    i_poly_xy_fc0.print_all();

    coutBoldRed("poly_xy_fc1:\n");
    poly_xy_fc1.print_all();

    coutBoldRed("i_poly_xy_fc1:\n");
    i_poly_xy_fc1.print_all();

  }

  // clean-up
  x_vc.DeleteAthenaArray();
  x_cc.DeleteAthenaArray();

  y_vc.DeleteAthenaArray();
  y_cc.DeleteAthenaArray();

  rds.DeleteAthenaArray();

  poly_xy_vc.DeleteAthenaArray();
  poly_xy_cc.DeleteAthenaArray();

  poly_xy_fc0.DeleteAthenaArray();
  poly_xy_fc1.DeleteAthenaArray();

  i_poly_xy_fc0.DeleteAthenaArray();
  i_poly_xy_fc1.DeleteAthenaArray();

  delete ig;
}

static void test_FC_3d(const bool stdout_dump)
{
  const int N_x = 6, N_y = 8, N_z = 4;

  const int sz_x_vc = 2*VC_NGHOST+N_x+1;
  const int sz_x_cc = 2*CC_NGHOST+N_x;

  const int sz_y_vc = 2*VC_NGHOST+N_y+1;
  const int sz_y_cc = 2*CC_NGHOST+N_y;

  const int sz_z_vc = 2*VC_NGHOST+N_z+1;
  const int sz_z_cc = 2*CC_NGHOST+N_z;

  const Real x_a = 1.2, x_b = 2.4;
  const Real y_a = 3.2, y_b = 4.4;
  const Real z_a = 2.2, z_b = 3.7;

  AthenaArray<Real> x_vc = make_ords(x_a, x_b, N_x, VC_NGHOST);
  AthenaArray<Real> x_cc = make_ords_cc(x_a, x_b, N_x, CC_NGHOST);

  AthenaArray<Real> y_vc = make_ords(y_a, y_b, N_y, VC_NGHOST);
  AthenaArray<Real> y_cc = make_ords_cc(y_a, y_b, N_y, CC_NGHOST);

  AthenaArray<Real> z_vc = make_ords(z_a, z_b, N_z, VC_NGHOST);
  AthenaArray<Real> z_cc = make_ords_cc(z_a, z_b, N_z, CC_NGHOST);

  AthenaArray<Real> rds;
  rds.NewAthenaArray(3);
  rds(0) = 1./(x_cc(1) - x_cc(0));
  rds(1) = 1./(y_cc(1) - y_cc(0));
  rds(2) = 1./(z_cc(1) - z_cc(0));

  AthenaArray<Real> poly_xyz_vc, poly_xyz_cc;
  AthenaArray<Real> poly_xyz_fc0, poly_xyz_fc1, poly_xyz_fc2;
  AthenaArray<Real> i_poly_xyz_fc0, i_poly_xyz_fc1, i_poly_xyz_fc2;

  poly_xyz_vc.NewAthenaArray(sz_z_vc, sz_y_vc, sz_x_vc);
  poly_xyz_cc.NewAthenaArray(sz_z_cc, sz_y_cc, sz_x_cc);

  poly_xyz_fc0.NewAthenaArray(sz_z_cc, sz_y_cc, sz_x_vc);
  poly_xyz_fc1.NewAthenaArray(sz_z_cc, sz_y_vc, sz_x_cc);
  poly_xyz_fc2.NewAthenaArray(sz_z_vc, sz_y_cc, sz_x_cc);

  i_poly_xyz_fc0.NewAthenaArray(sz_z_cc, sz_y_cc, sz_x_vc);
  i_poly_xyz_fc1.NewAthenaArray(sz_z_cc, sz_y_vc, sz_x_cc);
  i_poly_xyz_fc2.NewAthenaArray(sz_z_vc, sz_y_cc, sz_x_cc);


  // linear
  for(int k=0; k<sz_z_vc; ++k)
  for(int j=0; j<sz_y_vc; ++j)
  for(int i=0; i<sz_x_vc; ++i)
    poly_xyz_vc(k,j,i) = -2*(x_vc(i)-2)*(y_vc(j)+1)*(z_vc(k)-3);

  for(int k=0; k<sz_z_cc; ++k)
  for(int j=0; j<sz_y_cc; ++j)
  for(int i=0; i<sz_x_cc; ++i)
  {
    poly_xyz_cc(k,j,i) = -2*(x_cc(i)-2)*(y_cc(j)+1)*(z_cc(k)-3);
  }

  for(int k=0; k<sz_z_cc; ++k)
  for(int j=0; j<sz_y_cc; ++j)
  for(int i=0; i<sz_x_vc; ++i)
    poly_xyz_fc0(k,j,i) = -2*(x_vc(i)-2)*(y_cc(j)+1)*(z_cc(k)-3);

  for(int k=0; k<sz_z_cc; ++k)
  for(int j=0; j<sz_y_vc; ++j)
  for(int i=0; i<sz_x_cc; ++i)
    poly_xyz_fc1(k,j,i) = -2*(x_cc(i)-2)*(y_vc(j)+1)*(z_cc(k)-3);

  for(int k=0; k<sz_z_vc; ++k)
  for(int j=0; j<sz_y_cc; ++j)
  for(int i=0; i<sz_x_cc; ++i)
    poly_xyz_fc2(k,j,i) = -2*(x_cc(i)-2)*(y_cc(j)+1)*(z_vc(k)-3);


  const int dim = 3;
  const int cc_il = CC_NGHOST, cc_iu = sz_x_cc-CC_NGHOST;
  const int cc_jl = CC_NGHOST, cc_ju = sz_y_cc-CC_NGHOST;
  const int cc_kl = CC_NGHOST, cc_ku = sz_z_cc-CC_NGHOST;

  const int vc_il = VC_NGHOST, vc_iu = sz_x_vc-VC_NGHOST;
  const int vc_jl = VC_NGHOST, vc_ju = sz_y_vc-VC_NGHOST;
  const int vc_kl = VC_NGHOST, vc_ku = sz_z_vc-VC_NGHOST;

  // construct test for InterpIntergridLocal ----------------------------------
  int N[] = {N_x, N_y, N_z};
  Real rdx[] = {
    1./(x_vc(1)-x_vc(0)), 1./(y_vc(1)-y_vc(0)), 1./(z_vc(1)-z_vc(0))
  };

  InterpIntergridLocal * ig = new InterpIntergridLocal(dim, &N[0], &rdx[0]);

  int dir = 0;

  // note split in iteration idx char
  for(int n=cc_jl; n<=(cc_ju-1); ++n)
    for(int m=cc_jl; m<=(cc_ju-1); ++m)
    {
#pragma omp simd
    for(int l=vc_il; l<=(vc_iu-1); ++l)
      i_poly_xyz_fc0(n,m,l) = ig->map3d_VC2FC(dir, poly_xyz_vc(n,m,l));
    }

  dir = 1;

  // note split in iteration idx char
  for(int n=cc_jl; n<=(cc_ju-1); ++n)
    for(int m=vc_jl; m<=(vc_ju-1); ++m)
    {
#pragma omp simd
    for(int l=cc_il; l<=(cc_iu-1); ++l)
      i_poly_xyz_fc1(n,m,l) = ig->map3d_VC2FC(dir, poly_xyz_vc(n,m,l));
    }

  dir = 2;

  // note split in iteration idx char
  for(int n=vc_jl; n<=(vc_ju-1); ++n)
    for(int m=cc_jl; m<=(cc_ju-1); ++m)
    {
#pragma omp simd
    for(int l=cc_il; l<=(cc_iu-1); ++l)
      i_poly_xyz_fc2(n,m,l) = ig->map3d_VC2FC(dir, poly_xyz_vc(n,m,l));
    }


  // --------------------------------------------------------------------------


  // compute sum of abs error over all interpolated points
  Real err_fc = 0.0;

  for(int k=cc_kl; k<=cc_ku-1; ++k)
  for(int j=cc_jl; j<=cc_ju-1; ++j)
    for(int i=vc_il; i<=vc_iu-1; ++i)
    {
      err_fc += std::abs(
        i_poly_xyz_fc0(k,j,i) - poly_xyz_fc0(k,j,i)
      );

    }

  for(int k=cc_kl; k<=cc_ku-1; ++k)
  for(int j=vc_jl; j<=vc_ju-1; ++j)
    for(int i=cc_il; i<=cc_iu-1; ++i)
    {
      err_fc += std::abs(
        i_poly_xyz_fc1(k,j,i) - poly_xyz_fc1(k,j,i)
      );

    }

  for(int k=vc_kl; k<=vc_ku-1; ++k)
  for(int j=cc_jl; j<=cc_ju-1; ++j)
    for(int i=cc_il; i<=cc_iu-1; ++i)
    {
      err_fc += std::abs(
        i_poly_xyz_fc2(k,j,i) - poly_xyz_fc2(k,j,i)
      );

    }

  coutBoldBlue("(err_fc):\n");
  printf("(%1.4e)\n", err_fc);

  // clean-up
  x_vc.DeleteAthenaArray();
  x_cc.DeleteAthenaArray();

  y_vc.DeleteAthenaArray();
  y_cc.DeleteAthenaArray();

  z_vc.DeleteAthenaArray();
  z_cc.DeleteAthenaArray();

  rds.DeleteAthenaArray();

  poly_xyz_vc.DeleteAthenaArray();
  poly_xyz_cc.DeleteAthenaArray();

  poly_xyz_fc0.DeleteAthenaArray();
  poly_xyz_fc1.DeleteAthenaArray();
  poly_xyz_fc2.DeleteAthenaArray();

  i_poly_xyz_fc0.DeleteAthenaArray();
  i_poly_xyz_fc1.DeleteAthenaArray();
  i_poly_xyz_fc2.DeleteAthenaArray();

  delete ig;
}

int main(int argc, char *argv[]) {


  // testing(); // init testing stuff (such as random seed etc)

  test_1d(false);
  test_2d(false);
  test_3d(false);

  // face centered
  test_FC_2d(false);
  test_FC_3d(false);

  return (0);
}