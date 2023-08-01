//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wave_test.cpp
//  \brief Initial conditions for the wave equation

#include <cassert> // assert
#include <cmath> // abs, exp, sin, fmod
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "../wave/wave.hpp"

using namespace std;

int RefinementCondition(MeshBlock *pmb);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Initialize the problem.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  int il = pwave->mbi.il, iu = pwave->mbi.iu;
  int kl = pwave->mbi.kl, ku = pwave->mbi.ku;
  int jl = pwave->mbi.jl, ju = pwave->mbi.ju;

  Real c = pwave->c;
  const Real sig = 1. / (4. * std::sqrt(5.));

  for(int k = kl; k <= ku; ++k)
    for(int j = jl; j <= ju; ++j)
      for(int i = il; i <= iu; ++i) {

        Real x = pwave->mbi.x1(i);
        Real y = pwave->mbi.x2(j);
        Real z = pwave->mbi.x3(k);

        /*
        // Real cos_x = cos(PI*x);
        Real cos_2x = cos(2.*PI*x);
        Real sqr_cos_2x = SQR(cos_2x);
        // Real sqr_sin_x = SQR(sin(PI*x));

        pwave->u(0,k,j,i) = sqr_cos_2x;
        pwave->u(1,k,j,i) = -cos_2x / 2.;
        */

        // pwave->u(0,k,j,i) = (
        //   2. -
        //   4. * std::exp(-(SQR(x) / (2. * SQR(sig)))) +
        //   1. / 2. * std::cos(PI * x)
        // );

        // pwave->u(1,k,j,i) = (
        //   -320. * x * std::exp(-(SQR(x) / (2. * SQR(sig)))) +
        //   PI / 2. * std::sin(PI * x)
        // );

        pwave->u(0,k,j,i) = (
          2. +
          1. / 2. * std::cos(PI * x) -
          4. * std::exp(-40 * SQR(x))
        );

        pwave->u(1,k,j,i) = (
          PI / 2. * std::sin(PI * x) -
          320. * x * std::exp(-SQR(x) / (2. * SQR(sig)))
        );


        pwave->exact(k,j,i) = pwave->u(0,k,j,i);
        pwave->error(k,j,i) = 0.0;
      }


  return;
}

void MeshBlock::WaveUserWorkInLoop() {
  Real max_err = 0;
  Real fun_max = 0;

  Real c = pwave->c;
  const Real sig = 1. / (4. * std::sqrt(5.));

  Real t = pmy_mesh->time + pmy_mesh->dt;
  bool const debug_inspect_error = pwave->debug_inspect_error;
  Real const debug_abort_threshold = pwave->debug_abort_threshold;

  int il = pwave->mbi.il, iu = pwave->mbi.iu;
  int jl = pwave->mbi.jl, ju = pwave->mbi.ju;
  int kl = pwave->mbi.kl, ku = pwave->mbi.ku;

  for(int k = kl; k <= ku; ++k)
    for(int j = jl; j <= ju; ++j)
      for(int i = il; i <= iu; ++i) {

        Real x = pwave->mbi.x1(i);
        Real y = pwave->mbi.x2(j);
        Real z = pwave->mbi.x3(k);

        // Real cos_2x = cos(2.*PI*x);
        // Real cos_4x = cos(4.*PI*x);
        // Real cos_4ct = cos(4.*PI*c*t);
        // Real sin_2ct = sin(2.*PI*c*t);

        // pwave->u(0,k,j,i) = i + 1 + 100 * gid;
        // pwave->u(1,k,j,i) = 0;
        /*
        pwave->exact(k,j,i) = (2. + 2. * cos_4ct * cos_4x -
                               cos_2x * sin_2ct / ( c * PI )) / 4.;

        pwave->error(k,j,i) = pwave->u(0,k,j,i) - pwave->exact(k,j,i);
        */

        // pwave->exact(k,j,i) = (
        //   2. -
        //   4. * std::exp(-(SQR(x-t) / (2. * SQR(sig)))) +
        //   1. / 2. * std::cos(PI * (x-t))
        // );

        Real exp_arg = std::asin(std::sin(
          PI / 2. * (x-t)
        )) * 2. / PI;

        pwave->exact(k,j,i) = (
          2. +
          1. / 2. * std::cos(PI * (x-t)) -
          4. * std::exp(-SQR(exp_arg) / (2. * SQR(sig)))
        );

        pwave->error(k,j,i) = pwave->u(0,k,j,i) - pwave->exact(k,j,i);


        if (std::abs(pwave->error(k,j,i)) > max_err){
          max_err = std::abs(pwave->error(k,j,i));
          fun_max = pwave->u(0,k,j,i);
        }
      }

  // AthenaArray<Real> dbg;
  // dbg.NewAthenaArray(4);
  // dbg(0) = gid;
  // dbg(1) = pwave->mbi.x1(il);
  // dbg(2) = pwave->mbi.x1(iu);
  // dbg(3) = dbg(2) - dbg(1);

  if (debug_inspect_error) {
    printf(">>>\n");
    // coutBoldRed("MB::UWIL gid = ");
    printf("%d\n", gid);
    printf("(max_err, fun_max, t; x_a, x_b)=(%1.18f, %1.18f, %1.18f; %1.4f, %1.4f, %1.1e)\n",
          max_err, fun_max, t, pwave->mbi.x1(il), pwave->mbi.x1(iu),
          pwave->mbi.x1(il+1) - pwave->mbi.x1(il));

    if (max_err > debug_abort_threshold) {
      printf("pwave->u:\n");
      pwave->u.print_all("%1.5f");

      printf("pwave->exact:\n");
      pwave->exact.print_all("%1.5f");

      printf("pwave->error:\n");
      pwave->error.print_all("%1.5f");

    }

    printf("pwave->u:\n");
    pwave->u.print_all("%1.2f");

    printf("pwave->error:\n");
    pwave->error.print_all("%1.1e");
    printf("<<<\n");
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn
//  \brief refinement condition: simple time-dependent test

int RefinementCondition(MeshBlock *pmb){

  const Real ref_dx = 0.2;
  const Real t = pmb->pmy_mesh->time + pmb->pmy_mesh->dt;

  // consider wrapped t as the refinement region centre
  const Real ref_x_0 = std::asin(std::sin(PI / 2. * t)) * 2. / PI;
  const Real ref_x_l = ref_x_0 - ref_dx;
  const Real ref_x_r = ref_x_0 + ref_dx;

  // std::cout << ref_x_0 << std::endl;

  // check whether range formed by extent of this MeshBlock overlaps with
  // ref_x_0 + [-ref_dx, ref_dx]
  const Real Mb_x_l = pmb->pcoord->x1f(0);
  const Real Mb_x_r = pmb->pcoord->x1f(pmb->ncells1-1);

  bool overlapping = std::max(ref_x_l, Mb_x_l) <= std::min(ref_x_r, Mb_x_r);

  if (overlapping)
  {
    // std::cout << "ref: " << pmb->gid << std::endl;
    return 1;
  }
  else
  {
    // std::cout << "deref: " << pmb->gid << std::endl;
    return -1;
  }

}
