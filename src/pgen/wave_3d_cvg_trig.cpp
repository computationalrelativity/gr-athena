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

  for(int k = kl; k <= ku; ++k)
    for(int j = jl; j <= ju; ++j)
#pragma omp simd
      for(int i = il; i <= iu; ++i) {
        Real x = pwave->mbi.x1(i);
        Real y = pwave->mbi.x2(j);
        Real z = pwave->mbi.x3(k);

        Real cos_2x = cos(2 * PI * x);
        Real cos_y = cos(PI * y);
        Real cos_3z = cos(3 * PI * z);

        pwave->u(0,k,j,i) = SQR(cos_2x) * cos_y * cos_3z;
        pwave->u(1,k,j,i) = 0.;

        // pwave->u(0,k,j,i) = 1 + i + 100 * j + 1000 * k + 10000 * (gid + 1);
        // pwave->u(1,k,j,i) = -(pwave->u(0,k,j,i));


        pwave->exact(k,j,i) = pwave->u(0,k,j,i);
        pwave->error(k,j,i) = 0.0;
      }

  return;
}

void MeshBlock::WaveUserWorkInLoop() {
  Real max_err = 0;
  Real fun_err = 0;

  // int il = pwave->mbi.il, iu = pwave->mbi.iu;
  // int kl = pwave->mbi.kl, ku = pwave->mbi.ku;
  // int jl = pwave->mbi.jl, ju = pwave->mbi.ju;

  Real c = pwave->c;
  Real t = pmy_mesh->time + pmy_mesh->dt;

  int il = pwave->mbi.il, iu = pwave->mbi.iu;
  int jl = pwave->mbi.jl, ju = pwave->mbi.ju;
  int kl = pwave->mbi.kl, ku = pwave->mbi.ku;

  Real cos_s10t = cos(sqrt(10.) * c * PI * t);
  Real cos_s26t = cos(sqrt(26.) * c * PI * t);

  for(int k = kl; k <= ku; ++k){
    for(int j = jl; j <= ju; ++j){
#pragma omp simd
      for(int i = il; i <= iu; ++i) {
        Real x = pwave->mbi.x1(i);
        Real y = pwave->mbi.x2(j);
        Real z = pwave->mbi.x3(k);

        Real cos_4x = cos(4 * PI * x);
        Real cos_y = cos(PI * y);
        Real cos_3z = cos(3 * PI * z);


        pwave->exact(k,j,i) = 1. / 2. * (cos_s10t + cos_s26t * cos_4x) *
          cos_y * cos_3z;
        pwave->error(k,j,i) = pwave->u(0,k,j,i) - pwave->exact(k,j,i);

        if (std::abs(pwave->error(k,j,i)) > max_err){
          max_err = std::abs(pwave->error(k,j,i));
          fun_err = pwave->u(0,k,j,i);
        }
      }
    }
  }

  // printf(">>>\n");

  // printf("MB::UWIL gid = ");
  // printf("%d\n", gid);
  // printf("(max_err, fun_max, t)=(%1.18f, %1.18f, %1.18f)\n",
  //        max_err, fun_err, t);

  // if (max_err > 0.1) {
  //   printf("pwave->u:\n");
  //   pwave->u.print_all("%1.2f");

  //   printf("pwave->exact:\n");
  //   pwave->exact.print_all("%1.2f");

  //   printf("pwave->error:\n");
  //   pwave->error.print_all("%1.2f");

  // }

  // printf("<<<\n");
  return;
}

//----------------------------------------------------------------------------------------
//! \fn
//  \brief refinement condition: simple time-dependent test

int RefinementCondition(MeshBlock *pmb){

  const Real ref_dx = 0.2;
  const Real t = pmb->pmy_mesh->time + pmb->pmy_mesh->dt;

  // consider wrapped t as the refinement region centre
  const Real ref_x1_0 = std::asin(std::sin(PI / 2. * t / SQRT2)) * 2. / PI;
  const Real ref_x1_l = ref_x1_0 - ref_dx;
  const Real ref_x1_r = ref_x1_0 + ref_dx;


  // check whether range formed by extent of this MeshBlock overlaps with
  // ref_x1_0 + [-ref_dx, ref_dx]
  const Real Mb_x1_l = pmb->pcoord->x1f(0);
  const Real Mb_x1_r = pmb->pcoord->x1f(pmb->ncells1-1);

  bool ol_x1 = std::max(ref_x1_l, Mb_x1_l) <= std::min(ref_x1_r, Mb_x1_r);

  const Real ref_x2_0 = std::asin(std::sin(PI / 2. * t / SQRT2)) * 2. / PI;
  const Real ref_x2_l = ref_x2_0 - ref_dx;
  const Real ref_x2_r = ref_x2_0 + ref_dx;

  const Real Mb_x2_l = pmb->pcoord->x2f(0);
  const Real Mb_x2_r = pmb->pcoord->x2f(pmb->ncells2-1);

  bool ol_x2 = std::max(ref_x2_l, Mb_x2_l) <= std::min(ref_x2_r, Mb_x2_r);


  const Real ref_x3_0 = std::asin(std::sin(PI / 2. * t / SQRT2)) * 2. / PI;
  const Real ref_x3_l = ref_x3_0 - ref_dx;
  const Real ref_x3_r = ref_x3_0 + ref_dx;

  const Real Mb_x3_l = pmb->pcoord->x3f(0);
  const Real Mb_x3_r = pmb->pcoord->x3f(pmb->ncells2-1);

  bool ol_x3 = std::max(ref_x3_l, Mb_x3_l) <= std::min(ref_x3_r, Mb_x3_r);

  if (ol_x1 && ol_x2 && ol_x3)
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