//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file mesh_refinement.cpp
//  \brief implements functions for static/adaptive mesh refinement

// C headers

// C++ headers
#include <algorithm>  // max()
#include <cmath>
#include <cstring>  // strcmp()
#include <sstream>
#include <stdexcept>  // runtime_error
#include <string>
#include <tuple>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../parameter_input.hpp"
#include "../utils/floating_point.hpp"
#include "../utils/interp_univariate.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"

//----------------------------------------------------------------------------------------
//! \fn MeshRefinement::MeshRefinement(MeshBlock *pmb, ParameterInput *pin)
//  \brief constructor

MeshRefinement::MeshRefinement(MeshBlock* pmb, ParameterInput* pin)
    : pmy_block_(pmb),
      deref_count_(0),
      deref_threshold_(pin->GetOrAddInteger("mesh", "derefine_count", 10)),
      AMRFlag_(pmb->pmy_mesh->AMRFlag_)
{
  // Create coarse mesh object for parent grid
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0)
  {
    pcoarsec = new Cartesian(pmb, pin, true);
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0)
  {
    pcoarsec = new Cylindrical(pmb, pin, true);
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0)
  {
    pcoarsec = new SphericalPolar(pmb, pin, true);
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar_uniform") == 0)
  {
    pcoarsec = new SphericalPolarUniform(pmb, pin, true);
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "minkowski") == 0)
  {
    pcoarsec = new Minkowski(pmb, pin, true);
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "schwarzschild") == 0)
  {
    pcoarsec = new Schwarzschild(pmb, pin, true);
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "kerr-schild") == 0)
  {
    pcoarsec = new KerrSchild(pmb, pin, true);
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "gr_dynamical") == 0)
  {
    pcoarsec = new GRDynamical(pmb, pin, true);
  }

  int nc1 = pmb->ncells1;
  fvol_[0][0].NewAthenaArray(nc1);
  fvol_[0][1].NewAthenaArray(nc1);
  fvol_[1][0].NewAthenaArray(nc1);
  fvol_[1][1].NewAthenaArray(nc1);
  sarea_x1_[0][0].NewAthenaArray(nc1 + 1);
  sarea_x1_[0][1].NewAthenaArray(nc1 + 1);
  sarea_x1_[1][0].NewAthenaArray(nc1 + 1);
  sarea_x1_[1][1].NewAthenaArray(nc1 + 1);
  sarea_x2_[0][0].NewAthenaArray(nc1);
  sarea_x2_[0][1].NewAthenaArray(nc1);
  sarea_x2_[0][2].NewAthenaArray(nc1);
  sarea_x2_[1][0].NewAthenaArray(nc1);
  sarea_x2_[1][1].NewAthenaArray(nc1);
  sarea_x2_[1][2].NewAthenaArray(nc1);
  sarea_x3_[0][0].NewAthenaArray(nc1);
  sarea_x3_[0][1].NewAthenaArray(nc1);
  sarea_x3_[1][0].NewAthenaArray(nc1);
  sarea_x3_[1][1].NewAthenaArray(nc1);
  sarea_x3_[2][0].NewAthenaArray(nc1);
  sarea_x3_[2][1].NewAthenaArray(nc1);

  // --------------------------------------------------------------------------
  // init interpolation op based on underlying dimensionality
  Coordinates* pco = pmb->pcoord;

  /*
  int si, sj, sk, ei, ej, ek;
  si = pmb->cx_cis; ei = pmb->cx_cie;
  sj = pmb->cx_cjs; ej = pmb->cx_cje;
  sk = pmb->cx_cks; ek = pmb->cx_cke;
  */

  const int Ns_x3 = pmb->block_size.nx3 - 1;  // # phys. nodes - 1
  const int Ns_x2 = pmb->block_size.nx2 - 1;  // # phys. nodes - 1
  const int Ns_x1 = pmb->block_size.nx1 - 1;

  // Floater-Hormann blending parameter controls the formal order of approx.
  const int d = (NCGHOST_CX - 1) * 2 - 1;

#if defined(DBG_CX_ALL_BARY_RP)
  if (Ns_x3 > 0)
  {
    x1c_cx = linspace(pcoarsec->x1v(pmb->cx_cis),
                      pcoarsec->x1v(pmb->cx_cie),
                      pmb->block_size.nx1 / 2,
                      false,
                      NCGHOST_CX);

    x2c_cx = linspace(pcoarsec->x2v(pmb->cx_cis),
                      pcoarsec->x2v(pmb->cx_cie),
                      pmb->block_size.nx2 / 2,
                      false,
                      NCGHOST_CX);

    x3c_cx = linspace(pcoarsec->x3v(pmb->cx_cis),
                      pcoarsec->x3v(pmb->cx_cie),
                      pmb->block_size.nx3 / 2,
                      false,
                      NCGHOST_CX);

    x1f_cx = linspace(pco->x1v(pmb->cx_is),
                      pco->x1v(pmb->cx_ie),
                      pmb->block_size.nx1,
                      false,
                      NGHOST);

    x2f_cx = linspace(pco->x2v(pmb->cx_is),
                      pco->x2v(pmb->cx_ie),
                      pmb->block_size.nx2,
                      false,
                      NGHOST);

    x3f_cx = linspace(pco->x3v(pmb->cx_is),
                      pco->x3v(pmb->cx_ie),
                      pmb->block_size.nx3,
                      false,
                      NGHOST);

    ind_physical_p_op = new interp_nd(x1f_cx,
                                      x2f_cx,
                                      x3f_cx,
                                      x1c_cx,
                                      x2c_cx,
                                      x3c_cx,
                                      pmb->ncells1 - 1,
                                      pmb->ncells2 - 1,
                                      pmb->ncells3 - 1,
                                      pmb->cx_ncc1 - 1,
                                      pmb->cx_ncc2 - 1,
                                      pmb->cx_ncc3 - 1,
                                      d,
                                      0,
                                      0);

    ind_physical_r_op = new interp_nd(x1c_cx,
                                      x2c_cx,
                                      x3c_cx,
                                      x1f_cx,
                                      x2f_cx,
                                      x3f_cx,
                                      pmb->cx_ncc1 - 1,
                                      pmb->cx_ncc2 - 1,
                                      pmb->cx_ncc3 - 1,
                                      pmb->ncells1 - 1,
                                      pmb->ncells2 - 1,
                                      pmb->ncells3 - 1,
                                      d,
                                      0,
                                      0);
  }
#endif  // DBG_CX_ALL_BARY_RP

  if (Ns_x3 > 0)
  {
    Real* x1_s = &(pco->x1v(NGHOST));
    Real* x2_s = &(pco->x2v(NGHOST));
    Real* x3_s = &(pco->x3v(NGHOST));

    Real* x1_t = &(pcoarsec->x1v(NCGHOST + (pmb->cx_cis - NCGHOST_CX)));
    Real* x2_t = &(pcoarsec->x2v(NCGHOST + (pmb->cx_cjs - NCGHOST_CX)));
    Real* x3_t = &(pcoarsec->x3v(NCGHOST + (pmb->cx_cks - NCGHOST_CX)));

    const int Nt_x1 = pmb->block_size.nx1 / 2 - 1;
    const int Nt_x2 = pmb->block_size.nx2 / 2 - 1;
    const int Nt_x3 = pmb->block_size.nx3 / 2 - 1;

    ind_interior_r_op = new interp_nd(x1_t,
                                      x2_t,
                                      x3_t,
                                      x1_s,
                                      x2_s,
                                      x3_s,
                                      Nt_x1,
                                      Nt_x2,
                                      Nt_x3,
                                      Ns_x1,
                                      Ns_x2,
                                      Ns_x3,
                                      d,
                                      NCGHOST_CX,
                                      NGHOST);
  }
  else if (Ns_x2 > 0)
  {
    Real* x1_s = &(pco->x1v(NGHOST));
    Real* x2_s = &(pco->x2v(NGHOST));

    Real* x1_t = &(pcoarsec->x1v(NCGHOST + (pmb->cx_cis - NCGHOST_CX)));
    Real* x2_t = &(pcoarsec->x2v(NCGHOST + (pmb->cx_cjs - NCGHOST_CX)));

    const int Nt_x1 = pmb->block_size.nx1 / 2 - 1;
    const int Nt_x2 = pmb->block_size.nx2 / 2 - 1;

    ind_interior_r_op = new interp_nd(x1_t,
                                      x2_t,
                                      x1_s,
                                      x2_s,
                                      Nt_x1,
                                      Nt_x2,
                                      Ns_x1,
                                      Ns_x2,
                                      d,
                                      NCGHOST_CX,
                                      NGHOST);
  }
  else
  {
    Real* x1_s = &(pco->x1v(NGHOST));
    Real* x1_t = &(pcoarsec->x1v(NCGHOST + (pmb->cx_cis - NCGHOST_CX)));

    const int Nt_x1 = pmb->block_size.nx1 / 2 - 1;

    ind_interior_r_op =
      new interp_nd(x1_t, x1_s, Nt_x1, Ns_x1, d, NCGHOST_CX, NGHOST);
  }
  // --------------------------------------------------------------------------
}

//----------------------------------------------------------------------------------------
//! \fn MeshRefinement::~MeshRefinement()
//  \brief destructor

MeshRefinement::~MeshRefinement()
{
  delete pcoarsec;
  delete ind_interior_r_op;

#if defined(DBG_CX_ALL_BARY_RP)
  delete ind_physical_r_op;
  delete ind_physical_p_op;

  delete[] x1c_cx;
  delete[] x2c_cx;
  delete[] x3c_cx;

  delete[] x1f_cx;
  delete[] x2f_cx;
  delete[] x3f_cx;
#endif  // DBG_CX_ALL_BARY_RP
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::CheckRefinementCondition()
//  \brief Check refinement criteria

void MeshRefinement::CheckRefinementCondition()
{
  MeshBlock* pmb = pmy_block_;
  int ret = 0, aret = -1;
  refine_flag_ = 0;

  // *** should be implemented later ***
  // loop-over refinement criteria
  if (AMRFlag_ != nullptr)
    ret = AMRFlag_(pmb);
  aret = std::max(aret, ret);

  if (aret >= 0)
    deref_count_ = 0;
  if (aret > 0)
  {
    if (pmb->loc.level == pmb->pmy_mesh->max_level)
    {
      refine_flag_ = 0;
    }
    else
    {
      refine_flag_ = 1;
    }
  }
  else if (aret < 0)
  {
    if (pmb->loc.level == pmb->pmy_mesh->root_level)
    {
      refine_flag_ = 0;
      deref_count_ = 0;
    }
    else
    {
      deref_count_++;
      int ec = 0, js, je, ks, ke;
      if (pmb->block_size.nx2 > 1)
      {
        js = -1;
        je = 1;
      }
      else
      {
        js = 0;
        je = 0;
      }
      if (pmb->block_size.nx3 > 1)
      {
        ks = -1;
        ke = 1;
      }
      else
      {
        ks = 0;
        ke = 0;
      }
      for (int k = ks; k <= ke; k++)
      {
        for (int j = js; j <= je; j++)
        {
          for (int i = -1; i <= 1; i++)
            if (pmb->nc().neighbor_level(i, j, k) > pmb->loc.level)
              ec++;
        }
      }
      if (ec > 0)
      {
        refine_flag_ = 0;
      }
      else
      {
        if (deref_count_ >= deref_threshold_)
        {
          refine_flag_ = -1;
        }
        else
        {
          refine_flag_ = 0;
        }
      }
    }
  }
  return;
}
