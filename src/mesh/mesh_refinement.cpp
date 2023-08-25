//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mesh_refinement.cpp
//  \brief implements functions for static/adaptive mesh refinement

// C headers

// C++ headers
#include <algorithm>   // max()
#include <cmath>
#include <cstring>     // strcmp()
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
#include "mesh.hpp"
#include "mesh_refinement.hpp"

#include "../utils/floating_point.hpp"
#include "../utils/interp_univariate.hpp"

//----------------------------------------------------------------------------------------
//! \fn MeshRefinement::MeshRefinement(MeshBlock *pmb, ParameterInput *pin)
//  \brief constructor

MeshRefinement::MeshRefinement(MeshBlock *pmb, ParameterInput *pin) :
    pmy_block_(pmb), deref_count_(0),
    deref_threshold_(pin->GetOrAddInteger("mesh", "derefine_count", 10)),
    AMRFlag_(pmb->pmy_mesh->AMRFlag_) {

  // Create coarse mesh object for parent grid
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    pcoarsec = new Cartesian(pmb, pin, true);
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    pcoarsec = new Cylindrical(pmb, pin, true);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    pcoarsec = new SphericalPolar(pmb, pin, true);
  } else if (std::strcmp(COORDINATE_SYSTEM, "minkowski") == 0) {
    pcoarsec = new Minkowski(pmb, pin, true);
  } else if (std::strcmp(COORDINATE_SYSTEM, "schwarzschild") == 0) {
    pcoarsec = new Schwarzschild(pmb, pin, true);
  } else if (std::strcmp(COORDINATE_SYSTEM, "kerr-schild") == 0) {
    pcoarsec = new KerrSchild(pmb, pin, true);
  } else if (std::strcmp(COORDINATE_SYSTEM, "gr_user") == 0) {
    pcoarsec = new GRUser(pmb, pin, true);
  }

  int nc1 = pmb->ncells1;
  fvol_[0][0].NewAthenaArray(nc1);
  fvol_[0][1].NewAthenaArray(nc1);
  fvol_[1][0].NewAthenaArray(nc1);
  fvol_[1][1].NewAthenaArray(nc1);
  sarea_x1_[0][0].NewAthenaArray(nc1+1);
  sarea_x1_[0][1].NewAthenaArray(nc1+1);
  sarea_x1_[1][0].NewAthenaArray(nc1+1);
  sarea_x1_[1][1].NewAthenaArray(nc1+1);
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

  // KGF: probably don't need to preallocate space for pointers in these vectors
  pvars_cc_.reserve(3);
  pvars_fc_.reserve(3);
  pvars_vc_.reserve(3);
  pvars_cx_.reserve(3);

  // --------------------------------------------------------------------------
  // init interpolation op based on underlying dimensionality
  Coordinates* pco = pmb->pcoord;

  int si, sj, sk, ei, ej, ek;
  si = pmb->cx_cis; ei = pmb->cx_cie;
  sj = pmb->cx_cjs; ej = pmb->cx_cje;
  sk = pmb->cx_cks; ek = pmb->cx_cke;

  const int Ns_x3 = pmb->block_size.nx3 - 1;  // # phys. nodes - 1
  const int Ns_x2 = pmb->block_size.nx2 - 1;  // # phys. nodes - 1
  const int Ns_x1 = pmb->block_size.nx1 - 1;

  // Floater-Hormann blending parameter controls the formal order of approx.
  const int d = (NCGHOST_CX-1) * 2 - 1;

#if defined(DBG_CX_ALL_BARY_RP)
  if(Ns_x3 > 0)
  {
    x1c_cx = linspace(
      pcoarsec->x1v(pmb->cx_cis),
      pcoarsec->x1v(pmb->cx_cie),
      pmb->block_size.nx1 / 2,
      false,
      NCGHOST_CX
    );

    x2c_cx = linspace(
      pcoarsec->x2v(pmb->cx_cis),
      pcoarsec->x2v(pmb->cx_cie),
      pmb->block_size.nx2 / 2,
      false,
      NCGHOST_CX
    );

    x3c_cx = linspace(
      pcoarsec->x3v(pmb->cx_cis),
      pcoarsec->x3v(pmb->cx_cie),
      pmb->block_size.nx3 / 2,
      false,
      NCGHOST_CX
    );

    x1f_cx = linspace(
      pco->x1v(pmb->cx_is),
      pco->x1v(pmb->cx_ie),
      pmb->block_size.nx1,
      false,
      NGHOST
    );

    x2f_cx = linspace(
      pco->x2v(pmb->cx_is),
      pco->x2v(pmb->cx_ie),
      pmb->block_size.nx2,
      false,
      NGHOST
    );

    x3f_cx = linspace(
      pco->x3v(pmb->cx_is),
      pco->x3v(pmb->cx_ie),
      pmb->block_size.nx3,
      false,
      NGHOST
    );

    ind_physical_p_op = new interp_nd(
      x1f_cx, x2f_cx, x3f_cx,
      x1c_cx, x2c_cx, x3c_cx,
      pmb->ncells1-1, pmb->ncells2-1, pmb->ncells3-1,
      pmb->cx_ncc1-1, pmb->cx_ncc2-1, pmb->cx_ncc3-1,
      d,
      0, 0
    );

    ind_physical_r_op = new interp_nd(
      x1c_cx, x2c_cx, x3c_cx,
      x1f_cx, x2f_cx, x3f_cx,
      pmb->cx_ncc1-1, pmb->cx_ncc2-1, pmb->cx_ncc3-1,
      pmb->ncells1-1, pmb->ncells2-1, pmb->ncells3-1,
      d,
      0, 0
    );


  }
#endif // DBG_CX_ALL_BARY_RP

  if(Ns_x3 > 0)
  {
    Real* x1_s = &(pco->x1v(NGHOST));
    Real* x2_s = &(pco->x2v(NGHOST));
    Real* x3_s = &(pco->x3v(NGHOST));

    Real* x1_t = &(pcoarsec->x1v(NCGHOST+(pmb->cx_cis-NCGHOST_CX)));
    Real* x2_t = &(pcoarsec->x2v(NCGHOST+(pmb->cx_cjs-NCGHOST_CX)));
    Real* x3_t = &(pcoarsec->x3v(NCGHOST+(pmb->cx_cks-NCGHOST_CX)));

    const int Nt_x1 = pmb->block_size.nx1 / 2 - 1;
    const int Nt_x2 = pmb->block_size.nx2 / 2 - 1;
    const int Nt_x3 = pmb->block_size.nx3 / 2 - 1;

    ind_interior_r_op = new interp_nd(
      x1_t, x2_t, x3_t,
      x1_s, x2_s, x3_s,
      Nt_x1, Nt_x2, Nt_x3,
      Ns_x1, Ns_x2, Ns_x3,
      d,
      NCGHOST_CX, NGHOST
    );
  }
  else if(Ns_x2 > 0)
  {
    Real* x1_s = &(pco->x1v(NGHOST));
    Real* x2_s = &(pco->x2v(NGHOST));

    Real* x1_t = &(pcoarsec->x1v(NCGHOST+(pmb->cx_cis-NCGHOST_CX)));
    Real* x2_t = &(pcoarsec->x2v(NCGHOST+(pmb->cx_cjs-NCGHOST_CX)));

    const int Nt_x1 = pmb->block_size.nx1 / 2 - 1;
    const int Nt_x2 = pmb->block_size.nx2 / 2 - 1;

    ind_interior_r_op = new interp_nd(
      x1_t, x2_t,
      x1_s, x2_s,
      Nt_x1, Nt_x2,
      Ns_x1, Ns_x2,
      d,
      NCGHOST_CX, NGHOST
    );
  }
  else
  {
    Real* x1_s = &(pco->x1v(NGHOST));
    Real* x1_t = &(pcoarsec->x1v(NCGHOST+(pmb->cx_cis-NCGHOST_CX)));

    const int Nt_x1 = pmb->block_size.nx1 / 2 - 1;

    ind_interior_r_op = new interp_nd(
      x1_t, x1_s,
      Nt_x1, Ns_x1,
      d,
      NCGHOST_CX, NGHOST
    );
  }
  // --------------------------------------------------------------------------

}


//----------------------------------------------------------------------------------------
//! \fn MeshRefinement::~MeshRefinement()
//  \brief destructor

MeshRefinement::~MeshRefinement() {
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
#endif // DBG_CX_ALL_BARY_RP
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictCellCenteredValues(const AthenaArray<Real> &fine,
//                           AthenaArray<Real> &coarse, int sn, int en,
//                           int csi, int cei, int csj, int cej, int csk, int cek)
//  \brief restrict cell centered values

void MeshRefinement::RestrictCellCenteredValues(
    const AthenaArray<Real> &fine, AthenaArray<Real> &coarse, int sn, int en,
    int csi, int cei, int csj, int cej, int csk, int cek) {

  MeshBlock *pmb = pmy_block_;
  Coordinates *pco = pmb->pcoord;
  int si = (csi - pmb->cis)*2 + pmb->is, ei = (cei - pmb->cis)*2 + pmb->is + 1;

  // store the restricted data in the prolongation buffer for later use
  if (pmb->block_size.nx3>1) { // 3D
    for (int n=sn; n<=en; ++n) {
      for (int ck=csk; ck<=cek; ck++) {
        int k = (ck - pmb->cks)*2 + pmb->ks;
        for (int cj=csj; cj<=cej; cj++) {
          int j = (cj - pmb->cjs)*2 + pmb->js;
          pco->CellVolume(k,j,si,ei,fvol_[0][0]);
          pco->CellVolume(k,j+1,si,ei,fvol_[0][1]);
          pco->CellVolume(k+1,j,si,ei,fvol_[1][0]);
          pco->CellVolume(k+1,j+1,si,ei,fvol_[1][1]);
          for (int ci=csi; ci<=cei; ci++) {
            int i = (ci - pmb->cis)*2 + pmb->is;
            // KGF: add the off-centered quantities first to preserve FP symmetry
            Real tvol = ((fvol_[0][0](i) + fvol_[0][1](i))
                         + (fvol_[0][0](i+1) + fvol_[0][1](i+1)))
                        + ((fvol_[1][0](i) + fvol_[1][1](i))
                           + (fvol_[1][0](i+1) + fvol_[1][1](i+1)));
            // KGF: add the off-centered quantities first to preserve FP symmetry
            coarse(n,ck,cj,ci) =
                (((fine(n,k  ,j  ,i)*fvol_[0][0](i) + fine(n,k  ,j+1,i)*fvol_[0][1](i))
                  + (fine(n,k  ,j  ,i+1)*fvol_[0][0](i+1) +
                     fine(n,k  ,j+1,i+1)*fvol_[0][1](i+1)))
                 + ((fine(n,k+1,j  ,i)*fvol_[1][0](i) + fine(n,k+1,j+1,i)*fvol_[1][1](i))
                    + (fine(n,k+1,j  ,i+1)*fvol_[1][0](i+1) +
                       fine(n,k+1,j+1,i+1)*fvol_[1][1](i+1)))) / tvol;
          }
        }
      }
    }
  } else if (pmb->block_size.nx2>1) { // 2D
    for (int n=sn; n<=en; ++n) {
      for (int cj=csj; cj<=cej; cj++) {
        int j = (cj - pmb->cjs)*2 + pmb->js;
        pco->CellVolume(0,j  ,si,ei,fvol_[0][0]);
        pco->CellVolume(0,j+1,si,ei,fvol_[0][1]);
        for (int ci=csi; ci<=cei; ci++) {
          int i = (ci - pmb->cis)*2 + pmb->is;
          // KGF: add the off-centered quantities first to preserve FP symmetry
          Real tvol = (fvol_[0][0](i) + fvol_[0][1](i)) +
                      (fvol_[0][0](i+1) + fvol_[0][1](i+1));

          // KGF: add the off-centered quantities first to preserve FP symmetry
          coarse(n,0,cj,ci) =
              ((fine(n,0,j  ,i)*fvol_[0][0](i) + fine(n,0,j+1,i)*fvol_[0][1](i))
               + (fine(n,0,j ,i+1)*fvol_[0][0](i+1) + fine(n,0,j+1,i+1)*fvol_[0][1](i+1)))
              /tvol;
        }
      }
    }
  } else { // 1D
    // printf("1d_restr");
    int j = pmb->js, cj = pmb->cjs, k = pmb->ks, ck = pmb->cks;
    for (int n=sn; n<=en; ++n) {
      pco->CellVolume(k,j,si,ei,fvol_[0][0]);
      for (int ci=csi; ci<=cei; ci++) {
        int i = (ci - pmb->cis)*2 + pmb->is;
        Real tvol = fvol_[0][0](i) + fvol_[0][0](i+1);
        coarse(n,ck,cj,ci)
          = (fine(n,k,j,i)*fvol_[0][0](i) + fine(n,k,j,i+1)*fvol_[0][0](i+1))/tvol;

      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictFieldX1(const AthenaArray<Real> &fine
//      AthenaArray<Real> &coarse, int csi, int cei, int csj, int cej, int csk, int cek)
//  \brief restrict the x1 field data and set them into the coarse buffer

void MeshRefinement::RestrictFieldX1(
    const AthenaArray<Real> &fine, AthenaArray<Real> &coarse,
    int csi, int cei, int csj, int cej, int csk, int cek) {
  MeshBlock *pmb = pmy_block_;
  Coordinates *pco = pmb->pcoord;
  int si = (csi - pmb->cis)*2 + pmb->is, ei = (cei - pmb->cis)*2 + pmb->is;

  // store the restricted data in the prolongation buffer for later use
  if (pmb->block_size.nx3>1) { // 3D
    for (int ck=csk; ck<=cek; ck++) {
      int k = (ck - pmb->cks)*2 + pmb->ks;
      for (int cj=csj; cj<=cej; cj++) {
        int j = (cj - pmb->cjs)*2 + pmb->js;
        pco->Face1Area(k,   j,   si, ei, sarea_x1_[0][0]);
        pco->Face1Area(k,   j+1, si, ei, sarea_x1_[0][1]);
        pco->Face1Area(k+1, j,   si, ei, sarea_x1_[1][0]);
        pco->Face1Area(k+1, j+1, si, ei, sarea_x1_[1][1]);
        for (int ci=csi; ci<=cei; ci++) {
          int i = (ci - pmb->cis)*2 + pmb->is;
          Real tarea = sarea_x1_[0][0](i) + sarea_x1_[0][1](i) +
                       sarea_x1_[1][0](i) + sarea_x1_[1][1](i);
          coarse(ck,cj,ci) =
              (fine(k  ,j,i)*sarea_x1_[0][0](i) + fine(k  ,j+1,i)*sarea_x1_[0][1](i)
               + fine(k+1,j,i)*sarea_x1_[1][0](i) + fine(k+1,j+1,i)*sarea_x1_[1][1](i)
               )/tarea;
        }
      }
    }
  } else if (pmb->block_size.nx2>1) { // 2D
    int k = pmb->ks;
    for (int cj=csj; cj<=cej; cj++) {
      int j = (cj - pmb->cjs)*2 + pmb->js;
      pco->Face1Area(k,  j,   si, ei, sarea_x1_[0][0]);
      pco->Face1Area(k,  j+1, si, ei, sarea_x1_[0][1]);
      for (int ci=csi; ci<=cei; ci++) {
        int i = (ci - pmb->cis)*2 + pmb->is;
        Real tarea = sarea_x1_[0][0](i) + sarea_x1_[0][1](i);
        coarse(csk,cj,ci) =
            (fine(k,j,i)*sarea_x1_[0][0](i) + fine(k,j+1,i)*sarea_x1_[0][1](i))/tarea;
      }
    }
  } else { // 1D - no restriction, just copy
    for (int ci=csi; ci<=cei; ci++) {
      int i = (ci - pmb->cis)*2 + pmb->is;
      coarse(csk,csj,ci) = fine(pmb->ks,pmb->js,i);
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictFieldX2(const AthenaArray<Real> &fine
//      AthenaArray<Real> &coarse, int csi, int cei, int csj, int cej, int csk, int cek)
//  \brief restrict the x2 field data and set them into the coarse buffer

void MeshRefinement::RestrictFieldX2(
    const AthenaArray<Real> &fine, AthenaArray<Real> &coarse,
    int csi, int cei, int csj, int cej, int csk, int cek) {
  MeshBlock *pmb = pmy_block_;
  Coordinates *pco = pmb->pcoord;
  int si = (csi - pmb->cis)*2 + pmb->is, ei = (cei - pmb->cis)*2 + pmb->is + 1;

  // store the restricted data in the prolongation buffer for later use
  if (pmb->block_size.nx3>1) { // 3D
    for (int ck=csk; ck<=cek; ck++) {
      int k = (ck - pmb->cks)*2 + pmb->ks;
      for (int cj=csj; cj<=cej; cj++) {
        int j = (cj - pmb->cjs)*2 + pmb->js;
        bool pole = pco->IsPole(j);
        if (!pole) {
          pco->Face2Area(k,   j,  si, ei, sarea_x2_[0][0]);
          pco->Face2Area(k+1, j,  si, ei, sarea_x2_[1][0]);
        } else {
          for (int ci = csi; ci <= cei; ++ci) {
            int i = (ci - pmb->cis) * 2 + pmb->is;
            sarea_x2_[0][0](i) = pco->dx1f(i);
            sarea_x2_[1][0](i) = pco->dx1f(i);
          }
        }
        for (int ci=csi; ci<=cei; ci++) {
          int i = (ci - pmb->cis)*2 + pmb->is;
          Real tarea = sarea_x2_[0][0](i) + sarea_x2_[0][0](i+1) +
                       sarea_x2_[1][0](i) + sarea_x2_[1][0](i+1);
          coarse(ck,cj,ci) =
              (fine(k  ,j,i)*sarea_x2_[0][0](i) + fine(k  ,j,i+1)*sarea_x2_[0][0](i+1)
               +fine(k+1,j,i)*sarea_x2_[1][0](i) + fine(k+1,j,i+1)*sarea_x2_[1][0](i+1))
              /tarea;
        }
      }
    }
  } else if (pmb->block_size.nx2>1) { // 2D
    int k = pmb->ks;
    for (int cj=csj; cj<=cej; cj++) {
      int j = (cj - pmb->cjs)*2 + pmb->js;
      bool pole = pco->IsPole(j);
      if (!pole) {
        pco->Face2Area(k, j, si, ei, sarea_x2_[0][0]);
      } else {
        for (int ci = csi; ci <= cei; ++ci) {
          int i = (ci - pmb->cis) * 2 + pmb->is;
          sarea_x2_[0][0](i) = pco->dx1f(i);
        }
      }
      for (int ci=csi; ci<=cei; ci++) {
        int i = (ci - pmb->cis)*2 + pmb->is;
        Real tarea = sarea_x2_[0][0](i) + sarea_x2_[0][0](i+1);
        coarse(pmb->cks,cj,ci) =
            (fine(k,j,i)*sarea_x2_[0][0](i) + fine(k,j,i+1)*sarea_x2_[0][0](i+1))/tarea;
      }
    }
  } else { // 1D
    int k = pmb->ks, j = pmb->js;
    pco->Face2Area(k, j, si, ei, sarea_x2_[0][0]);
    for (int ci=csi; ci<=cei; ci++) {
      int i = (ci - pmb->cis)*2 + pmb->is;
      Real tarea = sarea_x2_[0][0](i) + sarea_x2_[0][0](i+1);
      coarse(pmb->cks,pmb->cjs,ci) =
          (fine(k,j,i)*sarea_x2_[0][0](i) + fine(k,j,i+1)*sarea_x2_[0][0](i+1))/tarea;
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictFieldX3(const AthenaArray<Real> &fine
//      AthenaArray<Real> &coarse, int csi, int cei, int csj, int cej, int csk, int cek)
//  \brief restrict the x3 field data and set them into the coarse buffer

void MeshRefinement::RestrictFieldX3(
    const AthenaArray<Real> &fine, AthenaArray<Real> &coarse,
    int csi, int cei, int csj, int cej, int csk, int cek) {
  MeshBlock *pmb = pmy_block_;
  Coordinates *pco = pmb->pcoord;
  int si = (csi - pmb->cis)*2 + pmb->is, ei = (cei - pmb->cis)*2 + pmb->is + 1;

  // store the restricted data in the prolongation buffer for later use
  if (pmb->block_size.nx3>1) { // 3D
    for (int ck=csk; ck<=cek; ck++) {
      int k = (ck - pmb->cks)*2 + pmb->ks;
      for (int cj=csj; cj<=cej; cj++) {
        int j = (cj - pmb->cjs)*2 + pmb->js;
        pco->Face3Area(k,   j,  si, ei, sarea_x3_[0][0]);
        pco->Face3Area(k, j+1,  si, ei, sarea_x3_[0][1]);
        for (int ci=csi; ci<=cei; ci++) {
          int i = (ci - pmb->cis)*2 + pmb->is;
          Real tarea = sarea_x3_[0][0](i) + sarea_x3_[0][0](i+1) +
                       sarea_x3_[0][1](i) + sarea_x3_[0][1](i+1);
          coarse(ck,cj,ci)  =
              (fine(k,j  ,i)*sarea_x3_[0][0](i) + fine(k,j  ,i+1)*sarea_x3_[0][0](i+1)
               + fine(k,j+1,i)*sarea_x3_[0][1](i) + fine(k,j+1,i+1)*sarea_x3_[0][1](i+1)
               ) /tarea;
        }
      }
    }
  } else if (pmb->block_size.nx2>1) { // 2D
    int k = pmb->ks;
    for (int cj=csj; cj<=cej; cj++) {
      int j = (cj - pmb->cjs)*2 + pmb->js;
      pco->Face3Area(k,   j, si, ei, sarea_x3_[0][0]);
      pco->Face3Area(k, j+1, si, ei, sarea_x3_[0][1]);
      for (int ci=csi; ci<=cei; ci++) {
        int i = (ci - pmb->cis)*2 + pmb->is;
        Real tarea = sarea_x3_[0][0](i) + sarea_x3_[0][0](i+1) +
                     sarea_x3_[0][1](i) + sarea_x3_[0][1](i+1);
        coarse(pmb->cks,cj,ci) =
            (fine(k,j  ,i)*sarea_x3_[0][0](i) + fine(k,j  ,i+1)*sarea_x3_[0][0](i+1)
             + fine(k,j+1,i)*sarea_x3_[0][1](i) + fine(k,j+1,i+1)*sarea_x3_[0][1](i+1)
             ) /tarea;
      }
    }
  } else { // 1D
    int k = pmb->ks, j = pmb->js;
    pco->Face3Area(k, j, si, ei, sarea_x3_[0][0]);
    for (int ci=csi; ci<=cei; ci++) {
      int i = (ci - pmb->cis)*2 + pmb->is;
      Real tarea = sarea_x3_[0][0](i) + sarea_x3_[0][0](i+1);
      coarse(pmb->cks,pmb->cjs,ci) =
          (fine(k,j,i)*sarea_x3_[0][0](i) + fine(k,j,i+1)*sarea_x3_[0][0](i+1))/tarea;
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn inline void MeshRefinement::RestrictVertexCenteredIndicialHelper(...)
//  \brief De-duplicate some indicial logic
inline void MeshRefinement::RestrictVertexCenteredIndicialHelper(
  int ix,
  int ix_cvs, int ix_cve,
  int ix_vs, int ix_ve,
  int &f_ix) {

  // map for fine-index
  if (ix < ix_cvs) {
    f_ix = ix_vs - 2 * (ix_cvs - ix);
  } else if (ix > ix_cve) {
    f_ix = ix_ve + 2 * (ix - ix_cve);
  } else { // map to interior+boundary nodes
    f_ix = 2 * (ix - ix_cvs) + ix_vs;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictVertexCenteredValues(const AthenaArray<Real> &fine,
//                           AthenaArray<Real> &coarse, int sn, int en,
//                           int csi, int cei, int csj, int cej, int csk, int cek)
//  \brief restrict vertex centered values

void MeshRefinement::RestrictVertexCenteredValues(
    const AthenaArray<Real> &fine, AthenaArray<Real> &coarse, int sn, int en,
    int csi, int cei, int csj, int cej, int csk, int cek) {
  MeshBlock *pmb = pmy_block_;

  // store the restricted data in the prolongation buffer for later use
  if (pmb->block_size.nx3 > 1) {
    int k, j, i;
    for (int n=sn; n<=en; ++n){
      for (int ck=csk; ck<=cek; ck++) {
        // int k = (ck - pmb->ckvs)*2 + pmb->kvs;
        RestrictVertexCenteredIndicialHelper(ck, pmb->ckvs, pmb->ckve,
                                             pmb->kvs, pmb->kve, k);
        for (int cj=csj; cj<=cej; cj++) {
          // int j = (cj - pmb->cjvs)*2 + pmb->jvs;
          RestrictVertexCenteredIndicialHelper(cj, pmb->cjvs, pmb->cjve,
                                              pmb->jvs, pmb->jve, j);
          for (int ci=csi; ci<=cei; ci++) {
            // int i = (ci - pmb->civs)*2 + pmb->ivs;
            RestrictVertexCenteredIndicialHelper(ci, pmb->civs, pmb->cive,
                                                pmb->ivs, pmb->ive, i);
            coarse(n, ck, cj, ci) = fine(n, k, j, i);
          }
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) {
    int k = pmb->kvs, ck = pmb->ckvs;
    int j, i;

    for (int n=sn; n<=en; ++n){
      for (int cj=csj; cj<=cej; cj++) {
        // int j = (cj - pmb->cjvs)*2 + pmb->jvs;
        RestrictVertexCenteredIndicialHelper(cj, pmb->cjvs, pmb->cjve,
                                             pmb->jvs, pmb->jve, j);
        for (int ci=csi; ci<=cei; ci++) {
          // int i = (ci - pmb->civs)*2 + pmb->ivs;
          RestrictVertexCenteredIndicialHelper(ci, pmb->civs, pmb->cive,
                                               pmb->ivs, pmb->ive, i);
          coarse(n, ck, cj, ci) = fine(n, k, j, i);
        }
      }
    }
  } else {
    int j = pmb->jvs, cj = pmb->cjvs, k = pmb->kvs, ck = pmb->ckvs;
    int i;
    for (int n=sn; n<=en; ++n) {
      for (int ci=csi; ci<=cei; ci++) {
        // int i = (ci - pmb->civs)*2 + pmb->ivs;
        RestrictVertexCenteredIndicialHelper(ci, pmb->civs, pmb->cive,
                                              pmb->ivs, pmb->ive, i);
        coarse(n, ck, cj, ci) = fine(n, k, j, i);
      }
    }

  }



}

void MeshRefinement::RestrictTwiceToBufferVertexCenteredValues(
    const AthenaArray<Real> &fine,
    Real *buf,
    int sn, int en,
    int csi, int cei, int csj, int cej, int csk, int cek,
    int &offset) {
  MeshBlock *pmb = pmy_block_;
  // Coordinates *pco = pmb->pcoord;

  // store the restricted data within input buffer
  if (pmb->block_size.nx3 > 1) { // 3D
    int k, j, i;
    for (int n=sn; n<=en; ++n){
      for (int ck=csk; ck<=cek; ck+=2) {
        // int k = (ck - pmb->ckvs)*2 + pmb->kvs;
        RestrictVertexCenteredIndicialHelper(ck, pmb->ckvs, pmb->ckve,
                                             pmb->kvs, pmb->kve, k);
        for (int cj=csj; cj<=cej; cj+=2) {
          // int j = (cj - pmb->cjvs)*2 + pmb->jvs;
          RestrictVertexCenteredIndicialHelper(cj, pmb->cjvs, pmb->cjve,
                                              pmb->jvs, pmb->jve, j);
          for (int ci=csi; ci<=cei; ci+=2) {
            // int i = (ci - pmb->civs)*2 + pmb->ivs;
            RestrictVertexCenteredIndicialHelper(ci, pmb->civs, pmb->cive,
                                                pmb->ivs, pmb->ive, i);
            buf[offset++] = fine(n, k, j, i);
          }
        }
      }
    }

  } else if (pmb->block_size.nx2 > 1) { // 2D
    int k = pmb->kvs;
    int j, i;
    for (int n=sn; n<=en; ++n){
      for (int cj=csj; cj<=cej; cj+=2) {
        // int j = (cj - pmb->cjvs)*2 + pmb->jvs;
        RestrictVertexCenteredIndicialHelper(cj, pmb->cjvs, pmb->cjve,
                                             pmb->jvs, pmb->jve, j);
        for (int ci=csi; ci<=cei; ci+=2) {
          // int i = (ci - pmb->civs)*2 + pmb->ivs;
          RestrictVertexCenteredIndicialHelper(ci, pmb->civs, pmb->cive,
                                               pmb->ivs, pmb->ive, i);
          buf[offset++] = fine(n, k, j, i);
        }
      }
    }

  } else { // 1D
    int j = pmb->jvs, k = pmb->kvs;
    int i;
    for (int n=sn; n<=en; ++n) {
      for (int ci=csi; ci<=cei; ci+=2) {
        // int i = (ci - pmb->civs)*2 + pmb->ivs;
        RestrictVertexCenteredIndicialHelper(ci, pmb->civs, pmb->cive,
                                             pmb->ivs, pmb->ive, i);
        buf[offset++] = fine(n, k, j, i);
      }
    }

  }


  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictCellCenteredXValues(const AthenaArray<Real> &fine,
//                           AthenaArray<Real> &coarse, int sn, int en,
//                           int csi, int cei, int csj, int cej, int csk, int cek)
//  \brief restrict cell centered (extended) values

void MeshRefinement::RestrictCellCenteredXValues(
  const AthenaArray<Real> &fine, AthenaArray<Real> &coarse,
  int sn, int en,
  int csi, int cei, int csj, int cej, int csk, int cek)
{
  // RestrictCellCenteredValues(fine, coarse, sn, en,
  //                            csi, cei, csj, cej, csk, cek);
  // return;

  // // BD: debug all rat. -------------------------------------------------------
  // AthenaArray<Real> & var_t = coarse;
  // AthenaArray<Real> & var_s = const_cast<AthenaArray<Real>&>(fine);
  // for(int n=sn; n<=en; ++n)
  // {
  //   const Real* const fcn_s = &(var_s(n,0,0,0));
  //   Real* fcn_t = &(var_t(n,0,0,0));

  //   // ind_interior_r_op->eval(fcn_t, fcn_s);
  //   ind_physical_r_op->eval_nn(fcn_t, fcn_s,
  //     csi, cei,
  //     csj, cej,
  //     csk, cek);
  // }
  // return;
  // // --------------------------------------------------------------------------

  // here H_SZ * 2 - 1 is resultant degree
  const int H_SZ = NGHOST;  // works for all physical nodes
  // const int H_SZ = 1;  // works for all physical nodes

  // BD: debug with LO
  // const int H_SZ = 1;  // works for all physical nodes

  MeshBlock *pmb = pmy_block_;
  // Coordinates *pco = pmb->pcoord;

  // // map to fine idx
  // int si = 2 * (csi - pmb->cx_cis) + pmb->cx_is;
  // int ei = 2 * (cei - pmb->cx_cis) + pmb->cx_is + 1;

  if (pmb->block_size.nx3>1)
  { // 3D
    for (int n=sn; n<=en; ++n)
    for (int cx_ck=csk; cx_ck<=cek; cx_ck++)
    {
      // left child idx on fine grid
      const int cx_fk = 2 * (cx_ck - pmb->cx_cks) + pmb->cx_ks;

      for (int cx_cj=csj; cx_cj<=cej; cx_cj++)
      {
        // left child idx on fine grid
        const int cx_fj = 2 * (cx_cj - pmb->cx_cjs) + pmb->cx_js;

        for (int cx_ci=csi; cx_ci<=cei; cx_ci++)
        {
          // left child idx on fine grid
          const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

          // use templated ----------------------------------------------------
          coarse(n,cx_ck,cx_cj,cx_ci) = 0.0;

          // #pragma unroll
          for (int dk=0; dk<H_SZ; ++dk)
          {
            int const cx_fk_l = cx_fk - dk;
            int const cx_fk_r = cx_fk + dk + 1;
            Real const lck = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-dk-1];

            // #pragma unroll
            for (int dj=0; dj<H_SZ; ++dj)
            {
              int const cx_fj_l = cx_fj - dj;
              int const cx_fj_r = cx_fj + dj + 1;
              Real const lckj = lck * InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-dj-1];

              // #pragma unroll
              for (int di=0; di<H_SZ; ++di)
              {
                int const cx_fi_l = cx_fi - di;
                int const cx_fi_r = cx_fi + di + 1;

                Real const lckji = lckj * InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-di-1];

                Real const f_rrr = fine(n, cx_fk_r, cx_fj_r, cx_fi_r);
                Real const f_lrr = fine(n, cx_fk_l, cx_fj_r, cx_fi_r);
                Real const f_rlr = fine(n, cx_fk_r, cx_fj_l, cx_fi_r);
                Real const f_rrl = fine(n, cx_fk_r, cx_fj_r, cx_fi_l);

                Real const f_llr = fine(n, cx_fk_l, cx_fj_l, cx_fi_r);
                Real const f_rll = fine(n, cx_fk_r, cx_fj_l, cx_fi_l);
                Real const f_lrl = fine(n, cx_fk_l, cx_fj_r, cx_fi_l);
                Real const f_lll = fine(n, cx_fk_l, cx_fj_l, cx_fi_l);

                coarse(n,cx_ck,cx_cj,cx_ci) += lckji * FloatingPoint::sum_associative(
                  f_rrr, f_lll, f_rrl, f_llr,
                  f_lrl, f_rlr, f_lrr, f_rll
                );
              }
            }
          }


        }

      }
    }


  }
  else if (pmb->block_size.nx2>1)
  { // 2D
    const int cx_fk = pmb->ks, cx_ck = pmb->cks;
    for (int n=sn; n<=en; ++n)
    for (int cx_cj=csj; cx_cj<=cej; cx_cj++)
    {
      // left child idx on fine grid
      const int cx_fj = 2 * (cx_cj - pmb->cx_cjs) + pmb->cx_js;

      for (int cx_ci=csi; cx_ci<=cei; cx_ci++)
      {
        // left child idx on fine grid
        const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

        // use templated ------------------------------------------------------
        coarse(n,cx_ck,cx_cj,cx_ci) = 0.0;

        #pragma unroll
        for (int dj=0; dj<H_SZ; ++dj)
        {
          int const cx_fj_l = cx_fj - dj;
          int const cx_fj_r = cx_fj + dj + 1;
          Real const lcj = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-dj-1];

          #pragma unroll
          for (int di=0; di<H_SZ; ++di)
          {
            int const cx_fi_l = cx_fi - di;
            int const cx_fi_r = cx_fi + di + 1;

            Real const lcji = lcj * InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-di-1];

            Real const f_uu = fine(n, 0, cx_fj_r, cx_fi_r);
            Real const f_ul = fine(n, 0, cx_fj_r, cx_fi_l);
            Real const f_lu = fine(n, 0, cx_fj_l, cx_fi_r);
            Real const f_ll = fine(n, 0, cx_fj_l, cx_fi_l);

            coarse(n,cx_ck,cx_cj,cx_ci) += lcji * FloatingPoint::sum_associative(
              f_uu, f_ll, f_lu, f_ul
            );
          }
        }


      }

    }
  }
  else
  { // 1D
    const int cx_fj = pmb->js, cx_cj = pmb->cjs;
    const int cx_fk = pmb->ks, cx_ck = pmb->cks;

    for (int n=sn; n<=en; ++n)
    for (int cx_ci=csi; cx_ci<=cei; cx_ci++)
    {
      // left child idx on fine grid
      const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

      // if ((cx_ci < pmb->cis + 1) | (cx_ci > pmb->cie - 1))
      // {
      // // linear interp. -------------------------------------------------------
      // coarse(n,cx_ck,cx_cj,cx_ci) = 0.5 * (
      //   fine(n,cx_fk,cx_fj,cx_fi  ) +
      //   fine(n,cx_fk,cx_fj,cx_fi+1)
      // );
      // // ----------------------------------------------------------------------
      // }
      // else
      // {
      // coarse(n,cx_ck,cx_cj,cx_ci) = (
      //   -1.0 * fine(n,cx_fk,cx_fj,cx_fi-1) +
      //   9.0  * fine(n,cx_fk,cx_fj,cx_fi  ) +
      //   9.0  * fine(n,cx_fk,cx_fj,cx_fi+1) +
      //   -1.0 * fine(n,cx_fk,cx_fj,cx_fi+2)
      // ) / 16.0;
      // }

      // linear interp. -------------------------------------------------------
      // coarse(n,cx_ck,cx_cj,cx_ci) = 0.5 * (
      //   fine(n,cx_fk,cx_fj,cx_fi  ) +
      //   fine(n,cx_fk,cx_fj,cx_fi+1)
      // );

      // deg 3 interp. --------------------------------------------------------
      // coarse(n,cx_ck,cx_cj,cx_ci) = (
      //   -1.0 * fine(n,cx_fk,cx_fj,cx_fi-1) +
      //   9.0  * fine(n,cx_fk,cx_fj,cx_fi  ) +
      //   9.0  * fine(n,cx_fk,cx_fj,cx_fi+1) +
      //   -1.0 * fine(n,cx_fk,cx_fj,cx_fi+2)
      // ) / 16.0;
      // ----------------------------------------------------------------------

      // deg 5 interp. --------------------------------------------------------
      // coarse(n,cx_ck,cx_cj,cx_ci) = (
      //   3.0   * fine(n,cx_fk,cx_fj,cx_fi-2) +
      //   -25.0 * fine(n,cx_fk,cx_fj,cx_fi-1) +
      //   150.0 * fine(n,cx_fk,cx_fj,cx_fi  ) +
      //   150.0 * fine(n,cx_fk,cx_fj,cx_fi+1) +
      //   -25.0 * fine(n,cx_fk,cx_fj,cx_fi+2) +
      //   3.0   * fine(n,cx_fk,cx_fj,cx_fi+3)
      // ) / 256.0;
      // ----------------------------------------------------------------------


      // use templated --------------------------------------------------------
      coarse(n,cx_ck,cx_cj,cx_ci) = 0.0;

      #pragma unroll
      for (int di=0; di<H_SZ; ++di)
      {
        Real const lc = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-di-1];
        int const cx_fi_l = cx_fi - di;
        int const cx_fi_r = cx_fi + di + 1;

        coarse(n,cx_ck,cx_cj,cx_ci) += lc * (
          fine(n,cx_fk,cx_fj,cx_fi_l) +
          fine(n,cx_fk,cx_fj,cx_fi_r)
        );
      }


      // ----------------------------------------------------------------------

    }
  }
}

void MeshRefinement::RestrictCellCenteredXValuesLO(
  const AthenaArray<Real> &fine, AthenaArray<Real> &coarse,
  int sn, int en,
  int csi, int cei, int csj, int cej, int csk, int cek)
{

  // RestrictCellCenteredValues(fine, coarse, sn, en,
  //                            csi, cei, csj, cej, csk, cek);
  // return;
  // // BD: debug all rat. -------------------------------------------------------
  // AthenaArray<Real> & var_t = coarse;
  // AthenaArray<Real> & var_s = const_cast<AthenaArray<Real>&>(fine);
  // for(int n=sn; n<=en; ++n)
  // {
  //   const Real* const fcn_s = &(var_s(n,0,0,0));
  //   Real* fcn_t = &(var_t(n,0,0,0));

  //   // ind_interior_r_op->eval(fcn_t, fcn_s);
  //   ind_physical_r_op->eval_nn(fcn_t, fcn_s,
  //     csi, cei,
  //     csj, cej,
  //     csk, cek);
  // }
  // return;
  // // --------------------------------------------------------------------------

  // here H_SZ * 2 - 1 is resultant degree
  const int H_SZ = 1;  // works for all physical nodes

  // BD: debug with LO
  // const int H_SZ = 1;  // works for all physical nodes

  MeshBlock *pmb = pmy_block_;
  // Coordinates *pco = pmb->pcoord;

  // // map to fine idx
  // int si = 2 * (csi - pmb->cx_cis) + pmb->cx_is;
  // int ei = 2 * (cei - pmb->cx_cis) + pmb->cx_is + 1;

  if (pmb->block_size.nx3>1)
  { // 3D
    for (int n=sn; n<=en; ++n)
    for (int cx_ck=csk; cx_ck<=cek; cx_ck++)
    {
      // left child idx on fine grid
      const int cx_fk = 2 * (cx_ck - pmb->cx_cks) + pmb->cx_ks;

      for (int cx_cj=csj; cx_cj<=cej; cx_cj++)
      {
        // left child idx on fine grid
        const int cx_fj = 2 * (cx_cj - pmb->cx_cjs) + pmb->cx_js;

        for (int cx_ci=csi; cx_ci<=cei; cx_ci++)
        {
          // left child idx on fine grid
          const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

          // use templated ----------------------------------------------------
          coarse(n,cx_ck,cx_cj,cx_ci) = 0.0;

          // #pragma unroll
          for (int dk=0; dk<H_SZ; ++dk)
          {
            int const cx_fk_l = cx_fk - dk;
            int const cx_fk_r = cx_fk + dk + 1;
            Real const lck = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-dk-1];

            // #pragma unroll
            for (int dj=0; dj<H_SZ; ++dj)
            {
              int const cx_fj_l = cx_fj - dj;
              int const cx_fj_r = cx_fj + dj + 1;
              Real const lckj = lck * InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-dj-1];

              // #pragma unroll
              for (int di=0; di<H_SZ; ++di)
              {
                int const cx_fi_l = cx_fi - di;
                int const cx_fi_r = cx_fi + di + 1;

                Real const lckji = lckj * InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-di-1];

                Real const f_rrr = fine(n, cx_fk_r, cx_fj_r, cx_fi_r);
                Real const f_lrr = fine(n, cx_fk_l, cx_fj_r, cx_fi_r);
                Real const f_rlr = fine(n, cx_fk_r, cx_fj_l, cx_fi_r);
                Real const f_rrl = fine(n, cx_fk_r, cx_fj_r, cx_fi_l);

                Real const f_llr = fine(n, cx_fk_l, cx_fj_l, cx_fi_r);
                Real const f_rll = fine(n, cx_fk_r, cx_fj_l, cx_fi_l);
                Real const f_lrl = fine(n, cx_fk_l, cx_fj_r, cx_fi_l);
                Real const f_lll = fine(n, cx_fk_l, cx_fj_l, cx_fi_l);

                coarse(n,cx_ck,cx_cj,cx_ci) += lckji * FloatingPoint::sum_associative(
                  f_rrr, f_lll, f_rrl, f_llr,
                  f_lrl, f_rlr, f_lrr, f_rll
                );
              }
            }
          }


        }

      }
    }


  }
  else if (pmb->block_size.nx2>1)
  { // 2D
    const int cx_fk = pmb->ks, cx_ck = pmb->cks;
    for (int n=sn; n<=en; ++n)
    for (int cx_cj=csj; cx_cj<=cej; cx_cj++)
    {
      // left child idx on fine grid
      const int cx_fj = 2 * (cx_cj - pmb->cx_cjs) + pmb->cx_js;

      for (int cx_ci=csi; cx_ci<=cei; cx_ci++)
      {
        // left child idx on fine grid
        const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

        // use templated ------------------------------------------------------
        coarse(n,cx_ck,cx_cj,cx_ci) = 0.0;

        #pragma unroll
        for (int dj=0; dj<H_SZ; ++dj)
        {
          int const cx_fj_l = cx_fj - dj;
          int const cx_fj_r = cx_fj + dj + 1;
          Real const lcj = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-dj-1];

          #pragma unroll
          for (int di=0; di<H_SZ; ++di)
          {
            int const cx_fi_l = cx_fi - di;
            int const cx_fi_r = cx_fi + di + 1;

            Real const lcji = lcj * InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-di-1];

            Real const f_uu = fine(n, 0, cx_fj_r, cx_fi_r);
            Real const f_ul = fine(n, 0, cx_fj_r, cx_fi_l);
            Real const f_lu = fine(n, 0, cx_fj_l, cx_fi_r);
            Real const f_ll = fine(n, 0, cx_fj_l, cx_fi_l);

            coarse(n,cx_ck,cx_cj,cx_ci) += lcji * FloatingPoint::sum_associative(
              f_uu, f_ll, f_lu, f_ul
            );
          }
        }


      }

    }
  }
  else
  { // 1D
    const int cx_fj = pmb->js, cx_cj = pmb->cjs;
    const int cx_fk = pmb->ks, cx_ck = pmb->cks;

    for (int n=sn; n<=en; ++n)
    for (int cx_ci=csi; cx_ci<=cei; cx_ci++)
    {
      // left child idx on fine grid
      const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

      // use templated --------------------------------------------------------
      coarse(n,cx_ck,cx_cj,cx_ci) = 0.0;

      #pragma unroll
      for (int di=0; di<H_SZ; ++di)
      {
        Real const lc = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-di-1];
        int const cx_fi_l = cx_fi - di;
        int const cx_fi_r = cx_fi + di + 1;

        coarse(n,cx_ck,cx_cj,cx_ci) += lc * (
          fine(n,cx_fk,cx_fj,cx_fi_l) +
          fine(n,cx_fk,cx_fj,cx_fi_r)
        );
      }


      // ----------------------------------------------------------------------

    }
  }
}


//----------------------------------------------------------------------------------------
// Restriction utilizing only physical data from fine to coarse grid
void MeshRefinement::RestrictCellCenteredXWithInteriorValues(
    const AthenaArray<Real> &fine,
    AthenaArray<Real> &coarse, int sn, int en)
{
  using namespace numprox::interpolation;

  MeshBlock* pmb = pmy_block_;
  Coordinates* pco = pmb->pcoord;

  int si, sj, sk, ei, ej, ek;
  si = pmb->cx_cis; ei = pmb->cx_cie;
  sj = pmb->cx_cjs; ej = pmb->cx_cje;
  sk = pmb->cx_cks; ek = pmb->cx_cke;

  // // BD: debug with LO
  // RestrictCellCenteredXValuesLO(
  //   fine, coarse, sn, en,
  //   si, ei,
  //   sj, ej,
  //   sk, ek);
  // return;

  const int Ns_x3 = pmb->block_size.nx3 - 1;  // # phys. nodes - 1
  const int Ns_x2 = pmb->block_size.nx2 - 1;  // # phys. nodes - 1
  const int Ns_x1 = pmb->block_size.nx1 - 1;

  if (Z4C_CX_NUM_RBC > 0)
  {
    RestrictCellCenteredXValuesLO(fine, coarse, sn, en,
                                  si, ei, sj, ej, sk, ek);
    return;
  }
  // RestrictCellCenteredXValuesLO(fine, coarse, sn, en,
  //                               si, ei, sj, ej, sk, ek);
  // return;

  // Floater-Hormann blending parameter controls the formal order of approx.
  // const int d = (NGHOST-1) * 2 + 1;

  AthenaArray<Real> & var_t = coarse;
  AthenaArray<Real> & var_s = const_cast<AthenaArray<Real>&>(fine);

  if(pmb->block_size.nx3>1)
  {
    /*
    for(int n=sn; n<=en; ++n)
    for(int ck=pmb->cx_cks; ck<=pmb->cx_cke; ++ck)
    {
      // left child idx on fundamental grid
      const int cx_fk = 2 * (ck - pmb->cx_cks) + pmb->cx_ks;

      Real* x3_s = &(pco->x3v(NGHOST));

      // coarse variable grids are constructed with CC ghost number
      const Real x3_t = pcoarsec->x3v(NCGHOST+(ck-NCGHOST_CX));

      for(int cj=pmb->cx_cjs; cj<=pmb->cx_cje; ++cj)
      {
        // left child idx on fundamental grid
        const int cx_fj = 2 * (cj - pmb->cx_cjs) + pmb->cx_js;

        Real* x2_s = &(pco->x2v(NGHOST));

        // coarse variable grids are constructed with CC ghost number
        const Real x2_t = pcoarsec->x2v(NCGHOST+(cj-NCGHOST_CX));

        for(int ci=pmb->cx_cis; ci<=pmb->cx_cie; ++ci)
        {
          // left child idx on fundamental grid
          const int cx_fi = 2 * (ci - pmb->cx_cis) + pmb->cx_is;

          Real* x1_s = &(pco->x1v(NGHOST));
          Real* fcn_s = &(var_s(n,0,0,0));

          const Real x1_t = pcoarsec->x1v(NCGHOST+(ci-NCGHOST_CX));

          var_t(n,ck,cj,ci) = Floater_Hormann::interp_3d(
            x1_t, x2_t, x3_t,
            x1_s, x2_s, x3_s,
            fcn_s,
            Ns_x1, Ns_x2, Ns_x3, d, NGHOST);
        }
      }
    }
    */


    /*
    // grids
    Real* x1_s = &(pco->x1v(NGHOST));
    Real* x2_s = &(pco->x2v(NGHOST));
    Real* x3_s = &(pco->x3v(NGHOST));

    Real* x1_t = &(pcoarsec->x1v(NCGHOST+(pmb->cx_cis-NCGHOST_CX)));
    Real* x2_t = &(pcoarsec->x2v(NCGHOST+(pmb->cx_cjs-NCGHOST_CX)));
    Real* x3_t = &(pcoarsec->x3v(NCGHOST+(pmb->cx_cks-NCGHOST_CX)));

    const int Nt_x1 = pmb->block_size.nx1 / 2 - 1;
    const int Nt_x2 = pmb->block_size.nx2 / 2 - 1;
    const int Nt_x3 = pmb->block_size.nx3 / 2 - 1;

    typedef Floater_Hormann::interp_nd_weights_precomputed<Real, Real>
      interp_nd_weights_precomputed;

    interp_nd_weights_precomputed * i3c = new interp_nd_weights_precomputed(
      x1_t, x2_t, x3_t,
      x1_s, x2_s, x3_s,
      Nt_x1, Nt_x2, Nt_x3,
      Ns_x1, Ns_x2, Ns_x3,
      d,
      NCGHOST_CX, NGHOST
    );

    for(int n=sn; n<=en; ++n)
    {
      const Real* const fcn_s = &(var_s(n,0,0,0));
      Real* fcn_t = &(var_t(n,0,0,0));

      i3c->eval(fcn_t, fcn_s);
    }

    delete i3c;
    */

    for(int n=sn; n<=en; ++n)
    {
      const Real* const fcn_s = &(var_s(n,0,0,0));
      Real* fcn_t = &(var_t(n,0,0,0));

      // ind_interior_r_op->eval(fcn_t, fcn_s);
      ind_interior_r_op->eval_nn(fcn_t, fcn_s);
    }

  }
  else if(pmb->block_size.nx2>1)
  {
    /*
    for(int n=sn; n<=en; ++n)
    for(int cj=pmb->cx_cjs; cj<=pmb->cx_cje; ++cj)
    {
      // left child idx on fundamental grid
      const int cx_fj = 2 * (cj - pmb->cx_cjs) + pmb->cx_js;

      Real* x2_s = &(pco->x2v(NGHOST));

      // coarse variable grids are constructed with CC ghost number
      const Real x2_t = pcoarsec->x2v(NCGHOST+(cj-NCGHOST_CX));

      for(int ci=pmb->cx_cis; ci<=pmb->cx_cie; ++ci)
      {
        // left child idx on fundamental grid
        const int cx_fi = 2 * (ci - pmb->cx_cis) + pmb->cx_is;

        Real* x1_s = &(pco->x1v(NGHOST));
        Real* fcn_s = &(var_s(n,0,0,0));

        const Real x1_t = pcoarsec->x1v(NCGHOST+(ci-NCGHOST_CX));

        var_t(n,0,cj,ci) = Floater_Hormann::interp_2d(
          x1_t, x2_t,
          x1_s, x2_s,
          fcn_s,
          Ns_x1, Ns_x2, d, NGHOST);
      }
    }
    */

    /*
    // grids
    Real* x1_s = &(pco->x1v(NGHOST));
    Real* x2_s = &(pco->x2v(NGHOST));

    Real* x1_t = &(pcoarsec->x1v(NCGHOST+(pmb->cx_cis-NCGHOST_CX)));
    Real* x2_t = &(pcoarsec->x2v(NCGHOST+(pmb->cx_cjs-NCGHOST_CX)));

    const int Nt_x1 = pmb->block_size.nx1 / 2 - 1;
    const int Nt_x2 = pmb->block_size.nx2 / 2 - 1;

    typedef Floater_Hormann::interp_nd_weights_precomputed<Real, Real>
      interp_nd_weights_precomputed;

    interp_nd_weights_precomputed * i2c = new interp_nd_weights_precomputed(
      x1_t, x2_t,
      x1_s, x2_s,
      Nt_x1, Nt_x2,
      Ns_x1, Ns_x2,
      d,
      NCGHOST_CX, NGHOST
    );

    for(int n=sn; n<=en; ++n)
    {
      const Real* const fcn_s = &(var_s(n,0,0,0));
      Real* fcn_t = &(var_t(n,0,0,0));

      i2c->eval(fcn_t, fcn_s);
    }

    delete i2c;
    */

    for(int n=sn; n<=en; ++n)
    {
      const Real* const fcn_s = &(var_s(n,0,0,0));
      Real* fcn_t = &(var_t(n,0,0,0));

      ind_interior_r_op->eval_nn(fcn_t, fcn_s);
      // ind_interior_r_op->eval(fcn_t, fcn_s);

      // ind_interior_r_op->eval(fcn_t, fcn_s,
      //                         pmb->cx_cis-NCGHOST_CX, pmb->cx_cie-NCGHOST_CX,
      //                         pmb->cx_cjs-NCGHOST_CX, pmb->cx_cje-NCGHOST_CX);

    }

  }
  else
  {
    /*
    for(int n=sn; n<=en; ++n)
    for(int ci=pmb->cx_cis; ci<=pmb->cx_cie; ++ci)
    {
      // left child idx on fundamental grid
      const int cx_fi = 2 * (ci - pmb->cx_cis) + pmb->cx_is;

      Real* x1_s = &(pco->x1v(NGHOST));
      Real* fcn_s = &(var_s(n,0,0,0));

      // coarse variable grids are constructed with CC ghost number
      const Real x1_t = pcoarsec->x1v(NCGHOST+(ci-NCGHOST_CX));

      var_t(n,0,0,ci) = Floater_Hormann::interp_1d(
        x1_t, x1_s,
        fcn_s,
        Ns_x1, d, NGHOST);
    }
    */

    /*
    // grids
    Real* x1_s = &(pco->x1v(NGHOST));
    Real* x1_t = &(pcoarsec->x1v(NCGHOST+(pmb->cx_cis-NCGHOST_CX)));

    const int Nt_x1 = pmb->block_size.nx1 / 2 - 1;

    typedef Floater_Hormann::interp_nd_weights_precomputed<Real, Real>
      interp_nd_weights_precomputed;

    interp_nd_weights_precomputed * i1c = new interp_nd_weights_precomputed(
      x1_t, x1_s,
      Nt_x1, Ns_x1,
      d,
      NCGHOST_CX, NGHOST
    );

    for(int n=sn; n<=en; ++n)
    {
      const Real* const fcn_s = &(var_s(n,0,0,0));
      Real* fcn_t = &(var_t(n,0,0,0));

      i1c->eval(fcn_t, fcn_s);
    }


    delete i1c;
    */

    for(int n=sn; n<=en; ++n)
    {
      const Real* const fcn_s = &(var_s(n,0,0,0));
      Real* fcn_t = &(var_t(n,0,0,0));

      // ind_interior_r_op->eval(fcn_t, fcn_s);
      ind_interior_r_op->eval_nn(fcn_t, fcn_s);

      // ind_interior_r_op->eval(fcn_t, fcn_s,
      //                         pmb->cx_cis-NCGHOST_CX, pmb->cx_cigs-NCGHOST_CX);

    }

    // {
    //   const Real* const fcn_s = &(var_s(n,0,0,0));

    //   for(int i=pmb->cx_is; i<=pmb->cx_igs; ++i)
    //   {
    //     var_t(n,0,0,i) = Floater_Hormann::interp_1d(
    //       pmb->pcoord->x1v(i),
    //       &(pmb->pcoord->x1v(NGHOST)),
    //       fcn_s,
    //       d+2, // stencil size
    //       d,   // order
    //       NGHOST
    //     );
    //   }

    // }

  }

}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateCellCenteredValues(
//        const AthenaArray<Real> &coarse,AthenaArray<Real> &fine, int sn, int en,,
//        int si, int ei, int sj, int ej, int sk, int ek)
//  \brief Prolongate cell centered values

void MeshRefinement::ProlongateCellCenteredValues(
    const AthenaArray<Real> &coarse, AthenaArray<Real> &fine,
    int sn, int en, int si, int ei, int sj, int ej, int sk, int ek) {
  MeshBlock *pmb = pmy_block_;
  Coordinates *pco = pmb->pcoord;

  if (pmb->block_size.nx3 > 1) {
    for (int n=sn; n<=en; n++) {
      for (int k=sk; k<=ek; k++) {
        int fk = (k - pmb->cks)*2 + pmb->ks;
        const Real& x3m = pcoarsec->x3v(k-1);
        const Real& x3c = pcoarsec->x3v(k);
        const Real& x3p = pcoarsec->x3v(k+1);
        Real dx3m = x3c - x3m;
        Real dx3p = x3p - x3c;
        const Real& fx3m = pco->x3v(fk);
        const Real& fx3p = pco->x3v(fk+1);
        Real dx3fm =  x3c - fx3m;
        Real dx3fp =  fx3p - x3c;
        for (int j = sj; j<=ej; j++) {
          int fj = (j - pmb->cjs)*2 + pmb->js;
          const Real& x2m = pcoarsec->x2v(j-1);
          const Real& x2c = pcoarsec->x2v(j);
          const Real& x2p = pcoarsec->x2v(j+1);
          Real dx2m = x2c - x2m;
          Real dx2p = x2p - x2c;
          const Real& fx2m = pco->x2v(fj);
          const Real& fx2p = pco->x2v(fj+1);
          Real dx2fm = x2c - fx2m;
          Real dx2fp = fx2p - x2c;
          for (int i=si; i<=ei; i++) {
            int fi = (i - pmb->cis)*2 + pmb->is;
            const Real& x1m = pcoarsec->x1v(i-1);
            const Real& x1c = pcoarsec->x1v(i);
            const Real& x1p = pcoarsec->x1v(i+1);
            Real dx1m = x1c - x1m;
            Real dx1p = x1p - x1c;
            const Real& fx1m = pco->x1v(fi);
            const Real& fx1p = pco->x1v(fi+1);
            Real dx1fm = x1c - fx1m;
            Real dx1fp = fx1p - x1c;
            Real ccval = coarse(n,k,j,i);

            // calculate 3D gradients using the minmod limiter
            Real gx1c, gx2c, gx3c;

            Real gx1m = (ccval - coarse(n,k,j,i-1))/dx1m;
            Real gx1p = (coarse(n,k,j,i+1) - ccval)/dx1p;
            gx1c = 0.5*(SIGN(gx1m) + SIGN(gx1p))*
              std::min(std::abs(gx1m), std::abs(gx1p));

            Real gx2m = (ccval - coarse(n,k,j-1,i))/dx2m;
            Real gx2p = (coarse(n,k,j+1,i) - ccval)/dx2p;
            gx2c = 0.5*(SIGN(gx2m) + SIGN(gx2p))*
              std::min(std::abs(gx2m), std::abs(gx2p));

            Real gx3m = (ccval - coarse(n,k-1,j,i))/dx3m;
            Real gx3p = (coarse(n,k+1,j,i) - ccval)/dx3p;
            gx3c = 0.5*(SIGN(gx3m) + SIGN(gx3p))*
              std::min(std::abs(gx3m), std::abs(gx3p));

            // KGF: add the off-centered quantities first to preserve FP symmetry
            // interpolate onto the finer grid
            fine(n,fk  ,fj  ,fi  ) = ccval - (gx1c*dx1fm + gx2c*dx2fm + gx3c*dx3fm);
            fine(n,fk  ,fj  ,fi+1) = ccval + (gx1c*dx1fp - gx2c*dx2fm - gx3c*dx3fm);
            fine(n,fk  ,fj+1,fi  ) = ccval - (gx1c*dx1fm - gx2c*dx2fp + gx3c*dx3fm);
            fine(n,fk  ,fj+1,fi+1) = ccval + (gx1c*dx1fp + gx2c*dx2fp - gx3c*dx3fm);
            fine(n,fk+1,fj  ,fi  ) = ccval - (gx1c*dx1fm + gx2c*dx2fm - gx3c*dx3fp);
            fine(n,fk+1,fj  ,fi+1) = ccval + (gx1c*dx1fp - gx2c*dx2fm + gx3c*dx3fp);
            fine(n,fk+1,fj+1,fi  ) = ccval - (gx1c*dx1fm - gx2c*dx2fp - gx3c*dx3fp);
            fine(n,fk+1,fj+1,fi+1) = ccval + (gx1c*dx1fp + gx2c*dx2fp + gx3c*dx3fp);
          }
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) {
    int k = pmb->cks, fk = pmb->ks;
    for (int n=sn; n<=en; n++) {
      for (int j=sj; j<=ej; j++) {
        int fj = (j - pmb->cjs)*2 + pmb->js;
        const Real& x2m = pcoarsec->x2v(j-1);
        const Real& x2c = pcoarsec->x2v(j);
        const Real& x2p = pcoarsec->x2v(j+1);
        Real dx2m = x2c - x2m;
        Real dx2p = x2p - x2c;
        const Real& fx2m = pco->x2v(fj);
        const Real& fx2p = pco->x2v(fj+1);
        Real dx2fm = x2c - fx2m;
        Real dx2fp = fx2p - x2c;
        for (int i=si; i<=ei; i++) {
          int fi = (i - pmb->cis)*2 + pmb->is;
          const Real& x1m = pcoarsec->x1v(i-1);
          const Real& x1c = pcoarsec->x1v(i);
          const Real& x1p = pcoarsec->x1v(i+1);
          Real dx1m = x1c - x1m;
          Real dx1p = x1p - x1c;
          const Real& fx1m = pco->x1v(fi);
          const Real& fx1p = pco->x1v(fi+1);
          Real dx1fm = x1c - fx1m;
          Real dx1fp = fx1p - x1c;
          Real ccval = coarse(n,k,j,i);

          Real gx1c, gx2c;
          Real gx1m = (ccval - coarse(n,k,j,i-1))/dx1m;
          Real gx1p = (coarse(n,k,j,i+1) - ccval)/dx1p;
          gx1c = 0.5*(SIGN(gx1m) + SIGN(gx1p))*
            std::min(std::abs(gx1m), std::abs(gx1p));

          Real gx2m = (ccval - coarse(n,k,j-1,i))/dx2m;
          Real gx2p = (coarse(n,k,j+1,i) - ccval)/dx2p;
          gx2c = 0.5*(SIGN(gx2m) + SIGN(gx2p))*
            std::min(std::abs(gx2m), std::abs(gx2p));

          // KGF: add the off-centered quantities first to preserve FP symmetry
          // interpolate onto the finer grid
          fine(n,fk  ,fj  ,fi  ) = ccval - (gx1c*dx1fm + gx2c*dx2fm);
          fine(n,fk  ,fj  ,fi+1) = ccval + (gx1c*dx1fp - gx2c*dx2fm);
          fine(n,fk  ,fj+1,fi  ) = ccval - (gx1c*dx1fm - gx2c*dx2fp);
          fine(n,fk  ,fj+1,fi+1) = ccval + (gx1c*dx1fp + gx2c*dx2fp);
        }
      }
    }
  } else { // 1D
    int k = pmb->cks, fk = pmb->ks, j = pmb->cjs, fj = pmb->js;
    for (int n=sn; n<=en; n++) {
      for (int i=si; i<=ei; i++) {
        int fi = (i - pmb->cis)*2 + pmb->is;
        const Real& x1m = pcoarsec->x1v(i-1);
        const Real& x1c = pcoarsec->x1v(i);
        const Real& x1p = pcoarsec->x1v(i+1);
        Real dx1m = x1c - x1m;
        Real dx1p = x1p - x1c;
        const Real& fx1m = pco->x1v(fi);
        const Real& fx1p = pco->x1v(fi+1);
        Real dx1fm = x1c - fx1m;
        Real dx1fp = fx1p - x1c;
        Real ccval = coarse(n,k,j,i);

        Real gx1c;

        // calculate 1D gradient using the min-mod limiter
        Real gx1m = (ccval - coarse(n,k,j,i-1))/dx1m;
        Real gx1p = (coarse(n,k,j,i+1) - ccval)/dx1p;
        gx1c = 0.5*(SIGN(gx1m) + SIGN(gx1p))*std::min(std::abs(gx1m),
                                                      std::abs(gx1p));

        // interpolate on to the finer grid
        fine(n,fk  ,fj  ,fi  ) = ccval - gx1c*dx1fm;
        fine(n,fk  ,fj  ,fi+1) = ccval + gx1c*dx1fp;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateSharedFieldX1(const AthenaArray<Real> &coarse,
//      AthenaArray<Real> &fine, int si, int ei, int sj, int ej, int sk, int ek)
//  \brief prolongate x1 face-centered fields shared between coarse and fine levels

void MeshRefinement::ProlongateSharedFieldX1(
    const AthenaArray<Real> &coarse, AthenaArray<Real> &fine,
    int si, int ei, int sj, int ej, int sk, int ek) {
  MeshBlock *pmb = pmy_block_;
  Coordinates *pco = pmb->pcoord;
  if (pmb->block_size.nx3 > 1) {
    for (int k=sk; k<=ek; k++) {
      int fk = (k - pmb->cks)*2 + pmb->ks;
      const Real& x3m = pcoarsec->x3s1(k-1);
      const Real& x3c = pcoarsec->x3s1(k);
      const Real& x3p = pcoarsec->x3s1(k+1);
      Real dx3m = x3c - x3m;
      Real dx3p = x3p - x3c;
      const Real& fx3m = pco->x3s1(fk);
      const Real& fx3p = pco->x3s1(fk+1);
      for (int j=sj; j<=ej; j++) {
        int fj = (j - pmb->cjs)*2 + pmb->js;
        const Real& x2m = pcoarsec->x2s1(j-1);
        const Real& x2c = pcoarsec->x2s1(j);
        const Real& x2p = pcoarsec->x2s1(j+1);
        Real dx2m = x2c - x2m;
        Real dx2p = x2p - x2c;
        const Real& fx2m = pco->x2s1(fj);
        const Real& fx2p = pco->x2s1(fj+1);
        for (int i=si; i<=ei; i++) {
          int fi = (i - pmb->cis)*2 + pmb->is;
          Real ccval = coarse(k,j,i);

          Real gx2m = (ccval - coarse(k,j-1,i))/dx2m;
          Real gx2p = (coarse(k,j+1,i) - ccval)/dx2p;
          Real gx2c = 0.5*(SIGN(gx2m) + SIGN(gx2p))*std::min(std::abs(gx2m),
                                                             std::abs(gx2p));
          Real gx3m = (ccval - coarse(k-1,j,i))/dx3m;
          Real gx3p = (coarse(k+1,j,i) - ccval)/dx3p;
          Real gx3c = 0.5*(SIGN(gx3m) + SIGN(gx3p))*std::min(std::abs(gx3m),
                                                             std::abs(gx3p));

          fine(fk  ,fj  ,fi) = ccval - gx2c*(x2c - fx2m) - gx3c*(x3c - fx3m);
          fine(fk  ,fj+1,fi) = ccval + gx2c*(fx2p - x2c) - gx3c*(x3c - fx3m);
          fine(fk+1,fj  ,fi) = ccval - gx2c*(x2c - fx2m) + gx3c*(fx3p - x3c);
          fine(fk+1,fj+1,fi) = ccval + gx2c*(fx2p - x2c) + gx3c*(fx3p - x3c);
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) {
    int k = pmb->cks, fk = pmb->ks;
    for (int j=sj; j<=ej; j++) {
      int fj = (j - pmb->cjs)*2 + pmb->js;
      const Real& x2m = pcoarsec->x2s1(j-1);
      const Real& x2c = pcoarsec->x2s1(j);
      const Real& x2p = pcoarsec->x2s1(j+1);
      Real dx2m = x2c - x2m;
      Real dx2p = x2p - x2c;
      const Real& fx2m = pco->x2s1(fj);
      const Real& fx2p = pco->x2s1(fj+1);
      for (int i=si; i<=ei; i++) {
        int fi = (i - pmb->cis)*2 + pmb->is;
        Real ccval = coarse(k,j,i);

        Real gx2m = (ccval - coarse(k,j-1,i))/dx2m;
        Real gx2p = (coarse(k,j+1,i) - ccval)/dx2p;
        Real gx2c = 0.5*(SIGN(gx2m) + SIGN(gx2p))*std::min(std::abs(gx2m),
                                                           std::abs(gx2p));

        fine(fk,fj  ,fi) = ccval - gx2c*(x2c - fx2m);
        fine(fk,fj+1,fi) = ccval + gx2c*(fx2p - x2c);
      }
    }
  } else { // 1D
    for (int i=si; i<=ei; i++) {
      int fi = (i - pmb->cis)*2 + pmb->is;
      fine(0,0,fi) = coarse(0,0,i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateSharedFieldX2(const AthenaArray<Real> &coarse,
//      AthenaArray<Real> &fine, int si, int ei, int sj, int ej, int sk, int ek)
//  \brief prolongate x2 face-centered fields shared between coarse and fine levels

void MeshRefinement::ProlongateSharedFieldX2(
    const AthenaArray<Real> &coarse, AthenaArray<Real> &fine,
    int si, int ei, int sj, int ej, int sk, int ek) {
  MeshBlock *pmb = pmy_block_;
  Coordinates *pco = pmb->pcoord;
  if (pmb->block_size.nx3 > 1) {
    for (int k=sk; k<=ek; k++) {
      int fk = (k - pmb->cks)*2 + pmb->ks;
      const Real& x3m = pcoarsec->x3s2(k-1);
      const Real& x3c = pcoarsec->x3s2(k);
      const Real& x3p = pcoarsec->x3s2(k+1);
      Real dx3m = x3c - x3m;
      Real dx3p = x3p - x3c;
      const Real& fx3m = pco->x3s2(fk);
      const Real& fx3p = pco->x3s2(fk+1);
      for (int j=sj; j<=ej; j++) {
        int fj = (j - pmb->cjs)*2 + pmb->js;
        for (int i=si; i<=ei; i++) {
          int fi = (i - pmb->cis)*2 + pmb->is;
          const Real& x1m = pcoarsec->x1s2(i-1);
          const Real& x1c = pcoarsec->x1s2(i);
          const Real& x1p = pcoarsec->x1s2(i+1);
          Real dx1m = x1c - x1m;
          Real dx1p = x1p - x1c;
          const Real& fx1m = pco->x1s2(fi);
          const Real& fx1p = pco->x1s2(fi+1);
          Real ccval = coarse(k,j,i);

          Real gx1m = (ccval - coarse(k,j,i-1))/dx1m;
          Real gx1p = (coarse(k,j,i+1) - ccval)/dx1p;
          Real gx1c = 0.5*(SIGN(gx1m) + SIGN(gx1p))*std::min(std::abs(gx1m),
                                                             std::abs(gx1p));
          Real gx3m = (ccval - coarse(k-1,j,i))/dx3m;
          Real gx3p = (coarse(k+1,j,i) - ccval)/dx3p;
          Real gx3c = 0.5*(SIGN(gx3m) + SIGN(gx3p))*std::min(std::abs(gx3m),
                                                             std::abs(gx3p));

          fine(fk  ,fj,fi  ) = ccval - gx1c*(x1c - fx1m) - gx3c*(x3c - fx3m);
          fine(fk  ,fj,fi+1) = ccval + gx1c*(fx1p - x1c) - gx3c*(x3c - fx3m);
          fine(fk+1,fj,fi  ) = ccval - gx1c*(x1c - fx1m) + gx3c*(fx3p - x3c);
          fine(fk+1,fj,fi+1) = ccval + gx1c*(fx1p - x1c) + gx3c*(fx3p - x3c);
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) {
    int k = pmb->cks, fk = pmb->ks;
    for (int j=sj; j<=ej; j++) {
      int fj = (j - pmb->cjs)*2 + pmb->js;
      for (int i=si; i<=ei; i++) {
        int fi = (i - pmb->cis)*2 + pmb->is;
        const Real& x1m = pcoarsec->x1s2(i-1);
        const Real& x1c = pcoarsec->x1s2(i);
        const Real& x1p = pcoarsec->x1s2(i+1);
        const Real& fx1m = pco->x1s2(fi);
        const Real& fx1p = pco->x1s2(fi+1);
        Real ccval = coarse(k,j,i);

        Real gx1m = (ccval - coarse(k,j,i-1))/(x1c - x1m);
        Real gx1p = (coarse(k,j,i+1) - ccval)/(x1p - x1c);
        Real gx1c = 0.5*(SIGN(gx1m) + SIGN(gx1p))*std::min(std::abs(gx1m),
                                                           std::abs(gx1p));

        fine(fk,fj,fi  ) = ccval - gx1c*(x1c - fx1m);
        fine(fk,fj,fi+1) = ccval + gx1c*(fx1p - x1c);
      }
    }
  } else {
    for (int i=si; i<=ei; i++) {
      int fi = (i - pmb->cis)*2 + pmb->is;
      Real gxm = (coarse(0,0,i) - coarse(0,0,i-1))
                 /(pcoarsec->x1s2(i) - pcoarsec->x1s2(i-1));
      Real gxp = (coarse(0,0,i+1) - coarse(0,0,i))
                 /(pcoarsec->x1s2(i+1) - pcoarsec->x1s2(i));
      Real gxc = 0.5*(SIGN(gxm) + SIGN(gxp))*std::min(std::abs(gxm),
                                                      std::abs(gxp));
      fine(0,0,fi  ) = fine(0,1,fi  )
                     = coarse(0,0,i) - gxc*(pcoarsec->x1s2(i) - pco->x1s2(fi));
      fine(0,0,fi+1) = fine(0,1,fi+1)
                     = coarse(0,0,i) + gxc*(pco->x1s2(fi+1) - pcoarsec->x1s2(i));
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateSharedFieldX3(const AthenaArray<Real> &coarse,
//      AthenaArray<Real> &fine, int si, int ei, int sj, int ej, int sk, int ek)
//  \brief prolongate x3 face-centered fields shared between coarse and fine levels

void MeshRefinement::ProlongateSharedFieldX3(
    const AthenaArray<Real> &coarse, AthenaArray<Real> &fine,
    int si, int ei, int sj, int ej, int sk, int ek) {
  MeshBlock *pmb = pmy_block_;
  Coordinates *pco = pmb->pcoord;
  if (pmb->block_size.nx3 > 1) {
    for (int k=sk; k<=ek; k++) {
      int fk = (k - pmb->cks)*2 + pmb->ks;
      for (int j=sj; j<=ej; j++) {
        int fj = (j - pmb->cjs)*2 + pmb->js;
        const Real& x2m = pcoarsec->x2s3(j-1);
        const Real& x2c = pcoarsec->x2s3(j);
        const Real& x2p = pcoarsec->x2s3(j+1);
        Real dx2m = x2c - x2m;
        Real dx2p = x2p - x2c;
        const Real& fx2m = pco->x2s3(fj);
        const Real& fx2p = pco->x2s3(fj+1);
        for (int i=si; i<=ei; i++) {
          int fi = (i - pmb->cis)*2 + pmb->is;
          const Real& x1m = pcoarsec->x1s3(i-1);
          const Real& x1c = pcoarsec->x1s3(i);
          const Real& x1p = pcoarsec->x1s3(i+1);
          Real dx1m = x1c - x1m;
          Real dx1p = x1p - x1c;
          const Real& fx1m = pco->x1s3(fi);
          const Real& fx1p = pco->x1s3(fi+1);
          Real ccval = coarse(k,j,i);

          Real gx1m = (ccval - coarse(k,j,i-1))/dx1m;
          Real gx1p = (coarse(k,j,i+1) - ccval)/dx1p;
          Real gx1c = 0.5*(SIGN(gx1m) + SIGN(gx1p))*std::min(std::abs(gx1m),
                                                             std::abs(gx1p));
          Real gx2m = (ccval - coarse(k,j-1,i))/dx2m;
          Real gx2p = (coarse(k,j+1,i) - ccval)/dx2p;
          Real gx2c = 0.5*(SIGN(gx2m) + SIGN(gx2p))*std::min(std::abs(gx2m),
                                                             std::abs(gx2p));

          fine(fk,fj  ,fi  ) = ccval - gx1c*(x1c - fx1m) - gx2c*(x2c - fx2m);
          fine(fk,fj  ,fi+1) = ccval + gx1c*(fx1p - x1c) - gx2c*(x2c - fx2m);
          fine(fk,fj+1,fi  ) = ccval - gx1c*(x1c - fx1m) + gx2c*(fx2p - x2c);
          fine(fk,fj+1,fi+1) = ccval + gx1c*(fx1p - x1c) + gx2c*(fx2p - x2c);
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) {
    int k = pmb->cks, fk = pmb->ks;
    for (int j=sj; j<=ej; j++) {
      int fj = (j - pmb->cjs)*2 + pmb->js;
      const Real& x2m = pcoarsec->x2s3(j-1);
      const Real& x2c = pcoarsec->x2s3(j);
      const Real& x2p = pcoarsec->x2s3(j+1);
      Real dx2m = x2c - x2m;
      Real dx2p = x2p - x2c;
      const Real& fx2m = pco->x2s3(fj);
      const Real& fx2p = pco->x2s3(fj+1);
      Real dx2fm = x2c - fx2m;
      Real dx2fp = fx2p - x2c;
      for (int i=si; i<=ei; i++) {
        int fi = (i - pmb->cis)*2 + pmb->is;
        const Real& x1m = pcoarsec->x1s3(i-1);
        const Real& x1c = pcoarsec->x1s3(i);
        const Real& x1p = pcoarsec->x1s3(i+1);
        Real dx1m = x1c - x1m;
        Real dx1p = x1p - x1c;
        const Real& fx1m = pco->x1s3(fi);
        const Real& fx1p = pco->x1s3(fi+1);
        Real dx1fm = x1c - fx1m;
        Real dx1fp = fx1p - x1c;
        Real ccval = coarse(k,j,i);

        // calculate 2D gradients using the minmod limiter
        Real gx1m = (ccval - coarse(k,j,i-1))/dx1m;
        Real gx1p = (coarse(k,j,i+1) - ccval)/dx1p;
        Real gx1c = 0.5*(SIGN(gx1m) + SIGN(gx1p))*std::min(std::abs(gx1m),
                                                           std::abs(gx1p));
        Real gx2m = (ccval - coarse(k,j-1,i))/dx2m;
        Real gx2p = (coarse(k,j+1,i) - ccval)/dx2p;
        Real gx2c = 0.5*(SIGN(gx2m) + SIGN(gx2p))*std::min(std::abs(gx2m),
                                                           std::abs(gx2p));

        // interpolate on to the finer grid
        fine(fk,fj  ,fi  ) = fine(fk+1,fj  ,fi  ) = ccval - gx1c*dx1fm-gx2c*dx2fm;
        fine(fk,fj  ,fi+1) = fine(fk+1,fj  ,fi+1) = ccval + gx1c*dx1fp-gx2c*dx2fm;
        fine(fk,fj+1,fi  ) = fine(fk+1,fj+1,fi  ) = ccval - gx1c*dx1fm+gx2c*dx2fp;
        fine(fk,fj+1,fi+1) = fine(fk+1,fj+1,fi+1) = ccval + gx1c*dx1fp+gx2c*dx2fp;
      }
    }
  } else {
    for (int i=si; i<=ei; i++) {
      int fi = (i - pmb->cis)*2 + pmb->is;
      Real gxm = (coarse(0,0,i)   - coarse(0,0,i-1))
                 / (pcoarsec->x1s3(i) - pcoarsec->x1s3(i-1));
      Real gxp = (coarse(0,0,i+1) - coarse(0,0,i))
                 / (pcoarsec->x1s3(i+1) - pcoarsec->x1s3(i));
      Real gxc = 0.5*(SIGN(gxm) + SIGN(gxp))*std::min(std::abs(gxm),
                                                      std::abs(gxp));
      fine(0,0,fi  ) = fine(1,0,fi  )
                     = coarse(0,0,i) - gxc*(pcoarsec->x1s3(i) - pco->x1s3(fi));
      fine(0,0,fi+1) = fine(1,0,fi+1)
                     = coarse(0,0,i) + gxc*(pco->x1s3(fi+1) - pcoarsec->x1s3(i));
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateInternalField(FaceField &fine,
//                           int si, int ei, int sj, int ej, int sk, int ek)
//  \brief prolongate the internal face-centered fields

void MeshRefinement::ProlongateInternalField(
    FaceField &fine, int si, int ei, int sj, int ej, int sk, int ek) {
  MeshBlock *pmb = pmy_block_;
  Coordinates *pco = pmb->pcoord;
  int fsi = (si - pmb->cis)*2 + pmb->is, fei = (ei - pmb->cis)*2 + pmb->is + 1;
  if (pmb->block_size.nx3 > 1) {
    for (int k=sk; k<=ek; k++) {
      int fk = (k - pmb->cks)*2 + pmb->ks;
      for (int j=sj; j<=ej; j++) {
        int fj = (j - pmb->cjs)*2 + pmb->js;
        pco->Face1Area(fk,   fj,   fsi, fei+1, sarea_x1_[0][0]);
        pco->Face1Area(fk,   fj+1, fsi, fei+1, sarea_x1_[0][1]);
        pco->Face1Area(fk+1, fj,   fsi, fei+1, sarea_x1_[1][0]);
        pco->Face1Area(fk+1, fj+1, fsi, fei+1, sarea_x1_[1][1]);
        pco->Face2Area(fk,   fj,   fsi, fei,   sarea_x2_[0][0]);
        pco->Face2Area(fk,   fj+1, fsi, fei,   sarea_x2_[0][1]);
        pco->Face2Area(fk,   fj+2, fsi, fei,   sarea_x2_[0][2]);
        pco->Face2Area(fk+1, fj,   fsi, fei,   sarea_x2_[1][0]);
        pco->Face2Area(fk+1, fj+1, fsi, fei,   sarea_x2_[1][1]);
        pco->Face2Area(fk+1, fj+2, fsi, fei,   sarea_x2_[1][2]);
        pco->Face3Area(fk,   fj,   fsi, fei,   sarea_x3_[0][0]);
        pco->Face3Area(fk,   fj+1, fsi, fei,   sarea_x3_[0][1]);
        pco->Face3Area(fk+1, fj,   fsi, fei,   sarea_x3_[1][0]);
        pco->Face3Area(fk+1, fj+1, fsi, fei,   sarea_x3_[1][1]);
        pco->Face3Area(fk+2, fj,   fsi, fei,   sarea_x3_[2][0]);
        pco->Face3Area(fk+2, fj+1, fsi, fei,   sarea_x3_[2][1]);
        for (int i=si; i<=ei; i++) {
          int fi = (i - pmb->cis)*2 + pmb->is;
          Real Uxx = 0.0, Vyy = 0.0, Wzz = 0.0;
          Real Uxyz = 0.0, Vxyz = 0.0, Wxyz = 0.0;
#pragma unroll
          for (int jj=0; jj<2; jj++) {
            int js = 2*jj - 1, fjj = fj + jj, fjp = fj + 2*jj;
#pragma unroll
            for (int ii=0; ii<2; ii++) {
              int is = 2*ii - 1, fii = fi + ii, fip = fi + 2*ii;
              Uxx += is*(js*(fine.x2f(fk  ,fjp,fii)*sarea_x2_[0][2*jj](fii) +
                             fine.x2f(fk+1,fjp,fii)*sarea_x2_[1][2*jj](fii))
                         +(fine.x3f(fk+2,fjj,fii)*sarea_x3_[2][  jj](fii) -
                           fine.x3f(fk  ,fjj,fii)*sarea_x3_[0][  jj](fii)));
              Vyy += js*(   (fine.x3f(fk+2,fjj,fii)*sarea_x3_[2][  jj](fii) -
                             fine.x3f(fk  ,fjj,fii)*sarea_x3_[0][  jj](fii))
                            +is*(fine.x1f(fk  ,fjj,fip)*sarea_x1_[0][  jj](fip) +
                                 fine.x1f(fk+1,fjj,fip)*sarea_x1_[1][  jj](fip)));
              Wzz +=     is*(fine.x1f(fk+1,fjj,fip)*sarea_x1_[1][  jj](fip) -
                             fine.x1f(fk  ,fjj,fip)*sarea_x1_[0][  jj](fip))
                         +js*(fine.x2f(fk+1,fjp,fii)*sarea_x2_[1][2*jj](fii) -
                              fine.x2f(fk  ,fjp,fii)*sarea_x2_[0][2*jj](fii));
              Uxyz += is*js*(fine.x1f(fk+1,fjj,fip)*sarea_x1_[1][  jj](fip) -
                             fine.x1f(fk  ,fjj,fip)*sarea_x1_[0][  jj](fip));
              Vxyz += is*js*(fine.x2f(fk+1,fjp,fii)*sarea_x2_[1][2*jj](fii) -
                             fine.x2f(fk  ,fjp,fii)*sarea_x2_[0][2*jj](fii));
              Wxyz += is*js*(fine.x3f(fk+2,fjj,fii)*sarea_x3_[2][  jj](fii) -
                             fine.x3f(fk  ,fjj,fii)*sarea_x3_[0][  jj](fii));
            }
          }
          Real Sdx1 = SQR(pco->dx1f(fi) + pco->dx1f(fi+1));
          Real Sdx2 = SQR(pco->GetEdge2Length(fk+1,fj,fi+1) +
                          pco->GetEdge2Length(fk+1,fj+1,fi+1));
          Real Sdx3 = SQR(pco->GetEdge3Length(fk,fj+1,fi+1) +
                          pco->GetEdge3Length(fk+1,fj+1,fi+1));
          Uxx *= 0.125; Vyy *= 0.125; Wzz *= 0.125;
          Uxyz *= 0.125/(Sdx2 + Sdx3);
          Vxyz *= 0.125/(Sdx1 + Sdx3);
          Wxyz *= 0.125/(Sdx1 + Sdx2);
          fine.x1f(fk  ,fj  ,fi+1) =
              (0.5*(fine.x1f(fk  ,fj  ,fi  )*sarea_x1_[0][0](fi  ) +
                    fine.x1f(fk  ,fj  ,fi+2)*sarea_x1_[0][0](fi+2))
               + Uxx - Sdx3*Vxyz - Sdx2*Wxyz) /sarea_x1_[0][0](fi+1);
          fine.x1f(fk  ,fj+1,fi+1) =
              (0.5*(fine.x1f(fk  ,fj+1,fi  )*sarea_x1_[0][1](fi  ) +
                    fine.x1f(fk  ,fj+1,fi+2)*sarea_x1_[0][1](fi+2))
               + Uxx - Sdx3*Vxyz + Sdx2*Wxyz) /sarea_x1_[0][1](fi+1);
          fine.x1f(fk+1,fj  ,fi+1) =
              (0.5*(fine.x1f(fk+1,fj  ,fi  )*sarea_x1_[1][0](fi  ) +
                    fine.x1f(fk+1,fj  ,fi+2)*sarea_x1_[1][0](fi+2))
               + Uxx + Sdx3*Vxyz - Sdx2*Wxyz) /sarea_x1_[1][0](fi+1);
          fine.x1f(fk+1,fj+1,fi+1) =
              (0.5*(fine.x1f(fk+1,fj+1,fi  )*sarea_x1_[1][1](fi  ) +
                    fine.x1f(fk+1,fj+1,fi+2)*sarea_x1_[1][1](fi+2))
               + Uxx + Sdx3*Vxyz + Sdx2*Wxyz) /sarea_x1_[1][1](fi+1);

          fine.x2f(fk  ,fj+1,fi  ) =
              (0.5*(fine.x2f(fk  ,fj  ,fi  )*sarea_x2_[0][0](fi  ) +
                    fine.x2f(fk  ,fj+2,fi  )*sarea_x2_[0][2](fi  ))
               + Vyy - Sdx3*Uxyz - Sdx1*Wxyz) /sarea_x2_[0][1](fi  );
          fine.x2f(fk  ,fj+1,fi+1) =
              (0.5*(fine.x2f(fk  ,fj  ,fi+1)*sarea_x2_[0][0](fi+1) +
                    fine.x2f(fk  ,fj+2,fi+1)*sarea_x2_[0][2](fi+1))
               + Vyy - Sdx3*Uxyz + Sdx1*Wxyz) /sarea_x2_[0][1](fi+1);
          fine.x2f(fk+1,fj+1,fi  ) =
              (0.5*(fine.x2f(fk+1,fj  ,fi  )*sarea_x2_[1][0](fi  ) +
                    fine.x2f(fk+1,fj+2,fi  )*sarea_x2_[1][2](fi  ))
               + Vyy + Sdx3*Uxyz - Sdx1*Wxyz) /sarea_x2_[1][1](fi  );
          fine.x2f(fk+1,fj+1,fi+1) =
              (0.5*(fine.x2f(fk+1,fj  ,fi+1)*sarea_x2_[1][0](fi+1) +
                    fine.x2f(fk+1,fj+2,fi+1)*sarea_x2_[1][2](fi+1))
               + Vyy + Sdx3*Uxyz + Sdx1*Wxyz) /sarea_x2_[1][1](fi+1);

          fine.x3f(fk+1,fj  ,fi  ) =
              (0.5*(fine.x3f(fk+2,fj  ,fi  )*sarea_x3_[2][0](fi  ) +
                    fine.x3f(fk  ,fj  ,fi  )*sarea_x3_[0][0](fi  ))
               + Wzz - Sdx2*Uxyz - Sdx1*Vxyz) /sarea_x3_[1][0](fi  );
          fine.x3f(fk+1,fj  ,fi+1) =
              (0.5*(fine.x3f(fk+2,fj  ,fi+1)*sarea_x3_[2][0](fi+1) +
                    fine.x3f(fk  ,fj  ,fi+1)*sarea_x3_[0][0](fi+1))
               + Wzz - Sdx2*Uxyz + Sdx1*Vxyz) /sarea_x3_[1][0](fi+1);
          fine.x3f(fk+1,fj+1,fi  ) =
              (0.5*(fine.x3f(fk+2,fj+1,fi  )*sarea_x3_[2][1](fi  ) +
                    fine.x3f(fk  ,fj+1,fi  )*sarea_x3_[0][1](fi  ))
               + Wzz + Sdx2*Uxyz - Sdx1*Vxyz) /sarea_x3_[1][1](fi  );
          fine.x3f(fk+1,fj+1,fi+1) =
              (0.5*(fine.x3f(fk+2,fj+1,fi+1)*sarea_x3_[2][1](fi+1) +
                    fine.x3f(fk  ,fj+1,fi+1)*sarea_x3_[0][1](fi+1))
               + Wzz + Sdx2*Uxyz + Sdx1*Vxyz) /sarea_x3_[1][1](fi+1);
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) {
    int fk = pmb->ks;
    for (int j=sj; j<=ej; j++) {
      int fj = (j - pmb->cjs)*2 + pmb->js;
      pco->Face1Area(fk,   fj,   fsi, fei+1, sarea_x1_[0][0]);
      pco->Face1Area(fk,   fj+1, fsi, fei+1, sarea_x1_[0][1]);
      pco->Face2Area(fk,   fj,   fsi, fei,   sarea_x2_[0][0]);
      pco->Face2Area(fk,   fj+1, fsi, fei,   sarea_x2_[0][1]);
      pco->Face2Area(fk,   fj+2, fsi, fei,   sarea_x2_[0][2]);
      for (int i=si; i<=ei; i++) {
        int fi = (i - pmb->cis)*2 + pmb->is;
        Real tmp1 = 0.25*(fine.x2f(fk,fj+2,fi+1)*sarea_x2_[0][2](fi+1)
                          - fine.x2f(fk,fj,  fi+1)*sarea_x2_[0][0](fi+1)
                          - fine.x2f(fk,fj+2,fi  )*sarea_x2_[0][2](fi  )
                          + fine.x2f(fk,fj,  fi  )*sarea_x2_[0][0](fi  ));
        Real tmp2 = 0.25*(fine.x1f(fk,fj,  fi  )*sarea_x1_[0][0](fi  )
                          - fine.x1f(fk,fj,  fi+2)*sarea_x1_[0][0](fi+2)
                          - fine.x1f(fk,fj+1,fi  )*sarea_x1_[0][1](fi  )
                          + fine.x1f(fk,fj+1,fi+2)*sarea_x1_[0][1](fi+2));
        fine.x1f(fk,fj  ,fi+1) =
            (0.5*(fine.x1f(fk,fj,  fi  )*sarea_x1_[0][0](fi  )
                  +fine.x1f(fk,fj,  fi+2)*sarea_x1_[0][0](fi+2)) + tmp1)
            /sarea_x1_[0][0](fi+1);
        fine.x1f(fk,fj+1,fi+1) =
            (0.5*(fine.x1f(fk,fj+1,fi  )*sarea_x1_[0][1](fi  )
                  +fine.x1f(fk,fj+1,fi+2)*sarea_x1_[0][1](fi+2)) + tmp1)
            /sarea_x1_[0][1](fi+1);
        fine.x2f(fk,fj+1,fi  ) =
            (0.5*(fine.x2f(fk,fj,  fi  )*sarea_x2_[0][0](fi  )
                  +fine.x2f(fk,fj+2,fi  )*sarea_x2_[0][2](fi  )) + tmp2)
            /sarea_x2_[0][1](fi  );
        fine.x2f(fk,fj+1,fi+1) =
            (0.5*(fine.x2f(fk,fj,  fi+1)*sarea_x2_[0][0](fi+1)
                  +fine.x2f(fk,fj+2,fi+1)*sarea_x2_[0][2](fi+1)) + tmp2)
            /sarea_x2_[0][1](fi+1);
      }
    }
  } else {
    pco->Face1Area(0, 0, fsi, fei+1, sarea_x1_[0][0]);
    for (int i=si; i<=ei; i++) {
      int fi = (i - pmb->cis)*2 + pmb->is;
      Real ph = sarea_x1_[0][0](fi)*fine.x1f(0,0,fi);
      fine.x1f(0,0,fi+1) = ph/sarea_x1_[0][0](fi+1);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn inline void MeshRefinement::ProlongateVertexCenteredIndicialHelper(...)
//  \brief De-duplicate some indicial logic
inline void MeshRefinement::ProlongateVertexCenteredIndicialHelper(
  int hs_sz, int ix,
  int ix_cvs, int ix_cve, int ix_cmp,
  int ix_vs, int ix_ve,
  int &f_ix, int &ix_b, int &ix_so, int &ix_eo, int &ix_l, int &ix_u) {

  // map for fine-index
  if (ix < ix_cvs) {
    f_ix = ix_vs - 2 * (ix_cvs - ix);
  } else if (ix > ix_cve) {
    f_ix = ix_ve + 2 * (ix - ix_cve);
  } else { // map to interior+boundary nodes
    f_ix = 2 * (ix - ix_cvs) + ix_vs;
  }

  // bias direction [nb. stencil still symmetric!]
  if (ix < ix_cmp) {
    ix_b = 1;
    ix_so = 0;
    ix_eo = 1;
  } else if (ix > ix_cmp) {
    ix_b = -1;
    ix_so = -1;
    ix_eo = 0;
  } else {
    // central node is unbiased, coincident, inject with no neighbors
    ix_so = ix_eo = 0;
    ix_b = -1;
  }

  ix_l = ix - hs_sz + 1 - (1 - ix_b) / 2;
  ix_u = ix + hs_sz - (1 - ix_b) / 2;

  return;

}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateVertexCenteredValues(
//        const AthenaArray<Real> &coarse,AthenaArray<Real> &fine, int sn, int en,,
//        int si, int ei, int sj, int ej, int sk, int ek)
//  \brief Prolongate vertex centered values;
//  Faster implementation, bias towards center by default

void MeshRefinement::ProlongateVertexCenteredValues(
    const AthenaArray<Real> &coarse, AthenaArray<Real> &fine,
    int sn, int en, int si, int ei, int sj, int ej, int sk, int ek) {

  // // half number of ghosts
  // int const H_NCGHOST = NCGHOST / 2;

  // // maximum stencil size for interpolator
  // int const H_SZ = H_NCGHOST + 1;

  // ghost shift parameter
  int const H_NCGHOST = (2 * NCGHOST - NGHOST) / 2;

  // maximum stencil size for interpolator
  int const H_SZ = (2*NCGHOST-NGHOST) / 2 + 1;


  MeshBlock *pmb = pmy_block_;

  if (pmb->pmy_mesh->ndim == 3) {
#if ISEVEN(NGHOST)
    int const eo_offset = 0;

    int const si_inj{(si)};
    int const ei_inj{(ei)};

    int const sj_inj{(sj)};
    int const ej_inj{(ej)};

    int const sk_inj{(sk)};
    int const ek_inj{(ek)};
#else
    int const eo_offset = -1;

    int const fis_inj{(2 * (si - H_NCGHOST) + eo_offset)};
    int const fie_inj{(2 * (ei - H_NCGHOST) + eo_offset)};

    int const fjs_inj{(2 * (sj - H_NCGHOST) + eo_offset)};
    int const fje_inj{(2 * (ej - H_NCGHOST) + eo_offset)};

    int const fks_inj{(2 * (sk - H_NCGHOST) + eo_offset)};
    int const fke_inj{(2 * (ek - H_NCGHOST) + eo_offset)};

    int const si_inj{(fis_inj < 0) ? si + 1 : si};
    int const ei_inj{(fie_inj > 2 * NGHOST + pmb->block_size.nx1) ? ei - 1 : ei};

    int const sj_inj{(fjs_inj < 0) ? sj + 1 : sj};
    int const ej_inj{(fje_inj > 2 * NGHOST + pmb->block_size.nx2) ? ej - 1 : ej};

    int const sk_inj{(fks_inj < 0) ? sk + 1 : sk};
    int const ek_inj{(fke_inj > 2 * NGHOST + pmb->block_size.nx3) ? ek - 1 : ek};
#endif


    //-------------------------------------------------------------------------
    // bias offsets for prolongation (depends on location)
    int const si_prl{(si > pmb->cimp) ? si - 1 : si};
    int const ei_prl{(ei < pmb->cimp) ? ei + 1 : ei};
    int const sj_prl{(sj > pmb->cjmp) ? sj - 1 : sj};
    int const ej_prl{(ej < pmb->cjmp) ? ej + 1 : ej};
    int const sk_prl{(sk > pmb->ckmp) ? sk - 1 : sk};
    int const ek_prl{(ek < pmb->ckmp) ? ek + 1 : ek};

    // [running ix]: op

    for (int n = sn; n<= en; ++n) {
      //-----------------------------------------------------------------------
      // [k, j, i]: interp. 3d
      for (int k = sk_prl; k < ek_prl; ++k) {
        int const fk_inj = 2 * (k - H_NCGHOST) + eo_offset;
        int const fk_prl = fk_inj + 1;

        for (int j = sj_prl; j < ej_prl; ++j) {
          int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
          int const fj_prl = fj_inj + 1;

          for (int i = si_prl; i < ei_prl; ++i) {
            int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
            int const fi_prl = fi_inj + 1;

            fine(n, fk_prl, fj_prl, fi_prl) = 0.;

            for (int dk=0; dk<H_SZ; ++dk) {
              int const ck_u = k + dk + 1;
              int const ck_l = k - dk;

              Real const lck = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-dk-1];

              for (int dj=0; dj<H_SZ; ++dj) {
                int const cj_u = j + dj + 1;
                int const cj_l = j - dj;

                Real const lckj = lck * InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-dj-1];

                for (int di=0; di<H_SZ; ++di) {
                  int const ci_u = i + di + 1;
                  int const ci_l = i - di;

                  Real const lckji = lckj * InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-di-1];

                  Real const fc_uuu = coarse(n, ck_u, cj_u, ci_u);
                  Real const fc_lll = coarse(n, ck_l, cj_l, ci_l);

                  Real const fc_luu = coarse(n, ck_l, cj_u, ci_u);
                  Real const fc_ulu = coarse(n, ck_u, cj_l, ci_u);
                  Real const fc_uul = coarse(n, ck_u, cj_u, ci_l);

                  Real const fc_llu = coarse(n, ck_l, cj_l, ci_u);
                  Real const fc_ull = coarse(n, ck_u, cj_l, ci_l);
                  Real const fc_lul = coarse(n, ck_l, cj_u, ci_l);

#ifdef DBG_SYMMETRIZE_P_OP
                  fine(n, fk_prl, fj_prl, fi_prl) += lckji * (
                    FloatingPoint::sum_associative(
                      fc_uuu, fc_lll, fc_uul, fc_llu,
                      fc_lul, fc_ulu, fc_luu, fc_ull
                    )
                  );
#else
                  fine(n, fk_prl, fj_prl, fi_prl) += lckji *
                    ((fc_uuu + fc_lll) +
                    (fc_uul + fc_llu) +
                    (fc_lul + fc_ulu) +
                    (fc_luu + fc_ull));
#endif // DBG_SYMMETRIZE_P_OP
                }
              }
            }
          }
        }
      } // (prl, prl, prl)
      //-----------------------------------------------------------------------

      //-----------------------------------------------------------------------
      // [k, j, i]: interp. 2d & inject 1d
      for (int k = sk_inj; k <= ek_inj; ++k) {
        int const fk_inj = 2 * (k - H_NCGHOST) + eo_offset;
        int const fk_prl = fk_inj + 1;
        (void)fk_prl; // DR: why is this not used?

        for (int j = sj_prl; j < ej_prl; ++j) {
          int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
          int const fj_prl = fj_inj + 1;

          for (int i = si_prl; i < ei_prl; ++i) {
            int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
            int const fi_prl = fi_inj + 1;

            fine(n, fk_inj, fj_prl, fi_prl) = 0.;

            for (int dj=0; dj<H_SZ; ++dj) {
              int const cj_u = j + dj + 1;
              int const cj_l = j - dj;

              Real const lcj = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-dj-1];

              for (int di=0; di<H_SZ; ++di) {
                int const ci_u = i + di + 1;
                int const ci_l = i - di;

                Real const lcji = lcj * InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-di-1];

                Real const fc_cuu = coarse(n, k, cj_u, ci_u);
                Real const fc_cul = coarse(n, k, cj_u, ci_l);
                Real const fc_clu = coarse(n, k, cj_l, ci_u);
                Real const fc_cll = coarse(n, k, cj_l, ci_l);

#ifdef DBG_SYMMETRIZE_P_OP
                fine(n, fk_inj, fj_prl, fi_prl) += lcji * (
                  FloatingPoint::sum_associative(
                    fc_cuu, fc_cll, fc_clu, fc_cul
                  )
                );
#else
                fine(n, fk_inj, fj_prl, fi_prl) += lcji * ((fc_cuu + fc_cll) + (fc_clu + fc_cul));
#endif // DBG_SYMMETRIZE_P_OP
              }
            }

          }
        }
      } // (inj, prl, prl)


      // [k, j, i]: interp. 2d & inject 1d
      for (int k = sk_prl; k < ek_prl; ++k) {
        int const fk_inj = 2 * (k - H_NCGHOST) + eo_offset;
        int const fk_prl = fk_inj + 1;

        for (int j = sj_inj; j <= ej_inj; ++j) {
          int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
          int const fj_prl = fj_inj + 1;
          (void)fj_prl; // DR: why is this not used?

          for (int i = si_prl; i < ei_prl; ++i) {
            int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
            int const fi_prl = fi_inj + 1;

            fine(n, fk_prl, fj_inj, fi_prl) = 0.;

            for (int dk=0; dk<H_SZ; ++dk) {
              int const ck_u = k + dk + 1;
              int const ck_l = k - dk;

              Real const lck = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-dk-1];

              for (int di=0; di<H_SZ; ++di) {
                int const ci_u = i + di + 1;
                int const ci_l = i - di;

                Real const lcki = lck * InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-di-1];

                Real const fc_ucu = coarse(n, ck_u, j, ci_u);
                Real const fc_ucl = coarse(n, ck_u, j, ci_l);
                Real const fc_lcu = coarse(n, ck_l, j, ci_u);
                Real const fc_lcl = coarse(n, ck_l, j, ci_l);

#ifdef DBG_SYMMETRIZE_P_OP
                fine(n, fk_prl, fj_inj, fi_prl) += lcki * (
                  FloatingPoint::sum_associative(
                    fc_lcu, fc_ucl, fc_ucu, fc_lcl
                  )
                );

#else
                fine(n, fk_prl, fj_inj, fi_prl) += lcki * ((fc_lcu + fc_ucl) + (fc_ucu + fc_lcl));
#endif // DBG_SYMMETRIZE_P_OP
              }
            }

          }
        }
      } // (prl, inj, prl)

      // [k, j, i]: interp. 2d & inject 1d
      for (int k = sk_prl; k < ek_prl; ++k) {
        int const fk_inj = 2 * (k - H_NCGHOST) + eo_offset;
        int const fk_prl = fk_inj + 1;

        for (int j = sj_prl; j < ej_prl; ++j) {
          int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
          int const fj_prl = fj_inj + 1;

          for (int i = si_inj; i <= ei_inj; ++i) {
            int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
            int const fi_prl = fi_inj + 1;
            (void)fi_prl;  // DR: why is this not used?

            fine(n, fk_prl, fj_prl, fi_inj) = 0.;

            for (int dk=0; dk<H_SZ; ++dk) {
              int const ck_u = k + dk + 1;
              int const ck_l = k - dk;

              Real const lck = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-dk-1];

              for (int dj=0; dj<H_SZ; ++dj) {
                int const cj_u = j + dj + 1;
                int const cj_l = j - dj;

                Real const lckj = lck * InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-dj-1];

                Real const fc_uuc = coarse(n, ck_u, cj_u, i);
                Real const fc_ulc = coarse(n, ck_u, cj_l, i);
                Real const fc_luc = coarse(n, ck_l, cj_u, i);
                Real const fc_llc = coarse(n, ck_l, cj_l, i);

#ifdef DBG_SYMMETRIZE_P_OP
                fine(n, fk_prl, fj_prl, fi_inj) += lckj * (
                  FloatingPoint::sum_associative(
                    fc_uuc, fc_llc, fc_luc, fc_ulc
                  )
                );
#else
                fine(n, fk_prl, fj_prl, fi_inj) += lckj * ((fc_uuc + fc_llc) + (fc_luc + fc_ulc));
#endif // DBG_SYMMETRIZE_P_OP
              }
            }

          }
        }
      } // (prl, prl, inj)
      //-----------------------------------------------------------------------

      //-----------------------------------------------------------------------
      // [k, j, i]: interp. 1d & inject 2d
      for (int k = sk_inj; k <= ek_inj; ++k) {
        int const fk_inj = 2 * (k - H_NCGHOST) + eo_offset;
        int const fk_prl = fk_inj + 1;
        (void)fk_prl;   // DR: why is this not used?

        for (int j = sj_inj; j <= ej_inj; ++j) {
          int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
          int const fj_prl = fj_inj + 1;
          (void)fj_prl;  // DR: why is this not used?

          for (int i = si_prl; i < ei_prl; ++i) {
            int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
            int const fi_prl = fi_inj + 1;

            fine(n, fk_inj, fj_inj, fi_prl) = 0.;

            for (int di=0; di<H_SZ; ++di) {
              int const ci_u = i + di + 1;
              int const ci_l = i - di;

              Real const lci = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-di-1];

              Real const fc_ccu = coarse(n, k, j, ci_u);
              Real const fc_ccl = coarse(n, k, j, ci_l);

              fine(n, fk_inj, fj_inj, fi_prl) += lci * (fc_ccl + fc_ccu);
            }

          }
        }
      } // (inj, inj, prl)


      // [k, j, i]: interp. 1d & inject 2d
      for (int k = sk_inj; k <= ek_inj; ++k) {
        int const fk_inj = 2 * (k - H_NCGHOST) + eo_offset;
        int const fk_prl = fk_inj + 1;
        (void)fk_prl;  // DR: why is this not used?

        for (int j = sj_prl; j < ej_prl; ++j) {
          int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
          int const fj_prl = fj_inj + 1;

          for (int i = si_inj; i <= ei_inj; ++i) {
            int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
            int const fi_prl = fi_inj + 1;
            (void)fi_prl;  // DR: why is this not used?

            fine(n, fk_inj, fj_prl, fi_inj) = 0.;

            for (int dj=0; dj<H_SZ; ++dj) {
              int const cj_u = j + dj + 1;
              int const cj_l = j - dj;

              Real const lcj = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-dj-1];

              Real const fc_cuc = coarse(n, k, cj_u, i);
              Real const fc_clc = coarse(n, k, cj_l, i);

              fine(n, fk_inj, fj_prl, fi_inj) += lcj * (fc_clc + fc_cuc);
            }

          }
        }
      } // (inj, prl, inj)

      // [k, j, i]: interp. 1d & inject 2d
      for (int k = sk_prl; k < ek_prl; ++k) {
        int const fk_inj = 2 * (k - H_NCGHOST) + eo_offset;
        int const fk_prl = fk_inj + 1;

        for (int j = sj_inj; j <= ej_inj; ++j) {
          int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
          int const fj_prl = fj_inj + 1;
          (void)fj_prl;  // DR: why is this not used?

          for (int i = si_inj; i <= ei_inj; ++i) {
            int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
            int const fi_prl = fi_inj + 1;
            (void)fi_prl;  // DR: why is this not used?

            fine(n, fk_prl, fj_inj, fi_inj) = 0.;

            for (int dk=0; dk<H_SZ; ++dk) {
              int const ck_u = k + dk + 1;
              int const ck_l = k - dk;

              Real const lck = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-dk-1];

              Real const fc_ucc = coarse(n, ck_u, j, i);
              Real const fc_lcc = coarse(n, ck_l, j, i);

              fine(n, fk_prl, fj_inj, fi_inj) += lck * (fc_lcc + fc_ucc);
            }

          }
        }
      } // (prl, inj, inj)
      //-----------------------------------------------------------------------

      //-----------------------------------------------------------------------
      // [k, j, i]: inject 3d
      for (int k = sk_inj; k <= ek_inj; ++k) {
        int const fk_inj = 2 * (k - H_NCGHOST) + eo_offset;
        int const fk_prl = fk_inj + 1;
        (void)fk_prl;  // DR: why is this not used?

        for (int j = sj_inj; j <= ej_inj; ++j) {
          int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
          int const fj_prl = fj_inj + 1;
          (void)fj_prl;  // DR: why is this not used?

          for (int i = si_inj; i <= ei_inj; ++i) {
            int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
            int const fi_prl = fi_inj + 1;
            (void)fi_prl;  // DR: why is this not used?

            fine(n, fk_inj, fj_inj, fi_inj) = coarse(n, k, j, i);

          }
        }
      } // (inj, inj, inj)
      //-----------------------------------------------------------------------

    } // function component loop

  } else if (pmb->pmy_mesh->ndim == 2) {
#if ISEVEN(NGHOST)
    int const eo_offset = 0;

    int const si_inj{(si)};
    int const sj_inj{(sj)};

    int const ei_inj{(ei)};
    int const ej_inj{(ej)};
#else
    int const eo_offset = -1;

    int const fis_inj{(2 * (si - H_NCGHOST) + eo_offset)};
    int const fie_inj{(2 * (ei - H_NCGHOST) + eo_offset)};

    int const fjs_inj{(2 * (sj - H_NCGHOST) + eo_offset)};
    int const fje_inj{(2 * (ej - H_NCGHOST) + eo_offset)};


    int const si_inj{(fis_inj < 0) ? si + 1 : si};
    int const ei_inj{(fie_inj > 2 * NGHOST + pmb->block_size.nx1) ? ei - 1 : ei};

    int const sj_inj{(fjs_inj < 0) ? sj + 1 : sj};
    int const ej_inj{(fje_inj > 2 * NGHOST + pmb->block_size.nx2) ? ej - 1 : ej};
#endif

    //-------------------------------------------------------------------------
    // bias offsets for prolongation (depends on location)
    int const si_prl{(si > pmb->cimp) ? si - 1 : si};
    int const ei_prl{(ei < pmb->cimp) ? ei + 1 : ei};
    int const sj_prl{(sj > pmb->cjmp) ? sj - 1 : sj};
    int const ej_prl{(ej < pmb->cjmp) ? ej + 1 : ej};


    // [running ix]: op

    for (int n = sn; n<= en; ++n) {

      //-----------------------------------------------------------------------
      // [j, i]: interp. 2d
      for (int j = sj_prl; j < ej_prl; ++j) {
        int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
        int const fj_prl = fj_inj + 1;

        for (int i = si_prl; i < ei_prl; ++i) {
          int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
          int const fi_prl = fi_inj + 1;

          fine(n, 0, fj_prl, fi_prl) = 0.;

          // apply stencil via Cartesian product relation
          for (int dj=0; dj<H_SZ; ++dj) {
            int const cj_u = j + dj + 1;
            int const cj_l = j - dj;

            Real const lcj = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-dj-1];

            for (int di=0; di<H_SZ; ++di) {
              int const ci_u = i + di + 1;
              int const ci_l = i - di;

              Real const lcji = lcj * InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-di-1];

              Real const fc_uu = coarse(n, 0, cj_u, ci_u);
              Real const fc_ul = coarse(n, 0, cj_u, ci_l);
              Real const fc_lu = coarse(n, 0, cj_l, ci_u);
              Real const fc_ll = coarse(n, 0, cj_l, ci_l);

#ifdef DBG_SYMMETRIZE_P_OP
              fine(n, 0, fj_prl, fi_prl) += lcji * FloatingPoint::sum_associative(
                fc_uu, fc_ll, fc_lu, fc_ul
              );
#else
              fine(n, 0, fj_prl, fi_prl) += lcji * ((fc_uu + fc_ll) + (fc_lu + fc_ul));
#endif // DBG_SYMMETRIZE_P_OP
            }
          }
        }
      } // (prl, prl)
      //-----------------------------------------------------------------------

      //-----------------------------------------------------------------------
      // [j, i]: (interp. 1d, inject 1d)
      for (int j = sj_prl; j < ej_prl; ++j) {
        int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
        int const fj_prl = fj_inj + 1;

        for (int i = si_inj; i <= ei_inj; ++i) {
          int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
          int const fi_prl = fi_inj + 1;
          (void)fi_prl;  // DR: why is this not used?

          fine(n, 0, fj_prl, fi_inj) = 0.;

          for (int dj=0; dj<H_SZ; ++dj) {
            int const cj_u = j + dj + 1;
            int const cj_l = j - dj;

            Real const lcj = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-dj-1];

            Real const fc_uc = coarse(n, 0, cj_u, i);
            Real const fc_lc = coarse(n, 0, cj_l, i);

            fine(n, 0, fj_prl, fi_inj) += lcj * (fc_uc + fc_lc);
          }
        }
      } // (prl, inj)

      // [j, i]: (inject 1d, interp. 1d)
      for (int j = sj_inj; j <= ej_inj; ++j) {
        int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
        int const fj_prl = fj_inj + 1;
        (void)fj_prl;  // DR: why is this not used?

        for (int i = si_prl; i < ei_prl; ++i) {
          int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
          int const fi_prl = fi_inj + 1;

          fine(n, 0, fj_inj, fi_prl) = 0.;

          for (int di=0; di<H_SZ; ++di) {
            int const ci_u = i + di + 1;
            int const ci_l = i - di;

            Real const lci = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-di-1];

            Real const fc_cu = coarse(n, 0, j, ci_u);
            Real const fc_cl = coarse(n, 0, j, ci_l);

            fine(n, 0, fj_inj, fi_prl) += lci * (fc_cl + fc_cu);
          }
        }
      } // (inj, prl)
      //-----------------------------------------------------------------------

      // [j, i]: inject 2d
      for (int j = sj_inj; j <= ej_inj; ++j) {
        int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
        int const fj_prl = fj_inj + 1;
        (void)fj_prl;  // DR: why is this not used?

        for (int i = si_inj; i <= ei_inj; ++i) {
          int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
          int const fi_prl = fi_inj + 1;
          (void)fi_prl;  // DR: why is this not used?

          // injected
          fine(n, 0, fj_inj, fi_inj) = coarse(n, 0, j, i);
        }
      } // (inj, inj)
      //-----------------------------------------------------------------------

    } // function component loop
    //-------------------------------------------------------------------------


  } else {

#if ISEVEN(NGHOST)
    int const eo_offset = 0;

    int const si_inj{(si)};
    int const ei_inj{(ei)};
#else
    int const eo_offset = -1;

    int const fis_inj{(2 * (si - H_NCGHOST) + eo_offset)};
    int const fie_inj{(2 * (ei - H_NCGHOST) + eo_offset)};

    int const si_inj{(fis_inj < 0) ? si + 1 : si};
    int const ei_inj{(fie_inj > 2 * NGHOST + pmb->block_size.nx1) ? ei - 1 : ei};
#endif

    //-------------------------------------------------------------------------
    // bias offsets for prolongation (depends on location)
    int const si_prl{(si > pmb->cimp) ? si - 1 : si};
    int const ei_prl{(ei < pmb->cimp) ? ei + 1 : ei};

    for (int n = sn; n<= en; ++n) {
      for (int i = si_prl; i < ei_prl; ++i) {
        int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
        int const fi_prl = fi_inj + 1;
        fine(n, 0, 0, fi_prl) = 0.;

        // apply stencil
        for (int di=0; di<H_SZ; ++di) {
          int const ci_u = i + di + 1;
          int const ci_l = i - di;

          Real const lc = InterpolateLagrangeUniform_opt<H_SZ>::coeff[H_SZ-di-1];

          Real const fc_u = coarse(n, 0, 0, ci_u);
          Real const fc_l = coarse(n, 0, 0, ci_l);

          fine(n, 0, 0, fi_prl) += lc * (fc_l + fc_u);
        }
      }
      //-----------------------------------------------------------------------
    } // function component loop

    // inject
    for (int n = sn; n<= en; ++n) {
      for (int i = si_inj; i <= ei_inj; ++i) {
        int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
        fine(n, 0, 0, fi_inj) = coarse(n, 0, 0, i);
      }
      //-----------------------------------------------------------------------
    } // function component loop

  }

  return;
}

void MeshRefinement::ProlongateCellCenteredXBCValues(
    const AthenaArray<Real> &coarse, AthenaArray<Real> &fine,
    int sn, int en, int csi, int cei, int csj, int cej, int csk, int cek)
{
  // ProlongateCellCenteredValues(coarse, fine, sn, en,
  //                              csi, cei, csj, cej, csk, cek);
  // return;

  MeshBlock *pmb = pmy_block_;

  // // BD: debug all rat. -------------------------------------------------------
  // const int cx_fsi = std::max(2 * (csi - pmb->cx_cis) + pmb->cx_is, 0);
  // const int cx_fsj = std::max(2 * (csj - pmb->cx_cjs) + pmb->cx_js, 0);
  // const int cx_fsk = std::max(2 * (csk - pmb->cx_cks) + pmb->cx_ks, 0);

  // const int cx_fei = std::min(2 * (cei - pmb->cx_cis) + pmb->cx_is + 1, pmb->ncells1-1);
  // const int cx_fej = std::min(2 * (cej - pmb->cx_cjs) + pmb->cx_js + 1, pmb->ncells2-1);
  // const int cx_fek = std::min(2 * (cek - pmb->cx_cks) + pmb->cx_ks + 1, pmb->ncells3-1);

  // AthenaArray<Real> & var_t = fine;
  // AthenaArray<Real> & var_s = const_cast<AthenaArray<Real>&>(coarse);
  // for(int n=sn; n<=en; ++n)
  // {
  //   const Real* const fcn_s = &(var_s(n,0,0,0));
  //   Real* fcn_t = &(var_t(n,0,0,0));

  //   // ind_interior_r_op->eval(fcn_t, fcn_s);
  //   ind_physical_p_op->eval_nn(fcn_t, fcn_s,
  //                              cx_fsi, cx_fei,
  //                              cx_fsj, cx_fej,
  //                              cx_fsk, cx_fek);
  // }
  // return;
  // // --------------------------------------------------------------------------

  // here H_SZ * 2 is resultant interpolant degree
  // formula compatible with ProlongateCellCenteredXGhosts
  const int H_SZ = (2 * NCGHOST_CX - NGHOST) / 2;
  // const int H_SZ = 2;

  if (pmb->block_size.nx3>1)
  { // 3D

    for(int n=sn; n<=en; ++n)
    for(int cx_ck=csk; cx_ck<=cek; cx_ck++)
    {
      // left child idx on fine grid
      const int cx_fk = 2 * (cx_ck - pmb->cx_cks) + pmb->cx_ks;

      for(int cx_cj=csj; cx_cj<=cej; cx_cj++)
      {
        // left child idx on fine grid
        const int cx_fj = 2 * (cx_cj - pmb->cx_cjs) + pmb->cx_js;

        for(int cx_ci=csi; cx_ci<=cei; cx_ci++)
        {
          // left child idx on fine grid
          const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

          // use templated ----------------------------------------------------
          if (true)
          {
            if((cx_fi >= 0) && (cx_fj >= 0) && (cx_fk >= 0))
              fine(n,cx_fk  ,cx_fj,  cx_fi  ) = 0.0;

            if((cx_fi >= 0) && (cx_fj >= 0) && (cx_fk+1 < pmb->ncells3))
              fine(n,cx_fk+1,cx_fj,  cx_fi  ) = 0.0;

            if((cx_fi >= 0) && (cx_fj+1 < pmb->ncells2) && (cx_fk >= 0))
              fine(n,cx_fk,  cx_fj+1,cx_fi  ) = 0.0;

            if((cx_fi+1 < pmb->ncells1) && (cx_fj >= 0) && (cx_fk >= 0))
              fine(n,cx_fk,  cx_fj,  cx_fi+1) = 0.0;

            if((cx_fi >= 0) && (cx_fj+1 < pmb->ncells2) && (cx_fk+1 < pmb->ncells3))
              fine(n,cx_fk+1,cx_fj+1,cx_fi  ) = 0.0;

            if((cx_fi+1 < pmb->ncells1) && (cx_fj >= 0) && (cx_fk+1 < pmb->ncells3))
              fine(n,cx_fk+1,cx_fj,  cx_fi+1) = 0.0;

            if((cx_fi+1 < pmb->ncells1) && (cx_fj+1 < pmb->ncells2) && (cx_fk >= 0))
              fine(n,cx_fk  ,cx_fj+1,cx_fi+1) = 0.0;

            if((cx_fi+1 < pmb->ncells1) && (cx_fj+1 < pmb->ncells2) && (cx_fk+1 < pmb->ncells3))
              fine(n,cx_fk+1,cx_fj+1,cx_fi+1) = 0.0;

            // #pragma unroll
            for (int dk=0; dk<2*H_SZ+1; ++dk)
            {
              // see 1d case for description of this
              Real const l_k = (
                InterpolateLagrangeUniformChildren<H_SZ>::coeff[dk]
              );

              const int cx_ckl = cx_ck-H_SZ+dk;
              const int cx_ckr = cx_ck+H_SZ-dk;

              // #pragma unroll
              for (int dj=0; dj<2*H_SZ+1; ++dj)
              {
                // see 1d case for description of this
                Real const l_kj = l_k * (
                  InterpolateLagrangeUniformChildren<H_SZ>::coeff[dj]
                );

                const int cx_cjl = cx_cj-H_SZ+dj;
                const int cx_cjr = cx_cj+H_SZ-dj;

                // #pragma unroll
                for (int di=0; di<2*H_SZ+1; ++di)
                {
                  Real const l_kji = l_kj * (
                    InterpolateLagrangeUniformChildren<H_SZ>::coeff[di]
                  );

                  const int cx_cil = cx_ci-H_SZ+di;
                  const int cx_cir = cx_ci+H_SZ-di;

                  Real const fc_lll = coarse(n,cx_ckl,cx_cjl,cx_cil);
                  Real const fc_lrr = coarse(n,cx_ckl,cx_cjr,cx_cir);
                  Real const fc_rrl = coarse(n,cx_ckr,cx_cjr,cx_cil);
                  Real const fc_rlr = coarse(n,cx_ckr,cx_cjl,cx_cir);
                  Real const fc_llr = coarse(n,cx_ckl,cx_cjl,cx_cir);
                  Real const fc_rll = coarse(n,cx_ckr,cx_cjl,cx_cil);
                  Real const fc_lrl = coarse(n,cx_ckl,cx_cjr,cx_cil);
                  Real const fc_rrr = coarse(n,cx_ckr,cx_cjr,cx_cir);

                  if((cx_fi >= 0) && (cx_fj >= 0) && (cx_fk >= 0))
                    fine(n,cx_fk  ,cx_fj,  cx_fi  ) += l_kji * fc_lll;

                  if((cx_fi >= 0) && (cx_fj >= 0) && (cx_fk+1 < pmb->ncells3))
                    fine(n,cx_fk+1,cx_fj,  cx_fi  ) += l_kji * fc_rll;

                  if((cx_fi >= 0) && (cx_fj+1 < pmb->ncells2) && (cx_fk >= 0))
                    fine(n,cx_fk,  cx_fj+1,cx_fi  ) += l_kji * fc_lrl;

                  if((cx_fi+1 < pmb->ncells1) && (cx_fj >= 0) && (cx_fk >= 0))
                    fine(n,cx_fk,  cx_fj,  cx_fi+1) += l_kji * fc_llr;

                  if((cx_fi >= 0) && (cx_fj+1 < pmb->ncells2) && (cx_fk+1 < pmb->ncells3))
                    fine(n,cx_fk+1,cx_fj+1,cx_fi  ) += l_kji * fc_rrl;

                  if((cx_fi+1 < pmb->ncells1) && (cx_fj >= 0) && (cx_fk+1 < pmb->ncells3))
                    fine(n,cx_fk+1,cx_fj,  cx_fi+1) += l_kji * fc_rlr;

                  if((cx_fi+1 < pmb->ncells1) && (cx_fj+1 < pmb->ncells2) && (cx_fk >= 0))
                    fine(n,cx_fk  ,cx_fj+1,cx_fi+1) += l_kji * fc_lrr;

                  if((cx_fi+1 < pmb->ncells1) && (cx_fj+1 < pmb->ncells2) && (cx_fk+1 < pmb->ncells3))
                    fine(n,cx_fk+1,cx_fj+1,cx_fi+1) += l_kji * fc_rrr;

                }

              }

            }

          }

        }

      }
    }



  }
  else if (pmb->block_size.nx2>1)
  { // 2D

    const int cx_fk = pmb->cx_ks, cx_ck = pmb->cx_cks;
    for(int n=sn; n<=en; ++n)
    for(int cx_cj=csj; cx_cj<=cej; cx_cj++)
    {
      // left child idx on fine grid
      const int cx_fj = 2 * (cx_cj - pmb->cx_cjs) + pmb->cx_js;

      for(int cx_ci=csi; cx_ci<=cei; cx_ci++)
      {
        // left child idx on fine grid
        const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

        // use templated ------------------------------------------------------
        if (true)
        {
          if((cx_fi >= 0) && (cx_fj >= 0))
            fine(n,cx_fk,cx_fj,  cx_fi  ) = 0.0;

          if((cx_fi+1 < pmb->ncells1) && (cx_fj >= 0))
            fine(n,cx_fk,cx_fj,  cx_fi+1) = 0.0;

          if((cx_fi >= 0) && (cx_fj+1 < pmb->ncells2))
            fine(n,cx_fk,cx_fj+1,cx_fi  ) = 0.0;

          if((cx_fi+1 < pmb->ncells1) && (cx_fj+1 < pmb->ncells2))
            fine(n,cx_fk,cx_fj+1,cx_fi+1) = 0.0;

          // #pragma unroll
          for (int dj=0; dj<2*H_SZ+1; ++dj)
          {
            // see 1d case for description of this
            Real const l_j = (
              InterpolateLagrangeUniformChildren<H_SZ>::coeff[dj]
            );

            const int cx_cjl = cx_cj-H_SZ+dj;
            const int cx_cjr = cx_cj+H_SZ-dj;

            // #pragma unroll
            for (int di=0; di<2*H_SZ+1; ++di)
            {
              Real const l_ji = l_j * (
                InterpolateLagrangeUniformChildren<H_SZ>::coeff[di]
              );

              const int cx_cil = cx_ci-H_SZ+di;
              const int cx_cir = cx_ci+H_SZ-di;

              Real const fc_ll = coarse(n,cx_ck,cx_cjl,cx_cil);
              Real const fc_lr = coarse(n,cx_ck,cx_cjl,cx_cir);
              Real const fc_rl = coarse(n,cx_ck,cx_cjr,cx_cil);
              Real const fc_rr = coarse(n,cx_ck,cx_cjr,cx_cir);

              if((cx_fi >= 0) && (cx_fj >= 0))
                fine(n,cx_fk,cx_fj,  cx_fi  ) += l_ji * fc_ll;
              if((cx_fi+1 < pmb->ncells1) && (cx_fj >= 0))
                fine(n,cx_fk,cx_fj,  cx_fi+1) += l_ji * fc_lr;
              if((cx_fi >= 0) && (cx_fj+1 < pmb->ncells2))
                fine(n,cx_fk,cx_fj+1,cx_fi  ) += l_ji * fc_rl;
              if((cx_fi+1 < pmb->ncells1) && (cx_fj+1 < pmb->ncells2))
                fine(n,cx_fk,cx_fj+1,cx_fi+1) += l_ji * fc_rr;

            }

          }

        }

      }

    }

  }
  else
  { // 1D
    const int cx_fj = pmb->cx_js, cx_cj = pmb->cx_cjs;
    const int cx_fk = pmb->cx_ks, cx_ck = pmb->cx_cks;

    for (int n=sn; n<=en; ++n)
    for (int cx_ci=csi; cx_ci<=cei; cx_ci++)
    {
      // left child idx on fine grid
      const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

      // use templated --------------------------------------------------------
      if(cx_fi >= 0)
        fine(n,cx_fk,cx_fj,cx_fi)   = 0.0;
      if(cx_fi+1 < pmb->ncells1)
        fine(n,cx_fk,cx_fj,cx_fi+1) = 0.0;

      #pragma unroll
      for (int di=0; di<2*H_SZ+1; ++di)
      {
        // left / right children coeffs
        // Real const l_li = InterpolateLagrangeUniformChildren<H_SZ>::coeff[di];
        // Real const l_ri = InterpolateLagrangeUniformChildren<H_SZ>::coeff[
        //   2 * H_SZ - di
        // ];

        // left / coeffs children coeffs
        // mask left to right vs right to left (see fc_l / fc_r below)
        Real const l_i = InterpolateLagrangeUniformChildren<H_SZ>::coeff[di];

        const int cx_cil = cx_ci-H_SZ+di;
        const int cx_cir = cx_ci+H_SZ-di;

        Real const fc_l = coarse(n,cx_ck,cx_cj,cx_cil);
        Real const fc_r = coarse(n,cx_ck,cx_cj,cx_cir);


        if(cx_fi >= 0)
          fine(n,cx_fk,cx_fj,cx_fi)   += l_i * fc_l;
        if(cx_fi+1 < pmb->ncells1)
          fine(n,cx_fk,cx_fj,cx_fi+1) += l_i * fc_r;
      }

      /*
      // deg 4 interp. --------------------------------------------------------
      // left child
      fine(n,cx_fk,cx_fj,cx_fi) = (
        -45.0  * coarse(n,cx_ck,cx_cj,cx_ci-2) +
        420.0  * coarse(n,cx_ck,cx_cj,cx_ci-1) +
        1890.0 * coarse(n,cx_ck,cx_cj,cx_ci  ) +
        -252.0 * coarse(n,cx_ck,cx_cj,cx_ci+1) +
        35.0   * coarse(n,cx_ck,cx_cj,cx_ci+2)
      ) / 2048.0;
      // right child
      fine(n,cx_fk,cx_fj,cx_fi+1) = (
        35.0   * coarse(n,cx_ck,cx_cj,cx_ci-2) +
        -252.0 * coarse(n,cx_ck,cx_cj,cx_ci-1) +
        1890.0 * coarse(n,cx_ck,cx_cj,cx_ci  ) +
        420.0  * coarse(n,cx_ck,cx_cj,cx_ci+1) +
        -45.0  * coarse(n,cx_ck,cx_cj,cx_ci+2)
      ) / 2048.0;
      // ----------------------------------------------------------------------
      */
    }


  }

  return;

}

void MeshRefinement::ProlongateCellCenteredXValues(
    const AthenaArray<Real> &coarse, AthenaArray<Real> &fine,
    int sn, int en, int csi, int cei, int csj, int cej, int csk, int cek)
{
  // ProlongateCellCenteredXBCValues(coarse, fine, sn, en,
  //                                 csi, cei, csj, cej, csk, cek);
  // return;

  MeshBlock *pmb = pmy_block_;

  // // BD: debug all rat. -------------------------------------------------------
  // const int cx_fsi = std::max(2 * (csi - pmb->cx_cis) + pmb->cx_is, 0);
  // const int cx_fsj = std::max(2 * (csj - pmb->cx_cjs) + pmb->cx_js, 0);
  // const int cx_fsk = std::max(2 * (csk - pmb->cx_cks) + pmb->cx_ks, 0);

  // const int cx_fei = std::min(2 * (cei - pmb->cx_cis) + pmb->cx_is + 1, pmb->ncells1-1);
  // const int cx_fej = std::min(2 * (cej - pmb->cx_cjs) + pmb->cx_js + 1, pmb->ncells2-1);
  // const int cx_fek = std::min(2 * (cek - pmb->cx_cks) + pmb->cx_ks + 1, pmb->ncells3-1);

  // AthenaArray<Real> & var_t = fine;
  // AthenaArray<Real> & var_s = const_cast<AthenaArray<Real>&>(coarse);

  // for(int n=sn; n<=en; ++n)
  // {
  //   const Real* const fcn_s = &(var_s(n,0,0,0));
  //   Real* fcn_t = &(var_t(n,0,0,0));

  //   // ind_interior_r_op->eval(fcn_t, fcn_s);
  //   ind_physical_p_op->eval_nn(fcn_t, fcn_s,
  //                              cx_fsi, cx_fei,
  //                              cx_fsj, cx_fej,
  //                              cx_fsk, cx_fek);
  // }
  // return;
  // // --------------------------------------------------------------------------

  // ProlongateCellCenteredXBCValues(coarse, fine,
  //   sn, en, csi, cei, csj, cej, csk, cek);
  // return;

  // here H_SZ * 2 is resultant interpolant degree
  // formula compatible with ProlongateCellCenteredXGhosts
  const int H_SZ = (2 * NCGHOST_CX - NGHOST) / 2 + 1;
  // const int H_SZ = 1;


  // for(int n=sn; n<=en; ++n)
  // for(int ci=pmb->cx_cis; ci<=pmb->cx_cie; ++ci)
  // {
  //   // left child idx on fundamental grid
  //   const int cx_fi = 2 * (ci - pmb->cx_cis) + pmb->cx_is;

  //   Real* x1_s = &(pco->x1v(NGHOST));
  //   Real* fcn_s = &(var_s(n,0,0,0));

  //   // coarse variable grids are constructed with CC ghost number
  //   const Real x1_t = pcoarsec->x1v(NCGHOST+(ci-NCGHOST_CX));

  //   var_t(n,0,0,ci) = Floater_Hormann::interp_1d(
  //     x1_t, x1_s,
  //     fcn_s,
  //     Ns_x1, d, NGHOST);
  // }

  // deal with odd ghosts through expanded scratch array
  // AthenaArray<Real> fscr;
  // fscr.NewAthenaArray(pmb->ncells1 + 2 * ((NGHOST % 2) != 0));

  // ffff
  /*
  AthenaArray<Real> &ncoarse = const_cast<AthenaArray<Real>&>(coarse);

  const int N = pmb->cx_ncc1 - 1; // # nodes - 1
  const int d = 4;

  AthenaArray<Real> cx_cx1v;
  cx_cx1v.NewAthenaArray(pmb->cx_ncc1);
  Real cx0 = pcoarsec->x1v(NCGHOST);
  Real cdx = pcoarsec->x1v(1) - pcoarsec->x1v(0);
  for(int i=0; i<pmb->cx_ncc1; ++i)
  {
    cx_cx1v(i) = (i - pmb->cx_dng) * cdx + pcoarsec->x1v(0);
  }


  for(int n=sn; n<=en; ++n)
  for(int ci=csi; ci<=cei; ++ci)
  {
    // left child idx on fundamental grid
    const int cx_fi = 2 * (ci - pmb->cx_cis) + pmb->cx_is;

    // Real* x1_s = &(pcoarsec->x1v(0));
    Real* x1_s = &(cx_cx1v(0));
    Real* fcn_s = &(ncoarse(n,0,0,0));

    // // const Real x1_t = x1_s[0] + ci * 0.01;
    // const Real x1_tl = pcoarsec->x1v(NCGHOST+(ci-NCGHOST_CX));
    // const Real x1_tr = pcoarsec->x1v(NCGHOST+(ci-NCGHOST_CX));


    if(cx_fi >= 0)
    {
      Real x1_tl = pmb->pcoord->x1v(cx_fi);
      fine(n,0,0,cx_fi) = interp_1d_FH(
        x1_tl, x1_s,
        fcn_s,
        N, d);
    }

    if(cx_fi+1 < pmb->ncells1)
    {
        Real x1_tr = pmb->pcoord->x1v(cx_fi+1);
        fine(n,0,0,cx_fi+1) = interp_1d_FH(
          x1_tr, x1_s,
          fcn_s,
          N, d);
    }



  }

  return;
  */

  // BD: TODO:
  // Use 1d scratch to pad to NGHOST % 2 = 0 then copy back
  // This deals with odd-ghosts

  // AthenaArray<Real> scr;
  // scr.NewAthenaArray(
  //   pmb->block_size.nx1 + )

  if (pmb->block_size.nx3>1)
  { // 3D

    for(int n=sn; n<=en; ++n)
    for(int cx_ck=csk; cx_ck<=cek; cx_ck++)
    {
      // left child idx on fine grid
      const int cx_fk = 2 * (cx_ck - pmb->cx_cks) + pmb->cx_ks;

      for(int cx_cj=csj; cx_cj<=cej; cx_cj++)
      {
        // left child idx on fine grid
        const int cx_fj = 2 * (cx_cj - pmb->cx_cjs) + pmb->cx_js;

        for(int cx_ci=csi; cx_ci<=cei; cx_ci++)
        {
          // left child idx on fine grid
          const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

          // use templated ----------------------------------------------------
          if (true)
          {
            // if((cx_fi >= 0) && (cx_fj >= 0))
            //   fine(n,cx_fk,cx_fj,  cx_fi  ) = 0.0;

            // if((cx_fi+1 < pmb->ncells1) && (cx_fj >= 0))
            //   fine(n,cx_fk,cx_fj,  cx_fi+1) = 0.0;

            // if((cx_fi >= 0) && (cx_fj+1 < pmb->ncells2))
            //   fine(n,cx_fk,cx_fj+1,cx_fi  ) = 0.0;

            // if((cx_fi+1 < pmb->ncells1) && (cx_fj+1 < pmb->ncells2))
            //   fine(n,cx_fk,cx_fj+1,cx_fi+1) = 0.0;

            if((cx_fi >= 0) && (cx_fj >= 0) && (cx_fk >= 0))
              fine(n,cx_fk  ,cx_fj,  cx_fi  ) = 0.0;

            if((cx_fi >= 0) && (cx_fj >= 0) && (cx_fk+1 < pmb->ncells3))
              fine(n,cx_fk+1,cx_fj,  cx_fi  ) = 0.0;

            if((cx_fi >= 0) && (cx_fj+1 < pmb->ncells2) && (cx_fk >= 0))
              fine(n,cx_fk,  cx_fj+1,cx_fi  ) = 0.0;

            if((cx_fi+1 < pmb->ncells1) && (cx_fj >= 0) && (cx_fk >= 0))
              fine(n,cx_fk,  cx_fj,  cx_fi+1) = 0.0;

            if((cx_fi >= 0) && (cx_fj+1 < pmb->ncells2) && (cx_fk+1 < pmb->ncells3))
              fine(n,cx_fk+1,cx_fj+1,cx_fi  ) = 0.0;

            if((cx_fi+1 < pmb->ncells1) && (cx_fj >= 0) && (cx_fk+1 < pmb->ncells3))
              fine(n,cx_fk+1,cx_fj,  cx_fi+1) = 0.0;

            if((cx_fi+1 < pmb->ncells1) && (cx_fj+1 < pmb->ncells2) && (cx_fk >= 0))
              fine(n,cx_fk  ,cx_fj+1,cx_fi+1) = 0.0;

            if((cx_fi+1 < pmb->ncells1) && (cx_fj+1 < pmb->ncells2) && (cx_fk+1 < pmb->ncells3))
              fine(n,cx_fk+1,cx_fj+1,cx_fi+1) = 0.0;

            #pragma unroll
            for (int dk=0; dk<2*H_SZ+1; ++dk)
            {
              // see 1d case for description of this
              Real const l_k = (
                InterpolateLagrangeUniformChildren<H_SZ>::coeff[dk]
              );

              const int cx_ckl = cx_ck-H_SZ+dk;
              const int cx_ckr = cx_ck+H_SZ-dk;

              #pragma unroll
              for (int dj=0; dj<2*H_SZ+1; ++dj)
              {
                // see 1d case for description of this
                Real const l_kj = l_k * (
                  InterpolateLagrangeUniformChildren<H_SZ>::coeff[dj]
                );

                const int cx_cjl = cx_cj-H_SZ+dj;
                const int cx_cjr = cx_cj+H_SZ-dj;

                #pragma unroll
                for (int di=0; di<2*H_SZ+1; ++di)
                {
                  Real const l_kji = l_kj * (
                    InterpolateLagrangeUniformChildren<H_SZ>::coeff[di]
                  );

                  const int cx_cil = cx_ci-H_SZ+di;
                  const int cx_cir = cx_ci+H_SZ-di;

                  Real const fc_lll = coarse(n,cx_ckl,cx_cjl,cx_cil);
                  Real const fc_lrr = coarse(n,cx_ckl,cx_cjr,cx_cir);
                  Real const fc_rrl = coarse(n,cx_ckr,cx_cjr,cx_cil);
                  Real const fc_rlr = coarse(n,cx_ckr,cx_cjl,cx_cir);
                  Real const fc_llr = coarse(n,cx_ckl,cx_cjl,cx_cir);
                  Real const fc_rll = coarse(n,cx_ckr,cx_cjl,cx_cil);
                  Real const fc_lrl = coarse(n,cx_ckl,cx_cjr,cx_cil);
                  Real const fc_rrr = coarse(n,cx_ckr,cx_cjr,cx_cir);

                  if((cx_fi >= 0) && (cx_fj >= 0) && (cx_fk >= 0))
                    fine(n,cx_fk  ,cx_fj,  cx_fi  ) += l_kji * fc_lll;

                  if((cx_fi >= 0) && (cx_fj >= 0) && (cx_fk+1 < pmb->ncells3))
                    fine(n,cx_fk+1,cx_fj,  cx_fi  ) += l_kji * fc_rll;

                  if((cx_fi >= 0) && (cx_fj+1 < pmb->ncells2) && (cx_fk >= 0))
                    fine(n,cx_fk,  cx_fj+1,cx_fi  ) += l_kji * fc_lrl;

                  if((cx_fi+1 < pmb->ncells1) && (cx_fj >= 0) && (cx_fk >= 0))
                    fine(n,cx_fk,  cx_fj,  cx_fi+1) += l_kji * fc_llr;

                  if((cx_fi >= 0) && (cx_fj+1 < pmb->ncells2) && (cx_fk+1 < pmb->ncells3))
                    fine(n,cx_fk+1,cx_fj+1,cx_fi  ) += l_kji * fc_rrl;

                  if((cx_fi+1 < pmb->ncells1) && (cx_fj >= 0) && (cx_fk+1 < pmb->ncells3))
                    fine(n,cx_fk+1,cx_fj,  cx_fi+1) += l_kji * fc_rlr;

                  if((cx_fi+1 < pmb->ncells1) && (cx_fj+1 < pmb->ncells2) && (cx_fk >= 0))
                    fine(n,cx_fk  ,cx_fj+1,cx_fi+1) += l_kji * fc_lrr;

                  if((cx_fi+1 < pmb->ncells1) && (cx_fj+1 < pmb->ncells2) && (cx_fk+1 < pmb->ncells3))
                    fine(n,cx_fk+1,cx_fj+1,cx_fi+1) += l_kji * fc_rrr;

                }

              }

            }

          }

        }

      }
    }



  }
  else if (pmb->block_size.nx2>1)
  { // 2D


    // const int cx_fk = pmb->cx_ks, cx_ck = pmb->cx_cks;
    // for (int n=sn; n<=en; ++n)
    // for (int cx_cj=csj; cx_cj<=cej; cx_cj++)
    // {
    //   // left child idx on fine grid
    //   const int cx_fj = 2 * (cx_cj - pmb->cx_cjs) + pmb->cx_js;

    //   for (int cx_ci=csi; cx_ci<=cei; cx_ci++)
    //   {
    //     // left child idx on fine grid
    //     const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

    //     // fine(n,cx_fk,cx_fj,cx_fi) = coarse(n,cx_ck,cx_cj,cx_ci);
    //     // fine(n,cx_fk,cx_fj,cx_fi+1) = coarse(n,cx_ck,cx_cj,cx_ci+1);
    //     // fine(n,cx_fk,cx_fj+1,cx_fi) = coarse(n,cx_ck,cx_cj+1,cx_ci);
    //     // fine(n,cx_fk,cx_fj+1,cx_fi+1) = coarse(n,cx_ck,cx_cj+1,cx_ci+1);

    //     // use templated ------------------------------------------------------
    //     if (false)
    //     {
    //     fine(n,cx_fk,cx_fj,  cx_fi  ) = 0.0;
    //     fine(n,cx_fk,cx_fj,  cx_fi+1) = 0.0;
    //     fine(n,cx_fk,cx_fj+1,cx_fi  ) = 0.0;
    //     fine(n,cx_fk,cx_fj+1,cx_fi+1) = 0.0;

    //     #pragma unroll
    //     for (int dj=0; dj<2*H_SZ+1; ++dj)
    //     {
    //       // left / right children coeffs
    //       Real const l_clj = (
    //         InterpolateLagrangeUniformChildren<H_SZ>::coeff[dj]
    //       );
    //       Real const l_crj = InterpolateLagrangeUniformChildren<H_SZ>::coeff[
    //         2 * H_SZ - dj
    //       ];

    //       const int cx_cjl = cx_cj-H_SZ+dj;
    //       const int cx_cjr = cx_cj+H_SZ-dj;


    //       #pragma unroll
    //       for (int di=0; di<2*H_SZ+1; ++di)
    //       {
    //         Real const l_cli = (
    //           InterpolateLagrangeUniformChildren<H_SZ>::coeff[di]
    //         );
    //         Real const l_cri = (
    //             InterpolateLagrangeUniformChildren<H_SZ>::coeff[
    //               2 * H_SZ - di
    //             ]
    //         );

    //         const int cx_cil = cx_ci-H_SZ+di;
    //         const int cx_cir = cx_ci+H_SZ-di;

    //         Real const fc_ll = coarse(n,cx_ck,cx_cjl,cx_cil);
    //         Real const fc_lr = coarse(n,cx_ck,cx_cjl,cx_cir);
    //         Real const fc_rl = coarse(n,cx_ck,cx_cjr,cx_cil);
    //         Real const fc_rr = coarse(n,cx_ck,cx_cjr,cx_cir);

    //         /*
    //         Real const fc_ll = coarse(n,cx_ck,cx_cj-H_SZ+dj,cx_ci-H_SZ+di);
    //         Real const fc_lr = coarse(n,cx_ck,cx_cj-H_SZ+dj,cx_ci+H_SZ-di);
    //         Real const fc_rl = coarse(n,cx_ck,cx_cj+H_SZ-dj,cx_ci-H_SZ+di);
    //         Real const fc_rr = coarse(n,cx_ck,cx_cj+H_SZ-dj,cx_ci+H_SZ-di);

    //         fine(n,cx_fk,cx_fj,  cx_fi  ) += l_clj * l_cli * fc_ll;
    //         fine(n,cx_fk,cx_fj,  cx_fi+1) += l_clj * l_cri * fc_lr;
    //         fine(n,cx_fk,cx_fj+1,cx_fi  ) += l_crj * l_cli * fc_rl;
    //         fine(n,cx_fk,cx_fj+1,cx_fi+1) += l_crj * l_cri * fc_rr;
    //         */

    //         // fine(n,cx_fk,cx_fj,  cx_fi  ) += l_cli * l_clj * (fc_ll + fc_lr + fc_rl + fc_rr);
    //         // fine(n,cx_fk,cx_fj,  cx_fi+1) += l_cri * l_clj * (fc_ll + fc_lr + fc_rl + fc_rr);
    //         // fine(n,cx_fk,cx_fj+1,  cx_fi  ) += l_cli * l_crj * (fc_ll + fc_lr + fc_rl + fc_rr);
    //         // fine(n,cx_fk,cx_fj+1,  cx_fi+1) += l_cri * l_crj * (fc_ll + fc_lr + fc_rl + fc_rr);

    //         fine(n,cx_fk,cx_fj,  cx_fi  ) += l_clj * l_cli * fc_ll;
    //         fine(n,cx_fk,cx_fj,  cx_fi+1) += l_clj * l_cri * fc_lr;
    //         fine(n,cx_fk,cx_fj+1,cx_fi  ) += l_crj * l_cli * fc_rl;
    //         fine(n,cx_fk,cx_fj+1,cx_fi+1) += l_crj * l_cri * fc_rr;

    //       }

    //     }

    //     }

    //   }

    // }

    // debug with poly injection on coarse 1 / 2
    if(false)
    {
      AthenaArray<Real> &ncoarse = const_cast<AthenaArray<Real>&>(coarse);
      ncoarse.Fill(0);

      AthenaArray<Real> cx_cx1v;
      AthenaArray<Real> cx_cx2v;
      cx_cx1v.NewAthenaArray(pmb->cx_ncc1);
      cx_cx2v.NewAthenaArray(pmb->cx_ncc2);

      Real cdx1 = pcoarsec->x1v(1) - pcoarsec->x1v(0);
      Real cdx2 = pcoarsec->x2v(1) - pcoarsec->x2v(0);

      for(int i=0; i<pmb->cx_ncc1; ++i)
        cx_cx1v(i) = (i - pmb->cx_dng) * cdx1 + pcoarsec->x1v(0);

      for(int j=0; j<pmb->cx_ncc2; ++j)
        cx_cx2v(j) = (j - pmb->cx_dng) * cdx2 + pcoarsec->x2v(0);


      for(int cj=0; cj<pmb->cx_ncc2; ++cj)
      for(int ci=0; ci<pmb->cx_ncc1; ++ci)
      {
        ncoarse(0,0,cj,ci) = (
          std::pow(cx_cx2v(cj), 3.) * std::pow(cx_cx1v(ci), 2.)
        );
      }

      fine.Fill(0);

      csi = NCGHOST_CX;
      csj = NCGHOST_CX;

      cei = pmb->cx_ncc1 - NCGHOST_CX - 1;
      cej = pmb->cx_ncc2 - NCGHOST_CX - 1;

    }

    const int cx_fk = pmb->cx_ks, cx_ck = pmb->cx_cks;
    for(int n=sn; n<=en; ++n)
    for(int cx_cj=csj; cx_cj<=cej; cx_cj++)
    {
      // left child idx on fine grid
      const int cx_fj = 2 * (cx_cj - pmb->cx_cjs) + pmb->cx_js;

      for(int cx_ci=csi; cx_ci<=cei; cx_ci++)
      {
        // left child idx on fine grid
        const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

        // use templated ------------------------------------------------------
        if (true)
        {
          if((cx_fi >= 0) && (cx_fj >= 0))
            fine(n,cx_fk,cx_fj,  cx_fi  ) = 0.0;

          if((cx_fi+1 < pmb->ncells1) && (cx_fj >= 0))
            fine(n,cx_fk,cx_fj,  cx_fi+1) = 0.0;

          if((cx_fi >= 0) && (cx_fj+1 < pmb->ncells2))
            fine(n,cx_fk,cx_fj+1,cx_fi  ) = 0.0;

          if((cx_fi+1 < pmb->ncells1) && (cx_fj+1 < pmb->ncells2))
            fine(n,cx_fk,cx_fj+1,cx_fi+1) = 0.0;

          #pragma unroll
          for (int dj=0; dj<2*H_SZ+1; ++dj)
          {
            // see 1d case for description of this
            Real const l_j = (
              InterpolateLagrangeUniformChildren<H_SZ>::coeff[dj]
            );

            const int cx_cjl = cx_cj-H_SZ+dj;
            const int cx_cjr = cx_cj+H_SZ-dj;

            #pragma unroll
            for (int di=0; di<2*H_SZ+1; ++di)
            {
              Real const l_ji = l_j * (
                InterpolateLagrangeUniformChildren<H_SZ>::coeff[di]
              );

              const int cx_cil = cx_ci-H_SZ+di;
              const int cx_cir = cx_ci+H_SZ-di;

              Real const fc_ll = coarse(n,cx_ck,cx_cjl,cx_cil);
              Real const fc_lr = coarse(n,cx_ck,cx_cjl,cx_cir);
              Real const fc_rl = coarse(n,cx_ck,cx_cjr,cx_cil);
              Real const fc_rr = coarse(n,cx_ck,cx_cjr,cx_cir);

              if((cx_fi >= 0) && (cx_fj >= 0))
                fine(n,cx_fk,cx_fj,  cx_fi  ) += l_ji * fc_ll;
              if((cx_fi+1 < pmb->ncells1) && (cx_fj >= 0))
                fine(n,cx_fk,cx_fj,  cx_fi+1) += l_ji * fc_lr;
              if((cx_fi >= 0) && (cx_fj+1 < pmb->ncells2))
                fine(n,cx_fk,cx_fj+1,cx_fi  ) += l_ji * fc_rl;
              if((cx_fi+1 < pmb->ncells1) && (cx_fj+1 < pmb->ncells2))
                fine(n,cx_fk,cx_fj+1,cx_fi+1) += l_ji * fc_rr;

            }

          }

        }

      }

    }

    // debug with poly injection on coarse 2 / 2
    if (false)
    {
      fine.print_all("% 3.2e", false, false);
      fine.Fill(0);

      for(int cx_cj=csj; cx_cj<=cej; cx_cj++)
      {
        // left child idx on fine grid
        const int cx_fj = 2 * (cx_cj - pmb->cx_cjs) + pmb->cx_js;

        for(int cx_ci=csi; cx_ci<=cei; cx_ci++)
        {
          // left child idx on fine grid
          const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

          Real powx1vl = std::pow(pmb->pcoord->x1v(cx_fi), 2.);
          Real powx1vr = std::pow(pmb->pcoord->x1v(cx_fi+1), 2.);

          Real powx2vl = std::pow(pmb->pcoord->x2v(cx_fj), 3.);
          Real powx2vr = std::pow(pmb->pcoord->x2v(cx_fj+1), 3.);

          fine(0,cx_fk,cx_fj,  cx_fi  ) = powx2vl * powx1vl;
          fine(0,cx_fk,cx_fj,  cx_fi+1) = powx2vl * powx1vr;
          fine(0,cx_fk,cx_fj+1,cx_fi  ) = powx2vr * powx1vl;
          fine(0,cx_fk,cx_fj+1,cx_fi+1) = powx2vr * powx1vr;

        }
      }

      fine.print_all("% 3.2e", false, false);

      fine.print_all("% 3.2e", false, false);
    }

  }
  else
  { // 1D
    const int cx_fj = pmb->cx_js, cx_cj = pmb->cx_cjs;
    const int cx_fk = pmb->cx_ks, cx_ck = pmb->cx_cks;

    for (int n=sn; n<=en; ++n)
    for (int cx_ci=csi; cx_ci<=cei; cx_ci++)
    {
      // left child idx on fine grid
      const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

      // use templated --------------------------------------------------------
      if(cx_fi >= 0)
        fine(n,cx_fk,cx_fj,cx_fi)   = 0.0;
      if(cx_fi+1 < pmb->ncells1)
        fine(n,cx_fk,cx_fj,cx_fi+1) = 0.0;

      #pragma unroll
      for (int di=0; di<2*H_SZ+1; ++di)
      {
        // left / right children coeffs
        // Real const l_li = InterpolateLagrangeUniformChildren<H_SZ>::coeff[di];
        // Real const l_ri = InterpolateLagrangeUniformChildren<H_SZ>::coeff[
        //   2 * H_SZ - di
        // ];

        // left / coeffs children coeffs
        // mask left to right vs right to left (see fc_l / fc_r below)
        Real const l_i = InterpolateLagrangeUniformChildren<H_SZ>::coeff[di];

        const int cx_cil = cx_ci-H_SZ+di;
        const int cx_cir = cx_ci+H_SZ-di;

        Real const fc_l = coarse(n,cx_ck,cx_cj,cx_cil);
        Real const fc_r = coarse(n,cx_ck,cx_cj,cx_cir);


        if(cx_fi >= 0)
          fine(n,cx_fk,cx_fj,cx_fi)   += l_i * fc_l;
        if(cx_fi+1 < pmb->ncells1)
          fine(n,cx_fk,cx_fj,cx_fi+1) += l_i * fc_r;
      }

      // deg 2 interp. --------------------------------------------------------
      // // left child
      // fine(n,cx_fk,cx_fj,cx_fi) = (
      //   5.0  * coarse(n,cx_ck,cx_cj,cx_ci-1) +
      //   30.0 * coarse(n,cx_ck,cx_cj,cx_ci  ) +
      //   -3.0 * coarse(n,cx_ck,cx_cj,cx_ci+1)
      // ) / 32.0;
      // // right child
      // fine(n,cx_fk,cx_fj,cx_fi+1) = (
      //   -3.0 * coarse(n,cx_ck,cx_cj,cx_ci-1) +
      //   30.0 * coarse(n,cx_ck,cx_cj,cx_ci  ) +
      //   5.0  * coarse(n,cx_ck,cx_cj,cx_ci+1)
      // ) / 32.0;
      // ----------------------------------------------------------------------

      // deg 4 interp. --------------------------------------------------------
      // left child
      // fine(n,cx_fk,cx_fj,cx_fi) = (
      //   -45.0  * coarse(n,cx_ck,cx_cj,cx_ci-2) +
      //   420.0  * coarse(n,cx_ck,cx_cj,cx_ci-1) +
      //   1890.0 * coarse(n,cx_ck,cx_cj,cx_ci  ) +
      //   -252.0 * coarse(n,cx_ck,cx_cj,cx_ci+1) +
      //   35.0   * coarse(n,cx_ck,cx_cj,cx_ci+2)
      // ) / 2048.0;
      // // right child
      // fine(n,cx_fk,cx_fj,cx_fi+1) = (
      //   35.0   * coarse(n,cx_ck,cx_cj,cx_ci-2) +
      //   -252.0 * coarse(n,cx_ck,cx_cj,cx_ci-1) +
      //   1890.0 * coarse(n,cx_ck,cx_cj,cx_ci  ) +
      //   420.0  * coarse(n,cx_ck,cx_cj,cx_ci+1) +
      //   -45.0  * coarse(n,cx_ck,cx_cj,cx_ci+2)
      // ) / 2048.0;
      // ----------------------------------------------------------------------

      // deg 6 interp. --------------------------------------------------------
      // left child
      // fine(n,cx_fk,cx_fj,cx_fi) = (
      //   273.0   * coarse(n,cx_ck,cx_cj,cx_ci-3) +
      //   -2574.0 * coarse(n,cx_ck,cx_cj,cx_ci-2) +
      //   15015.0 * coarse(n,cx_ck,cx_cj,cx_ci-1) +
      //   60060.0 * coarse(n,cx_ck,cx_cj,cx_ci  ) +
      //   -9009.0 * coarse(n,cx_ck,cx_cj,cx_ci+1) +
      //   2002.0  * coarse(n,cx_ck,cx_cj,cx_ci+2) +
      //   -231.0  * coarse(n,cx_ck,cx_cj,cx_ci+3)
      // ) / 65536.0;
      // // right child
      // fine(n,cx_fk,cx_fj,cx_fi+1) = (
      //   -231.0  * coarse(n,cx_ck,cx_cj,cx_ci-3) +
      //   2002.0  * coarse(n,cx_ck,cx_cj,cx_ci-2) +
      //   -9009.0 * coarse(n,cx_ck,cx_cj,cx_ci-1) +
      //   60060.0 * coarse(n,cx_ck,cx_cj,cx_ci  ) +
      //   15015.0 * coarse(n,cx_ck,cx_cj,cx_ci+1) +
      //   -2574.0 * coarse(n,cx_ck,cx_cj,cx_ci+2) +
      //   273.0   * coarse(n,cx_ck,cx_cj,cx_ci+3)
      // ) / 65536.0;
      // ----------------------------------------------------------------------


    }


  }

  return;

}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::CheckRefinementCondition()
//  \brief Check refinement criteria

void MeshRefinement::CheckRefinementCondition() {
  MeshBlock *pmb = pmy_block_;
  int ret = 0, aret = -1;
  refine_flag_ = 0;

  // *** should be implemented later ***
  // loop-over refinement criteria
  if (AMRFlag_ != nullptr)
    ret = AMRFlag_(pmb);
  aret = std::max(aret,ret);

  if (aret >= 0)
    deref_count_ = 0;
  if (aret > 0) {
    if (pmb->loc.level == pmb->pmy_mesh->max_level) {
      refine_flag_ = 0;
    } else {
      refine_flag_ = 1;
    }
  } else if (aret < 0) {
    if (pmb->loc.level == pmb->pmy_mesh->root_level) {
      refine_flag_ = 0;
      deref_count_ = 0;
    } else {
      deref_count_++;
      int ec = 0, js, je, ks, ke;
      if (pmb->block_size.nx2 > 1) {
        js = -1;
        je = 1;
      } else {
        js = 0;
        je = 0;
      }
      if (pmb->block_size.nx3 > 1) {
        ks = -1;
        ke = 1;
      } else {
        ks = 0;
        ke = 0;
      }
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=-1; i<=1; i++)
            if (pmb->pbval->nblevel[k+1][j+1][i+1]>pmb->loc.level) ec++;
        }
      }
      if (ec > 0) {
        refine_flag_ = 0;
      } else {
        if (deref_count_ >= deref_threshold_) {
          refine_flag_ = -1;
        } else {
          refine_flag_ = 0;
        }
      }
    }
  }
  return;
}

// TODO(felker): consider merging w/ MeshBlock::pvars_cc, etc. See meshblock.cpp
int MeshRefinement::AddToRefinementVC(AthenaArray<Real> *pvar_in,
                                    AthenaArray<Real> *pcoarse_in) {
  pvars_vc_.push_back(std::make_tuple(pvar_in, pcoarse_in));
  return static_cast<int>(pvars_vc_.size() - 1);
}

int MeshRefinement::AddToRefinementCC(AthenaArray<Real> *pvar_in,
                                    AthenaArray<Real> *pcoarse_in) {
  pvars_cc_.push_back(std::make_tuple(pvar_in, pcoarse_in));
  return static_cast<int>(pvars_cc_.size() - 1);
}

int MeshRefinement::AddToRefinementFC(FaceField *pvar_fc, FaceField *pcoarse_fc) {
  pvars_fc_.push_back(std::make_tuple(pvar_fc, pcoarse_fc));
  return static_cast<int>(pvars_fc_.size() - 1);
}

int MeshRefinement::AddToRefinementCX(AthenaArray<Real> *pvar_in,
                                      AthenaArray<Real> *pcoarse_in) {
  pvars_cx_.push_back(std::make_tuple(pvar_in, pcoarse_in));
  return static_cast<int>(pvars_cx_.size() - 1);
}

// Currently, only called in 2x functions in bvals_refine.cpp:
// ----------
// - BoundaryValues::RestrictGhostCellsOnSameLevel()--- to perform additional
// restriction on primitive Hydro standard/coarse arrays (only for GR) without changing
// the var_cc/coarse_buf pointer members of the HydroBoundaryVariable.

// - BoundaryValues::ProlongateGhostCells()--- to ensure prolongation occurs on conserved
// (not primitive) variable standard/coarse arrays for Hydro, PassiveScalars

// Should probably consolidate this function and std::vector of tuples with
// BoundaryVariable interface ptr members. Too much independent switching of ptrs!
// ----------
// Even though we currently do not have special GR functionality planned for
// PassiveScalars::coarse_r_ like Hydro::coarse_prim_
// (it is never transferred in Mesh::LoadBalancingAndAdaptiveMeshRefinement)
// the physical (non-periodic) boundary functions will still apply only to the PRIMITIVE
// scalar variable arrays, thus S/AMR demand 1) AthenaArray<Real> PassiveScalars::coarse_r
// 2) ability to switch (s, coarse_s) and (r, coarse_r) ptrs in MeshRefinement::bvals_cc_

void MeshRefinement::SetHydroRefinement(HydroBoundaryQuantity hydro_type) {
  // TODO(felker): make more general so it can be used as SetPassiveScalarsRefinement()
  // e.g. refer to "int Hydro::refinement_idx" instead of assuming that the correct tuple
  // is in the first vector entry
  Hydro *ph = pmy_block_->phydro;
  // hard-coded assumption that, if multilevel, then Hydro is always present
  // and enrolled in mesh refinement in the first pvars_cc_ vector entry
  switch (hydro_type) {
    case (HydroBoundaryQuantity::cons): {
      pvars_cc_.front() = std::make_tuple(&ph->u, &ph->coarse_cons_);
      break;
    }
    case (HydroBoundaryQuantity::prim): {
      pvars_cc_.front() = std::make_tuple(&ph->w, &ph->coarse_prim_);
      break;
    }
  }
  return;
}
