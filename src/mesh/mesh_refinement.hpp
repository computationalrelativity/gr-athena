#ifndef MESH_MESH_REFINEMENT_HPP_
#define MESH_MESH_REFINEMENT_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mesh_refinement.hpp
//  \brief defines MeshRefinement class used for static/adaptive mesh refinement

// C headers

// C++ headers
#include <tuple>
#include <vector>

// Athena++ headers
#include "../athena.hpp"                    // Real
#include "../athena_arrays.hpp"             // AthenaArray
#include "../utils/finite_differencing.hpp"
#include "../utils/interp_barycentric.hpp"  // New interpolation ops

// MPI headers
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// ----------------------------------------------------------------------------
using namespace numprox::interpolation;
// ----------------------------------------------------------------------------

class MeshBlock;
class ParameterInput;
class Coordinates;
struct FaceField;
class BoundaryValues;
class FaceCenteredBoundaryVariable;
class VertexCenteredBoundaryVariable;
class HydroBoundaryVariable;

//----------------------------------------------------------------------------------------
//! \class MeshRefinement
//  \brief

class MeshRefinement {
  // needs to access pcoarsec in ProlongateBoundaries() for passing to BoundaryFunc()
  friend class BoundaryValues;
  // needs to access refine_flag_ in Mesh::AdaptiveMeshRefinement(). Make var public?
  friend class Mesh;

 public:
  MeshRefinement(MeshBlock *pmb, ParameterInput *pin);
  ~MeshRefinement();

  // functions
  void RestrictCellCenteredValues(const AthenaArray<Real> &fine,
                                  AthenaArray<Real> &coarse, int sn, int en,
                                  int csi, int cei, int csj, int cej, int csk, int cek);
  void RestrictFieldX1(const AthenaArray<Real> &fine, AthenaArray<Real> &coarse,
                       int csi, int cei, int csj, int cej, int csk, int cek);
  void RestrictFieldX2(const AthenaArray<Real> &fine, AthenaArray<Real> &coarse,
                       int csi, int cei, int csj, int cej, int csk, int cek);
  void RestrictFieldX3(const AthenaArray<Real> &fine, AthenaArray<Real> &coarse,
                       int csi, int cei, int csj, int cej, int csk, int cek);

  void RestrictVertexCenteredValues(const AthenaArray<Real> &fine,
                                    AthenaArray<Real> &coarse, int sn, int en,
                                    int csi, int cei, int csj, int cej, int csk, int cek);
  void RestrictTwiceToBufferVertexCenteredValues(
    const AthenaArray<Real> &fine,
    Real *buf,
    int sn, int en,
    int csi, int cei, int csj, int cej, int csk, int cek, int &offset);
  void ProlongateCellCenteredValues(const AthenaArray<Real> &coarse,
                                    AthenaArray<Real> &fine, int sn, int en,
                                    int si, int ei, int sj, int ej, int sk, int ek);
  void ProlongateSharedFieldX1(const AthenaArray<Real> &coarse, AthenaArray<Real> &fine,
                               int si, int ei, int sj, int ej, int sk, int ek);
  void ProlongateSharedFieldX2(const AthenaArray<Real> &coarse, AthenaArray<Real> &fine,
                               int si, int ei, int sj, int ej, int sk, int ek);
  void ProlongateSharedFieldX3(const AthenaArray<Real> &coarse, AthenaArray<Real> &fine,
                               int si, int ei, int sj, int ej, int sk, int ek);
  void ProlongateInternalField(FaceField &fine,
                               int si, int ei, int sj, int ej, int sk, int ek);

  void ProlongateVertexCenteredValues(const AthenaArray<Real> &coarse,
                                      AthenaArray<Real> &fine, int sn, int en,
                                      int si, int ei, int sj, int ej, int sk, int ek);

  // Cell-centered extended
  void ProlongateCellCenteredXBCValues(const AthenaArray<Real> &coarse,
                                       AthenaArray<Real> &fine, int sn, int en,
                                       int csi, int cei, int csj, int cej, int csk, int cek);

  void ProlongateCellCenteredXValues(const AthenaArray<Real> &coarse,
                                      AthenaArray<Real> &fine, int sn, int en,
                                      int csi, int cei, int csj, int cej, int csk, int cek);

  template<int H_SZ>
  void ProlongateCellCenteredX(const AthenaArray<Real> &coarse,
                              AthenaArray<Real> &fine, int sn, int en,
                              int csi, int cei, int csj, int cej, int csk, int cek);


  void RestrictCellCenteredXValues(const AthenaArray<Real> &fine,
                                   AthenaArray<Real> &coarse, int sn, int en,
                                   int csi, int cei, int csj, int cej, int csk, int cek);

  void RestrictCellCenteredXValuesLO(const AthenaArray<Real> &fine,
                                     AthenaArray<Real> &coarse, int sn, int en,
                                     int csi, int cei, int csj, int cej, int csk, int cek);

  void RestrictCellCenteredXWithInteriorValues(
    const AthenaArray<Real> &fine,
    AthenaArray<Real> &coarse, int sn, int en
  );

  void CheckRefinementCondition();

  // setter functions for "enrolling" variable arrays in refinement via Mesh::AMR()
  // and/or in BoundaryValues::ProlongateBoundaries() (for SMR and AMR)
  int AddToRefinementCC(AthenaArray<Real> *pvar_in, AthenaArray<Real> *pcoarse_in);
  int AddToRefinementVC(AthenaArray<Real> *pvar_in, AthenaArray<Real> *pcoarse_in);
  int AddToRefinementCX(AthenaArray<Real> *pvar_in, AthenaArray<Real> *pcoarse_in);
  int AddToRefinementFC(FaceField *pvar_fc, FaceField *pcoarse_fc);

  // as above but for use with auxiliary task list
  int AddToRefinementAuxCC(AthenaArray<Real> *pvar_in, AthenaArray<Real> *pcoarse_in);
  int AddToRefinementAuxVC(AthenaArray<Real> *pvar_in, AthenaArray<Real> *pcoarse_in);
  int AddToRefinementAuxCX(AthenaArray<Real> *pvar_in, AthenaArray<Real> *pcoarse_in);
  int AddToRefinementAuxFC(FaceField *pvar_fc, FaceField *pcoarse_fc);

  // switch internal pvars_X_ <-> pvars_aux_X_
  void SwapRefinementAux();

  // for switching first entry in pvars_cc_ to/from: (w, coarse_prim); (u, coarse_cons_)
  void SetHydroRefinement(HydroBoundaryQuantity hydro_type);

 private:
  // data
  MeshBlock *pmy_block_;
  Coordinates *pcoarsec;

  AthenaArray<Real> fvol_[2][2], sarea_x1_[2][2], sarea_x2_[2][3], sarea_x3_[3][2];
  int refine_flag_, neighbor_rflag_, deref_count_, deref_threshold_;

  // functions
  AMRFlagFunc AMRFlag_; // duplicate of Mesh class member

  void RestrictVertexCenteredIndicialHelper(
    int ix,
    int ix_cvs, int ix_cve,
    int ix_vs, int ix_ve,
    int &f_ix);

  void ProlongateVertexCenteredIndicialHelper(
    int hs_sz, int ix,
    int ix_cvs, int ix_cve, int ix_cmp,
    int ix_vs, int ix_ve,
    int &f_ix, int &ix_b, int &ix_so, int &ix_eo, int &ix_l, int &ix_u);

  // tuples of references to AMR-enrolled arrays (quantity, coarse_quantity)
  std::vector<std::tuple<AthenaArray<Real> *, AthenaArray<Real> *>> pvars_cc_;
  std::vector<std::tuple<FaceField *, FaceField *>> pvars_fc_;
  std::vector<std::tuple<AthenaArray<Real> *, AthenaArray<Real> *>> pvars_vc_;
  std::vector<std::tuple<AthenaArray<Real> *, AthenaArray<Real> *>> pvars_cx_;

  // for aux. lists
  std::vector<std::tuple<AthenaArray<Real> *,
                         AthenaArray<Real> *>> pvars_aux_cc_;
  std::vector<std::tuple<FaceField *, FaceField *>> pvars_aux_fc_;
  std::vector<std::tuple<AthenaArray<Real> *,
                         AthenaArray<Real> *>> pvars_aux_vc_;
  std::vector<std::tuple<AthenaArray<Real> *,
                         AthenaArray<Real> *>> pvars_aux_cx_;

  // --------------------------------------------------------------------------
  // BD: cx coarse ghosts can exceed NCGHOST so grid regen needed
  static inline int sz_linspace(
    int num,
    bool staggered,
    int num_ghost)
  {
    return num + 2 * num_ghost;
    // return num + 2 * num_ghost - staggered;
  }

  template <typename Tsta, typename Tsto>
  static inline auto linspace(
    Tsta start, Tsto stop, int num,
    bool staggered,
    int num_ghost) -> decltype(start + stop) *
  {
    // Reduce to common value
    typedef typename std::common_type<Tsta, Tsto>::type R;

    const int inum = num + staggered;

    // spacing [note promotion]
    auto ds = (stop - start) / (inum - 1 + 0.0);
    R s_0 = (staggered) ? ds / 2 : 0;

    R * ret = new R[sz_linspace(inum, staggered, num_ghost)];

    if (std::is_integral<R>::value)
    {
      for (int six = -num_ghost; six < (inum + num_ghost - staggered); ++six)
      {
        ret[six + num_ghost] = std::round(start + s_0 + six * ds);
      }

    }
    else
    {
      for (int six = -num_ghost; six < (inum + num_ghost - staggered); ++six)
      {
        ret[six + num_ghost] = start + s_0 + six * ds;
      }
    }

    return ret;
  }

  // --------------------------------------------------------------------------
  // Class for barycentric interpolation; weights precomputed in MeshRefinement
  // ctor
  typedef Floater_Hormann::interp_nd<Real, Real>
    interp_nd;

  interp_nd * ind_interior_r_op;
#if defined(DBG_CX_ALL_BARY_RP)
  interp_nd * ind_physical_r_op;
  interp_nd * ind_physical_p_op;

  Real *x1c_cx, *x2c_cx, *x3c_cx;
  Real *x1f_cx, *x2f_cx, *x3f_cx;
#endif // DBG_CX_ALL_BARY_RP
  // --------------------------------------------------------------------------

};

#endif // MESH_MESH_REFINEMENT_HPP_
