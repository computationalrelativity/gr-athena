#ifndef COORDINATES_COORDINATES_HPP_
#define COORDINATES_COORDINATES_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file coordinates.hpp
//  \brief defines abstract base and derived classes for coordinates.  These classes
//  provide data and functions to compute/store coordinate positions and spacing, as well
//  as geometrical factors (areas, volumes, coordinate source terms) for various
//  coordinate systems.

// C headers

// C++ headers
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../finite_differencing.hpp"
#include "../z4c/z4c.hpp"
#include "../lagrange_interp.hpp"

// forward declarations
class MeshBlock;
class ParameterInput;

//----------------------------------------------------------------------------------------
//! \class Coordinates
//  \brief abstract base class for all coordinate derived classes

class Coordinates {
 public:
  friend class HydroSourceTerms;
  Coordinates(MeshBlock *pmb, ParameterInput *pin, bool flag = false);
  virtual ~Coordinates() = default;

  // data
  MeshBlock *pmy_block;  // ptr to MeshBlock containing this Coordinates
  AthenaArray<Real> dx1f, dx2f, dx3f, x1f, x2f, x3f;    // face   spacing and positions
  AthenaArray<Real> dx1v, dx2v, dx3v, x1v, x2v, x3v;    // volume spacing and positions
  AthenaArray<Real> x1s2, x1s3, x2s1, x2s3, x3s1, x3s2; // area averaged positions for AMR
  // geometry coefficients (only used in SphericalPolar, Cylindrical, Cartesian)
  AthenaArray<Real> h2f, dh2fd1, h31f, h32f, dh31fd1, dh32fd2;
  AthenaArray<Real> h2v, dh2vd1, h31v, h32v, dh31vd1, dh32vd2;

  // functions...
  // ...to compute length of edges
  virtual void UpdateMetric();
  virtual void Edge1Length(const int k, const int j, const int il, const int iu,
                           AthenaArray<Real> &len);
  virtual void Edge2Length(const int k, const int j, const int il, const int iu,
                           AthenaArray<Real> &len);
  virtual void Edge3Length(const int k, const int j, const int il, const int iu,
                           AthenaArray<Real> &len);
  virtual Real GetEdge1Length(const int k, const int j, const int i);
  virtual Real GetEdge2Length(const int k, const int j, const int i);
  virtual Real GetEdge3Length(const int k, const int j, const int i);
  // ...to compute length connecting cell centers (for non-ideal MHD)
  virtual void VolCenter1Length(const int k, const int j, const int il, const int iu,
                                AthenaArray<Real> &len);
  virtual void VolCenter2Length(const int k, const int j, const int il, const int iu,
                                AthenaArray<Real> &len);
  virtual void VolCenter3Length(const int k, const int j, const int il, const int iu,
                                AthenaArray<Real> &len);
  // ...to compute physical width at cell center
  virtual void CenterWidth1(const int k, const int j, const int il, const int iu,
                            AthenaArray<Real> &dx1);
  virtual void CenterWidth2(const int k, const int j, const int il, const int iu,
                            AthenaArray<Real> &dx2);
  virtual void CenterWidth3(const int k, const int j, const int il, const int iu,
                            AthenaArray<Real> &dx3);

  // ...to compute area of faces
  virtual void Face1Area(const int k, const int j, const int il, const int iu,
                         AthenaArray<Real> &area);
  virtual void Face2Area(const int k, const int j, const int il, const int iu,
                         AthenaArray<Real> &area);
  virtual void Face3Area(const int k, const int j, const int il, const int iu,
                         AthenaArray<Real> &area);
  virtual Real GetFace1Area(const int k, const int j, const int i);
  virtual Real GetFace2Area(const int k, const int j, const int i);
  virtual Real GetFace3Area(const int k, const int j, const int i);
  // ...to compute area of faces joined by cell centers (for non-ideal MHD)
  virtual void VolCenterFace1Area(const int k, const int j, const int il, const int iu,
                                  AthenaArray<Real> &area);
  virtual void VolCenterFace2Area(const int k, const int j, const int il, const int iu,
                                  AthenaArray<Real> &area);
  virtual void VolCenterFace3Area(const int k, const int j, const int il, const int iu,
                                  AthenaArray<Real> &area);

  // ...to compute Laplacian of quantities in the coord system and orthogonal subspaces
  virtual void Laplacian(
      const AthenaArray<Real> &s, AthenaArray<Real> &delta_s,
      const int il, const int iu, const int jl, const int ju, const int kl, const int ku,
      const int nl, const int nu);
  virtual void LaplacianX1(
      const AthenaArray<Real> &s, AthenaArray<Real> &delta_s,
      const int n, const int k, const int j, const int il, const int iu);
  virtual void LaplacianX1All(
      const AthenaArray<Real> &s, AthenaArray<Real> &delta_s,
      const int nl, const int nu, const int kl, const int ku,
      const int jl, const int ju, const int il, const int iu);
  virtual void LaplacianX2(
      const AthenaArray<Real> &s, AthenaArray<Real> &delta_s,
      const int n, const int k, const int j, const int il, const int iu);
  virtual void LaplacianX2All(
      const AthenaArray<Real> &s, AthenaArray<Real> &delta_s,
      const int nl, const int nu, const int kl, const int ku,
      const int jl, const int ju, const int il, const int iu);
  virtual void LaplacianX3(
      const AthenaArray<Real> &s, AthenaArray<Real> &delta_s,
      const int n, const int k, const int j, const int il, const int iu);
  virtual void LaplacianX3All(
      const AthenaArray<Real> &s, AthenaArray<Real> &delta_s,
      const int nl, const int nu, const int kl, const int ku,
      const int jl, const int ju, const int il, const int iu);

  // ...to compute volume of cells
  virtual void CellVolume(const int k, const int j, const int il, const int iu,
                          AthenaArray<Real> &vol);
  virtual Real GetCellVolume(const int k, const int j, const int i);

  // ...to compute geometrical source terms
  virtual void AddCoordTermsDivergence(const Real dt, const AthenaArray<Real> *flux,
                             const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                             AthenaArray<Real> &u);

  // ...to determine if index is a pole
  bool IsPole(int j);


  // In GR, functions...
  // ...to return private variables
  Real GetMass() const {return bh_mass_;}
  Real GetSpin() const {return bh_spin_;}

  // ...to compute metric
  void Metric(
      Real x1, Real x2, Real x3, ParameterInput *pin, AthenaArray<Real> &g,
      AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1, AthenaArray<Real> &dg_dx2,
      AthenaArray<Real> &dg_dx3);
  virtual void CellMetric(const int k, const int j, const int il, const int iu,
                          AthenaArray<Real> &g, AthenaArray<Real> &gi) {}
  virtual void Face1Metric(const int k, const int j, const int il, const int iu,
                           AthenaArray<Real> &g, AthenaArray<Real> &g_inv) {}
  virtual void Face2Metric(const int k, const int j, const int il, const int iu,
                           AthenaArray<Real> &g, AthenaArray<Real> &g_inv) {}
  virtual void Face3Metric(const int k, const int j, const int il, const int iu,
                           AthenaArray<Real> &g, AthenaArray<Real> &g_inv) {}

  // ...to transform primitives to locally flat space
  virtual void PrimToLocal1(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &b1_vals, AthenaArray<Real> &prim_left,
      AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) {}
  virtual void PrimToLocal2(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &b2_vals, AthenaArray<Real> &prim_left,
      AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) {}
  virtual void PrimToLocal3(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &b3_vals, AthenaArray<Real> &prim_left,
      AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) {}

  // ...to transform fluxes in locally flat space to global frame
  virtual void FluxToGlobal1(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) {}
  virtual void FluxToGlobal2(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) {}
  virtual void FluxToGlobal3(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) {}

  // ...to raise (lower) covariant (contravariant) components of a vector
  virtual void RaiseVectorCell(Real a_0, Real a_1, Real a_2, Real a_3, int k, int j,
                               int i, Real *pa0, Real *pa1, Real *pa2, Real *pa3) {}
  virtual void LowerVectorCell(Real a0, Real a1, Real a2, Real a3, int k, int j, int i,
                               Real *pa_0, Real *pa_1, Real *pa_2, Real *pa_3) {}

 protected:
  bool coarse_flag;  // true if this coordinate object is parent (coarse) mesh in AMR
  Mesh *pm;
  int il, iu, jl, ju, kl, ku, ng;  // limits of indices of arrays (normal or coarse)
  int nc1, nc2, nc3;               // # cells in each dir of arrays (normal or coarse)
  // Scratch arrays for coordinate factors
  // Format: coord_<type>[<direction>]_<index>[<count>]_
  //   type: vol[ume], area, etc.
  //   direction: 1/2/3 depending on which face, edge, etc. is in play
  //   index: i/j/k indicating which coordinates index array
  //   count: 1/2/... in case multiple arrays are needed for different terms
  AthenaArray<Real> coord_vol_i_, coord_vol_i1_, coord_vol_i2_;
  AthenaArray<Real> coord_vol_j_, coord_vol_j1_, coord_vol_j2_;
  AthenaArray<Real> coord_vol_k1_;
  AthenaArray<Real> coord_vol_kji_;
  AthenaArray<Real> coord_area1_i_, coord_area1_i1_;
  AthenaArray<Real> coord_area1_j_, coord_area1_j1_, coord_area1_j2_;
  AthenaArray<Real> coord_area1_k1_;
  AthenaArray<Real> coord_area1_kji_;
  AthenaArray<Real> coord_area2_i_, coord_area2_i1_, coord_area2_i2_;
  AthenaArray<Real> coord_area2_j_, coord_area2_j1_, coord_area2_j2_;
  AthenaArray<Real> coord_area2_k1_;
  AthenaArray<Real> coord_area2_kji_;
  AthenaArray<Real> coord_area3_i_, coord_area3_i1_, coord_area3_i2_;
  AthenaArray<Real> coord_area3_j1_, coord_area3_j2_;
  AthenaArray<Real> coord_area3_kji_;
  AthenaArray<Real> coord_area1vc_i_,coord_area1vc_j_; //nonidealmhd additions
  AthenaArray<Real> coord_area2vc_i_,coord_area2vc_j_; //nonidealmhd additions
  AthenaArray<Real> coord_area3vc_i_; //nonidealmhd addition
  AthenaArray<Real> coord_len1_i1_, coord_len1_i2_;
  AthenaArray<Real> coord_len1_j1_, coord_len1_j2_;
  AthenaArray<Real> coord_len1_kji_;
  AthenaArray<Real> coord_len2_i1_;
  AthenaArray<Real> coord_len2_j1_, coord_len2_j2_;
  AthenaArray<Real> coord_len2_kji_;
  AthenaArray<Real> coord_len3_i1_;
  AthenaArray<Real> coord_len3_j1_, coord_len3_j2_;
  AthenaArray<Real> coord_len3_k1_;
  AthenaArray<Real> coord_len3_kji_;
  AthenaArray<Real> coord_width1_i1_;
  AthenaArray<Real> coord_width1_kji_;
  AthenaArray<Real> coord_width2_i1_;
  AthenaArray<Real> coord_width2_j1_;
  AthenaArray<Real> coord_width2_kji_;
  AthenaArray<Real> coord_width3_j1_, coord_width3_j2_, coord_width3_j3_;
  AthenaArray<Real> coord_width3_k1_;
  AthenaArray<Real> coord_width3_ji1_;
  AthenaArray<Real> coord_width3_kji_;
  AthenaArray<Real> coord_src_j1_, coord_src_j2_;
  AthenaArray<Real> coord_src_kji_;
  AthenaArray<Real> coord_src1_i_;
  AthenaArray<Real> coord_src1_j_;
  AthenaArray<Real> coord_src2_i_;
  AthenaArray<Real> coord_src2_j_;
  AthenaArray<Real> coord_src3_j_;

  // Scratch arrays for physical source terms
  AthenaArray<Real> phy_src1_i_, phy_src2_i_;

  // GR-specific scratch arrays
  AthenaArray<Real> metric_cell_i1_, metric_cell_i2_;
  AthenaArray<Real> metric_cell_j1_, metric_cell_j2_;
  AthenaArray<Real> metric_cell_kji_;
  AthenaArray<Real> metric_face1_i1_, metric_face1_i2_;
  AthenaArray<Real> metric_face1_j1_, metric_face1_j2_;
  AthenaArray<Real> metric_face1_kji_;
  AthenaArray<Real> metric_face2_i1_, metric_face2_i2_;
  AthenaArray<Real> metric_face2_j1_, metric_face2_j2_;
  AthenaArray<Real> metric_face2_kji_;
  AthenaArray<Real> metric_face3_i1_, metric_face3_i2_;
  AthenaArray<Real> metric_face3_j1_, metric_face3_j2_;
  AthenaArray<Real> metric_face3_kji_;
  AthenaArray<Real> trans_face1_i1_, trans_face1_i2_;
  AthenaArray<Real> trans_face1_j1_;
  AthenaArray<Real> trans_face1_ji1_, trans_face1_ji2_, trans_face1_ji3_,
    trans_face1_ji4_, trans_face1_ji5_, trans_face1_ji6_, trans_face1_ji7_;
  AthenaArray<Real> trans_face1_kji_;
  AthenaArray<Real> trans_face2_i1_, trans_face2_i2_;
  AthenaArray<Real> trans_face2_j1_;
  AthenaArray<Real> trans_face2_ji1_, trans_face2_ji2_, trans_face2_ji3_,
    trans_face2_ji4_, trans_face2_ji5_, trans_face2_ji6_;
  AthenaArray<Real> trans_face2_kji_;
  AthenaArray<Real> trans_face3_i1_, trans_face3_i2_;
  AthenaArray<Real> trans_face3_j1_;
  AthenaArray<Real> trans_face3_ji1_, trans_face3_ji2_, trans_face3_ji3_,
    trans_face3_ji4_, trans_face3_ji5_, trans_face3_ji6_;
  AthenaArray<Real> trans_face3_kji_;
  AthenaArray<Real> g_, gi_;

// Dynamical GR arrays
  AthenaArray<Real>  excurv_kji_;
  AthenaArray<Real>  coord_3vol_kji_;



  // GR-specific variables
  Real bh_mass_;
  Real bh_spin_;
  Real chi_psi_power;
};

//----------------------------------------------------------------------------------------
//! \class Cartesian
//  \brief derived class for Cartesian coordinates.  None of the virtual funcs
//  in the Coordinates abstract base class need to be overridden.

class Cartesian : public Coordinates {
  friend class HydroSourceTerms;

 public:
  Cartesian(MeshBlock *pmb, ParameterInput *pin, bool flag);
  //void UpdateMetric() final;
};

//----------------------------------------------------------------------------------------
//! \class Cylindrical
//  \brief derived class for Cylindrical coordinates.  Some of the length, area,
//  and volume functions in the Coordinates abstract base class are overridden

class Cylindrical : public Coordinates {
  friend class HydroSourceTerms;

 public:
  Cylindrical(MeshBlock *pmb, ParameterInput *pin, bool flag);

  // functions...
  // ...to compute length of edges
  //void UpdateMetric() final;
  void Edge2Length(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &len) final;
  Real GetEdge2Length(const int k, const int j, const int i) final;
  // ...to compute length connecting cell centers (for non-ideal MHD)
  void VolCenter2Length(const int k, const int j, const int il, const int iu,
                        AthenaArray<Real> &len) final;
  // ...to compute physical width at cell center
  void CenterWidth2(const int k, const int j, const int il, const int iu,
                    AthenaArray<Real> &dx2) final;

  // ...to compute area of faces
  void Face1Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  void Face3Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  Real GetFace1Area(const int k, const int j, const int i) final;
  Real GetFace3Area(const int k, const int j, const int i) final;
  // ...to compute area of faces joined by cell centers (for non-ideal MHD)
  void VolCenterFace1Area(const int k, const int j, const int il, const int iu,
                          AthenaArray<Real> &area) final;
  void VolCenterFace3Area(const int k, const int j, const int il, const int iu,
                          AthenaArray<Real> &area) final;
  // ...to compute volumes of cells
  void CellVolume(const int k, const int j, const int il, const int iu,
                  AthenaArray<Real> &vol) final;
  Real GetCellVolume(const int k, const int j, const int i) final;

  // ...to compute geometrical source terms
  void AddCoordTermsDivergence(const Real dt, const AthenaArray<Real> *flux,
                     const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                     AthenaArray<Real> &u) final;
};

//----------------------------------------------------------------------------------------
//! \class SphericalPolar
//  \brief derived class for spherical polar coordinates.  Many of the length, area,
//  and volume functions in the Coordinates abstract base class are overridden.

class SphericalPolar : public Coordinates {
  friend class HydroSourceTerms;

 public:
  SphericalPolar(MeshBlock *pmb, ParameterInput *pin, bool flag);
  //void UpdateMetric() final;

  // functions...
  // ...to compute length of edges
  void Edge2Length(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &len) final;
  void Edge3Length(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &len) final;
  Real GetEdge2Length(const int k, const int j, const int i) final;
  Real GetEdge3Length(const int k, const int j, const int i) final;
  // ...to compute length connecting cell centers (for non-ideal MHD)
  void VolCenter2Length(const int k, const int j, const int il, const int iu,
                        AthenaArray<Real> &len) final;
  void VolCenter3Length(const int k, const int j, const int il, const int iu,
                        AthenaArray<Real> &len) final;
  // ...to compute physical width at cell center
  void CenterWidth2(const int k, const int j, const int il, const int iu,
                    AthenaArray<Real> &dx2) final;
  void CenterWidth3(const int k, const int j, const int il, const int iu,
                    AthenaArray<Real> &dx3) final;

  // ...to compute area of faces
  void Face1Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  void Face2Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  void Face3Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  Real GetFace1Area(const int k, const int j, const int i) final;
  Real GetFace2Area(const int k, const int j, const int i) final;
  Real GetFace3Area(const int k, const int j, const int i) final;
  // ...to compute area of faces joined by cell centers (for non-ideal MHD)
  void VolCenterFace1Area(const int k, const int j, const int il, const int iu,
                          AthenaArray<Real> &area) final;
  void VolCenterFace2Area(const int k, const int j, const int il, const int iu,
                          AthenaArray<Real> &area) final;
  void VolCenterFace3Area(const int k, const int j, const int il, const int iu,
                          AthenaArray<Real> &area) final;
  // ...to compute volumes of cells
  void CellVolume(const int k, const int j, const int il, const int iu,
                  AthenaArray<Real> &vol) final;
  Real GetCellVolume(const int k, const int j, const int i) final;

  // ...to compute geometrical source terms
  void AddCoordTermsDivergence(const Real dt, const AthenaArray<Real> *flux,
                     const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                     AthenaArray<Real> &u) final;
};

//----------------------------------------------------------------------------------------
//! \class Minkowski
//  \brief derived class for Minkowski (flat) spacetime and Cartesian coordinates in GR.
//  None of the length, area, and volume functions in the abstract base class need to be
//  overridden, but all the metric and transforms functions are.

class Minkowski : public Coordinates {
  friend class HydroSourceTerms;

 public:
  Minkowski(MeshBlock *pmb, ParameterInput *pin, bool flag);

  // In GR, functions...
  // ...to compute metric
  //void UpdateMetric() final;
  void CellMetric(const int k, const int j, const int il, const int iu,
                  AthenaArray<Real> &g, AthenaArray<Real> &gi) final;
  void Face1Metric(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &g, AthenaArray<Real> &g_inv) final;
  void Face2Metric(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &g, AthenaArray<Real> &g_inv) final;
  void Face3Metric(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &g, AthenaArray<Real> &g_inv) final;

  // ...to transform primitives to locally flat space
  void PrimToLocal1(const int k, const int j, const int il, const int iu,
                    const AthenaArray<Real> &b1_vals, AthenaArray<Real> &prim_left,
                    AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) final;
  void PrimToLocal2(const int k, const int j, const int il, const int iu,
                    const AthenaArray<Real> &b2_vals, AthenaArray<Real> &prim_left,
                    AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) final;
  void PrimToLocal3(const int k, const int j, const int il, const int iu,
                    const AthenaArray<Real> &b3_vals, AthenaArray<Real> &prim_left,
                    AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) final;

  // ...to transform fluxes in locally flat space to global frame
  void FluxToGlobal1(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) final;
  void FluxToGlobal2(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) final;
  void FluxToGlobal3(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) final;

  // for raising (lowering) covariant (contravariant) components of a vector
  void RaiseVectorCell(Real a_0, Real a_1, Real a_2, Real a_3, int k, int j, int i,
                       Real *pa0, Real *pa1, Real *pa2, Real *pa3) final;
  void LowerVectorCell(Real a0, Real a1, Real a2, Real a3, int k, int j, int i,
                       Real *pa_0, Real *pa_1, Real *pa_2, Real *pa_3) final;
};

//----------------------------------------------------------------------------------------
//! \class Schwarzschild
//  \brief derived class for Schwarzschild spacetime and spherical polar coordinates in GR
//  Nearly every function in the abstract base class need to be overridden.

class Schwarzschild : public Coordinates {
  friend class HydroSourceTerms;

 public:
  Schwarzschild(MeshBlock *pmb, ParameterInput *pin, bool flag);

  // functions...
  // ...to compute length of edges
  //void UpdateMetric() final;
  void Edge1Length(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &len) final;
  void Edge2Length(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &len) final;
  void Edge3Length(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &len) final;
  Real GetEdge1Length(const int k, const int j, const int i) final;
  Real GetEdge2Length(const int k, const int j, const int i) final;
  Real GetEdge3Length(const int k, const int j, const int i) final;

  // ...to compute physical width at cell center
  void CenterWidth1(const int k, const int j, const int il, const int iu,
                    AthenaArray<Real> &dx1) final;
  void CenterWidth2(const int k, const int j, const int il, const int iu,
                    AthenaArray<Real> &dx2) final;
  void CenterWidth3(const int k, const int j, const int il, const int iu,
                    AthenaArray<Real> &dx3) final;

  // ...to compute area of faces
  void Face1Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  void Face2Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  void Face3Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  Real GetFace1Area(const int k, const int j, const int i) final;
  Real GetFace2Area(const int k, const int j, const int i) final;
  Real GetFace3Area(const int k, const int j, const int i) final;

  // ...to compute volumes of cells
  void CellVolume(const int k, const int j, const int il, const int iu,
                  AthenaArray<Real> &vol) final;
  Real GetCellVolume(const int k, const int j, const int i) final;

  // ...to compute geometrical source terms
  void AddCoordTermsDivergence(const Real dt, const AthenaArray<Real> *flux,
                     const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                     AthenaArray<Real> &u) final;

  // In GR, functions...
  // ...to compute metric
  void CellMetric(const int k, const int j, const int il, const int iu,
                  AthenaArray<Real> &g, AthenaArray<Real> &gi) final;
  void Face1Metric(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &g, AthenaArray<Real> &g_inv) final;
  void Face2Metric(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &g, AthenaArray<Real> &g_inv) final;
  void Face3Metric(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &g, AthenaArray<Real> &g_inv) final;

  // ...to transform primitives to locally flat space
  void PrimToLocal1(const int k, const int j, const int il, const int iu,
                    const AthenaArray<Real> &b1_vals, AthenaArray<Real> &prim_left,
                    AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) final;
  void PrimToLocal2(const int k, const int j, const int il, const int iu,
                    const AthenaArray<Real> &b2_vals, AthenaArray<Real> &prim_left,
                    AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) final;
  void PrimToLocal3(const int k, const int j, const int il, const int iu,
                    const AthenaArray<Real> &b3_vals, AthenaArray<Real> &prim_left,
                    AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) final;

  // ...to transform fluxes in locally flat space to global frame
  void FluxToGlobal1(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) final;
  void FluxToGlobal2(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) final;
  void FluxToGlobal3(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) final;

  // for raising (lowering) covariant (contravariant) components of a vector
  void RaiseVectorCell(Real a_0, Real a_1, Real a_2, Real a_3, int k, int j, int i,
                       Real *pa0, Real *pa1, Real *pa2, Real *pa3) final;
  void LowerVectorCell(Real a0, Real a1, Real a2, Real a3, int k, int j, int i,
                       Real *pa_0, Real *pa_1, Real *pa_2, Real *pa_3) final;
};

//----------------------------------------------------------------------------------------
//! \class KerrSchild
//  \brief derived class for Kerr spacetime and Kerr-Schild coordinates in GR.
//  Nearly every function in the abstract base class need to be overridden.

class KerrSchild : public Coordinates {
  friend class HydroSourceTerms;

 public:
  KerrSchild(MeshBlock *pmb, ParameterInput *pin, bool flag);

  // functions...
  // ...to compute length of edges
  //void UpdateMetric() final;
  void Edge1Length(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &len) final;
  void Edge2Length(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &len) final;
  void Edge3Length(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &len) final;
  Real GetEdge1Length(const int k, const int j, const int i) final;
  Real GetEdge2Length(const int k, const int j, const int i) final;
  Real GetEdge3Length(const int k, const int j, const int i) final;

  // ...to compute physical width at cell center
  void CenterWidth1(const int k, const int j, const int il, const int iu,
                    AthenaArray<Real> &dx1) final;
  void CenterWidth2(const int k, const int j, const int il, const int iu,
                    AthenaArray<Real> &dx2) final;
  void CenterWidth3(const int k, const int j, const int il, const int iu,
                    AthenaArray<Real> &dx3) final;

  // ...to compute area of faces
  void Face1Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  void Face2Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  void Face3Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  Real GetFace1Area(const int k, const int j, const int i) final;
  Real GetFace2Area(const int k, const int j, const int i) final;
  Real GetFace3Area(const int k, const int j, const int i) final;

  // ...to compute volumes of cells
  void CellVolume(const int k, const int j, const int il, const int iu,
                  AthenaArray<Real> &vol) final;
  Real GetCellVolume(const int k, const int j, const int i) final;

  // ...to compute geometrical source terms
  void AddCoordTermsDivergence(const Real dt, const AthenaArray<Real> *flux,
                     const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                     AthenaArray<Real> &u) final;

  // In GR, functions...
  // ...to compute metric
  void CellMetric(const int k, const int j, const int il, const int iu,
                  AthenaArray<Real> &g, AthenaArray<Real> &gi) final;
  void Face1Metric(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &g, AthenaArray<Real> &g_inv) final;
  void Face2Metric(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &g, AthenaArray<Real> &g_inv) final;
  void Face3Metric(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &g, AthenaArray<Real> &g_inv) final;

  // ...to transform primitives to locally flat space
  void PrimToLocal1(const int k, const int j, const int il, const int iu,
                    const AthenaArray<Real> &b1_vals, AthenaArray<Real> &prim_left,
                    AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) final;
  void PrimToLocal2(const int k, const int j, const int il, const int iu,
                    const AthenaArray<Real> &b2_vals, AthenaArray<Real> &prim_left,
                    AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) final;
  void PrimToLocal3(const int k, const int j, const int il, const int iu,
                    const AthenaArray<Real> &b3_vals, AthenaArray<Real> &prim_left,
                    AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) final;

  // ...to transform fluxes in locally flat space to global frame
  void FluxToGlobal1(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) final;
  void FluxToGlobal2(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) final;
  void FluxToGlobal3(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) final;

  // for raising (lowering) covariant (contravariant) components of a vector
  void RaiseVectorCell(Real a_0, Real a_1, Real a_2, Real a_3, int k, int j, int i,
                       Real *pa0, Real *pa1, Real *pa2, Real *pa3) final;
  void LowerVectorCell(Real a0, Real a1, Real a2, Real a3, int k, int j, int i,
                       Real *pa_0, Real *pa_1, Real *pa_2, Real *pa_3) final;
};

//----------------------------------------------------------------------------------------
//! \class GRUser
//  \brief derived class for arbitrary (stationary) user-defined coordinates in GR.
//  Nearly every function in the abstract base class need to be overridden.

class GRUser : public Coordinates {
  friend class HydroSourceTerms;

 public:
  GRUser(MeshBlock *pmb, ParameterInput *pin, bool flag);

  // functions...
  // ...to compute length of edges
  //void UpdateMetric() final;
  void Edge1Length(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &len) final;
  void Edge2Length(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &len) final;
  void Edge3Length(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &len) final;
  Real GetEdge1Length(const int k, const int j, const int i) final;
  Real GetEdge2Length(const int k, const int j, const int i) final;
  Real GetEdge3Length(const int k, const int j, const int i) final;

  // ...to compute physical width at cell center
  void CenterWidth1(const int k, const int j, const int il, const int iu,
                    AthenaArray<Real> &dx1) final;
  void CenterWidth2(const int k, const int j, const int il, const int iu,
                    AthenaArray<Real> &dx2) final;
  void CenterWidth3(const int k, const int j, const int il, const int iu,
                    AthenaArray<Real> &dx3) final;

  // ...to compute area of faces
  void Face1Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  void Face2Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  void Face3Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  Real GetFace1Area(const int k, const int j, const int i) final;
  Real GetFace2Area(const int k, const int j, const int i) final;
  Real GetFace3Area(const int k, const int j, const int i) final;

  // ...to compute volumes of cells
  void CellVolume(const int k, const int j, const int il, const int iu,
                  AthenaArray<Real> &vol) final;
  Real GetCellVolume(const int k, const int j, const int i) final;

  // ...to compute geometrical source terms
  void AddCoordTermsDivergence(const Real dt, const AthenaArray<Real> *flux,
                     const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                     AthenaArray<Real> &u) final;

  // In GR, functions...
  // ...to compute metric
  void CellMetric(const int k, const int j, const int il, const int iu,
                  AthenaArray<Real> &g, AthenaArray<Real> &gi) final;
  void Face1Metric(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &g, AthenaArray<Real> &g_inv) final;
  void Face2Metric(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &g, AthenaArray<Real> &g_inv) final;
  void Face3Metric(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &g, AthenaArray<Real> &g_inv) final;

  // ...to transform primitives to locally flat space
  void PrimToLocal1(const int k, const int j, const int il, const int iu,
                    const AthenaArray<Real> &b1_vals, AthenaArray<Real> &prim_left,
                    AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) final;
  void PrimToLocal2(const int k, const int j, const int il, const int iu,
                    const AthenaArray<Real> &b2_vals, AthenaArray<Real> &prim_left,
                    AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) final;
  void PrimToLocal3(const int k, const int j, const int il, const int iu,
                    const AthenaArray<Real> &b3_vals, AthenaArray<Real> &prim_left,
                    AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) final;

  // ...to transform fluxes in locally flat space to global frame
  void FluxToGlobal1(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) final;
  void FluxToGlobal2(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) final;
  void FluxToGlobal3(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) final;

  // ...for raising (lowering) covariant (contravariant) components of a vector
  void RaiseVectorCell(Real a_0, Real a_1, Real a_2, Real a_3, int k, int j, int i,
                       Real *pa0, Real *pa1, Real *pa2, Real *pa3) final;
  void LowerVectorCell(Real a0, Real a1, Real a2, Real a3, int k, int j, int i,
                       Real *pa_0, Real *pa_1, Real *pa_2, Real *pa_3) final;
};

//----------------------------------------------------------------------------------------
//! \class GRDynamical
//  \brief derived class for arbitrary dynamically evolving coordinates in GR.
//  Nearly every function in the abstract base class need to be overridden.

class GRDynamical : public Coordinates {
  friend class HydroSourceTerms;

 public:
  GRDynamical(MeshBlock *pmb, ParameterInput *pin, bool flag);

  // functions...
  // for obtaining CC from VC
  void GetMetric(const int k, const int j, const int i, AthenaArray<Real>& g, AthenaArray<Real>& g_inv);
  void GetFace1Metric(const int k, const int j, const int i, AthenaArray<Real>& g, AthenaArray<Real>& g_inv);
  void GetFace2Metric(const int k, const int j, const int i, AthenaArray<Real>& g, AthenaArray<Real>& g_inv);
  void GetFace3Metric(const int k, const int j, const int i, AthenaArray<Real>& g, AthenaArray<Real>& g_inv);
  void GetExCurv(const int k, const int j, const int i,  AthenaArray<Real>& K);
  void GetDerivs(const int i, const int j, const int k, AthenaArray<Real>& dg_dx1, AthenaArray<Real>& dg_dx2, AthenaArray<Real>& dg_dx3);
  void UpdateMetric() final;
  // ...to compute length of edges

  void Edge1Length(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &len) final;
  void Edge2Length(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &len) final;
  void Edge3Length(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &len) final;
  Real GetEdge1Length(const int k, const int j, const int i) final;
  Real GetEdge2Length(const int k, const int j, const int i) final;
  Real GetEdge3Length(const int k, const int j, const int i) final;

  // ...to compute physical width at cell center
  void CenterWidth1(const int k, const int j, const int il, const int iu,
                    AthenaArray<Real> &dx1) final;
  void CenterWidth2(const int k, const int j, const int il, const int iu,
                    AthenaArray<Real> &dx2) final;
  void CenterWidth3(const int k, const int j, const int il, const int iu,
                    AthenaArray<Real> &dx3) final;

  // ...to compute area of faces
  void Face1Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  void Face2Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  void Face3Area(const int k, const int j, const int il, const int iu,
                 AthenaArray<Real> &area) final;
  Real GetFace1Area(const int k, const int j, const int i) final;
  Real GetFace2Area(const int k, const int j, const int i) final;
  Real GetFace3Area(const int k, const int j, const int i) final;

  // ...to compute volumes of cells
  void CellVolume(const int k, const int j, const int il, const int iu,
                  AthenaArray<Real> &vol) final;
  Real GetCellVolume(const int k, const int j, const int i) final;

  // ...to compute geometrical source terms
  void AddCoordTermsDivergence(const Real dt, const AthenaArray<Real> *flux,
                     const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                     AthenaArray<Real> &u) final;

  // In GR, functions...
  // ...to compute metric
  void CellMetric(const int k, const int j, const int il, const int iu,
                  AthenaArray<Real> &g, AthenaArray<Real> &gi) final;
  void Face1Metric(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &g, AthenaArray<Real> &g_inv) final;
  void Face2Metric(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &g, AthenaArray<Real> &g_inv) final;
  void Face3Metric(const int k, const int j, const int il, const int iu,
                   AthenaArray<Real> &g, AthenaArray<Real> &g_inv) final;

  // ...to transform primitives to locally flat space
  void PrimToLocal1(const int k, const int j, const int il, const int iu,
                    const AthenaArray<Real> &b1_vals, AthenaArray<Real> &prim_left,
                    AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) final;
  void PrimToLocal2(const int k, const int j, const int il, const int iu,
                    const AthenaArray<Real> &b2_vals, AthenaArray<Real> &prim_left,
                    AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) final;
  void PrimToLocal3(const int k, const int j, const int il, const int iu,
                    const AthenaArray<Real> &b3_vals, AthenaArray<Real> &prim_left,
                    AthenaArray<Real> &prim_right, AthenaArray<Real> &bx) final;

  // ...to transform fluxes in locally flat space to global frame
  void FluxToGlobal1(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) final;
  void FluxToGlobal2(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) final;
  void FluxToGlobal3(
      const int k, const int j, const int il, const int iu,
      const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx,
      AthenaArray<Real> &flux, AthenaArray<Real> &ey, AthenaArray<Real> &ez) final;

  // ...for raising (lowering) covariant (contravariant) components of a vector
  void RaiseVectorCell(Real a_0, Real a_1, Real a_2, Real a_3, int k, int j, int i,
                       Real *pa0, Real *pa1, Real *pa2, Real *pa3) final;
  void LowerVectorCell(Real a0, Real a1, Real a2, Real a3, int k, int j, int i,
                       Real *pa_0, Real *pa_1, Real *pa_2, Real *pa_3) final;
   private:
// not needed for 1st order interp
//    LagrangeInterpND<1, 3> * pinterp;

// derivative for calculating hydro source terms - copied from z4c
  struct {
    // 1st derivative stecil
    typedef FDCenteredStencil<1, NGHOST-1> s1;
    // 2nd derivative stencil
    typedef FDCenteredStencil<2, NGHOST-1> s2;
    // dissipation operator
    typedef FDCenteredStencil<
      FDDissChoice<NGHOST-1>::degree,
      FDDissChoice<NGHOST-1>::nghost
      > sd;
    // left-biased derivative
    typedef FDLeftBiasedStencil<
        FDBiasedChoice<1, NGHOST-1>::degree,
        FDBiasedChoice<1, NGHOST-1>::nghost,
        FDBiasedChoice<1, NGHOST-1>::lopsize
      > sl;
    // right-biased derivative
    typedef FDRightBiasedStencil<
        FDBiasedChoice<1, NGHOST-1>::degree,
        FDBiasedChoice<1, NGHOST-1>::nghost,
        FDBiasedChoice<1, NGHOST-1>::lopsize
      > sr;

    int stride[3];
    Real idx[3];
    Real diss;

    // 1st derivative (high order centered)
    inline Real Dx(int dir, Real & u) {
      Real * pu = &u - s1::offset*stride[dir];

      Real out(0.);
      for(int n1 = 0; n1 < s1::nghost; ++n1) {
        int const n2  = s1::width - n1 - 1;
        Real const c1 = s1::coeff[n1] * pu[n1*stride[dir]];
        Real const c2 = s1::coeff[n2] * pu[n2*stride[dir]];
        out += (c1 + c2);
      }
      out += s1::coeff[s1::nghost] * pu[s1::nghost*stride[dir]];
      return out * idx[dir];
    }
    // 1st derivative 2nd order centered
    inline Real Ds(int dir, Real & u) {
      Real * pu = &u;
      return 0.5 * idx[dir] * (pu[stride[dir]] - pu[-stride[dir]]);
    }
    // Advective derivative
    // The advective derivative is for an equation in the form
    //    d_t u = vx d_x u
    // So negative vx means advection from the *left* to the *right*, so we use
    // *left* biased FD stencils
    inline Real Lx(int dir, Real & vx, Real & u) {
      Real * pu = &u;

      Real dl(0.);
      for(int n = 0; n < sl::width; ++n) {
        dl += sl::coeff[n] * pu[(n - sl::offset)*stride[dir]];
      }

      Real dr(0.);
      for(int n = sr::width-1; n >= 0; --n) {
        dr += sr::coeff[n] * pu[(n - sr::offset)*stride[dir]];
      }
      return ((vx < 0) ? (vx * dl) : (vx * dr)) * idx[dir];
    }
    // Homogeneous 2nd derivative
    inline Real Dxx(int dir, Real & u) {
      Real * pu = &u - s2::offset*stride[dir];

      Real out(0.);
      for(int n1 = 0; n1 < s2::nghost; ++n1) {
        int const n2  = s2::width - n1 - 1;
        Real const c1 = s2::coeff[n1] * pu[n1*stride[dir]];
        Real const c2 = s2::coeff[n2] * pu[n2*stride[dir]];
        out += (c1 + c2);
      }
      out += s2::coeff[s2::nghost] * pu[s2::nghost*stride[dir]];
      return out * SQR(idx[dir]);
    }
    // Mixed 2nd derivative
    inline Real Dxy(int dirx, int diry, Real & u) {
      Real * pu = &u - s1::offset*(stride[dirx] + stride[diry]);
      Real out(0.);

      for(int nx1 = 0; nx1 < s1::nghost; ++nx1) {
        int const nx2 = s1::width - nx1 - 1;
        for(int ny1 = 0; ny1 < s1::nghost; ++ny1) {
          int const ny2 = s1::width - ny1 - 1;

          Real const c11 = s1::coeff[nx1] * s1::coeff[ny1] * pu[nx1*stride[dirx] + ny1*stride[diry]];
          Real const c12 = s1::coeff[nx1] * s1::coeff[ny2] * pu[nx1*stride[dirx] + ny2*stride[diry]];
          Real const c21 = s1::coeff[nx2] * s1::coeff[ny1] * pu[nx2*stride[dirx] + ny1*stride[diry]];
          Real const c22 = s1::coeff[nx2] * s1::coeff[ny2] * pu[nx2*stride[dirx] + ny2*stride[diry]];

          Real const ca = (1./6.)*((c11 + c12) + (c21 + c22));
          Real const cb = (1./6.)*((c11 + c21) + (c12 + c22));
          Real const cc = (1./6.)*((c11 + c22) + (c12 + c21));

          out += ((ca + cb) + cc) + ((ca + cc) + cb);
        }
        int const ny = s1::nghost;

        Real const c1 = s1::coeff[nx1] * s1::coeff[ny] * pu[nx1*stride[dirx] + ny*stride[diry]];
        Real const c2 = s1::coeff[nx2] * s1::coeff[ny] * pu[nx2*stride[dirx] + ny*stride[diry]];

        out += (c1 + c2);
      }
      int const nx = s1::nghost;
      for(int ny1 = 0; ny1 < s1::nghost; ++ny1) {
        int const ny2 = s1::width - ny1 - 1;

        Real const c1 = s1::coeff[nx] * s1::coeff[ny1] * pu[nx*stride[dirx] + ny1*stride[diry]];
        Real const c2 = s1::coeff[nx] * s1::coeff[ny2] * pu[nx*stride[dirx] + ny2*stride[diry]];

        out += (c1 + c2);
      }
      int const ny = s1::nghost;
      out += s1::coeff[nx] * s1::coeff[ny] * pu[nx*stride[dirx] + ny*stride[diry]];

      return out * idx[dirx] * idx[diry];
    }
    // Kreiss-Oliger dissipation operator
    inline Real Diss(int dir, Real & u) {
      Real * pu = &u - sd::offset*stride[dir];

      Real out(0.);
      for(int n1 = 0; n1 < sd::nghost; ++n1) {
        int const n2  = sd::width - n1 - 1;
        Real const c1 = sd::coeff[n1] * pu[n1*stride[dir]];
        Real const c2 = sd::coeff[n2] * pu[n2*stride[dir]];
        out += (c1 + c2);
      }
      out += sd::coeff[sd::nghost] * pu[sd::nghost*stride[dir]];

      return out * idx[dir] * diss;
    }
  } coordFD;

};

#endif // COORDINATES_COORDINATES_HPP_
