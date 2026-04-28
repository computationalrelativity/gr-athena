#ifndef COMM_PHYSICAL_BCS_HPP_
#define COMM_PHYSICAL_BCS_HPP_
//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file physical_bcs.hpp
//  \brief Physical boundary condition dispatch for the data-driven comm system.
//
//  Replaces the virtual BoundaryPhysics interface and DispatchBoundaryFunctions()
//  with free functions dispatched by PhysicalBC enum + Sampling.
//
//  BC operations fill ghost zones at domain faces using only the variable array
//  and index ranges - no subclass polymorphism required.  For BCs that require
//  parity sign flips (Reflect, PolarWedge), the dispatch layer applies sign
//  corrections after the geometric fill using the parity module.

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh_topology.hpp"  // BoundaryFace
#include "comm_enums.hpp"

// forward declarations
class MeshBlock;
class Coordinates;

namespace comm {

struct CommSpec;
class NeighborConnectivity;

//----------------------------------------------------------------------------------------
// Apply a single physical BC implementation on one face of one variable.
// This is the per-face, per-variable inner call.  Includes parity sign flip
// post-processing for Reflect and PolarWedge BCs (computed from spec.component_groups).
//
//   var    - array to fill (may be fine or coarse, depending on caller)
//   face   - which domain face (inner/outer x1/x2/x3)
//   il,iu,jl,ju,kl,ku - index range (interior bounds, BC extends beyond)
//   ngh    - ghost zone width
//   spec   - CommSpec for this channel (provides nvar, sampling, component_groups)

void ApplyPhysicalBCFace(AthenaArray<Real> &var,
                         BoundaryFace face,
                         int il, int iu, int jl, int ju, int kl, int ku,
                         int ngh, PhysicalBC bc, const CommSpec &spec);

//----------------------------------------------------------------------------------------
// Apply physical BCs on all active faces for one channel, fine level.
//
//   pmb    - MeshBlock
//   spec   - CommSpec for this channel
//   time, dt - for user-enrolled BCs

void ApplyPhysicalBCs(MeshBlock *pmb, const CommSpec &spec,
                      Real time, Real dt);

//----------------------------------------------------------------------------------------
// Apply physical BCs on the coarse representation for prolongation.
//
// Operates on spec.coarse_var with coarse-level coordinates (pmr->pcoarsec).
// The face ordering and transverse-limit extension logic matches the old system
// exactly: x1 faces first, then x2 (with extended x1 limits), then x3 (with
// extended x1 and x2 limits).
//
// Does NOT call InterchangeFundamentalCoarse - instead it simply passes the coarse
// array directly. The old system needed the swap because BC functions operated on
// the `var_cc` member; the new system passes the target array explicitly.

void ApplyPhysicalBCsOnCoarseLevel(MeshBlock *pmb, const CommSpec &spec,
                                   Real time, Real dt,
                                   int cis, int cie, int cjs, int cje,
                                   int cks, int cke, int cng);

//----------------------------------------------------------------------------------------
// Individual BC fill implementations.
// These are pure ghost-zone fill operations: they read interior data and write ghost data.
// Parity sign flips are applied by the dispatch layer (ApplyPhysicalBCFace) after these.

// Reflect: mirror-image copy. Ghost cell (boundary-i) <- interior cell (boundary+i-1).
void ReflectBC(AthenaArray<Real> &var, int nvar, BoundaryFace face,
               int il, int iu, int jl, int ju, int kl, int ku, int ngh);

// Outflow: zeroth-order extrapolation. All ghost cells = boundary cell value.
void OutflowBC(AthenaArray<Real> &var, int nvar, BoundaryFace face,
               int il, int iu, int jl, int ju, int kl, int ku, int ngh);

// ExtrapolateOutflow: 4th-order polynomial extrapolation into ghost zones.
void ExtrapolateOutflowBC(AthenaArray<Real> &var, int nvar, BoundaryFace face,
                          int il, int iu, int jl, int ju, int kl, int ku, int ngh);

// PolarWedge: mirror copy across x2 boundary with j-reversal.
// Only valid for inner/outer x2. Parity sign flips applied by dispatch layer.
void PolarWedgeBC(AthenaArray<Real> &var, int nvar, BoundaryFace face,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//========================================================================================
// Face-centered (FC) boundary condition functions
//========================================================================================
//
// FC BCs operate on FaceField (x1f, x2f, x3f) rather than a single 4D array.
// Each component has a stagger-direction loop range extended by +1 in its own direction.
// Parity is NOT handled by the component_groups/parity module; instead, sign flips are
// hardcoded in the BC functions themselves (normal component negated for Reflect,
// per-component signs from flip_across_pole_field for PolarWedge).

// Reflect: mirror-image copy.  Normal component is negated; tangentials are not.
// Each component extends its own stagger direction by +1 in the i-loop range.
void ReflectBC_FC(AthenaArray<Real> *fc[3], BoundaryFace face,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);

// Outflow: zeroth-order extrapolation.  All components copied from boundary face/cell.
// Normal component copies from the boundary face; tangentials from the boundary cell.
void OutflowBC_FC(AthenaArray<Real> *fc[3], BoundaryFace face,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);

// PolarWedge: mirror copy across x2 boundary with per-component sign from
// flip_across_pole_field = {false, true, true} -> signs {+1, -1, -1}.
// Additionally zeros the normal (x2f) face at the pole.
// Only valid for inner/outer x2.
void PolarWedgeBC_FC(AthenaArray<Real> *fc[3], BoundaryFace face,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//----------------------------------------------------------------------------------------
// FC single-face dispatch: select the appropriate BC implementation.
// GRSommerfeld -> OutflowBC_FC.  ExtrapolateOutflow and User -> ATHENA_ERROR.
void ApplyPhysicalBCFace_FC(AthenaArray<Real> *fc[3], BoundaryFace face,
                            int il, int iu, int jl, int ju, int kl, int ku,
                            int ngh, PhysicalBC bc);

//----------------------------------------------------------------------------------------
// FC multi-face dispatch: apply BCs on all active domain faces for one FC channel.
// Face ordering and transverse-limit extension match ApplyPhysicalBCs (CC version).
void ApplyPhysicalBCs_FC(MeshBlock *pmb, const CommSpec &spec,
                         Real time, Real dt);

//----------------------------------------------------------------------------------------
// FC coarse-level multi-face dispatch for prolongation.
// Operates on spec.coarse_fc with coarse-level coordinates.
void ApplyPhysicalBCsOnCoarseLevel_FC(MeshBlock *pmb, const CommSpec &spec,
                                      Real time, Real dt,
                                      int cis, int cie, int cjs, int cje,
                                      int cks, int cke, int cng);

} // namespace comm

#endif // COMM_PHYSICAL_BCS_HPP_
