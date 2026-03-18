#ifndef COMM_AMR_SPEC_HPP_
#define COMM_AMR_SPEC_HPP_
//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file amr_spec.hpp
//  \brief AMRSpec: metadata for enrolling a variable in AMR redistribution.
//
//  Analogous to CommSpec (which drives ghost-zone exchange), AMRSpec declares
//  what a physics module needs from the AMR block-transfer system.  One
//  AMRSpec per logical variable (e.g. "hydro cons", "Z4c u", "M1 cons").
//
//  Key differences from CommSpec:
//    - AMR enrollment can differ from ghost-exchange enrollment.
//      E.g. z4c_aux_adm has a CommSpec but no AMR enrollment; M1 enrolls in
//      both pvars_cc_ AND pvars_m1_cc_; Weyl toggles enrollment via
//      DBG_REDUCE_AUX_COMM.
//    - Group-based selection (AMRGroup) replaces the std::swap(pvars_cc_,
//    pvars_aux_cc_)
//      pattern for multi-pass AMR.
//    - Per-variable prolong/restrict ops, matching CommSpec's flexibility.
//    - No physical BCs, no parity, no flux correction - those are
//    ghost-exchange concerns.

#include <string>

#include "../athena.hpp"  // Real, AthenaArray
#include "../athena_arrays.hpp"
#include "comm_enums.hpp"  // Sampling, ProlongOp, RestrictOp

// forward declarations
struct FaceField;

namespace comm
{

//----------------------------------------------------------------------------------------
// AMRGroup: which pass of AMR redistribution a variable participates in.
//
// During AMR, the orchestrator may run multiple passes (e.g. main hydro
// variables, then Z4c auxiliary variables, then M1).  Each pass selects an
// AMRGroup to pack/unpack. This replaces the legacy std::swap(pvars_cc_,
// pvars_aux_cc_) mechanism.
//
// A variable belongs to exactly one group.

enum class AMRGroup : int
{
  Main = 0,  // primary integrator variables (hydro cons, field b, scalars, z4c
             // u, wave)
  Aux = 1,   // auxiliary variables (z4c Weyl abvar, z4c aux ADM derivatives)
  M1  = 2,   // M1 radiation transport
  NumGroups = 3  // sentinel - total number of groups
};

//----------------------------------------------------------------------------------------
//! \struct AMRSpec
//  \brief Complete specification for enrolling a variable in AMR block
//  redistribution.
//
//  Pure data struct with no behaviour - the AMRRegistry creates operational
//  state from it.

struct AMRSpec
{
  // --- identity ---
  std::string label;  // human-readable name (e.g. "hydro_cons", "z4c_u")

  // --- data references ---
  // CC/VC/CX: single 4D array (nvar, nk, nj, ni) via var/coarse_var.
  // FC: use face_var/coarse_face_var (FaceField*) which each contain
  // x1f/x2f/x3f.
  AthenaArray<Real>* var;         // fine-level state array (CC/VC/CX only)
  AthenaArray<Real>* coarse_var;  // coarse buffer for restriction/prolongation
  FaceField* face_var;            // fine-level face field (FC only)
  FaceField* coarse_face_var;     // coarse face field buffer (FC only)
  int nvar;  // number of variable components (leading index)
             // (FC ignores this - always 3 face directions)

  // --- grid placement ---
  Sampling sampling;  // CC, VC, CX, or FC

  // --- AMR group ---
  AMRGroup group;  // which redistribution pass this variable belongs to

  // --- refinement operators ---
  // Used for cross-level transfers (f2c restriction before packing, c2f
  // prolongation after unpacking).  Same-level transfers are simple block
  // copies, no operators needed.
  ProlongOp prolong_op;    // coarse-to-fine interpolation
  RestrictOp restrict_op;  // fine-to-coarse averaging/injection

  // --- convenience defaults ---
  AMRSpec()
      : label("unnamed"),
        var(nullptr),
        coarse_var(nullptr),
        face_var(nullptr),
        coarse_face_var(nullptr),
        nvar(0),
        sampling(Sampling::CC),
        group(AMRGroup::Main),
        prolong_op(ProlongOp::MinmodLinear),
        restrict_op(RestrictOp::VolumeWeighted)
  {
  }
};

}  // namespace comm

#endif  // COMM_AMR_SPEC_HPP_
