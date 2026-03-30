#ifndef COMM_COMM_SPEC_HPP_
#define COMM_COMM_SPEC_HPP_
//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file comm_spec.hpp
//  \brief CommSpec: all metadata needed to register a variable for
//  communication.
//
//  One CommSpec per logical variable (e.g. "hydro cons", "Z4c u").  The
//  registry creates a CommChannel from this spec; the spec itself is a pure
//  data struct with no behaviour.

#include <string>
#include <vector>

#include "../athena.hpp"  // Real, AthenaArray
#include "../athena_arrays.hpp"
#include "../mesh/mesh_topology.hpp"  // BoundaryFace, BoundaryFlag
#include "comm_enums.hpp"
#include "neighbor_connectivity.hpp"  // NeighborConnectivity

// forward declarations
class MeshBlock;
class Coordinates;

namespace comm
{

//----------------------------------------------------------------------------------------
// Parity component group: describes how a contiguous run of components
// transforms under coordinate reflections (needed for reflecting BCs and polar
// boundaries).
//
// Example for Z4c:  component_groups = {{GeomType::SymTensor, 6},
// {GeomType::Vector, 3}} means components 0-5 are a symmetric 3-tensor, 6-8
// are a 3-vector. Empty vector = all even parity (scalar-like, no sign flips).

struct ComponentGroup
{
  GeomType type;
  int count;  // number of components in this group
};

//----------------------------------------------------------------------------------------
// Physical boundary condition function signature.
// Called per-face to fill ghost zones for physical (non-block) boundaries.
// Receives the array, coordinates, and the ghost zone index range.

using PhysicalBCFn = void (*)(MeshBlock* pmb,
                              Coordinates* pco,
                              AthenaArray<Real>& u,
                              Real time,
                              Real dt,
                              int is,
                              int ie,
                              int js,
                              int je,
                              int ks,
                              int ke,
                              int ngh);

//----------------------------------------------------------------------------------------
//! \struct CommSpec
//  \brief Complete specification for registering a variable with the comm
//  system.

struct CommSpec
{
  // --- identity ---
  std::string label;  // human-readable name (e.g. "hydro_cons", "z4c_u")

  // --- data reference ---
  // CC/VC/CX: single 4D array (nvar, nk, nj, ni) via var/coarse_var.
  // FC: three 3D arrays via var_fc[3]/coarse_fc[3], one per face direction
  //     (var_fc[0]=x1f, var_fc[1]=x2f, var_fc[2]=x3f).  var/coarse_var
  //     ignored.
  AthenaArray<Real>* var;  // fine-level state array (CC/VC/CX only)
  AthenaArray<Real>*
    coarse_var;  // coarse buffer (CC/VC/CX only)
                 // TODO: allocate lazily at Finalize() only for blocks
                 //       with cross-level neighbors.
  AthenaArray<Real>* var_fc[3];     // fine-level face components (FC only)
  AthenaArray<Real>* coarse_fc[3];  // coarse face component buffers (FC only)
  int nvar;  // number of variable components in the leading index
             // (FC ignores this - component count is always 3)

  // --- flux correction data ---
  // CC: flx_cc[d] points to the flux AthenaArray for direction d.
  // FC: flx_fc[d] points to the edge EMF AthenaArray for direction d.
  // Null pointers = no flux correction for this channel.
  AthenaArray<Real>* flx_cc[3];  // CC flux per coordinate direction
  AthenaArray<Real>* flx_fc[3];  // FC edge-EMF per coordinate direction
  FluxCorrMode flcor_mode;       // how received flux data is applied
  CommGroup flux_group;          // group for flux correction messages
                         // (NumGroups = "none", i.e. no flux correction)

  // --- grid placement ---
  Sampling sampling;

  // --- communication scope ---
  CommTarget targets;  // which neighbor relationships to communicate
  CommGroup group;     // fusion group - channels in the same group are fused

  // --- ghost zone width ---
  int nghost;  // defaults to NGHOST; VC/CX may differ

  // --- refinement operators ---
  ProlongOp prolong_op;    // coarse-to-fine interpolation
  RestrictOp restrict_op;  // fine-to-coarse averaging/injection

  // --- physical boundary conditions ---
  // Per-face BC type.  Indexed by BoundaryFace (inner_x1=0 .. outer_x3=5).
  // Different domain faces can have different BC types (e.g. reflect on x1,
  // outflow on x2, polar_wedge on x2).  PhysicalBC::None means the face is
  // internal (block/periodic/polar) and needs no function-based fill.
  PhysicalBC physical_bc[6];
  PhysicalBCFn
    physical_bc_fn;  // custom function (only used when physical_bc[f] == User)

  // --- unpack mode ---
  UnpackMode unpack_mode;  // how received ghost data merges with local data

  // --- parity / sign flips ---
  std::vector<ComponentGroup> component_groups;
  // Empty = all even parity (no sign flips).
  // Non-empty = describes how consecutive component runs transform
  // geometrically.

  // --- convenience defaults ---
  CommSpec()
      : label("unnamed"),
        var(nullptr),
        coarse_var(nullptr),
        var_fc{ nullptr, nullptr, nullptr },
        coarse_fc{ nullptr, nullptr, nullptr },
        nvar(0),
        flx_cc{ nullptr, nullptr, nullptr },
        flx_fc{ nullptr, nullptr, nullptr },
        flcor_mode(FluxCorrMode::None),
        flux_group(CommGroup::NumGroups),
        sampling(Sampling::CC),
        targets(CommTarget::All),
        group(CommGroup::MainInt),
        nghost(NGHOST),
        prolong_op(ProlongOp::MinmodLinear),
        restrict_op(RestrictOp::VolumeWeighted),
        physical_bc{ PhysicalBC::None, PhysicalBC::None, PhysicalBC::None,
                     PhysicalBC::None, PhysicalBC::None, PhysicalBC::None },
        physical_bc_fn(nullptr),
        unpack_mode(UnpackMode::Default),
        component_groups()
  {
  }

  // --- helpers ---

  // True if any face has a non-None physical BC.
  bool HasAnyPhysicalBC() const
  {
    for (int f = 0; f < 6; ++f)
      if (physical_bc[f] != PhysicalBC::None)
        return true;
    return false;
  }
};

//----------------------------------------------------------------------------------------
// Map old BoundaryFlag enum to new PhysicalBC enum.
// Flags that correspond to communication (block, periodic, polar) map to None.

inline PhysicalBC MapBoundaryFlag(BoundaryFlag bf)
{
  switch (bf)
  {
    case BoundaryFlag::reflect:
      return PhysicalBC::Reflect;
    case BoundaryFlag::outflow:
      return PhysicalBC::Outflow;
    case BoundaryFlag::extrapolate_outflow:
      return PhysicalBC::ExtrapolateOutflow;
    case BoundaryFlag::gr_sommerfeld:
      return PhysicalBC::GRSommerfeld;
    case BoundaryFlag::polar_wedge:
      return PhysicalBC::PolarWedge;
    case BoundaryFlag::user:
      return PhysicalBC::User;
    default:
      return PhysicalBC::None;
  }
}

//----------------------------------------------------------------------------------------
// Populate physical_bc[6] from the MeshBlock's neighbor connectivity.
// Call after MeshBlock is constructed so that boundary flags are available.

inline void SetPhysicalBCFromBlockBCs(CommSpec& spec,
                                      const NeighborConnectivity& nc)
{
  for (int f = 0; f < 6; ++f)
    spec.physical_bc[f] = MapBoundaryFlag(nc.boundary_flag(f));
}

}  // namespace comm

#endif  // COMM_COMM_SPEC_HPP_
