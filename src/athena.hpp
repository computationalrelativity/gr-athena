#ifndef ATHENA_HPP_
#define ATHENA_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file athena.hpp
//  \brief contains Athena++ general purpose types, structures, enums, etc.

// C headers

// C++ headers
#include <cmath>
#include <cstdint>  // std::int64_t

// Athena++ headers
#include "athena_arrays.hpp"
#include "defs.hpp"

// primitive type alias that allows code to run with either floats or doubles
#if SINGLE_PRECISION_ENABLED
using Real = float;
#ifdef MPI_PARALLEL
#define MPI_ATHENA_REAL MPI_FLOAT
#endif
#else
using Real = double;
#ifdef MPI_PARALLEL
#define MPI_ATHENA_REAL MPI_DOUBLE
#endif
#endif

// for OpenMP 4.0 SIMD vectorization, control width of SIMD lanes
#if defined(__AVX512F__)
#define SIMD_WIDTH 8
#elif defined(__AVX__)
#define SIMD_WIDTH 4
#elif defined(__SSE2__)
#define SIMD_WIDTH 2
#else
#define SIMD_WIDTH 4
#endif

#define CACHELINE_BYTES 64

// forward declarations needed for function pointer type aliases
class MeshBlock;
class Coordinates;
class ParameterInput;

//--------------------------------------------------------------------------------------
//! \struct LogicalLocation
//  \brief stores logical location and level of MeshBlock

struct LogicalLocation { // aggregate and POD type
  // These values can exceed the range of std::int32_t even if the root grid has only a
  // single MeshBlock if >30 levels of AMR are used, since the corresponding max index =
  // 1*2^31 > INT_MAX = 2^31 -1 for most 32-bit signed integer type impelementations
  std::int64_t lx1, lx2, lx3;
  int level;

  // operators useful for sorting
  bool operator==(LogicalLocation &ll) {
    return ((ll.level == level) && (ll.lx1 == lx1) && (ll.lx2 == lx2) && (ll.lx3 == lx3));
  }
  static bool Lesser(const LogicalLocation &left, const LogicalLocation &right) {
    return left.level < right.level;
  }
  static bool Greater(const LogicalLocation & left, const LogicalLocation &right) {
    return left.level > right.level;
  }
};

//----------------------------------------------------------------------------------------
//! \struct RegionSize
//  \brief physical size and number of cells in a Mesh or a MeshBlock

struct RegionSize {  // aggregate and POD type; do NOT reorder member declarations:
  Real x1min, x2min, x3min;
  Real x1max, x2max, x3max;
  Real x1rat, x2rat, x3rat; // ratio of dxf(i)/dxf(i-1)
  // the size of the root grid or a MeshBlock should not exceed std::int32_t limits
  int nx1, nx2, nx3;        // number of active cells (not including ghost zones)
};

//------------------------------------------------------------------------------------// enums used everywhere
// (not specifying underlying integral type (C++11) for portability & performance)

// TODO(felker): C++ Core Guidelines Enum.5: Don’t use ALL_CAPS for enumerators
// (avoid clashes with preprocessor macros). Enumerated type definitions in this file and:
// athena_fft.hpp, io_wrapper.hpp, bvals.hpp, hydro_diffusion.hpp, field_diffusion.hpp,
// task_list.hpp, ???

//------------------
// named, weakly typed / unscoped enums:
//------------------

// enumerators only used for indexing AthenaArray and regular arrays; typename and
// explicitly specified enumerator values aare unnecessary, but provided for clarity:

// array indices for 4-metric and triangular matrices in GR
enum MetricIndex {I00=0, I01=1, I02=2, I03=3, I11=4, I12=5, I13=6, I22=7, I23=8, I33=9,
  NMETRIC=10};
enum TriangleIndex {T00=0, T10=1, T11=2, T20=3, T21=4, T22=5, T30=6, T31=7, T32=8, T33=9,
  NTRIANGULAR=10};

// array indices for 3-metric and extrinsic curv in GR in 3D
enum SpatialMetricIndex {S11=0, S12=1, S13=2, S22=3, S23=4, S33=5,
  NSPMETRIC=6};

// enumerator types that are used for variables and function parameters:

// needed for arrays dimensioned over grid directions
// enumerator type only used in Mesh::EnrollUserMeshGenerator()
enum CoordinateDirection {X1DIR=0, X2DIR=1, X3DIR=2};

//------------------
// strongly typed / scoped enums (C++11):
//------------------
// KGF: Except for the 2x MG* enums, these may be unnessary w/ the new class inheritance
// Now, only passed to BoundaryVariable::InitBoundaryData(); could replace w/ bool switch
enum class BoundaryQuantity {cc, cx, fc, vc};
enum class BoundaryCommSubset {mesh_init, all};
enum class UserHistoryOperation {sum, max, min};

//----------------------------------------------------------------------------------------
// function pointer prototypes for user-defined modules set at runtime

using BValFunc = void (*)(
    MeshBlock *pmb, Coordinates *pco,
    Real time, Real dt,
    int is, int ie, int js, int je, int ks, int ke, int ngh);
using AMRFlagFunc = int (*)(MeshBlock *pmb);
using MeshGenFunc = Real (*)(Real x, RegionSize rs);
using TimeStepFunc = Real (*)(MeshBlock *pmb);
using HistoryOutputFunc = Real (*)(MeshBlock *pmb, int iout);
using MetricFunc = void (*)(
    Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv,
    AthenaArray<Real> &dg_dx1, AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3);

#endif // ATHENA_HPP_
