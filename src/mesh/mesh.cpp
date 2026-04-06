//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file mesh.cpp
//  \brief implementation of functions in Mesh class

// C headers
// pre-C11: needed before including inttypes.h, else won't define int64_t for
// C++ code #define __STDC_FORMAT_MACROS

// C++ headers
#include <algorithm>
#include <cinttypes>  // format macro "PRId64" for fixed-width integer type std::int64_t
#include <cmath>    // std::abs(), std::pow()
#include <cstdint>  // std::int64_t fixed-wdith integer type alias
#include <cstdio>   // std::printf
#include <cstdlib>
#include <cstring>  // std::memcpy()
#include <iomanip>  // std::setprecision()
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../outputs/io_wrapper.hpp"
#include "../parameter_input.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/buffer_utils.hpp"
#include "../z4c/ahf.hpp"
#include "../z4c/puncture_tracker.hpp"
#include "../z4c/wave_extract.hpp"
#include "../z4c/wave_extract_rwz.hpp"
#include "../z4c/z4c.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"
#include "meshblock_tree.hpp"
#include "surfaces.hpp"

#if CCE_ENABLED
#include "../z4c/cce/cce.hpp"
#endif
#include "../trackers/extrema_tracker.hpp"

#ifdef EJECTA_ENABLED
#include "../z4c/ejecta.hpp"
#endif

#include "../comm/amr_registry.hpp"
#include "../comm/comm_registry.hpp"
#include "../comm/reconcile_faces.hpp"
#include "../m1/m1.hpp"
#include "../wave/wave.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// Mesh constructor, builds mesh at start of calculation using parameters in
// input file

Mesh::Mesh(ParameterInput* pin, int mesh_test)
    :  // public members:
       // aggregate initialization of RegionSize struct:
       // (x1rat, x2rat, x3rat, nx1, nx2, nx3 are set in this initializer list,
       // which is called before Mesh body; remaining 6 members are initialized
       // in the Mesh() body)
      mesh_size{ pin->GetReal("mesh", "x1min"),
                 pin->GetReal("mesh", "x2min"),
                 pin->GetReal("mesh", "x3min"),
                 pin->GetReal("mesh", "x1max"),
                 pin->GetReal("mesh", "x2max"),
                 pin->GetReal("mesh", "x3max"),
                 pin->GetOrAddReal("mesh", "x1rat", 1.0),
                 pin->GetOrAddReal("mesh", "x2rat", 1.0),
                 pin->GetOrAddReal("mesh", "x3rat", 1.0),
                 pin->GetInteger("mesh", "nx1"),
                 pin->GetInteger("mesh", "nx2"),
                 pin->GetInteger("mesh", "nx3") },
      mesh_bcs{ GetBoundaryFlag(pin->GetOrAddString("mesh", "ix1_bc", "none")),
                GetBoundaryFlag(pin->GetOrAddString("mesh", "ox1_bc", "none")),
                GetBoundaryFlag(pin->GetOrAddString("mesh", "ix2_bc", "none")),
                GetBoundaryFlag(pin->GetOrAddString("mesh", "ox2_bc", "none")),
                GetBoundaryFlag(pin->GetOrAddString("mesh", "ix3_bc", "none")),
                GetBoundaryFlag(
                  pin->GetOrAddString("mesh", "ox3_bc", "none")) },
      f2(mesh_size.nx2 > 1 ? true : false),
      f3(mesh_size.nx3 > 1 ? true : false),
      ndim(f3 ? 3 : (f2 ? 2 : 1)),
      adaptive(pin->GetOrAddString("mesh", "refinement", "none") == "adaptive"
                 ? true
                 : false),
      multilevel(
        (adaptive ||
         pin->GetOrAddString("mesh", "refinement", "none") == "static")
          ? true
          : false),
      start_time(pin->GetOrAddReal("time", "start_time", 0.0)),
      time(start_time),
      tlim(pin->GetReal("time", "tlim")),
      dt(std::numeric_limits<Real>::max()),
      dt_hyperbolic(dt),
      dt_parabolic(dt),
      dt_user(dt),
      cfl_number(pin->GetReal("time", "cfl_number")),
      nlim(pin->GetOrAddInteger("time", "nlim", -1)),
      ncycle(),
      ncycle_out(pin->GetOrAddInteger("time", "ncycle_out", 1)),
      dt_diagnostics(pin->GetOrAddInteger("time", "dt_diagnostics", -1)),
      nbnew(),
      nbdel(),
      step_since_lb(),
      // private members:
      num_mesh_threads_(pin->GetOrAddInteger("mesh", "num_threads", 1)),
      tree(this),
      use_uniform_meshgen_fn_{ true, true, true },
      nreal_user_mesh_data_(),
      nint_user_mesh_data_(),
      lb_flag_(true),
      lb_automatic_(),
      lb_manual_(),
      MeshGenerator_{ UniformMeshGeneratorX1,
                      UniformMeshGeneratorX2,
                      UniformMeshGeneratorX3 },
      BoundaryFunction_{
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
      },
      AMRFlag_{},
      UserTimeStep_{},
      UserMainLoopBreak_{}
{
  std::stringstream msg;
  RegionSize block_size;
  MeshBlock* pfirst{};
  BoundaryFlag block_bcs[6];
  std::int64_t nbmax;
  resume_flag = false;
  // mesh test
  if (mesh_test > 0)
    Globals::nranks = mesh_test;

  // check number of OpenMP threads for mesh
  if (num_mesh_threads_ < 1)
  {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Number of OpenMP threads must be >= 1, but num_threads="
        << num_mesh_threads_ << std::endl;
    ATHENA_ERROR(msg);
  }

  // check number of grid cells in root level of mesh from input file.
  if (mesh_size.nx1 < 4)
  {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file nx1 must be >= 4, but nx1="
        << mesh_size.nx1 << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mesh_size.nx2 < 1)
  {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file nx2 must be >= 1, but nx2="
        << mesh_size.nx2 << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mesh_size.nx3 < 1)
  {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file nx3 must be >= 1, but nx3="
        << mesh_size.nx3 << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mesh_size.nx2 == 1 && mesh_size.nx3 > 1)
  {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file: nx2=1, nx3=" << mesh_size.nx3
        << ", 2D problems in x1-x3 plane not supported" << std::endl;
    ATHENA_ERROR(msg);
  }

  // check physical size of mesh (root level) from input file.
  if (mesh_size.x1max <= mesh_size.x1min)
  {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Input x1max must be larger than x1min: x1min=" << mesh_size.x1min
        << " x1max=" << mesh_size.x1max << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mesh_size.x2max <= mesh_size.x2min)
  {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Input x2max must be larger than x2min: x2min=" << mesh_size.x2min
        << " x2max=" << mesh_size.x2max << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mesh_size.x3max <= mesh_size.x3min)
  {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Input x3max must be larger than x3min: x3min=" << mesh_size.x3min
        << " x3max=" << mesh_size.x3max << std::endl;
    ATHENA_ERROR(msg);
  }

  // check the consistency of the periodic boundaries
  if (((mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic &&
        mesh_bcs[BoundaryFace::outer_x1] != BoundaryFlag::periodic) ||
       (mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic &&
        mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::periodic)) ||
      (mesh_size.nx2 > 1 &&
       ((mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic &&
         mesh_bcs[BoundaryFace::outer_x2] != BoundaryFlag::periodic) ||
        (mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic &&
         mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::periodic))) ||
      (mesh_size.nx3 > 1 &&
       ((mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic &&
         mesh_bcs[BoundaryFace::outer_x3] != BoundaryFlag::periodic) ||
        (mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic &&
         mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::periodic))))
  {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "When periodic boundaries are in use, both sides must be periodic."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  // read and set MeshBlock parameters
  block_size.x1rat = mesh_size.x1rat;
  block_size.x2rat = mesh_size.x2rat;
  block_size.x3rat = mesh_size.x3rat;
  block_size.nx1   = pin->GetOrAddInteger("meshblock", "nx1", mesh_size.nx1);
  if (f2)
    block_size.nx2 = pin->GetOrAddInteger("meshblock", "nx2", mesh_size.nx2);
  else
    block_size.nx2 = mesh_size.nx2;
  if (f3)
    block_size.nx3 = pin->GetOrAddInteger("meshblock", "nx3", mesh_size.nx3);
  else
    block_size.nx3 = mesh_size.nx3;

  // Allocate per-thread scratch caches
  {
    int nc1 = block_size.nx1 + 2 * NGHOST;
    int nc2 = (f2) ? block_size.nx2 + 2 * NGHOST : 1;
    int nc3 = (f3) ? block_size.nx3 + 2 * NGHOST : 1;
    thread_caches_.resize(num_mesh_threads_);
    for (auto& tc : thread_caches_)
    {
      tc.AllocateFCGeom(nc1, nc2, nc3);
      if (FLUID_ENABLED &&
          pin->GetOrAddBoolean("time", "xorder_use_fb", false))
        tc.AllocateLOFlux(nc1, nc2, nc3, f2, f3);
      if (M1_ENABLED)
      {
        bool m1_fb_E = pin->GetOrAddBoolean("M1", "flux_lo_fallback_E", false);
        bool m1_fb_nG =
          pin->GetOrAddBoolean("M1", "flux_lo_fallback_nG", false);
        if (m1_fb_E || m1_fb_nG)
        {
          int ngrps = pin->GetOrAddInteger("M1", "ngroups", 1);
          int nspcs = pin->GetOrAddInteger("M1", "nspecies", 1);
          int ngs   = M1::M1::ixn_Lab::N * ngrps * nspcs;
          tc.AllocateM1LOFlux(nc1, nc2, nc3, ngs);
        }
      }
    }
  }

  // check consistency of the block and mesh
  if (mesh_size.nx1 % block_size.nx1 != 0 ||
      mesh_size.nx2 % block_size.nx2 != 0 ||
      mesh_size.nx3 % block_size.nx3 != 0)
  {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "the Mesh must be evenly divisible by the MeshBlock" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (block_size.nx1 < 4 || (block_size.nx2 < 4 && f2) ||
      (block_size.nx3 < 4 && f3))
  {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "block_size must be larger than or equal to 4 cells." << std::endl;
    ATHENA_ERROR(msg);
  }

  if (multilevel)
  {
#ifdef DBG_VC_DOUBLE_RESTRICT
    // MeshBlock minimum size is constrained in multilevel due to
    // double-restrict
    int const min_mb_nx = std::max(4, 4 * NCGHOST - 2);

    if (block_size.nx1 < min_mb_nx || (block_size.nx2 < min_mb_nx && f2) ||
        (block_size.nx3 < min_mb_nx && f3))
    {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "for multilevel with '-vertex' and NCGHOST = " << NCGHOST << ", "
          << "block_size must be at least " << min_mb_nx << std::endl
          << "Criterion: max(4, 4 * NCGHOST - 2)" << std::endl;
      ATHENA_ERROR(msg);
    }
#endif  // DBG_VC_DOUBLE_RESTRICT
  }

  // calculate the number of the blocks
  nrbx1 = mesh_size.nx1 / block_size.nx1;
  nrbx2 = mesh_size.nx2 / block_size.nx2;
  nrbx3 = mesh_size.nx3 / block_size.nx3;
  nbmax = (nrbx1 > nrbx2) ? nrbx1 : nrbx2;
  nbmax = (nbmax > nrbx3) ? nbmax : nrbx3;

  // initialize user-enrollable functions
  if (mesh_size.x1rat != 1.0)
  {
    use_uniform_meshgen_fn_[X1DIR] = false;
    MeshGenerator_[X1DIR]          = DefaultMeshGeneratorX1;
  }
  if (mesh_size.x2rat != 1.0)
  {
    use_uniform_meshgen_fn_[X2DIR] = false;
    MeshGenerator_[X2DIR]          = DefaultMeshGeneratorX2;
  }
  if (mesh_size.x3rat != 1.0)
  {
    use_uniform_meshgen_fn_[X3DIR] = false;
    MeshGenerator_[X3DIR]          = DefaultMeshGeneratorX3;
  }

  // calculate the logical root level and maximum level
  for (root_level = 0; (1 << root_level) < nbmax; root_level++)
  {
  }
  current_level = root_level;

  tree.CreateRootGrid();

  // Load balancing flag and parameters
#ifdef MPI_PARALLEL
  if (pin->GetOrAddString("loadbalancing", "balancer", "default") ==
      "automatic")
    lb_automatic_ = true;
  else if (pin->GetOrAddString("loadbalancing", "balancer", "default") ==
           "manual")
    lb_manual_ = true;
  lb_tolerance_ = pin->GetOrAddReal("loadbalancing", "tolerance", 0.5);
  lb_interval_  = pin->GetOrAddReal("loadbalancing", "interval", 10);
#endif

  // SMR / AMR:
  if (adaptive)
  {
    max_level = pin->GetOrAddInteger("mesh", "numlevel", 1) + root_level - 1;
    if (max_level > 63)
    {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The number of the refinement level must be smaller than "
          << 63 - root_level + 1 << "." << std::endl;
      ATHENA_ERROR(msg);
    }
  }
  else
  {
    max_level = 63;
  }

  if (Z4C_ENABLED)
  {
    int nrad = pin->GetOrAddInteger("psi4_extraction", "num_radii", 0);
    if (nrad > 0)
    {
      pwave_extr.reserve(nrad);
      for (int n = 0; n < nrad; ++n)
      {
        pwave_extr.push_back(new WaveExtract(this, pin, n));
      }
    }
    int nrad_rwz = pin->GetOrAddInteger("rwz_extraction", "num_radii", 0);
    if (nrad_rwz > 0)
    {
      pwave_extr_rwz.reserve(nrad_rwz);
      for (int n = 0; n < nrad_rwz; ++n)
      {
        pwave_extr_rwz.push_back(new WaveExtractRWZ(this, pin, n));
      }
    }

#if CCE_ENABLED
    // CCE
    int ncce = pin->GetOrAddInteger("cce", "num_radii", 0);
    pcce.reserve(10 * ncce);  // 10 different components for each radius
    for (int n = 0; n < ncce; ++n)
    {
      // NOTE: these names are used for pittnull code, so DON'T change the
      // convention
      pcce.push_back(new CCE(this, pin, "gxx", n));
      pcce.push_back(new CCE(this, pin, "gxy", n));
      pcce.push_back(new CCE(this, pin, "gxz", n));
      pcce.push_back(new CCE(this, pin, "gyy", n));
      pcce.push_back(new CCE(this, pin, "gyz", n));
      pcce.push_back(new CCE(this, pin, "gzz", n));
      pcce.push_back(new CCE(this, pin, "betax", n));
      pcce.push_back(new CCE(this, pin, "betay", n));
      pcce.push_back(new CCE(this, pin, "betaz", n));
      pcce.push_back(new CCE(this, pin, "alp", n));
    }
#endif

#ifdef EJECTA_ENABLED
    const int nejecta = pin->GetOrAddInteger("ejecta", "num_rad", 0);
    pej_extract.reserve(nejecta);
    for (int n = 0; n < nejecta; ++n)
    {
      pej_extract.push_back(new Ejecta(this, pin, n));
    }

    /*
    // Ejecta analysis
    const int nejecta = pin->GetOrAddRealArray("ejecta", "R", -1, 0).GetSize();
    pej_extract.reserve(nejecta);
    for (int n=0; n<nejecta; ++n)
    {
      pej_extract.push_back(new Ejecta(this, pin, n, nejecta));
    }
    */
#endif

    // Puncture Trackers
    int npunct = pin->GetOrAddInteger("z4c", "npunct", 0);
    if (npunct > 0)
    {
      pz4c_tracker.reserve(npunct);
      for (int n = 0; n < npunct; ++n)
      {
        pz4c_tracker.push_back(new PunctureTracker(this, pin, n));
      }
    }
  }

  // Last entry says if it is restart run or not
  ptracker_extrema = new ExtremaTracker(this, pin, 0);

  if (Z4C_ENABLED)
  {
    // AHF (0 is restart flag for restart)
    int nhorizon = pin->GetOrAddInteger("ahf", "num_horizons", 0);
    pah_finder.reserve(nhorizon);
    for (int n = 0; n < nhorizon; ++n)
    {
      pah_finder.push_back(new AHF(this, pin, n));
    }
  }

  InitUserMeshData(pin);

  if (multilevel)
  {
    if (block_size.nx1 % 2 == 1 || (block_size.nx2 % 2 == 1 && f2) ||
        (block_size.nx3 % 2 == 1 && f3))
    {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The size of MeshBlock must be divisible by 2 in order to use "
             "SMR or AMR."
          << std::endl;
      ATHENA_ERROR(msg);
    }

    InputBlock* pib = pin->pfirst_block;
    while (pib != nullptr)
    {
      if (pib->block_name.compare(0, 10, "refinement") == 0)
      {
        RegionSize ref_size;
        ref_size.x1min = pin->GetReal(pib->block_name, "x1min");
        ref_size.x1max = pin->GetReal(pib->block_name, "x1max");
        if (f2)
        {
          ref_size.x2min = pin->GetReal(pib->block_name, "x2min");
          ref_size.x2max = pin->GetReal(pib->block_name, "x2max");
        }
        else
        {
          ref_size.x2min = mesh_size.x2min;
          ref_size.x2max = mesh_size.x2max;
        }
        if (ndim == 3)
        {
          ref_size.x3min = pin->GetReal(pib->block_name, "x3min");
          ref_size.x3max = pin->GetReal(pib->block_name, "x3max");
        }
        else
        {
          ref_size.x3min = mesh_size.x3min;
          ref_size.x3max = mesh_size.x3max;
        }
        int ref_lev = pin->GetInteger(pib->block_name, "level");
        int lrlev   = ref_lev + root_level;
        if (lrlev > current_level)
          current_level = lrlev;
        // range check
        if (ref_lev < 1)
        {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Refinement level must be larger than 0 (root level = 0)"
              << std::endl;
          ATHENA_ERROR(msg);
        }
        if (lrlev > max_level)
        {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Refinement level exceeds the maximum level (specify "
              << "'maxlevel' parameter in <mesh> input block if adaptive)."
              << std::endl;
          ATHENA_ERROR(msg);
        }
        if (ref_size.x1min > ref_size.x1max ||
            ref_size.x2min > ref_size.x2max || ref_size.x3min > ref_size.x3max)
        {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Invalid refinement region is specified." << std::endl;
          ATHENA_ERROR(msg);
        }
        if (ref_size.x1min < mesh_size.x1min ||
            ref_size.x1max > mesh_size.x1max ||
            ref_size.x2min < mesh_size.x2min ||
            ref_size.x2max > mesh_size.x2max ||
            ref_size.x3min < mesh_size.x3min ||
            ref_size.x3max > mesh_size.x3max)
        {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Refinement region must be smaller than the whole mesh."
              << std::endl;
          ATHENA_ERROR(msg);
        }
        // find the logical range in the ref_level
        // note: if this is too slow, this should be replaced with bi-section
        // search.
        std::int64_t lx1min = 0, lx1max = 0, lx2min = 0, lx2max = 0,
                     lx3min = 0, lx3max = 0;
        std::int64_t lxmax = nrbx1 * (1LL << ref_lev);
        for (lx1min = 0; lx1min < lxmax; lx1min++)
        {
          Real rx = ComputeMeshGeneratorX(
            lx1min + 1, lxmax, use_uniform_meshgen_fn_[X1DIR]);
          if (MeshGenerator_[X1DIR](rx, mesh_size) > ref_size.x1min)
            break;
        }
        for (lx1max = lx1min; lx1max < lxmax; lx1max++)
        {
          Real rx = ComputeMeshGeneratorX(
            lx1max + 1, lxmax, use_uniform_meshgen_fn_[X1DIR]);
          if (MeshGenerator_[X1DIR](rx, mesh_size) >= ref_size.x1max)
            break;
        }
        if (lx1min % 2 == 1)
          lx1min--;
        if (lx1max % 2 == 0)
          lx1max++;
        if (f2)
        {  // 2D or 3D
          lxmax = nrbx2 * (1LL << ref_lev);
          for (lx2min = 0; lx2min < lxmax; lx2min++)
          {
            Real rx = ComputeMeshGeneratorX(
              lx2min + 1, lxmax, use_uniform_meshgen_fn_[X2DIR]);
            if (MeshGenerator_[X2DIR](rx, mesh_size) > ref_size.x2min)
              break;
          }
          for (lx2max = lx2min; lx2max < lxmax; lx2max++)
          {
            Real rx = ComputeMeshGeneratorX(
              lx2max + 1, lxmax, use_uniform_meshgen_fn_[X2DIR]);
            if (MeshGenerator_[X2DIR](rx, mesh_size) >= ref_size.x2max)
              break;
          }
          if (lx2min % 2 == 1)
            lx2min--;
          if (lx2max % 2 == 0)
            lx2max++;
        }
        if (ndim == 3)
        {  // 3D
          lxmax = nrbx3 * (1LL << ref_lev);
          for (lx3min = 0; lx3min < lxmax; lx3min++)
          {
            Real rx = ComputeMeshGeneratorX(
              lx3min + 1, lxmax, use_uniform_meshgen_fn_[X3DIR]);
            if (MeshGenerator_[X3DIR](rx, mesh_size) > ref_size.x3min)
              break;
          }
          for (lx3max = lx3min; lx3max < lxmax; lx3max++)
          {
            Real rx = ComputeMeshGeneratorX(
              lx3max + 1, lxmax, use_uniform_meshgen_fn_[X3DIR]);
            if (MeshGenerator_[X3DIR](rx, mesh_size) >= ref_size.x3max)
              break;
          }
          if (lx3min % 2 == 1)
            lx3min--;
          if (lx3max % 2 == 0)
            lx3max++;
        }
        // create the finest level
        if (ndim == 1)
        {
          for (std::int64_t i = lx1min; i < lx1max; i += 2)
          {
            LogicalLocation nloc;
            nloc.level = lrlev, nloc.lx1 = i, nloc.lx2 = 0, nloc.lx3 = 0;
            int nnew;
            tree.AddMeshBlock(nloc, nnew);
          }
        }
        if (ndim == 2)
        {
          for (std::int64_t j = lx2min; j < lx2max; j += 2)
          {
            for (std::int64_t i = lx1min; i < lx1max; i += 2)
            {
              LogicalLocation nloc;
              nloc.level = lrlev, nloc.lx1 = i, nloc.lx2 = j, nloc.lx3 = 0;
              int nnew;
              tree.AddMeshBlock(nloc, nnew);
            }
          }
        }
        if (ndim == 3)
        {
          for (std::int64_t k = lx3min; k < lx3max; k += 2)
          {
            for (std::int64_t j = lx2min; j < lx2max; j += 2)
            {
              for (std::int64_t i = lx1min; i < lx1max; i += 2)
              {
                LogicalLocation nloc;
                nloc.level = lrlev, nloc.lx1 = i, nloc.lx2 = j, nloc.lx3 = k;
                int nnew;
                tree.AddMeshBlock(nloc, nnew);
              }
            }
          }
        }
      }
      pib = pib->pnext;
    }
  }

  // initial mesh hierarchy construction is completed here
  tree.CountMeshBlock(nbtotal);
  loclist = new LogicalLocation[nbtotal];
  tree.GetMeshBlockList(loclist, nullptr, nbtotal);

#ifdef MPI_PARALLEL
  // check if there are sufficient blocks
  if (nbtotal < Globals::nranks)
  {
    if (mesh_test == 0)
    {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "Too few mesh blocks: nbtotal (" << nbtotal << ") < nranks ("
          << Globals::nranks << ")" << std::endl;
      ATHENA_ERROR(msg);
    }
    else
    {  // test
      std::cout << "### Warning in Mesh constructor" << std::endl
                << "Too few mesh blocks: nbtotal (" << nbtotal
                << ") < nranks (" << Globals::nranks << ")" << std::endl;
    }
  }
#endif

  ranklist = new int[nbtotal];
  nslist   = new int[Globals::nranks];
  nblist   = new int[Globals::nranks];
  costlist = new double[nbtotal];
  if (adaptive)
  {  // allocate arrays for AMR
    nref    = new int[Globals::nranks];
    nderef  = new int[Globals::nranks];
    rdisp   = new int[Globals::nranks];
    ddisp   = new int[Globals::nranks];
    bnref   = new int[Globals::nranks];
    bnderef = new int[Globals::nranks];
    brdisp  = new int[Globals::nranks];
    bddisp  = new int[Globals::nranks];
  }

  // initialize cost array with the simplest estimate; all the blocks are equal
  for (int i = 0; i < nbtotal; i++)
    costlist[i] = 1.0;

  CalculateLoadBalance(costlist, ranklist, nslist, nblist, nbtotal);

  // Output some diagnostic information to terminal

  // Output MeshBlock list and quit (mesh test only); do not create meshes
  if (mesh_test > 0)
  {
    if (Globals::my_rank == 0)
      OutputMeshStructure(ndim);
    return;
  }

  // create MeshBlock list for this process
  int nbs = nslist[Globals::my_rank];
  int nbe = nbs + nblist[Globals::my_rank] - 1;
  // create MeshBlock list for this process
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
  tree.ComputeNeighborLevelFlags();
#endif
  for (int i = nbs; i <= nbe; i++)
  {
    SetBlockSizeAndBoundaries(loclist[i], block_size, block_bcs);
    // create a block and add into the link list
    if (i == nbs)
    {
      pblock = new MeshBlock(
        i, i - nbs, loclist[i], block_size, block_bcs, this, pin);
      pfirst = pblock;
    }
    else
    {
      pblock->next = new MeshBlock(
        i, i - nbs, loclist[i], block_size, block_bcs, this, pin);
      pblock->next->prev = pblock;
      pblock             = pblock->next;
    }
    pblock->SearchAndSetNeighbors(tree, ranklist, nslist);
  }
  pblock = pfirst;

  RebuildBlockByGid();
  ResetLoadBalanceVariables();

#if FLUID_ENABLED
  presc = new gra::hydro::rescaling::Rescaling(this, pin);
#endif

  // storage for Mesh info struct
  M_info.x_min.NewAthenaArray(ndim);
  M_info.x_max.NewAthenaArray(ndim);
  M_info.dx_min.NewAthenaArray(ndim);
  M_info.dx_max.NewAthenaArray(ndim);
  M_info.max_level = 0;

  // Surface init needs to come after MeshBlocks have been initialized
  gra::mesh::surfaces::InitSurfaces(this, pin);
}

//----------------------------------------------------------------------------------------
// Mesh constructor for restarts. Load the restart file

Mesh::Mesh(ParameterInput* pin, IOWrapper& resfile, int mesh_test)
    :  // public members:
       // aggregate initialization of RegionSize struct:
       // (will be overwritten by memcpy from restart file, in this case)
      mesh_size{ pin->GetReal("mesh", "x1min"),
                 pin->GetReal("mesh", "x2min"),
                 pin->GetReal("mesh", "x3min"),
                 pin->GetReal("mesh", "x1max"),
                 pin->GetReal("mesh", "x2max"),
                 pin->GetReal("mesh", "x3max"),
                 pin->GetOrAddReal("mesh", "x1rat", 1.0),
                 pin->GetOrAddReal("mesh", "x2rat", 1.0),
                 pin->GetOrAddReal("mesh", "x3rat", 1.0),
                 pin->GetInteger("mesh", "nx1"),
                 pin->GetInteger("mesh", "nx2"),
                 pin->GetInteger("mesh", "nx3") },
      mesh_bcs{ GetBoundaryFlag(pin->GetOrAddString("mesh", "ix1_bc", "none")),
                GetBoundaryFlag(pin->GetOrAddString("mesh", "ox1_bc", "none")),
                GetBoundaryFlag(pin->GetOrAddString("mesh", "ix2_bc", "none")),
                GetBoundaryFlag(pin->GetOrAddString("mesh", "ox2_bc", "none")),
                GetBoundaryFlag(pin->GetOrAddString("mesh", "ix3_bc", "none")),
                GetBoundaryFlag(
                  pin->GetOrAddString("mesh", "ox3_bc", "none")) },
      f2(mesh_size.nx2 > 1 ? true : false),
      f3(mesh_size.nx3 > 1 ? true : false),
      ndim(f3 ? 3 : (f2 ? 2 : 1)),
      adaptive(pin->GetOrAddString("mesh", "refinement", "none") == "adaptive"
                 ? true
                 : false),
      multilevel(
        (adaptive ||
         pin->GetOrAddString("mesh", "refinement", "none") == "static")
          ? true
          : false),
      start_time(pin->GetOrAddReal("time", "start_time", 0.0)),
      time(start_time),
      tlim(pin->GetReal("time", "tlim")),
      dt(std::numeric_limits<Real>::max()),
      dt_hyperbolic(dt),
      dt_parabolic(dt),
      dt_user(dt),
      cfl_number(pin->GetReal("time", "cfl_number")),
      nlim(pin->GetOrAddInteger("time", "nlim", -1)),
      ncycle(),
      ncycle_out(pin->GetOrAddInteger("time", "ncycle_out", 1)),
      dt_diagnostics(pin->GetOrAddInteger("time", "dt_diagnostics", -1)),
      nbnew(),
      nbdel(),
      step_since_lb(),
      // private members:
      num_mesh_threads_(pin->GetOrAddInteger("mesh", "num_threads", 1)),
      tree(this),
      use_uniform_meshgen_fn_{ true, true, true },
      nreal_user_mesh_data_(),
      nint_user_mesh_data_(),
      lb_flag_(true),
      lb_automatic_(),
      lb_manual_(),
      MeshGenerator_{ UniformMeshGeneratorX1,
                      UniformMeshGeneratorX2,
                      UniformMeshGeneratorX3 },
      BoundaryFunction_{
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
      },
      AMRFlag_{},
      UserTimeStep_{},
      UserMainLoopBreak_{}
{
  std::stringstream msg;
  RegionSize block_size;
  BoundaryFlag block_bcs[6];
  MeshBlock* pfirst{};
  IOWrapperSizeT datasize, listsize, headeroffset;

  // mesh test
  if (mesh_test > 0)
    Globals::nranks = mesh_test;

  // check the number of OpenMP threads for mesh
  if (num_mesh_threads_ < 1)
  {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Number of OpenMP threads must be >= 1, but num_threads="
        << num_mesh_threads_ << std::endl;
    ATHENA_ERROR(msg);
  }

  // get the end of the header
  headeroffset = resfile.GetPosition();
  // read the restart file
  // the file is already open and the pointer is set to after <par_end>
  IOWrapperSizeT headersize = sizeof(int) * 3 + sizeof(Real) * 2 +
                              sizeof(RegionSize) + sizeof(IOWrapperSizeT);
  char* headerdata = new char[headersize];
  if (Globals::my_rank == 0)
  {  // the master process reads the header data
    if (resfile.Read(headerdata, 1, headersize) != headersize)
    {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The restart file is broken." << std::endl;
      ATHENA_ERROR(msg);
    }
  }
#ifdef MPI_PARALLEL
  // then broadcast the header data
  MPI_Bcast(headerdata, headersize, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
  IOWrapperSizeT hdos = 0;
  std::memcpy(&nbtotal, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  std::memcpy(&root_level, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  current_level = root_level;
  std::memcpy(&mesh_size, &(headerdata[hdos]), sizeof(RegionSize));
  hdos += sizeof(RegionSize);
  std::memcpy(&time, &(headerdata[hdos]), sizeof(Real));
  hdos += sizeof(Real);
  std::memcpy(&dt, &(headerdata[hdos]), sizeof(Real));
  hdos += sizeof(Real);
  std::memcpy(&ncycle, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  std::memcpy(&datasize, &(headerdata[hdos]), sizeof(IOWrapperSizeT));
  hdos += sizeof(IOWrapperSizeT);  // (this updated value is never used)

  delete[] headerdata;

  // initialize
  loclist  = new LogicalLocation[nbtotal];
  costlist = new double[nbtotal];
  ranklist = new int[nbtotal];
  nslist   = new int[Globals::nranks];
  nblist   = new int[Globals::nranks];

  block_size.nx1 = pin->GetOrAddInteger("meshblock", "nx1", mesh_size.nx1);
  block_size.nx2 = pin->GetOrAddInteger("meshblock", "nx2", mesh_size.nx2);
  block_size.nx3 = pin->GetOrAddInteger("meshblock", "nx3", mesh_size.nx3);

  // Allocate per-thread scratch caches
  {
    int nc1 = block_size.nx1 + 2 * NGHOST;
    int nc2 = (f2) ? block_size.nx2 + 2 * NGHOST : 1;
    int nc3 = (f3) ? block_size.nx3 + 2 * NGHOST : 1;
    thread_caches_.resize(num_mesh_threads_);
    for (auto& tc : thread_caches_)
    {
      tc.AllocateFCGeom(nc1, nc2, nc3);
      if (FLUID_ENABLED &&
          pin->GetOrAddBoolean("time", "xorder_use_fb", false))
        tc.AllocateLOFlux(nc1, nc2, nc3, f2, f3);
      if (M1_ENABLED)
      {
        bool m1_fb_E = pin->GetOrAddBoolean("M1", "flux_lo_fallback_E", false);
        bool m1_fb_nG =
          pin->GetOrAddBoolean("M1", "flux_lo_fallback_nG", false);
        if (m1_fb_E || m1_fb_nG)
        {
          int ngrps = pin->GetOrAddInteger("M1", "ngroups", 1);
          int nspcs = pin->GetOrAddInteger("M1", "nspecies", 1);
          int ngs   = M1::M1::ixn_Lab::N * ngrps * nspcs;
          tc.AllocateM1LOFlux(nc1, nc2, nc3, ngs);
        }
      }
    }
  }

  // calculate the number of the blocks
  nrbx1 = mesh_size.nx1 / block_size.nx1;
  nrbx2 = mesh_size.nx2 / block_size.nx2;
  nrbx3 = mesh_size.nx3 / block_size.nx3;

  // initialize user-enrollable functions
  if (mesh_size.x1rat != 1.0)
  {
    use_uniform_meshgen_fn_[X1DIR] = false;
    MeshGenerator_[X1DIR]          = DefaultMeshGeneratorX1;
  }
  if (mesh_size.x2rat != 1.0)
  {
    use_uniform_meshgen_fn_[X2DIR] = false;
    MeshGenerator_[X2DIR]          = DefaultMeshGeneratorX2;
  }
  if (mesh_size.x3rat != 1.0)
  {
    use_uniform_meshgen_fn_[X3DIR] = false;
    MeshGenerator_[X3DIR]          = DefaultMeshGeneratorX3;
  }

  // Load balancing flag and parameters
#ifdef MPI_PARALLEL
  if (pin->GetOrAddString("loadbalancing", "balancer", "default") ==
      "automatic")
    lb_automatic_ = true;
  else if (pin->GetOrAddString("loadbalancing", "balancer", "default") ==
           "manual")
    lb_manual_ = true;
  lb_tolerance_ = pin->GetOrAddReal("loadbalancing", "tolerance", 0.5);
  lb_interval_  = pin->GetOrAddReal("loadbalancing", "interval", 10);
#endif

  // SMR / AMR
  if (adaptive)
  {
    max_level = pin->GetOrAddInteger("mesh", "numlevel", 1) + root_level - 1;
    if (max_level > 63)
    {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The number of the refinement level must be smaller than "
          << 63 - root_level + 1 << "." << std::endl;
      ATHENA_ERROR(msg);
    }
  }
  else
  {
    max_level = 63;
  }

  if (Z4C_ENABLED)
  {
    int nrad = pin->GetOrAddInteger("psi4_extraction", "num_radii", 0);
    if (nrad > 0)
    {
      pwave_extr.reserve(nrad);
      for (int n = 0; n < nrad; ++n)
      {
        pwave_extr.push_back(new WaveExtract(this, pin, n));
      }
    }
    int nrad_rwz = pin->GetOrAddInteger("rwz_extraction", "num_radii", 0);
    if (nrad_rwz > 0)
    {
      pwave_extr_rwz.reserve(nrad_rwz);
      for (int n = 0; n < nrad_rwz; ++n)
      {
        pwave_extr_rwz.push_back(new WaveExtractRWZ(this, pin, n));
      }
    }

#if CCE_ENABLED
    // CCE
    int ncce = pin->GetOrAddInteger("cce", "num_radii", 0);
    pcce.reserve(10 * ncce);  // 10 different components for each radius
    for (int n = 0; n < ncce; ++n)
    {
      pcce.push_back(new CCE(this, pin, "gxx", n));
      pcce.push_back(new CCE(this, pin, "gxy", n));
      pcce.push_back(new CCE(this, pin, "gxz", n));
      pcce.push_back(new CCE(this, pin, "gyy", n));
      pcce.push_back(new CCE(this, pin, "gyz", n));
      pcce.push_back(new CCE(this, pin, "gzz", n));
      pcce.push_back(new CCE(this, pin, "betax", n));
      pcce.push_back(new CCE(this, pin, "betay", n));
      pcce.push_back(new CCE(this, pin, "betaz", n));
      pcce.push_back(new CCE(this, pin, "alp", n));
    }
#endif

#ifdef EJECTA_ENABLED
    const int nejecta = pin->GetOrAddInteger("ejecta", "num_rad", 0);
    pej_extract.reserve(nejecta);
    for (int n = 0; n < nejecta; ++n)
    {
      pej_extract.push_back(new Ejecta(this, pin, n));
    }

    /*
    const int nejecta = pin->GetOrAddRealArray("ejecta", "R", -1, 0).GetSize();
    pej_extract.reserve(nejecta);
    for (int n=0; n<nejecta; ++n)
    {
      pej_extract.push_back(new Ejecta(this, pin, n, nejecta));
    }
    */
#endif

    int npunct = pin->GetOrAddInteger("z4c", "npunct", 0);
    if (npunct > 0)
    {
      pz4c_tracker.reserve(npunct);
      for (int n = 0; n < npunct; ++n)
      {
        pz4c_tracker.push_back(new PunctureTracker(this, pin, n));
      }
    }
  }

  // Last entry says if it is restart run or not
  ptracker_extrema = new ExtremaTracker(this, pin, 1);

  if (Z4C_ENABLED)
  {
    // BD: By default do not add any horizon searching
    int nhorizon = pin->GetOrAddInteger("ahf", "num_horizons", 0);
    pah_finder.reserve(nhorizon);
    for (int n = 0; n < nhorizon; ++n)
    {
      pah_finder.push_back(new AHF(this, pin, n));
    }
  }

  InitUserMeshData(pin);
  // read user Mesh data
  IOWrapperSizeT udsize = 0;
  for (int n = 0; n < nint_user_mesh_data_; n++)
    udsize += iuser_mesh_data[n].GetSizeInBytes();
  for (int n = 0; n < nreal_user_mesh_data_; n++)
    udsize += ruser_mesh_data[n].GetSizeInBytes();

  udsize += 2 * NDIM * sizeof(Real) * pz4c_tracker.size();

  if (!ptracker_extrema->use_new_style)
  {
    // c_x1, c_x2, c_x3
    udsize += ptracker_extrema->c_x1.GetSizeInBytes();
    udsize += ptracker_extrema->c_x2.GetSizeInBytes();
    udsize += ptracker_extrema->c_x3.GetSizeInBytes();
  }

  if (udsize != 0)
  {
    char* userdata = new char[udsize];
    if (Globals::my_rank == 0)
    {  // only the master process reads the ID list
      if (resfile.Read(userdata, 1, udsize) != udsize)
      {
        msg << "### FATAL ERROR in Mesh constructor" << std::endl
            << "The restart file is broken." << std::endl;
        ATHENA_ERROR(msg);
      }
    }
#ifdef MPI_PARALLEL
    // then broadcast the ID list
    MPI_Bcast(userdata, udsize, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif

    IOWrapperSizeT udoffset = 0;
    for (int n = 0; n < nint_user_mesh_data_; n++)
    {
      std::memcpy(iuser_mesh_data[n].data(),
                  &(userdata[udoffset]),
                  iuser_mesh_data[n].GetSizeInBytes());
      udoffset += iuser_mesh_data[n].GetSizeInBytes();
    }
    for (int n = 0; n < nreal_user_mesh_data_; n++)
    {
      std::memcpy(ruser_mesh_data[n].data(),
                  &(userdata[udoffset]),
                  ruser_mesh_data[n].GetSizeInBytes());
      udoffset += ruser_mesh_data[n].GetSizeInBytes();
    }
    for (auto ptracker : pz4c_tracker)
    {
      std::memcpy(ptracker->pos, &userdata[udoffset], NDIM * sizeof(Real));
      udoffset += NDIM * sizeof(Real);
      std::memcpy(ptracker->betap, &userdata[udoffset], NDIM * sizeof(Real));
      udoffset += NDIM * sizeof(Real);
    }

    if (!ptracker_extrema->use_new_style)
    {
      std::memcpy(ptracker_extrema->c_x1.data(),
                  &(userdata[udoffset]),
                  ptracker_extrema->c_x1.GetSizeInBytes());
      udoffset += ptracker_extrema->c_x1.GetSizeInBytes();

      std::memcpy(ptracker_extrema->c_x2.data(),
                  &(userdata[udoffset]),
                  ptracker_extrema->c_x2.GetSizeInBytes());
      udoffset += ptracker_extrema->c_x2.GetSizeInBytes();

      std::memcpy(ptracker_extrema->c_x3.data(),
                  &(userdata[udoffset]),
                  ptracker_extrema->c_x3.GetSizeInBytes());
      udoffset += ptracker_extrema->c_x3.GetSizeInBytes();
    }

    delete[] userdata;
  }

  // read the ID list
  listsize = sizeof(LogicalLocation) + sizeof(double);
  // allocate the idlist buffer
  char* idlist = new char[listsize * nbtotal];
  if (Globals::my_rank == 0)
  {  // only the master process reads the ID list
    if (resfile.Read(idlist, listsize, nbtotal) !=
        static_cast<unsigned int>(nbtotal))
    {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The restart file is broken." << std::endl;
      ATHENA_ERROR(msg);
    }
  }
#ifdef MPI_PARALLEL
  // then broadcast the ID list
  MPI_Bcast(idlist, listsize * nbtotal, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif

  int os = 0;
  for (int i = 0; i < nbtotal; i++)
  {
    std::memcpy(&(loclist[i]), &(idlist[os]), sizeof(LogicalLocation));
    os += sizeof(LogicalLocation);
    std::memcpy(&(costlist[i]), &(idlist[os]), sizeof(double));
    os += sizeof(double);
    if (loclist[i].level > current_level)
      current_level = loclist[i].level;
  }
  delete[] idlist;

  // calculate the header offset and seek
  headeroffset += headersize + udsize + listsize * nbtotal;
  if (Globals::my_rank != 0)
    resfile.Seek(headeroffset);

  // rebuild the Block Tree
  tree.CreateRootGrid();
  for (int i = 0; i < nbtotal; i++)
    tree.AddMeshBlockWithoutRefine(loclist[i]);
  int nnb;
  // check the tree structure, and assign GID
  tree.GetMeshBlockList(loclist, nullptr, nnb);
  if (nnb != nbtotal)
  {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Tree reconstruction failed. The total numbers of the blocks do "
           "not match. ("
        << nbtotal << " != " << nnb << ")" << std::endl;
    ATHENA_ERROR(msg);
  }

#ifdef MPI_PARALLEL
  if (nbtotal < Globals::nranks)
  {
    if (mesh_test == 0)
    {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "Too few mesh blocks: nbtotal (" << nbtotal << ") < nranks ("
          << Globals::nranks << ")" << std::endl;
      ATHENA_ERROR(msg);
    }
    else
    {  // test
      std::cout << "### Warning in Mesh constructor" << std::endl
                << "Too few mesh blocks: nbtotal (" << nbtotal
                << ") < nranks (" << Globals::nranks << ")" << std::endl;
      return;
    }
  }
#endif

  if (adaptive)
  {  // allocate arrays for AMR
    nref    = new int[Globals::nranks];
    nderef  = new int[Globals::nranks];
    rdisp   = new int[Globals::nranks];
    ddisp   = new int[Globals::nranks];
    bnref   = new int[Globals::nranks];
    bnderef = new int[Globals::nranks];
    brdisp  = new int[Globals::nranks];
    bddisp  = new int[Globals::nranks];
  }

  CalculateLoadBalance(costlist, ranklist, nslist, nblist, nbtotal);

  // Output MeshBlock list and quit (mesh test only); do not create meshes
  if (mesh_test > 0)
  {
    if (Globals::my_rank == 0)
      OutputMeshStructure(ndim);
    return;
  }

  // allocate data buffer
  int nb  = nblist[Globals::my_rank];
  int nbs = nslist[Globals::my_rank];
  int nbe = nbs + nb - 1;

#if !defined(DBG_RST_WRITE_PER_MB)
  char* mbdata = new char[datasize * nb];
  // load MeshBlocks (parallel)
  if (resfile.Read_at_all(
        mbdata, datasize, nb, headeroffset + nbs * datasize) !=
      static_cast<unsigned int>(nb))
  {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "The restart file is broken or input parameters are inconsistent."
        << std::endl;
    ATHENA_ERROR(msg);
  }
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
  tree.ComputeNeighborLevelFlags();
#endif
  for (int i = nbs; i <= nbe; i++)
  {
    // Match fixed-width integer precision of IOWrapperSizeT datasize
    std::uint64_t buff_os = datasize * (i - nbs);
    SetBlockSizeAndBoundaries(loclist[i], block_size, block_bcs);
    // create a block and add into the link list
    if (i == nbs)
    {
      pblock = new MeshBlock(i,
                             i - nbs,
                             this,
                             pin,
                             loclist[i],
                             block_size,
                             block_bcs,
                             costlist[i],
                             mbdata + buff_os);
      pfirst = pblock;
    }
    else
    {
      pblock->next       = new MeshBlock(i,
                                   i - nbs,
                                   this,
                                   pin,
                                   loclist[i],
                                   block_size,
                                   block_bcs,
                                   costlist[i],
                                   mbdata + buff_os);
      pblock->next->prev = pblock;
      pblock             = pblock->next;
    }

    // BD: needed for cons<->prim after restart
    // if(Z4C_ENABLED && FLUID_ENABLED)
    // {
    //   pblock->pz4c->Z4cToADM(pblock->pz4c->storage.u,
    //   pblock->pz4c->storage.adm);
    // }

    pblock->SearchAndSetNeighbors(tree, ranklist, nslist);
  }
#else
  int nbmin = nblist[0];
  for (int n = 1; n < Globals::nranks; ++n)
  {
    if (nbmin > nblist[n])
      nbmin = nblist[n];
  }

  char* mbdata = new char[datasize];

#if defined(DBG_NO_REF_NN_SAME_LEVEL)
  tree.ComputeNeighborLevelFlags();
#endif
  for (int i = nbs; i <= nbe; i++)
  {
    if (i - nbs < nbmin)
    {
      // load MeshBlock (parallel)
      if (resfile.Read_at_all(
            mbdata, datasize, 1, headeroffset + i * datasize) != 1)
      {
        msg
          << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The restart file is broken or input parameters are inconsistent."
          << std::endl;
        ATHENA_ERROR(msg);
      }
    }
    else
    {
      // load MeshBlock (serial)
      if (resfile.Read_at(mbdata, datasize, 1, headeroffset + i * datasize) !=
          1)
      {
        msg
          << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The restart file is broken or input parameters are inconsistent."
          << std::endl;
        ATHENA_ERROR(msg);
      }
    }

    SetBlockSizeAndBoundaries(loclist[i], block_size, block_bcs);
    // create a block and add into the link list
    if (i == nbs)
    {
      pblock = new MeshBlock(i,
                             i - nbs,
                             this,
                             pin,
                             loclist[i],
                             block_size,
                             block_bcs,
                             costlist[i],
                             mbdata);
      pfirst = pblock;
    }
    else
    {
      pblock->next       = new MeshBlock(i,
                                   i - nbs,
                                   this,
                                   pin,
                                   loclist[i],
                                   block_size,
                                   block_bcs,
                                   costlist[i],
                                   mbdata);
      pblock->next->prev = pblock;
      pblock             = pblock->next;
    }

    // BD: needed for cons<->prim after restart
    // if(Z4C_ENABLED && FLUID_ENABLED)
    // {
    //   pblock->pz4c->Z4cToADM(pblock->pz4c->storage.u,
    //   pblock->pz4c->storage.adm);
    // }

    pblock->SearchAndSetNeighbors(tree, ranklist, nslist);
  }
#endif  // DBG_RST_WRITE_PER_MB

  pblock = pfirst;
  RebuildBlockByGid();
  delete[] mbdata;
  // check consistency
  if (datasize != pblock->GetBlockSizeInBytes())
  {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "The restart file is broken or input parameters are inconsistent."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  ResetLoadBalanceVariables();

#if FLUID_ENABLED
  presc = new gra::hydro::rescaling::Rescaling(this, pin);
#endif

  // storage for Mesh info struct
  M_info.x_min.NewAthenaArray(ndim);
  M_info.x_max.NewAthenaArray(ndim);
  M_info.dx_min.NewAthenaArray(ndim);
  M_info.dx_max.NewAthenaArray(ndim);
  M_info.max_level = 0;

  // Surface init needs to come after MeshBlocks have been initialized
  gra::mesh::surfaces::InitSurfaces(this, pin);
}

//----------------------------------------------------------------------------------------
// destructor

Mesh::~Mesh()
{
  if (pblock != nullptr)
  {
    while (pblock->prev != nullptr)  // should not be true
      delete pblock->prev;
    while (pblock->next != nullptr)
      delete pblock->next;
    delete pblock;
  }
  delete[] nslist;
  delete[] nblist;
  delete[] ranklist;
  delete[] costlist;
  delete[] loclist;

  if (Z4C_ENABLED)
  {
    for (auto pwextr : pwave_extr)
    {
      delete pwextr;
    }
    pwave_extr.resize(0);

    for (auto pwextr_rwz : pwave_extr_rwz)
    {
      delete pwextr_rwz;
    }
    pwave_extr_rwz.resize(0);

#if CCE_ENABLED
    for (auto cce : pcce)
    {
      delete cce;
    }
    pcce.resize(0);
#endif

    for (auto pah_f : pah_finder)
    {
      delete pah_f;
    }
    pah_finder.resize(0);

#ifdef EJECTA_ENABLED
    for (auto pej : pej_extract)
    {
      delete pej;
    }
    pej_extract.resize(0);
#endif

    for (auto tracker : pz4c_tracker)
    {
      delete tracker;
    }
    pz4c_tracker.resize(0);
  }

  delete ptracker_extrema;

  for (auto surf : psurfs)
  {
    delete surf;
  }
  psurfs.resize(0);

  if (adaptive)
  {  // deallocate arrays for AMR
    delete[] nref;
    delete[] nderef;
    delete[] rdisp;
    delete[] ddisp;
    delete[] bnref;
    delete[] bnderef;
    delete[] brdisp;
    delete[] bddisp;
  }
  // delete user Mesh data
  if (nreal_user_mesh_data_ > 0)
    delete[] ruser_mesh_data;
  if (nint_user_mesh_data_ > 0)
    delete[] iuser_mesh_data;

#if FLUID_ENABLED
  delete presc;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::OutputMeshStructure(int ndim)
//  \brief print the mesh structure information

void Mesh::OutputMeshStructure(int ndim)
{
  RegionSize block_size;
  BoundaryFlag block_bcs[6];
  FILE* fp = nullptr;

  // open 'mesh_structure.dat' file
  if (f2)
  {
    if ((fp = std::fopen("mesh_structure.dat", "wb")) == nullptr)
    {
      std::cout << "### ERROR in function Mesh::OutputMeshStructure"
                << std::endl
                << "Cannot open mesh_structure.dat" << std::endl;
      return;
    }
  }

  // Write overall Mesh structure to stdout and file
  std::cout << std::endl;
  std::cout << "Root grid = " << nrbx1 << " x " << nrbx2 << " x " << nrbx3
            << " MeshBlocks" << std::endl;
  std::cout << "Total number of MeshBlocks = " << nbtotal << std::endl;
  std::cout << "Number of physical refinement levels = "
            << (current_level - root_level) << std::endl;
  std::cout << "Number of logical  refinement levels = " << current_level
            << std::endl;

  // compute/output number of blocks per level, and cost per level
  const int nb_plevel  = max_level - root_level + 1;
  int* nb_per_plevel   = new int[nb_plevel]();
  int* cost_per_plevel = new int[nb_plevel]();
  for (int i = 0; i < nbtotal; i++)
  {
    nb_per_plevel[(loclist[i].level - root_level)]++;
    cost_per_plevel[(loclist[i].level - root_level)] += costlist[i];
  }
  for (int i = root_level; i <= max_level; i++)
  {
    if (nb_per_plevel[i - root_level] != 0)
    {
      std::cout << "  Physical level = " << i - root_level
                << " (logical level = " << i
                << "): " << nb_per_plevel[i - root_level]
                << " MeshBlocks, cost = " << cost_per_plevel[i - root_level]
                << std::endl;
    }
  }

  // compute/output number of blocks per rank, and cost per rank
  std::cout << "Number of parallel ranks = " << Globals::nranks << std::endl;
  int* nb_per_rank   = new int[Globals::nranks];
  int* cost_per_rank = new int[Globals::nranks];
  for (int i = 0; i < Globals::nranks; ++i)
  {
    nb_per_rank[i]   = 0;
    cost_per_rank[i] = 0;
  }
  for (int i = 0; i < nbtotal; i++)
  {
    nb_per_rank[ranklist[i]]++;
    cost_per_rank[ranklist[i]] += costlist[i];
  }
  for (int i = 0; i < Globals::nranks; ++i)
  {
    std::cout << "  Rank = " << i << ": " << nb_per_rank[i]
              << " MeshBlocks, cost = " << cost_per_rank[i] << std::endl;
  }

  // output relative size/locations of meshblock to file, for plotting
  double real_max = std::numeric_limits<double>::max();
  double mincost = real_max, maxcost = 0.0, totalcost = 0.0;
  for (int i = root_level; i <= max_level; i++)
  {
    for (int j = 0; j < nbtotal; j++)
    {
      if (loclist[j].level == i)
      {
        SetBlockSizeAndBoundaries(loclist[j], block_size, block_bcs);
        std::int64_t& lx1 = loclist[j].lx1;
        std::int64_t& lx2 = loclist[j].lx2;
        std::int64_t& lx3 = loclist[j].lx3;
        int& ll           = loclist[j].level;
        mincost           = std::min(mincost, costlist[j]);
        maxcost           = std::max(maxcost, costlist[j]);
        totalcost += costlist[j];
        std::fprintf(fp,
                     "#MeshBlock %d on rank=%d with cost=%g\n",
                     j,
                     ranklist[j],
                     costlist[j]);
        std::fprintf(fp,
                     "#  Logical level %d, location = (%" PRId64 " %" PRId64
                     " %" PRId64 ")\n",
                     ll,
                     lx1,
                     lx2,
                     lx3);
        if (ndim == 2)
        {
          std::fprintf(fp, "%g %g\n", block_size.x1min, block_size.x2min);
          std::fprintf(fp, "%g %g\n", block_size.x1max, block_size.x2min);
          std::fprintf(fp, "%g %g\n", block_size.x1max, block_size.x2max);
          std::fprintf(fp, "%g %g\n", block_size.x1min, block_size.x2max);
          std::fprintf(fp, "%g %g\n", block_size.x1min, block_size.x2min);
          std::fprintf(fp, "\n\n");
        }
        if (ndim == 3)
        {
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1min,
                       block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1max,
                       block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1max,
                       block_size.x2max,
                       block_size.x3min);
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1min,
                       block_size.x2max,
                       block_size.x3min);
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1min,
                       block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1min,
                       block_size.x2min,
                       block_size.x3max);
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1max,
                       block_size.x2min,
                       block_size.x3max);
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1max,
                       block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1max,
                       block_size.x2min,
                       block_size.x3max);
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1max,
                       block_size.x2max,
                       block_size.x3max);
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1max,
                       block_size.x2max,
                       block_size.x3min);
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1max,
                       block_size.x2max,
                       block_size.x3max);
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1min,
                       block_size.x2max,
                       block_size.x3max);
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1min,
                       block_size.x2max,
                       block_size.x3min);
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1min,
                       block_size.x2max,
                       block_size.x3max);
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1min,
                       block_size.x2min,
                       block_size.x3max);
          std::fprintf(fp,
                       "%g %g %g\n",
                       block_size.x1min,
                       block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp, "\n\n");
        }
      }
    }
  }

  // close file, final outputs
  if (f2)
    std::fclose(fp);
  std::cout << "Load Balancing:" << std::endl;
  std::cout << "  Minimum cost = " << mincost << ", Maximum cost = " << maxcost
            << ", Average cost = " << totalcost / nbtotal << std::endl
            << std::endl;
  std::cout << "See the 'mesh_structure.dat' file for a complete list"
            << " of MeshBlocks." << std::endl;
  std::cout << "Use 'python ../vis/python/plot_mesh.py' or gnuplot"
            << " to visualize mesh structure." << std::endl
            << std::endl;

  delete[] nb_per_plevel;
  delete[] cost_per_plevel;
  delete[] nb_per_rank;
  delete[] cost_per_rank;

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::NewTimeStep()
// \brief function that loops over all MeshBlocks and find new timestep
//        this assumes that phydro->NewBlockTimeStep is already called

void Mesh::NewTimeStep(bool limit_dt_growth)
{
  const auto& pmb_array = GetMeshBlocksCached();
  const int nmb         = pmb_array.size();

  MeshBlock* pmb0 = pmb_array[0];

  if (limit_dt_growth)
  {
    // prevent timestep from growing too fast in between 2x cycles (even if
    // every MeshBlock has new_block_dt > 2.0*dt_old)
    dt = static_cast<Real>(2.0) * dt;
    // consider first MeshBlock on this MPI rank's linked list of blocks:
    dt = std::min(dt, pmb0->new_block_dt_);
  }
  else
  {
    dt = pmb0->new_block_dt_;
  }

  dt_hyperbolic = pmb0->new_block_dt_hyperbolic_;
  dt_parabolic  = pmb0->new_block_dt_parabolic_;
  dt_user       = pmb0->new_block_dt_user_;

  for (int i = 1; i < nmb; ++i)
  {
    MeshBlock* pmb = pmb_array[i];
    dt             = std::min(dt, pmb->new_block_dt_);
    dt_hyperbolic  = std::min(dt_hyperbolic, pmb->new_block_dt_hyperbolic_);
    dt_parabolic   = std::min(dt_parabolic, pmb->new_block_dt_parabolic_);
    dt_user        = std::min(dt_user, pmb->new_block_dt_user_);
  }

#ifdef MPI_PARALLEL
  // pack array, MPI allreduce over array, then unpack into Mesh variables
  Real dt_array[4] = { dt, dt_hyperbolic, dt_parabolic, dt_user };
  MPI_Allreduce(
    MPI_IN_PLACE, dt_array, 4, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  dt            = dt_array[0];
  dt_hyperbolic = dt_array[1];
  dt_parabolic  = dt_array[2];
  dt_user       = dt_array[3];
#endif

  if (time < tlim &&
      (tlim - time) < dt)  // timestep would take us past desired endpoint
    dt = tlim - time;

  return;
}

// no arg. limit_dt_growth
void Mesh::NewTimeStep()
{
  Mesh::NewTimeStep(true);
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::NewTimeStepBegin(bool limit_dt_growth)
// \brief Non-blocking first half of NewTimeStep.
//
//  Computes the rank-local minimum of per-block timesteps and posts a
//  non-blocking MPI_Iallreduce (MPI_MIN) over all ranks.  The caller should
//  perform unrelated work (e.g. file I/O) and then call NewTimeStepFinish()
//  to complete the reduction and apply the tlim clamp.
//
//  Without MPI the function behaves identically to NewTimeStep() (the
//  reduction is trivially local and Finish is a no-op apart from the clamp).

void Mesh::NewTimeStepBegin(bool limit_dt_growth)
{
  const auto& pmb_array = GetMeshBlocksCached();
  const int nmb         = pmb_array.size();

  MeshBlock* pmb0 = pmb_array[0];

  if (limit_dt_growth)
  {
    dt = static_cast<Real>(2.0) * dt;
    dt = std::min(dt, pmb0->new_block_dt_);
  }
  else
  {
    dt = pmb0->new_block_dt_;
  }

  dt_hyperbolic = pmb0->new_block_dt_hyperbolic_;
  dt_parabolic  = pmb0->new_block_dt_parabolic_;
  dt_user       = pmb0->new_block_dt_user_;

  for (int i = 1; i < nmb; ++i)
  {
    MeshBlock* pmb = pmb_array[i];
    dt             = std::min(dt, pmb->new_block_dt_);
    dt_hyperbolic  = std::min(dt_hyperbolic, pmb->new_block_dt_hyperbolic_);
    dt_parabolic   = std::min(dt_parabolic, pmb->new_block_dt_parabolic_);
    dt_user        = std::min(dt_user, pmb->new_block_dt_user_);
  }

#ifdef MPI_PARALLEL
  // Pack local minima into the staging buffer and post a non-blocking
  // allreduce.  The result is collected in NewTimeStepFinish().
  dt_reduce_buf_[0] = dt;
  dt_reduce_buf_[1] = dt_hyperbolic;
  dt_reduce_buf_[2] = dt_parabolic;
  dt_reduce_buf_[3] = dt_user;

  MPI_Iallreduce(MPI_IN_PLACE,
                 dt_reduce_buf_,
                 4,
                 MPI_ATHENA_REAL,
                 MPI_MIN,
                 MPI_COMM_WORLD,
                 &dt_reduce_req_);
#endif
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::NewTimeStepFinish()
// \brief Non-blocking second half of NewTimeStep.
//
//  Waits for the MPI_Iallreduce posted by NewTimeStepBegin(), unpacks the
//  global minima, and applies the tlim clamp.  Without MPI, only the clamp
//  is applied (the local reduction was already complete in Begin).

void Mesh::NewTimeStepFinish()
{
#ifdef MPI_PARALLEL
  MPI_Status mpi_status;
  {
    int rc = MPI_Wait(&dt_reduce_req_, &mpi_status);
    if (rc != MPI_SUCCESS)
    {
      int err_class = 0;
      MPI_Error_class(rc, &err_class);
      char err_str[MPI_MAX_ERROR_STRING];
      int err_len = 0;
      MPI_Error_string(rc, err_str, &err_len);
      std::printf(
        "[MPI ERROR] MPI_Wait(NewTimeStepFinish) failed on rank %d\n"
        "  error_class=%d error_string=\"%.*s\"\n",
        Globals::my_rank,
        err_class,
        err_len,
        err_str);
      MPI_Abort(MPI_COMM_WORLD, rc);
    }
  }

  dt            = dt_reduce_buf_[0];
  dt_hyperbolic = dt_reduce_buf_[1];
  dt_parabolic  = dt_reduce_buf_[2];
  dt_user       = dt_reduce_buf_[3];
#endif

  if (time < tlim && (tlim - time) < dt)
    dt = tlim - time;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::ResetAllBlockDt()
//  \brief Reset new_block_dt_ on every MeshBlock to the maximum sentinel
//  value.
//
//  Called once per cycle before any physics task list runs, so that each
//  subsystem's NewBlockTimeStep() can min-reduce against the sentinel.

void Mesh::ResetAllBlockDt()
{
  const auto& pmb_array = GetMeshBlocksCached();
  const int nmb         = pmb_array.size();
  for (int i = 0; i < nmb; ++i)
  {
    pmb_array[i]->ResetBlockDt();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserBoundaryFunction(BoundaryFace dir, BValHydro
//! my_bc)
//  \brief Enroll a user-defined boundary function

void Mesh::EnrollUserBoundaryFunction(BoundaryFace dir, BValFunc my_bc)
{
  std::stringstream msg;
  if (dir < 0 || dir > 5)
  {
    msg << "### FATAL ERROR in EnrollBoundaryCondition function" << std::endl
        << "dirName = " << dir << " not valid" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mesh_bcs[dir] != BoundaryFlag::user)
  {
    msg
      << "### FATAL ERROR in EnrollUserBoundaryFunction" << std::endl
      << "The boundary condition flag must be set to the string 'user' in the "
      << " <mesh> block in the input file to use user-enrolled BCs"
      << std::endl;
    ATHENA_ERROR(msg);
  }
  BoundaryFunction_[static_cast<int>(dir)] = my_bc;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserRefinementCondition(AMRFlagFunc amrflag)
//  \brief Enroll a user-defined function for checking refinement criteria

void Mesh::EnrollUserRefinementCondition(AMRFlagFunc amrflag)
{
  if (adaptive)
    AMRFlag_ = amrflag;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserMeshGenerator(CoordinateDirection,MeshGenFunc
//! my_mg)
//  \brief Enroll a user-defined function for Mesh generation

void Mesh::EnrollUserMeshGenerator(CoordinateDirection dir, MeshGenFunc my_mg)
{
  std::stringstream msg;
  if (dir < 0 || dir >= 3)
  {
    msg << "### FATAL ERROR in EnrollUserMeshGenerator function" << std::endl
        << "dirName = " << dir << " not valid" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (dir == X1DIR && mesh_size.x1rat > 0.0)
  {
    msg << "### FATAL ERROR in EnrollUserMeshGenerator function" << std::endl
        << "x1rat = " << mesh_size.x1rat
        << " must be negative for user-defined mesh generator in X1DIR "
        << std::endl;
    ATHENA_ERROR(msg);
  }
  if (dir == X2DIR && mesh_size.x2rat > 0.0)
  {
    msg << "### FATAL ERROR in EnrollUserMeshGenerator function" << std::endl
        << "x2rat = " << mesh_size.x2rat
        << " must be negative for user-defined mesh generator in X2DIR "
        << std::endl;
    ATHENA_ERROR(msg);
  }
  if (dir == X3DIR && mesh_size.x3rat > 0.0)
  {
    msg << "### FATAL ERROR in EnrollUserMeshGenerator function" << std::endl
        << "x3rat = " << mesh_size.x3rat
        << " must be negative for user-defined mesh generator in X3DIR "
        << std::endl;
    ATHENA_ERROR(msg);
  }
  use_uniform_meshgen_fn_[dir] = false;
  MeshGenerator_[dir]          = my_mg;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserTimeStepFunction(TimeStepFunc my_func)
//  \brief Enroll a user-defined time step function

void Mesh::EnrollUserTimeStepFunction(TimeStepFunc my_func)
{
  UserTimeStep_ = my_func;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserMainLoopBreak(MainLoopBreakFunc my_func)
//  \brief Enroll a user-defined function to check for early main-loop exit

void Mesh::EnrollUserMainLoopBreak(MainLoopBreakFunc my_func)
{
  UserMainLoopBreak_ = my_func;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn bool Mesh::CheckUserMainLoopBreak(ParameterInput *pin)
//  \brief Check the enrolled break condition; returns true to break main loop

bool Mesh::CheckUserMainLoopBreak(ParameterInput* pin)
{
  if (UserMainLoopBreak_ != nullptr)
    return UserMainLoopBreak_(this, pin);
  return false;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserHistoryOutput(int i, HistoryOutputFunc my_func,
//                                         const char *name,
//                                         UserHistoryOperation op)
//  \brief Enroll a user-defined history output function and set its name

void Mesh::EnrollUserHistoryOutput(HistoryOutputFunc my_func,
                                   const char* name,
                                   UserHistoryOperation op)
{
  user_history_output_names_.push_back(std::move(name));
  user_history_func_.push_back(std::move(my_func));
  user_history_ops_.push_back(std::move(op));
}

void Mesh::EnrollUserHistoryOutput(
  std::function<Real(MeshBlock*, int)> my_func,
  const char* name,
  UserHistoryOperation op)
{
  user_history_output_names_.push_back(std::move(name));
  user_history_func_.push_back(my_func);
  user_history_ops_.push_back(std::move(op));
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::AllocateRealUserMeshDataField(int n)
//  \brief Allocate Real AthenaArrays for user-defned data in Mesh

void Mesh::AllocateRealUserMeshDataField(int n)
{
  if (nreal_user_mesh_data_ != 0)
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::AllocateRealUserMeshDataField"
        << std::endl
        << "User Mesh data arrays are already allocated" << std::endl;
    ATHENA_ERROR(msg);
  }
  nreal_user_mesh_data_ = n;
  ruser_mesh_data       = new AthenaArray<Real>[n];
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::AllocateIntUserMeshDataField(int n)
//  \brief Allocate integer AthenaArrays for user-defned data in Mesh

void Mesh::AllocateIntUserMeshDataField(int n)
{
  if (nint_user_mesh_data_ != 0)
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::AllocateIntUserMeshDataField" << std::endl
        << "User Mesh data arrays are already allocated" << std::endl;
    ATHENA_ERROR(msg);
  }
  nint_user_mesh_data_ = n;
  iuser_mesh_data      = new AthenaArray<int>[n];
  return;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::ApplyUserWorkBeforeOutput(ParameterInput *pin)
// \brief Apply MeshBlock::UserWorkBeforeOutput

void Mesh::ApplyUserWorkBeforeOutput(ParameterInput* pin)
{
  const auto& pmb_array = GetMeshBlocksCached();
  const int nmb         = pmb_array.size();
  for (int i = 0; i < nmb; ++i)
  {
    pmb_array[i]->UserWorkBeforeOutput(pin);
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::ApplyUserWorkAfterOutput(ParameterInput *pin)
// \brief Apply MeshBlock::UserWorkAfterOutput

void Mesh::ApplyUserWorkAfterOutput(ParameterInput* pin)
{
  const auto& pmb_array = GetMeshBlocksCached();
  const int nmb         = pmb_array.size();
  for (int i = 0; i < nmb; ++i)
  {
    pmb_array[i]->UserWorkAfterOutput(pin);
  }
}

// ----------------------------------------------------------------------------
// Apply MeshBlock::UserWorkMeshUpdatedPrePostAMRHooks
void Mesh::ApplyUserWorkMeshUpdatedPrePostAMRHooks(ParameterInput* pin)
{
  const auto& pmb_array = GetMeshBlocksCached();
  const int nmb         = pmb_array.size();
  for (int i = 0; i < nmb; ++i)
  {
    pmb_array[i]->UserWorkMeshUpdatedPrePostAMRHooks(pin);
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::Initialize(int res_flag, ParameterInput *pin)
// \brief  initialization before the main loop

void Mesh::Initialize(initialize_style init_style, ParameterInput* pin)
{
  bool iflag   = true;
  int inb      = nbtotal;
  int nthreads = GetNumMeshThreads();
  (void)nthreads;

  int nmb = -1;

  do
  {
    // initialize a vector of MeshBlock pointers
    const auto& pmb_array = GetMeshBlocksCached();
    nmb                   = pmb_array.size();

    if (init_style == initialize_style::pgen)
    {
#pragma omp parallel for num_threads(nthreads)
      for (int i = 0; i < nmb; ++i)
      {
        MeshBlock* pmb = pmb_array[i];
        pmb->ProblemGenerator(pin);
        pmb->CheckUserBoundaries();
      }
    }

// Finalize new comm system: allocate buffers and create persistent MPI
// requests for all CommChannels registered during physics module construction.
// After AMR regrid, surviving blocks already have finalized comm state with
// stale MPI requests and buffer sizes - Reinitialize() tears those down first.
// Newly created blocks have never been finalized and need first-time
// Finalize().
#pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nmb; ++i)
    {
      MeshBlock* pmb = pmb_array[i];
      if (pmb->pcomm->is_finalized())
      {
        pmb->pcomm->Reinitialize();
      }
      else
      {
        pmb->pcomm->Finalize();
      }
      // AMRRegistry: freeze registration and compute buffer sizes.
      // Unlike CommRegistry, AMRRegistry owns no MPI state, so surviving
      // blocks need no tear-down - just skip if already finalized.
      if (pmb->pamr != nullptr && !pmb->pamr->is_finalized())
        pmb->pamr->Finalize();
    }

#pragma omp parallel num_threads(nthreads)
    {
#if FLUID_ENABLED && Z4C_ENABLED
      // Early interior C2P: compute ADM metric on interior, refresh bcc,
      // then run C2P on interior cells before ghost exchange.
      if ((init_style == initialize_style::pgen) ||
          (init_style == initialize_style::regrid) ||
          (init_style == initialize_style::restart))
      {
        const bool enforce_alg = init_style != initialize_style::restart;
        FinalizeZ4cADMPhysical(pmb_array, enforce_alg);

        // Recompute cell-centered B from (prolongated) face-centered B
        // before C2P. Newly created blocks from AMR have stale bcc.
        if (MAGNETIC_FIELDS_ENABLED)
        {
          const int nmb_loc = pmb_array.size();
#pragma omp for
          for (int i = 0; i < nmb_loc; ++i)
          {
            Field* pf        = pmb_array[i]->pfield;
            MeshBlock* pmb_i = pmb_array[i];
            pf->CalculateCellCenteredField(pf->b,
                                           pf->bcc,
                                           pmb_i->pcoord,
                                           pmb_i->is,
                                           pmb_i->ie,
                                           pmb_i->js,
                                           pmb_i->je,
                                           pmb_i->ks,
                                           pmb_i->ke);
          }
        }

        // C2P on interior polishes conserved via reset_floor before exchange.
        static const bool interior_only = true;
        PreparePrimitives(pmb_array, interior_only);
      }
#endif  // FLUID_ENABLED && Z4C_ENABLED

      if ((init_style == initialize_style::pgen) ||
          (init_style == initialize_style::regrid) ||
          (init_style == initialize_style::restart))
      {
        CommunicateConserved(pmb_array);
      }

      // ReconcileSharedFacesFC is disabled for all init styles.
      // pgen should now consistently use discrete Stokes theorem

      // Finalize sub-systems that only need conserved vars -------------------
#if Z4C_ENABLED
      // To finalize Z4c/ADM
      // Prolongate z4c
      // Apply BC [CC,CX,VC]
      // Enforce alg. constraints
      // Prepare ADM variables
      if ((init_style == initialize_style::pgen) ||
          (init_style == initialize_style::regrid) ||
          (init_style == initialize_style::restart))
      {
        const bool enforce_alg = init_style != initialize_style::restart;
        FinalizeZ4cADMGhosts(pmb_array, enforce_alg);
      }
#endif

#if WAVE_ENABLED
      // Prolongate wave
      // Apply BC
      if ((init_style == initialize_style::pgen) ||
          (init_style == initialize_style::regrid) ||
          (init_style == initialize_style::restart))
      {
        FinalizeWave(pmb_array);
      }
#endif
      // ----------------------------------------------------------------------

      // Hydro finalization: seed w1, prolongation & BC, C2P -----------------
      // FinalizeHydroState seeds w1 from current w for potential C2P
      // warm-start.  FinalizeHydroConsRP handles prolongation (if
      // multilevel), physical BCs, and bcc recomputation.  PreparePrimitives
      // runs full-domain C2P so that ghost-zone primitives are valid after
      // load-balance redistribution, and updates w1 via RetainState.
#if FLUID_ENABLED
      if ((init_style == initialize_style::pgen) ||
          (init_style == initialize_style::regrid) ||
          (init_style == initialize_style::restart))
      {
        FinalizeHydroState(pmb_array);
        FinalizeHydroConsRP(pmb_array);

        // C2P on full domain after prolongation. skip_physical avoids
        // floor corrections in ghost zones when Z4c provides the metric.
        PreparePrimitives(pmb_array, false, Z4C_ENABLED);
      }
#endif
      // ----------------------------------------------------------------------

      // ----------------------------------------------------------------------

      // M1 needs to slice into hydro, hence after that R/P -------------------
#if M1_ENABLED
      // Prolongate m1
      // Apply BC
      // Not all registers are reloaded from rst, do this on all init calls
      if ((init_style == initialize_style::pgen) ||
          (init_style == initialize_style::regrid) ||
          (init_style == initialize_style::restart))
      {
        FinalizeM1(pmb_array);
      }
#endif
      // ----------------------------------------------------------------------

#if FLUID_ENABLED && Z4C_ENABLED
      if ((init_style == initialize_style::pgen) ||
          (init_style == initialize_style::regrid) ||
          (init_style == initialize_style::restart))
      {
        // Prepare ADM sources
        // Requires B-field in ghost-zones
        //
        // If M1 is activated then ADM variable coupling requires
        // (sc_E, sp_F_d) which in turn requires (sc_chi, ) etc..
        // Therefore this comes after FinalizeM1
        FinalizeZ4cADM_Matter(pmb_array);
      }
#endif

      if (adaptive)
        if (init_style == initialize_style::pgen)
        {
#pragma omp for
          for (int i = 0; i < nmb; ++i)
          {
            pmb_array[i]->pmr->CheckRefinementCondition();
          }
        }
    }  // omp parallel

    // Prepare various derived field quantities -------------------------------
#if FLUID_ENABLED
    if ((init_style == initialize_style::pgen) ||
        (init_style == initialize_style::regrid) ||
        (init_style == initialize_style::restart))
    {
      CalculateHydroFieldDerived();
    }
#endif  // FLUID_ENABLED
    // ------------------------------------------------------------------------

    // Prepare z4c diagnostic quantities --------------------------------------
#if Z4C_ENABLED
    if ((init_style == initialize_style::pgen) ||
        (init_style == initialize_style::regrid) ||
        (init_style == initialize_style::restart))
    {
      CalculateZ4cInitDiagnostics();
    }
#endif  // Z4C_ENABLED
    // ------------------------------------------------------------------------

    // Further re-gridding as required ----------------------------------------
    if (adaptive)
      if (init_style == initialize_style::pgen)
      {
        iflag                   = false;
        int onb                 = nbtotal;
        const bool mesh_updated = (LoadBalancingAndAdaptiveMeshRefinement(
                                     pin) != AMRStatus::unchanged);

        if (nbtotal == onb)
        {
          iflag = true;
          if (mesh_updated)
          {
            // Cache is already rebuilt by
            // LoadBalancingAndAdaptiveMeshRefinement
            nmb = GetMeshBlocksCached().size();
          }
        }
        else if (nbtotal < onb && Globals::my_rank == 0)
        {
          std::cout << "### Warning in Mesh::Initialize" << std::endl
                    << "The number of MeshBlocks decreased during AMR grid "
                       "initialization."
                    << std::endl
                    << "Possibly the refinement criteria have a problem."
                    << std::endl;
        }
        if (nbtotal > 2 * inb && Globals::my_rank == 0)
        {
          std::cout
            << "### Warning in Mesh::Initialize" << std::endl
            << "The number of MeshBlocks increased more than twice "
               "during initialization."
            << std::endl
            << "More computing power than you expected may be required."
            << std::endl;
        }
      }
    // ------------------------------------------------------------------------
  } while (!iflag);

  // calculate the first time step --------------------------------------------
  {
    const auto& pmb_array = GetMeshBlocksCached();
    nmb                   = pmb_array.size();

#pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nmb; ++i)
    {
      pmb_array[i]->ResetBlockDt();

      if (FLUID_ENABLED)
        pmb_array[i]->phydro->NewBlockTimeStep();

      if (WAVE_ENABLED)
        pmb_array[i]->pwave->NewBlockTimeStep();

      if (Z4C_ENABLED)
        pmb_array[i]->pz4c->NewBlockTimeStep();

      if (M1_ENABLED)
        pmb_array[i]->pm1->NewBlockTimeStep();
    }
  }
  // --------------------------------------------------------------------------

  NewTimeStep();
  return;
}

void Mesh::InitializePostFirstInitialize(initialize_style init_style,
                                         ParameterInput* pin)
{
  // Initialized any required rescalings
#if FLUID_ENABLED
  presc->Initialize();
#endif
  // Whenever we initialize the Mesh, we record global properties
  const bool res = GetGlobalGridGeometry(M_info.x_min,
                                         M_info.x_max,
                                         M_info.dx_min,
                                         M_info.dx_max,
                                         M_info.max_level);

  diagnostic_grid_updated = res || diagnostic_grid_updated;

  // Compute diagnostic quantities associated with pgen:
  if (init_style == initialize_style::pgen)
    CalculateZ4cInitDiagnostics();

  // Evaluate values of fields over trackers
  // Needs to be here as various field classes need a first assembly
  if (init_style == initialize_style::pgen)
    ptracker_extrema->EvaluateAndWriteFields(ncycle, time);
}

void Mesh::InitializePostMainUpdatedMesh(ParameterInput* pin)
{
  // Rescale as required
#if FLUID_ENABLED
  presc->Apply();
#endif
  // Whenever we initialize the Mesh, we record global properties
  const bool res = GetGlobalGridGeometry(M_info.x_min,
                                         M_info.x_max,
                                         M_info.dx_min,
                                         M_info.dx_max,
                                         M_info.max_level);

  diagnostic_grid_updated = res || diagnostic_grid_updated;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::RebuildBlockByGid()
//  \brief Rebuild the O(1) gid -> MeshBlock* lookup table from the linked
//  list. Must be called after constructing or redistributing MeshBlocks.

void Mesh::RebuildBlockByGid()
{
  gid_base_     = nslist[Globals::my_rank];
  const int nmb = nblist[Globals::my_rank];
  block_by_gid_.assign(nmb, nullptr);
  MeshBlock* pmbl = pblock;
  while (pmbl != nullptr)
  {
    int idx = pmbl->gid - gid_base_;
    if (idx >= 0 && idx < nmb)
      block_by_gid_[idx] = pmbl;
    pmbl = pmbl->next;
  }
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlock* Mesh::FindMeshBlock(int tgid)
//  \brief return the MeshBlock whose gid is tgid

MeshBlock* Mesh::FindMeshBlock(int tgid)
{
  int idx = tgid - gid_base_;
  if (idx >= 0 && idx < static_cast<int>(block_by_gid_.size()))
    return block_by_gid_[idx];
  return nullptr;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::SetBlockSizeAndBoundaries(LogicalLocation loc,
//                 RegionSize &block_size, BundaryFlag *block_bcs)
// \brief Set the physical part of a block_size structure and block boundary
// conditions

void Mesh::SetBlockSizeAndBoundaries(LogicalLocation loc,
                                     RegionSize& block_size,
                                     BoundaryFlag* block_bcs)
{
  std::int64_t& lx1    = loc.lx1;
  int& ll              = loc.level;
  std::int64_t nrbx_ll = nrbx1 << (ll - root_level);

  // calculate physical block size, x1
  if (lx1 == 0)
  {
    block_size.x1min                  = mesh_size.x1min;
    block_bcs[BoundaryFace::inner_x1] = mesh_bcs[BoundaryFace::inner_x1];
  }
  else
  {
    Real rx =
      ComputeMeshGeneratorX(lx1, nrbx_ll, use_uniform_meshgen_fn_[X1DIR]);
    block_size.x1min                  = MeshGenerator_[X1DIR](rx, mesh_size);
    block_bcs[BoundaryFace::inner_x1] = BoundaryFlag::block;
  }
  if (lx1 == nrbx_ll - 1)
  {
    block_size.x1max                  = mesh_size.x1max;
    block_bcs[BoundaryFace::outer_x1] = mesh_bcs[BoundaryFace::outer_x1];
  }
  else
  {
    Real rx =
      ComputeMeshGeneratorX(lx1 + 1, nrbx_ll, use_uniform_meshgen_fn_[X1DIR]);
    block_size.x1max                  = MeshGenerator_[X1DIR](rx, mesh_size);
    block_bcs[BoundaryFace::outer_x1] = BoundaryFlag::block;
  }

  // calculate physical block size, x2
  if (mesh_size.nx2 == 1)
  {
    block_size.x2min                  = mesh_size.x2min;
    block_size.x2max                  = mesh_size.x2max;
    block_bcs[BoundaryFace::inner_x2] = mesh_bcs[BoundaryFace::inner_x2];
    block_bcs[BoundaryFace::outer_x2] = mesh_bcs[BoundaryFace::outer_x2];
  }
  else
  {
    std::int64_t& lx2 = loc.lx2;
    nrbx_ll           = nrbx2 << (ll - root_level);
    if (lx2 == 0)
    {
      block_size.x2min                  = mesh_size.x2min;
      block_bcs[BoundaryFace::inner_x2] = mesh_bcs[BoundaryFace::inner_x2];
    }
    else
    {
      Real rx =
        ComputeMeshGeneratorX(lx2, nrbx_ll, use_uniform_meshgen_fn_[X2DIR]);
      block_size.x2min                  = MeshGenerator_[X2DIR](rx, mesh_size);
      block_bcs[BoundaryFace::inner_x2] = BoundaryFlag::block;
    }
    if (lx2 == (nrbx_ll)-1)
    {
      block_size.x2max                  = mesh_size.x2max;
      block_bcs[BoundaryFace::outer_x2] = mesh_bcs[BoundaryFace::outer_x2];
    }
    else
    {
      Real rx = ComputeMeshGeneratorX(
        lx2 + 1, nrbx_ll, use_uniform_meshgen_fn_[X2DIR]);
      block_size.x2max                  = MeshGenerator_[X2DIR](rx, mesh_size);
      block_bcs[BoundaryFace::outer_x2] = BoundaryFlag::block;
    }
  }

  // calculate physical block size, x3
  if (mesh_size.nx3 == 1)
  {
    block_size.x3min                  = mesh_size.x3min;
    block_size.x3max                  = mesh_size.x3max;
    block_bcs[BoundaryFace::inner_x3] = mesh_bcs[BoundaryFace::inner_x3];
    block_bcs[BoundaryFace::outer_x3] = mesh_bcs[BoundaryFace::outer_x3];
  }
  else
  {
    std::int64_t& lx3 = loc.lx3;
    nrbx_ll           = nrbx3 << (ll - root_level);
    if (lx3 == 0)
    {
      block_size.x3min                  = mesh_size.x3min;
      block_bcs[BoundaryFace::inner_x3] = mesh_bcs[BoundaryFace::inner_x3];
    }
    else
    {
      Real rx =
        ComputeMeshGeneratorX(lx3, nrbx_ll, use_uniform_meshgen_fn_[X3DIR]);
      block_size.x3min                  = MeshGenerator_[X3DIR](rx, mesh_size);
      block_bcs[BoundaryFace::inner_x3] = BoundaryFlag::block;
    }
    if (lx3 == (nrbx_ll)-1)
    {
      block_size.x3max                  = mesh_size.x3max;
      block_bcs[BoundaryFace::outer_x3] = mesh_bcs[BoundaryFace::outer_x3];
    }
    else
    {
      Real rx = ComputeMeshGeneratorX(
        lx3 + 1, nrbx_ll, use_uniform_meshgen_fn_[X3DIR]);
      block_size.x3max                  = MeshGenerator_[X3DIR](rx, mesh_size);
      block_bcs[BoundaryFace::outer_x3] = BoundaryFlag::block;
    }
  }

  block_size.x1rat = mesh_size.x1rat;
  block_size.x2rat = mesh_size.x2rat;
  block_size.x3rat = mesh_size.x3rat;

  return;
}

void Mesh::OutputCycleDiagnostics()
{
  const int Real_prec       = std::numeric_limits<Real>::max_digits10 - 1;
  const int dt_precision    = Real_prec;
  const int ratio_precision = 3;
  if (ncycle_out != 0)
  {
    if (ncycle % ncycle_out == 0)
    {
      // std::cout << "cycle=" << ncycle << std::scientific
      //           << std::setprecision(dt_precision)
      //           << " time=" << time << " dt=" << dt;

      std::cout << "cycle=" << ncycle << std::scientific
                << std::setprecision(dt_precision) << " time=" << time
                << " dt=" << dt;
      std::printf(" dtime[hr^-1]=%.2e", evo_rate);

      int nthreads = GetNumMeshThreads();
      std::printf(" MB/thr~=%.1f",
                  nbtotal / static_cast<Real>(Globals::nranks * nthreads));

      if (step_since_lb == 0)
      {
        std::cout << "\nTree changed: MeshBlocks=" << nbtotal;
        if (adaptive)
        {
          std::cout << "; created=" << nbnew << "; destroyed=" << nbdel;
        }

        if (diagnostic_grid_updated)
        {
          std::cout << "\nGrid global properties changed:";

          std::cout << std::setprecision(Real_prec);
          for (int n = 0; n < ndim; ++n)
          {
            std::cout << "\nx_min(" << n << "); " << M_info.x_min(n);
            std::cout << "\nx_max(" << n << "); " << M_info.x_max(n);
          }

          for (int n = 0; n < ndim; ++n)
          {
            std::cout << "\ndx_min(" << n << "); " << M_info.dx_min(n);
            std::cout << "\ndx_max(" << n << "); " << M_info.dx_max(n);
          }

          std::cout << "\nmax_level: " << M_info.max_level;
          diagnostic_grid_updated = false;
        }
      }

      if (dt_diagnostics != -1)
      {
        {
          Real ratio = dt / dt_hyperbolic;
          std::cout << "\ndt_hyperbolic=" << dt_hyperbolic
                    << " ratio=" << std::setprecision(ratio_precision) << ratio
                    << std::setprecision(dt_precision);
          ratio = dt / dt_parabolic;
          std::cout << "\ndt_parabolic=" << dt_parabolic
                    << " ratio=" << std::setprecision(ratio_precision) << ratio
                    << std::setprecision(dt_precision);
        }
        if (UserTimeStep_ != nullptr)
        {
          Real ratio = dt / dt_user;
          std::cout << "\ndt_user=" << dt_user
                    << " ratio=" << std::setprecision(ratio_precision) << ratio
                    << std::setprecision(dt_precision);
        }
      }  // else (empty): dt_diagnostics = -1 -> provide no additional timestep
         // diagnostics
      std::cout << std::endl;
    }
  }
  return;
}

void Mesh::FinalizePostAMR()
{
  // iterate over rank-local MeshBlock
  int nmb         = GetNumMeshBlocksThisRank(Globals::my_rank);
  MeshBlock* pmbl = pblock;
  for (int i = 0; i < nmb; ++i)
  {
    // main logic here....
    pmbl->new_from_amr = false;

    pmbl = pmbl->next;
  }
}

bool Mesh::GetGlobalGridGeometry(AthenaArray<Real>& x_min,
                                 AthenaArray<Real>& x_max,
                                 AthenaArray<Real>& dx_min,
                                 AthenaArray<Real>& dx_max,
                                 int& max_level)
{
  int nmb = -1;
  // initialize a vector of MeshBlock pointers
  const auto& pmb_array = GetMeshBlocksCached();
  nmb                   = pmb_array.size();

  bool grid_updated = false;

  AthenaArray<Real> dx_min_old(ndim);
  AthenaArray<Real> dx_max_old(ndim);
  int max_level_old = max_level;

  AA dx;
  dx.NewAthenaArray(2 * ndim);

  for (int n = 0; n < ndim; ++n)
  {
    dx_min_old(n) = dx_min(n);
    dx_max_old(n) = dx_max(n);
  }

  dx.Fill(std::numeric_limits<Real>::infinity());

  int nthreads = GetNumMeshThreads();
  (void)nthreads;

  switch (ndim)
  {
    case 3:
    {
      x_min(2) = mesh_size.x3min;
      x_max(2) = mesh_size.x3max;

      Real dx2_min = std::numeric_limits<Real>::infinity();
      Real dx2_max = std::numeric_limits<Real>::infinity();

#pragma omp parallel for num_threads(nthreads) \
  reduction(min : dx2_min, dx2_max)
      for (int i = 0; i < nmb; ++i)
      {
        MeshBlock* pmb = pmb_array[i];

        for (int ix = 0; ix < pmb->ncells3 - 1; ++ix)
        {
          dx2_min = std::min(dx2_min, pmb->pcoord->dx3v(ix));
          dx2_max = std::min(dx2_max, -pmb->pcoord->dx3v(ix));
        }
      }

      dx(2)        = dx2_min;
      dx(ndim + 2) = dx2_max;
    }
    case 2:
    {
      x_min(1) = mesh_size.x2min;
      x_max(1) = mesh_size.x2max;

      Real dx1_min = std::numeric_limits<Real>::infinity();
      Real dx1_max = std::numeric_limits<Real>::infinity();

#pragma omp parallel for num_threads(nthreads) \
  reduction(min : dx1_min, dx1_max)
      for (int i = 0; i < nmb; ++i)
      {
        MeshBlock* pmb = pmb_array[i];

        for (int ix = 0; ix < pmb->ncells2 - 1; ++ix)
        {
          dx1_min = std::min(dx1_min, pmb->pcoord->dx2v(ix));
          dx1_max = std::min(dx1_max, -pmb->pcoord->dx2v(ix));
        }
      }

      dx(1)        = dx1_min;
      dx(ndim + 1) = dx1_max;
    }
    case 1:
    {
      x_min(0) = mesh_size.x1min;
      x_max(0) = mesh_size.x1max;

      Real dx0_min = std::numeric_limits<Real>::infinity();
      Real dx0_max = std::numeric_limits<Real>::infinity();

#pragma omp parallel for num_threads(nthreads) \
  reduction(min : dx0_min, dx0_max)
      for (int i = 0; i < nmb; ++i)
      {
        MeshBlock* pmb = pmb_array[i];

        for (int ix = 0; ix < pmb->ncells1 - 1; ++ix)
        {
          dx0_min = std::min(dx0_min, pmb->pcoord->dx1v(ix));
          dx0_max = std::min(dx0_max, -pmb->pcoord->dx1v(ix));
        }
      }

      dx(0)        = dx0_min;
      dx(ndim + 0) = dx0_max;

      break;
    }
    default:
    {
      assert(false);
    }
  }

  int ml = 0;
#pragma omp parallel for num_threads(nthreads) reduction(max : ml)
  for (int nix = 0; nix < nmb; ++nix)
  {
    MeshBlock* pmb = pmb_array[nix];
    ml             = std::max(ml, pmb->loc.level - root_level);
  }
  max_level = ml;

#ifdef MPI_PARALLEL
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Allreduce(MPI_IN_PLACE,
                dx.data(),
                2 * ndim,
                MPI_ATHENA_REAL,
                MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_level, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

#endif

  for (int n = 0; n < ndim; ++n)
  {
    dx_min(n) = dx(n);
    dx_max(n) = -dx(ndim + n);
  }

  for (int n = 0; n < ndim; ++n)
  {
    grid_updated = grid_updated || (dx_min(n) != dx_min_old(n));
    grid_updated = grid_updated || (dx_max(n) != dx_max_old(n));
  }

  return grid_updated || (max_level != max_level_old);
}

void Mesh::CalculateExcisionMask()
{
#if FLUID_ENABLED && Z4C_ENABLED
  const auto& pmb_array = GetMeshBlocksCached();
  const int nmb         = pmb_array.size();

  if (pmb_array[0]->phydro->opt_excision.use_taper ||
      pmb_array[0]->phydro->opt_excision.excise_hydro_damping)
  {
    int nthreads = GetNumMeshThreads();
    (void)nthreads;

#pragma omp parallel num_threads(nthreads)
    {
      MeshBlock* pmb        = nullptr;
      Hydro* ph             = nullptr;
      EquationOfState* peos = nullptr;
      Z4c* pz4c             = nullptr;

#pragma omp for private(pmb, ph, peos, pz4c)
      for (int nix = 0; nix < nmb; ++nix)
      {
        pmb  = pmb_array[nix];
        ph   = pmb->phydro;
        peos = pmb->peos;
        pz4c = pmb->pz4c;

        AT_N_sca adm_alpha(pz4c->storage.adm, Z4c::I_ADM_alpha);

        CC_GLOOP3(k, j, i)
        {
          Real excision_factor;
          const bool can_excise      = peos->CanExcisePoint(excision_factor,
                                                       false,
                                                       adm_alpha,
                                                       pmb->pcoord->x1v,
                                                       pmb->pcoord->x2v,
                                                       pmb->pcoord->x3v,
                                                       i,
                                                       j,
                                                       k);
          ph->excision_mask(k, j, i) = excision_factor;
        }
      }
    }
  }
#endif
}

void Mesh::CalculateHydroFieldDerived()
{
#if FLUID_ENABLED && Z4C_ENABLED

  int nthreads = GetNumMeshThreads();
  (void)nthreads;
  const auto& pmb_array = GetMeshBlocksCached();
  const int nmb         = pmb_array.size();

#pragma omp parallel num_threads(nthreads)
  {
#pragma omp for
    for (int nix = 0; nix < nmb; ++nix)
    {
      MeshBlock* pmb     = pmb_array[nix];
      Hydro* ph          = pmb->phydro;
      Field* pf          = pmb->pfield;
      PassiveScalars* ps = pmb->pscalars;
      Z4c* pz4c          = pmb->pz4c;

      EquationOfState* peos = pmb->peos;

      EquationOfState::geom_sliced_cc gsc;

      AT_N_sca& alpha_          = gsc.alpha_;
      AT_N_sym& gamma_dd_       = gsc.gamma_dd_;
      AT_N_sym& gamma_uu_       = gsc.gamma_uu_;
      AT_N_sca& sqrt_det_gamma_ = gsc.sqrt_det_gamma_;

      // sanitize loop limits (coarse / fine auto-switched)
      int IL = 0;
      int IU = pmb->ncells1 - 1;
      int JL = 0;
      int JU = pmb->ncells2 - 1;
      int KL = 0;
      int KU = pmb->ncells3 - 1;

      const bool coarse_flag   = false;
      const bool skip_physical = false;

      peos->SanitizeLoopLimits(
        IL, IU, JL, JU, KL, KU, coarse_flag, pmb->pcoord);

      for (int k = KL; k <= KU; ++k)
        for (int j = JL; j <= JU; ++j)
        {
          peos->GeometryToSlicedCC(
            gsc, k, j, IL, IU, coarse_flag, pmb->pcoord);
          peos->DerivedQuantities(ph->derived_ms,
                                  ph->derived_int,
                                  pf->derived_ms,
                                  ph->u,
                                  ps->s,
                                  ph->w,
                                  ps->r,
                                  pf->bcc,
                                  gsc,
                                  pmb->pcoord,
                                  k,
                                  j,
                                  IL,
                                  IU,
                                  coarse_flag,
                                  skip_physical);
        }
    }
  }

#endif
}

void Mesh::CalculateZ4cInitDiagnostics()
{
#if Z4C_ENABLED

  int nthreads = GetNumMeshThreads();
  (void)nthreads;
  const auto& pmb_array = GetMeshBlocksCached();
  const int nmb         = pmb_array.size();

#pragma omp parallel num_threads(nthreads)
  {
#pragma omp for
    for (int nix = 0; nix < nmb; ++nix)
    {
      MeshBlock* pmb = pmb_array[nix];
      Z4c* pz4c      = pmb->pz4c;

      pz4c->ADMConstraints(pz4c->storage.con,
                           pz4c->storage.adm,
                           pz4c->storage.mat,
                           pz4c->storage.u);
    }
  }

#endif
}
