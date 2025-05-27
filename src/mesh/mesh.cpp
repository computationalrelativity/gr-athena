//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mesh.cpp
//  \brief implementation of functions in Mesh class

// C headers
// pre-C11: needed before including inttypes.h, else won't define int64_t for C++ code
// #define __STDC_FORMAT_MACROS

// C++ headers
#include <algorithm>
#include <cinttypes>  // format macro "PRId64" for fixed-width integer type std::int64_t
#include <cmath>      // std::abs(), std::pow()
#include <cstdint>    // std::int64_t fixed-wdith integer type alias
#include <cstdlib>
#include <cstring>    // std::memcpy()
#include <iomanip>    // std::setprecision()
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../fft/athena_fft.hpp"
#include "../fft/turbulence.hpp"
#include "../field/field.hpp"
#include "../field/field_diffusion/field_diffusion.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../outputs/io_wrapper.hpp"
#include "../parameter_input.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/buffer_utils.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"
#include "meshblock_tree.hpp"
#include "surfaces.hpp"

#include "../z4c/z4c.hpp"
#include "../z4c/puncture_tracker.hpp"
#include "../z4c/wave_extract.hpp"
#include "../z4c/ahf.hpp"

#if CCE_ENABLED
#include "../z4c/cce/cce.hpp"
#endif
#include "../trackers/extrema_tracker.hpp"

#ifdef EJECTA_ENABLED
#include "../z4c/ejecta.hpp"
#endif

#include "../wave/wave.hpp"
#include "../m1/m1.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// Mesh constructor, builds mesh at start of calculation using parameters in input file

Mesh::Mesh(ParameterInput *pin, int mesh_test) :
    // public members:
    // aggregate initialization of RegionSize struct:
    mesh_size{pin->GetReal("mesh", "x1min"), pin->GetReal("mesh", "x2min"),
              pin->GetReal("mesh", "x3min"), pin->GetReal("mesh", "x1max"),
              pin->GetReal("mesh", "x2max"), pin->GetReal("mesh", "x3max"),
              pin->GetOrAddReal("mesh", "x1rat", 1.0),
              pin->GetOrAddReal("mesh", "x2rat", 1.0),
              pin->GetOrAddReal("mesh", "x3rat", 1.0),
              pin->GetInteger("mesh", "nx1"), pin->GetInteger("mesh", "nx2"),
              pin->GetInteger("mesh", "nx3") },
    mesh_bcs{GetBoundaryFlag(pin->GetOrAddString("mesh", "ix1_bc", "none")),
             GetBoundaryFlag(pin->GetOrAddString("mesh", "ox1_bc", "none")),
             GetBoundaryFlag(pin->GetOrAddString("mesh", "ix2_bc", "none")),
             GetBoundaryFlag(pin->GetOrAddString("mesh", "ox2_bc", "none")),
             GetBoundaryFlag(pin->GetOrAddString("mesh", "ix3_bc", "none")),
             GetBoundaryFlag(pin->GetOrAddString("mesh", "ox3_bc", "none"))},
    f2(mesh_size.nx2 > 1 ? true : false), f3(mesh_size.nx3 > 1 ? true : false),
    ndim(f3 ? 3 : (f2 ? 2 : 1)),
    adaptive(pin->GetOrAddString("mesh", "refinement", "none") == "adaptive"
             ? true : false),
    multilevel((adaptive || pin->GetOrAddString("mesh", "refinement", "none") == "static")
               ? true : false),
    use_split_grmhd_z4c(pin->GetOrAddBoolean("hydro", "use_split_grmhd_z4c", false)),
    fluid_setup(GetFluidFormulation(pin->GetOrAddString("hydro", "active", "true"))),
    start_time(pin->GetOrAddReal("time", "start_time", 0.0)), time(start_time),
    tlim(pin->GetReal("time", "tlim")), dt(std::numeric_limits<Real>::max()),
    dt_hyperbolic(dt), dt_parabolic(dt), dt_user(dt),
    cfl_number(pin->GetReal("time", "cfl_number")),
    nlim(pin->GetOrAddInteger("time", "nlim", -1)), ncycle(),
    ncycle_out(pin->GetOrAddInteger("time", "ncycle_out", 1)),
    dt_diagnostics(pin->GetOrAddInteger("time", "dt_diagnostics", -1)),
    muj(), nuj(), muj_tilde(),
    nbnew(), nbdel(),
    step_since_lb(), gflag(), turb_flag(),
    // private members:
    next_phys_id_(), num_mesh_threads_(pin->GetOrAddInteger("mesh", "num_threads", 1)),
    tree(this),
    use_uniform_meshgen_fn_{true, true, true},
    nreal_user_mesh_data_(), nint_user_mesh_data_(),
    four_pi_G_(), grav_eps_(-1.0), grav_mean_rho_(-1.0),
    lb_flag_(true), lb_automatic_(), lb_manual_(),
    MeshGenerator_{UniformMeshGeneratorX1, UniformMeshGeneratorX2,
                   UniformMeshGeneratorX3},
    BoundaryFunction_{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr},
    AMRFlag_{}, UserSourceTerm_{}, UserTimeStep_{}, ViscosityCoeff_{},
    ConductionCoeff_{}, FieldDiffusivity_{}
{

  std::stringstream msg;
  RegionSize block_size;
  MeshBlock *pfirst{};
  BoundaryFlag block_bcs[6];
  std::int64_t nbmax;
  resume_flag=false;
  // mesh test
  if (mesh_test > 0) Globals::nranks = mesh_test;

#ifdef MPI_PARALLEL
  // reserve phys=0 for former TAG_AMR=8; now hard-coded in Mesh::CreateAMRMPITag()
  next_phys_id_  = 1;
  ReserveMeshBlockPhysIDs();
#endif

  // check number of OpenMP threads for mesh
  if (num_mesh_threads_ < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Number of OpenMP threads must be >= 1, but num_threads="
        << num_mesh_threads_ << std::endl;
    ATHENA_ERROR(msg);
  }

  // check number of grid cells in root level of mesh from input file.
  if (mesh_size.nx1 < 4) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file nx1 must be >= 4, but nx1="
        << mesh_size.nx1 << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mesh_size.nx2 < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file nx2 must be >= 1, but nx2="
        << mesh_size.nx2 << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mesh_size.nx3 < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file nx3 must be >= 1, but nx3="
        << mesh_size.nx3 << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mesh_size.nx2 == 1 && mesh_size.nx3 > 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file: nx2=1, nx3=" << mesh_size.nx3
        << ", 2D problems in x1-x3 plane not supported" << std::endl;
    ATHENA_ERROR(msg);
  }

  // check physical size of mesh (root level) from input file.
  if (mesh_size.x1max <= mesh_size.x1min) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Input x1max must be larger than x1min: x1min=" << mesh_size.x1min
        << " x1max=" << mesh_size.x1max << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mesh_size.x2max <= mesh_size.x2min) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Input x2max must be larger than x2min: x2min=" << mesh_size.x2min
        << " x2max=" << mesh_size.x2max << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mesh_size.x3max <= mesh_size.x3min) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Input x3max must be larger than x3min: x3min=" << mesh_size.x3min
        << " x3max=" << mesh_size.x3max << std::endl;
    ATHENA_ERROR(msg);
  }

  // check the consistency of the periodic boundaries
  if ( ((mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic
    &&   mesh_bcs[BoundaryFace::outer_x1] != BoundaryFlag::periodic)
    ||  (mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic
    &&   mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::periodic))
    ||  (mesh_size.nx2 > 1
    && ((mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic
    &&   mesh_bcs[BoundaryFace::outer_x2] != BoundaryFlag::periodic)
    ||  (mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic
    &&   mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::periodic)))
    ||  (mesh_size.nx3 > 1
    && ((mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic
    &&   mesh_bcs[BoundaryFace::outer_x3] != BoundaryFlag::periodic)
    ||  (mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic
    &&   mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::periodic)))) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "When periodic boundaries are in use, both sides must be periodic."
        << std::endl;
    ATHENA_ERROR(msg);
  }
  if ( ((mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic
    &&   mesh_bcs[BoundaryFace::outer_x1] != BoundaryFlag::shear_periodic)
    ||  (mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::shear_periodic
    &&   mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::shear_periodic))) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "When shear_periodic boundaries are in use, "
        << "both sides must be shear_periodic." << std::endl;
    ATHENA_ERROR(msg);
  }

  // read and set MeshBlock parameters
  block_size.x1rat = mesh_size.x1rat;
  block_size.x2rat = mesh_size.x2rat;
  block_size.x3rat = mesh_size.x3rat;
  block_size.nx1 = pin->GetOrAddInteger("meshblock", "nx1", mesh_size.nx1);
  if (f2)
    block_size.nx2 = pin->GetOrAddInteger("meshblock", "nx2", mesh_size.nx2);
  else
    block_size.nx2 = mesh_size.nx2;
  if (f3)
    block_size.nx3 = pin->GetOrAddInteger("meshblock", "nx3", mesh_size.nx3);
  else
    block_size.nx3 = mesh_size.nx3;

  // check consistency of the block and mesh
  if (mesh_size.nx1 % block_size.nx1 != 0
      || mesh_size.nx2 % block_size.nx2 != 0
      || mesh_size.nx3 % block_size.nx3 != 0) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "the Mesh must be evenly divisible by the MeshBlock" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (block_size.nx1 < 4 || (block_size.nx2 < 4 && f2)
      || (block_size.nx3 < 4 && f3)) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "block_size must be larger than or equal to 4 cells." << std::endl;
    ATHENA_ERROR(msg);
  }

  if (multilevel) {
#ifdef DBG_VC_DOUBLE_RESTRICT
    // MeshBlock minimum size is constrained in multilevel due to double-restrict
    int const min_mb_nx = std::max(4, 4 * NCGHOST - 2);

    if (block_size.nx1 < min_mb_nx
        || (block_size.nx2 < min_mb_nx && f2)
        || (block_size.nx3 < min_mb_nx && f3)) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "for multilevel with '-vertex' and NCGHOST = " << NCGHOST << ", "
          << "block_size must be at least " << min_mb_nx << std::endl
          << "Criterion: max(4, 4 * NCGHOST - 2)" << std::endl;
      ATHENA_ERROR(msg);
    }
#endif // DBG_VC_DOUBLE_RESTRICT
  }


  // calculate the number of the blocks
  nrbx1 = mesh_size.nx1/block_size.nx1;
  nrbx2 = mesh_size.nx2/block_size.nx2;
  nrbx3 = mesh_size.nx3/block_size.nx3;
  nbmax = (nrbx1 > nrbx2) ? nrbx1:nrbx2;
  nbmax = (nbmax > nrbx3) ? nbmax:nrbx3;

  // initialize user-enrollable functions
  if (mesh_size.x1rat != 1.0) {
    use_uniform_meshgen_fn_[X1DIR] = false;
    MeshGenerator_[X1DIR] = DefaultMeshGeneratorX1;
  }
  if (mesh_size.x2rat != 1.0) {
    use_uniform_meshgen_fn_[X2DIR] = false;
    MeshGenerator_[X2DIR] = DefaultMeshGeneratorX2;
  }
  if (mesh_size.x3rat != 1.0) {
    use_uniform_meshgen_fn_[X3DIR] = false;
    MeshGenerator_[X3DIR] = DefaultMeshGeneratorX3;
  }

  // calculate the logical root level and maximum level
  for (root_level=0; (1<<root_level) < nbmax; root_level++) {}
  current_level = root_level;

  tree.CreateRootGrid();

  // Load balancing flag and parameters
#ifdef MPI_PARALLEL
  if (pin->GetOrAddString("loadbalancing","balancer","default") == "automatic")
    lb_automatic_ = true;
  else if (pin->GetOrAddString("loadbalancing","balancer","default") == "manual")
    lb_manual_ = true;
  lb_tolerance_ = pin->GetOrAddReal("loadbalancing","tolerance",0.5);
  lb_interval_ = pin->GetOrAddReal("loadbalancing","interval",10);
#endif

  // SMR / AMR:
  if (adaptive) {
    max_level = pin->GetOrAddInteger("mesh", "numlevel", 1) + root_level - 1;
    if (max_level > 63) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The number of the refinement level must be smaller than "
          << 63 - root_level + 1 << "." << std::endl;
      ATHENA_ERROR(msg);
    }
  } else {
    max_level = 63;
  }

  if (use_split_grmhd_z4c)
  {
#if !defined(USETM)
      msg << "### FATAL ERROR " << std::endl
          << "hydro/use_split_grmhd_z4c can only be used with PrimitiveSolver"
          << std::endl;
      ATHENA_ERROR(msg);
#endif
  }

  if (Z4C_ENABLED)
  {
    int nrad = pin->GetOrAddInteger("psi4_extraction", "num_radii", 0);
    if (nrad > 0) {
      pwave_extr.reserve(nrad);
      for(int n = 0; n < nrad; ++n){
        pwave_extr.push_back(new WaveExtract(this, pin, n));
      }
    }
#if CCE_ENABLED
    // CCE
    int ncce = pin->GetOrAddInteger("cce", "num_radii", 0);
    pcce.reserve(10*ncce);// 10 different components for each radius
    for(int n = 0; n < ncce; ++n)
    {
      // NOTE: these names are used for pittnull code, so DON'T change the convention
      pcce.push_back(new CCE(this, pin, "gxx",n));
      pcce.push_back(new CCE(this, pin, "gxy",n));
      pcce.push_back(new CCE(this, pin, "gxz",n));
      pcce.push_back(new CCE(this, pin, "gyy",n));
      pcce.push_back(new CCE(this, pin, "gyz",n));
      pcce.push_back(new CCE(this, pin, "gzz",n));
      pcce.push_back(new CCE(this, pin, "betax",n));
      pcce.push_back(new CCE(this, pin, "betay",n));
      pcce.push_back(new CCE(this, pin, "betaz",n));
      pcce.push_back(new CCE(this, pin, "alp",n));
    }
#endif

#ifdef EJECTA_ENABLED
    const int nejecta = pin->GetOrAddInteger("ejecta", "num_rad", 0);
    pej_extract.reserve(nejecta);
    for (int n=0; n<nejecta; ++n)
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
    if (npunct > 0) {
      pz4c_tracker.reserve(npunct);
      for (int n = 0; n < npunct; ++n) {
        pz4c_tracker.push_back(new PunctureTracker(this, pin, n));
      }
    }
  }

  // Last entry says if it is restart run or not
  ptracker_extrema = new ExtremaTracker(this, pin, 0);

  if (Z4C_ENABLED)
  {
    // AHF (0 is restart flag for restart)
    int nhorizon = pin->GetOrAddInteger("ahf", "num_horizons",0);
    pah_finder.reserve(nhorizon);
    for (int n = 0; n < nhorizon; ++n) {
      pah_finder.push_back(new AHF(this, pin, n));
    }
  }

  if (EOS_TABLE_ENABLED) peos_table = new EosTable(pin);
  InitUserMeshData(pin);

  if (multilevel) {
    if (block_size.nx1 % 2 == 1 || (block_size.nx2 % 2 == 1 && f2)
        || (block_size.nx3 % 2 == 1 && f3)) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The size of MeshBlock must be divisible by 2 in order to use SMR or AMR."
          << std::endl;
      ATHENA_ERROR(msg);
    }

    InputBlock *pib = pin->pfirst_block;
    while (pib != nullptr) {
      if (pib->block_name.compare(0, 10, "refinement") == 0) {
        RegionSize ref_size;
        ref_size.x1min = pin->GetReal(pib->block_name, "x1min");
        ref_size.x1max = pin->GetReal(pib->block_name, "x1max");
        if (f2) {
          ref_size.x2min = pin->GetReal(pib->block_name, "x2min");
          ref_size.x2max = pin->GetReal(pib->block_name, "x2max");
        } else {
          ref_size.x2min = mesh_size.x2min;
          ref_size.x2max = mesh_size.x2max;
        }
        if (ndim == 3) {
          ref_size.x3min = pin->GetReal(pib->block_name, "x3min");
          ref_size.x3max = pin->GetReal(pib->block_name, "x3max");
        } else {
          ref_size.x3min = mesh_size.x3min;
          ref_size.x3max = mesh_size.x3max;
        }
        int ref_lev = pin->GetInteger(pib->block_name, "level");
        int lrlev = ref_lev + root_level;
        if (lrlev > current_level) current_level = lrlev;
        // range check
        if (ref_lev < 1) {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Refinement level must be larger than 0 (root level = 0)" << std::endl;
          ATHENA_ERROR(msg);
        }
        if (lrlev > max_level) {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Refinement level exceeds the maximum level (specify "
              << "'maxlevel' parameter in <mesh> input block if adaptive)."
              << std::endl;
          ATHENA_ERROR(msg);
        }
        if (ref_size.x1min > ref_size.x1max || ref_size.x2min > ref_size.x2max
            || ref_size.x3min > ref_size.x3max)  {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Invalid refinement region is specified."<<  std::endl;
          ATHENA_ERROR(msg);
        }
        if (ref_size.x1min < mesh_size.x1min || ref_size.x1max > mesh_size.x1max
            || ref_size.x2min < mesh_size.x2min || ref_size.x2max > mesh_size.x2max
            || ref_size.x3min < mesh_size.x3min || ref_size.x3max > mesh_size.x3max) {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Refinement region must be smaller than the whole mesh." << std::endl;
          ATHENA_ERROR(msg);
        }
        // find the logical range in the ref_level
        // note: if this is too slow, this should be replaced with bi-section search.
        std::int64_t lx1min = 0, lx1max = 0, lx2min = 0, lx2max = 0,
                     lx3min = 0, lx3max = 0;
        std::int64_t lxmax = nrbx1*(1LL<<ref_lev);
        for (lx1min=0; lx1min<lxmax; lx1min++) {
          Real rx = ComputeMeshGeneratorX(lx1min+1, lxmax,
                                          use_uniform_meshgen_fn_[X1DIR]);
          if (MeshGenerator_[X1DIR](rx, mesh_size) > ref_size.x1min)
            break;
        }
        for (lx1max=lx1min; lx1max<lxmax; lx1max++) {
          Real rx = ComputeMeshGeneratorX(lx1max+1, lxmax,
                                          use_uniform_meshgen_fn_[X1DIR]);
          if (MeshGenerator_[X1DIR](rx, mesh_size) >= ref_size.x1max)
            break;
        }
        if (lx1min % 2 == 1) lx1min--;
        if (lx1max % 2 == 0) lx1max++;
        if (f2) { // 2D or 3D
          lxmax = nrbx2*(1LL << ref_lev);
          for (lx2min=0; lx2min<lxmax; lx2min++) {
            Real rx = ComputeMeshGeneratorX(lx2min+1, lxmax,
                                            use_uniform_meshgen_fn_[X2DIR]);
            if (MeshGenerator_[X2DIR](rx, mesh_size) > ref_size.x2min)
              break;
          }
          for (lx2max=lx2min; lx2max<lxmax; lx2max++) {
            Real rx = ComputeMeshGeneratorX(lx2max+1, lxmax,
                                            use_uniform_meshgen_fn_[X2DIR]);
            if (MeshGenerator_[X2DIR](rx, mesh_size) >= ref_size.x2max)
              break;
          }
          if (lx2min % 2 == 1) lx2min--;
          if (lx2max % 2 == 0) lx2max++;
        }
        if (ndim == 3) { // 3D
          lxmax = nrbx3*(1LL<<ref_lev);
          for (lx3min=0; lx3min<lxmax; lx3min++) {
            Real rx = ComputeMeshGeneratorX(lx3min+1, lxmax,
                                            use_uniform_meshgen_fn_[X3DIR]);
            if (MeshGenerator_[X3DIR](rx, mesh_size) > ref_size.x3min)
              break;
          }
          for (lx3max=lx3min; lx3max<lxmax; lx3max++) {
            Real rx = ComputeMeshGeneratorX(lx3max+1, lxmax,
                                            use_uniform_meshgen_fn_[X3DIR]);
            if (MeshGenerator_[X3DIR](rx, mesh_size) >= ref_size.x3max)
              break;
          }
          if (lx3min % 2 == 1) lx3min--;
          if (lx3max % 2 == 0) lx3max++;
        }
        // create the finest level
        if (ndim == 1) {
          for (std::int64_t i=lx1min; i<lx1max; i+=2) {
            LogicalLocation nloc;
            nloc.level=lrlev, nloc.lx1=i, nloc.lx2=0, nloc.lx3=0;
            int nnew;
            tree.AddMeshBlock(nloc, nnew);
          }
        }
        if (ndim == 2) {
          for (std::int64_t j=lx2min; j<lx2max; j+=2) {
            for (std::int64_t i=lx1min; i<lx1max; i+=2) {
              LogicalLocation nloc;
              nloc.level=lrlev, nloc.lx1=i, nloc.lx2=j, nloc.lx3=0;
              int nnew;
              tree.AddMeshBlock(nloc, nnew);
            }
          }
        }
        if (ndim == 3) {
          for (std::int64_t k=lx3min; k<lx3max; k+=2) {
            for (std::int64_t j=lx2min; j<lx2max; j+=2) {
              for (std::int64_t i=lx1min; i<lx1max; i+=2) {
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
  if (nbtotal < Globals::nranks) {
    if (mesh_test == 0) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "Too few mesh blocks: nbtotal ("<< nbtotal <<") < nranks ("
          << Globals::nranks << ")" << std::endl;
      ATHENA_ERROR(msg);
    } else { // test
      std::cout << "### Warning in Mesh constructor" << std::endl
                << "Too few mesh blocks: nbtotal ("<< nbtotal <<") < nranks ("
                << Globals::nranks << ")" << std::endl;
    }
  }
#endif

  ranklist = new int[nbtotal];
  nslist = new int[Globals::nranks];
  nblist = new int[Globals::nranks];
  costlist = new double[nbtotal];
  if (adaptive) { // allocate arrays for AMR
    nref = new int[Globals::nranks];
    nderef = new int[Globals::nranks];
    rdisp = new int[Globals::nranks];
    ddisp = new int[Globals::nranks];
    bnref = new int[Globals::nranks];
    bnderef = new int[Globals::nranks];
    brdisp = new int[Globals::nranks];
    bddisp = new int[Globals::nranks];
  }

  // initialize cost array with the simplest estimate; all the blocks are equal
  for (int i=0; i<nbtotal; i++) costlist[i] = 1.0;

  CalculateLoadBalance(costlist, ranklist, nslist, nblist, nbtotal);

  // Output some diagnostic information to terminal

  // Output MeshBlock list and quit (mesh test only); do not create meshes
  if (mesh_test > 0) {
    if (Globals::my_rank == 0) OutputMeshStructure(ndim);
    return;
  }

  // create MeshBlock list for this process
  int nbs = nslist[Globals::my_rank];
  int nbe = nbs + nblist[Globals::my_rank] - 1;
  // create MeshBlock list for this process
  for (int i=nbs; i<=nbe; i++) {
    SetBlockSizeAndBoundaries(loclist[i], block_size, block_bcs);
    // create a block and add into the link list
    if (i == nbs) {
      pblock = new MeshBlock(i, i-nbs, loclist[i], block_size, block_bcs, this,
                             pin, gflag);
      pfirst = pblock;
    } else {
      pblock->next = new MeshBlock(i, i-nbs, loclist[i], block_size, block_bcs,
                                   this, pin, gflag);
      pblock->next->prev = pblock;
      pblock = pblock->next;
    }
    pblock->pbval->SearchAndSetNeighbors(tree, ranklist, nslist);
  }
  pblock = pfirst;

  ResetLoadBalanceVariables();

  if (turb_flag > 0) // TurbulenceDriver depends on the MeshBlock ctor
  {
#ifdef FFT
    ptrbd = new TurbulenceDriver(this, pin);
#endif // FFT
  }

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

Mesh::Mesh(ParameterInput *pin, IOWrapper& resfile, int mesh_test) :
    // public members:
    // aggregate initialization of RegionSize struct:
    // (will be overwritten by memcpy from restart file, in this case)
    mesh_size{pin->GetReal("mesh", "x1min"), pin->GetReal("mesh", "x2min"),
              pin->GetReal("mesh", "x3min"), pin->GetReal("mesh", "x1max"),
              pin->GetReal("mesh", "x2max"), pin->GetReal("mesh", "x3max"),
              pin->GetOrAddReal("mesh", "x1rat", 1.0),
              pin->GetOrAddReal("mesh", "x2rat", 1.0),
              pin->GetOrAddReal("mesh", "x3rat", 1.0),
              pin->GetInteger("mesh", "nx1"), pin->GetInteger("mesh", "nx2"),
              pin->GetInteger("mesh", "nx3") },
    mesh_bcs{GetBoundaryFlag(pin->GetOrAddString("mesh", "ix1_bc", "none")),
             GetBoundaryFlag(pin->GetOrAddString("mesh", "ox1_bc", "none")),
             GetBoundaryFlag(pin->GetOrAddString("mesh", "ix2_bc", "none")),
             GetBoundaryFlag(pin->GetOrAddString("mesh", "ox2_bc", "none")),
             GetBoundaryFlag(pin->GetOrAddString("mesh", "ix3_bc", "none")),
             GetBoundaryFlag(pin->GetOrAddString("mesh", "ox3_bc", "none"))},
    f2(mesh_size.nx2 > 1 ? true : false), f3(mesh_size.nx3 > 1 ? true : false),
    ndim(f3 ? 3 : (f2 ? 2 : 1)),
    adaptive(pin->GetOrAddString("mesh", "refinement", "none") == "adaptive"
             ? true : false),
    multilevel((adaptive || pin->GetOrAddString("mesh", "refinement", "none") == "static")
               ? true : false),
    use_split_grmhd_z4c(pin->GetOrAddBoolean("hydro", "use_split_grmhd_z4c", false)),
    fluid_setup(GetFluidFormulation(pin->GetOrAddString("hydro", "active", "true"))),
    start_time(pin->GetOrAddReal("time", "start_time", 0.0)), time(start_time),
    tlim(pin->GetReal("time", "tlim")), dt(std::numeric_limits<Real>::max()),
    dt_hyperbolic(dt), dt_parabolic(dt), dt_user(dt),
    cfl_number(pin->GetReal("time", "cfl_number")),
    nlim(pin->GetOrAddInteger("time", "nlim", -1)), ncycle(),
    ncycle_out(pin->GetOrAddInteger("time", "ncycle_out", 1)),
    dt_diagnostics(pin->GetOrAddInteger("time", "dt_diagnostics", -1)),
    muj(), nuj(), muj_tilde(),
    nbnew(), nbdel(),
    step_since_lb(), gflag(), turb_flag(),
    // private members:
    next_phys_id_(), num_mesh_threads_(pin->GetOrAddInteger("mesh", "num_threads", 1)),
    tree(this),
    use_uniform_meshgen_fn_{true, true, true},
    nreal_user_mesh_data_(), nint_user_mesh_data_(),
    four_pi_G_(), grav_eps_(-1.0), grav_mean_rho_(-1.0),
    lb_flag_(true), lb_automatic_(), lb_manual_(),
    MeshGenerator_{UniformMeshGeneratorX1, UniformMeshGeneratorX2,
                   UniformMeshGeneratorX3},
    BoundaryFunction_{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr},
    AMRFlag_{}, UserSourceTerm_{}, UserTimeStep_{}, ViscosityCoeff_{},
    ConductionCoeff_{}, FieldDiffusivity_{}
{

  std::stringstream msg;
  RegionSize block_size;
  BoundaryFlag block_bcs[6];
  MeshBlock *pfirst{};
  IOWrapperSizeT *offset{};
  IOWrapperSizeT datasize, listsize, headeroffset;

  // mesh test
  if (mesh_test > 0) Globals::nranks = mesh_test;

#ifdef MPI_PARALLEL
  // reserve phys=0 for former TAG_AMR=8; now hard-coded in Mesh::CreateAMRMPITag()
  next_phys_id_  = 1;
  ReserveMeshBlockPhysIDs();
#endif

  // check the number of OpenMP threads for mesh
  if (num_mesh_threads_ < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Number of OpenMP threads must be >= 1, but num_threads="
        << num_mesh_threads_ << std::endl;
    ATHENA_ERROR(msg);
  }

  // get the end of the header
  headeroffset = resfile.GetPosition();
  // read the restart file
  // the file is already open and the pointer is set to after <par_end>
  IOWrapperSizeT headersize = sizeof(int)*3+sizeof(Real)*2
                              + sizeof(RegionSize)+sizeof(IOWrapperSizeT);
  char *headerdata = new char[headersize];
  if (Globals::my_rank == 0) { // the master process reads the header data
    if (resfile.Read(headerdata, 1, headersize) != headersize) {
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
  hdos += sizeof(IOWrapperSizeT);   // (this updated value is never used)

  delete [] headerdata;

  // initialize
  loclist = new LogicalLocation[nbtotal];
  offset = new IOWrapperSizeT[nbtotal];
  costlist = new double[nbtotal];
  ranklist = new int[nbtotal];
  nslist = new int[Globals::nranks];
  nblist = new int[Globals::nranks];

  block_size.nx1 = pin->GetOrAddInteger("meshblock", "nx1", mesh_size.nx1);
  block_size.nx2 = pin->GetOrAddInteger("meshblock", "nx2", mesh_size.nx2);
  block_size.nx3 = pin->GetOrAddInteger("meshblock", "nx3", mesh_size.nx3);

  // calculate the number of the blocks
  nrbx1 = mesh_size.nx1/block_size.nx1;
  nrbx2 = mesh_size.nx2/block_size.nx2;
  nrbx3 = mesh_size.nx3/block_size.nx3;

  // initialize user-enrollable functions
  if (mesh_size.x1rat != 1.0) {
    use_uniform_meshgen_fn_[X1DIR] = false;
    MeshGenerator_[X1DIR] = DefaultMeshGeneratorX1;
  }
  if (mesh_size.x2rat != 1.0) {
    use_uniform_meshgen_fn_[X2DIR] = false;
    MeshGenerator_[X2DIR] = DefaultMeshGeneratorX2;
  }
  if (mesh_size.x3rat != 1.0) {
    use_uniform_meshgen_fn_[X3DIR] = false;
    MeshGenerator_[X3DIR] = DefaultMeshGeneratorX3;
  }


  // Load balancing flag and parameters
#ifdef MPI_PARALLEL
  if (pin->GetOrAddString("loadbalancing", "balancer", "default") == "automatic")
    lb_automatic_ = true;
  else if (pin->GetOrAddString("loadbalancing", "balancer", "default") == "manual")
    lb_manual_ = true;
  lb_tolerance_ = pin->GetOrAddReal("loadbalancing", "tolerance", 0.5);
  lb_interval_ = pin->GetOrAddReal("loadbalancing", "interval", 10);
#endif

  // SMR / AMR
  if (adaptive) {
    max_level = pin->GetOrAddInteger("mesh", "numlevel", 1) + root_level - 1;
    if (max_level > 63) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The number of the refinement level must be smaller than "
          << 63 - root_level + 1 << "." << std::endl;
      ATHENA_ERROR(msg);
    }
  } else {
    max_level = 63;
  }

  if (use_split_grmhd_z4c)
  {
#if !defined(USETM)
      msg << "### FATAL ERROR " << std::endl
          << "hydro/use_split_grmhd_z4c can only be used with PrimitiveSolver"
          << std::endl;
      ATHENA_ERROR(msg);
#endif
  }

  if (Z4C_ENABLED) {
    int nrad = pin->GetOrAddInteger("psi4_extraction", "num_radii", 0);
    if (nrad > 0) {
      pwave_extr.reserve(nrad);
      for(int n = 0; n < nrad; ++n) {
        pwave_extr.push_back(new WaveExtract(this, pin, n));
      }
    }
#if CCE_ENABLED
    // CCE
    int ncce = pin->GetOrAddInteger("cce", "num_radii", 0);
    pcce.reserve(10*ncce);// 10 different components for each radius
    for(int n = 0; n < ncce; ++n)
    {
      pcce.push_back(new CCE(this, pin, "gxx",n));
      pcce.push_back(new CCE(this, pin, "gxy",n));
      pcce.push_back(new CCE(this, pin, "gxz",n));
      pcce.push_back(new CCE(this, pin, "gyy",n));
      pcce.push_back(new CCE(this, pin, "gyz",n));
      pcce.push_back(new CCE(this, pin, "gzz",n));
      pcce.push_back(new CCE(this, pin, "betax",n));
      pcce.push_back(new CCE(this, pin, "betay",n));
      pcce.push_back(new CCE(this, pin, "betaz",n));
      pcce.push_back(new CCE(this, pin, "alp",n));
    }
#endif

#ifdef EJECTA_ENABLED
    const int nejecta = pin->GetOrAddInteger("ejecta", "num_rad", 0);
    pej_extract.reserve(nejecta);
    for (int n=0; n<nejecta; ++n)
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
    if (npunct > 0) {
      pz4c_tracker.reserve(npunct);
      for (int n = 0; n < npunct; ++n) {
        pz4c_tracker.push_back(new PunctureTracker(this, pin, n));
      }
    }
  }

  // Last entry says if it is restart run or not
  ptracker_extrema = new ExtremaTracker(this, pin, 1);

  if (Z4C_ENABLED)
  {
    // BD: By default do not add any horizon searching
    int nhorizon = pin->GetOrAddInteger("ahf", "num_horizons",0);
    pah_finder.reserve(nhorizon);
    for (int n = 0; n < nhorizon; ++n) {
      pah_finder.push_back(new AHF(this, pin, n));
    }
  }

  if (EOS_TABLE_ENABLED) peos_table = new EosTable(pin);
  InitUserMeshData(pin);
  // read user Mesh data
  IOWrapperSizeT udsize = 0;
  for (int n=0; n<nint_user_mesh_data_; n++)
    udsize += iuser_mesh_data[n].GetSizeInBytes();
  for (int n=0; n<nreal_user_mesh_data_; n++)
    udsize += ruser_mesh_data[n].GetSizeInBytes();

  udsize += 2*NDIM*sizeof(Real)*pz4c_tracker.size();

  if (!ptracker_extrema->use_new_style)
  {
    // c_x1, c_x2, c_x3
    udsize += ptracker_extrema->c_x1.GetSizeInBytes();
    udsize += ptracker_extrema->c_x2.GetSizeInBytes();
    udsize += ptracker_extrema->c_x3.GetSizeInBytes();
  }

  if (udsize != 0) {
    char *userdata = new char[udsize];
    if (Globals::my_rank == 0) { // only the master process reads the ID list
      if (resfile.Read(userdata, 1, udsize) != udsize) {
        msg << "### FATAL ERROR in Mesh constructor" << std::endl
            << "The restart file is broken." << std::endl;
        ATHENA_ERROR(msg);
      }
    }
#ifdef MPI_PARALLEL
    // then broadcast the ID list
    MPI_Bcast(userdata, udsize, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif

    IOWrapperSizeT udoffset=0;
    for (int n=0; n<nint_user_mesh_data_; n++) {
      std::memcpy(iuser_mesh_data[n].data(), &(userdata[udoffset]),
                  iuser_mesh_data[n].GetSizeInBytes());
      udoffset += iuser_mesh_data[n].GetSizeInBytes();
    }
    for (int n=0; n<nreal_user_mesh_data_; n++) {
      std::memcpy(ruser_mesh_data[n].data(), &(userdata[udoffset]),
                  ruser_mesh_data[n].GetSizeInBytes());
      udoffset += ruser_mesh_data[n].GetSizeInBytes();
    }
    for (auto ptracker : pz4c_tracker) {
      std::memcpy(ptracker->pos, &userdata[udoffset], NDIM*sizeof(Real));
      udoffset += 3*sizeof(Real);
      std::memcpy(ptracker->betap, &userdata[udoffset], NDIM*sizeof(Real));
      udoffset += 3*sizeof(Real);
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

    delete [] userdata;
  }

  // read the ID list
  listsize = sizeof(LogicalLocation)+sizeof(Real);
  //allocate the idlist buffer
  char *idlist = new char[listsize*nbtotal];
  if (Globals::my_rank == 0) { // only the master process reads the ID list
    if (resfile.Read(idlist, listsize, nbtotal) != static_cast<unsigned int>(nbtotal)) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The restart file is broken." << std::endl;
      ATHENA_ERROR(msg);
    }
  }
#ifdef MPI_PARALLEL
  // then broadcast the ID list
  MPI_Bcast(idlist, listsize*nbtotal, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif

  int os = 0;
  for (int i=0; i<nbtotal; i++) {
    std::memcpy(&(loclist[i]), &(idlist[os]), sizeof(LogicalLocation));
    os += sizeof(LogicalLocation);
    std::memcpy(&(costlist[i]), &(idlist[os]), sizeof(double));
    os += sizeof(double);
    if (loclist[i].level > current_level) current_level = loclist[i].level;
  }
  delete [] idlist;

  // calculate the header offset and seek
  headeroffset += headersize + udsize + listsize*nbtotal;
  if (Globals::my_rank != 0)
    resfile.Seek(headeroffset);

  // rebuild the Block Tree
  tree.CreateRootGrid();
  for (int i=0; i<nbtotal; i++)
    tree.AddMeshBlockWithoutRefine(loclist[i]);
  int nnb;
  // check the tree structure, and assign GID
  tree.GetMeshBlockList(loclist, nullptr, nnb);
  if (nnb != nbtotal) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Tree reconstruction failed. The total numbers of the blocks do not match. ("
        << nbtotal << " != " << nnb << ")" << std::endl;
    ATHENA_ERROR(msg);
  }

#ifdef MPI_PARALLEL
  if (nbtotal < Globals::nranks) {
    if (mesh_test == 0) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "Too few mesh blocks: nbtotal ("<< nbtotal <<") < nranks ("
          << Globals::nranks << ")" << std::endl;
      ATHENA_ERROR(msg);
    } else { // test
      std::cout << "### Warning in Mesh constructor" << std::endl
                << "Too few mesh blocks: nbtotal ("<< nbtotal <<") < nranks ("
                << Globals::nranks << ")" << std::endl;
      delete [] offset;
      return;
    }
  }
#endif

  if (adaptive) { // allocate arrays for AMR
    nref = new int[Globals::nranks];
    nderef = new int[Globals::nranks];
    rdisp = new int[Globals::nranks];
    ddisp = new int[Globals::nranks];
    bnref = new int[Globals::nranks];
    bnderef = new int[Globals::nranks];
    brdisp = new int[Globals::nranks];
    bddisp = new int[Globals::nranks];
  }

  CalculateLoadBalance(costlist, ranklist, nslist, nblist, nbtotal);

  // Output MeshBlock list and quit (mesh test only); do not create meshes
  if (mesh_test > 0) {
    if (Globals::my_rank == 0) OutputMeshStructure(ndim);
    delete [] offset;
    return;
  }

  // allocate data buffer
  int nb = nblist[Globals::my_rank];
  int nbs = nslist[Globals::my_rank];
  int nbe = nbs + nb - 1;

#if !defined(DBG_RST_WRITE_PER_MB)
  char *mbdata = new char[datasize*nb];
  // load MeshBlocks (parallel)
  if (resfile.Read_at_all(mbdata, datasize, nb, headeroffset+nbs*datasize) !=
      static_cast<unsigned int>(nb)) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "The restart file is broken or input parameters are inconsistent."
        << std::endl;
    ATHENA_ERROR(msg);
  }
  for (int i=nbs; i<=nbe; i++) {
    // Match fixed-width integer precision of IOWrapperSizeT datasize
    std::uint64_t buff_os = datasize * (i-nbs);
    SetBlockSizeAndBoundaries(loclist[i], block_size, block_bcs);
    // create a block and add into the link list
    if (i == nbs) {
      pblock = new MeshBlock(i, i-nbs, this, pin, loclist[i], block_size,
                             block_bcs, costlist[i], mbdata+buff_os, gflag);
      pfirst = pblock;
    } else {
      pblock->next = new MeshBlock(i, i-nbs, this, pin, loclist[i], block_size,
                                   block_bcs, costlist[i], mbdata+buff_os, gflag);
      pblock->next->prev = pblock;
      pblock = pblock->next;
    }

    // BD: needed for cons<->prim after restart
    // if(Z4C_ENABLED && FLUID_ENABLED)
    // {
    //   pblock->pz4c->Z4cToADM(pblock->pz4c->storage.u, pblock->pz4c->storage.adm);
    // }

    pblock->pbval->SearchAndSetNeighbors(tree, ranklist, nslist);
  }
#else
  int nbmin = nblist[0];
  for (int n = 1; n < Globals::nranks; ++n) {
    if (nbmin > nblist[n])
      nbmin = nblist[n];
  }

  char *mbdata = new char[datasize];

  for (int i=nbs; i<=nbe; i++) {

    if (i - nbs < nbmin) {
      // load MeshBlock (parallel)
      if (resfile.Read_at_all(mbdata, datasize, 1, headeroffset+i*datasize) != 1) {
        msg << "### FATAL ERROR in Mesh constructor" << std::endl
            << "The restart file is broken or input parameters are inconsistent."
            << std::endl;
        ATHENA_ERROR(msg);
      }
    } else {
      // load MeshBlock (serial)
      if (resfile.Read_at(mbdata, datasize, 1, headeroffset+i*datasize) != 1) {
        msg << "### FATAL ERROR in Mesh constructor" << std::endl
            << "The restart file is broken or input parameters are inconsistent."
            << std::endl;
        ATHENA_ERROR(msg);
      }
    }

    SetBlockSizeAndBoundaries(loclist[i], block_size, block_bcs);
    // create a block and add into the link list
    if (i == nbs) {
      pblock = new MeshBlock(i, i-nbs, this, pin, loclist[i], block_size,
                             block_bcs, costlist[i], mbdata, gflag);
      pfirst = pblock;
    } else {
      pblock->next = new MeshBlock(i, i-nbs, this, pin, loclist[i], block_size,
                                   block_bcs, costlist[i], mbdata, gflag);
      pblock->next->prev = pblock;
      pblock = pblock->next;
    }

    // BD: needed for cons<->prim after restart
    // if(Z4C_ENABLED && FLUID_ENABLED)
    // {
    //   pblock->pz4c->Z4cToADM(pblock->pz4c->storage.u, pblock->pz4c->storage.adm);
    // }

    pblock->pbval->SearchAndSetNeighbors(tree, ranklist, nslist);
  }
#endif // DBG_RST_WRITE_PER_MB

  pblock = pfirst;
  delete [] mbdata;
  // check consistency
  if (datasize != pblock->GetBlockSizeInBytes()) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "The restart file is broken or input parameters are inconsistent."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  ResetLoadBalanceVariables();

  // clean up
  delete [] offset;

  if (turb_flag > 0) // TurbulenceDriver depends on the MeshBlock ctor
  {
#ifdef FFT
    ptrbd = new TurbulenceDriver(this, pin);
#endif // FFT
  }

#if FLUID_ENABLED
  presc = new gra::hydro::rescaling::Rescaling(this, pin);
#endif

  // storage for Mesh info struct
  M_info.x_min.NewAthenaArray(ndim);
  M_info.x_max.NewAthenaArray(ndim);
  M_info.dx_min.NewAthenaArray(ndim);
  M_info.dx_max.NewAthenaArray(ndim);

  // Surface init needs to come after MeshBlocks have been initialized
  gra::mesh::surfaces::InitSurfaces(this, pin);
}

//----------------------------------------------------------------------------------------
// destructor

Mesh::~Mesh() {
  while (pblock->prev != nullptr) // should not be true
    delete pblock->prev;
  while (pblock->next != nullptr)
    delete pblock->next;
  delete pblock;
  delete [] nslist;
  delete [] nblist;
  delete [] ranklist;
  delete [] costlist;
  delete [] loclist;

#ifdef FFT
  if (turb_flag > 0) delete ptrbd;
#endif // FFT

  if (Z4C_ENABLED) {
    for (auto pwextr : pwave_extr) {
      delete pwextr;
    }
    pwave_extr.resize(0);

#if CCE_ENABLED
    for (auto cce : pcce) {
      delete cce;
    }
    pcce.resize(0);
#endif

    for (auto pah_f : pah_finder) {
      delete pah_f;
    }
    pah_finder.resize(0);

#ifdef EJECTA_ENABLED
    for (auto pej : pej_extract) {
      delete pej;
    }
    pej_extract.resize(0);
#endif

    for (auto tracker : pz4c_tracker) {
      delete tracker;
    }
    pz4c_tracker.resize(0);
  }

  delete ptracker_extrema;

  for (auto surf : psurfs) {
    delete surf;
  }
  psurfs.resize(0);

  if (adaptive) { // deallocate arrays for AMR
    delete [] nref;
    delete [] nderef;
    delete [] rdisp;
    delete [] ddisp;
    delete [] bnref;
    delete [] bnderef;
    delete [] brdisp;
    delete [] bddisp;
  }
  // delete user Mesh data
  if (nreal_user_mesh_data_>0) delete [] ruser_mesh_data;
  if (nint_user_mesh_data_>0) delete [] iuser_mesh_data;
  if (EOS_TABLE_ENABLED) delete peos_table;

#if FLUID_ENABLED
  delete presc;
#endif
}

//-----------------------------------------------------------------------------
// Fill vector with pointers to all MeshBlock objects on current rank
void Mesh::GetMeshBlocksMyRank(std::vector<MeshBlock*> & pmb_array)
{
  const int nmb = GetNumMeshBlocksThisRank(Globals::my_rank);

  if (static_cast<unsigned int>(nmb) != pmb_array.size())
    pmb_array.resize(nmb);

  MeshBlock *pmbl = pblock;

  for (int i = 0; i < nmb; ++i)
  {
    pmb_array[i] = pmbl;
    pmbl = pmbl->next;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::OutputMeshStructure(int ndim)
//  \brief print the mesh structure information

void Mesh::OutputMeshStructure(int ndim) {
  RegionSize block_size;
  BoundaryFlag block_bcs[6];
  FILE *fp = nullptr;

  // open 'mesh_structure.dat' file
  if (f2) {
    if ((fp = std::fopen("mesh_structure.dat","wb")) == nullptr) {
      std::cout << "### ERROR in function Mesh::OutputMeshStructure" << std::endl
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
  std::cout << "Number of logical  refinement levels = " << current_level << std::endl;

  // compute/output number of blocks per level, and cost per level
  int *nb_per_plevel = new int[max_level];
  int *cost_per_plevel = new int[max_level];
  for (int i=0; i<=max_level; ++i) {
    nb_per_plevel[i] = 0;
    cost_per_plevel[i] = 0;
  }
  for (int i=0; i<nbtotal; i++) {
    nb_per_plevel[(loclist[i].level - root_level)]++;
    cost_per_plevel[(loclist[i].level - root_level)] += costlist[i];
  }
  for (int i=root_level; i<=max_level; i++) {
    if (nb_per_plevel[i-root_level] != 0) {
      std::cout << "  Physical level = " << i-root_level << " (logical level = " << i
                << "): " << nb_per_plevel[i-root_level] << " MeshBlocks, cost = "
                << cost_per_plevel[i-root_level] <<  std::endl;
    }
  }

  // compute/output number of blocks per rank, and cost per rank
  std::cout << "Number of parallel ranks = " << Globals::nranks << std::endl;
  int *nb_per_rank = new int[Globals::nranks];
  int *cost_per_rank = new int[Globals::nranks];
  for (int i=0; i<Globals::nranks; ++i) {
    nb_per_rank[i] = 0;
    cost_per_rank[i] = 0;
  }
  for (int i=0; i<nbtotal; i++) {
    nb_per_rank[ranklist[i]]++;
    cost_per_rank[ranklist[i]] += costlist[i];
  }
  for (int i=0; i<Globals::nranks; ++i) {
    std::cout << "  Rank = " << i << ": " << nb_per_rank[i] <<" MeshBlocks, cost = "
              << cost_per_rank[i] << std::endl;
  }

  // output relative size/locations of meshblock to file, for plotting
  double real_max = std::numeric_limits<double>::max();
  double mincost = real_max, maxcost = 0.0, totalcost = 0.0;
  for (int i=root_level; i<=max_level; i++) {
    for (int j=0; j<nbtotal; j++) {
      if (loclist[j].level == i) {
        SetBlockSizeAndBoundaries(loclist[j], block_size, block_bcs);
        std::int64_t &lx1 = loclist[j].lx1;
        std::int64_t &lx2 = loclist[j].lx2;
        std::int64_t &lx3 = loclist[j].lx3;
        int &ll = loclist[j].level;
        mincost = std::min(mincost,costlist[i]);
        maxcost = std::max(maxcost,costlist[i]);
        totalcost += costlist[i];
        std::fprintf(fp,"#MeshBlock %d on rank=%d with cost=%g\n", j, ranklist[j],
                     costlist[j]);
        std::fprintf(
            fp, "#  Logical level %d, location = (%" PRId64 " %" PRId64 " %" PRId64")\n",
            ll, lx1, lx2, lx3);
        if (ndim == 2) {
          std::fprintf(fp, "%g %g\n", block_size.x1min, block_size.x2min);
          std::fprintf(fp, "%g %g\n", block_size.x1max, block_size.x2min);
          std::fprintf(fp, "%g %g\n", block_size.x1max, block_size.x2max);
          std::fprintf(fp, "%g %g\n", block_size.x1min, block_size.x2max);
          std::fprintf(fp, "%g %g\n", block_size.x1min, block_size.x2min);
          std::fprintf(fp, "\n\n");
        }
        if (ndim == 3) {
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2max,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2max,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2min,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2min,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2min,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2max,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2max,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2max,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2max,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2max,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2max,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2min,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp, "\n\n");
        }
      }
    }
  }

  // close file, final outputs
  if (f2) std::fclose(fp);
  std::cout << "Load Balancing:" << std::endl;
  std::cout << "  Minimum cost = " << mincost << ", Maximum cost = " << maxcost
            << ", Average cost = " << totalcost/nbtotal << std::endl << std::endl;
  std::cout << "See the 'mesh_structure.dat' file for a complete list"
            << " of MeshBlocks." << std::endl;
  std::cout << "Use 'python ../vis/python/plot_mesh.py' or gnuplot"
            << " to visualize mesh structure." << std::endl << std::endl;

  delete [] nb_per_plevel;
  delete [] cost_per_plevel;
  delete [] nb_per_rank;
  delete [] cost_per_rank;

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::NewTimeStep()
// \brief function that loops over all MeshBlocks and find new timestep
//        this assumes that phydro->NewBlockTimeStep is already called

void Mesh::NewTimeStep(bool limit_dt_growth) {
  MeshBlock *pmb = pblock;

  if (limit_dt_growth)
  {
  // prevent timestep from growing too fast in between 2x cycles (even if every MeshBlock
    // has new_block_dt > 2.0*dt_old)
    dt = static_cast<Real>(2.0)*dt;
    // consider first MeshBlock on this MPI rank's linked list of blocks:
    dt = std::min(dt, pmb->new_block_dt_);
  }
  else
  {
    dt = pmb->new_block_dt_;
  }

  dt_hyperbolic = pmb->new_block_dt_hyperbolic_;
  dt_parabolic = pmb->new_block_dt_parabolic_;
  dt_user = pmb->new_block_dt_user_;
  pmb = pmb->next;

  while (pmb != nullptr)  {
    dt = std::min(dt, pmb->new_block_dt_);
    dt_hyperbolic  = std::min(dt_hyperbolic, pmb->new_block_dt_hyperbolic_);
    dt_parabolic  = std::min(dt_parabolic, pmb->new_block_dt_parabolic_);
    dt_user  = std::min(dt_user, pmb->new_block_dt_user_);
    pmb = pmb->next;
  }

#ifdef MPI_PARALLEL
  // pack array, MPI allreduce over array, then unpack into Mesh variables
  Real dt_array[4] = {dt, dt_hyperbolic, dt_parabolic, dt_user};
  MPI_Allreduce(MPI_IN_PLACE, dt_array, 4, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  dt            = dt_array[0];
  dt_hyperbolic = dt_array[1];
  dt_parabolic  = dt_array[2];
  dt_user       = dt_array[3];
#endif

  if (time < tlim && (tlim - time) < dt) // timestep would take us past desired endpoint
    dt = tlim - time;

  return;
}

// no arg. limit_dt_growth
void Mesh::NewTimeStep()
{
  Mesh::NewTimeStep(true);
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserBoundaryFunction(BoundaryFace dir, BValHydro my_bc)
//  \brief Enroll a user-defined boundary function

void Mesh::EnrollUserBoundaryFunction(BoundaryFace dir, BValFunc my_bc) {
  std::stringstream msg;
  if (dir < 0 || dir > 5) {
    msg << "### FATAL ERROR in EnrollBoundaryCondition function" << std::endl
        << "dirName = " << dir << " not valid" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mesh_bcs[dir] != BoundaryFlag::user) {
    msg << "### FATAL ERROR in EnrollUserBoundaryFunction" << std::endl
        << "The boundary condition flag must be set to the string 'user' in the "
        << " <mesh> block in the input file to use user-enrolled BCs" << std::endl;
    ATHENA_ERROR(msg);
  }
  BoundaryFunction_[static_cast<int>(dir)]=my_bc;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserRefinementCondition(AMRFlagFunc amrflag)
//  \brief Enroll a user-defined function for checking refinement criteria

void Mesh::EnrollUserRefinementCondition(AMRFlagFunc amrflag) {
  if (adaptive)
    AMRFlag_ = amrflag;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserMeshGenerator(CoordinateDirection,MeshGenFunc my_mg)
//  \brief Enroll a user-defined function for Mesh generation

void Mesh::EnrollUserMeshGenerator(CoordinateDirection dir, MeshGenFunc my_mg) {
  std::stringstream msg;
  if (dir < 0 || dir >= 3) {
    msg << "### FATAL ERROR in EnrollUserMeshGenerator function" << std::endl
        << "dirName = " << dir << " not valid" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (dir == X1DIR && mesh_size.x1rat > 0.0) {
    msg << "### FATAL ERROR in EnrollUserMeshGenerator function" << std::endl
        << "x1rat = " << mesh_size.x1rat <<
        " must be negative for user-defined mesh generator in X1DIR " << std::endl;
    ATHENA_ERROR(msg);
  }
  if (dir == X2DIR && mesh_size.x2rat > 0.0) {
    msg << "### FATAL ERROR in EnrollUserMeshGenerator function" << std::endl
        << "x2rat = " << mesh_size.x2rat <<
        " must be negative for user-defined mesh generator in X2DIR " << std::endl;
    ATHENA_ERROR(msg);
  }
  if (dir == X3DIR && mesh_size.x3rat > 0.0) {
    msg << "### FATAL ERROR in EnrollUserMeshGenerator function" << std::endl
        << "x3rat = " << mesh_size.x3rat <<
        " must be negative for user-defined mesh generator in X3DIR " << std::endl;
    ATHENA_ERROR(msg);
  }
  use_uniform_meshgen_fn_[dir] = false;
  MeshGenerator_[dir] = my_mg;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserExplicitSourceFunction(SrcTermFunc my_func)
//  \brief Enroll a user-defined source function

void Mesh::EnrollUserExplicitSourceFunction(SrcTermFunc my_func) {
  UserSourceTerm_ = my_func;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserTimeStepFunction(TimeStepFunc my_func)
//  \brief Enroll a user-defined time step function

void Mesh::EnrollUserTimeStepFunction(TimeStepFunc my_func) {
  UserTimeStep_ = my_func;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserHistoryOutput(int i, HistoryOutputFunc my_func,
//                                         const char *name, UserHistoryOperation op)
//  \brief Enroll a user-defined history output function and set its name

void Mesh::EnrollUserHistoryOutput(HistoryOutputFunc my_func, const char *name,
                                   UserHistoryOperation op)
{
  user_history_output_names_.push_back(std::move(name));
  user_history_func_.push_back(std::move(my_func));
  user_history_ops_.push_back(std::move(op));
}

void Mesh::EnrollUserHistoryOutput(std::function<Real(MeshBlock*, int)> my_func,
                                   const char *name,
                                   UserHistoryOperation op)
{
  user_history_output_names_.push_back(std::move(name));
  user_history_func_.push_back(my_func);
  user_history_ops_.push_back(std::move(op));
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserMetric(MetricFunc my_func)
//  \brief Enroll a user-defined metric for arbitrary GR coordinates

void Mesh::EnrollUserMetric(MetricFunc my_func) {
  UserMetric_ = my_func;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollViscosityCoefficient(ViscosityCoeff my_func)
//  \brief Enroll a user-defined magnetic field diffusivity function

void Mesh::EnrollViscosityCoefficient(ViscosityCoeffFunc my_func) {
  ViscosityCoeff_ = my_func;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollConductionCoefficient(ConductionCoeff my_func)
//  \brief Enroll a user-defined thermal conduction function

void Mesh::EnrollConductionCoefficient(ConductionCoeffFunc my_func) {
  ConductionCoeff_ = my_func;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollFieldDiffusivity(FieldDiffusionCoeff my_func)
//  \brief Enroll a user-defined magnetic field diffusivity function

void Mesh::EnrollFieldDiffusivity(FieldDiffusionCoeffFunc my_func) {
  FieldDiffusivity_ = my_func;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::AllocateRealUserMeshDataField(int n)
//  \brief Allocate Real AthenaArrays for user-defned data in Mesh

void Mesh::AllocateRealUserMeshDataField(int n) {
  if (nreal_user_mesh_data_ != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::AllocateRealUserMeshDataField"
        << std::endl << "User Mesh data arrays are already allocated" << std::endl;
    ATHENA_ERROR(msg);
  }
  nreal_user_mesh_data_ = n;
  ruser_mesh_data = new AthenaArray<Real>[n];
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::AllocateIntUserMeshDataField(int n)
//  \brief Allocate integer AthenaArrays for user-defned data in Mesh

void Mesh::AllocateIntUserMeshDataField(int n) {
  if (nint_user_mesh_data_ != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::AllocateIntUserMeshDataField"
        << std::endl << "User Mesh data arrays are already allocated" << std::endl;
    ATHENA_ERROR(msg);
  }
  nint_user_mesh_data_ = n;
  iuser_mesh_data = new AthenaArray<int>[n];
  return;
}


//----------------------------------------------------------------------------------------
// \!fn void Mesh::ApplyUserWorkBeforeOutput(ParameterInput *pin)
// \brief Apply MeshBlock::UserWorkBeforeOutput

void Mesh::ApplyUserWorkBeforeOutput(ParameterInput *pin) {
  MeshBlock *pmb = pblock;
  while (pmb != nullptr)  {
    pmb->UserWorkBeforeOutput(pin);
    pmb = pmb->next;
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::ApplyUserWorkAfterOutput(ParameterInput *pin)
// \brief Apply MeshBlock::UserWorkAfterOutput

void Mesh::ApplyUserWorkAfterOutput(ParameterInput *pin) {
  MeshBlock *pmb = pblock;
  while (pmb != nullptr)  {
    pmb->UserWorkAfterOutput(pin);
    pmb = pmb->next;
  }
}

// ----------------------------------------------------------------------------
// Apply MeshBlock::UserWorkMeshUpdatedPrePostAMRHooks
void Mesh::ApplyUserWorkMeshUpdatedPrePostAMRHooks(ParameterInput *pin) {
  MeshBlock *pmb = pblock;
  while (pmb != nullptr)  {
    pmb->UserWorkMeshUpdatedPrePostAMRHooks(pin);
    pmb = pmb->next;
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::Initialize(int res_flag, ParameterInput *pin)
// \brief  initialization before the main loop

void Mesh::Initialize(initialize_style init_style, ParameterInput *pin)
{
  bool iflag = true;
  int inb = nbtotal;
  int nthreads = GetNumMeshThreads();
  (void)nthreads;

  int nmb = -1;
  std::vector<MeshBlock*> pmb_array;

  do
  {
    // initialize a vector of MeshBlock pointers
    GetMeshBlocksMyRank(pmb_array);
    nmb = pmb_array.size();

    if (init_style == initialize_style::pgen)
    {
      #pragma omp parallel for num_threads(nthreads)
      for (int i = 0; i < nmb; ++i)
      {
        MeshBlock *pmb = pmb_array[i];
        pmb->ProblemGenerator(pin);
        pmb->pbval->CheckUserBoundaries();
      }
    }

    // Create send/recv MPI_Requests for all BoundaryData objects
    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nmb; ++i)
    {
      MeshBlock *pmb = pmb_array[i];
      // BoundaryVariable objects evolved in main TimeIntegratorTaskList:
      pmb->pbval->SetupPersistentMPI();
    }

    #pragma omp parallel num_threads(nthreads)
    {
      MeshBlock *pmb;
      BoundaryValues *pbval;

#if defined(DBG_EARLY_INIT_CONSTOPRIM) && FLUID_ENABLED && Z4C_ENABLED
      if ((init_style == initialize_style::pgen)   ||
          (init_style == initialize_style::regrid) ||
          (init_style == initialize_style::restart))
      {
        // ADM on physical
        const bool enforce_alg = init_style != initialize_style::restart;
        FinalizeZ4cADMPhysical(pmb_array, enforce_alg);

        // reset_floor with PrimitiveSolver adjusts the conserved
        // Put this here to further polish values after global regridding
        static const bool interior_only = true;
        PreparePrimitives(pmb_array, interior_only);
      }
#endif // DBG_EARLY_INIT_CONSTOPRIM

      if ((init_style == initialize_style::pgen)   ||
          (init_style == initialize_style::regrid) ||
          (init_style == initialize_style::restart))
      {
        CommunicateConserved(pmb_array);
      }

      // Finalize sub-systems that only need conserved vars -------------------
#if Z4C_ENABLED
      // To finalize Z4c/ADM
      // Prolongate z4c
      // Apply BC [CC,CX,VC]
      // Enforce alg. constraints
      // Prepare ADM variables
      if ((init_style == initialize_style::pgen)   ||
          (init_style == initialize_style::regrid) ||
          (init_style == initialize_style::restart))
      {

#ifdef DBG_EARLY_INIT_CONSTOPRIM
        const bool enforce_alg = init_style != initialize_style::restart;
        FinalizeZ4cADMGhosts(pmb_array, enforce_alg);
#else
        const bool enforce_alg = init_style != initialize_style::restart;
        FinalizeZ4cADM(pmb_array, enforce_alg);
#endif // DBG_EARLY_INIT_CONSTOPRIM
      }
#endif

#if WAVE_ENABLED
      // Prolongate wave
      // Apply BC
      if ((init_style == initialize_style::pgen)   ||
          (init_style == initialize_style::regrid) ||
          (init_style == initialize_style::restart))
      {
        FinalizeWave(pmb_array);
      }
#endif
      // ----------------------------------------------------------------------

      // ----------------------------------------------------------------------
      // Deal with retention of old prim state of fluid in case of rootfinder
#if FLUID_ENABLED
      if ((init_style == initialize_style::pgen)   ||
          (init_style == initialize_style::regrid) ||
          (init_style == initialize_style::restart))
      {
        FinalizeHydro_pgen(pmb_array);
      }
#endif
      // ----------------------------------------------------------------------

      // Treat R/P with prim_rp in the case of fluid + gr + z4c ---------------
      //
      // This requires:
      // - ConservedToPrimitive on interior (physical) grid points
      // - communication of primitives
#if FLUID_ENABLED && GENERAL_RELATIVITY && !defined(DBG_USE_CONS_BC)
      if (multilevel)
      if ((init_style == initialize_style::pgen)   ||
          (init_style == initialize_style::regrid) ||
          (init_style == initialize_style::restart))
      {
        const bool interior_only = true;
        PreparePrimitives(pmb_array, interior_only);
        CommunicatePrimitives(pmb_array);
      }
#endif
      // ----------------------------------------------------------------------

      // Deal with matter prol. & BC ------------------------------------------
#if FLUID_ENABLED && !defined(DBG_USE_CONS_BC)
      if (multilevel)
      if ((init_style == initialize_style::pgen)   ||
          (init_style == initialize_style::regrid) ||
          (init_style == initialize_style::restart))
      {
        FinalizeHydroPrimRP(pmb_array);
      }
#elif FLUID_ENABLED && defined(DBG_USE_CONS_BC)
      if (multilevel)
      if ((init_style == initialize_style::pgen)   ||
          (init_style == initialize_style::regrid) ||
          (init_style == initialize_style::restart))
      {
        FinalizeHydroConsRP(pmb_array);

#if defined(DBG_EARLY_INIT_CONSTOPRIM) && FLUID_ENABLED && Z4C_ENABLED
        PreparePrimitivesGhosts(pmb_array);
#else
        const bool interior_only = false;
        PreparePrimitives(pmb_array, interior_only);
#endif // DBG_EARLY_INIT_CONSTOPRIM
      }
#endif
      // ----------------------------------------------------------------------

      // Initial diffusion coefficients ---------------------------------------
// #if FLUID_ENABLED
//       FinalizeDiffusion(pmb_array);
// #endif
      // ----------------------------------------------------------------------

      // M1 needs to slice into hydro, hence after that R/P -------------------
#if M1_ENABLED
      // Prolongate m1
      // Apply BC
      // Not all registers are reloaded from rst, do this on all init calls
      if ((init_style == initialize_style::pgen)   ||
          (init_style == initialize_style::regrid) ||
          (init_style == initialize_style::restart))
      {
        FinalizeM1(pmb_array);
      }
#endif
      // ----------------------------------------------------------------------

#if FLUID_ENABLED && Z4C_ENABLED
      if ((init_style == initialize_style::pgen)   ||
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
        for (int i = 0; i < nmb; ++i) {
          pmb_array[i]->pmr->CheckRefinementCondition();
        }
      }
    } // omp parallel

    // Prepare various derived field quantities -------------------------------
#if FLUID_ENABLED
    if ((init_style == initialize_style::pgen)   ||
        (init_style == initialize_style::regrid) ||
        (init_style == initialize_style::restart))
    {
      CalculateHydroFieldDerived();
    }
#endif // FLUID_ENABLED
    // ------------------------------------------------------------------------

    // Prepare z4c diagnostic quantities --------------------------------------
#if FLUID_ENABLED
    if ((init_style == initialize_style::pgen)   ||
        (init_style == initialize_style::regrid) ||
        (init_style == initialize_style::restart))
    {
      CalculateZ4cInitDiagnostics();
    }
#endif // FLUID_ENABLED
    // ------------------------------------------------------------------------

    // Further re-gridding as required ----------------------------------------
    if (adaptive)
    if (init_style == initialize_style::pgen)
    {
      iflag = false;
      int onb = nbtotal;
      bool mesh_updated = LoadBalancingAndAdaptiveMeshRefinement(pin);

      if (nbtotal == onb)
      {
        iflag = true;
        if (mesh_updated)
        {
          GetMeshBlocksMyRank(pmb_array);
          nmb = pmb_array.size();
        }
      } else if (nbtotal < onb && Globals::my_rank == 0) {
        std::cout << "### Warning in Mesh::Initialize" << std::endl
                  << "The number of MeshBlocks decreased during AMR grid "
                     "initialization."
                  << std::endl
                  << "Possibly the refinement criteria have a problem."
                  << std::endl;
      }
      if (nbtotal > 2 * inb && Globals::my_rank == 0) {
        std::cout << "### Warning in Mesh::Initialize" << std::endl
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
  #pragma omp parallel for num_threads(nthreads)
  for (int i=0; i<nmb; ++i) {
    if (FLUID_ENABLED)
      pmb_array[i]->phydro->NewBlockTimeStep();

    if (WAVE_ENABLED)
      pmb_array[i]->pwave->NewBlockTimeStep();

    if (Z4C_ENABLED)
      pmb_array[i]->pz4c->NewBlockTimeStep();

    if (M1_ENABLED)
      pmb_array[i]->pm1->NewBlockTimeStep();
  }
  // --------------------------------------------------------------------------

  NewTimeStep();
  return;
}

void Mesh::InitializePostFirstInitialize(initialize_style init_style,
                                         ParameterInput *pin)
{
  // Initialized any required rescalings
#if FLUID_ENABLED
  presc->Initialize();
#endif
  // Whenever we initialize the Mesh, we record global properties
  const bool res = GetGlobalGridGeometry(M_info.x_min, M_info.x_max,
                                         M_info.dx_min, M_info.dx_max,
                                         M_info.max_level);

  diagnostic_grid_updated = res || diagnostic_grid_updated;

  // Compute diagnostic quantities associated with pgen:
  if (init_style == initialize_style::pgen)
    CalculateZ4cInitDiagnostics();
}

void Mesh::InitializePostMainUpdatedMesh(ParameterInput *pin)
{
  // Rescale as required
#if FLUID_ENABLED
  presc->Apply();
#endif
  // Whenever we initialize the Mesh, we record global properties
  const bool res = GetGlobalGridGeometry(M_info.x_min, M_info.x_max,
                                         M_info.dx_min, M_info.dx_max,
                                         M_info.max_level);

  diagnostic_grid_updated = res || diagnostic_grid_updated;
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlock* Mesh::FindMeshBlock(int tgid)
//  \brief return the MeshBlock whose gid is tgid

MeshBlock* Mesh::FindMeshBlock(int tgid) {
  MeshBlock *pbl = pblock;
  while (pbl != nullptr) {
    if (pbl->gid == tgid)
      break;
    pbl = pbl->next;
  }
  return pbl;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::SetBlockSizeAndBoundaries(LogicalLocation loc,
//                 RegionSize &block_size, BundaryFlag *block_bcs)
// \brief Set the physical part of a block_size structure and block boundary conditions

void Mesh::SetBlockSizeAndBoundaries(LogicalLocation loc, RegionSize &block_size,
                                     BoundaryFlag *block_bcs) {
  std::int64_t &lx1 = loc.lx1;
  int &ll = loc.level;
  std::int64_t nrbx_ll = nrbx1 << (ll - root_level);

  // calculate physical block size, x1
  if (lx1 == 0) {
    block_size.x1min = mesh_size.x1min;
    block_bcs[BoundaryFace::inner_x1] = mesh_bcs[BoundaryFace::inner_x1];
  } else {
    Real rx = ComputeMeshGeneratorX(lx1, nrbx_ll, use_uniform_meshgen_fn_[X1DIR]);
    block_size.x1min = MeshGenerator_[X1DIR](rx, mesh_size);
    block_bcs[BoundaryFace::inner_x1] = BoundaryFlag::block;
  }
  if (lx1 == nrbx_ll - 1) {
    block_size.x1max = mesh_size.x1max;
    block_bcs[BoundaryFace::outer_x1] = mesh_bcs[BoundaryFace::outer_x1];
  } else {
    Real rx = ComputeMeshGeneratorX(lx1+1, nrbx_ll, use_uniform_meshgen_fn_[X1DIR]);
    block_size.x1max = MeshGenerator_[X1DIR](rx, mesh_size);
    block_bcs[BoundaryFace::outer_x1] = BoundaryFlag::block;
  }

  // calculate physical block size, x2
  if (mesh_size.nx2 == 1) {
    block_size.x2min = mesh_size.x2min;
    block_size.x2max = mesh_size.x2max;
    block_bcs[BoundaryFace::inner_x2] = mesh_bcs[BoundaryFace::inner_x2];
    block_bcs[BoundaryFace::outer_x2] = mesh_bcs[BoundaryFace::outer_x2];
  } else {
    std::int64_t &lx2 = loc.lx2;
    nrbx_ll = nrbx2 << (ll - root_level);
    if (lx2 == 0) {
      block_size.x2min = mesh_size.x2min;
      block_bcs[BoundaryFace::inner_x2] = mesh_bcs[BoundaryFace::inner_x2];
    } else {
      Real rx = ComputeMeshGeneratorX(lx2, nrbx_ll, use_uniform_meshgen_fn_[X2DIR]);
      block_size.x2min = MeshGenerator_[X2DIR](rx, mesh_size);
      block_bcs[BoundaryFace::inner_x2] = BoundaryFlag::block;
    }
    if (lx2 == (nrbx_ll) - 1) {
      block_size.x2max = mesh_size.x2max;
      block_bcs[BoundaryFace::outer_x2] = mesh_bcs[BoundaryFace::outer_x2];
    } else {
      Real rx = ComputeMeshGeneratorX(lx2+1, nrbx_ll, use_uniform_meshgen_fn_[X2DIR]);
      block_size.x2max = MeshGenerator_[X2DIR](rx, mesh_size);
      block_bcs[BoundaryFace::outer_x2] = BoundaryFlag::block;
    }
  }

  // calculate physical block size, x3
  if (mesh_size.nx3 == 1) {
    block_size.x3min = mesh_size.x3min;
    block_size.x3max = mesh_size.x3max;
    block_bcs[BoundaryFace::inner_x3] = mesh_bcs[BoundaryFace::inner_x3];
    block_bcs[BoundaryFace::outer_x3] = mesh_bcs[BoundaryFace::outer_x3];
  } else {
    std::int64_t &lx3 = loc.lx3;
    nrbx_ll = nrbx3 << (ll - root_level);
    if (lx3 == 0) {
      block_size.x3min = mesh_size.x3min;
      block_bcs[BoundaryFace::inner_x3] = mesh_bcs[BoundaryFace::inner_x3];
    } else {
      Real rx = ComputeMeshGeneratorX(lx3, nrbx_ll, use_uniform_meshgen_fn_[X3DIR]);
      block_size.x3min = MeshGenerator_[X3DIR](rx, mesh_size);
      block_bcs[BoundaryFace::inner_x3] = BoundaryFlag::block;
    }
    if (lx3 == (nrbx_ll) - 1) {
      block_size.x3max = mesh_size.x3max;
      block_bcs[BoundaryFace::outer_x3] = mesh_bcs[BoundaryFace::outer_x3];
    } else {
      Real rx = ComputeMeshGeneratorX(lx3+1, nrbx_ll, use_uniform_meshgen_fn_[X3DIR]);
      block_size.x3max = MeshGenerator_[X3DIR](rx, mesh_size);
      block_bcs[BoundaryFace::outer_x3] = BoundaryFlag::block;
    }
  }

  block_size.x1rat = mesh_size.x1rat;
  block_size.x2rat = mesh_size.x2rat;
  block_size.x3rat = mesh_size.x3rat;

  return;
}

// Public function for advancing next_phys_id_ counter
// E.g. if chemistry or radiation elects to communicate additional information with MPI
// outside the framework of the BoundaryVariable classes

// Store signed, but positive, integer corresponding to the next unused value to be used
// as unique ID for a BoundaryVariable object's single set of MPI calls (formerly "enum
// AthenaTagMPI"). 5 bits of unsigned integer representation are currently reserved
// for this "phys" part of the bitfield tag, making 0, ..., 31 legal values

int Mesh::ReserveTagPhysIDs(int num_phys) {
  // TODO(felker): add safety checks? input, output are positive, obey <= 31= MAX_NUM_PHYS
  int start_id = next_phys_id_;
  next_phys_id_ += num_phys;
  return start_id;
}

// private member fn, called in Mesh() ctor

// depending on compile- and runtime options, reserve the maximum number of "int physid"
// that might be necessary for each MeshBlock's BoundaryValues object to perform MPI
// communication for all BoundaryVariable objects

// TODO(felker): deduplicate this logic, which combines conditionals in MeshBlock ctor

void Mesh::ReserveMeshBlockPhysIDs() {
#ifdef MPI_PARALLEL
  if (FLUID_ENABLED) {
    // Advance Mesh's shared counter (initialized to next_phys_id=1 if MPI)
    // Greedy reservation of phys IDs (only 1 of 2 needed for Hydro if multilevel==false)
    ReserveTagPhysIDs(HydroBoundaryVariable::max_phys_id);
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    ReserveTagPhysIDs(FaceCenteredBoundaryVariable::max_phys_id);
  }
  if (NSCALARS > 0) {
    ReserveTagPhysIDs(CellCenteredBoundaryVariable::max_phys_id);
  }

  if (WAVE_ENABLED) {
    if (WAVE_CC_ENABLED)
      ReserveTagPhysIDs(CellCenteredBoundaryVariable::max_phys_id);

    if (WAVE_VC_ENABLED)
      ReserveTagPhysIDs(VertexCenteredBoundaryVariable::max_phys_id);

    if (WAVE_CX_ENABLED)
      ReserveTagPhysIDs(CellCenteredXBoundaryVariable::max_phys_id);

  }

  if (Z4C_ENABLED) {
    #if defined(Z4C_CC_ENABLED)
      ReserveTagPhysIDs(CellCenteredBoundaryVariable::max_phys_id);
    #elif defined(Z4C_CX_ENABLED)
      ReserveTagPhysIDs(CellCenteredXBoundaryVariable::max_phys_id);
    #else // VC
      ReserveTagPhysIDs(VertexCenteredBoundaryVariable::max_phys_id);
    #endif
  }

  if (M1_ENABLED)
  {
    ReserveTagPhysIDs(CellCenteredBoundaryVariable::max_phys_id);
  }

#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn GetFluidFormulation(std::string input_string)
//  \brief Parses input string to return scoped enumerator flag specifying boundary
//  condition. Typically called in Mesh() ctor initializer list

FluidFormulation GetFluidFormulation(const std::string& input_string) {
  if (input_string == "true") {
    return FluidFormulation::evolve;
  } else if (input_string == "disabled") {
    return FluidFormulation::disabled;
  } else if (input_string == "background") {
    return FluidFormulation::background;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in GetFluidFormulation" << std::endl
        << "Input string=" << input_string << "\n"
        << "is an invalid fluid formulation" << std::endl;
    ATHENA_ERROR(msg);
  }
}

void Mesh::OutputCycleDiagnostics() {
  const int Real_prec = std::numeric_limits<Real>::max_digits10 - 1;
  const int dt_precision = Real_prec;
  const int ratio_precision = 3;
  if (ncycle_out != 0) {
    if (ncycle % ncycle_out == 0) {
      // std::cout << "cycle=" << ncycle << std::scientific
      //           << std::setprecision(dt_precision)
      //           << " time=" << time << " dt=" << dt;

      std::cout << "cycle=" << ncycle << std::scientific
                << std::setprecision(dt_precision)
                << " time=" << time << " dt=" << dt;
      std::printf(" dtime[hr^-1]=%.2e", evo_rate);

      int nthreads = GetNumMeshThreads();
      std::printf(" MB/thr~=%.1f",
        nbtotal / static_cast<Real>(Globals::nranks * GetNumMeshThreads())
      );

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
          for (int n=0; n<ndim; ++n)
          {
            std::cout << "\nx_min(" << n << "); " << M_info.x_min(n);
            std::cout << "\nx_max(" << n << "); " << M_info.x_max(n);
          }

          for (int n=0; n<ndim; ++n)
          {
            std::cout << "\ndx_min(" << n << "); " << M_info.dx_min(n);
            std::cout << "\ndx_max(" << n << "); " << M_info.dx_max(n);
          }

          std::cout << "\nmax_level: " << M_info.max_level;
          diagnostic_grid_updated = false;
        }
      }

      if (dt_diagnostics != -1) {
        if (STS_ENABLED) {
          if (UserTimeStep_ == nullptr)
            std::cout << "=dt_hyperbolic";
          // remaining dt_parabolic diagnostic output handled in STS StartupTaskList
        } else {
          Real ratio = dt / dt_hyperbolic;
          std::cout << "\ndt_hyperbolic=" << dt_hyperbolic << " ratio="
                    << std::setprecision(ratio_precision) << ratio
                    << std::setprecision(dt_precision);
          ratio = dt / dt_parabolic;
          std::cout << "\ndt_parabolic=" << dt_parabolic << " ratio="
                    << std::setprecision(ratio_precision) << ratio
                    << std::setprecision(dt_precision);
        }
        if (UserTimeStep_ != nullptr) {
          Real ratio = dt / dt_user;
          std::cout << "\ndt_user=" << dt_user << " ratio="
                    << std::setprecision(ratio_precision) << ratio
                    << std::setprecision(dt_precision);
        }
      } // else (empty): dt_diagnostics = -1 -> provide no additional timestep diagnostics
      std::cout << std::endl;
    }
  }
  return;
}


void Mesh::FinalizePostAMR()
{
  // iterate over rank-local MeshBlock
  int nmb = GetNumMeshBlocksThisRank(Globals::my_rank);
  MeshBlock *pmbl = pblock;
  for (int i=0; i<nmb; ++i)
  {
    // main logic here....
    pmbl->new_from_amr = false;

    pmbl = pmbl->next;
  }
}

void Mesh::CalculateStoreMetricDerivatives()
{
#if Z4C_ENABLED
  if (!(pblock->pz4c->opt.store_metric_drvts)) return;

  // Compute and store ADM metric drvts at this iteration
  MeshBlock * pmb = pblock;

  while (pmb != nullptr)
  {
    Z4c *pz4c = pmb->pz4c;

    pz4c->ADMDerivatives(pz4c->storage.u, pz4c->storage.adm,
                          pz4c->storage.aux);
    pmb = pmb->next;
  }
#endif // Z4C_ENABLED
}

bool Mesh::GetGlobalGridGeometry(AthenaArray<Real> & x_min,
                                 AthenaArray<Real> & x_max,
                                 AthenaArray<Real> & dx_min,
                                 AthenaArray<Real> & dx_max,
                                 int & max_level)
{
  int nmb = -1;
  std::vector<MeshBlock*> pmb_array;
  // initialize a vector of MeshBlock pointers
  GetMeshBlocksMyRank(pmb_array);
  nmb = pmb_array.size();

  bool grid_updated = false;

  AthenaArray<Real> dx_min_old(ndim);
  AthenaArray<Real> dx_max_old(ndim);
  int max_level_old = max_level;

  AA dx;
  dx.NewAthenaArray(2*ndim);

  for (int n=0; n<ndim; ++n)
  {
    dx_min_old(n) = dx_min(n);
    dx_max_old(n) = dx_max(n);
  }

  dx.Fill(std::numeric_limits<Real>::infinity());

  switch (ndim)
  {
    case 3:
    {
      x_min(2) = mesh_size.x3min;
      x_max(2) = mesh_size.x3max;

      for (int i = 0; i < nmb; ++i)
      {
        MeshBlock *pmb = pmb_array[i];

        for (int ix=0; ix<pmb->ncells3-1; ++ix)
        {
          dx(2)      = std::min(dx(2),       pmb->pcoord->dx3v(ix));
          dx(ndim+2) = std::min(dx(ndim+2), -pmb->pcoord->dx3v(ix));
        }
      }
    }
    case 2:
    {
      x_min(1) = mesh_size.x2min;
      x_max(1) = mesh_size.x2max;

      for (int i = 0; i < nmb; ++i)
      {
        MeshBlock *pmb = pmb_array[i];

        for (int ix=0; ix<pmb->ncells2-1; ++ix)
        {
          dx(1)      = std::min(dx(1),       pmb->pcoord->dx2v(ix));
          dx(ndim+1) = std::min(dx(ndim+1), -pmb->pcoord->dx2v(ix));
        }
      }

    }
    case 1:
    {
      x_min(0) = mesh_size.x1min;
      x_max(0) = mesh_size.x1max;

      for (int i = 0; i < nmb; ++i)
      {
        MeshBlock *pmb = pmb_array[i];

        for (int ix=0; ix<pmb->ncells1-1; ++ix)
        {
          dx(0)      = std::min(dx(0),       pmb->pcoord->dx1v(ix));
          dx(ndim+0) = std::min(dx(ndim+0), -pmb->pcoord->dx1v(ix));
        }
      }

      break;
    }
    default:
    {
      assert(false);
    }
  }

  for (int nix = 0; nix < nmb; ++nix)
  {
    MeshBlock *pmb = pmb_array[nix];
    max_level = pmb->loc.level - root_level;
  }


#ifdef MPI_PARALLEL
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Allreduce(MPI_IN_PLACE, dx.data(), 2*ndim,
                MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_level, 1,
                MPI_INT, MPI_MAX, MPI_COMM_WORLD);

#endif

  for (int n=0; n<ndim; ++n)
  {
    dx_min(n) =  dx(n);
    dx_max(n) = -dx(ndim+n);
  }

  for (int n=0; n<ndim; ++n)
  {
    grid_updated = grid_updated || (dx_min(n) != dx_min_old(n));
    grid_updated = grid_updated || (dx_max(n) != dx_max_old(n));
  }

  return grid_updated || (max_level != max_level_old);
}


void Mesh::CalculateExcisionMask()
{
#if FLUID_ENABLED && Z4C_ENABLED
  if (pblock->phydro->opt_excision.use_taper || pblock->phydro->opt_excision.excise_hydro_damping)
  {
    int inb = nbtotal;
    int nthreads = GetNumMeshThreads();
    (void)nthreads;
    int nmb = GetNumMeshBlocksThisRank(Globals::my_rank);
    std::vector<MeshBlock*> pmb_array(nmb);

    if (static_cast<unsigned int>(nmb) != pmb_array.size())
      pmb_array.resize(nmb);

    MeshBlock *pmbl = pblock;
    for (int i=0; i<nmb; ++i)
    {
      pmb_array[i] = pmbl;
      pmbl = pmbl->next;
    }

    #pragma omp parallel num_threads(nthreads)
    {
      MeshBlock *pmb = nullptr;
      Hydro *ph = nullptr;
      EquationOfState * peos = nullptr;
      Z4c * pz4c = nullptr;

      #pragma omp for private(pmb,ph,peos,pz4c)
      for (int nix=0; nix<nmb; ++nix)
      {
        pmb = pmb_array[nix];
        ph = pmb->phydro;
        peos = pmb->peos;
        pz4c = pmb->pz4c;

        AT_N_sca adm_alpha(pz4c->storage.adm, Z4c::I_ADM_alpha);

        CC_GLOOP3(k, j, i)
        {
          Real excision_factor;
          const bool can_excise = peos->CanExcisePoint(
            excision_factor,
            false, adm_alpha,
            pmb->pcoord->x1v,
            pmb->pcoord->x2v,
            pmb->pcoord->x3v, i, j, k);
          ph->excision_mask(k,j,i) = excision_factor;
        }
      }
    }

  }
#endif
}

void Mesh::CalculateHydroFieldDerived()
{
#if FLUID_ENABLED && Z4C_ENABLED

  int inb = nbtotal;
  int nthreads = GetNumMeshThreads();
  (void)nthreads;
  int nmb = GetNumMeshBlocksThisRank(Globals::my_rank);
  std::vector<MeshBlock*> pmb_array(nmb);

  if (static_cast<unsigned int>(nmb) != pmb_array.size())
    pmb_array.resize(nmb);

  MeshBlock *pmbl = pblock;
  for (int i=0; i<nmb; ++i)
  {
    pmb_array[i] = pmbl;
    pmbl = pmbl->next;
  }

  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for
    for (int nix=0; nix<nmb; ++nix)
    {
      MeshBlock *pmb = pmb_array[nix];
      Hydro *ph = pmb->phydro;
      Field *pf = pmb->pfield;
      PassiveScalars *ps = pmb->pscalars;
      Z4c * pz4c = pmb->pz4c;

      EquationOfState * peos = pmb->peos;

      EquationOfState::geom_sliced_cc gsc;

      AT_N_sca & alpha_    = gsc.alpha_;
      AT_N_sym & gamma_dd_ = gsc.gamma_dd_;
      AT_N_sym & gamma_uu_    = gsc.gamma_uu_;
      AT_N_sca & sqrt_det_gamma_  = gsc.sqrt_det_gamma_;
      AT_N_sca & det_gamma_   = gsc.det_gamma_;

      // sanitize loop limits (coarse / fine auto-switched)
      int IL = 0; int IU = pmb->ncells1-1;
      int JL = 0; int JU = pmb->ncells2-1;
      int KL = 0; int KU = pmb->ncells3-1;

      const bool coarse_flag = false;
      const bool skip_physical = false;

      peos->SanitizeLoopLimits(IL, IU, JL, JU, KL, KU,
                               coarse_flag, pmb->pcoord);

      for (int k = KL; k <= KU; ++k)
      for (int j = JL; j <= JU; ++j)
      {
        peos->GeometryToSlicedCC(gsc, k, j, IL, IU,
                                 coarse_flag, pmb->pcoord);
        peos->DerivedQuantities(
          ph->derived_ms, ph->derived_int, pf->derived_ms,
          ph->u, ps->s,
          ph->w, ps->r,
          pf->bcc, gsc,
          pmb->pcoord,
          k, j, IL, IU,
          coarse_flag, skip_physical
        );
      }
    }
  }

#endif
}


void Mesh::CalculateZ4cInitDiagnostics()
{
#if Z4C_ENABLED

  int inb = nbtotal;
  int nthreads = GetNumMeshThreads();
  (void)nthreads;
  int nmb = GetNumMeshBlocksThisRank(Globals::my_rank);
  std::vector<MeshBlock*> pmb_array(nmb);

  if (static_cast<unsigned int>(nmb) != pmb_array.size())
    pmb_array.resize(nmb);

  MeshBlock *pmbl = pblock;
  for (int i=0; i<nmb; ++i)
  {
    pmb_array[i] = pmbl;
    pmbl = pmbl->next;
  }

  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for
    for (int nix=0; nix<nmb; ++nix)
    {
      MeshBlock *pmb = pmb_array[nix];
      Z4c * pz4c = pmb->pz4c;

      pz4c->ADMConstraints(pz4c->storage.con,
                           pz4c->storage.adm,
                           pz4c->storage.mat,
                           pz4c->storage.u);
    }
  }

#endif
}