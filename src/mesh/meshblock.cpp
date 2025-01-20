//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mesh.cpp
//  \brief implementation of functions in MeshBlock class

// C headers

// C++ headers
#include <algorithm>  // sort()
#include <cstdlib>
#include <cstring>    // memcpy()
#include <ctime>      // clock(), CLOCKS_PER_SEC, clock_t
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../fft/athena_fft.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../parameter_input.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/buffer_utils.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"
#include "meshblock_tree.hpp"
#include "../z4c/z4c.hpp"
#include "../z4c/wave_extract.hpp"
#include "../wave/wave.hpp"
#include "../trackers/extrema_tracker.hpp"
#include "../m1/m1.hpp"

//----------------------------------------------------------------------------------------
// MeshBlock constructor: constructs coordinate, boundary condition, hydro, field
//                        and mesh refinement objects.

MeshBlock::MeshBlock(int igid, int ilid, LogicalLocation iloc, RegionSize input_block,
                     BoundaryFlag *input_bcs, Mesh *pm, ParameterInput *pin,
                     int igflag, bool ref_flag) :
    pmy_mesh(pm), loc(iloc), block_size(input_block),
    gid(igid), lid(ilid), gflag(igflag), nuser_out_var(), prev(nullptr), next(nullptr),
    new_block_dt_{}, new_block_dt_hyperbolic_{}, new_block_dt_parabolic_{},
    new_block_dt_user_{},
    nreal_user_meshblock_data_(), nint_user_meshblock_data_(), cost_(1.0) {

  this->new_from_amr = ref_flag;

  // BD:
  // As this needs to be done twice (here and restarts), is verbose and prone
  // to parablepsis we collect logic..
  SetAllIndicialParameters();

  // (probably don't need to preallocate space for references in these vectors)
  vars_cc_.reserve(3);
  vars_fc_.reserve(3);
  vars_vc_.reserve(3);
  vars_cx_.reserve(1);

  // construct objects stored in MeshBlock class.  Note in particular that the initial
  // conditions for the simulation are set in problem generator called from main, not
  // in the Hydro constructor

  // mesh-related objects
  // Boundary
  pbval  = new BoundaryValues(this, input_bcs, pin);

  // Coordinates
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0)
  {
    pcoord = new Cartesian(this, pin, false);
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0)
  {
    pcoord = new Cylindrical(this, pin, false);
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0)
  {
    pcoord = new SphericalPolar(this, pin, false);
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar_uniform") == 0)
  {
    pcoord = new SphericalPolarUniform(this, pin, false);
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "minkowski") == 0)
  {
    pcoord = new Minkowski(this, pin, false);
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "schwarzschild") == 0)
  {
    pcoord = new Schwarzschild(this, pin, false);
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "kerr-schild") == 0)
  {
    pcoord = new KerrSchild(this, pin, false);
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "gr_user") == 0)
  {
    pcoord = new GRUser(this, pin, false);
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "gr_dynamical") == 0)
  {
    pcoord = new GRDynamical(this, pin, false);
  }

  if (FLUID_ENABLED) {
    // Reconstruction: constructor may implicitly depend on Coordinates, and PPM variable
    // floors depend on EOS, but EOS isn't needed in Reconstruction constructor-> this is ok
    precon = new Reconstruction(this, pin);
  }

  if (pm->multilevel) pmr = new MeshRefinement(this, pin);

  // physics-related, per-MeshBlock objects: may depend on Coordinates for diffusion
  // terms, and may enroll quantities in AMR and BoundaryVariable objs. in BoundaryValues

  // TODO(felker): prepare this section of the MeshBlock ctor to become more complicated
  // for several extensions:
  // 1) allow solver to compile without a Hydro class (or with a Hydro class for the
  // background fluid that is not dynamically evolved)
  // 2) MPI ranks containing MeshBlocks that solve a subset of the physics, e.g. Gravity
  // but not Hydro.
  // 3) MAGNETIC_FIELDS_ENABLED, NSCALARS, (future) FLUID_ENABLED,
  // etc. become runtime switches

  if (FLUID_ENABLED) {
    // if (this->hydro_block)
    phydro = new Hydro(this, pin);
    // } else
    // }
    // Regardless, advance MeshBlock's local counter (initialized to bvars_next_phys_id=1)
    // Greedy reservation of phys IDs (only 1 of 2 needed for Hydro if multilevel==false)
    pbval->AdvanceCounterPhysID(HydroBoundaryVariable::max_phys_id);
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    // if (this->field_block)
    pfield = new Field(this, pin);
    pbval->AdvanceCounterPhysID(FaceCenteredBoundaryVariable::max_phys_id);
  }

  if (NSCALARS > 0) {
    // if (this->scalars_block)
    pscalars = new PassiveScalars(this, pin);
    pbval->AdvanceCounterPhysID(CellCenteredBoundaryVariable::max_phys_id);
  }

  if (WAVE_ENABLED) {
    pwave = new Wave(this, pin);

    if (WAVE_CC_ENABLED)
      pbval->AdvanceCounterPhysID(CellCenteredBoundaryVariable::max_phys_id);

    if (WAVE_VC_ENABLED)
      pbval->AdvanceCounterPhysID(VertexCenteredBoundaryVariable::max_phys_id);

    if (WAVE_CX_ENABLED)
      pbval->AdvanceCounterPhysID(CellCenteredXBoundaryVariable::max_phys_id);

  }


  if (Z4C_ENABLED) {
    pz4c = new Z4c(this, pin);
    int nrad = pin->GetOrAddInteger("psi4_extraction", "num_radii", 0);
    if (nrad > 0) {
      pwave_extr_loc.reserve(nrad);
      for (int n = 0; n < nrad; ++n) {
        pwave_extr_loc.push_back(new WaveExtractLocal(this->pmy_mesh->pwave_extr[n]->psphere, this, pin, n));
      }
    }
    #if defined(Z4C_CC_ENABLED)
      pbval->AdvanceCounterPhysID(CellCenteredBoundaryVariable::max_phys_id);
    #elif defined(Z4C_CX_ENABLED)
      pbval->AdvanceCounterPhysID(CellCenteredXBoundaryVariable::max_phys_id);
    #else // VC
      pbval->AdvanceCounterPhysID(VertexCenteredBoundaryVariable::max_phys_id);
    #endif
  }

  if (FLUID_ENABLED) {
    peos = new EquationOfState(this, pin);
  }

  if (M1_ENABLED)
  {
    pm1 = new M1::M1(this, pin);
    pbval->AdvanceCounterPhysID(CellCenteredBoundaryVariable::max_phys_id);
  }


  // KGF: suboptimal solution, since developer must copy/paste BoundaryVariable derived
  // class type that is used in each PassiveScalars, Gravity, Field, Hydro, ... etc. class
  // in order to correctly advance the BoundaryValues::bvars_next_phys_id_ local counter.

  // TODO(felker): check that local counter pbval->bvars_next_phys_id_ agrees with shared
  // Mesh::next_phys_id_ counter (including non-BoundaryVariable / per-MeshBlock reserved
  // values). Compare both private member variables via BoundaryValues::CheckCounterPhysID


    // must come after pvar to register variables
  ptracker_extrema_loc = new ExtremaTrackerLocal(this, pin);

  // Create user mesh data
  InitUserMeshBlockData(pin);

  return;
}

//----------------------------------------------------------------------------------------
// MeshBlock constructor for restarts

MeshBlock::MeshBlock(int igid, int ilid, Mesh *pm, ParameterInput *pin,
                     LogicalLocation iloc, RegionSize input_block,
                     BoundaryFlag *input_bcs,
                     double icost, char *mbdata, int igflag) :
    pmy_mesh(pm), loc(iloc), block_size(input_block),
    gid(igid), lid(ilid), gflag(igflag), nuser_out_var(), prev(nullptr), next(nullptr),
    new_block_dt_{}, new_block_dt_hyperbolic_{}, new_block_dt_parabolic_{},
    new_block_dt_user_{},
    nreal_user_meshblock_data_(), nint_user_meshblock_data_(), cost_(icost) {
  // BD:
  // As this needs to be done twice (here and restarts), is verbose and prone
  // to parablepsis we collect logic..
  SetAllIndicialParameters();

  // (re-)create mesh-related objects in MeshBlock

  // Boundary
  pbval = new BoundaryValues(this, input_bcs, pin);

  // Coordinates
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    pcoord = new Cartesian(this, pin, false);
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    pcoord = new Cylindrical(this, pin, false);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    pcoord = new SphericalPolar(this, pin, false);
  } else if (std::strcmp(COORDINATE_SYSTEM, "minkowski") == 0) {
    pcoord = new Minkowski(this, pin, false);
  } else if (std::strcmp(COORDINATE_SYSTEM, "schwarzschild") == 0) {
    pcoord = new Schwarzschild(this, pin, false);
  } else if (std::strcmp(COORDINATE_SYSTEM, "kerr-schild") == 0) {
    pcoord = new KerrSchild(this, pin, false);
  } else if (std::strcmp(COORDINATE_SYSTEM, "gr_user") == 0) {
    pcoord = new GRUser(this, pin, false);
  } else if (std::strcmp(COORDINATE_SYSTEM, "gr_dynamical") == 0) {
    pcoord = new GRDynamical(this, pin, false);
  }

  if (FLUID_ENABLED) {
    // Reconstruction (constructor may implicitly depend on Coordinates)
    precon = new Reconstruction(this, pin);
  }

  if (pm->multilevel) pmr = new MeshRefinement(this, pin);

  // (re-)create physics-related objects in MeshBlock

  if (FLUID_ENABLED) {
    // if (this->hydro_block)
    phydro = new Hydro(this, pin);
    // } else
    // }
    // Regardless, advance MeshBlock's local counter (initialized to bvars_next_phys_id=1)
    // Greedy reservation of phys IDs (only 1 of 2 needed for Hydro if multilevel==false)
    pbval->AdvanceCounterPhysID(HydroBoundaryVariable::max_phys_id);
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    // if (this->field_block)
    pfield = new Field(this, pin);
    pbval->AdvanceCounterPhysID(FaceCenteredBoundaryVariable::max_phys_id);
  }

  if (NSCALARS > 0) {
    // if (this->scalars_block)
    pscalars = new PassiveScalars(this, pin);
    pbval->AdvanceCounterPhysID(CellCenteredBoundaryVariable::max_phys_id);
  }

  if (WAVE_ENABLED) {
    pwave = new Wave(this, pin);

    if (WAVE_CC_ENABLED)
      pbval->AdvanceCounterPhysID(CellCenteredBoundaryVariable::max_phys_id);

    if (WAVE_VC_ENABLED)
      pbval->AdvanceCounterPhysID(VertexCenteredBoundaryVariable::max_phys_id);

    if (WAVE_CX_ENABLED)
      pbval->AdvanceCounterPhysID(CellCenteredXBoundaryVariable::max_phys_id);

  }

  if (Z4C_ENABLED) {
    pz4c = new Z4c(this, pin);
    int nrad = pin->GetOrAddInteger("psi4_extraction", "num_radii", 0);
    if (nrad > 0) {
      pwave_extr_loc.reserve(nrad);
      for (int n = 0; n < nrad; ++n) {
        pwave_extr_loc.push_back(new WaveExtractLocal(this->pmy_mesh->pwave_extr[n]->psphere, this, pin, n));
      }
    }
    #if defined(Z4C_CC_ENABLED)
      pbval->AdvanceCounterPhysID(CellCenteredBoundaryVariable::max_phys_id);
    #elif defined(Z4C_CX_ENABLED)
      pbval->AdvanceCounterPhysID(CellCenteredXBoundaryVariable::max_phys_id);
    #else // VC
      pbval->AdvanceCounterPhysID(VertexCenteredBoundaryVariable::max_phys_id);
    #endif
  }

  if (FLUID_ENABLED) {
    peos = new EquationOfState(this, pin);
  }

  if (M1_ENABLED)
  {
    pm1 = new M1::M1(this, pin);
    pbval->AdvanceCounterPhysID(CellCenteredBoundaryVariable::max_phys_id);
  }


  // must come after var to register variables
  ptracker_extrema_loc = new ExtremaTrackerLocal(this, pin);

  InitUserMeshBlockData(pin);

  std::size_t os = 0;
  // NEW_OUTPUT_TYPES:

  auto load_data = [&](const AA & data, const bool advance_position=true)
  {
    std::memcpy(const_cast<AA&>(data).data(), &(mbdata[os]), data.GetSizeInBytes());

    if (advance_position)
    {
      os += data.GetSizeInBytes();
    }
  };

  if (FLUID_ENABLED)
  {
    // load data into register (do not advance)
    load_data(phydro->u,  false);
    // load & advance counter
    load_data(phydro->u1);
  }

  if (GENERAL_RELATIVITY)
  {
    load_data(phydro->w);
    load_data(phydro->w1);
  }
  if (MAGNETIC_FIELDS_ENABLED)
  {
    load_data(pfield->b.x1f, false);
    load_data(pfield->b1.x1f);

    load_data(pfield->b.x2f, false);
    load_data(pfield->b1.x2f);

    load_data(pfield->b.x3f, false);
    load_data(pfield->b1.x3f);
  }

  // (conserved variable) Passive scalars:
  if (NSCALARS > 0)
  {
    // load data into multiple registers (do not advance)
    load_data(pscalars->s, false);
    // load & advance counter
    load_data(pscalars->s1);
  }

  if (WAVE_ENABLED)
  {
    load_data(pwave->u);
  }

  if (Z4C_ENABLED)
  {
    load_data(pz4c->storage.u);
    load_data(pz4c->storage.mat);
  }

  if (M1_ENABLED)
  {
    load_data(pm1->storage.u);
  }


  // load user MeshBlock data
  for (int n=0; n<nint_user_meshblock_data_; n++)
  {
    std::memcpy(iuser_meshblock_data[n].data(), &(mbdata[os]),
                iuser_meshblock_data[n].GetSizeInBytes());
    os += iuser_meshblock_data[n].GetSizeInBytes();
  }
  for (int n=0; n<nreal_user_meshblock_data_; n++)
  {
    std::memcpy(ruser_meshblock_data[n].data(), &(mbdata[os]),
                ruser_meshblock_data[n].GetSizeInBytes());
    os += ruser_meshblock_data[n].GetSizeInBytes();
  }

  return;
}

//----------------------------------------------------------------------------------------
// MeshBlock destructor

MeshBlock::~MeshBlock() {
  if (prev != nullptr) prev->next = next;
  if (next != nullptr) next->prev = prev;

  delete pcoord;
  if (FLUID_ENABLED) delete precon;
  if (pmy_mesh->multilevel) delete pmr;

  if (FLUID_ENABLED) delete phydro;
  if (MAGNETIC_FIELDS_ENABLED) delete pfield;
  if (FLUID_ENABLED) delete peos;
  if (NSCALARS > 0) delete pscalars;
  if (WAVE_ENABLED) delete pwave;

  if (Z4C_ENABLED) {
    delete pz4c;
    for(auto pwextr : pwave_extr_loc) {
      delete pwextr;
    }
    pwave_extr_loc.resize(0);
  }

  delete ptracker_extrema_loc;

  if (M1_ENABLED) delete pm1;

  // BoundaryValues should be destructed AFTER all BoundaryVariable objects are destroyed
  delete pbval;
  // delete user output variables array
  if (nuser_out_var > 0) {
    delete [] user_out_var_names_;
  }
  // delete user MeshBlock data
  if (nreal_user_meshblock_data_ > 0) delete [] ruser_meshblock_data;
  if (nint_user_meshblock_data_ > 0) delete [] iuser_meshblock_data;
}

//----------------------------------------------------------------------------------------
//! \fn inline void MeshBlock::SetAllIndicialParameters(...)
//  \brief Set all requisite indicial parameters
inline void MeshBlock::SetAllIndicialParameters() {
  ng = NGHOST;
  cnghost = (NGHOST + 1)/2 + 1;

  // allow decoupling of coarse ghosts for vertex-centered
  cng = NCGHOST;
  rcng = ng / 2;

  // fundamental grid indicial parameters
  SetIndicialParameters(ng, block_size.nx1,
                        is, ie, ncells1,
                        ivs, ive,
                        ims, ime, ips, ipe,
                        igs, ige, imp,
                        nverts1,
                        true, true);

  SetIndicialParameters(ng, block_size.nx2,
                        js, je, ncells2,
                        jvs, jve,
                        jms, jme, jps, jpe,
                        jgs, jge, jmp,
                        nverts2,
                        true, pmy_mesh->f2);

  SetIndicialParameters(ng, block_size.nx3,
                        ks, ke, ncells3,
                        kvs, kve,
                        kms, kme, kps, kpe,
                        kgs, kge, kmp,
                        nverts3,
                        true, pmy_mesh->f3);

  // coarse grid indicial parameters
  SetIndicialParameters(cng, block_size.nx1 / 2,
                        cis, cie, ncc1,
                        civs, cive,
                        cims, cime, cips, cipe,
                        cigs, cige, cimp,
                        ncv1,
                        pmy_mesh->multilevel, true);

  SetIndicialParameters(cng, block_size.nx2 / 2,
                        cjs, cje, ncc2,
                        cjvs, cjve,
                        cjms, cjme, cjps, cjpe,
                        cjgs, cjge, cjmp,
                        ncv2,
                        pmy_mesh->multilevel, pmy_mesh->f2);

  SetIndicialParameters(cng, block_size.nx3 / 2,
                        cks, cke, ncc3,
                        ckvs, ckve,
                        ckms, ckme, ckps, ckpe,
                        ckgs, ckge, ckmp,
                        ncv3,
                        pmy_mesh->multilevel, pmy_mesh->f3);

  // cell-centered extended ---------------------------------------------------
  SetIndicialParametersCX(NGHOST,
                          block_size.nx1,
                          ncells1,
                          cx_is, cx_ie,
                          cx_ims, cx_ime,
                          cx_ips, cx_ipe,
                          cx_igs, cx_ige,
                          true, true);

  SetIndicialParametersCX(NGHOST,
                          block_size.nx2,
                          ncells2,
                          cx_js, cx_je,
                          cx_jms, cx_jme,
                          cx_jps, cx_jpe,
                          cx_jgs, cx_jge,
                          true,
                          pmy_mesh->f2);

  SetIndicialParametersCX(NGHOST,
                          block_size.nx3,
                          ncells3,
                          cx_ks, cx_ke,
                          cx_kms, cx_kme,
                          cx_kps, cx_kpe,
                          cx_kgs, cx_kge,
                          true,
                          pmy_mesh->f3);

  // coarse grid indicial parameters
  cx_dng = NCGHOST_CX - NGHOST;

  SetIndicialParametersCX(NCGHOST_CX,
                          block_size.nx1 / 2,
                          cx_ncc1,
                          cx_cis, cx_cie,
                          cx_cims, cx_cime,
                          cx_cips, cx_cipe,
                          cx_cigs, cx_cige,
                          pmy_mesh->multilevel,
                          true);

  SetIndicialParametersCX(NCGHOST_CX,
                          block_size.nx2 / 2,
                          cx_ncc2,
                          cx_cjs, cx_cje,
                          cx_cjms, cx_cjme,
                          cx_cjps, cx_cjpe,
                          cx_cjgs, cx_cjge,
                          pmy_mesh->multilevel,
                          pmy_mesh->f2);

  SetIndicialParametersCX(NCGHOST_CX,
                          block_size.nx3 / 2,
                          cx_ncc3,
                          cx_cks, cx_cke,
                          cx_ckms, cx_ckme,
                          cx_ckps, cx_ckpe,
                          cx_ckgs, cx_ckge,
                          pmy_mesh->multilevel,
                          pmy_mesh->f3);


  // // x1
  // cx_ims = 0;
  // cx_ime = NGHOST - 1;

  // cx_ips = block_size.nx1 + NGHOST;
  // cx_ipe = block_size.nx1 + 2 * NGHOST - 1;

  // cx_is = NGHOST;
  // cx_ie = cx_is + block_size.nx1 - 1;

  // cx_igs = cx_is + NGHOST - 1;
  // cx_ige = cx_ie - (NGHOST - 1);

  // // x2
  // cx_jms = 0;
  // cx_jme = NGHOST - 1;

  // cx_jps = block_size.nx2 + NGHOST;
  // cx_jpe = block_size.nx2 + 2 * NGHOST - 1;

  // cx_js = NGHOST;
  // cx_je = cx_js + block_size.nx2 - 1;

  // cx_jgs = cx_js + NGHOST - 1;
  // cx_jge = cx_je - (NGHOST - 1);

  // // x3
  // cx_kms = 0;
  // cx_kme = NGHOST - 1;

  // cx_kps = block_size.nx3 + NGHOST;
  // cx_kpe = block_size.nx3 + 2 * NGHOST - 1;

  // cx_ks = NGHOST;
  // cx_ke = cx_ks + block_size.nx3 - 1;

  // cx_kgs = cx_ks + NGHOST - 1;
  // cx_kge = cx_ke - (NGHOST - 1);

  // // multi-level
  // if (pmy_mesh->multilevel)
  // {
  //   cx_dng = NCGHOST_CX - NGHOST;

  //   // x1
  //   cx_cims = 0;
  //   cx_cime = NCGHOST_CX - 1;

  //   cx_cips = (block_size.nx1 / 2) + NCGHOST_CX;
  //   cx_cipe = (block_size.nx1 / 2) + 2 * NCGHOST_CX - 1;

  //   cx_cis = NCGHOST_CX;
  //   cx_cie = cx_cis + (block_size.nx1 / 2) - 1;

  //   cx_cigs = cx_cis + NCGHOST_CX - 1;
  //   cx_cige = cx_cie - (NCGHOST_CX - 1);

  //   cx_ncc1 = (block_size.nx1 / 2) + 2 * NCGHOST_CX;

  //   // x2
  //   cx_cjms = 0;
  //   cx_cjme = NCGHOST_CX - 1;

  //   cx_cjps = (block_size.nx2 / 2) + NCGHOST_CX;
  //   cx_cjpe = (block_size.nx2 / 2) + 2 * NCGHOST_CX - 1;

  //   cx_cjs = NCGHOST_CX;
  //   cx_cje = cx_cjs + (block_size.nx2 / 2) - 1;

  //   cx_cjgs = cx_cjs + NCGHOST_CX - 1;
  //   cx_cjge = cx_cje - (NCGHOST_CX - 1);

  //   cx_ncc2 = (block_size.nx2 / 2) + 2 * NCGHOST_CX;

  //   // x3
  //   cx_c3ms = 0;
  //   cx_c3me = NCGHOST_CX - 1;

  //   cx_c3ps = (block_size.nx3 / 2) + NCGHOST_CX;
  //   cx_c3pe = (block_size.nx3 / 2) + 2 * NCGHOST_CX - 1;

  //   cx_c3s = NCGHOST_CX;
  //   cx_c3e = cx_c3s + (block_size.nx3 / 2) - 1;

  //   cx_c3gs = cx_c3s + NCGHOST_CX - 1;
  //   cx_c3ge = cx_c3e - (NCGHOST_CX - 1);

  //   cx_ncc3 = (block_size.nx3 / 2) + 2 * NCGHOST_CX;
  // }

  // --------------------------------------------------------------------------


  return;
}

//----------------------------------------------------------------------------------------
//! \fn inline void MeshBlock::SetIndicialParameters(...)
//  \brief Set indicial parameters for a given dimension
inline void MeshBlock::SetIndicialParameters(int num_ghost,
                                             int block_size,
                                             int &ix_s, int &ix_e,
                                             int &ncells,            // CC end
                                             int &ix_vs, int &ix_ve, // bnd
                                             int &ix_ms, int &ix_me, // neg
                                             int &ix_ps, int &ix_pe, // pos
                                             int &ix_gs, int &ix_ge, // int. g
                                             int &ix_mp,
                                             int &nverts,            // VC end
                                             bool populate_ix,
                                             bool is_dim_nontrivial) {

  if (is_dim_nontrivial) {
    // cell centered
    ncells = block_size + 2 * num_ghost;
    // vertex centered
    nverts = block_size + 2 * num_ghost + 1;

    if (populate_ix) {
      // cell centered
      ix_s = num_ghost;
      ix_e = ix_s + block_size - 1;

      // vertex centered
      ix_vs = num_ghost;
      ix_ve = ix_vs + block_size;

      ix_ms = 0;
      ix_me = num_ghost - 1;

      ix_ps = block_size + num_ghost + 1;
      ix_pe = block_size + 2 * num_ghost;

      ix_gs = ix_vs + num_ghost;
      ix_ge = ix_ve - num_ghost;

      ix_mp = ix_vs + block_size / 2;
    }

  } else {
    ncells = nverts = 1;
    if (populate_ix) {
      // cell centered
      ix_s = ix_e = 0;

      // vertex centered
      ix_vs = ix_ve = 0;

      ix_ms = ix_me = ix_ps = ix_pe = 0;
      ix_gs = ix_ge = 0;
      ix_mp = 0;
    }

  }
  return;
}

inline void MeshBlock::SetIndicialParametersCX(int num_ghost,
                                               int block_size,
                                               int &ncells,
                                               int &ix_is, int &ix_ie,
                                               int &ix_ms, int &ix_me,
                                               int &ix_ps, int &ix_pe,
                                               int &ix_gs, int &ix_ge,
                                               bool populate_ix,
                                               bool is_dim_nontrivial) {

  if (is_dim_nontrivial) {
    // cell centered
    ncells = block_size + 2 * num_ghost;

    if (populate_ix) {
      // cell centered
      ix_is = num_ghost;
      ix_ie = ix_is + block_size - 1;

      ix_ms = 0;
      ix_me = num_ghost - 1;

      ix_ps = block_size + num_ghost;
      ix_pe = block_size + 2 * num_ghost - 1;

      ix_gs = ix_is + num_ghost - 1;
      ix_ge = ix_ie - (num_ghost - 1);
    }

  } else {
    ncells = 1;
    if (populate_ix) {
      // cell centered
      ix_is = ix_ie = 0;
      ix_ms = ix_me = ix_ps = ix_pe = 0;
      ix_gs = ix_ge = 0;
    }

  }
  return;
}
//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::AllocateRealUserMeshBlockDataField(int n)
//  \brief Allocate Real AthenaArrays for user-defned data in MeshBlock

void MeshBlock::AllocateRealUserMeshBlockDataField(int n) {
  if (nreal_user_meshblock_data_ != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MeshBlock::AllocateRealUserMeshBlockDataField"
        << std::endl << "User MeshBlock data arrays are already allocated" << std::endl;
    ATHENA_ERROR(msg);
  }
  nreal_user_meshblock_data_ = n;
  ruser_meshblock_data = new AthenaArray<Real>[n];
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::AllocateIntUserMeshBlockDataField(int n)
//  \brief Allocate integer AthenaArrays for user-defned data in MeshBlock

void MeshBlock::AllocateIntUserMeshBlockDataField(int n) {
  if (nint_user_meshblock_data_ != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MeshBlock::AllocateIntusermeshblockDataField"
        << std::endl << "User MeshBlock data arrays are already allocated" << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
  nint_user_meshblock_data_=n;
  iuser_meshblock_data = new AthenaArray<int>[n];
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::AllocateUserOutputVariables(int n)
//  \brief Allocate user-defined output variables

void MeshBlock::AllocateUserOutputVariables(int n) {
  if (n <= 0) return;
  if (nuser_out_var != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MeshBlock::AllocateUserOutputVariables"
        << std::endl << "User output variables are already allocated." << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
  nuser_out_var = n;
  user_out_var.NewAthenaArray(nuser_out_var, ncells3, ncells2, ncells1);
  user_out_var_names_ = new std::string[n];
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::SetUserOutputVariableName(int n, const char *name)
//  \brief set the user-defined output variable name

void MeshBlock::SetUserOutputVariableName(int n, const char *name) {
  if (n >= nuser_out_var) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MeshBlock::SetUserOutputVariableName"
        << std::endl << "User output variable is not allocated." << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
  user_out_var_names_[n] = name;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn std::size_t MeshBlock::GetBlockSizeInBytes()
//  \brief Calculate the block data size required for restart.

std::size_t MeshBlock::GetBlockSizeInBytes() {
  std::size_t size = 0;
  // NEW_OUTPUT_TYPES:
  if (FLUID_ENABLED)
    size += phydro->u.GetSizeInBytes();

  if (GENERAL_RELATIVITY) {
    size += phydro->w.GetSizeInBytes();
    size += phydro->w1.GetSizeInBytes();
  }
  if (MAGNETIC_FIELDS_ENABLED)
    size += (pfield->b.x1f.GetSizeInBytes() + pfield->b.x2f.GetSizeInBytes()
             + pfield->b.x3f.GetSizeInBytes());
  if (NSCALARS > 0)
    size += pscalars->s.GetSizeInBytes();

  if (WAVE_ENABLED) {
    size += pwave->u.GetSizeInBytes();
  }

  if (Z4C_ENABLED) {
    size+=pz4c->storage.u.GetSizeInBytes();
    size+=pz4c->storage.mat.GetSizeInBytes();
  }

  if (M1_ENABLED)
  {
    size += pm1->storage.u.GetSizeInBytes();
  }

  // calculate user MeshBlock data size
  for (int n=0; n<nint_user_meshblock_data_; n++)
    size += iuser_meshblock_data[n].GetSizeInBytes();
  for (int n=0; n<nreal_user_meshblock_data_; n++)
    size += ruser_meshblock_data[n].GetSizeInBytes();

  return size;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::SetCostForLoadBalancing(double cost)
//  \brief stop time measurement and accumulate it in the MeshBlock cost

void MeshBlock::SetCostForLoadBalancing(double cost) {
  if (pmy_mesh->lb_manual_) {
    cost_ = std::min(cost, TINY_NUMBER);
    pmy_mesh->lb_flag_ = true;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ResetTimeMeasurement()
//  \brief reset the MeshBlock cost for automatic load balancing

void MeshBlock::ResetTimeMeasurement() {
  if (pmy_mesh->lb_automatic_) cost_ = TINY_NUMBER;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::StartTimeMeasurement()
//  \brief start time measurement for automatic load balancing

void MeshBlock::StartTimeMeasurement() {
  // coutGreen("MeshBlock::StartTimeMeasurement\n");
  if (pmy_mesh->lb_automatic_) {
#ifdef OPENMP_PARALLEL
    lb_time_ = omp_get_wtime();
#else
    lb_time_ = static_cast<double>(clock());
#endif
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::StartTimeMeasurement()
//  \brief stop time measurement and accumulate it in the MeshBlock cost

void MeshBlock::StopTimeMeasurement() {
  // coutGreen("MeshBlock::StopTimeMeasurement\n");
  if (pmy_mesh->lb_automatic_) {
#ifdef OPENMP_PARALLEL
    lb_time_ = omp_get_wtime() - lb_time_;
#else
    lb_time_ = static_cast<double>(clock()) - lb_time_;
#endif
    cost_ += lb_time_;
  }
}

void MeshBlock::RegisterMeshBlockDataCC(AthenaArray<Real> &pvar_in) {
  vars_cc_.push_back(pvar_in);
  return;
}

void MeshBlock::RegisterMeshBlockDataVC(AthenaArray<Real> &pvar_in) {
  vars_vc_.push_back(pvar_in);
  return;
}

void MeshBlock::RegisterMeshBlockDataFC(FaceField &pvar_fc) {
  vars_fc_.push_back(pvar_fc);
  return;
}

void MeshBlock::RegisterMeshBlockDataCX(AthenaArray<Real> &pvar_in) {
  vars_cx_.push_back(pvar_in);
  return;
}

// TODO(felker): consider merging the MeshRefinement::pvars_cc/fc_ into the
// MeshBlock::pvars_cc/fc_. Would need to weaken the MeshBlock std::vector to use tuples
// of pointers instead of a std::vector of references, so that:
// - nullptr can be passed for the second entry if multilevel==false
// - we can rebind the pointers to Hydro for GR purposes in bvals_refine.cpp
// If GR, etc. in the future requires additional flexiblity from non-refinement load
// balancing, we will need to use ptrs instead of references anyways, and add:

// void MeshBlock::SetHydroData(HydroBoundaryQuantity hydro_type)
//   Hydro *ph = pmy_block_->phydro;
//   // hard-coded assumption that, if multilevel, then Hydro is always present
//   // and enrolled in mesh refinement in the first pvars_cc_ vector entry
//   switch (hydro_type) {
//     case (HydroBoundaryQuantity::cons): {
//       pvars_cc_.front() = &ph->u;
//       break;
//     }
//     case (HydroBoundaryQuantity::prim): {
//       pvars_cc_.front() = &ph->w;
//       break;
//     }
//   }
//   return;
// }


//----------------------------------------------------------------------------------------
//! \fn bool MeshBlock::PointContained(Real const x, Real const y,
//                                     Real const z)
//  \brief Check whether a point is contained in the MeshBlock.
bool MeshBlock::PointContained(Real const x, Real const y, Real const z) {

  Real const mb_mi_x1 = block_size.x1min;
  Real const mb_ma_x1 = block_size.x1max;

  Real const mb_mi_x2 = block_size.x2min;
  Real const mb_ma_x2 = block_size.x2max;

  Real const mb_mi_x3 = block_size.x3min;
  Real const mb_ma_x3 = block_size.x3max;

  return ((mb_mi_x1 <= x) && (x <= mb_ma_x1) &&
          (mb_mi_x2 <= y) && (y <= mb_ma_x2) &&
          (mb_mi_x3 <= z) && (z <= mb_ma_x3));
}

//----------------------------------------------------------------------------------------
//! \fn bool MeshBlock::PointContainedExclusive(Real const x, Real const y,
//                                     Real const z)
//  \brief Check whether a point is exclusively contained in the MeshBlock.
//
// note: this function should be used for those points that are away from
// the computational grid boundaries as the upper limits are not included.
bool MeshBlock::PointContainedExclusive(Real const x, Real const y, Real const z) {

  Real const mb_mi_x1 = block_size.x1min;
  Real const mb_ma_x1 = block_size.x1max;

  Real const mb_mi_x2 = block_size.x2min;
  Real const mb_ma_x2 = block_size.x2max;

  Real const mb_mi_x3 = block_size.x3min;
  Real const mb_ma_x3 = block_size.x3max;

  return ((mb_mi_x1 <= x) && (x < mb_ma_x1) &&
          (mb_mi_x2 <= y) && (y < mb_ma_x2) &&
          (mb_mi_x3 <= z) && (z < mb_ma_x3));
}

//----------------------------------------------------------------------------------------
//! \fn bool MeshBlock::PointContainedExtended(
// Real const x, Real const y,
// Real const z)
//  \brief As in PointContained but also check ghost-layer
bool MeshBlock::PointContainedExtended(Real const x, Real const y, Real const z) {

  Real const mb_mi_x1 = pcoord->x1f(0);
  Real const mb_ma_x1 = pcoord->x1f(nverts1-1);

  Real const mb_mi_x2 = pcoord->x2f(0);
  Real const mb_ma_x2 = pcoord->x2f(nverts2-1);

  Real const mb_mi_x3 = pcoord->x3f(0);
  Real const mb_ma_x3 = pcoord->x3f(nverts3-1);

  return ((mb_mi_x1 <= x) && (x <= mb_ma_x1) &&
          (mb_mi_x2 <= y) && (y <= mb_ma_x2) &&
          (mb_mi_x3 <= z) && (z <= mb_ma_x3));
}

//----------------------------------------------------------------------------------------
//! \fn Real MeshBlock::PointCentralDistanceSquared(Real const x, Real const y,
//                                                  Real const z)
//  \brief Squared distance from center of MeshBlock to some point.
Real MeshBlock::PointCentralDistanceSquared(Real const x, Real const y,
                                            Real const z) {

  Real const mb_mi_x1 = block_size.x1min;
  Real const mb_ma_x1 = block_size.x1max;

  Real const mb_mi_x2 = block_size.x2min;
  Real const mb_ma_x2 = block_size.x2max;

  Real const mb_mi_x3 = block_size.x3min;
  Real const mb_ma_x3 = block_size.x3max;

  Real const mb_cx1 = mb_mi_x1 + (mb_ma_x1 - mb_mi_x1) / 2.;
  Real const mb_cx2 = mb_mi_x2 + (mb_ma_x2 - mb_mi_x2) / 2.;
  Real const mb_cx3 = mb_mi_x3 + (mb_ma_x3 - mb_mi_x3) / 2.;

  return SQR(mb_cx1 - x) + SQR(mb_cx2 - y) + SQR(mb_cx3 - z);

}

//----------------------------------------------------------------------------------------
//! \fn bool MeshBlock::SphereIntersects(
// Real const Sx0, Real const Sy0, Real const Sz0, Real const radius)
//  \brief Check if some sphere intersects current MeshBlock
bool MeshBlock::SphereIntersects(
  Real const Sx0, Real const Sy0, Real const Sz0, Real const radius) {

  // Check if center is contained in MeshBlock
  if (PointContained(Sx0, Sy0, Sz0))
    return true;

  // We require the MeshBlock vertices
  Real const mb_mi_x1 = block_size.x1min;
  Real const mb_ma_x1 = block_size.x1max;

  Real const mb_mi_x2 = block_size.x2min;
  Real const mb_ma_x2 = block_size.x2max;

  Real const mb_mi_x3 = block_size.x3min;
  Real const mb_ma_x3 = block_size.x3max;

  Real const Srad = SQR(radius);

  if ((SQR(mb_mi_x1 - Sx0) + SQR(mb_mi_x2 - Sy0) + SQR(mb_mi_x3 - Sz0)) < Srad)
    return true;

  if ((SQR(mb_ma_x1 - Sx0) + SQR(mb_mi_x2 - Sy0) + SQR(mb_mi_x3 - Sz0)) < Srad)
    return true;

  if ((SQR(mb_mi_x1 - Sx0) + SQR(mb_ma_x2 - Sy0) + SQR(mb_mi_x3 - Sz0)) < Srad)
    return true;

  if ((SQR(mb_mi_x1 - Sx0) + SQR(mb_mi_x2 - Sy0) + SQR(mb_ma_x3 - Sz0)) < Srad)
    return true;

  if ((SQR(mb_ma_x1 - Sx0) + SQR(mb_ma_x2 - Sy0) + SQR(mb_ma_x3 - Sz0)) < Srad)
    return true;

  if ((SQR(mb_mi_x1 - Sx0) + SQR(mb_ma_x2 - Sy0) + SQR(mb_ma_x3 - Sz0)) < Srad)
    return true;

  if ((SQR(mb_ma_x1 - Sx0) + SQR(mb_mi_x2 - Sy0) + SQR(mb_ma_x3 - Sz0)) < Srad)
    return true;

  if ((SQR(mb_ma_x1 - Sx0) + SQR(mb_ma_x2 - Sy0) + SQR(mb_mi_x3 - Sz0)) < Srad)
    return true;


  return false;

}

void MeshBlock::SetBoundaryVariablesConserved()
{

#if FLUID_ENABLED
  Hydro *ph = phydro;
  ph->hbvar.var_cc     = &(ph->u);
  ph->hbvar.coarse_buf = &(ph->coarse_cons_);
#endif

#if NSCALARS > 0
  pscalars->sbvar.var_cc     = &(pscalars->s);
  pscalars->sbvar.coarse_buf = &(pscalars->coarse_s_);
#endif

}

void MeshBlock::SetBoundaryVariablesPrimitive()
{

#if FLUID_ENABLED
  Hydro *ph = phydro;
  ph->hbvar.var_cc     = &(ph->w);
  ph->hbvar.coarse_buf = &(ph->coarse_prim_);
#endif

#if NSCALARS > 0
  pscalars->sbvar.var_cc     = &(pscalars->r);
  pscalars->sbvar.coarse_buf = &(pscalars->coarse_r_);
#endif

}

//
// :D
//
