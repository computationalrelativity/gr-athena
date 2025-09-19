// C++ standard headers
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <unistd.h>

// Athena++ headers
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/linear_algebra.hpp"     // Det. & friends

#if FLUID_ENABLED
#include "../hydro/hydro.hpp"
#endif // FLUID_ENABLED

#if MAGNETIC_FIELDS_ENABLED
#include "../field/field.hpp"
#endif // MAGNETIC_FIELDS_ENABLED

#if Z4C_ENABLED
#include "../z4c/z4c.hpp"
#endif // Z4C_ENABLED

#if M1_ENABLED
#include "../m1/m1.hpp"
#endif // M1_ENABLED

#include "trivialize_fields.hpp"

// External libraries

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// ============================================================================
namespace gra::trivialize {
// ============================================================================

Real get_oo_ms_sqrt_detgamma(MeshBlock * pmb, const int k, const int j, const int i)
{
#if defined(Z4C_VC_ENABLED)
  // not supported
  assert(false);
#endif

#if Z4C_ENABLED
  // avoid aliases
  return OO(
    pmb->pz4c->storage.aux_extended(
      Z4c::I_AUX_EXTENDED_ms_sqrt_detgamma,
      k, j, i
    )
  );
#endif
  return 1.0;
}

Real get_hydro_D(MeshBlock *pmb, const Real oo_ms_sqrt_detgamma,
                 const int k, const int j, const int i)
{
  Real ret = 0;
#if FLUID_ENABLED
  ret = oo_ms_sqrt_detgamma * pmb->phydro->u(IDN,k,j,i);
#endif

  return ret;
}

TrivializeFields::TrivializeFields(Mesh *pm, ParameterInput *pin) :
  pm (pm),
  pin (pin)
{
  // scrape settings ----------------------------------------------------------
  {
    opt.retain_nn_for_cut = pin->GetOrAddBoolean(
      "trivialize_fields", "retain_nn_for_cut", true
    );

    // hydro specific settings
    opt.hydro.active = pin->GetOrAddBoolean(
      "trivialize_fields", "hydro_use", false
    );
    opt.apply_on_substeps = pin->GetOrAddBoolean(
      "trivialize_fields", "apply_on_substeps", true
    );
    opt.hydro.set_vacuum = pin->GetOrAddBoolean(
      "trivialize_fields", "hydro_set_vacuum", false
    );
    opt.hydro.correct_layer = pin->GetOrAddBoolean(
      "trivialize_fields", "hydro_correct_layer", false
    );
    opt.hydro.cut_D = pin->GetOrAddReal(
      "trivialize_fields", "hydro_cut_D", 1.1e-14
    );

    opt.hydro.flux_fac = pin->GetOrAddReal(
      "trivialize_fields", "hydro_flux_fac", 1
    );

    opt.hydro.num_neighbors = pin->GetOrAddInteger(
      "trivialize_fields", "hydro_num_neighbors", 1
    );

    opt.hydro.num_neighbors_layer_extend = pin->GetOrAddInteger(
      "trivialize_fields", "hydro_num_neighbors_layer_extend", 0
    );

    // reduce to any active
    opt.active = opt.hydro.active;

    opt.verbose = pin->GetOrAddBoolean(
      "trivialize_fields", "verbose", true
    );
  }

}

void TrivializeFields::Update()
{
  if (!opt.active)
  {
    return;
  }

  PrepareMasks();

  if ((Globals::my_rank == 0) && opt.verbose)
  {
    std::printf("TrivializeFields:\n");
    // if (opt.hydro.active)
    // {
    //   std::printf("  hydro.r_star_D=%.3e\n", storage.hydro.r_star_D);
    // }
    // if (opt.magnetic.active)
    // {
    //   std::printf("  magnetic.r_star_b2=%.3e\n",
    //               storage.magnetic.r_star_b2);
    // }
    // if (opt.m1.active)
    // {
    //   std::printf("  m1.r_star_E=%.3e\n",
    //               storage.m1.r_star_E);
    // }
  }
}

void TrivializeFields::Update_()
{
  if (!opt.active)
  {
    return;
  }

  PrepareMasks();

  if ((Globals::my_rank == 0) && opt.verbose)
  {
    std::printf("TrivializeFields_post_tasklist:\n");
    // if (opt.hydro.active)
    // {
    //   std::printf("  hydro.r_star_D=%.3e\n", storage.hydro.r_star_D);
    // }
    // if (opt.magnetic.active)
    // {
    //   std::printf("  magnetic.r_star_b2=%.3e\n",
    //               storage.magnetic.r_star_b2);
    // }
    // if (opt.m1.active)
    // {
    //   std::printf("  m1.r_star_E=%.3e\n",
    //               storage.m1.r_star_E);
    // }
  }
}

void TrivializeFields::AllocateLocal(MeshBlock *pmb)
{
  // make and fill a default mask; used regardless of trivialization
  auto &tf = pmb->storage.trivialize_fields;

  if (opt.active)
  {
    tf.oo_ms_sqrt_detgamma.NewAthenaArray(
      pmb->ncells3, pmb->ncells2, pmb->ncells1
    );
  }

  if (opt.hydro.active)
  {
    AA_B & mask_nn_hydro = GetMaskHydroNN(pmb);
    AA_B & mask_pt_hydro = GetMaskHydroPT(pmb);
    AA_B & mask_ly_hydro = GetMaskHydroLY(pmb);

    mask_nn_hydro.NewAthenaArray(
      pmb->ncells3, pmb->ncells2, pmb->ncells1
    );
    mask_pt_hydro.NewAthenaArray(
      pmb->ncells3, pmb->ncells2, pmb->ncells1
    );
    mask_ly_hydro.NewAthenaArray(
      pmb->ncells3, pmb->ncells2, pmb->ncells1
    );


    // fill defaults
    // mask_nn_hydro.Fill(true);
    // mask_pt_hydro.Fill(true);
    // mask_ly_hydro.Fill(false);
  }
}

bool NeighborhoodUnderCut(MeshBlock * pmb,
                          const bool densitized,
                          const Real cut,
                          AA & field,
                          const int n,
                          const int k, const int j, const int i,
                          const int num_neighbors)
{
  bool ret = true;
  auto &tf = pmb->storage.trivialize_fields;

  for (int kk = -num_neighbors; kk <= num_neighbors; ++kk)
  for (int jj = -num_neighbors; jj <= num_neighbors; ++jj)
  for (int ii = -num_neighbors; ii <= num_neighbors; ++ii)
  {
    const int i_ix = i + ii;
    const int j_ix = j + jj;
    const int k_ix = k + kk;

    if ((i_ix < 0) || (i_ix > pmb->ncells1-1))
      continue;

    if ((j_ix < 0) || (j_ix > pmb->ncells2-1))
      continue;

    if ((k_ix < 0) || (k_ix > pmb->ncells3-1))
      continue;

    Real oo_sqrt_det_gamma = (densitized)
      ? tf.oo_ms_sqrt_detgamma(k_ix,j_ix,i_ix)
      : 1.0;
    ret = ret && (oo_sqrt_det_gamma * field(n,k_ix,j_ix,i_ix) < cut);

    if (!ret)
      break;
  }

  return ret;
}

void TrivializeFields::Reweight()
{
  if (!opt.active)
  {
    return;
  }

  int nthreads = pm->GetNumMeshThreads();
  (void)nthreads;

  std::vector<MeshBlock*> pmb_array;
  pm->GetMeshBlocksMyRank(pmb_array);
  const int nmb = pmb_array.size();

  #pragma omp parallel for num_threads(nthreads)
  for (int ix = 0; ix < nmb; ++ix)
  {
    MeshBlock *pmb = pmb_array[ix];

    auto &tf = pmb->storage.trivialize_fields;

    AA & oo_ms_sqrt_detgamma = tf.oo_ms_sqrt_detgamma;

    AA_B & mask_nn_hydro = tf.hydro.mask_nn;

    CC_GLOOP2(k, j)
    for (int i = 0; i <= pmb->ncells1-1; ++i)
    {
      if (opt.hydro.active)
      {
        if (mask_nn_hydro(k,j,i))
        {
          const Real ms_sqrt_detgamma = OO(get_oo_ms_sqrt_detgamma(pmb,k,j,i));

          for (int n=0; n<NHYDRO; ++n)
          {
            pmb->phydro->u(IDN,k,j,i) *= oo_ms_sqrt_detgamma(k,j,i);
            pmb->phydro->u(IDN,k,j,i) *= ms_sqrt_detgamma;
          }
        }
      }
    }
  }
}

void TrivializeFields::PrepareMask(MeshBlock *pmb)
{
  auto &tf = pmb->storage.trivialize_fields;

  AA_B & mask_nn_hydro = tf.hydro.mask_nn;
  AA_B & mask_pt_hydro = tf.hydro.mask_pt;
  AA_B & mask_ly_hydro = tf.hydro.mask_ly;

  bool & is_trivial_hydro = tf.hydro.is_trivial;

  AA & oo_ms_sqrt_detgamma = tf.oo_ms_sqrt_detgamma;
  CC_GLOOP3(k, j, i)
  {
    oo_ms_sqrt_detgamma(k,j,i) = get_oo_ms_sqrt_detgamma(pmb,k,j,i);
  }

  // Check whole block:
  // diffusion / evo will leak data into block thus activating if frozen
  //
  // no simd here, otherwise need local + reduction
  is_trivial_hydro = true;

  if (opt.hydro.active)
  CC_GLOOP2(k, j)
  for (int i = 0; i <= pmb->ncells1-1; ++i)
  {
    const Real D = oo_ms_sqrt_detgamma(k,j,i) * pmb->phydro->u(IDN,k,j,i);
    const bool pt_trivial = D < opt.hydro.cut_D;

    const bool nn_trivial = NeighborhoodUnderCut(
      pmb, true, opt.hydro.cut_D, pmb->phydro->u,
      IDN, k, j, i, opt.hydro.num_neighbors
    );

    mask_pt_hydro(k,j,i) = !pt_trivial;
    mask_nn_hydro(k,j,i) = !nn_trivial;

    is_trivial_hydro = is_trivial_hydro && nn_trivial;
  }

  // construct layer mask
  if (opt.hydro.active)
  {
    const int nn_layer = opt.hydro.num_neighbors_layer_extend;

    CC_GLOOP2(k, j)
    for (int i = 0; i <= pmb->ncells1-1; ++i)
    {
      bool is_layer = mask_nn_hydro(k,j,i) && !mask_pt_hydro(k,j,i);

      if (is_layer)
      {
        for (int kk = -nn_layer; kk <= nn_layer; ++kk)
        for (int jj = -nn_layer; jj <= nn_layer; ++jj)
        for (int ii = -nn_layer; ii <= nn_layer; ++ii)
        {
          const int i_ix = i + ii;
          const int j_ix = j + jj;
          const int k_ix = k + kk;

          if ((i_ix < 0) || (i_ix > pmb->ncells1-1))
            continue;

          if ((j_ix < 0) || (j_ix > pmb->ncells2-1))
            continue;

          if ((k_ix < 0) || (k_ix > pmb->ncells3-1))
            continue;

          mask_ly_hydro(k_ix,j_ix,i_ix) = true;
        }
      }
    }
  }


}

bool TrivializeFields::CutMaskHydro(
  MeshBlock *pmb,
  const int k, const int j, const int i)
{
  if (opt.hydro.active && opt.hydro.set_vacuum)
  {
    // AA_B & mask_nn_hydro = GetMaskHydroNN(pmb);
    // AA_B & mask_pt_hydro = GetMaskHydroPT(pmb);

    AA_B & mask_hydro = (opt.retain_nn_for_cut)
      ? GetMaskHydroNN(pmb)
      : GetMaskHydroPT(pmb);

    Hydro * ph = pmb->phydro;
    PassiveScalars *ps = pmb->pscalars;
    EquationOfState *peos = pmb->peos;

    if (!mask_hydro(k,j,i))
    {
      for (int n=0; n<NHYDRO; ++n)
      {
        ph->u(n,k,j,i) = 0;
        ph->w(n,k,j,i) = 0;
      }

      for (int n=0; n<NSCALARS; ++n)
      {
        ps->r(n,k,j,i) = (
          pmb->peos->GetEOS().GetSpeciesAtmosphere(n)
        );
        ps->s(n,k,j,i) = 0;
      }

      // BD: TODO- confirm this is the way to set derived quantities for cut
      //           points. Cross-check EquationOfState::DerivedQuantities
      ph->derived_ms(IX_T,k,j,i) = 0; // peos->GetEOS().GetTemperatureFloor();
      ph->derived_ms(IX_LOR,k,j,i) = 1.0;
      ph->derived_ms(IX_ETH,k,j,i) = 0; // peos->GetEOS().GetMinimumEnthalpy();

      return true;
    }
  }
  return false;
}

void TrivializeFields::CutMask(MeshBlock *pmb)
{
  if (opt.hydro.active && opt.hydro.set_vacuum)
  {
    CC_GLOOP2(k, j)
    for (int i = 0; i <= pmb->ncells1-1; ++i)
    {
      CutMaskHydro(pmb,k,j,i);
    }
  }
}

void TrivializeFields::PrepareMasks()
{
  int nthreads = pm->GetNumMeshThreads();
  (void)nthreads;

  std::vector<MeshBlock*> pmb_array;
  pm->GetMeshBlocksMyRank(pmb_array);
  const int nmb = pmb_array.size();

  #pragma omp parallel for num_threads(nthreads)
  for (int ix = 0; ix < nmb; ++ix)
  {
    MeshBlock *pmb = pmb_array[ix];
    PrepareMask(pmb);
  }
}

void TrivializeFields::CutMasks()
{
  int nthreads = pm->GetNumMeshThreads();
  (void)nthreads;

  std::vector<MeshBlock*> pmb_array;
  pm->GetMeshBlocksMyRank(pmb_array);
  const int nmb = pmb_array.size();

  #pragma omp parallel for num_threads(nthreads)
  for (int ix = 0; ix < nmb; ++ix)
  {
    MeshBlock *pmb = pmb_array[ix];
    CutMask(pmb);
  }
}

bool TrivializeFields::IsTrivialHydro(MeshBlock * pmb)
{
  return pmb->storage.trivialize_fields.hydro.is_trivial;
}

bool TrivializeFields::IsTrivialMatter(MeshBlock * pmb)
{
  // BD: TODO- fix for magnetic fields
  return IsTrivialHydro(pmb);
}

bool TrivializeFields::MaskHydro(MeshBlock * pmb,
                                 const int k, const int j, const int i)
{
  assert(false);
  // const bool is_trivial = (
  //   pmb->storage.trivialize_fields.hydro.is_trivial &&
  //   opt.hydro.active && // mask needs to be allocated
  //   !pmb->storage.trivialize_fields.hydro.mask(k,j,i) // mask true if non-triv.
  // );
  // return !is_trivial;
}

/*
bool TrivializeFields::MaskMatter(MeshBlock * pmb,
                                  const int k, const int j, const int i)
{
  assert(false);
  // return MaskHydro(pmb, k, j, i);
}

AA_B & TrivializeFields::GetMatterMaskEvolve(MeshBlock *pmb)
{
  return pmb->storage.trivialize_fields.matter.mask_evolve;
}

AA_B & TrivializeFields::GetMatterMaskFlux(MeshBlock *pmb)
{
  return pmb->storage.trivialize_fields.matter.mask_flux;
}
*/

AA_B & TrivializeFields::GetMaskHydroPT(MeshBlock *pmb)
{
  return pmb->storage.trivialize_fields.hydro.mask_pt;
}
AA_B & TrivializeFields::GetMaskHydroNN(MeshBlock *pmb)
{
  return pmb->storage.trivialize_fields.hydro.mask_nn;
}
AA_B & TrivializeFields::GetMaskHydroLY(MeshBlock *pmb)
{
  return pmb->storage.trivialize_fields.hydro.mask_ly;
}

// ============================================================================
} // namespace gra::trivialize
// ============================================================================

//
// :D
//