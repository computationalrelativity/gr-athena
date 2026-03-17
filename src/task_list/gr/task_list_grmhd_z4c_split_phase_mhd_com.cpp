// C/C++ headers
#include <cmath>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../eos/eos.hpp"
#include "../../field/field.hpp"
#include "../../hydro/hydro.hpp"
#include "../../mesh/mesh.hpp"
#include "../../z4c/z4c.hpp"
#include "../../z4c/wave_extract.hpp"
#include "../../z4c/puncture_tracker.hpp"
#include "../../trackers/extrema_tracker.hpp"
#include "../../reconstruct/reconstruction.hpp"
#include "../../scalars/scalars.hpp"
#include "../../utils/linear_algebra.hpp"
#include "../../comm/comm_registry.hpp"

#include "task_list.hpp"
#include "task_names.hpp"

// #if M1_ENABLED
// #include "../../m1/m1.hpp"
// #endif

// ----------------------------------------------------------------------------
using namespace TaskLists::GeneralRelativity;
using namespace TaskLists::Integrators;

using namespace gra::triggers;
typedef Triggers::TriggerVariant TriggerVariant;
// ----------------------------------------------------------------------------

GRMHD_Z4c_Phase_MHD_com::GRMHD_Z4c_Phase_MHD_com(ParameterInput *pin,
                                                 Mesh *pm,
                                                 Triggers &trgs)
  : LowStorage(pin, pm),
    trgs(trgs)
{
  using namespace TaskNames::GeneralRelativity::GRMHD_Z4c_Split::Phase_MHD_com;

  // Take the number of stages from the integrator
  nstages = LowStorage::nstages;

  //---------------------------------------------------------------------------

  const bool multilevel = pm->multilevel;  // for SMR or AMR logic
  const bool adaptive   = pm->adaptive;    // AMR

  Add(CONS2PRIMP, NONE, &GRMHD_Z4c_Phase_MHD_com::PrimitivesPhysical);

  Add(SEND_HYD, CONS2PRIMP, &GRMHD_Z4c_Phase_MHD_com::SendHydro);
  Add(RECV_HYD, NONE,       &GRMHD_Z4c_Phase_MHD_com::ReceiveHydro);

  Add(SETB_HYD, RECV_HYD, &GRMHD_Z4c_Phase_MHD_com::SetBoundariesHydro);

  if (NSCALARS > 0)
  {
    Add(SEND_SCLR, CONS2PRIMP, &GRMHD_Z4c_Phase_MHD_com::SendScalars);
    Add(RECV_SCLR, NONE,       &GRMHD_Z4c_Phase_MHD_com::ReceiveScalars);

    Add(SETB_SCLR, RECV_SCLR, &GRMHD_Z4c_Phase_MHD_com::SetBoundariesScalars);
  }

  if (MAGNETIC_FIELDS_ENABLED)
  {
    Add(SEND_FLD, CONS2PRIMP, &GRMHD_Z4c_Phase_MHD_com::SendField);
    Add(RECV_FLD, NONE,       &GRMHD_Z4c_Phase_MHD_com::ReceiveField);

    Add(SETB_FLD, RECV_FLD,   &GRMHD_Z4c_Phase_MHD_com::SetBoundariesField);
  }

  // ensure all boundary buffers are set --------------------------------------
  TaskID BLOCK_SETB = SETB_HYD | SEND_HYD;

  if (NSCALARS > 0)
    BLOCK_SETB = BLOCK_SETB | SETB_SCLR | SEND_SCLR;

  if (MAGNETIC_FIELDS_ENABLED)
    BLOCK_SETB = BLOCK_SETB | SETB_FLD | SEND_FLD;
  // --------------------------------------------------------------------------

  if (multilevel)
  {
    Add(PROLONG_HYD,  BLOCK_SETB,
        &GRMHD_Z4c_Phase_MHD_com::Prolongation_Hyd);
    Add(PHY_BVAL_HYD, PROLONG_HYD,
        &GRMHD_Z4c_Phase_MHD_com::PhysicalBoundary_Hyd);
  }
  else
  {
    Add(PHY_BVAL_HYD, BLOCK_SETB,
        &GRMHD_Z4c_Phase_MHD_com::PhysicalBoundary_Hyd);
  }

  // We are done for the (M)HD phase
  Add(CLEAR_ALLBND, PHY_BVAL_HYD, &GRMHD_Z4c_Phase_MHD_com::ClearAllBoundary);

}

// ----------------------------------------------------------------------------
void GRMHD_Z4c_Phase_MHD_com::StartupTaskList(MeshBlock *pmb, int stage)
{
  if (stage == 1)
  {
    // Initialize time abscissae (also done in Phase_MHD; repeated here for
    // consistency in case task lists are ever assembled separately)
    PrepareStageAbscissae(stage, pmb);
  }

  // Post persistent receives for all MainInt channels (hydro, field, scalars).
  pmb->pcomm->StartReceiving(comm::CommGroup::MainInt);
  return;
}

//-----------------------------------------------------------------------------
// Functions to end MPI communication
TaskStatus GRMHD_Z4c_Phase_MHD_com::ClearAllBoundary(MeshBlock *pmb, int stage)
{
  // Wait on outstanding sends and reset channel flags for all MainInt channels.
  pmb->pcomm->ClearBoundary(comm::CommGroup::MainInt);
  return TaskStatus::next;
}

//-----------------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Phase_MHD_com::PrimitivesPhysical(
  MeshBlock *pmb, int stage)
{
  // Construct primitives from conserved on the whole MeshBlock.
  if (stage <= nstages)
  {
    Hydro *ph = pmb->phydro;
    Field *pf = pmb->pfield;
    PassiveScalars *ps = pmb->pscalars;
    EquationOfState *peos = pmb->peos;

    int il = pmb->is, iu = pmb->ie;
    int jl = pmb->js, ju = pmb->je;
    int kl = pmb->ks, ku = pmb->ke;

    // Recompute cell-centred B from the (already updated) face-centred field
    // on the physical interior so that C2P sees a bcc consistent with the
    // current stage.  The global bcc refresh happens later in
    // PhysicalBoundary_Hyd, but PrimitivesPhysical runs before that task.
    if (MAGNETIC_FIELDS_ENABLED)
    {
      pf->CalculateCellCenteredField(pf->b, pf->bcc, pmb->pcoord,
                                     il, iu, jl, ju, kl, ku);
    }

    static const int coarseflag = 0;
    peos->ConservedToPrimitive(ph->u, ph->w1, ph->w,
                               ps->s, ps->r,
                               pf->bcc, pmb->pcoord,
                               il, iu, jl, ju, kl, ku,
                               coarseflag);

    // Update w1 to have the state of w
    ph->RetainState(ph->w1, ph->w, il, iu, jl, ju, kl, ku);
    return TaskStatus::next;
  }

  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Ghost-zone exchange: pack and send all MainInt channels (hydro, field, scalars).
TaskStatus GRMHD_Z4c_Phase_MHD_com::SendHydro(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    // Group-level send handles all MainInt channels (hydro, field, scalars).
    // Restriction into coarse buffers (for cross-level neighbors) is embedded
    // inside SendBoundaryBuffers when mesh is multilevel.
    pmb->pcomm->SendBoundaryBuffers(comm::CommGroup::MainInt);
    return TaskStatus::next;
  }

  return TaskStatus::fail;
}

TaskStatus GRMHD_Z4c_Phase_MHD_com::SendField(MeshBlock *pmb, int stage)
{
  // Field is sent as part of the MainInt group in SendHydro.
  if (stage <= nstages) return TaskStatus::next;
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Ghost-zone exchange: poll for received data.
TaskStatus GRMHD_Z4c_Phase_MHD_com::ReceiveHydro(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    // Poll all MainInt channels.  Returns true when every channel has received
    // all expected messages from same-level, coarser, and finer neighbors.
    bool done = pmb->pcomm->ReceiveBoundaryBuffers(comm::CommGroup::MainInt);
    if (done) return TaskStatus::next;
    return TaskStatus::fail;
  }

  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c_Phase_MHD_com::ReceiveField(MeshBlock *pmb, int stage)
{
  // Field is received as part of the MainInt group in ReceiveHydro.
  if (stage <= nstages) return TaskStatus::next;
  return TaskStatus::fail;
}

TaskStatus GRMHD_Z4c_Phase_MHD_com::SetBoundariesHydro(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    // Unpack received data for all MainInt channels into state arrays.
    pmb->pcomm->SetBoundaries(comm::CommGroup::MainInt);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c_Phase_MHD_com::SetBoundariesField(MeshBlock *pmb, int stage)
{
  // Field is unpacked as part of the MainInt group in SetBoundariesHydro.
  if (stage <= nstages) return TaskStatus::next;
  return TaskStatus::fail;
}

TaskStatus GRMHD_Z4c_Phase_MHD_com::SendScalars(MeshBlock *pmb, int stage)
{
  // Scalars are sent as part of the MainInt group in SendHydro.
  if (stage <= nstages) return TaskStatus::next;
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c_Phase_MHD_com::ReceiveScalars(MeshBlock *pmb, int stage)
{
  // Scalars are received as part of the MainInt group in ReceiveHydro.
  if (stage <= nstages) return TaskStatus::next;
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c_Phase_MHD_com::SetBoundariesScalars(MeshBlock *pmb, int stage)
{
  // Scalars are unpacked as part of the MainInt group in SetBoundariesHydro.
  if (stage <= nstages) return TaskStatus::next;
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Functions for everything else
TaskStatus GRMHD_Z4c_Phase_MHD_com::Prolongation_Hyd(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    comm::CommRegistry *pcomm = pmb->pcomm;

    // For re-scatter (stage < 1): use current time and zero dt (no time
    // interpolation needed). Avoids out-of-bounds access on stage_wghts[-1].
    // Mirrors the pattern in Mesh::FinalizeHydroConsRP.
    const Real t_end     = (stage >= 1) ? this->t_end(stage, pmb)
                                        : pmb->pmy_mesh->time;
    const Real dt_scaled = (stage >= 1) ? this->dt_scaled(stage, pmb)
                                        : 0.0;

    // Coarse-level index range (from MeshBlock).
    const int cis = pmb->cis, cie = pmb->cie;
    const int cjs = pmb->cjs, cje = pmb->cje;
    const int cks = pmb->cks, cke = pmb->cke;

    // Apply coarse-level physical BCs then prolongate each MainInt channel.
    pcomm->ProlongateAndApplyPhysicalBCs(
        comm::CommGroup::MainInt, t_end, dt_scaled,
        cis, cie, cjs, cje, cks, cke, NGHOST);

    // Recompute cell-centred B from face-centred field on the prolongated fine
    // ghost-zone slabs so that bcc is consistent with the divergence-preserving
    // prolongation of B.
    if (MAGNETIC_FIELDS_ENABLED)
      pmb->CalculateCellCenteredFieldOnProlongedBoundaries();

    return TaskStatus::next;
  }

  return TaskStatus::fail;
}

TaskStatus GRMHD_Z4c_Phase_MHD_com::PhysicalBoundary_Hyd(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Field *pf = pmb->pfield;
    comm::CommRegistry *pcomm = pmb->pcomm;

    // For re-scatter (stage < 1): use current time and zero dt (no time
    // interpolation needed). Avoids out-of-bounds access on stage_wghts[-1].
    // Mirrors the pattern in Mesh::FinalizeHydroConsRP.
    const Real t_end     = (stage >= 1) ? this->t_end(stage, pmb)
                                        : pmb->pmy_mesh->time;
    const Real dt_scaled = (stage >= 1) ? this->dt_scaled(stage, pmb)
                                        : 0.0;

    // Apply fine-level physical BCs for every MainInt channel.
    pcomm->ApplyPhysicalBCs(comm::CommGroup::MainInt, t_end, dt_scaled);

    // Recompute bcc globally (ghost zones now filled by physical BCs above).
    if (MAGNETIC_FIELDS_ENABLED)
    {
      pf->CalculateCellCenteredField(pf->b,
                                     pf->bcc,
                                     pmb->pcoord,
                                     0, pmb->ncells1-1,
                                     0, pmb->ncells2-1,
                                     0, pmb->ncells3-1);
    }

    return TaskStatus::next;
  }

  return TaskStatus::fail;
}

//
// :D
//
