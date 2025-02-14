// C/C++ headers
#include <cmath>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../bvals/bvals.hpp"
#include "../../eos/eos.hpp"
#include "../../field/field.hpp"
#include "../../hydro/hydro.hpp"
#include "../../hydro/srcterms/hydro_srcterms.hpp"
#include "../../mesh/mesh.hpp"
#include "../../z4c/z4c.hpp"
#include "../../z4c/wave_extract.hpp"
#include "../../z4c/puncture_tracker.hpp"
#include "../../trackers/extrema_tracker.hpp"
#include "../../reconstruct/reconstruction.hpp"
#include "../../scalars/scalars.hpp"
#include "../../utils/linear_algebra.hpp"

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
  BoundaryValues *pb   = pmb->pbval;
  Z4c            *pz4c = pmb->pz4c;

  if (stage == 1)
  {
    // Initialize time abscissae
    PrepareStageAbscissae(stage, pmb);
  }

  pb->StartReceiving(BoundaryCommSubset::matter);
  return;
}

//-----------------------------------------------------------------------------
// Functions to end MPI communication
TaskStatus GRMHD_Z4c_Phase_MHD_com::ClearAllBoundary(MeshBlock *pmb, int stage)
{
  BoundaryValues *pb = pmb->pbval;
  pb->ClearBoundary(BoundaryCommSubset::matter);

  // pmb->DebugMeshBlock(-15,-15,-15, 2, 20, 3, "@T:MHD_com\n", "@E:MHD_com\n");
  return TaskStatus::success;
}

//-----------------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Phase_MHD_com::PrimitivesPhysical(
  MeshBlock *pmb, int stage)
{
  // Construct primitives from conserved.
  // In the case of `cons_bc` the whole MeshBlock is populated
  // otherwise only points on the interior of the computational domain are
  // populated
  if (stage <= nstages)
  {
    Hydro *ph = pmb->phydro;
    Field *pf = pmb->pfield;
    PassiveScalars *ps = pmb->pscalars;
    BoundaryValues *pb = pmb->pbval;
    EquationOfState *peos = pmb->peos;

    int il = pmb->is, iu = pmb->ie;
    int jl = pmb->js, ju = pmb->je;
    int kl = pmb->ks, ku = pmb->ke;

    static const int coarseflag = 0;
    peos->ConservedToPrimitive(ph->u, ph->w1, ph->w,
                               ps->s, ps->r,
                               pf->bcc, pmb->pcoord,
                               il, iu, jl, ju, kl, ku,
                               coarseflag);

    // Update w1 to have the state of w
    ph->RetainState(ph->w1, ph->w, il, iu, jl, ju, kl, ku);
    return TaskStatus::success;
  }

  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Functions to communicate conserved variables between MeshBlocks
TaskStatus GRMHD_Z4c_Phase_MHD_com::SendHydro(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro * ph = pmb->phydro;

    // Swap Hydro quantity in BoundaryVariable interface back to conserved var
    // formulation (also needed in SetBoundariesHydro(), since the tasks are
    // independent)
#ifndef DBG_USE_CONS_BC
    pmb->SetBoundaryVariablesConserved();
#endif
    ph->hbvar.SendBoundaryBuffers();

    return TaskStatus::success;
  }

  return TaskStatus::fail;
}

TaskStatus GRMHD_Z4c_Phase_MHD_com::SendField(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Field * pf = pmb->pfield;

    pf->fbvar.SendBoundaryBuffers();
    return TaskStatus::success;
  }

  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Functions to receive conserved variables between MeshBlocks
TaskStatus GRMHD_Z4c_Phase_MHD_com::ReceiveHydro(MeshBlock *pmb, int stage)
{
  bool ret;
  if (stage <= nstages)
  {
    ret = pmb->phydro->hbvar.ReceiveBoundaryBuffers();
  }
  else
  {
    return TaskStatus::fail;
  }

  if (ret) {
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
}


TaskStatus GRMHD_Z4c_Phase_MHD_com::ReceiveField(MeshBlock *pmb, int stage)
{
  bool ret;
  if (stage <= nstages) {
    ret = pmb->pfield->fbvar.ReceiveBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  if (ret) {
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
}

TaskStatus GRMHD_Z4c_Phase_MHD_com::SetBoundariesHydro(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro * ph = pmb->phydro;

#ifndef DBG_USE_CONS_BC
    pmb->SetBoundaryVariablesConserved();
#endif  // DBG_USE_CONS_BC
    ph->hbvar.SetBoundaries();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c_Phase_MHD_com::SetBoundariesField(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Field *pf = pmb->pfield;

    pf->fbvar.SetBoundaries();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

TaskStatus GRMHD_Z4c_Phase_MHD_com::SendScalars(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    PassiveScalars * ps = pmb->pscalars;

    // Swap PassiveScalars quantity in BoundaryVariable interface back to
    // conserved var formulation (also needed in SetBoundariesScalars() since
    // the tasks are independent)
#ifndef DBG_USE_CONS_BC
    pmb->SetBoundaryVariablesConserved();
#endif // DBG_USE_CONS_BC
    ps->sbvar.SendBoundaryBuffers();
  }
  else
  {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c_Phase_MHD_com::ReceiveScalars(MeshBlock *pmb, int stage)
{
  bool ret;
  if (stage <= nstages)
  {
    PassiveScalars * ps = pmb->pscalars;
    ret = ps->sbvar.ReceiveBoundaryBuffers();
  }
  else
  {
    return TaskStatus::fail;
  }

  if (ret)
  {
    return TaskStatus::success;
  }
  else
  {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c_Phase_MHD_com::SetBoundariesScalars(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    PassiveScalars * ps = pmb->pscalars;

    // Set PassiveScalars quantity in BoundaryVariable interface to cons var
    // formulation
#ifndef DBG_USE_CONS_BC
    pmb->SetBoundaryVariablesConserved();
#endif // DBG_USE_CONS_BC
    ps->sbvar.SetBoundaries();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Functions for everything else
TaskStatus GRMHD_Z4c_Phase_MHD_com::Prolongation_Hyd(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    BoundaryValues * pb = pmb->pbval;

    const Real t_end = this->t_end(stage, pmb);
    const Real dt_scaled = this->dt_scaled(stage, pmb);

    pb->ProlongateBoundariesHydro(t_end, dt_scaled);

    return TaskStatus::success;
  }

  return TaskStatus::fail;
}

TaskStatus GRMHD_Z4c_Phase_MHD_com::PhysicalBoundary_Hyd(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Field *pf = pmb->pfield;
    Hydro *ph = pmb->phydro;
    PassiveScalars *ps = pmb->pscalars;
    BoundaryValues *pbval = pmb->pbval;

    const Real t_end = this->t_end(stage, pmb);
    const Real dt_scaled = this->dt_scaled(stage, pmb);

    // Swap Hydro and (possibly) passive scalar quantities in BoundaryVariable
    // interface from conserved to primitive formulations:
#ifndef DBG_USE_CONS_BC
    pmb->SetBoundaryVariablesPrimitive();
#endif // DBG_USE_CONS_BC

    // Apply boundary conditions on either prim or con
    pbval->ApplyPhysicalBoundaries(
      t_end, dt_scaled,
      pbval->GetBvarsMatter(),
      pmb->is, pmb->ie,
      pmb->js, pmb->je,
      pmb->ks, pmb->ke,
      NGHOST);

    // Compute bcc globally
    if (MAGNETIC_FIELDS_ENABLED)
    {
      pf->CalculateCellCenteredField(pf->b,
                                     pf->bcc,
                                     pmb->pcoord,
                                     0, pmb->ncells1-1,
                                     0, pmb->ncells2-1,
                                     0, pmb->ncells3-1);
    }

    // Compute conserved fields in the boundary if required
#ifndef DBG_USE_CONS_BC
    pbval->PrimitiveToConservedOnPhysicalBoundaries();
    pmb->SetBoundaryVariablesConserved();
#endif // DBG_USE_CONS_BC

    return TaskStatus::success;
  }

  return TaskStatus::fail;
}

//
// :D
//
