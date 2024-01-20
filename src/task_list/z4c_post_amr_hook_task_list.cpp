// C/C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ classes headers
#include "../athena.hpp"
#include "../bvals/bvals.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"
#include "../z4c/z4c_macro.hpp"
#include "../z4c/wave_extract.hpp"
#include "../parameter_input.hpp"
#include "task_list.hpp"


//----------------------------------------------------------------------------------------
//  Z4cPostAMRTaskList constructor

Z4cPostAMRTaskList::Z4cPostAMRTaskList(ParameterInput *pin, Mesh *pm){

  nstages = 1;

  {
    using namespace Z4cPostAMRTaskNames;

    // storage.u contains z4c (populated via R/P)

    // Enforce algebraic constraints (post-R/P)
    AddTask(ALG_CONSTR, NONE);

    // Map geometric description to ADM
    AddTask(Z4C_TO_ADM, ALG_CONSTR);

    if (FLUID_ENABLED || MAGNETIC_FIELDS_ENABLED)
    {
      // ADM sources need updating (fluid populated via R/P)
      AddTask(UPDATE_SRC, Z4C_TO_ADM);
      AddTask(ADM_CONSTR, UPDATE_SRC);
    }
    else
    {
      // vacuum
      AddTask(ADM_CONSTR, Z4C_TO_ADM);
    }

    // Recompute Weyl (strictly only needed for 3d dump)
    AddTask(Z4C_WEYL, Z4C_TO_ADM);

  } // end of using namespace block

}

//---------------------------------------------------------------------------------------
//  Sets id and dependency for "ntask" member of task_list_ array, then iterates value of
//  ntask.

void Z4cPostAMRTaskList::AddTask(const TaskID& id, const TaskID& dep) {
    task_list_[ntasks].task_id = id;
    task_list_[ntasks].dependency = dep;

    using namespace Z4cPostAMRTaskNames; // NOLINT (build/namespace)

    if (id == NOP)
    {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cPostAMRTaskList::Nop);
      task_list_[ntasks].lb_time = false;
    }
    else if (id == ALG_CONSTR)
    {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cPostAMRTaskList::EnforceAlgConstr);
      task_list_[ntasks].lb_time = true;
    }
    else if (id == Z4C_TO_ADM)
    {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cPostAMRTaskList::Z4cToADM);
      task_list_[ntasks].lb_time = true;
    }
    else if (id == UPDATE_SRC)
    {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cPostAMRTaskList::UpdateSource);
      task_list_[ntasks].lb_time = true;
    }
    else if (id == ADM_CONSTR)
    {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cPostAMRTaskList::ADM_Constraints);
      task_list_[ntasks].lb_time = true;
    }
    else if (id == Z4C_WEYL)
    {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cPostAMRTaskList::Z4c_Weyl);
      task_list_[ntasks].lb_time = true;
    }
    else
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in AddTask" << std::endl
          << "Invalid Task is specified" << std::endl;
      ATHENA_ERROR(msg);
    }

    ntasks++;
    return;
}


void Z4cPostAMRTaskList::StartupTaskList(MeshBlock *pmb, int stage)
{
  return;
}

TaskStatus Z4cPostAMRTaskList::Nop(MeshBlock *pmb, int stage)
{
  std::cout << "GetMatter, ADMConstraints" << std::endl;

  Hydro *ph = pmb->phydro;
  Z4c *pz4c = pmb->pz4c;

  // UpdateSource
  pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, ph->w, pmb->pfield->bcc);

  // ADM_Constraints
  pz4c->ADMConstraints(pz4c->storage.con, pz4c->storage.adm,
                       pz4c->storage.mat, pz4c->storage.u);

  return TaskStatus::success;
}


TaskStatus Z4cPostAMRTaskList::EnforceAlgConstr(MeshBlock *pmb, int stage)
{
  Z4c *pz4c = pmb->pz4c;
  pz4c->AlgConstr(pz4c->storage.u);
  return TaskStatus::success;
}

TaskStatus Z4cPostAMRTaskList::Z4cToADM(MeshBlock *pmb, int stage)
{
  Z4c *pz4c = pmb->pz4c;
  pz4c->Z4cToADM(pz4c->storage.u, pz4c->storage.adm);
  return TaskStatus::success;
}

TaskStatus Z4cPostAMRTaskList::UpdateSource(MeshBlock *pmb, int stage)
{
  Hydro *ph = pmb->phydro;
  Z4c *pz4c = pmb->pz4c;

  // UpdateSource
  pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, ph->w, pmb->pfield->bcc);

  return TaskStatus::success;
}

TaskStatus Z4cPostAMRTaskList::ADM_Constraints(MeshBlock *pmb, int stage)
{
  Z4c *pz4c = pmb->pz4c;

  pz4c->Z4cToADM(pz4c->storage.u, pz4c->storage.adm);

  pz4c->GetMatter(pz4c->storage.mat,
                  pz4c->storage.adm,
                  pmb->phydro->w,
                  pmb->pfield->bcc);

  pz4c->ADMConstraints(pz4c->storage.con, pz4c->storage.adm,
                       pz4c->storage.mat, pz4c->storage.u);

  return TaskStatus::success;
}

TaskStatus Z4cPostAMRTaskList::Z4c_Weyl(MeshBlock *pmb, int stage)
{
  Z4c *pz4c = pmb->pz4c;

  pz4c->Z4cWeyl(pz4c->storage.adm,
                pz4c->storage.mat,
                pz4c->storage.weyl);
  return TaskStatus::success;
}

//
// :D
//
