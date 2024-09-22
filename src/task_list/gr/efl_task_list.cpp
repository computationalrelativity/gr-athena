//! \file efl_task_list.cpp
//! \brief function implementation for EFLTaskList

// C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../../athena.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../task_list.hpp"
#include "efl_task_list.hpp"
#include "../../hydro/hydro.hpp"

//----------------------------------------------------------------------------------------
//! EntropFluxLimiterTaskList constructor

EFLTaskList::EFLTaskList(ParameterInput *pin, Mesh *pm){

    std::string integrator;
#if 0
    //if (efl_choice)
    //{
    //  integrator = pin->GetOrAddString("time", "integrator", "vl2");
    //  if      (integrator == "vl2") {nstages = 2};
    //  else if (integrator == "rk1") {nstages = 1};
      else if (integrator == "rk2") {nstages = 2};
      else if (integrator == "rk3") {nstages = 3};
      else if (integrator == "rk4") {nstages = 4};
      else if (integrator == "ssprk5_4") {nstages = 5};
    }
#endif
    nstages = 1;
    
    //initialize boundary conditions
    {using namespace EFLTaskNames;
        AddTask(Get_Entropy,NONE);
        AddTask(Get_EFL,Get_Entropy);
        AddTask(Set_Entropy,Get_EFL);

    }// end of using namespace block
 
}

//----------------------------------------------------------------------------------------
//! \fn void EFLTaskList:: AddTask(const TaskID& id, const TaskID& dep)
//! \brief Sets id and dependency for "ntask" member of task_list_ array, then iterates
//! value of ntask.

void EFLTaskList:: AddTask(const TaskID& id, const TaskID& dep){
    task_list_[ntasks].task_id =id;
    task_list_[ntasks].dependency=dep;
    using namespace EFLTaskNames;
    if (id == Get_Entropy){
        task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&EFLTaskList::GetEntropy);
    }
    else if(id == Get_EFL){
        task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&EFLTaskList::GetEFL);
    }
    else if (id == Set_Entropy){
        task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&EFLTaskList::SetEntropy);
    }

    else{
    std::stringstream msg;
    msg << "### FATAL ERROR in EFLTaskList::AddTask" << std::endl
        << "Invalid Task is specified" << std::endl;
    ATHENA_ERROR(msg);
    }
    ntasks++;
    return;
}

void EFLTaskList::StartupTaskList(MeshBlock *pmb, int stage) {
  return;
}


TaskStatus EFLTaskList::GetEntropy(MeshBlock *pmb, int stage){
    Hydro *phydro = pmb->phydro;
    phydro->CalculateEntropy(phydro->w , phydro->ent );
    return TaskStatus::success;
}

TaskStatus EFLTaskList::GetEFL(MeshBlock *pmb, int stage){
    Hydro *phydro = pmb->phydro;
    phydro->CalculateEFL(phydro->w,phydro->ent,
    phydro->ent1,phydro->ent2,phydro->ent3);
    return TaskStatus::success;
}

TaskStatus EFLTaskList::SetEntropy(MeshBlock *pmb, int stage){
    Hydro *phydro = pmb->phydro;
    phydro->ent3=phydro->ent2;
    phydro->ent2=phydro->ent1;
    phydro->ent1=phydro->ent;
    return TaskStatus::next;
}
