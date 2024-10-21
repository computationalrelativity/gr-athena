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
    if  ( pin->GetOrAddBoolean("hydro", "efl_in", false) )
    {
        integrator = pin->GetOrAddString("time", "integrator", "vl2");
        if      (integrator == "vl2") {nstages = 2;}
        else if (integrator == "rk1") {nstages = 1;}
        else if (integrator == "rk2") {nstages = 2;}
        else if (integrator == "rk3") {nstages = 3;}
        else if (integrator == "rk4") {nstages = 4;}
        else if (integrator == "ssprk5_4") {nstages = 5;}
    }
    else
    {
        nstages = 1;
    }
    {using namespace EFLTaskNames;  
        Add(Get_Entropy,NONE, &EFLTaskList::GetEntropy);
        Add(Get_EFL,Get_Entropy,&EFLTaskList::GetEFL);
        Add(Set_Entropy, (Get_Entropy | Get_EFL),&EFLTaskList::SetEntropy);

    }// end of using namespace block
}

//----------------------------------------------------------------------------------------
//! \fn void EFLTaskList:: AddTask(const TaskID& id, const TaskID& dep)
//! \brief Sets id and dependency for "ntask" member of task_list_ array, then iterates
//! value of ntask.

void EFLTaskList::StartupTaskList(MeshBlock *pmb, int stage) {
  return;
}


TaskStatus EFLTaskList::GetEntropy(MeshBlock *pmb, int stage){
    if (stage <= nstages)
    {
        Hydro *ph = pmb->phydro;
        ph->CalculateEntropy(ph->w , ph->entropy_0 );
        return TaskStatus::next;
    }
    return TaskStatus::fail;
}

TaskStatus EFLTaskList::GetEFL(MeshBlock *pmb, int stage){

    if (stage <= nstages)
    {
        Hydro *ph = pmb->phydro;
        ph->CalculateEFL(ph->w,ph->entropy_0,
        ph->entropy_1,ph->entropy_2,ph->entropy_3);
        return TaskStatus::next;
    }
    return TaskStatus::fail;
}

TaskStatus EFLTaskList::SetEntropy(MeshBlock *pmb, int stage){
    if (stage <= nstages)
    {
        Hydro *ph = pmb->phydro;
        ph->SetAtmMask(pmb->peos->GetEOS().GetDensityFloor(),
            ph->w, ph->atm_mask);
        ph->SetEntropy(ph->entropy_0,
        ph->entropy_1,ph->entropy_2,ph->entropy_3);
        return TaskStatus::success;
    }
    return TaskStatus::fail;
}
