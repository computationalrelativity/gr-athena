#ifndef Task_LIST_EFL_TASK_LIST_HPP_
#define Task_LIST_EFL_TASK_LIST_HPP_

//! \file efl_task_list.hpp
//! \brief define Entrop Flux Limiter TaskList

// Athena++ headers
#include "../../athena.hpp"
#include "../task_list.hpp"
//forward declarations
class Mesh;
class MeshBlock;

//----------------------------------------------------------------------------------------
//! \class EFLTaskList
//! \brief data and function definitions for EFL

class EFLTaskList: public TaskList
{
public:
  // Constructor
  EFLTaskList(ParameterInput *pin, Mesh *pm);

  //functions 
  TaskStatus GetEntropy(MeshBlock *pmb, int stage);
  TaskStatus GetEFL(MeshBlock *pmb,int stage);
  TaskStatus SetEntropy(MeshBlock *pmb, int stage);
public:
  using TaskList::nstages;
private:
  void AddTask(const TaskID& id, const TaskID& dep) override { };
  void StartupTaskList(MeshBlock *pmb, int stage) override;

private:
  // For slightly cleaner & more flexible, treatment of tasklist graph assembly
  void Add(
    const TaskID& id, const TaskID& dep,
    TaskStatus (EFLTaskList::*fcn)(MeshBlock*, int),
    bool lb_time=true)
  {
    TaskList::Add(id, dep, static_cast<TaskStatus (TaskList::*)(
      MeshBlock*, int
    )>(fcn));
  }
};

namespace EFLTaskNames {
const TaskID NONE(0);
const TaskID Get_Entropy(1);
const TaskID Get_EFL(2);
const TaskID Set_Entropy(3);
}

#endif //Task_LIST_EFL_TASK_LIST_HPP_