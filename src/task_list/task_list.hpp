#ifndef TASK_LIST_TASK_LIST_HPP_
#define TASK_LIST_TASK_LIST_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//!   \file task_list.hpp
//    \brief provides functionality to control dynamic execution using tasks

// C headers

// C++ headers
#include <cstdint>      // std::uint64_t
#include <string>       // std::string

// Athena++ headers
#include "../athena.hpp"


// forward declarations
class Mesh;
class MeshBlock;
class TaskList;
class TaskID;

// TODO(felker): these 4x declarations can be nested in TaskList if MGTaskList is derived

// constants = return codes for functions working on individual Tasks and TaskList
enum class TaskStatus {fail, success, next};
enum class TaskListStatus {running, stuck, complete, nothing_to_do};

//----------------------------------------------------------------------------------------
//! \class TaskID
//  \brief generalization of bit fields for Task IDs, status, and dependencies.

class TaskID {  // POD but not aggregate (there is a user-provided ctor)
public:
  TaskID() = default;
  explicit TaskID(unsigned int id);
  void Clear();
  bool IsUnfinished(const TaskID& id) const;
  bool CheckDependencies(const TaskID& dep) const;
  void SetFinished(const TaskID& id);

  bool operator== (const TaskID& rhs) const;
  TaskID operator| (const TaskID& rhs) const;

private:
  constexpr static int kNField_ = 1;
  std::uint64_t bitfld_[kNField_];

  friend class TaskList;
};


//----------------------------------------------------------------------------------------
//! \struct Task
//  \brief data and function pointer for an individual Task

struct Task { // aggregate and POD
  TaskID task_id;    // encodes task with bit positions in HydroIntegratorTaskNames
  TaskID dependency; // encodes dependencies to other tasks using " " " "
  TaskStatus (TaskList::*TaskFunc)(MeshBlock*, int);  // ptr to member function
  bool lb_time; // flag for automatic load balancing based on timing
};

//---------------------------------------------------------------------------------------
//! \struct TaskStates
//  \brief container for task states on a single MeshBlock

struct TaskStates { // aggregate and POD
  TaskID finished_tasks;
  int indx_first_task, num_tasks_left;
  void Reset(int ntasks) {
    indx_first_task = 0;
    num_tasks_left = ntasks;
    finished_tasks.Clear();
  }
};

//----------------------------------------------------------------------------------------
//! \class TaskList
//  \brief data and function definitions for task list base class

class TaskList {
public:
  TaskList() : ntasks(0), nstages(0), task_list_{} {} // 2x direct + zero initialization
  // rule of five:
  virtual ~TaskList() = default;

  // data
  int ntasks;     // number of tasks in this list
  int nstages;    // number of times the tasklist is repeated per each full timestep

  // functions
  TaskListStatus DoAllAvailableTasks(MeshBlock *pmb, int stage, TaskStates &ts);
  void DoTaskListOneStage(Mesh *pmesh, int stage);

protected:
  // TODO(felker): rename to avoid confusion with class name
  Task task_list_[64*TaskID::kNField_];

private:
  virtual void AddTask(const TaskID& id, const TaskID& dep) = 0;
  virtual void StartupTaskList(MeshBlock *pmb, int stage) = 0;
};

//----------------------------------------------------------------------------------------

class WaveIntegratorTaskList : public TaskList {
public:
  WaveIntegratorTaskList(ParameterInput *pin, Mesh *pm);

  //--------------------------------------------------------------------------------------
  //! \struct IntegratorWeight
  //  \brief weights used in time integrator tasks

  struct IntegratorWeight {
    // 2S or 3S* low-storage RK coefficients, Ketchenson (2010)
    Real delta; // low-storage coefficients to avoid double F() evaluation per substage
    Real gamma_1, gamma_2, gamma_3; // low-storage coeff for weighted ave of registers
    Real beta; // coeff. from bidiagonal Shu-Osher form Beta matrix, -1 diagonal terms
  };

  // data
  std::string integrator;
  Real cfl_limit; // dt stability limit for the particular time integrator + spatial order

  // functions
  TaskStatus ClearAllBoundary(MeshBlock *pmb, int stage);  // CLEAR_ALLBND [x]
  TaskStatus UserWork(MeshBlock *pmb, int stage);          // USERWORK     [x]
  TaskStatus NewBlockTimeStep(MeshBlock *pmb, int stage);  // NEW_DT       [x]
  TaskStatus CheckRefinement(MeshBlock *pmb, int stage);   // FLAG_AMR     [x]
  TaskStatus CalculateWaveRHS(MeshBlock *pmb, int stage);  // CALC_WAVERHS [x]
  TaskStatus IntegrateWave(MeshBlock *pmb, int stage);     // INT_WAVE     [x]
  TaskStatus SendWave(MeshBlock *pmb, int stage);          // SEND_WAVE    [x]
  TaskStatus ReceiveWave(MeshBlock *pmb, int stage);       // RECV_WAVE    [x]

  TaskStatus SetBoundariesWave(MeshBlock *pmb, int stage); // SETB_WAVE    [x]

  TaskStatus Prolongation(MeshBlock *pmb, int stage);      // PROLONG      [x]
  TaskStatus PhysicalBoundary(MeshBlock *pmb, int stage);  // PHY_BVAL     [x]

private:
  IntegratorWeight stage_wghts[MAX_NSTAGE];

  void AddTask(const TaskID& id, const TaskID& dep) override;
  void StartupTaskList(MeshBlock *pmb, int stage) override;
};

//----------------------------------------------------------------------------------------
// 64-bit integers with "1" in different bit positions used to ID each wave task.
namespace WaveIntegratorTaskNames {

  const TaskID NONE(0);
  const TaskID CLEAR_ALLBND(1);

  const TaskID CALC_WAVERHS(2);
  const TaskID INT_WAVE(3);
  const TaskID SEND_WAVE(4);
  const TaskID RECV_WAVE(5);
  const TaskID SETB_WAVE(6);
  const TaskID PROLONG(7);
  const TaskID PHY_BVAL(8);
  const TaskID USERWORK(9);
  const TaskID NEW_DT(10);
  const TaskID FLAG_AMR(11);
}  // namespace WaveIntegratorTaskNames
// -BD
//----------------------------------------------------------------------------------------

#endif  // TASK_LIST_TASK_LIST_HPP_
