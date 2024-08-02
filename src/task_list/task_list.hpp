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

// success vs. next: They are different (only) when there are more than one MeshBlock per
// node.  When a task returns “next”, then the code processes the next Task in the same
// MeshBlock; when it returns “success”, then the TaskList processes the next MeshBlock.
// “next” should be used when you want to immediately start the next task, for example,
// start sending the data just calculated in the previous task.  Otherwise, use “success”
// to process MeshBlocks as evenly as possible.

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

  // Couple function prototype directly at the point the task is added
  //
  // By default all tasks are considered to contribute to load-balance timing
  void Add(
    const TaskID& id, const TaskID& dep,
    TaskStatus (TaskList::*fcn)(MeshBlock*, int),
    bool lb_time=true)
  {
    task_list_[ntasks].task_id = id;
    task_list_[ntasks].dependency = dep;
    task_list_[ntasks].TaskFunc = fcn;
    task_list_[ntasks].lb_time = lb_time;

    ntasks++;
  }

private:
  virtual void AddTask(const TaskID& id, const TaskID& dep) = 0;
  virtual void StartupTaskList(MeshBlock *pmb, int stage) = 0;

};

#endif  // TASK_LIST_TASK_LIST_HPP_
