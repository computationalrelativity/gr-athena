#ifndef PARAMETER_INPUT_HPP_
#define PARAMETER_INPUT_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file parameter_input.hpp
//  \brief definition of class ParameterInput
// Contains data structures used to store, and functions used to access, parameters
// read from the input file.  See comments at start of parameter_input.cpp for more
// information on the Athena++ input file format.

// C headers

// C++ headers
#include <cstddef>  // std::size_t
#include <ostream>  // ostream
#include <string>   // string
#include <vector>

// Athena++ headers
#include "athena.hpp"
#include "defs.hpp"
#include "outputs/io_wrapper.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \struct InputLine
//  \brief  node in a singly linked list of parameters contained within 1x input block

struct InputLine {
  std::string param_name;
  std::string param_value;   // value of the parameter is stored as a string!
  std::string param_comment;
  InputLine *pnext;   // pointer to the next node in this nested singly linked list
};

//----------------------------------------------------------------------------------------
//! \class InputBlock
//  \brief node in a singly linked list of all input blocks contained within input file

class InputBlock {
 public:
  InputBlock() = default;
  ~InputBlock();

  // data
  std::string block_name;
  std::size_t max_len_parname;  // length of longest param_name, for nice-looking output
  std::size_t max_len_parvalue; // length of longest param_value, to format outputs
  InputBlock *pnext;  // pointer to the next node in InputBlock singly linked list

  InputLine *pline;   // pointer to head node in nested singly linked list (in this block)
  // (not storing a reference to the tail node)

  // functions
  InputLine* GetPtrToLine(std::string name);
};

//----------------------------------------------------------------------------------------
//! \class ParameterInput
//  \brief data and definitions of functions used to store and access input parameters
//  Functions are implemented in parameter_input.cpp

class ParameterInput {
 public:
  // constructor/destructor
  ParameterInput();
  ~ParameterInput();

  // data
  InputBlock* pfirst_block;   // pointer to head node in singly linked list of InputBlock
  // (not storing a reference to the tail node)

  // functions
  void LoadFromStream(std::istream &is);
  void LoadFromFile(IOWrapper &input);
  void ModifyFromCmdline(int argc, char *argv[]);
  void ParameterDump(std::ostream& os);
  int  DoesParameterExist(std::string block, std::string name);
  int  GetInteger(std::string block, std::string name);
  int  GetOrAddInteger(std::string block, std::string name, int value);
  int  SetInteger(std::string block, std::string name, int value);
  Real GetReal(std::string block, std::string name);
  Real GetOrAddReal(std::string block, std::string name, Real value);
  Real SetReal(std::string block, std::string name, Real value);
  bool GetBoolean(std::string block, std::string name);
  bool GetOrAddBoolean(std::string block, std::string name, bool value);
  bool SetBoolean(std::string block, std::string name, bool value);
  std::string GetString(std::string block, std::string name);
  std::string GetOrAddString(std::string block, std::string name, std::string value);
  std::string SetString(std::string block, std::string name, std::string value);

  // Overwrite a parameter; useful if some state is achieve and a toggle in
  // behaviour needs to be set thereafter (including with restarts)
  template<class T>
  void OverwriteParameter(
    const std::string & block,
    const std::string & name,
    T value)
  {
    InputBlock* pb;
    InputLine *pl;

    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);

    Lock();

    if (DoesParameterExist(block, name))
    {
      pb = GetPtrToBlock(block);
      pl = pb->GetPtrToLine(name);
      pl->param_value = std::to_string(value);
    }
    else
    {
      pb = FindOrAddBlock(block);
      AddParameter(pb,
                   name,
                   std::to_string(value),
                   "# Value inject during run time");
    }

    Unlock();
  }

  // Convenience functions for dealing with array-like parameters -------------
  typedef std::vector<std::string> t_vec_str;
  typedef std::vector<Real> t_vec_Real;
  typedef std::vector<int> t_vec_int;
  typedef std::vector<bool> t_vec_bool;

  // std::vector varieties
  t_vec_str GetOrAddStringArray(std::string block,
                                std::string name,
                                t_vec_str def_values);

  t_vec_Real GetOrAddRealArray(std::string block,
                               std::string name,
                               t_vec_Real def_values);

  t_vec_int GetOrAddIntegerArray(std::string block,
                                 std::string name,
                                 t_vec_int def_values);

  t_vec_bool GetOrAddBooleanArray(std::string block,
                                  std::string name,
                                  t_vec_bool def_values);

  // AthenaArray<type> varieties
  AthenaArray<Real> GetOrAddRealArray(const std::string & block,
                                      const std::string & name,
                                      const AthenaArray<Real> & def_values);
  AthenaArray<int> GetOrAddIntegerArray(const std::string & block,
                                        const std::string & name,
                                        const AthenaArray<int> & def_values);
  AthenaArray<bool> GetOrAddBooleanArray(const std::string & block,
                                         const std::string & name,
                                         const AthenaArray<bool> & def_values);
  AthenaArray<std::string> GetOrAddStringArray(
    const std::string & block,
    const std::string & name,
    const AthenaArray<std::string> & def_values);

  AthenaArray<Real> GetOrAddRealArray(const std::string & block,
                                      const std::string & name,
                                      const Real & def_value);
  AthenaArray<int> GetOrAddIntegerArray(const std::string & block,
                                        const std::string & name,
                                        const int & def_value);
  AthenaArray<bool> GetOrAddBooleanArray(const std::string & block,
                                         const std::string & name,
                                         const bool & def_value);
  AthenaArray<std::string> GetOrAddStringArray(
    const std::string & block,
    const std::string & name,
    const std::string & def_value);

  // --------------------------------------------------------------------------

  void RollbackNextTime();
  void ForwardNextTime(Real time);

 private:
  const int max_pars_array = 1024;

  std::string last_filename_;  // last input file opened, to prevent duplicate reads

  InputBlock* FindOrAddBlock(std::string name);
  InputBlock* GetPtrToBlock(std::string name);
  void ParseLine(InputBlock *pib, std::string line, std::string& name,
                 std::string& value, std::string& comment);
  void AddParameter(InputBlock *pib, std::string name, std::string value,
                    std::string comment);

  // thread safety
#ifdef OPENMP_PARALLEL
  omp_lock_t lock_;
#endif

  void Lock();
  void Unlock();

  void GetExistingStringArray(const std::string & block,
                              const std::string & name,
                              t_vec_str & vec);

  void AddParameterStringArray(const std::string & block,
                               const std::string & name,
                               const t_vec_str & values);

  template <class T>
  T GetOrAddArray(std::string block, std::string name, T def_values);

  template <class T>
  AthenaArray<T> GetOrAddArray(const std::string & block,
                               const std::string & name,
                               const AthenaArray<T> & def_values);

};
#endif // PARAMETER_INPUT_HPP_
