//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file change_rundir.cpp
//! \brief executes unix 'chdir' command to change dir in which Athena++ runs

// C headers
// POSIX C extensions
#include <unistd.h>    // access()

// C++ headers
#include <filesystem>

// Athena++ headers
// #include "../defs.hpp"
#include "../athena.hpp"
#include "../globals.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ChangeRunDir(const char *pdir)
// Check if file exists

bool file_exists(const char *fname)
{
#ifdef MPI_PARALLEL
  if (0 == Globals::my_rank)
  {
#endif
    if (access(fname, F_OK) == 0)
    {
      return 1;
    }
    else
    {
      return 0;
    }
#ifdef MPI_PARALLEL
  }

  return 1;
#endif

}

void file_copy(const std::string &from, const std::string &to)
{
  const auto copyOptions = std::filesystem::copy_options::update_existing;

  std::filesystem::copy(from, to, copyOptions);
}
