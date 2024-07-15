// C headers
#include <unistd.h>
#include <stdio.h>
#include <linux/limits.h>

// C++ headers
#include <iostream>
#include <sstream>

// Athena++ headers
#include "../athena.hpp"

void GetRunDir(std::string &scwd)
{
  std::stringstream msg;

  char cwd[PATH_MAX];
  if (getcwd(cwd, sizeof(cwd)) != NULL)
  {
    scwd.assign(cwd);
  }
  else
  {
    msg << "### FATAL ERROR in function [GetRunDir]"
        << std::endl;
    ATHENA_ERROR(msg);
  }

  return;
}

//
// :D
//