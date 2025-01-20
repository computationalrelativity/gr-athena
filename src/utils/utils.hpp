#ifndef UTILS_UTILS_HPP_
#define UTILS_UTILS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file utils.hpp
//  \brief prototypes of functions and class definitions for utils/*.cpp files

// C headers

// C++ headers
#include <csignal>   // sigset_t POSIX C extension
#include <cstdint>   // std::int64_t
#include <iostream>
#include <vector>

// Athena++ headers

void ChangeRunDir(const char *pdir);
void GetRunDir(std::string &scwd);

double ran2(std::int64_t *idum);
void ShowConfig();

// Check if file exists
bool file_exists(const char *fname);

void file_copy(const std::string &from, const std::string &to);

// Tokenize a string
void tokenize(const std::string & to_tok,
              const char token,
              std::vector<std::string> & vs);

long count_char(const std::string & source,
                const char to_count);
//----------------------------------------------------------------------------------------
//! SignalHandler
//  \brief static data and functions that implement a simple signal handling system

namespace SignalHandler {
const int nsignal = 3;
extern volatile int signalflag[nsignal];
const int ITERM = 0, IINT = 1, IALRM = 2;
extern sigset_t mask;
void SignalHandlerInit();
int CheckSignalFlags();
int GetSignalFlag(int s);
void SetSignalFlag(int s);
void SetWallTimeAlarm(int t);
void CancelWallTimeAlarm();
void BackTraceHandler(int s);
} // namespace SignalHandler

#endif // UTILS_UTILS_HPP_
