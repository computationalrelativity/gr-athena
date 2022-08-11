//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file read_lorene.hpp
//  \brief defines struct LoreneTable
//  Contains data and functions to read .lorene ASCII

// C headers

// C++ headers
#include <cmath>   // sqrt()
#include <fstream>
#include <iostream> // ifstream
#include <sstream>
#include <stdexcept> // std::invalid_argument
#include <string>

// Athena++ headers
#include "../parameter_input.hpp"

#ifndef DEBUG
#define DEBUG (1)
#endif 

#define STRLEN (1024)
#define UNIT_DENS     (1.782662696e12)         // [MeV/fm^3] --> [g/cm^3]
#define DENS_TO_GEOM  (1.6200170038654943e-18) // [g/cm^3]   --> [dimensionless]
#define PRESS_TO_GEOM (1.802512010158757e-39)  // [g/cm^3]   --> [dimensionless]
#define MN            (939.56542)              // [MeV]      --> neutron mass in MeV

// LoreneTable
enum{tab_idx, tab_nb, tab_rho, tab_p, tab_e, tab_eps, tab_logp, tab_logrho, tab_dpdrho, tab_dlogpdlogrho, tab_nv};
struct LoreneTable 
{
int size;
double *data[tab_nv];
double rho_max, rho_min, rho_atm;

int size_header;
char **header_Y = NULL;
double **Y = NULL;
};

/* LoreneTable funcs */
void AllocTable(LoreneTable **Table, const int size);
void FreeTable(LoreneTable *Table);
void ReadLoreneTable(std::string filename, LoreneTable *Table);
void ReadLoreneFractions(std::string filename, LoreneTable *Table);
void ConvertLoreneTable(LoreneTable *Table);
int FindLoreneFractions(std::string Y_i, LoreneTable *Table);

/* ASCII */
void remove_comments(char *line, const char *delimiters);
char *trim(char *str);
int is_blank(const char *line);
int noentries(const char *string);

/* Others */
int D0_x_2(double *f, double *x, int n, double *df);
