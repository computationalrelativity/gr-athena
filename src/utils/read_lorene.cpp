//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file read_lorene.cpp
//  \brief implements functions to read .lorene ASCII files into a LoreneTable struct (mostly C)

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
#include "read_lorene.hpp"

const char COMMENTS[]    = "#";
const char * DELIMITER[] = {" "};

// LoreneTable functions

//----------------------------------------------------------------------------------------
// \!fn void  AllocTable(LoreneTable **Table, int size)
// \brief Allocate memory for LoreneTable struct, except for header and Y_i
void AllocTable(LoreneTable **Table, int size)
{
    (*Table)->size = size;
    for(int i=0; i<tab_nv; i++) 
        (*Table)->data[i] = (double*) malloc((*Table)->size*sizeof(double));
    /* header and Y are allocated in ReadLoreneFractions */
}

//----------------------------------------------------------------------------------------
// \!fn void  FreeTable(LoreneTable **Table, int size)
// \brief Free memory for LoreneTable struct
void FreeTable(LoreneTable *Table)
{
    if (!Table) return;
    for(int i=0; i<tab_nv; i++)
        if (Table->data[i]) free(Table->data[i]);

    for(int i=0; i<Table->size_header; i++){
        if (Table->Y[i]) free(Table->Y[i]);
        if (Table->header_Y[i]) free(Table->header_Y[i]);
    }

    free(Table->Y);
    free(Table->header_Y);
    delete Table;
}

//----------------------------------------------------------------------------------------
// \!fn void ReadLoreneTable(std::string filename, LoreneTable *Table)
// \brief Read a .lorene ASCII file into a LoreneTable
void ReadLoreneTable(std::string filename, LoreneTable *Table)
{
    char line[3*STRLEN];
    int size;
    int row  = 0;

    FILE *fp = fopen(filename.c_str(), "r");
    if (!fp){
        std::stringstream msg;
        msg << "### FATAL ERROR in function [ReadLoreneTable]"
            << std::endl << "File not found";
        ATHENA_ERROR(msg);
    } 
    // TODO: check that the extension is of a .lorene file

    /* read the lines into the table struct */
    while (fgets(line,3*STRLEN,fp)!=NULL) {
        if (is_blank(line)) continue;
        if (line[0] == '#') continue;
        /* remove comments and trailing whitespaces */
        remove_comments(line, COMMENTS);
        trim(line);
        /* read the data */
        if(row == 0){
            sscanf(line, "%d", &size);
            AllocTable(&Table, size);
        }
        else sscanf(line, "%lf %lf %lf %lf", &Table->data[tab_idx][row-1],  &Table->data[tab_nb][row-1], &Table->data[tab_e][row-1], &Table->data[tab_p][row-1]);
        row++;
    }
    /* done with the .lorene file */
    fclose(fp);

    /* compute epsilon and rho*/
    for(int i=0; i<size; i++){
        Table->data[tab_rho][i] = Table->data[tab_nb][i]*MN;
        Table->data[tab_eps][i] = Table->data[tab_e][i]/(MN*Table->data[tab_nb][i]*UNIT_DENS) - 1.;
    }
    if (DEBUG) printf("ReadLoreneTable: done reading the table\n");
}       

//----------------------------------------------------------------------------------------
// \!fn void ReadLoreneFractions(std::string filename, LoreneTable *Table)
// \brief Read an input file containing abundances Y of an arbitrary number of species, Y_i
void ReadLoreneFractions(std::string filename, LoreneTable *Table)
{
    char line[8*STRLEN];
    char tmp[8*STRLEN];
    double val = 0.;
    int size = 1; int size_header = 1; int row = 0; int pl_hold = 0;

    FILE *fp = fopen(filename.c_str(), "r");
    if (!fp){
        std::stringstream msg;
        msg << "### FATAL ERROR in function [ReadLoreneTable]"
            << std::endl << "File not found";
        ATHENA_ERROR(msg);
    } 

    /* read the lines into the Table*/
    while (fgets(line,8*STRLEN,fp)!=NULL) {
        if (is_blank(line)) continue;
        if (line[0] == '#') continue;
        remove_comments(line, COMMENTS);
        trim(line);

        int offset = 0; int j = 0; 
        char *buffer = line;

        /* read the header */
        if(row == 0){
            /* alloc the Table header and the Y*/
            sscanf(buffer, "%d%n", &size, &offset);
            buffer += offset;

            size_header = noentries(line) -1;
            Table->size_header = size_header;
            Table->header_Y    = (char **) calloc(size_header,sizeof(char *));
            while (sscanf(buffer, "%s%n",tmp, &offset) == 1)
            {
                Table->header_Y[j] = (char *) calloc(strlen(tmp)+1,sizeof(char));
                strcpy(Table->header_Y[j],tmp);
                buffer +=offset;
                j++;
            }
            /* Finally, allocate Y */
            Table->Y = (double **) malloc(size_header*sizeof(double *));
            for(int k=0; k<size_header; k++) Table->Y[k] = (double *) calloc(size,sizeof(double));
        
        } else {
        /* read the data */
            sscanf(buffer, "%d%n", &pl_hold, &offset);
            buffer +=offset;
            while (sscanf(buffer, "%lf%n", &val, &offset) == 1)
            {
                Table->Y[j][row-1] = val;
                buffer +=offset;
                j++;
            }
        }
        row++;
    }

    fclose(fp);
    if (DEBUG) printf("ReadLoreneFractions: done reading the Y table\n");
}

//----------------------------------------------------------------------------------------
// \!fn void ConvertLoreneTable(LoreneTable *Table)
// \brief Convert to dimensionless units p, rho, e and compute: 
// logp, logrho, dpdrho, dlogpdlogrho, rho_min and rho_max
void ConvertLoreneTable(LoreneTable *Table)
{
    double conversion_factors[tab_nv];
    std::fill(conversion_factors, conversion_factors+tab_nv, 1.);
    conversion_factors[tab_rho] = UNIT_DENS*DENS_TO_GEOM;
    conversion_factors[tab_p]   = PRESS_TO_GEOM;
    conversion_factors[tab_e]   = UNIT_DENS*DENS_TO_GEOM;

    for(int i=0; i < Table->size; i++){
        for(int k=1; k<tab_nv;k++){
            if (k==tab_logp)        continue;
            if (k==tab_logrho)      continue;
            if (k==tab_dpdrho)      continue;
            if (k==tab_dlogpdlogrho) continue;
            Table->data[k][i] *=conversion_factors[k];
        }
        Table->data[tab_logp][i]   = log(Table->data[tab_p][i]);
        Table->data[tab_logrho][i] = log(Table->data[tab_rho][i]);
    }
    /* Compute dpdrho */
    // TODO: move it somewhere else? */
    D0_x_2(Table->data[tab_logp], Table->data[tab_logrho], Table->size, Table->data[tab_dlogpdlogrho]);
    for(int i=0; i<Table->size;i++)
        Table->data[tab_dpdrho][i]=Table->data[tab_dlogpdlogrho][i]*(Table->data[tab_p][i])/(Table->data[tab_rho][i]);

    Table->rho_min = Table->data[tab_rho][0];
    Table->rho_max = Table->data[tab_rho][Table->size-1];
    if (DEBUG) printf("ConvertLoreneTable: done converting to dimensionless units\n");
}

//----------------------------------------------------------------------------------------
// \!fn void FindLoreneFractions(std::string Y_i, LoreneTable *Table)
// \brief Given an input string specifying the desired Y_i, 
// returns the header index of the corresponding data
int FindLoreneFractions(std::string Y_i, LoreneTable *Table)
{   
    int k; int err=1;
    for(k=0; k<Table->size_header; k++){
        printf("%s %s\n", Y_i.c_str(),  Table->header_Y[k]);
        if(!strcmp(Y_i.c_str(), Table->header_Y[k])){
            err=0;
            break;
        }
    }
    if(err){
        std::stringstream msg;
        msg << "### FATAL ERROR in function [FindLoreneFractions]"
            << std::endl << "Specified fraction not found";
        ATHENA_ERROR(msg);
    }
    return k;
}

// Functions for ASCII parsing 

//--------------------------------------------------------------------------------------
//! \fn void remove_comments(char *line, const char *delimiters)
// \brief cut string to the first delimiter
void remove_comments(char *line, const char *delimiters)
{
    int sz = strcspn(line,delimiters);
    char *newline = (char *) calloc(sz+1, sizeof(char));
    strncpy(newline,line,sz);
    newline[sz] = '\0';
    strcpy(line, newline);
    free(newline);
}
//--------------------------------------------------------------------------------------
//! \fn char *trim(char *str)
// \brief get rid of trailing and leading whitespace
char *trim(char *str)
{
    char *start = str;
    char *end = str + strlen(str);  
    while(*start && isspace(*start))
        start++;
    while(end > start && isspace(*(end - 1)))
        end--;
    *end = '\0';
    return start;
}

//--------------------------------------------------------------------------------------
//! \fn int is_blank(const char *line) 
// \brief checks for blank string
int is_blank(const char *line) 
{
    const char accept[]=" \t\r\n"; 
    return (strspn(line, accept) == strlen(line));
}

//--------------------------------------------------------------------------------------
//! \fn noentries(const char *line)
// \brief Returns the number of entries in line according to
// delimiter
int noentries(const char *line)
{
  int n = 0;
  char *s, *t; 
  int len = strlen(line);
  s = strndup(line,len);
  t = strtok(s,DELIMITER[0]);
  while (t != NULL) {
    n++;
    t = strtok(NULL,DELIMITER[0]);
  }
  free(s);
  return n;
}

// Others

//--------------------------------------------------------------------------------------
//! \fn int D0_x_2(double *f, double *x, int n, double *df)
// \brief 1st order centered stencil first derivative, nonuniform grids
int D0_x_2(double *f, double *x, int n, double *df)
{
  int i;
  for(i=1; i<n-1; i++) {
    df[i] = (f[i]-f[i-1])/(x[i]-x[i-1]);
  }
  i = 0;
  df[i] = (f[i]-f[i+1])/(x[i]-x[i+1]);
  i = n-1;
  df[i] = (f[i-1]-f[i])/(x[i-1]-x[i]);
  return 0;
}