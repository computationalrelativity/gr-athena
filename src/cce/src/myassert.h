#ifndef MYASSERT_H
#define MYASSERT_H

#include<cctk.h>

#define myassert(x)                                     \
  do {                                                  \
       const long int  _err = (long int) (x);           \
       if (!_err)                                       \
       {                                                \
         CCTK_VWarn(CCTK_WARN_ABORT, __LINE__, __FILE__,\
         CCTK_THORNSTRING, "Test '%s' failed", #x);     \
       }                                                \
    } while(0)

#endif
