#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import h5py
import glob


# In[31]:


h5files = glob.glob('ejecta_*.h5')
h5files.sort()


# In[32]:


h5f = h5py.File('ejecta.h5', 'w')
h5f_old = h5py.File(h5files[0], 'r')

h5f_old.copy('theta', h5f)
h5f_old.copy('phi', h5f)

shape0 = (len(h5files),)
dset = h5f.create_dataset('time', shape0, dtype=h5f_old['time'].dtype)
for itime, h5fname in enumerate(h5files):
    h5f_in = h5py.File(h5fname, 'r')
    dset[itime] = h5f_in['time'][0]
    h5f_in.close()

h5f.create_group('prim')
for dsname in h5f_old['prim'].keys():
    shape_new = tuple(np.append(shape0, h5f_old['prim'][dsname].shape))
    dset = h5f['prim'].create_dataset(dsname, shape_new, dtype=h5f_old['prim'][dsname].dtype)
    for itime, h5fname in enumerate(h5files):
        h5f_in = h5py.File(h5fname, 'r')
        dset[itime] = h5f_in['prim'][dsname]
        h5f_in.close()

h5f_old.close()
h5f.close()


# In[ ]:




