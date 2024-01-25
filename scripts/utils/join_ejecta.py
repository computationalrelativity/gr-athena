#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import h5py
import glob


# In[31]:


h5files_rad = glob.glob('ejecta*_000000.h5')
h5files_rad.sort()

nrad = len(h5files_rad)

h5files = []
for irad in range(nrad):
    h5ff = glob.glob('ejecta' + str(irad) + '_*.h5')
    h5ff.sort()
    h5files.append(h5ff)

ntime = len(h5files[0])
# In[32]:
h5f = h5py.File('ejecta.h5', 'w')
h5f_old = h5py.File(h5files[0][0], 'r')

h5f_old.copy('theta', h5f)
h5f_old.copy('phi', h5f)

shape0 = (nrad, ntime,)

dset = h5f.create_dataset('time', (ntime,), dtype=h5f_old['time'].dtype)
for itime, h5fname in enumerate(h5files[0]):
    h5f_in = h5py.File(h5fname, 'r')
    dset[itime] = h5f_in['time'][0]
    h5f_in.close()

dset = h5f.create_dataset('radius', (nrad,), dtype=h5f_old['radius'].dtype)
for irad, h5fname in enumerate(h5files_rad):
    h5f_in = h5py.File(h5fname, 'r')
    dset[irad] = h5f_in['radius'][0]
    h5f_in.close()

h5f.create_group('prim')
for dsname in h5f_old['prim'].keys():
    shape_new = tuple(np.append(shape0, h5f_old['prim'][dsname].shape))
    dset = h5f['prim'].create_dataset(dsname, shape_new, dtype=h5f_old['prim'][dsname].dtype)
    for irad in range(nrad):
        for itime, h5fname in enumerate(h5files[irad]):
            h5f_in = h5py.File(h5fname, 'r')
            dset[irad, itime] = h5f_in['prim'][dsname]
            h5f_in.close()

h5f.create_group('cons')
for dsname in h5f_old['cons'].keys():
    shape_new = tuple(np.append(shape0, h5f_old['cons'][dsname].shape))
    dset = h5f['cons'].create_dataset(dsname, shape_new, dtype=h5f_old['cons'][dsname].dtype)
    for irad in range(nrad):
        for itime, h5fname in enumerate(h5files[irad]):
            h5f_in = h5py.File(h5fname, 'r')
            dset[irad, itime] = h5f_in['cons'][dsname]
            h5f_in.close()

h5f.create_group('Bcc')
for dsname in h5f_old['Bcc'].keys():
    shape_new = tuple(np.append(shape0, h5f_old['Bcc'][dsname].shape))
    dset = h5f['Bcc'].create_dataset(dsname, shape_new, dtype=h5f_old['Bcc'][dsname].dtype)
    for irad in range(nrad):
        for itime, h5fname in enumerate(h5files[irad]):
            h5f_in = h5py.File(h5fname, 'r')
            dset[irad, itime] = h5f_in['Bcc'][dsname]
            h5f_in.close()

h5f.create_group('adm')
for dsname in h5f_old['adm'].keys():
    shape_new = tuple(np.append(shape0, h5f_old['adm'][dsname].shape))
    dset = h5f['adm'].create_dataset(dsname, shape_new, dtype=h5f_old['adm'][dsname].dtype)
    for irad in range(nrad):
        for itime, h5fname in enumerate(h5files[irad]):
            h5f_in = h5py.File(h5fname, 'r')
            dset[irad, itime] = h5f_in['adm'][dsname]
            h5f_in.close()

h5f.create_group('z4c')
for dsname in h5f_old['z4c'].keys():
    shape_new = tuple(np.append(shape0, h5f_old['z4c'][dsname].shape))
    dset = h5f['z4c'].create_dataset(dsname, shape_new, dtype=h5f_old['z4c'][dsname].dtype)
    for irad in range(nrad):
        for itime, h5fname in enumerate(h5files[irad]):
            h5f_in = h5py.File(h5fname, 'r')
            dset[irad, itime] = h5f_in['z4c'][dsname]
            h5f_in.close()

h5f.create_group('other')
for dsname in h5f_old['other'].keys():
    shape_new = tuple(np.append(shape0, h5f_old['other'][dsname].shape))
    dset = h5f['other'].create_dataset(dsname, shape_new, dtype=h5f_old['other'][dsname].dtype)
    for irad in range(nrad):
        for itime, h5fname in enumerate(h5files[irad]):
            h5f_in = h5py.File(h5fname, 'r')
            dset[irad, itime] = h5f_in['other'][dsname]
            h5f_in.close()

dset = h5f.create_dataset('mass', (nrad, ntime,), dtype=h5f_old['mass'].dtype)
for irad in range(nrad):
    for itime, h5fname in enumerate(h5files[irad]):
        h5f_in = h5py.File(h5fname, 'r')
        dset[irad, itime] = h5f_in['mass'][0]
        h5f_in.close()

h5f_old.close()
h5f.close()


# In[ ]:




