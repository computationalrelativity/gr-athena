#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import h5py
import glob
import argparse
import os.path

p = argparse.ArgumentParser(description="Plotting all outputs")
p.add_argument("-d", type=str, required=True, help="dir")

args = p.parse_args()

if os.path.isfile(args.d + '/ejecta.h5') is False:
    h5files_rad = glob.glob(args.d + '/output-0000/ejecta*_000000.h5')
    h5files_rad.sort()

    nrad = len(h5files_rad)
    h5f_old = h5py.File(h5files_rad[0], 'r')

    h5f = h5py.File(args.d + '/ejecta.h5', 'w')
    h5f_old.copy('theta', h5f)
    h5f_old.copy('phi', h5f)

    shape0 = (nrad, None,)

    h5f.create_dataset('time', (0,), maxshape=(None,), dtype=h5f_old['time'].dtype)
    dset = h5f.create_dataset('radius', (nrad,), dtype=h5f_old['radius'].dtype)
    for irad, h5fname in enumerate(h5files_rad):
        h5f_in = h5py.File(h5fname, 'r')
        dset[irad] = h5f_in['radius'][0]
        h5f_in.close()

    h5f.create_group('prim')
    for dsname in h5f_old['prim'].keys():
        shape_new = tuple(np.append((nrad, 0), h5f_old['prim'][dsname].shape))
        h5f['prim'].create_dataset(dsname, shape_new, maxshape=tuple(np.append(shape0, h5f_old['prim'][dsname].shape)),
                dtype=h5f_old['prim'][dsname].dtype)
    h5f.create_group('Bcc')
    for dsname in h5f_old['Bcc'].keys():
        shape_new = tuple(np.append((nrad, 0), h5f_old['Bcc'][dsname].shape))
        h5f['Bcc'].create_dataset(dsname, shape_new, maxshape=tuple(np.append(shape0, h5f_old['Bcc'][dsname].shape)),
                dtype=h5f_old['Bcc'][dsname].dtype)
    h5f.create_group('adm')
    for dsname in h5f_old['adm'].keys():
        shape_new = tuple(np.append((nrad, 0), h5f_old['adm'][dsname].shape))
        h5f['adm'].create_dataset(dsname, shape_new, maxshape=tuple(np.append(shape0, h5f_old['adm'][dsname].shape)),
                dtype=h5f_old['adm'][dsname].dtype)
    h5f.create_group('z4c')
    for dsname in h5f_old['z4c'].keys():
        shape_new = tuple(np.append((nrad, 0), h5f_old['z4c'][dsname].shape))
        h5f['z4c'].create_dataset(dsname, shape_new, maxshape=tuple(np.append(shape0, h5f_old['z4c'][dsname].shape)),
                dtype=h5f_old['z4c'][dsname].dtype)
    h5f.create_group('other')
    for dsname in h5f_old['other'].keys():
        shape_new = tuple(np.append((nrad, 0), h5f_old['other'][dsname].shape))
        h5f['other'].create_dataset(dsname, shape_new, maxshape=tuple(np.append(shape0, h5f_old['other'][dsname].shape)),
                dtype=h5f_old['other'][dsname].dtype)
    h5f.create_dataset('mass', (nrad, 0), maxshape=shape0, dtype=h5f_old['mass'].dtype)
    h5f.create_dataset('Mdot_total', (nrad, 0,), maxshape=shape0, dtype=h5f_old['Mdot_total'].dtype)

    h5f_old.close()
    h5f.close()

dirs = glob.glob(args.d + '/output-????')
dirs.sort()

h5f = h5py.File(args.d + '/ejecta.h5', 'a')
nrad = h5f['radius'].shape[0]

print(dirs)

for d in dirs:
    nt_old = h5f['time'].shape[0]
    h5files = []
    for irad in range(nrad):
        h5ff = glob.glob(d + '/ejecta' + str(irad) + '_*.h5')
        h5ff.sort()
        h5files.append(h5ff)
    if len(h5files[0]) > 0:
        nt_ = np.int32(h5files[0][-1][-7:-3]) + 1
    else:
        nt_ = 0
    print(d, nt_old, nt_)

    if nt_ > nt_old:
        h5f['time'].resize((nt_old + len(h5files[0]),))
        h5f['mass'].resize((nrad, nt_old + len(h5files[0]),))
        h5f['Mdot_total'].resize((nrad, nt_old + len(h5files[0]),))
        for dsname in h5f['prim'].keys():
            sh0 = h5f['prim'][dsname].shape
            h5f['prim'][dsname].resize((nrad, nt_old + len(h5files[0]), sh0[2], sh0[3]))
        for dsname in h5f['Bcc'].keys():
            sh0 = h5f['Bcc'][dsname].shape
            h5f['Bcc'][dsname].resize((nrad, nt_old + len(h5files[0]), sh0[2], sh0[3]))
        for dsname in h5f['adm'].keys():
            sh0 = h5f['adm'][dsname].shape
            h5f['adm'][dsname].resize((nrad, nt_old + len(h5files[0]), sh0[2], sh0[3]))
        for dsname in h5f['z4c'].keys():
            sh0 = h5f['z4c'][dsname].shape
            h5f['z4c'][dsname].resize((nrad, nt_old + len(h5files[0]), sh0[2], sh0[3]))
        for dsname in h5f['other'].keys():
            sh0 = h5f['other'][dsname].shape
            h5f['other'][dsname].resize((nrad, nt_old + len(h5files[0]), sh0[2], sh0[3]))

        for itime, h5fname in enumerate(h5files[0]):
            h5f_in = h5py.File(h5fname, 'r')
            h5f['time'][nt_old + itime] = h5f_in['time'][0]
            h5f_in.close()

        for irad in range(nrad):
            for itime, h5fname in enumerate(h5files[irad]):
                h5f_in = h5py.File(h5fname, 'r')
                for dsname in h5f_in['prim'].keys():
                    h5f['prim'][dsname][irad, nt_old + itime] = h5f_in['prim'][dsname]
                for dsname in h5f_in['Bcc'].keys():
                    h5f['Bcc'][dsname][irad, nt_old + itime] = h5f_in['Bcc'][dsname]
                for dsname in h5f_in['adm'].keys():
                    h5f['adm'][dsname][irad, nt_old + itime] = h5f_in['adm'][dsname]
                for dsname in h5f_in['z4c'].keys():
                    h5f['z4c'][dsname][irad, nt_old + itime] = h5f_in['z4c'][dsname]
                for dsname in h5f_in['other'].keys():
                    h5f['other'][dsname][irad, nt_old + itime] = h5f_in['other'][dsname]

                h5f['mass'][irad, nt_old + itime] = h5f_in['mass'][0]
                h5f['Mdot_total'][irad, nt_old + itime] = h5f_in['Mdot_total'][0]
                h5f_in.close()

h5f.close()
