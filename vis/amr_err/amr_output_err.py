#!/usr/bin/env python

## Alireza Rashti Oct. 2022
##
## A framework to define and compute error on a slice of the grid in athena++.
## 
## Usage: bash example_par.sh
##

import sys
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
## you need "python3 -m pip install --user --upgrade findiff"
from findiff import FinDiff
## set athena hdf5 reader python file (athena_z4c/vis/python)
#athena_read_src="athena_z4c/vis/python"
#sys.path.insert(0, athena_read_src)
import athena_read

## global vars. currently there is not argument input for these:
_deriv_acc = 6   ## derivative accuracy in z4c Athena++ => h^{_deriv_acc}
_findiff_acc = 4 ## finite difference accuracy here for FinDiff
_out_prefix  = "" ## output file prefix
_hdf5_suffix = ".athdf" ## suffix of the hdf5 files to glob
## general env for 2d plot, change it
_cmap_2d = mpl.cm.cool ## [cool,jet]
_norm_2d = "normalized" ## "log"
        
## given parameters, get all hdf5 files of interest
class Params:
    
    ## a quick set of vars using arg
    def __init__(self,args):
        self.hdf5_dir    = args.i       ## dir. to save plots
        self.hdf5_prefix = args.p       ## hdf5 prefix, e.g., 'z4c_z' or 'adm'
        self.hdf5_suffix = _hdf5_suffix ## suffix of the hdf5 files to glob

        self.out_dir    = args.o + '/' ## dir. to read hdf5 files
        self.out_format = args.f       ## plot format pdf, png, txt, etc.
        self.out_prefix = _out_prefix  ## prefix of output files
        
        self.cut        = args.c ## slice of 3d data, 
        self.field_name = args.n ## the field to plot, z4c.chi and etc.
        self.step       = args.s ## reading files every step
        self.nghost     = args.g ## number of ghost zones
        self.radius     = args.r ## criterion to pick the meshblock (radii <= _radius)
        self.analysis   = args.a ## what kind of analysis we want, L2 norm, derivative, plot, etc.
        self.coord_1d   = args.x ## the fixed coordinate along which the txt_1d(?) plot(s) is plotted
        
        self.findiff_ord  = args.d       ## finite difference order for FinDiff
        self.findiff_acc  = _findiff_acc ## finite difference accuracy here for FinDiff
        self.deriv_acc    = _deriv_acc   ## derivative accuracy in z4c Athena++
        self.deriv_pow    = args.m       ## The results are powered to the m-th.
        
        self.resolution   = args.e ## resolution of the run, e.g., 128, 96,...
        self.output_field = None   ## the name of field to be output


## calc. the L2 norm and add it to the db. note: this is for a slice
def L2(params,db,mbs,slice,file):
    db[params.output_field+'_L2'] = np.zeros(shape=db[params.output_field].shape)

    small_norm = 1e-14
    
    for mb in mbs.keys():
        v = db[params.output_field][mb][ mbs[mb]['kI']:mbs[mb]['kF'],
                                          mbs[mb]['jI']:mbs[mb]['jF'],
                                          mbs[mb]['iI']:mbs[mb]['iF']]
        nx = len(range(mbs[mb]['iI'],mbs[mb]['iF']))
        ny = len(range(mbs[mb]['jI'],mbs[mb]['jF']))
        nz = len(range(mbs[mb]['kI'],mbs[mb]['kF']))
        v_L2 = np.linalg.norm(v)/np.sqrt(nx*ny*nz)
        
        ## for plotting purposes set it to a small number:
        if v_L2 == 0:
            print("NOTE: in meshblock = {} L2-norm({}) = 0! We set it to {:e}.".
                  format(mb,params.output_field,small_norm))
            v_L2 = small_norm
            
        for k in range(mbs[mb]['kI'],mbs[mb]['kF']):
            for j in range(mbs[mb]['jI'],mbs[mb]['jF']):
                for i in range(mbs[mb]['iI'],mbs[mb]['iF']):
                    db[params.output_field+'_L2'][mb][k,j,i] = v_L2

## calc. the Linf norm and add it to the db. note: this is for a slice
def Linf(params,db,mbs,slice,file,field='v'):
    
    if field == 'v':
        name    = params.output_field
        outname = params.output_field + '_Linf'
    elif field == 'dv':
        name    = params.output_field + 'without_h'
        outname = params.output_field + 'dv_Linf'
    else:
        raise Exception("No such {} option defined!".format(field))
        
    db[outname] = np.zeros(shape=db[params.output_field].shape)
    small_norm = 1e-14
    
    
    for mb in mbs.keys():
        v = db[name][mb][ mbs[mb]['kI']:mbs[mb]['kF'],
                          mbs[mb]['jI']:mbs[mb]['jF'],
                          mbs[mb]['iI']:mbs[mb]['iF']]
        
        nx = len(range(mbs[mb]['iI'],mbs[mb]['iF']))
        ny = len(range(mbs[mb]['jI'],mbs[mb]['jF']))
        nz = len(range(mbs[mb]['kI'],mbs[mb]['kF']))
        v_Linf = np.abs(np.max(v))
        
        ## for plotting purposes set it to a small number:
        if v_Linf == 0:
            print("NOTE: in meshblock = {} Linf-norm({}) = 0! We set it to {:e}.".
                  format(mb,params.output_field,small_norm))
            v_Linf = small_norm
            
        for k in range(mbs[mb]['kI'],mbs[mb]['kF']):
            for j in range(mbs[mb]['jI'],mbs[mb]['jF']):
                for i in range(mbs[mb]['iI'],mbs[mb]['iF']):
                    db[outname][mb][k,j,i] = v_Linf

## calc. the L2 norm average: (1/nmb * sum_m {L2_m}^2)^0.5 for all meshblocks.
def L2_avg(params,db,mbs):
    L2_avg = 0.

    for mb in mbs.keys():
        v = db[params.output_field][mb][ mbs[mb]['kI']:mbs[mb]['kF'],
                                          mbs[mb]['jI']:mbs[mb]['jF'],
                                          mbs[mb]['iI']:mbs[mb]['iF']]
        nx = len(range(mbs[mb]['iI'],mbs[mb]['iF']))
        ny = len(range(mbs[mb]['jI'],mbs[mb]['jF']))
        nz = len(range(mbs[mb]['kI'],mbs[mb]['kF']))
        v_L2 = np.linalg.norm(v)
        ## note: it includes puncture error
        L2_avg += v_L2**2/(nx*ny*nz)
        
    L2_avg = (L2_avg/len(mbs.keys()))**(0.5)
    
    return L2_avg

## replace the value with abs value
def Abs(params,db,mbs,slice,file):
    db[params.output_field+'_abs'] = np.zeros(shape=db[params.output_field].shape)

    for mb in mbs.keys():
        db[params.output_field+'_abs'][mb] = np.abs(db[params.output_field][mb])
        

## region of interest where the mesh-block should reside.
## it's simple but one can later make it more flexible for a complex geometry of interest
class Region:
    radius = None ## radius where the meshblock should be
    mbs    = None ## a list of sieved meshblocks
    
    def __init__(self,params):
        self.radius = params.radius
    
    ## finding meshblocks according to a specific rule
    def FindMeshBlocks(self,db):
        print("{} ...".format(self.FindMeshBlocks.__name__))
        sys.stdout.flush()

        self.mbs = self.MeshBlockByRadius(db)
        return self.mbs
    
    ## return a list of meshblocks that reside within a certain radius
    ## db is the Athena data file
    def MeshBlockByRadius(self,db):
        mbs=[]
        for mb in range(db["NumMeshBlocks"]):
            ## take the MIDDLE POINT and calc. the radius for this block wrt the origine
            x=db["x1v"][mb][int(np.size(db["x1v"][mb])/2)]
            y=db["x2v"][mb][int(np.size(db["x2v"][mb])/2)]
            z=db["x3v"][mb][int(np.size(db["x3v"][mb])/2)]
            r=(x**2+y**2+z**2)**(0.5)
            if r < self.radius:
                mbs.append(mb)

        if len(mbs) == 0:
            raise Exception("No meshblock found for radius <= {}!".format(self.radius))
        return mbs



## slice/cut the 3d grid in a given direction
## it finds the initial(I) and final(F) values of indices in all 3d if the slicing 
## plane is taking place in the meshblock. it checks if the constant coordinate of 
## the plane resides in the meshblock up to the grid space error.
class Slice:
    ## public
    slice_dir = None ## direction of slice, 3 for z, 2 for y, 1 for x
    slice_val = None ## plane value for slice, e.g., 2.0 in  x=2.0
    coordv = None    ## normal coordinate to the plane, "x1v","x2v", "x3v" (center)
    coordf = None    ## normal coordinate to the plane, "x1v","x2v", "x3v" (face)
    mbs    = dict()  ## sliced mesh blocks by its range  
                     ## dict={'mb': {'kI' = ,'kF' = ,'jI' = ,'Jf' = ,'iI' = ,'iF' = }}
    
    ## parse cut arg and find the indices bound
    def __init__(self,params):
        ## pars cut:
        cs = params.cut.split("=")
        if cs[0] == 'z' or cs[0] == 'Z':
            self.slice_dir = 3
            self.coordv = "x3v"
            self.coordf = "x3f"
            
        elif cs[0] == 'y' or cs[0] == 'Y':
            self.slice_dir = 2
            self.coordv = "x2v"
            self.coordf = "x2f"
            
        elif cs[0] == 'x' or cs[0] == 'X':
            self.slice_dir = 1
            self.coordv = "x1v"
            self.coordf = "x1f"
            
        else:
            raise Exception("could not pars '{}'! Set -c arg to, e.g., z=0.1.".format(cut))
       
        self.slice_val = float(cs[1])

    
    ## find the range of indices in all 3d and save it into the mbs dictionary
    def SliceMeshBlock(self,params,db,vol):
        print("{} ...".format(self.SliceMeshBlock.__name__))
        sys.stdout.flush()
        
        self.mbs.clear()
        for m in range(len(vol)):
            mb = vol[m]
            h = db[self.coordf][mb][1]-db[self.coordf][mb][0]
            ## only check the interior points
            ng = params.nghost
            ## if this is already sliced =>
            if len(db[self.coordv][mb])-ng < 0:
                ng = 0
            for i in range(ng,len(db[self.coordv][mb])-ng):
                coord = db[self.coordv][mb][i]
                if np.abs(coord - self.slice_val) <= h:
                    ## NOTE: the correction for ng is done later
                    self.mbs[mb] = dict()
                    self.mbs[mb]['iI'] = 0
                    self.mbs[mb]['iF'] = len(db["x1v"][mb])
                    self.mbs[mb]['jI'] = 0
                    self.mbs[mb]['jF'] = len(db["x2v"][mb])
                    self.mbs[mb]['kI'] = 0
                    self.mbs[mb]['kF'] = len(db["x3v"][mb])

                    if self.slice_dir == 3:
                        self.mbs[mb]['kI'] = i
                        self.mbs[mb]['kF'] = i+1
                    elif self.slice_dir == 2:
                        self.mbs[mb]['jI'] = i
                        self.mbs[mb]['jF'] = i+1
                    elif self.slice_dir == 1:
                        self.mbs[mb]['iI'] = i
                        self.mbs[mb]['iF'] = i+1
                    break
                    
        if len(self.mbs) == 0:
            raise Exception("could not find any {} = {} slice.".format(self.coordv,self.slice_val))
        
        return self.mbs

## collect files
class Files:
    ## public
    files = dict() ## path to all hdf5 files of interest
    
    def __init__(self,params):
        ## get all files
        files = glob.glob("{}/*{}{}".format(params.hdf5_dir,params.hdf5_prefix,
                                            "*"+params.hdf5_suffix))
        if len(files) == 0:
            raise Exception("could not find any file '{}*{}' in:\n{}!".format(params.hdf5_prefix,
                                                                 params.hdf5_suffix,
                                                                 params.hdf5_dir))

        ## pick by every given step
        file_counter = -1
        files.sort()
        for file in files:
            file_counter +=1
            if file_counter % params.step != 0:
                continue
            
            cycle = file[-12:-len(params.hdf5_suffix)]
            ## we assumed 5 digits so check this 
            assert(cycle[0] == '.')
            cycle = cycle[1:]
            
            #print("Adding ...\n'{}'".format(file))
            #print("---")
            #sys.stdout.flush()
            self.files[file] = dict()
            self.files[file]['cycle'] = cycle
            ## when txt output is asked
            self.files[file]['txt_2d'] = params.out_dir + params.out_prefix + \
                                         params.field_name + "_" + params.analysis + "_2d.txt"
            
            self.files[file]['txt_1d'] = params.out_dir + params.out_prefix + \
                                         params.field_name + "_" + params.analysis + "_1d.txt"

            ## each meshblock has its own file                                        
            self.files[file]['txt_1d_mb'] = params.out_dir + params.out_prefix + \
                                            params.field_name + "_" + params.analysis 

            ## each meshblock has its own file                                        
            self.files[file]['txt_2d_mb'] = params.out_dir + params.out_prefix + \
                                            params.field_name + "_" + params.analysis 
                                         
            self.files[file]['txt_2d_L2'] = params.out_dir + params.out_prefix + \
                                            params.field_name + "_" + params.analysis + "_L2"+"_2d.txt"

            self.files[file]['txt_2d_Linf'] = params.out_dir + params.out_prefix + \
                                            params.field_name + "_" + params.analysis + "_Linf"+"_2d.txt"
            self.files[file]['txt_2d_dv_Linf'] = params.out_dir + params.out_prefix + \
                                            params.field_name + "_" + params.analysis + "_dv_Linf"+"_2d.txt"
                                            
            self.files[file]['color_2d'] = params.out_dir + params.out_prefix + \
                                           params.field_name + "_" + params.analysis
            self.files[file]['color_2d_L2'] = params.out_dir + params.out_prefix + \
                                              params.field_name + "_" + params.analysis + "_L2"

            self.files[file]['color_2d_Linf'] = params.out_dir + params.out_prefix + \
                                              params.field_name + "_" + params.analysis + "_Linf"
            
            self.files[file]['color_2d_abs'] = params.out_dir + params.out_prefix + \
                                              params.field_name + "_" + params.analysis + "_abs"


## plot the quantity of interest
class Plot:
    def __init__(self,params,db,mbs,slice,file):
        if params.out_format == "txt":
            self.plot_2d_txt(params,db,mbs,slice,file['cycle'],"value",file['txt_2d'])

            L2(params,db,mbs,slice,file)
            self.plot_2d_txt(params,db,mbs,slice,file['cycle'],"L2",file['txt_2d_L2'])

            Linf(params,db,mbs,slice,file)
            self.plot_2d_txt(params,db,mbs,slice,file['cycle'],"Linf",file['txt_2d_Linf'])

            if params.analysis == "der":
                Linf(params,db,mbs,slice,file,'dv')
                self.plot_2d_txt(params,db,mbs,slice,file['cycle'],"dv_Linf",file['txt_2d_dv_Linf'])
            
        elif params.out_format == "txt1d":
            self.plot_1d_txt(params,db,mbs,slice,file['cycle'],"value",file['txt_1d'])

        elif params.out_format == "txt1d_mbx":
            self.plot_1d_txt_mb(params,db,mbs,slice,file['cycle'],"value",file['txt_1d_mb'],'x')
        
        elif params.out_format == "txt2d_mb":
            self.plot_2d_txt_mb(params,db,mbs,slice,file['cycle'],"value",file['txt_2d_mb'])
        
        elif params.out_format == "txt1d_mby":
            self.plot_1d_txt_mb(params,db,mbs,slice,file['cycle'],"value",file['txt_1d_mb'],'y')
        
#        elif params.out_format == "txt1d_mbz":
#            self.plot_1d_txt_mb(params,db,mbs,slice,file['cycle'],"value",file['txt_1d_mb'],'z')
            
        elif params.out_format == "pdf" or params.out_format == "png":
            self.plot_2d_color(params,db,mbs,slice,file['cycle'],"value",file['color_2d'])

            L2(params,db,mbs,slice,file)
            self.plot_2d_color(params,db,mbs,slice,file['cycle'],"L2",file['color_2d_L2'],
                               mynorm="log")

            Linf(params,db,mbs,slice,file)
            self.plot_2d_color(params,db,mbs,slice,file['cycle'],"Linf",file['color_2d_Linf'],
                               mynorm="log")

            Abs(params,db,mbs,slice,file)
            self.plot_2d_color(params,db,mbs,slice,file['cycle'],"abs",file['color_2d_abs'],
                               mynorm="log")

        else:
            raise Exception("No such {} option defined!".format(params.out_format))
    
    ## plotting in 2d color format
    def plot_2d_color(self,params,db,mbs,slice,cycle,type,output,mynorm=_norm_2d,cmap=_cmap_2d):
        print("{} ...".format(self.plot_2d_color.__name__))
        sys.stdout.flush()
        
        ng = params.nghost
        
        ## set plot env.
        fig  = plt.figure()
        ax   = fig.add_subplot()
        
        if type == "value":
            fld  = params.output_field
        elif type == "L2":
            fld  = params.output_field+'_L2'
        elif type == "Linf":
            fld  = params.output_field+'_Linf'
        elif type == "abs":
            fld  = params.output_field+'_abs'
        else:
            raise Exception("No such option {}!".format(type))
        
        ## find vmin and vmax
        vmin = sys.float_info.max
        vmax = 0
        for mb in mbs.keys():
            v = db[fld][mb]
            if slice.slice_dir == 3:
                if ng == 0:
                    vminp = np.amin(v[mbs[mb]['kI'], :, :])
                    vmaxp = np.amax(v[mbs[mb]['kI'], :, :])
                else:
                    vminp = np.amin(v[mbs[mb]['kI'], ng:-ng, ng:-ng])
                    vmaxp = np.amax(v[mbs[mb]['kI'], ng:-ng, ng:-ng])

            elif slice.slice_dir == 2:
                if ng == 0:
                    vminp = np.amin(v[: ,mbs[mb]['jI'], :])
                    vmaxp = np.amax(v[: ,mbs[mb]['jI'], :])
                else:
                    vminp = np.amin(v[ng:-ng ,mbs[mb]['jI'], ng:-ng])
                    vmaxp = np.amax(v[ng:-ng ,mbs[mb]['jI'], ng:-ng])

            elif slice.slice_dir == 1:
                if ng == 0:
                    vminp = np.amin(v[:, :, mbs[mb]['iI']])
                    vmaxp = np.amax(v[:, :, mbs[mb]['iI']])
                else:
                    vminp = np.amin(v[ng:-ng, ng:-ng, mbs[mb]['iI']])
                    vmaxp = np.amax(v[ng:-ng, ng:-ng, mbs[mb]['iI']])

            else:
                raise Exception("No such slice {}!".format(slice.slice_dir))
            
            vmin = vmin if vmin < vminp else vminp
            vmax = vmax if vmax > vmaxp else vmaxp
        
        if mynorm == "normalized":
            norm=mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        elif mynorm == "log":
            ## ensure we don't have log(0)
            if vmin == 0.0:
                vmin = 1e-14
            if vmax == 0.0:
                vmax = 1e-14
            norm=mpl.colors.LogNorm(vmin=vmin,vmax=vmax)
        else:
            raise Exception("No such norm {}!".format(mynorm))
            
        for mb in mbs.keys():
        
            if slice.slice_dir == 3:
                x = db["x1v"][mb][ mbs[mb]['iI']+ng:mbs[mb]['iF']-ng ] 
                y = db["x2v"][mb][ mbs[mb]['jI']+ng:mbs[mb]['jF']-ng ] 
            elif slice.slice_dir == 2:
                x = db["x1v"][mb][ mbs[mb]['iI']+ng:mbs[mb]['iF']-ng ] 
                z = db["x3v"][mb][ mbs[mb]['kI']+ng:mbs[mb]['kF']-ng ] 
            elif slice.slice_dir == 1:
                y = db["x2v"][mb][ mbs[mb]['jI']+ng:mbs[mb]['jF']-ng ] 
                z = db["x3v"][mb][ mbs[mb]['kI']+ng:mbs[mb]['kF']-ng ] 
            else:
                raise Exception("No such slice {}!".format(slice.slice_dir))
                
            v = db[fld][mb]
            
            if slice.slice_dir == 3:
                X,Y = np.meshgrid(x,y)
                if ng == 0:
                    plt.pcolor(Y,X,v[mbs[mb]['kI'], :, :],norm=norm)
                else:
                    plt.pcolor(Y,X,v[mbs[mb]['kI'], ng:-ng, ng:-ng],norm=norm)
                xlabel="y"
                ylabel="x"
                
            elif slice.slice_dir == 2:
                X,Z = np.meshgrid(x,z)
                if ng == 0:
                    plt.pcolor(Z,X,v[: ,mbs[mb]['jI'], :],norm=norm)
                else:
                    plt.pcolor(Z,X,v[ng:-ng ,mbs[mb]['jI'], ng:-ng],norm=norm)
                xlabel="z"
                ylabel="x"
                
            elif slice.slice_dir == 1:
                Y,Z = np.meshgrid(y,z)
                if ng == 0:
                    plt.pcolor(Z,Y,v[:, :, mbs[mb]['iI']],norm=norm)
                else:
                    plt.pcolor(Z,Y,v[ng:-ng, ng:-ng, mbs[mb]['iI']],norm=norm)
                xlabel="z"
                ylabel="y"
                
            else:
                raise Exception("No such slice {}!".format(slice.slice_dir))

        ax.set_title("cycle:" + cycle + ", slice:{}".format(params.cut))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(label="h^6 " + fld,cmap=cmap,orientation="horizontal")
        plt.savefig(output+ "_" + cycle + "." + params.out_format)
        plt.close('all')

    ## plotting in 2d txt format
    def plot_2d_txt(self,params,db,mbs,slice,cycle,type,output):
        print("{} ...".format(self.plot_2d_txt.__name__))
        sys.stdout.flush()
        
        ng = params.nghost
        
        txt_file = open(output,"a")
        txt_file.write("# \"time = {}\"\n".format(cycle))

        if type == "value":
            fld  = params.output_field
        elif type == "L2":
            fld  = params.output_field+'_L2'
        elif type == "Linf":
            fld  = params.output_field+'_Linf'
        elif type == "dv_Linf":
            fld  = params.output_field+'dv_Linf'
        else:
            raise Exception("No such option {}!".format(type))

        for mb in mbs.keys():
            x = db["x1v"][mb]
            y = db["x2v"][mb]
            z = db["x3v"][mb]
            v = db[fld][mb]
            hf = db[params.output_field+'grid-space'][mb]
            
            if slice.slice_dir == 3:
                for j in range(mbs[mb]['jI']+ng,mbs[mb]['jF']-ng):
                    for i in range(mbs[mb]['iI']+ng,mbs[mb]['iF']-ng):
                        txt_file.write("{} {} {} {} \n".format(y[j],x[i],v[mbs[mb]['kI'], j, i],hf[mbs[mb]['kI'], j, i]))
                    txt_file.write("\n") ## the newline is mandatory for the wired frame plot
            
            elif slice.slice_dir == 2:
                for k in range(mbs[mb]['kI']+ng,mbs[mb]['kF']-ng):
                    for i in range(mbs[mb]['iI']+ng,mbs[mb]['iF']-ng):
                        txt_file.write("{} {} {} {}\n".format(z[k],x[i],v[k, mbs[mb]['jI'], i],hf[k, mbs[mb]['jI'], i]))
                    txt_file.write("\n") ## the newline is mandatory for the wired frame plot
                
            elif slice.slice_dir == 1:
                for k in range(mbs[mb]['kI']+ng,mbs[mb]['kF']-ng):
                    for j in range(mbs[mb]['jI']+ng,mbs[mb]['jF']-ng):
                        txt_file.write("{} {} {} {}\n".format(z[k],y[j],v[k, j, mbs[mb]['iI']],hf[k, j, mbs[mb]['iI']]))
                    txt_file.write("\n") ## the newline is mandatory for the wired frame plot
            
            else:
                raise Exception("No such slice {}!".format(slice.slice_dir))

        txt_file.close()
        
        if type == 'L2':
            txt_file = open(output+'_L2_average',"a")
            txt_file.write("# \"time = {}\"\n".format(cycle))
            txt_file.write("{} {}\n".format(params.resolution, L2_avg(params,db,mbs) ))
            txt_file.close()
        

    ## plotting in 1d txt format for ALL meshblocks (aggregates all mbs into one file)
    def plot_1d_txt(self,params,db,mbs,slice,cycle,type,output):
        print("{} ...".format(self.plot_1d_txt.__name__))
        sys.stdout.flush()
        
        ng = params.nghost
        
        txt_file = open(output,"a")
        txt_file.write("# \"time = {}\"\n".format(cycle))

        if type == "value":
            fld  = params.output_field
        else:
            raise Exception("No such option {}!".format(type))

        if slice.slice_dir == 3:
            for mb in mbs.keys():
                x = db["x1v"][mb]
                y = db["x2v"][mb]
                v = db[fld][mb]
                hx = x[1]-x[0]
                
                found_i = 0
                
                for i in range(mbs[mb]['iI']-ng,mbs[mb]['iF']-ng):
                    if np.abs(x[i] - params.coord_1d) < hx:
                        found_i = 1;
                        break
                if found_i == 1:
                    ## plot along y-axis
                    for j in range(mbs[mb]['jI']+ng,mbs[mb]['jF']-ng):
                        txt_file.write("{} {}\n".format(y[j],v[mbs[mb]['kI'], j, i]))
        else:
            txt_file.close()
            raise Exception("No such slice {}!".format(slice.slice_dir))
     
        txt_file.close()

    ## plotting in 1d txt format for EACH meshblocks separately in each file
    def plot_1d_txt_mb(self,params,db,mbs,slice,cycle,type,output_root,kind):
        print("{} ...".format(self.plot_1d_txt_mb.__name__))
        sys.stdout.flush()
        
        ng = params.nghost

        if type == "value":
            fld = params.output_field
        else:
            raise Exception("No such option {}!".format(type))

        ## plot along y
        if kind == 'y' and slice.slice_dir == 3:
            for mb in mbs.keys():
                x = db["x1v"][mb]
                y = db["x2v"][mb]
                v = db[fld][mb]
                hx = x[1]-x[0]
                
                ## find the fixed coord x
                found_i = 0
                for i in range(mbs[mb]['iI']+ng,mbs[mb]['iF']-ng):
                    if np.abs(x[i] - params.coord_1d) < hx:
                        I = i;
                        found_i = 1;
                        break

                if found_i == 0:
                    continue

                output = output_root + f"_mb{mb}" + "_1dy.txt"
                txt_file = open(output,"a")
                txt_file.write("# \"time = {}\", {}, x = {}\n".format(cycle,params.cut,x[I]))
                
                ## plot along y-axis
                for j in range(mbs[mb]['jI']+ng,mbs[mb]['jF']-ng):
                    txt_file.write("{} {}\n".format(y[j],v[mbs[mb]['kI'], j, I]))
        
                txt_file.close()

        ## plot along x
        elif kind == 'x' and slice.slice_dir == 3:
            for mb in mbs.keys():
                x = db["x1v"][mb]
                y = db["x2v"][mb]
                v = db[fld][mb]
                hy = y[1]-y[0]
                
                ## find the fixed coord
                found_j = 0
                for j in range(mbs[mb]['jI']+ng,mbs[mb]['jF']-ng):
                    if np.abs(y[j] - params.coord_1d) < hy:
                        J = j
                        found_j = 1;
                        break

                if found_j == 0:
                    continue

                output = output_root + f"_mb{mb}" + "_1dx.txt"
                txt_file = open(output,"a")
                txt_file.write("# \"time = {}\", {}, y = {}\n".format(cycle,params.cut,y[J]))
                
                ## plot along x-axis
                for i in range(mbs[mb]['iI']+ng,mbs[mb]['iF']-ng):
                    txt_file.write("{} {}\n".format(x[i],v[mbs[mb]['kI'], J, i]))
        
                txt_file.close()
        
        else:
            raise Exception("No such slice {} or kind {}!".format(slice.slice_dir,kind))

    ## plotting in 2d txt format for EACH meshblocks separately in each file
    def plot_2d_txt_mb(self,params,db,mbs,slice,cycle,type,output_root):
        print("{} ...".format(self.plot_2d_txt_mb.__name__))
        sys.stdout.flush()
        
        ng = params.nghost

        if type == "value":
            fld = params.output_field
        else:
            raise Exception("No such option {}!".format(type))

        ## plot along xy
        if slice.slice_dir == 3:
            for mb in mbs.keys():
                x = db["x1v"][mb]
                y = db["x2v"][mb]
                v = db[fld][mb]

                output = output_root + f"_mb{mb}" + "_2d.txt"
                txt_file = open(output,"a")
                txt_file.write("# \"time = {}\", {}, ([0]=x ,[1]=y, [2]=field)\n".format(cycle,params.cut))
                
                for j in range(mbs[mb]['jI']+ng,mbs[mb]['jF']-ng):
                    for i in range(mbs[mb]['iI']+ng,mbs[mb]['iF']-ng):
                        txt_file.write("{:.6f} {:.6f} {:.6f}\n".format(x[i],y[j],v[mbs[mb]['kI'], j, i]))
                    txt_file.write("\n") ## the newline is mandatory for the wired frame plot
                   
                txt_file.close()
        else:
            raise Exception("No such slice {} or kind {}!".format(slice.slice_dir))

## do the post processing here
class Analysis:
   def __init__(self,params,db,mbs,slice,file):
       
       ## plot a quantity
       if params.analysis == "plot":
           params.output_field = params.field_name
           self.max_grid_space(params,db,mbs,slice,file)
       
       ## calc. derivative
       elif params.analysis == "der":
           params.output_field = "d^{0}/dX^{0} ({1})".format(params.findiff_ord,params.field_name)
           self.derivative(params,db,mbs,slice,file)

       else:
           raise Exception("Unknown analysis '{}'!".format(params.analysis))
   
   ## calc. derivative, note: this is for a slice
   def derivative(self,params,db,mbs,slice,file):
        print("{} ...".format(self.derivative.__name__))
        sys.stdout.flush()

        ## Note: this is multiplied by h to measure FD error
        db[params.output_field] = np.zeros(shape=db[params.field_name].shape)
        db[params.output_field+'without_h'] = np.zeros(shape=db[params.field_name].shape)
        db[params.output_field+'grid-space'] = np.zeros(shape=db[params.field_name].shape)
        
        for mb in mbs.keys():
            v = db[params.field_name][mb][ mbs[mb]['kI']:mbs[mb]['kF'],
                                           mbs[mb]['jI']:mbs[mb]['jF'],
                                           mbs[mb]['iI']:mbs[mb]['iF']]
                                               
            ## set the function(v) and diff operator(op) and then the derive (dv)
            if slice.slice_dir == 3:
                x = db["x1v"][mb]
                y = db["x2v"][mb]
                v = db[params.field_name][mb][mbs[mb]['kI'], :, :]
                dx = x[1]-x[0]
                dy = y[1]-y[0]
                h = max(dx,dy)
                
                op1 = 0.5*FinDiff(1,dx,params.findiff_ord,acc=params.findiff_acc)
                op0 = 0.5*FinDiff(0,dy,params.findiff_ord,acc=params.findiff_acc)
                
                dv1 = op1(v)
                dv0 = op0(v)
                
                dv = ( dv1**(params.deriv_pow) + dv0**(params.deriv_pow) ) ## pure dv
                hdv = (h**params.deriv_acc) * dv
                
                for k in range(mbs[mb]['kI'],mbs[mb]['kF']):
                    for j in range(mbs[mb]['jI'],mbs[mb]['jF']):
                        for i in range(mbs[mb]['iI'],mbs[mb]['iF']):
                            db[params.output_field][mb][k,j,i] = hdv[j,i]
                            db[params.output_field+'without_h'][mb][k,j,i] = dv[j,i]
                            db[params.output_field+'grid-space'][mb][k,j,i] = h
        
       
            elif slice.slice_dir == 2:
                x = db["x1v"][mb]
                z = db["x3v"][mb]
                v = db[params.field_name][mb][:, mbs[mb]['jI'], :]
                dx = x[1]-x[0]
                dz = z[1]-z[0]
                h = max(dx,dz)
                
                op1 = 0.5*FinDiff(1,dx,params.findiff_ord,acc=params.findiff_acc)
                op0 = 0.5*FinDiff(0,dz,params.findiff_ord,acc=params.findiff_acc)
                
                dv1 = op1(v)
                dv0 = op0(v)
                
                dv = ( dv1**(params.deriv_pow) + dv0**(params.deriv_pow) ) ## pure dv
                hdv = (h**params.deriv_acc) * dv
                
                for k in range(mbs[mb]['kI'],mbs[mb]['kF']):
                    for j in range(mbs[mb]['jI'],mbs[mb]['jF']):
                        for i in range(mbs[mb]['iI'],mbs[mb]['iF']):
                            db[params.output_field][mb][k,j,i] = hdv[j,i]
                            db[params.output_field+'without_h'][mb][k,j,i] = dv[j,i]
                            db[params.output_field+'grid-space'][mb][k,j,i] = h

            elif slice.slice_dir == 1:
                z = db["x3v"][mb]
                y = db["x2v"][mb]
                v = db[params.field_name][mb][:, :, mbs[mb]['iI']]
                dz = z[1]-z[0]
                dy = y[1]-y[0]
                h = max(dz,dy)
                
                op1 = 0.5*FinDiff(1,dy,params.findiff_ord,acc=params.findiff_acc)
                op0 = 0.5*FinDiff(0,dz,params.findiff_ord,acc=params.findiff_acc)
                
                dv1 = op1(v)
                dv0 = op0(v)
                
                dv = ( dv1**(params.deriv_pow) + dv0**(params.deriv_pow) ) ## pure dv
                hdv = (h**params.deriv_acc) * dv
                
                for k in range(mbs[mb]['kI'],mbs[mb]['kF']):
                    for j in range(mbs[mb]['jI'],mbs[mb]['jF']):
                        for i in range(mbs[mb]['iI'],mbs[mb]['iF']):
                            db[params.output_field][mb][k,j,i] = hdv[j,i]
                            db[params.output_field+'without_h'][mb][k,j,i] = dv[j,i]
                            db[params.output_field+'grid-space'][mb][k,j,i] = h

            else:
                raise Exception("No such slice {}!".format(slice.slice_dir))
            

   ## compute the largest grid-space in each mesh-block
   def max_grid_space(self,params,db,mbs,slice,file):
        print("{} ...".format(self.derivative.__name__))
        sys.stdout.flush()

        db[params.output_field+'grid-space'] = np.zeros(shape=db[params.field_name].shape)
        
        for mb in mbs.keys():
            if slice.slice_dir == 3:
                x = db["x1v"][mb]
                y = db["x2v"][mb]
                dx = x[1]-x[0]
                dy = y[1]-y[0]
                h = max(dx,dy)
                
                for k in range(mbs[mb]['kI'],mbs[mb]['kF']):
                    for j in range(mbs[mb]['jI'],mbs[mb]['jF']):
                        for i in range(mbs[mb]['iI'],mbs[mb]['iF']):
                            db[params.output_field+'grid-space'][mb][k,j,i] = h
        
       
            elif slice.slice_dir == 2:
                x = db["x1v"][mb]
                z = db["x3v"][mb]
                dx = x[1]-x[0]
                dz = z[1]-z[0]
                h = max(dx,dz)
                
                for k in range(mbs[mb]['kI'],mbs[mb]['kF']):
                    for j in range(mbs[mb]['jI'],mbs[mb]['jF']):
                        for i in range(mbs[mb]['iI'],mbs[mb]['iF']):
                            db[params.output_field+'grid-space'][mb][k,j,i] = h

            elif slice.slice_dir == 1:
                z = db["x3v"][mb]
                y = db["x2v"][mb]
                dz = z[1]-z[0]
                dy = y[1]-y[0]
                h = max(dz,dy)
                
                for k in range(mbs[mb]['kI'],mbs[mb]['kF']):
                    for j in range(mbs[mb]['jI'],mbs[mb]['jF']):
                        for i in range(mbs[mb]['iI'],mbs[mb]['iF']):
                            db[params.output_field+'grid-space'][mb][k,j,i] = h

            else:
                raise Exception("No such slice {}!".format(slice.slice_dir))
            


## --------------------------------------------------------------------------------
if __name__=="__main__":
    ## read and pars input args (we're running out of letter!!)
    p = argparse.ArgumentParser(description="Plotting errors in a BBH Athena++ run.")
    p.add_argument("-i",type=str,required=True,help="path/to/hdf5/dir")
    p.add_argument("-o",type=str,required=True,help="path/to/output/dir")
    p.add_argument("-e",type=int,default = 128, help="resolution of the run, e.g., 128, 96,...")
    p.add_argument("-p",type=str,required= True,help="hdf5 prefix, e.g., 'z4c_z' or 'adm'.")
    p.add_argument("-f",type=str,default = "txt", help="output format = {pdf,png,txt,txt1d,txt1d_mbx,txt1d_mby,txt2d_mb}.")
    p.add_argument("-n",type=str,default = "z4c.chi" , help="field name, e.g., z4c.chi, con.H.")
    p.add_argument("-c",type=str,default = "z=0.0", help="clipping/cutting of the 3D grid, e.g., z=0.")
    p.add_argument("-s",type=int,default = 10, help="read every step-th file.")
    p.add_argument("-g",type=int,default = 4, help="number of ghost zone.")
    p.add_argument("-r",type=float,default = 5.0,help="select all meshblocks whose radii are <= this value.")
    p.add_argument("-a",type=str,default = "plot",help="analysis = {plot,der}.")
    p.add_argument("-d",type=int,default = 2, help="derivative order.")
    p.add_argument("-m",type=float,default = 1, help="derivative power. The results are powered to the m-th.")
    p.add_argument("-x",type=float,default = 0, help="the fixed coordinate along which the txt_1d(?) plot(s) is plotted")
    
    args = p.parse_args()

    ## init
    params = Params(args)
    files  = Files(params)
    slice  = Slice(params)
    region = Region(params)

    ## open the db files
    for f in files.files.keys():
        print("{}'...".format(f))
        sys.stdout.flush()
        
        ## open the Athena files. True is set to open the file mesh-block by mesh-block
        db = athena_read.athdf(f,True)
        
        ## pick those meshblocks where lying in a particulate region,
        ## namely, their radii is <= raduis
        vol = region.FindMeshBlocks(db)
        
        ## only pick meshblocks that reside on the slice (i.e., plane)
        mbs = slice.SliceMeshBlock(params,db,vol)
        
        ## Post Processing
        Analysis(params,db,mbs,slice,files.files[f])
        
        Plot(params,db,mbs,slice,files.files[f])
        
        
    
