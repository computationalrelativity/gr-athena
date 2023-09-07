#! /usr/bin/env python

"""
Script for plotting mesh structure in mesh_structure.dat (default) file produced
by running Athena++ with "-m <np>" argument.

Can optionally specify "-i <input_file>" and/or "-o <output_file>". Output
defaults to using "show()" command rather than saving to file.
"""

# Python modules
import argparse


# Main function
def main(**kwargs):

    # Extract inputs
    input_file = kwargs['input']
    output_file = kwargs['output']
    dimension = kwargs['dimension']
    xlim = int(kwargs['xlim'])
    ylim = int(kwargs['ylim'])

    xshift = int(kwargs['xshift']) if not kwargs['xshift'] == None else kwargs['xshift']
    yshift = int(kwargs['yshift']) if not kwargs['yshift'] == None else kwargs['yshift']

    # Load Python plotting modules
    if output_file != 'show':
        import matplotlib
        matplotlib.use('agg')
    import matplotlib.pyplot as plt
    # not used explicitly, but required for 3D projections
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    # Read and plot block edges
    if dimension == '3D':
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    else:
        fig, ax = plt.subplots()
    x = []
    y = []
    z = []
    with open(input_file) as f:
        for line in f:
            if line[0] != '\n' and line[0] != '#':
                numbers_str = line.split()
                x.append(float(numbers_str[0]))
                y.append(float(numbers_str[1]))
                # append zero if 2D
                if(len(numbers_str) > 2):
                    z.append(float(numbers_str[2]))
                else:
                    z.append(0.0)
            if line[0] == '\n' and len(x) != 0:
                if dimension == '3D':
                    ax.plot(x, y, z, 'k-', lw = 0.5)
                else:
                    ax.plot(x, y, 'k-', lw = 0.5)
                x = []
                y = []
                z = []
    shift = 0
    if xlim > 0:
        if not xshift == None:
            shift = xshift
        ax.set_xlim(-xlim+shift,xlim+shift)
    shift = 0
    if ylim > 0:
        if not yshift == None:
            shift = yshift
        ax.set_ylim(-ylim+shift,ylim+shift)
    if output_file == 'show':
        plt.show()
    else:
        plt.savefig(output_file, bbox_inches='tight')


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        default='mesh_structure.dat',
                        help='name of mesh structure file')
    parser.add_argument('-o',
                        '--output',
                        default='show',
                        help=('name of output image file to create; omit to '
                              'display rather than save image'))
    parser.add_argument('-d',
                        '--dimension',
                        default='2D',
                        choices = ['2D', '3D'],
                        help=('choose z=0 2D slice or 3D representation of the '
                              'mesh grid'))
    parser.add_argument('-xl',
                        '--xlim',
                        default='-1',
                        help=('xlim for plot, must be positive'
                              '(-xlim,xlim)'))
    parser.add_argument('-yl',
                        '--ylim',
                        default='-1',
                        help=('ylim for plot, must be positive'
                              '(-ylim,ylim)'))
    parser.add_argument('-xs',
                        '--xshift',
                        default=None,
                        help=('xshift from center '
                              ))
    parser.add_argument('-ys',
                        '--yshift',
                        default=None,
                        help=('yshift from plot '
                              ))
    args = parser.parse_args()
    main(**vars(args))
