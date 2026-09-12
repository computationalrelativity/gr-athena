#!/usr/bin/env python

"""
Plot Athena++ AMR mesh / oct-tree structure.

Features
--------
1. Different colors for different refinement levels.
2. Prints number of blocks at each level.
3. Prints cell resolution (dx, dy, dz) at each level.
4. Prints spatial extent of each refinement level.
5. Meshblock dimensions are supplied with --meshblock.
6. Supports 2D and 3D.

Examples
--------
    python plot_mesh.py -i mesh_structure.dat -d 2D

    python plot_mesh.py -i mesh_structure.dat -d 3D -n 16

    python plot_mesh.py -i mesh_structure.dat -d 2D -n 32 -o mesh.png
"""

import argparse
from collections import defaultdict

import matplotlib
from matplotlib.lines import Line2D


# ================================================================
# Read mesh structure
# ================================================================

def read_mesh_structure(filename):

    blocks = []

    x = []
    y = []
    z = []

    with open(filename) as f:

        for line in f:

            line = line.strip()

            # End of block
            if not line:

                if x:
                    blocks.append({
                        'x': x,
                        'y': y,
                        'z': z
                    })

                x = []
                y = []
                z = []

                continue

            # Skip comments
            if line.startswith('#'):
                continue

            numbers = line.split()

            if len(numbers) < 2:
                continue

            x.append(float(numbers[0]))
            y.append(float(numbers[1]))

            if len(numbers) >= 3:
                z.append(float(numbers[2]))
            else:
                z.append(0.0)

    # Last block
    if x:
        blocks.append({
            'x': x,
            'y': y,
            'z': z
        })

    return blocks


# ================================================================
# Meshblock physical size
# ================================================================

def block_size(block, dimension):

    dx = max(block['x']) - min(block['x'])
    dy = max(block['y']) - min(block['y'])

    if dimension == '3D':
        dz = max(block['z']) - min(block['z'])
    else:
        dz = 0.0

    return dx, dy, dz


# ================================================================
# Infer refinement levels
# ================================================================

def infer_levels(blocks, dimension):
    """
    Infer AMR level from physical meshblock size.

    Largest blocks are level 0.
    Smaller blocks are progressively higher refinement levels.
    """

    sizes = []

    for block in blocks:

        dx, dy, dz = block_size(
            block,
            dimension
        )

        if dimension == '3D':
            size = min(dx, dy, dz)
        else:
            size = min(dx, dy)

        sizes.append(size)

    # ------------------------------------------------------------
    # Identify unique block sizes
    # ------------------------------------------------------------

    unique_sizes = []

    tolerance = 1.e-8

    for size in sorted(sizes, reverse=True):

        if not unique_sizes:
            unique_sizes.append(size)
            continue

        reference = unique_sizes[-1]

        if abs(size - reference) > tolerance * max(
                abs(size),
                abs(reference),
                1.0):

            unique_sizes.append(size)

    # ------------------------------------------------------------
    # Assign level
    # ------------------------------------------------------------

    levels = []

    for size in sizes:

        level = min(
            range(len(unique_sizes)),
            key=lambda i: abs(size - unique_sizes[i])
        )

        levels.append(level)

    return levels


# ================================================================
# Statistics
# ================================================================

def level_statistics(
        blocks,
        levels,
        dimension,
        meshblock):

    stats = defaultdict(lambda: {
        'nblocks': 0,

        'block_dx': [],
        'block_dy': [],
        'block_dz': [],

        'dx': [],
        'dy': [],
        'dz': [],

        'xmin': [],
        'xmax': [],
        'ymin': [],
        'ymax': [],
        'zmin': [],
        'zmax': []
    })

    for block, level in zip(blocks, levels):

        x = block['x']
        y = block['y']
        z = block['z']

        block_dx, block_dy, block_dz = block_size(
            block,
            dimension
        )

        # Actual cell resolution
        dx = block_dx / meshblock
        dy = block_dy / meshblock

        if dimension == '3D':
            dz = block_dz / meshblock
        else:
            dz = 0.0

        s = stats[level]

        s['nblocks'] += 1

        s['block_dx'].append(block_dx)
        s['block_dy'].append(block_dy)

        if dimension == '3D':
            s['block_dz'].append(block_dz)

        s['dx'].append(dx)
        s['dy'].append(dy)

        if dimension == '3D':
            s['dz'].append(dz)

        s['xmin'].append(min(x))
        s['xmax'].append(max(x))

        s['ymin'].append(min(y))
        s['ymax'].append(max(y))

        if dimension == '3D':
            s['zmin'].append(min(z))
            s['zmax'].append(max(z))

    return stats


# ================================================================
# Print statistics
# ================================================================

def print_level_statistics(
        stats,
        dimension,
        meshblock):

    print()
    print("=" * 80)
    print("ATHENA++ AMR MESH STRUCTURE")
    print("=" * 80)

    print(
        "Meshblock cells: {}{}".format(
            meshblock,
            " x {} x {}".format(meshblock, meshblock)
            if dimension == '3D'
            else " x {}".format(meshblock)
        )
    )

    print("=" * 80)

    for level in sorted(stats):

        s = stats[level]

        print()
        print(
            "Refinement level {}".format(level)
        )

        print("-" * 80)

        print(
            "  Number of meshblocks : {:d}".format(
                s['nblocks']
            )
        )

        # --------------------------------------------------------
        # Cell resolution
        # --------------------------------------------------------

        dx = min(s['dx'])
        dy = min(s['dy'])

        if dimension == '3D':
            dz = min(s['dz'])

            print(
                "  Cell resolution      : "
                "dx = {:.10g}, dy = {:.10g}, dz = {:.10g}".format(
                    dx,
                    dy,
                    dz
                )
            )

        else:

            print(
                "  Cell resolution      : "
                "dx = {:.10g}, dy = {:.10g}".format(
                    dx,
                    dy
                )
            )

        # --------------------------------------------------------
        # Meshblock physical size
        # --------------------------------------------------------

        block_dx = min(s['block_dx'])
        block_dy = min(s['block_dy'])

        if dimension == '3D':

            block_dz = min(s['block_dz'])

            print(
                "  Meshblock size       : "
                "Lx = {:.10g}, Ly = {:.10g}, Lz = {:.10g}".format(
                    block_dx,
                    block_dy,
                    block_dz
                )
            )

        else:

            print(
                "  Meshblock size       : "
                "Lx = {:.10g}, Ly = {:.10g}".format(
                    block_dx,
                    block_dy
                )
            )

        # --------------------------------------------------------
        # Spatial extent
        # --------------------------------------------------------

        xmin = min(s['xmin'])
        xmax = max(s['xmax'])

        ymin = min(s['ymin'])
        ymax = max(s['ymax'])

        print(
            "  Spatial extent       : "
            "x = [{:.10g}, {:.10g}]".format(
                xmin,
                xmax
            )
        )

        print(
            "                         "
            "y = [{:.10g}, {:.10g}]".format(
                ymin,
                ymax
            )
        )

        if dimension == '3D':

            zmin = min(s['zmin'])
            zmax = max(s['zmax'])

            print(
                "                         "
                "z = [{:.10g}, {:.10g}]".format(
                    zmin,
                    zmax
                )
            )

    print()
    print("=" * 80)
    print()


# ================================================================
# Draw 2D block
# ================================================================

def draw_block_2d(
        ax,
        block,
        color,
        lw=0.6):

    x = block['x']
    y = block['y']

    xmin = min(x)
    xmax = max(x)

    ymin = min(y)
    ymax = max(y)

    ax.plot(
        [xmin, xmax, xmax, xmin, xmin],
        [ymin, ymin, ymax, ymax, ymin],
        color=color,
        lw=lw
    )


# ================================================================
# Draw 3D block
# ================================================================

def draw_block_3d(
        ax,
        block,
        color,
        lw=0.6):

    x = block['x']
    y = block['y']
    z = block['z']

    xmin = min(x)
    xmax = max(x)

    ymin = min(y)
    ymax = max(y)

    zmin = min(z)
    zmax = max(z)

    vertices = [
        (xmin, ymin, zmin),
        (xmax, ymin, zmin),
        (xmax, ymax, zmin),
        (xmin, ymax, zmin),

        (xmin, ymin, zmax),
        (xmax, ymin, zmax),
        (xmax, ymax, zmax),
        (xmin, ymax, zmax)
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for i, j in edges:

        ax.plot(
            [vertices[i][0], vertices[j][0]],
            [vertices[i][1], vertices[j][1]],
            [vertices[i][2], vertices[j][2]],
            color=color,
            lw=lw
        )


# ================================================================
# Main
# ================================================================

def main(**kwargs):

    input_file = kwargs['input']
    output_file = kwargs['output']
    dimension = kwargs['dimension']
    meshblock = kwargs['meshblock']

    # ------------------------------------------------------------
    # Backend
    # ------------------------------------------------------------

    if output_file != 'show':
        matplotlib.use('agg')

    import matplotlib.pyplot as plt

    # ------------------------------------------------------------
    # Read mesh
    # ------------------------------------------------------------

    blocks = read_mesh_structure(
        input_file
    )

    if not blocks:

        raise RuntimeError(
            "No mesh blocks found in {}".format(
                input_file
            )
        )

    print(
        "\nTotal meshblocks: {}".format(
            len(blocks)
        )
    )

    # ------------------------------------------------------------
    # Determine AMR levels
    # ------------------------------------------------------------

    levels = infer_levels(
        blocks,
        dimension
    )

    for block, level in zip(
            blocks,
            levels):

        block['level'] = level

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    stats = level_statistics(
        blocks,
        levels,
        dimension,
        meshblock
    )

    print_level_statistics(
        stats,
        dimension,
        meshblock
    )

    # ------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------

    nlevels = max(levels) + 1

    cmap = plt.get_cmap(
        'turbo',
        max(nlevels, 2)
    )

    colors = {
        level: cmap(level)
        for level in range(nlevels)
    }

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    if dimension == '3D':

        fig = plt.figure(
            figsize=(10, 8)
        )

        ax = fig.add_subplot(
            111,
            projection='3d'
        )

    else:

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

    # ------------------------------------------------------------
    # Draw blocks
    # ------------------------------------------------------------

    for block in blocks:

        level = block['level']

        if dimension == '3D':

            draw_block_3d(
                ax,
                block,
                colors[level]
            )

        else:

            draw_block_2d(
                ax,
                block,
                colors[level]
            )

    # ------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if dimension == '3D':
        ax.set_zlabel('z')
    else:
        ax.set_aspect(
            'equal',
            adjustable='box'
        )

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------

    legend_lines = []

    for level in sorted(colors):

        legend_lines.append(
            Line2D(
                [0],
                [0],
                color=colors[level],
                lw=2,
                label="Level {}".format(level)
            )
        )

    ax.legend(
        handles=legend_lines,
        title="AMR refinement",
        loc="best"
    )

    ax.set_title(
        "Athena++ AMR mesh structure"
    )

    # ------------------------------------------------------------
    # Save / show
    # ------------------------------------------------------------

    if output_file == 'show':

        plt.show()

    else:

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches='tight'
        )

        print(
            "Saved: {}".format(
                output_file
            )
        )


# ================================================================
# Command line
# ================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Plot Athena++ AMR mesh/oct-tree structure."
    )

    parser.add_argument(
        '-i',
        '--input',
        default='mesh_structure.dat',
        help='mesh structure file'
    )

    parser.add_argument(
        '-o',
        '--output',
        default='show',
        help='output image; default: show'
    )

    parser.add_argument(
        '-d',
        '--dimension',
        default='2D',
        choices=['2D', '3D'],
        help='2D or 3D'
    )

    parser.add_argument(
        '-n',
        '--meshblock',
        type=int,
        default=16,
        help='number of cells per meshblock dimension (default: 16)'
    )

    args = parser.parse_args()

    if args.meshblock <= 0:
        parser.error(
            '--meshblock must be a positive integer'
        )

    main(**vars(args))
