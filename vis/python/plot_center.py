#! /usr/bin/env python

"""
Script for plotting the center of a TOV star from .athdf files.
"""

# Python modules
import argparse
import glob
import matplotlib.pyplot as plt

# Athena++ modules
import athena_read

# Main function
def main(**kwargs):
  
  # Extract inputs
  file_heads   = kwargs['file_heads'].split(',')
  output_file = kwargs['output_file']
  #fields = kwargs['fields'].split(',')
  nghosts = kwargs['nghosts']
  log_scale = kwargs['log']
  labels = kwargs['labels'].split(',')

  files = []
  n_lines = len(file_heads)
  labels_used = True
  if len(labels) == 1 and labels[0] == '':
    for i in range(n_lines):
      labels.append('')
    labels_used = False
  for n in range(n_lines):
    files.append(glob.glob('{}.*.athdf'.format(file_heads[n])))
    if len(files[n]) == 0:
      raise RuntimeError('Could not find any files matching {}'.format(file_head))
  if len(labels) != len(file_heads):
    raise RuntimeError('Number of labels does not match files')
  #if fields[0] == '':
  #  raise RuntimeError('First entry in fields must be nonempty')

  # Read data
  t_sets = []
  y_sets = []
  for i in range(n_lines):
    t = []
    rho = []
    for n in range(len(files[n])):
      data = athena_read.athdf(files[i][n], num_ghost=nghosts)
      t.append(data['Time'])
      rho.append(data['rho'][0,0,nghosts])
    # Sort the data
    rho = [x for _,x in sorted(zip(t, rho))]
    t = sorted(t)
    t_sets.append(t)
    y_sets.append(rho)

  print(t_sets)
  print(y_sets)

  # Plot data
  plt.figure()
  for i in range(n_lines):
    plt.plot(t_sets[i], y_sets[i], label=labels[i])
  if log_scale:
    plt.yscale('log')
  if labels_used:
    plt.legend(loc='best')

  plt.savefig(output_file, bbox_inches='tight')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-f',
    '--file_heads',
    help=('comma-separated list of file heads (i.e., file_head.<n>.athdf)'),
  )
  parser.add_argument(
    '-o',
    '--output_file',
    help=('name of output file'),
  )
  parser.add_argument(
    '-n',
    '--nghosts',
    type=int,
    default=0,
    help=('Number of ghost points in the data sets')
  )
  parser.add_argument(
    '-l',
    '--log',
    action='store_true',
    help=('flag indicating y-axis should be log scaled')
  )
  parser.add_argument(
    '--labels',
    default='',
    help='comma-separated list of labels for legend'
  )
  args = parser.parse_args()
  main(**vars(args))
