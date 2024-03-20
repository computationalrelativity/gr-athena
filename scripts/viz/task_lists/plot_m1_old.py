#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_)

@author: Boris Daszuta
@function: Plot tasklists for various problems

Needs networkx, graphviz, pygraphviz
"""
import networkx as _nx
import matplotlib.pyplot as _plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
# -----------------------------------------------------------------------------

class con(object):
  def __init__(self, NODE_LABEL=None, NODE_CHILDREN=None, NODE_PARENTS=None,
    description=None):

    self.NODE_LABEL = NODE_LABEL
    self.NODE_CHILDREN = self._san(NODE_CHILDREN)
    self.NODE_PARENTS = self._san(NODE_PARENTS)
    if isinstance(description, str):
      description = [description, ]

    description[0] = self.prettify(description[0])
    self.description = description

  def _san(self, input=None):
    if input is None:
      input = []
    elif isinstance(input, str):
      input = [input, ]

    return input

  def prettify(self, string):
    tstr = string.split(' ')
    return ' '.join((tstr[0], tstr[1].replace('[', r'$[\mathbf{').replace(
      ']', '}]:$').replace('_', r'\_')))

class col_con(object):
  def __init__(self):
    self.nodes = {}

    self.added_nodes = {}

  def add_node_direct(self, con):
    for NP in con.NODE_PARENTS:
      if NP in self.nodes:
        if con.NODE_LABEL not in self.nodes[NP].NODE_CHILDREN:
          self.nodes[NP].NODE_CHILDREN.append(con.NODE_LABEL)

    self.nodes[con.NODE_LABEL] = con

  def add_node(self, NODE_LABEL=None, NODE_CHILDREN=None, NODE_PARENTS=None,
               description=None):
    # append task name
    description[0] = description[0] + f" [{NODE_LABEL}]"
    nc = con(NODE_LABEL=NODE_LABEL, NODE_CHILDREN=NODE_CHILDREN,
             NODE_PARENTS=NODE_PARENTS, description=description)
    self.add_node_direct(nc)

  def add_node_unparsed(
    self, NODE_LABEL=None, NODE_PARENTS=None,
    description=None):
    # append task name
    description[0] = description[0] + f" [{NODE_LABEL}]"
    self.added_nodes[NODE_LABEL] = {
      "description": description,
      "NODE_PARENTS": NODE_PARENTS,
      "NODE_CHILDREN": []
    }

  def node_parse_added(self):

    # ensure all nodes have their children populated and connected
    for NODE_LABEL, item in self.added_nodes.items():
      NODE_PARENTS = item["NODE_PARENTS"]
      if NODE_PARENTS is not None:
        for NP in NODE_PARENTS:
          self.added_nodes[NP]["NODE_CHILDREN"].append(NODE_LABEL)

    # make the nodes
    for NODE_LABEL, item in self.added_nodes.items():
      nc = con(NODE_LABEL=NODE_LABEL, **item)
      self.add_node_direct(nc)

  def get_edges(self):
    edges = []
    for node in self.nodes:
      for NODE_CHILD in self.nodes[node].NODE_CHILDREN:
        edges.append([self.nodes[node].NODE_LABEL, NODE_CHILD])

    return edges

  def _san_desc(self, string):
    try:
      return '\n'.join(string)
    except:
      return string

  def get_descriptions(self):
    return {
      label: self._san_desc(node.description) for label, node in self.nodes.items()
    }

# -----------------------------------------------------------------------------

show_old = False

flags_M1 = 'm1_task_list (old): (pm->adaptive, pm->multilevel)'

cc_M1 = col_con()

cc_M1.add_node_unparsed(
  NODE_LABEL='CALC_FIDU',
  NODE_PARENTS=None,
  description=['CalcFiducialVelocity',
              r'<[...]>'])

cc_M1.add_node_unparsed(
  NODE_LABEL='CALC_CLOSURE',
  NODE_PARENTS=["CALC_FIDU",],
  description=['CalcClosure',
              r'<[...]>'])

cc_M1.add_node_unparsed(
  NODE_LABEL='CALC_OPAC',
  NODE_PARENTS=["CALC_CLOSURE",],
  description=['CalcOpacity',
              r'<[...]>'])

cc_M1.add_node_unparsed(
  NODE_LABEL='CALC_GRSRC',
  NODE_PARENTS=["CALC_CLOSURE",],
  description=['CalcGRSources',
              r'<[...]>'])

# com

cc_M1.add_node_unparsed(
  NODE_LABEL='CALC_FLUX',
  NODE_PARENTS=["CALC_CLOSURE",],
  description=['CalcFlux',
              r'<[...]>'])

cc_M1.add_node_unparsed(
  NODE_LABEL='SEND_FLUX',
  NODE_PARENTS=["CALC_FLUX",],
  description=['SendFlux',
              r'<[...]>'])

cc_M1.add_node_unparsed(
  NODE_LABEL='RECV_FLUX',
  NODE_PARENTS=["CALC_FLUX",],
  description=['ReceiveAndCorrectFlux',
              r'<[...]>'])

cc_M1.add_node_unparsed(
  NODE_LABEL='ADD_FLX_DIV',
  NODE_PARENTS=["RECV_FLUX",],
  description=['AddFluxDivergence',
              r'<[...]>'])

# upd
cc_M1.add_node_unparsed(
  NODE_LABEL='CALC_UPDATE',
  NODE_PARENTS=["ADD_FLX_DIV","CALC_OPAC", "CALC_GRSRC"],
  description=['CalcUpdate',
              r'<[...]>'])


# com 2

cc_M1.add_node_unparsed(
  NODE_LABEL='SEND',
  NODE_PARENTS=["CALC_UPDATE",],
  description=['Send',
              r'<[...]>'])

cc_M1.add_node_unparsed(
  NODE_LABEL='RECV',
  NODE_PARENTS=["CALC_UPDATE",],
  description=['Receive',
              r'<[...]>'])


cc_M1.add_node_unparsed(
  NODE_LABEL='SETB',
  NODE_PARENTS=["RECV","CALC_UPDATE"],
  description=['SetBoundaries',
              r'<[...]>'])


if "multilevel" in flags_M1:
  cc_M1.add_node_unparsed(
    NODE_LABEL='PROLONG',
    NODE_PARENTS=["SEND","SETB"],
    description=['Prolongation',
                 r'<[...]>'])

  cc_M1.add_node_unparsed(
    NODE_LABEL='PHY_BVAL',
    NODE_PARENTS=["PROLONG"],
    description=['PhysicalBoundary',
                 r'<[...]>'])

else:
  cc_M1.add_node_unparsed(
    NODE_LABEL='PHY_BVAL',
    NODE_PARENTS=["SETB",],
    description=['PhysicalBoundary',
                 r'<[...]>'])


cc_M1.add_node_unparsed(
  NODE_LABEL='USERWORK',
  NODE_PARENTS=["PHY_BVAL"],
  description=['UserWork',
              r'<[...]>'])

cc_M1.add_node_unparsed(
  NODE_LABEL='NEW_DT',
  NODE_PARENTS=["PHY_BVAL"],
  description=['NewBlockTimeStep',
              r'<[...]>'])

if "adaptive" in flags_M1:
  cc_M1.add_node_unparsed(
    NODE_LABEL='FLAG_AMR',
    NODE_PARENTS=["USERWORK"],
    description=['CheckRefinement',
                r'<[...]>'])

  cc_M1.add_node_unparsed(
    NODE_LABEL='CLEAR_ALLBND',
    NODE_PARENTS=["FLAG_AMR"],
    description=['ClearAllBoundary',
                r'<[...]>'])


else:
  cc_M1.add_node_unparsed(
    NODE_LABEL='CLEAR_ALLBND',
    NODE_PARENTS=["NEW_DT"],
    description=['ClearAllBoundary',
                r'<[...]>'])


cc_M1.node_parse_added()


# assemble --------------------------------------------------------------------
edges_M1 = cc_M1.get_edges()

g_M1 = _nx.DiGraph()
g_M1.add_edges_from(edges_M1, label='some_label')


g_M1_pos = _nx.nx_agraph.graphviz_layout(g_M1, prog='dot')
for p, vals in g_M1_pos.items():
  g_M1_pos[p] = (vals[0], vals[1] + 55)

gfig = _plt.figure(1)
_nx.draw_networkx(g_M1,
  # pos=_nx.planar_layout(g_M1),
  pos=g_M1_pos,
  node_shape='s',
  node_size=5,
  node_color='black',
  arrowsize=13,
  with_labels=False)


info_nodes_M1 = cc_M1.get_descriptions()

for p, vals in g_M1_pos.items():
  try:
    p = info_nodes_M1[p]
  except:
    pass

  _plt.text(vals[0]+10, vals[1], p,
    fontsize=10,
    horizontalalignment='left',
    verticalalignment='center',
    linespacing=0.7,
    bbox=dict(facecolor='white', alpha=0.6, boxstyle='Round', pad=0.1))


gax = gfig.gca()
spines = ['top', 'left', 'right', 'bottom']
for spine in spines:
  gax.spines[spine].set_visible(False)

gax.set_title(flags_M1, y=0.98, fontsize=10, fontweight='bold')
# gax.text(
#   0.02, 1,
#   r'Start_state: $(u,\,w;\,{}^A u,\,{}^Z u,\,{}^A\mathcal{S}){}^{n}$',
#   fontsize=10,
#   horizontalalignment='left',
#   verticalalignment='top',
#   linespacing=0.7,
#   bbox=dict(facecolor='white', alpha=0.8, boxstyle='Round', pad=0.1),
#   transform=gax.transAxes)

# gax.text(
#   0.02, 0.1,
#   r'End_state: $(u,\,w;\,{}^A u,\,{}^Z u,\,{}^A\mathcal{S}){}^{n+1}$',
#   fontsize=10,
#   horizontalalignment='left',
#   verticalalignment='top',
#   linespacing=0.7,
#   bbox=dict(facecolor='white', alpha=0.8, boxstyle='Round', pad=0.1),
#   transform=gax.transAxes)

# _plt.tight_layout()

_plt.show()

#
# :D
#