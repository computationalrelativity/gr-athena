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

MAGNETIC_FIELDS_ENABLED = True
NSCALARS = 1

MODE = "GRMHD" if MAGNETIC_FIELDS_ENABLED else "GRHD"

flags_Mat = f'z4c_matter_task_list ({MODE}):'
flags_Mat +=' (pm->adaptive, pm->multilevel)'

cc_Mat = col_con()

# -----------------------------------------------------------------------------
cc_Mat.add_node_unparsed(
  NODE_LABEL='DIFFUSE_HYD',
  NODE_PARENTS=None,
  description=['DiffuseHydro'])

if MAGNETIC_FIELDS_ENABLED:
  cc_Mat.add_node_unparsed(
    NODE_LABEL='DIFFUSE_FLD',
    NODE_PARENTS=None,
    description=['DiffuseField'])

  cc_Mat.add_node_unparsed(
    NODE_LABEL='CALC_HYDFLX',
    NODE_PARENTS=["DIFFUSE_HYD", "DIFFUSE_FLD"],
    description=['CalculateHydroFlux'])
else:
  cc_Mat.add_node_unparsed(
    NODE_LABEL='CALC_HYDFLX',
    NODE_PARENTS=["DIFFUSE_HYD"],
    description=['CalculateHydroFlux'])

if NSCALARS > 0:
  cc_Mat.add_node_unparsed(
    NODE_LABEL='DIFFUSE_SCLR',
    NODE_PARENTS=None,
    description=['DiffuseScalars'])

  cc_Mat.add_node_unparsed(
    NODE_LABEL='CALC_SCLRFLX',
    NODE_PARENTS=["CALC_HYDFLX", "DIFFUSE_SCLR"],
    description=['CalculateScalarFlux'])

if "multilevel" in flags_Mat:
  cc_Mat.add_node_unparsed(
    NODE_LABEL='SEND_HYDFLX',
    NODE_PARENTS=["CALC_HYDFLX"],
    description=['SendHydroFlux'])

  cc_Mat.add_node_unparsed(
    NODE_LABEL='RECV_HYDFLX',
    NODE_PARENTS=["CALC_HYDFLX"],
    description=['ReceiveAndCorrectHydroFlux'])

  cc_Mat.add_node_unparsed(
    NODE_LABEL='INT_HYD',
    NODE_PARENTS=["RECV_HYDFLX"],
    description=['IntegrateHydro'])

else:
  cc_Mat.add_node_unparsed(
    NODE_LABEL='INT_HYD',
    NODE_PARENTS=["CALC_HYDFLX"],
    description=['IntegrateHydro'])

# -----------------------------------------------------------------------------
cc_Mat.add_node_unparsed(
  NODE_LABEL='SRCTERM_HYD',
  NODE_PARENTS=["INT_HYD"],
  description=['AddSourceTermsHydro'])

cc_Mat.add_node_unparsed(
  NODE_LABEL='SEND_HYD',
  NODE_PARENTS=["SRCTERM_HYD"],
  description=['SendHydro'])

cc_Mat.add_node_unparsed(
  NODE_LABEL='RECV_HYD',
  NODE_PARENTS=["INT_HYD"],
  description=['ReceiveHydro'])

cc_Mat.add_node_unparsed(
  NODE_LABEL='SETB_HYD',
  NODE_PARENTS=["RECV_HYD", "SRCTERM_HYD"],
  description=['SetBoundariesHydro'])

if (NSCALARS > 0):
  if "multilevel" in flags_Mat:
    cc_Mat.add_node_unparsed(
      NODE_LABEL='SEND_SCLRFLX',
      NODE_PARENTS=["CALC_SCLRFLX"],
      description=['SendScalarFlux'])
    cc_Mat.add_node_unparsed(
      NODE_LABEL='RECV_SCLRFLX',
      NODE_PARENTS=["CALC_SCLRFLX"],
      description=['ReceiveScalarFlux'])
    cc_Mat.add_node_unparsed(
      NODE_LABEL='INT_SCLR',
      NODE_PARENTS=["RECV_SCLRFLX"],
      description=['IntegrateScalars'])
  else:
    cc_Mat.add_node_unparsed(
      NODE_LABEL='INT_SCLR',
      NODE_PARENTS=["CALC_SCLRFLX"],
      description=['IntegrateScalars'])

  cc_Mat.add_node_unparsed(
    NODE_LABEL='SEND_SCLR',
    NODE_PARENTS=["INT_SCLR"],
    description=['SendScalars'])

  cc_Mat.add_node_unparsed(
    NODE_LABEL='RECV_SCLR',
    NODE_PARENTS=None,
    description=['ReceiveScalars'])

  cc_Mat.add_node_unparsed(
    NODE_LABEL='SETB_SCLR',
    NODE_PARENTS=["RECV_SCLR", "INT_SCLR"],
    description=['SetBoundariesScalars'])

if MAGNETIC_FIELDS_ENABLED:
  cc_Mat.add_node_unparsed(
    NODE_LABEL='CALC_FLDFLX',
    NODE_PARENTS=["CALC_HYDFLX"],
    description=['CalculateEMF'])

  cc_Mat.add_node_unparsed(
    NODE_LABEL='SEND_FLDFLX',
    NODE_PARENTS=["CALC_FLDFLX"],
    description=['SendEMF'])

  cc_Mat.add_node_unparsed(
    NODE_LABEL='RECV_FLDFLX',
    NODE_PARENTS=["SEND_FLDFLX"],
    description=['ReceiveAndCorrectEMF'])

  cc_Mat.add_node_unparsed(
    NODE_LABEL='INT_FLD',
    NODE_PARENTS=["RECV_FLDFLX"],
    description=['IntegrateField'])

  cc_Mat.add_node_unparsed(
    NODE_LABEL='SEND_FLD',
    NODE_PARENTS=["INT_FLD"],
    description=['SendField'])

  cc_Mat.add_node_unparsed(
    NODE_LABEL='RECV_FLD',
    NODE_PARENTS=None,
    description=['ReceiveField'])

  cc_Mat.add_node_unparsed(
    NODE_LABEL='SETB_FLD',
    NODE_PARENTS=["RECV_FLD", "INT_FLD"],
    description=['SetBoundariesField'])


  if "multilevel" in flags_Mat:
    if (NSCALARS > 0):
      cc_Mat.add_node_unparsed(
        NODE_LABEL='PROLONG_HYD',
        NODE_PARENTS=["SEND_HYD", "SETB_HYD", "SEND_FLD", "SETB_FLD", "SEND_SCLR", "SETB_SCLR", "Z4C_TO_ADM"],
        description=['Prolongation_Hyd'])
    else:
      cc_Mat.add_node_unparsed(
        NODE_LABEL='PROLONG_HYD',
        NODE_PARENTS=["SEND_HYD", "SETB_HYD", "SEND_FLD", "SETB_FLD", "Z4C_TO_ADM"],
        description=['Prolongation_Hyd'])

    cc_Mat.add_node_unparsed(
      NODE_LABEL='CONS2PRIM',
      NODE_PARENTS=["PROLONG_HYD", "Z4C_TO_ADM"],
      description=['Primitives'])
  else:
    if (NSCALARS > 0):
      cc_Mat.add_node_unparsed(
        NODE_LABEL='CONS2PRIM',
        NODE_PARENTS=["SETB_HYD", "SETB_FLD", "SETB_SCLR"],
        description=['Primitives'])
    else:
      cc_Mat.add_node_unparsed(
        NODE_LABEL='CONS2PRIM',
        NODE_PARENTS=["SETB_HYD", "SETB_FLD", "Z4C_TO_ADM"],
        description=['Primitives'])

else: # HYDRO
  if "multilevel" in flags_Mat:
    if (NSCALARS > 0):
      cc_Mat.add_node_unparsed(
        NODE_LABEL='PROLONG_HYD',
        NODE_PARENTS=["SEND_HYD", "SETB_HYD", "SETB_SCLR", "SEND_SCLR", "Z4C_TO_ADM"],
        description=['Prolongation_Hyd'])
    else:
      cc_Mat.add_node_unparsed(
        NODE_LABEL='PROLONG_HYD',
        NODE_PARENTS=["SEND_HYD", "SETB_HYD"],
        description=['Prolongation_Hyd'])

    cc_Mat.add_node_unparsed(
      NODE_LABEL='CONS2PRIM',
      NODE_PARENTS=["Z4C_TO_ADM"],
      description=['Primitives'])

  else:
    if (NSCALARS > 0):
      cc_Mat.add_node_unparsed(
        NODE_LABEL='CONS2PRIM',
        NODE_PARENTS=["SETB_HYD", "SETB_SCLR"],
        description=['Primitives'])
    else:
      cc_Mat.add_node_unparsed(
        NODE_LABEL='CONS2PRIM',
        NODE_PARENTS=["Z4C_TO_ADM"],
        description=['Primitives'])

# -----------------------------------------------------------------------------
cc_Mat.add_node_unparsed(
  NODE_LABEL='PHY_BVAL_HYD',
  NODE_PARENTS=["CONS2PRIM"],
  description=['PhysicalBoundary_Hyd'])

cc_Mat.add_node_unparsed(
  NODE_LABEL='UPDATE_SRC',
  NODE_PARENTS=["PHY_BVAL_HYD"],
  description=['UpdateSource'])

cc_Mat.add_node_unparsed(
  NODE_LABEL='CALC_Z4CRHS',
  NODE_PARENTS=None,
  description=['CalculateZ4cRHS'])

cc_Mat.add_node_unparsed(
  NODE_LABEL='INT_Z4C',
  NODE_PARENTS=["CALC_Z4CRHS"],
  description=['IntegrateZ4c'])

cc_Mat.add_node_unparsed(
  NODE_LABEL='SEND_Z4C',
  NODE_PARENTS=["INT_Z4C"],
  description=['SendZ4c'])

if MAGNETIC_FIELDS_ENABLED:
  cc_Mat.add_node_unparsed(
    NODE_LABEL='RECV_Z4C',
    NODE_PARENTS=["INT_Z4C", "RECV_HYD", "RECV_FLD", "RECV_FLDFLX"],
    description=['ReceiveZ4c'])
else:
  cc_Mat.add_node_unparsed(
    NODE_LABEL='RECV_Z4C',
    NODE_PARENTS=["INT_Z4C"],
    description=['ReceiveZ4c'])

cc_Mat.add_node_unparsed(
  NODE_LABEL='SETB_Z4C',
  NODE_PARENTS=["RECV_Z4C"],
  description=['SetBoundariesZ4c'])

if "multilevel" in flags_Mat:
  cc_Mat.add_node_unparsed(
    NODE_LABEL='PROLONG_Z4C',
    NODE_PARENTS=["SEND_Z4C", "SETB_Z4C"],
    description=['Prolongation_Z4c'])
  cc_Mat.add_node_unparsed(
    NODE_LABEL='PHY_BVAL_Z4C',
    NODE_PARENTS=["PROLONG_Z4C"],
    description=['PhysicalBoundary_Z4c'])
else:
  cc_Mat.add_node_unparsed(
    NODE_LABEL='PHY_BVAL_Z4C',
    NODE_PARENTS=["SETB_Z4C"],
    description=['PhysicalBoundary_Z4c'])

cc_Mat.add_node_unparsed(
  NODE_LABEL='ALG_CONSTR',
  NODE_PARENTS=["PHY_BVAL_Z4C"],
  description=['EnforceAlgConstr'])

if MAGNETIC_FIELDS_ENABLED:
  cc_Mat.add_node_unparsed(
    NODE_LABEL='Z4C_TO_ADM',
    NODE_PARENTS=["ALG_CONSTR", "INT_HYD", "INT_FLD"],
    description=['Z4cToADM'])
else:
  if "multilevel" in flags_Mat:
    cc_Mat.add_node_unparsed(
      NODE_LABEL='Z4C_TO_ADM',
      NODE_PARENTS=["ALG_CONSTR", "PROLONG_HYD"],
      description=['Z4cToADM'])
  else:
    cc_Mat.add_node_unparsed(
      NODE_LABEL='Z4C_TO_ADM',
      NODE_PARENTS=["ALG_CONSTR", "SETB_HYD"],
      description=['Z4cToADM'])

# -----------------------------------------------------------------------------
cc_Mat.add_node_unparsed(
  NODE_LABEL='ADM_CONSTR',
  NODE_PARENTS=["UPDATE_SRC"],
  description=['ADM_Constraints'])

cc_Mat.add_node_unparsed(
  NODE_LABEL='Z4C_WEYL',
  NODE_PARENTS=["Z4C_TO_ADM"],
  description=['Z4c_Weyl'])

cc_Mat.add_node_unparsed(
  NODE_LABEL='WAVE_EXTR',
  NODE_PARENTS=["Z4C_WEYL"],
  description=['WaveExtract'])

cc_Mat.add_node_unparsed(
  NODE_LABEL='USERWORK',
  NODE_PARENTS=["ADM_CONSTR"],
  description=['UserWork'])

cc_Mat.add_node_unparsed(
  NODE_LABEL='NEW_DT',
  NODE_PARENTS=["USERWORK"],
  description=['NewBlockTimeStep'])

if "adaptive" in flags_Mat:
  cc_Mat.add_node_unparsed(
    NODE_LABEL='FLAG_AMR',
    NODE_PARENTS=["USERWORK"],
    description=['CheckRefinement'])
  cc_Mat.add_node_unparsed(
    NODE_LABEL='CLEAR_ALLBND',
    NODE_PARENTS=["FLAG_AMR"],
    description=['ClearAllBoundary'])
else:
  cc_Mat.add_node_unparsed(
    NODE_LABEL='CLEAR_ALLBND',
    NODE_PARENTS=["NEW_DT"],
    description=['ClearAllBoundary'])

cc_Mat.node_parse_added()


# assemble --------------------------------------------------------------------
edges_Mat = cc_Mat.get_edges()

g_Mat = _nx.DiGraph()
g_Mat.add_edges_from(edges_Mat, label='some_label')


g_Mat_pos = _nx.nx_agraph.graphviz_layout(g_Mat, prog='dot')
for p, vals in g_Mat_pos.items():
  g_Mat_pos[p] = (vals[0], vals[1] + 55)

gfig = _plt.figure(1)
_nx.draw_networkx(g_Mat,
  # pos=_nx.planar_layout(g_Mat),
  pos=g_Mat_pos,
  node_shape='s',
  node_size=5,
  node_color='black',
  arrowsize=13,
  with_labels=False)


info_nodes_Mat = cc_Mat.get_descriptions()

for p, vals in g_Mat_pos.items():
  try:
    p = info_nodes_Mat[p]
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

gax.set_title(flags_Mat, y=0.98, fontsize=10, fontweight='bold')
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

_plt.tight_layout()

_plt.show()

#
# :D
#