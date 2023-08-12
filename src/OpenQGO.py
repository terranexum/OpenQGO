
#import folium
#import fiona 

#import openqaoa

import networkx as nx
#import geonetworkx as gnx
#import pyproj
#import rtree
#import shapely
#import geojson
#import matplotlib as mpl
import matplotlib.pyplot as pyp
#import geopandas as gpd
#import geodatasets
#import dimod

#import libpysal
#from libpysal import weights

#import numpy as np
#import sklearn
#list


#import numpy as np
#import numpy.random as random

#import more_itertools

    
#import cplex
#import sympy as sp

#import matplotlib.pyplot as plt

#from math import acos

# Printing configuration
from sympy.interactive import printing
printing.init_printing(use_latex=True)
#from IPython.display import display, Markdown


#from openqaoa.problems import MaximumCut, NumberPartition, MinimumVertexCover, QUBO
#from openqaoa.utilities import plot_graph, ground_state_hamiltonian
#from openqaoa.qaoa_components import Hamiltonian


#import the QAOA workflow model
#from openqaoa import QAOA

#import method to specify the device
#from openqaoa.backends import create_device



#from sklearn.metrics import pairwise_distances

#from docplex.mp.model import Model
#from docplex.mp.model_reader import ModelReader

#from qiskit_optimization import QuadraticProgram
#from qiskit_optimization.algorithms import CplexOptimizer
#from qiskit_optimization.translators import from_docplex_mp



#from qiskit_optimization.converters import (
#    InequalityToEquality,     # converts inequality constraints to equality constraints by adding slack variables
#    LinearEqualityToPenalty,  # converts linear equality constraints to quadratic penalty terms 
#    IntegerToBinary,          # converts integer variables to binary variables
#    QuadraticProgramToQubo    # combines the previous three converters
#)

#from qiskit_optimization.algorithms.admm_optimizer import ADMMParameters, ADMMOptimizer
#from qiskit.algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
#from qiskit.algorithms.optimizers import COBYLA
#from qiskit.primitives import Sampler
#from qiskit_optimization.algorithms import CobylaOptimizer, MinimumEigenOptimizer
#from qiskit_optimization.algorithms import CplexOptimizer



pyp.rcParams["figure.figsize"] = (20, 20)

from QGONexum import QGOProblem, QGOGraph, QGOOptimizer
from QGOOut import QGOExporter

# paths to input data in data_in
links = ["/home/OpenQGO/data_in/mineral_production_facilities.geojson"]
links.append("/home/OpenQGO/data_in/mineral_sources.geojson")
links.append("/home/OpenQGO/data_in/possible_solar.geojson")

# forming the QGOProblem using QAOA
cutoffs = [5, 5, 10]
prefixes = ["possible_solar", "powerplants", "high_schools"]
ccc_problem = QGOProblem(links, prefixes, cutoffs)

# generating the graph network represented by the three input data files
thresholds = [3, 10]
ccc_graph = QGOGraph(ccc_problem, thresholds)
ccc_graph.createGraph()
ccc_graph.draw()

# optimizing the graph network using QAOA as the problem solver
ccc_optimize = QGOOptimizer()
ccc_optimize.optimize(problem=ccc_problem, QGOGraph=ccc_graph)

# getting the solution graph
ccc_sol_graph = ccc_optimize.createSolutionGraph(ccc_problem, ccc_graph)
print(ccc_sol_graph)

# exporting both the problem and solution graphs as GeoJSONs for UI display
exporter = QGOExporter(ccc_problem, ccc_graph, ccc_sol_graph)
