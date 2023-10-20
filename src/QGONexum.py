#import pandas
#import pyogrio 
#import requests

#import folium
#import fiona 

#import openqaoa

import networkx as nx
from networkx.classes.reportviews import OutEdgeView, InEdgeView
#import geonetworkx as gnx
#import pyproj
#import rtree
#import shapely
#import geojson
#import matplotlib as mpl
import matplotlib.pyplot as pyp
import geopandas as gpd
#import geodatasets
#import dimod

#import libpysal
#from libpysal import weights

import numpy as np
from qiskit_optimization import QuadraticProgram
#import sklearn

from sklearn.metrics import pairwise_distances



#import numpy as np
import numpy.random as random

#import more_itertools

    
import cplex
#import sympy as sp

#import matplotlib.pyplot as plt


import qiskit

#from math import acos

# Printing configuration
from sympy.interactive import printing
printing.init_printing(use_latex=True)
#from IPython.display import display, Markdown


#from openqaoa.problems import MaximumCut, NumberPartition, MinimumVertexCover, QUBO
#from openqaoa.utilities import plot_graph, ground_state_hamiltonian
#from openqaoa.qaoa_components import Hamiltonian


#import the QAOA workflow model
from openqaoa import QAOA

#import method to specify the device
#from openqaoa.backends import create_device

#import more_itertools

#rom qiskit import Aer, execute, QuantumCircuit
#from docplex.mp.model import Model
from docplex.mp.model_reader import ModelReader

#from qiskit_optimization import QuadraticProgram
#from qiskit_optimization.algorithms import CplexOptimizer
from qiskit_optimization.translators import from_docplex_mp

from qiskit_optimization.converters import (
    InequalityToEquality,     # converts inequality constraints to equality constraints by adding slack variables
    LinearEqualityToPenalty,  # converts linear equality constraints to quadratic penalty terms 
    IntegerToBinary,          # converts integer variables to binary variables
    QuadraticProgramToQubo    # combines the previous three converters
)


from qiskit_optimization.algorithms.admm_optimizer import ADMMParameters, ADMMOptimizer, ADMMOptimizationResult
from qiskit.algorithms.minimum_eigensolvers import QAOA # NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import CobylaOptimizer, MinimumEigenOptimizer
#from qiskit_optimization.algorithms import CplexOptimizer



class QAOANode:

    neighbors = []

    def __init__(self, point: gpd.GeoSeries, name: str):
        self.point = point
        self.coordinates: tuple[float, float] = ()
        try:
            self.coordinates = (point["geometry"].x, point["geometry"].y)
        except: 
            self.coordinates = (point["geometry"].centroid.x, point["geometry"].centroid.y)
            
        self.name = name
    
    def setCoordinates(self, coordinates: tuple[float, float]):
        self.coordinates = coordinates

    def getCoordinates(self) -> tuple[float, float]:
        return self.coordinates
    
    def changeName(self, new_name: str):
        self.name = new_name
        
    def __repr__(self):
        return self.name


def getEdgeNames(edge_list):
    
    edge_names = {}
    
    for i in range(len(edge_list)):
        
        edge_names[edge_list[i]] = "edge" + str(i)
    
    return edge_names

def tupleToStr(tuple):

    returnStr = "("
    for element in tuple:
        returnStr += element + ", "
    
    return returnStr + ")"

def getEdgesAndCapacities(
        nodes_1: list[QAOANode],
        nodes_2: list[QAOANode],
        binaryMatrix: np.ndarray,
        scale_factor: int
    ) -> tuple[
        list[tuple[QAOANode, QAOANode]],
        dict[tuple[QAOANode, QAOANode], float],
        dict[tuple[QAOANode, QAOANode], str]
    ]:
    edge_list: list[tuple[QAOANode, QAOANode]] = []

    edge_labels: dict[tuple[QAOANode, QAOANode], str] = {}
    edge_capacities: dict[tuple[QAOANode, QAOANode], float] = {}
    
    #for each node in nodes_1, iterate over each node in nodes_2
    i: QAOANode
    for i in range(len(nodes_1)):
        j: QAOANode
        for j in range(len(nodes_2)):
            #if these two nodes are within our distance threshold - create edge
            if(binaryMatrix[i, j] > 0):
                
                edge: tuple[QAOANode, QAOANode] = (nodes_1[i], nodes_2[j])
                edge_list.append(edge)
                
                edge_capacities[edge] = (1 / binaryMatrix[i, j]) * scale_factor
                edge_labels[edge] = str(round(edge_capacities[edge], 2))
                
    
    return edge_list, edge_capacities, edge_labels
    

edge_list = []
edge_capacities = {}

def getNodeLabels(node_list):
    
    node_labels = {}
    
    for node in node_list:
        node_labels[node] = node.name
        
    return node_labels


def normalizeEdgeCapacities(edge_capacities):
    
    
    min_val = np.min(np.array(list(edge_capacities.values())))
    max_val = np.max(np.array(list(edge_capacities.values())))
    
    capacity_range = max_val - min_val
    
    for key in edge_capacities.keys():
        edge_capacities[key] = (edge_capacities[key] - min_val) / capacity_range
    


def combineNodeCoords(node_coords_1, node_coords_2):
    
    return {**node_coords_1, **node_coords_2}


class QGOProblem:

    #GeoJSON Files
    files: list[gpd.GeoDataFrame] = []

    #Node Lists for each GeoJSON File
    nodes_lists: list[list[QAOANode]] = []

    #Coordinate Lists for each GeoJSON File
    coords_lists: list[tuple[float, float]] = []

    #Node Coord Dictionaries for each GeoJSON File
    ncds: list[dict[QAOANode, tuple[float, float]]] = []

    #Prefixes for the names of nodes for each GeoJSON File
    prefixes: list[str] = []

    def __init__(self, links: list[str], prefixes: list[str], cutoffs: list[str]):

        self.setFiles(links)
        self.prefixes = prefixes
        
        self.setNodeLists(cutoffs, prefixes)
        self.setCoordsLists()
        self.setNodeCoordDictionaries()
    
    def setFiles(self, links: list[str]):

        for link in links:
            with open(link) as f:
                self.files.append(gpd.read_file(f))

    def getNodes(self, geo_file: gpd.GeoDataFrame, prefix: str, idx_cutoff=np.inf) -> list[QAOANode]:
    
        node_list: list[QAOANode] = []

        row: gpd.GeoSeries
        for idx, row in geo_file.iterrows():

            if idx == idx_cutoff: break

            node = QAOANode(row, f"{prefix} {idx}")

            node_list.append(node)

        return node_list
    
    def getNodeCoordDict(self, node_list: list[QAOANode]) -> dict[QAOANode, tuple[float, float]]:
        
        node_coord_dict = {}

        node: QAOANode
        for node in node_list:

            node_coord_dict[node] = node.coordinates

        return node_coord_dict

    def getCoordList(self, node_list: list[QAOANode]) -> list[tuple[float, float]]:
    
        node_coords: list[tuple[float, float]] = []

        for node in node_list:
            node_coords.append(node.coordinates)

        return node_coords
    
    def splitNodes(node_list, idx_split):
    
        new_list = node_list[:idx_split]
        del node_list[:idx_split]

        return new_list
    
    def renameNodes(node_list, prefix):
    
        for idx, node in enumerate(node_list):
            node.changeName(prefix + " " + str(idx))
        
    
    def setNodeLists(self, cutoffs: list[str], prefixes: list[str]):
        
        for i, file in enumerate(self.files):
            
            self.nodes_lists.append(self.getNodes(file, self.prefixes[i], cutoffs[i]))


    def setNodeCoordDictionaries(self):
        node_list: list[QAOANode]
        for node_list in self.nodes_lists:

            self.ncds.append(self.getNodeCoordDict(node_list))

    def setCoordsLists(self):
        node_list: list[QAOANode]
        for node_list in self.nodes_lists:
            self.coords_lists.append(self.getCoordList(node_list))


class QGOGraph:

    qgo_problem: QGOProblem = None

    edges_lists: list[tuple[list[tuple[QAOANode, QAOANode]]]] = []
    edge_list_names: dict[tuple[QAOANode, QAOANode], str] = {}

    edges_capacities: list[dict[tuple[QAOANode, QAOANode], float]] = []
    all_edge_capacities: dict[tuple[QAOANode, QAOANode], float] = {}

    edges_labels: list[dict[tuple[QAOANode, QAOANode], str]] = []
    
    graph: nx.DiGraph = nx.DiGraph()

    def __init__(self, qgo_problem: QGOProblem, thresholds: list[int]):

        self.qgo_problem = qgo_problem
        self.thresholds = thresholds
        
        self.setEdgeListsCapacitiesLabels(thresholds=thresholds)

        self.setAllEdgeCapacities()
        
        self.setEdgeListNames()
        #print(self.edge_list_names)
    
    def getSquareMatrix(self, old_matrix: np.ndarray) -> np.ndarray:
    
        rows, cols = old_matrix.shape
        max_dim = max(rows, cols)

        square_mat = np.zeros((max_dim, max_dim))
        square_mat[:rows, :cols] = old_matrix

        return square_mat

    def applyThreshold(self, matrix: np.ndarray, threshold: int):
        matrix[matrix > threshold] = 0
    
    def setEdgeListsCapacitiesLabels(self, thresholds: list[int]):
        
        nodes_lists = self.qgo_problem.nodes_lists
        coords_lists = self.qgo_problem.coords_lists
        
        for i, _ in enumerate(coords_lists):
            
            if i < len(coords_lists) - 1:
                first_coords_list: tuple[float, float] = coords_lists[i]
                second_coords_list: tuple[float, float] = coords_lists[i + 1]
                
                first_nodes_list: list[QAOANode] = nodes_lists[i]
                second_nodes_list: list[QAOANode] = nodes_lists[i + 1]
                
                dist_matrix: np.ndarray = self.getSquareMatrix(pairwise_distances(first_coords_list, second_coords_list, metric='manhattan'))
                self.applyThreshold(dist_matrix, thresholds[i])
                
                edge_list: tuple[list[tuple[QAOANode, QAOANode]]]
                edge_capacities: dict[tuple[QAOANode, QAOANode], float]
                edge_labels: dict[tuple[QAOANode, QAOANode], str]
                edge_list, edge_capacities, edge_labels = getEdgesAndCapacities(first_nodes_list, second_nodes_list, dist_matrix, 1)
                
                self.edges_lists.append(edge_list)
                self.edges_capacities.append(edge_capacities)
                self.edges_labels.append(edge_labels)
    
    def setAllEdgeCapacities(self):

        self.all_edge_capacities = {**self.edges_capacities[0], **self.edges_capacities[1]}
        nx.set_edge_attributes(self.graph, self.all_edge_capacities, "capacity")

    def getEdgeNames(self, edge_list: list[tuple[QAOANode, QAOANode]]) -> dict[tuple[QAOANode, QAOANode], str]:
    
        edge_names = {}
        
        for i in range(len(edge_list)):
            
            edge_names[edge_list[i]] = "edge" + str(i)
        
        return edge_names
    
    def setEdgeListNames(self):

        self.edge_list_names = self.getEdgeNames([*self.all_edge_capacities.keys()])  
        #print(self.edge_list_names)

    def createGraph(self):

        ncds = self.qgo_problem.ncds

        for ncd in ncds:

            self.graph.add_nodes_from(ncd)

        for edge_list in self.edges_lists:

            self.graph.add_edges_from(edge_list)


    def draw(self):
        
        fig, ax = pyp.subplots(1, 1)
        
        first_node_list = self.qgo_problem.nodes_lists[0]
        second_node_list = self.qgo_problem.nodes_lists[1]
        third_node_list = self.qgo_problem.nodes_lists[2]
        
        first_ncd = self.qgo_problem.ncds[0]
        second_ncd = self.qgo_problem.ncds[1]
        third_ncd = self.qgo_problem.ncds[2]
        
        
        nx.draw_networkx_nodes(self.graph, ax=ax, pos=first_ncd, nodelist=first_node_list, node_color="red", node_size=80)
        nx.draw_networkx_nodes(self.graph, ax=ax, pos=second_ncd, nodelist=second_node_list, node_color="blue", node_size=70)
        nx.draw_networkx_nodes(self.graph, ax=ax, pos=third_ncd, nodelist=third_node_list, node_color="black", node_size=40)

        #Draw Edges
        nx.draw_networkx_edges(self.graph, ax=ax, edgelist=self.edges_lists[0], pos=combineNodeCoords(first_ncd, second_ncd), edge_color='green', width=1, arrowsize=10)
        nx.draw_networkx_edges(self.graph, ax=ax, edgelist=self.edges_lists[1], pos=combineNodeCoords(second_ncd, third_ncd), edge_color='purple', width=1, arrowsize=10)


        nx.draw_networkx_edge_labels(self.graph, ax=ax, pos=combineNodeCoords(first_ncd, second_ncd), edge_labels=self.edges_labels[0], font_size=12)
        nx.draw_networkx_edge_labels(self.graph, ax=ax, pos=combineNodeCoords(second_ncd, third_ncd), edge_labels=self.edges_labels[1], font_size=12)
        
        
class QGOOptimizer: 

    solution: ADMMOptimizationResult = None
    
    def getObjectiveFunction(self, graph: nx.DiGraph, source_nodes: list[QAOANode], edge_list_names: dict[tuple[QAOANode, QAOANode], str]) -> list[float]:
    
        obj_func = []

        edge: tuple[QAOANode, QAOANode]
        for edge in edge_list_names.keys():

            out_node = edge[0]

            if out_node in source_nodes:

                obj_func.append(1.0)
            else:

                obj_func.append(0.0)

        return obj_func
    


    def getConstraints(
            self,
            graph: nx.DiGraph,
            intermediary_nodes: list[QAOANode],
            edge_list_names: dict[tuple[QAOANode, QAOANode], str],
            edge_capacities: dict[tuple[QAOANode, QAOANode], float]
        ) -> tuple[
            list[str],
            list[str],
            list[list[list[str], list[int]]],
            list[int]
        ]:

        constraints: list[list[list[str], list[int]]] = [] #gets filled in



        #iterate over intermediate nodes
        for node in intermediary_nodes:

            #get the edges going INTO the powerplants(from the coal mines)
            in_edges: InEdgeView = graph.in_edges(node)

            #get the edges going OUT of the powerplants(into the cities)
            out_edges: OutEdgeView = graph.out_edges(node)

            if len(in_edges) == 0 or len(out_edges) == 0: 

                continue

            all_edges: list[tuple[QAOANode, QAOANode]] = list(in_edges) + list(out_edges)
            names: list[str] = []

            constraint_values: list[int] = []



            #iterate over the in_edges
            for edge in in_edges:

                names.append(edge_list_names[edge])


                #the in_edges need a constraint value of 1.0
                constraint_values.append(1)


            for edge in out_edges:

                names.append(edge_list_names[edge])



                #the out_edges need a constraint value of -1
                constraint_values.append(-1)

            #the constraints are a list of lists - names of edges associated with constraint values
            constraints.append([names, constraint_values])



            constraint_names = []

            for idx, val in enumerate(constraints):

                constraint_names.append(str(idx))

            constraint_senses: list[str] = ['E'] * len(constraints) #should all be equality -- the outflow should never be more than the inflow -- should be equal at most
            righthand_sides: list[int] = [0] * len(constraints) #just zeroes

            # QUESTION: this returns after evaluating only a single intermediary node?
            return constraint_names, constraint_senses, constraints, righthand_sides

    
    #GitHub Gist Link: https://gist.github.com/stevenwalton/601645612161fb6ebf82dd7687de0060 
    #Credit to Stephen Walton and Zayd H for general structure

    # Max Flow example from slides of UO 510: Multi-Agent Systems
    # Class: https://classes.cs.uoregon.edu/19S/cis410mas/

    # Flow moves to the right 
    #      2
    #   1-----3
    # 4/ \     \3
    # /   \     \
    #s     \1    t
    # \     \   /
    # 5\     \ /2
    #   2-----4 
    #      4
    # Nodes {s,1,2,3,4,t}
    # Edges {s1(4), 13(2), 3t(3), 14(1), s2(5), 24(4), 4t(2)}
    # Problem
    # max_x x_{s1} + x_{s2}
    #        (Conservation)             (Capacity)
    # s.t. x_{s1} = x_{13} + x_{14}     x_{s1} <= 4
    #      x_{s2} = x_{24}              x_{s2} <= 5
    #      x_{13} = x_{3t}              x_{13} <= 2
    #      x_{14} + x_{24} = x_{4t}     x_{14} <= 1
    #                                   x_{24} <= 4
    #                                   x_{3t} <= 3 
    #                                   x_{4t} <= 2




    # Initialize

    def getMaxFlowModel(
            self,
            graph: nx.DiGraph,
            source_nodes: list[QAOANode],
            intermediary_nodes: list[QAOANode],
            edge_list_names: dict[tuple[QAOANode, QAOANode], str],
            edge_capacities: dict[tuple[QAOANode, QAOANode], float]
        ) -> cplex.Cplex:

        p = cplex.Cplex()

        #basic problem details -- Max Flow
        p.set_problem_name("Max Flow")
        p.set_problem_type(cplex.Cplex.problem_type.LP)
        p.objective.set_sense(p.objective.sense.maximize) # max problem hence the name Max Flow

        #       0     1     2     3     4     5     6

        #v   = ["s1", "13", "3t", "14", "s2", "24", "4t"]  EQUIVALENT TO EDGE LIST

        #names of the edges are defined in the edge_list_names dictionary
        names: list[tuple[QAOANode, QAOANode]] = [*edge_list_names.values()]

        #the total flow is equal to the inflow - outflow -- inflow is defined by the flow directly following the source node, we want to maximize this hence the 1s
        obj_func: list[float] = self.getObjectiveFunction(graph, source_nodes, edge_list_names)

        #lowest flow value is zero for any edge
        low_bnd: np.ndarray[float]  = np.zeros(shape=len(edge_capacities))

        #upper bounds are the edge capacities
        upr_bnd: list[float]  = [*edge_capacities.values()] # Capacity constraints



        #add model variables
        p.variables.add(obj=obj_func, lb=low_bnd.tolist(), ub=upr_bnd, names=names)



        #set linear constraints
        cnames: list[str]
        csenses: list[str]
        constraints: list[list[list[str], list[int]]]
        rhs: list[int]
        cnames, csenses, constraints, rhs = self.getConstraints(graph, intermediary_nodes, edge_list_names=edge_list_names, edge_capacities=edge_capacities)
        p.linear_constraints.add(lin_expr=constraints,
                                senses = csenses,
                                rhs = rhs,
                                names = cnames)


    #pretty print
    # print("Problem Type: %s" % p.problem_type[p.get_problem_type()])
    # p.solve()
    # print("Solution result is: %s" % p.solution.get_status_string())
    # print(p.solution.get_values())


        return p

    def optimize(self, problem: QGOProblem, qgo_graph: QGOGraph):

        source_nodes: list[QAOANode] = problem.nodes_lists[0]
        intermediary_nodes: list[QAOANode] = problem.nodes_lists[1]
        
        edge_list_names: dict[tuple[QAOANode, QAOANode], str] = qgo_graph.edge_list_names
        all_edge_capacities: dict[tuple[QAOANode, QAOANode], float] = qgo_graph.all_edge_capacities
        
        model: cplex.Cplex = self.getMaxFlowModel(qgo_graph.graph, source_nodes=source_nodes, intermediary_nodes=intermediary_nodes, edge_list_names=edge_list_names, edge_capacities=all_edge_capacities)

        model.write("max_flow_model.lp", filetype='lp')
        m = ModelReader.read("max_flow_model.lp")

        qp: QuadraticProgram = from_docplex_mp(m)
        #print(qp.prettyprint())

        #OPTIMIZERS TO TEST
        #Just trying QAOA or possibly even adding our own unique branch to an already existing QAOA to fit our needs
        #Qiskit, Pennylane
        #QAOA, VQE, QADMM, QPPA, QSGD
        #Accelerate these qiskit optimizers using NVIDIA's QuQuantum

        admm_params = ADMMParameters(rho_initial=1001, beta=1000, factor_c=900, maxiter=500, three_block=True, tol=1.0)
        admm_quantum = ADMMOptimizer(params=admm_params, qubo_optimizer=MinimumEigenOptimizer(QAOA(sampler=Sampler(), optimizer=COBYLA())), continuous_optimizer=CobylaOptimizer())

        self.solution = admm_quantum.solve(qp)
        return str(self.solution)
    
    def createSolutionGraph(self, problem: QGOProblem, graph: QGOGraph) -> nx.DiGraph:
        
        sol_edge_list_names = {}
        sol_edge_list_two_names = {}
        
        first_edge_labels = graph.edges_labels[0]
        second_edge_labels = graph.edges_labels[1]

        for idx, edge in enumerate(first_edge_labels):

            sol_edge_list_names[edge] = str(round(self.solution.x[idx], 2))


        offset = len(first_edge_labels)


        for idx, edge in enumerate(second_edge_labels):

            sol_edge_list_two_names[edge] = str(round(self.solution.x[idx + offset], 2))

            
        fig, ax = pyp.subplots(1, 1)
        
        first_node_list = problem.nodes_lists[0]
        second_node_list = problem.nodes_lists[1]
        third_node_list = problem.nodes_lists[2]
        
        first_ncd = problem.ncds[0]
        second_ncd = problem.ncds[1]
        third_ncd = problem.ncds[2]
        
        sol_graph = nx.DiGraph()
    
    
        #Add Nodes
        sol_graph.add_nodes_from(first_ncd)
        sol_graph.add_nodes_from(second_ncd)
        sol_graph.add_nodes_from(third_ncd)

        #Add Edges 
        sol_graph.add_edges_from(graph.edges_lists[0])
        sol_graph.add_edges_from(graph.edges_lists[1])

        #Draw Nodes
        #nx.draw_networkx_nodes(sol_graph, ax=ax, pos=first_ncd, nodelist=first_node_list, node_color="red", node_size=80)
        #nx.draw_networkx_nodes(sol_graph, ax=ax, pos=second_ncd, nodelist=second_node_list, node_color="black", node_size=70)
        #nx.draw_networkx_nodes(sol_graph, ax=ax, pos=third_ncd, nodelist=third_node_list, node_color="blue", node_size=20)

        #Draw Edges
        #nx.draw_networkx_edges(sol_graph, ax=ax, edgelist=graph.edges_lists[0], pos=combineNodeCoords(first_ncd, second_ncd), edge_color='green', width=1, arrowsize=10)
        #nx.draw_networkx_edges(sol_graph, ax=ax, edgelist=graph.edges_lists[1], pos=combineNodeCoords(second_ncd, third_ncd), edge_color='purple', width=1, arrowsize=10)

        #Draw Edge Labels
        #nx.draw_networkx_edge_labels(sol_graph, ax=ax, pos=combineNodeCoords(first_ncd, second_ncd), edge_labels=sol_edge_list_names, font_size=12)
        #nx.draw_networkx_edge_labels(sol_graph, ax=ax, pos=combineNodeCoords(second_ncd, third_ncd), edge_labels=sol_edge_list_two_names, font_size=12)

        #pyp.show()
        return sol_graph