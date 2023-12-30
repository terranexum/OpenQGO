from QAOANode import QAOANode
import geopandas as gpd
import numpy as np
import networkx as nx
import matplotlib.pyplot as pyp
from sklearn.metrics import pairwise_distances



def combineNodeCoords(node_coords_1, node_coords_2):
    
    return {**node_coords_1, **node_coords_2}

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


#this class defines a QGOProblem through nodes and edges
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

            #when we name the node, we want to format its name in a specific way
            node = QAOANode(row, f"{prefix} {idx}")

            node_list.append(node)

        return node_list
    
    def getNodeCoordDict(self, node_list: list[QAOANode]) -> dict[QAOANode, tuple[float, float]]:
        

        #this is a very useful dictionary to have as we'll often need to access node to coordinate information directly
        node_coord_dict = {}

        node: QAOANode
        for node in node_list:

            node_coord_dict[node] = node.coordinates

        return node_coord_dict

    def getCoordList(self, node_list: list[QAOANode]) -> list[tuple[float, float]]:
    
        #these are just the coordinates in each node (this could probably get replaced)
        node_coords: list[tuple[float, float]] = []

        for node in node_list:
            node_coords.append(node.coordinates)

        return node_coords
    
    def splitNodes(node_list, idx_split):
    
        #sometimes we want a small sample case to work with to check for errors, so it's nice to split the node list

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
    
    #directional graph for showing how nodes lead into each other
    graph: nx.DiGraph = nx.DiGraph()

    def __init__(self, qgo_problem: QGOProblem, thresholds: list[int]):

        self.qgo_problem = qgo_problem
        self.thresholds = thresholds
        
        self.setEdgeListsCapacitiesLabels(thresholds=thresholds)

        self.setAllEdgeCapacities()
        
        self.setEdgeListNames()
        #print(self.edge_list_names)
    
    def getSquareMatrix(self, old_matrix: np.ndarray) -> np.ndarray:
    

        #must be square matrix, cannot have different sized rows and columns
        rows, cols = old_matrix.shape
        max_dim = max(rows, cols)


        #fill the square matrix with zeroes at the start, and then copy in our values from the old matrix
        square_mat = np.zeros((max_dim, max_dim))
        square_mat[:rows, :cols] = old_matrix

        return square_mat

    def applyThreshold(self, matrix: np.ndarray, threshold: int):

        #if the matrix value is greater than some value, we deem it at insignificant and set it to 0
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
                
                #matrix that defines distance between each node in the first coordinate list and the second coordinate list
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
            #just a formatting function
            edge_names[edge_list[i]] = "edge" + str(i)
        
        return edge_names
    
    def setEdgeListNames(self):
      self.edge_list_names = self.getEdgeNames([*self.all_edge_capacities.keys()])  

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