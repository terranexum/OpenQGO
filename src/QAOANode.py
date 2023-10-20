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