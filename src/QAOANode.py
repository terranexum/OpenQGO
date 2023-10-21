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


