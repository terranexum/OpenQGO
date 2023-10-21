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