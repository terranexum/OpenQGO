from src.QGONexum import QGOProblem, QGOGraph, QGOOptimizer


class QGOExporter: 

    def __init__(self, qgo_problem, qgo_graph, qgo_sol_graph):

        self.qgo_graph = qgo_graph
        self.qgo_sol_graph = qgo_sol_graph

        self.files = qgo_problem.files
        self.prefixes = qgo_problem.prefixes

        # open the qgo_sol_graph

        # get solution nodes of all 3 types (according to prefix)

        # generate 3 point geojsons from each of the 3 input geojsons containing only those points in the solution

        # get solution edges of both types

        # generate 2 line geojsons from each of the first pair and last pair of edges 

        
        self.setEdgeListsCapacitiesLabels(thresholds=thresholds)
        
        self.setAllEdgeCapacities()
        
        self.setEdgeListNames()
        print(self.edge_list_names)

    def getGeoData(self):
    
        obj_func = []

        for edge in edge_list_names.keys():

            out_node = edge[0]

            if out_node in source_nodes:

                obj_func.append(1.0)
            else:

                obj_func.append(0.0)

        return obj_func