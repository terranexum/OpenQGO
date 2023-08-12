import pandas as pd
import geopandas as gpd
import shapely


class QGOExporter: 

    def __init__(self, qgo_problem, qgo_graph, qgo_sol_graph):

        self.qgo_graph = qgo_graph.graph
        self.qgo_sol_graph = qgo_sol_graph

        self.files = qgo_problem.files
        self.prefixes = qgo_problem.prefixes

        self.outdict = {}

        print("files", self.files)
        print("prefixes", self.prefixes)

        prob_nodes = self.qgo_graph.nodes
        prob_edges = self.qgo_graph.edges
        sol_nodes = self.qgo_sol_graph.nodes
        sol_edges = self.qgo_sol_graph.edges

        objs = {
                "prob_nodes": prob_nodes, 
                "prob_edges": prob_edges, 
                "sol_nodes": sol_nodes, 
                "sol_edges": sol_edges
                }

        def retNodeCoords(node_item):
            return node_item.getCoordinates()
        
        def retEdgeCoords(edge_item):
            edge_coord_list = []
            for node in edge_item:
                print(node)
                edge_coord_list.append(node.getCoordinates())
            return edge_coord_list
        
        def retNames(item):
            return item.__str__()

        # List Setup
        prob_node_coord_list = map(retNodeCoords, objs["prob_nodes"])
        prob_node_name_list = map(retNames, objs["prob_nodes"])
        prob_node_list = zip(prob_node_coord_list, prob_node_name_list)

        prob_edge_coord_list = map(retEdgeCoords, objs["prob_edges"])
        prob_edge_name_list = map(retNames, objs["prob_edges"])
        prob_edge_list = zip(prob_edge_coord_list, prob_edge_name_list)

        sol_node_coord_list = map(retNodeCoords, objs["sol_nodes"])
        sol_node_name_list = map(retNames, objs["sol_nodes"])
        sol_node_list = zip(sol_node_coord_list, sol_node_name_list)

        sol_edge_coord_list = map(retEdgeCoords, objs["sol_edges"])
        sol_edge_name_list = map(retNames, objs["sol_edges"])
        sol_edge_list = zip(sol_edge_coord_list, sol_edge_name_list)

        # List of Dataframes that will later be concatenated into one large dataframe
        pre_problem_dfs = self.get_pre_dfs(prob_node_list, prob_edge_list)
        pre_sol_dfs = self.get_pre_dfs(sol_node_list, sol_edge_list)

        # Concatenating all the separate dataframes into one big DataFrame
        single_prob_df = pd.concat(pre_problem_dfs, ignore_index=True).reset_index(drop=True)
        single_sol_df = pd.concat(pre_sol_dfs, ignore_index=True).reset_index(drop=True)

        # Finally, generating the actual GeoDataFrame that can be manipulated
        geo_prob_df = gpd.GeoDataFrame(single_prob_df, geometry='geometry', crs='epsg:4326')
        geo_sol_df = gpd.GeoDataFrame(single_sol_df, geometry='geometry', crs='epsg:4326')
        
        # Output to file
        geo_prob_df.to_file('/home/OpenQGO/data_out/QGOprob.geojson', driver='GeoJSON')  
        geo_sol_df.to_file('/home/OpenQGO/data_out/QGOsol.geojson', driver='GeoJSON')  

    def get_pre_dfs(self, node_list, edge_list):

        pre_dfs = []

        for node_item in node_list:
            # Generating a shapely geometry
            geometry = shapely.Point(node_item[0])
            name = node_item[1]
            
            # Creating a single-row-DataFrame.
            this_df = pd.DataFrame({'geometry':[geometry],
                                    'name':[name]
                                    })
            
            # Appending this single-row-DataFrame to the `pre_dfs` list
            pre_dfs.append(this_df)

        for edge_item in edge_list:
            # Generating a shapely geometry
            geometry = shapely.LineString(edge_item[0])
            name = edge_item[1]
            
            # Creating a single-row-DataFrame.
            this_df = pd.DataFrame({'geometry':[geometry],
                                    'name':[name]
                                    })
            
            # Appending this single-row-DataFrame to the `pre_dfs` list
            pre_dfs.append(this_df)

        return pre_dfs