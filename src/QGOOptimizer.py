from QAOANode import QAOANode
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


import networkx as nx
from networkx.classes.reportviews import OutEdgeView, InEdgeView

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