import networkx as nx

import matplotlib.pyplot as pyp


# Printing configuration
from sympy.interactive import printing
printing.init_printing(use_latex=True)

from QGOProblem import QGOProblem, QGOGraph
from QGOOptimizer import QGOOptimizer

from QGOOut import QGOExporter

from pathlib import Path

DATA_IN_DIR = Path(__file__).resolve().parent.parent / "data_in"

# paths to input data in data_in
links = [
    DATA_IN_DIR / "mineral_production_facilities.geojson",
    DATA_IN_DIR / "mineral_sources.geojson",
    DATA_IN_DIR / "possible_solar.geojson",
]

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
ccc_optimize.optimize(problem=ccc_problem, qgo_graph=ccc_graph)

# getting the solution graph
ccc_sol_graph: nx.DiGraph = ccc_optimize.createSolutionGraph(ccc_problem, ccc_graph)
#print(ccc_sol_graph)

# exporting both the problem and solution graphs as GeoJSONs for UI display
exporter = QGOExporter(ccc_problem, ccc_graph, ccc_sol_graph)
