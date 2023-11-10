import __future__

import attrs
import matplotlib.pyplot as plt


@attrs.define
class Operation:
    """A class representing an operation in a job-shop scheduling problem."""
        
    job: int
    machine: int
    duration: float


@attrs.define
class JSSPInstance:
    """A class representing a job-shop scheduling problem instance."""
    
    n_jobs: int
    n_machines: int
    operations: list[Operation] = attrs.Factory(list)

    def plot_disjunctive_graph():
        """Plots the disjunctive graph of the instance.
        
        A disjunctive graph is a directed graph where each node represents an
        operation. There are two types of edges in the graph:
        
        - Conjunctive arcs: an arc from operation A to operation B means that
          operation B cannot start before operation A finishes.
        - Disjunctive arcs: an edge from operation A to operation B means that
            both operations are on the same machine and that they cannot be
            executed in parallel.
        """