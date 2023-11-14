from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import matplotlib.pyplot as plt


@dataclass(slots=True)
class JobShopInstance:
    """A class representing a job-shop scheduling problem instance."""

    n_machines: int = None
    jobs: list[list[int]] = None
    times: list[list[float]] = None
    name: str = ""

    def __post_init__(self):
        self.jobs = [] if self.jobs is None else self.jobs
        self.times = [] if self.times is None else self.times
        self.n_machines = 0 if self.n_machines is None else self.n_machines
    
    @property
    def n_jobs(self) -> int:
        """Returns the number of jobs in the instance."""
        return len(self.jobs)

    def get_n_machines(self) -> int:
        """Returns the number of machines in the instance."""
        mx = 0
        for job in self.jobs:
            mx = max(mx, max(job))
        return mx + 1

    @staticmethod
    def disjuntive_graph(job_shop_instance: JobShopInstance) -> nx.DiGraph:
        """Creates the disjunctive graph of a job-shop instance.
        
        A disjunctive graph is a directed graph where each node represents an
        operation. There are two types of edges in the graph:
        
        - Conjunctive arcs: an arc from operation A to operation B means that
            operation B cannot start before operation A finishes.
        - Disjunctive arcs: an edge from operation A to operation B means that
            both operations are on the same machine and that they cannot be
            executed in parallel.
        
        Args:
            job_shop_instance (JobShopInstance): The job-shop scheduling problem instance.
        
        Returns:
            nx.DiGraph: The disjunctive graph.
        """
        graph = nx.DiGraph()

        # Adding nodes (operations) to the graph
        for job_id, job in enumerate(job_shop_instance.jobs):
            for operation_id, machine in enumerate(job):
                # Each node is represented as a tuple (job_id, operation_id)
                graph.add_node((job_id, operation_id))

        # Adding conjunctive arcs (sequential operations within the same job)
        for job_id, job in enumerate(job_shop_instance.jobs):
            for operation_id, _ in enumerate(job[:-1]):
                # Adding an edge from the current operation to the next
                graph.add_edge((job_id, operation_id), (job_id, operation_id + 1))

        # Adding disjunctive arcs (operations on the same machine)
        # This requires identifying operations that use the same machine
        for machine_id in range(job_shop_instance.get_n_machines()):
            operations_on_machine = [(job_id, operation_id) 
                                    for job_id, job in enumerate(job_shop_instance.jobs)
                                    for operation_id, machine in enumerate(job) if machine == machine_id]

            # Add edges between all pairs of operations on the same machine
            # Note: These are undirected edges as the order is not yet determined
            for i, op_i in enumerate(operations_on_machine):
                for op_j in operations_on_machine[i + 1:]:
                    graph.add_edge(op_i, op_j, disjunctive=True)
                    graph.add_edge(op_j, op_i, disjunctive=True)

        return graph

    @staticmethod
    def plot_disjuntive_graph(job_shop_instance: JobShopInstance, 
                              graph: nx.DiGraph,
                              node_size: int = 1500,
                              figsize: tuple = (12, 8),
                              arrow_size: int = 20,
                              edge_width: int = 2,
                              font_size: int = 12):
        """Plots the disjunctive graph of a job-shop instance.

        Args:
            job_shop_instance (JobShopInstance): The job-shop scheduling problem instance.
            graph (nx.DiGraph): The disjunctive graph.
        """
        plt.figure(figsize=figsize)
        # Position nodes using a suitable layout
        pos = nx.spring_layout(graph)

        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=node_size)

        # Draw conjunctive edges (directed)
        conjunctive_edges = [(u, v) for u, v, d in graph.edges(data=True) if not d.get("disjunctive", False)]
        nx.draw_networkx_edges(graph,
                               pos,
                               arrowsize=arrow_size,
                               edgelist=conjunctive_edges,
                               edge_color="black",
                               arrows=True)

        # Draw disjunctive edges (undirected)
        disjunctive_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("disjunctive", False)]
        nx.draw_networkx_edges(graph,
                               pos,
                               width=edge_width,
                               arrowsize=arrow_size,
                               edgelist=disjunctive_edges,
                               edge_color="red",
                               arrows=False,
                               style="dashed")

        # Labels for nodes
        labels = {}
        for job_id, job in enumerate(job_shop_instance.jobs):
            for op_id, machine in enumerate(job):
                labels[(job_id, op_id)] = f"J{job_id}M{machine}"
        nx.draw_networkx_labels(graph, pos, labels, font_size=font_size)

        name = "Job-Shop Scheduling" if not job_shop_instance.name else job_shop_instance.name
        plt.title(f"Disjunctive Graph for {name}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    jssp = JobShopInstance(
        n_machines=3,
        jobs=[[0, 1, 2], [1, 0, 2]],
        times=[[1, 2, 3], [2, 1, 3]],
        name="test",
    )

    digraph = JobShopInstance.disjuntive_graph(jssp)
    JobShopInstance.plot_disjuntive_graph(jssp, digraph, arrow_size=40)
