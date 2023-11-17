from __future__ import annotations

import itertools
import functools
from typing import NamedTuple, Optional, Iterable
import os
import warnings
import json

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib


class Operation(NamedTuple):
    """Stores information about an operation in a job-shop scheduling problem."""
    machine_id: int
    duration: float
    
    def get_id(self, job_id: int, position: int) -> str:
        """Returns the id of the operation."""
        return f"J{job_id}M{self.machine_id}P{position}"


class JobShopInstance:
    """Stores a job-shop scheduling problem instance."""

    def __init__(self,
                 jobs: list[list[Operation]],
                 name: str = "JobShopInstance",
                 optimum: Optional[float] = None,
                 upper_bound: Optional[float] = None,
                 lower_bound: Optional[float] = None,
                 ):
        self.jobs = jobs
        self.n_machines = self._get_n_machines()
        self.name = name
        self.time = 0

        # List of lists of job ids. Each list represents a machine:
        self.current_solution = [[] for _ in range(self.n_machines)]

        self.optimum = optimum
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
 
        self._disjunctive_graph = None

    @property
    def n_jobs(self) -> int:
        """Returns the number of jobs in the instance."""
        return len(self.jobs)

    @property
    def disjunctive_graph(self) -> nx.DiGraph:
        """Returns the disjunctive graph of the instance."""
        if self._disjunctive_graph is None:
            self._disjunctive_graph = self._create_disjunctive_graph()
        return self._disjunctive_graph

    @property
    def bounds(self) -> tuple[float, float]:
        """Returns the lower and upper bounds of the instance."""
        return self.lower_bound, self.upper_bound

    def _get_n_machines(self) -> int:
        """Returns the number of machines in the instance."""
        mx = 0
        for job in self.jobs:
            mx_machine = max(operation.machine_id for operation in job)
            mx = max(mx, mx_machine)
        return mx + 1

    def _add_conjuctive_edges(self, 
                              disjunctive_graph: nx.DiGraph) -> nx.DiGraph:
        # Adding operations as nodes and conjunctive arcs as edges
        for job_id, job in enumerate(self.jobs):
            prev_op = "S"  # start from source
            
            for position, operation in enumerate(job):
                op_id = operation.get_id(job_id, position)
                disjunctive_graph.add_node(op_id,
                                           duration=operation.duration,
                                           machine_id=operation.machine_id,
                                           job_id=job_id)
                disjunctive_graph.add_edge(prev_op,
                                        op_id,
                                        type="conjunctive")
                prev_op = op_id
            
            # from last operation to sink
            disjunctive_graph.add_edge(prev_op, "T", type="conjunctive")
        
        return disjunctive_graph

    def _add_disjunctive_edges(self, 
                               disjunctive_graph: nx.DiGraph) -> nx.DiGraph:
        # Adding disjunctive arcs (edges) between operations on the same machine
        machine_operations = {i: [] for i in range(self.n_machines)}
        for job_id, job in enumerate(self.jobs):
            for position, operation in enumerate(job):
                op_id = operation.get_id(job_id, position)
                machine_operations[operation.machine_id].append(op_id)
        
        # Adding disjunctive arcs
        for operations in machine_operations.values():
            for op1, op2 in itertools.combinations(operations, 2):
                disjunctive_graph.add_edge(op1, op2, type="disjunctive")
                disjunctive_graph.add_edge(op2, op1, type="disjunctive")

        return disjunctive_graph

    def _create_disjunctive_graph(self) -> nx.DiGraph:
        """Creates the disjunctive graph of the instance."""
        disjunctive_graph = nx.DiGraph()
        
        # Adding source and sink nodes
        disjunctive_graph.add_node("S")
        disjunctive_graph.add_node("T")

        disjunctive_graph = self._add_conjuctive_edges(disjunctive_graph)
        disjunctive_graph = self._add_disjunctive_edges(disjunctive_graph)

        return disjunctive_graph

    def plot_disjunctive_graph(self,
                               figsize: tuple[float, float] = (12, 8),
                               node_size: int = 1600,
                               title: Optional[str] = None,
                               layout: Optional[callable] = None,
                               edge_width: int = 2,
                               font_size: int = 10,
                               arrow_size: int = 35,
                               alpha=0.95,
                               node_font_color: str = "white",
                               ) -> plt.Figure:
        """Returns a plot of the disjunctive graph of the instance."""
        # Set up the plot
        # ----------------
        plt.figure(figsize=figsize)
        if title is None:
            title = f"Disjunctive Graph Visualization: {self.name}"
        plt.title(title)
        
        # Set up the layout
        # -----------------
        if layout is None:
            try:
                from networkx.drawing.nx_agraph import graphviz_layout
                
                layout = functools.partial(graphviz_layout, prog="dot", args="-Grankdir=LR")
            except ImportError:
                warnings.warn("Could not import graphviz_layout. "
                              "Using spring_layout instead.")
                layout = nx.spring_layout
        
        temp_graph = self.disjunctive_graph.copy()
        # Remove disjunctive edges to get a better layout
        temp_graph.remove_edges_from(
            [(u, v) for u, v, d
                in self.disjunctive_graph.edges(data=True)
                if d["type"] == "disjunctive"]
        )
        pos = layout(temp_graph)
        
        # Draw nodes
        # ----------
        node_colors = [node.get("machine_id", -1)
                       for node in temp_graph.nodes.values()]

        nx.draw_networkx_nodes(self.disjunctive_graph,
                               pos,
                               node_size=node_size,
                               node_color=node_colors,
                               alpha=alpha,
                               cmap=matplotlib.colormaps.get_cmap("Dark2_r"),
                               )

        # Draw edges
        # ----------
        conjunctive_edges = [(u, v) for u, v, d
                             in self.disjunctive_graph.edges(data=True)
                             if d["type"] == "conjunctive"]
        disjunctive_edges = [(u, v) for u, v, d
                             in self.disjunctive_graph.edges(data=True)
                             if d["type"] == "disjunctive"]

        nx.draw_networkx_edges(
            self.disjunctive_graph, pos,
            edgelist=conjunctive_edges,
            width=edge_width,
            edge_color="black",
            arrowsize=arrow_size,
        )

        nx.draw_networkx_edges(
            self.disjunctive_graph, pos,
            edgelist=disjunctive_edges,
            width=edge_width,
            edge_color="red",
            arrowsize=arrow_size,
        )
        
        # Draw node labels
        # ----------------
        durations = nx.get_node_attributes(
            self.disjunctive_graph, "duration"
            ).values()
        nodes = list(self.disjunctive_graph.nodes.keys())[2:]

        labels = {}
        labels["S"] = "S"
        labels["T"] = "T"
        for node, machine, duration in zip(nodes,
                                           node_colors[2:],
                                           durations):
            labels[node] = f"m={machine}\nd={duration}"

        nx.draw_networkx_labels(self.disjunctive_graph,
                                pos,
                                labels=labels,
                                font_color=node_font_color,
                                font_size=font_size,
                                font_family="sans-serif")

        # Final touches
        # -------------
        plt.axis("off")
        plt.tight_layout()
        # Create a legend to indicate the meaning of the edge colors
        conjunctive_patch = matplotlib.patches.Patch(color="black",
                                                     label="conjunctive edges")
        disjunctive_patch = matplotlib.patches.Patch(color="red",
                                                     label="disjunctive edges")
        
        # Add to the legend the meaning of m and d
        text = "m = machine_id\nd = duration"
        extra = matplotlib.patches.Rectangle((0, 0),
                                             1,
                                             1,
                                             fc="w", 
                                             fill=False,
                                             edgecolor="none",
                                             linewidth=0,
                                             label=text)
        plt.legend(handles=[conjunctive_patch, disjunctive_patch, extra],
                   loc="upper left",
                   bbox_to_anchor=(1.05, 1),
                   borderaxespad=0.,
                   )
        return plt.gcf()

    @staticmethod
    def _read_taillard_file(lines: Iterable[str],
                            comment_symbol: str = "#",
                            **kwargs,
                            ) -> JobShopInstance:
        """Returns a job-shop instance from a Taillard file.
        
        Example of a Taillard file:
            #+++++++++++++++++++++++++++++
            # instance la02
            #+++++++++++++++++++++++++++++
            # Lawrence 10x5 instance (Table 3, instance 2); also called (setf2) or (F2)
            10 5
            0 20 3 87 1 31 4 76 2 17
            4 25 2 32 0 24 1 18 3 81
            1 72 2 23 4 28 0 58 3 99
            2 86 1 76 4 97 0 45 3 90
            4 27 0 42 3 48 2 17 1 46
            1 67 0 98 4 48 3 27 2 62
            4 28 1 12 3 19 0 80 2 50
            1 63 0 94 2 98 3 50 4 80
            4 14 0 75 2 50 1 41 3 55
            4 72 2 18 1 37 3 79 0 61
        """
        
        first_non_comment_line_reached = False
        jobs = []
        for line in lines:
            line = line.strip()
            if line.startswith(comment_symbol):
                continue
            if not first_non_comment_line_reached:
                first_non_comment_line_reached = True
                continue
            
            row = list(map(int, line.split()))
            
            pairs = zip(row[::2], row[1::2])
            operations = [Operation(machine_id=machine_id,
                                    duration=duration)
                          for machine_id, duration in pairs]
            jobs.append(operations)
        
        return JobShopInstance(jobs=jobs, **kwargs)

    @staticmethod
    def load_from_file(path: os.PathLike | str | bytes,
                       comment_symbol: str = "#",
                       specification: str = "taillard",
                       encoding: str = "utf-8",
                       **kwargs,
                       ) -> JobShopInstance:
        """Loads a job-shop instance from a file."""
        
        with open(path, "r", encoding=encoding) as f:
            lines = f.readlines()
        
        if specification == "taillard":
            return JobShopInstance._read_taillard_file(lines,
                                                       comment_symbol,
                                                       **kwargs)

        raise NotImplementedError(
            f"Specification '{specification}' is not implemented."
            )
    
    @staticmethod
    def load_from_benchmark(path: os.PathLike | str | bytes,
                            instance_name: str,
                            encoding: str = "utf-8",
                            ) -> JobShopInstance:
        """Loads a job-shop instance from a benchmark file."""
        
        # get metadata from instances.json file
        instances_path = os.path.join(path, "instances.json")
        with open(instances_path, "r", encoding=encoding) as f:
            instances: list[dict] = json.load(f)
        
        optimum = None
        upper_bound = None
        lower_bound = None
        file_path = None
        for instance in instances:
            if instance["name"] != instance_name:
                continue
            optimum = instance["optimum"]
            if optimum is None and instance.get("bounds", None) is not None:
                upper_bound, lower_bound = instance["bounds"]
            file_path = os.path.join(path, instance["path"])
            break
        
        return JobShopInstance.load_from_file(file_path,
                                              name=instance_name,
                                              optimum=optimum,
                                              upper_bound=upper_bound,
                                              lower_bound=lower_bound,
                                              encoding=encoding,
                                              )
    
if __name__ == "__main__":    
    pass
