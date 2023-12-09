from __future__ import annotations

import collections
import itertools
import functools
import enum
from typing import Optional, Callable
import warnings

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from gnn_scheduler.jssp.job_shop_instance import JobShopInstance


class EdgeType(enum.Enum):
    CONJUNCTIVE = 0
    DISJUNCTIVE = 1


Layout = Callable[[nx.Graph], dict[str, tuple[float, float]]]


class DisjunctiveGraph(nx.DiGraph):
    """Stores the disjunctive graph of a job-shop instance."""

    def __init__(self, incoming_graph_data=None, name: str = "", **attr):
        super().__init__(incoming_graph_data, **attr)
        self.name = name

    @functools.cached_property
    def n_machines(self) -> int:
        """Returns the number of machines from the graph."""
        machines = set()
        for _, node_data in self.nodes(data=True):
            machines.add(node_data["machine_id"])
        return len(machines)

    @functools.cached_property
    def n_jobs(self) -> int:
        """Returns the number of jobs from the graph."""
        jobs = set()
        for _, node_data in self.nodes(data=True):
            jobs.add(node_data["job_id"])
        return len(jobs)

    @functools.cached_property
    def total_processing_time(self) -> float:
        """Returns the total processing time of the graph."""
        total_processing_time = 0
        for _, node_data in self.nodes(data=True):
            total_processing_time += node_data["duration"]
        return total_processing_time

    @functools.cached_property
    def job_loads(self) -> dict[int, float]:
        """Returns the load of each job."""
        job_loads = collections.defaultdict(float)
        for _, node_data in self.nodes(data=True):
            job_loads[node_data["job_id"]] += node_data["duration"]
        return job_loads

    @functools.cached_property
    def max_job_load(self) -> float:
        """Returns the maximum load of a job."""
        return max(self.job_loads.values())

    @functools.cached_property
    def machine_loads(self) -> dict[int, float]:
        """Returns the load of each machine."""
        machine_loads = collections.defaultdict(float)
        for _, node_data in self.nodes(data=True):
            machine_loads[node_data["machine_id"]] += node_data["duration"]
        return machine_loads

    @functools.cached_property
    def max_machine_load(self) -> float:
        """Returns the maximum load of a machine."""
        return max(self.machine_loads.values())

    @functools.cached_property
    def min_makespan(self) -> float:
        """Cumputes the cumulative processing time of each job and return the
        maximum.
        """
        job_loads = self.job_loads
        return max(job_loads.values())

    @functools.cached_property
    def max_graph_duration(self) -> float:
        """Calculates the maximum operation time across the graph."""
        max_duration = 0
        for _, node_data in self.nodes(data=True):
            max_duration = max(max_duration, node_data["duration"])
        return max_duration

    @functools.cached_property
    def max_job_durations(self) -> float:
        """Calculates the maximum operation time across each job."""
        max_durations = collections.defaultdict(float)
        for _, node_data in self.nodes(data=True):
            duration = node_data["duration"]
            job_id = node_data["job_id"]
            max_durations[node_data["job_id"]] = max(
                max_durations[job_id], duration
            )
        return max_durations

    @functools.cached_property
    def max_machine_durations(self) -> float:
        """Calculates the maximum operation time across each machine."""
        max_durations = collections.defaultdict(float)
        for _, node_data in self.nodes(data=True):
            duration = node_data["duration"]
            machine_id = node_data["machine_id"]
            max_durations[machine_id] = max(
                max_durations[machine_id], duration
            )
        return max_durations

    @functools.cached_property
    def n_operations_per_job(self) -> list[int]:
        """Returns the number of operations per job."""
        n_operations_per_job = [0] * self.n_jobs
        for _, node_data in self.nodes(data=True):
            n_operations_per_job[node_data["job_id"]] += 1
        return n_operations_per_job

    @staticmethod
    def add_conjuctive_edges(
        instance: JobShopInstance, disjunctive_graph: nx.DiGraph
    ) -> nx.DiGraph:
        counter = 0
        # Adding operations as nodes and conjunctive arcs as edges
        for job_id, job in enumerate(instance.jobs):
            prev_op = "S"  # start from source

            for position, operation in enumerate(job):
                op_id = operation.get_id(job_id, position)
                disjunctive_graph.add_node(
                    op_id,
                    duration=operation.duration,
                    machine_id=operation.machine_id,
                    job_id=job_id,
                    position=position,
                    node_index=counter,
                )
                disjunctive_graph.add_edge(
                    prev_op, op_id, type=EdgeType.CONJUNCTIVE
                )
                prev_op = op_id
                counter += 1

            # from last operation to sink
            disjunctive_graph.add_edge(
                prev_op, "T", type=EdgeType.CONJUNCTIVE
            )

        return disjunctive_graph

    @staticmethod
    def add_disjunctive_edges(
        instance: JobShopInstance, disjunctive_graph: nx.DiGraph
    ) -> nx.DiGraph:
        # Adding disjunctive arcs (edges) between operations on the same machine
        machine_operations = {i: [] for i in range(instance.n_machines)}
        for job_id, job in enumerate(instance.jobs):
            for position, operation in enumerate(job):
                op_id = operation.get_id(job_id, position)
                machine_operations[operation.machine_id].append(op_id)

        # Adding disjunctive arcs
        for operations in machine_operations.values():
            for op1, op2 in itertools.combinations(operations, 2):
                disjunctive_graph.add_edge(
                    op1, op2, type=EdgeType.DISJUNCTIVE
                )
                disjunctive_graph.add_edge(
                    op2, op1, type=EdgeType.DISJUNCTIVE
                )

        return disjunctive_graph

    @functools.cached_property
    def disjuncive_edges(self) -> list[tuple[str, str]]:
        """Returns the disjunctive edges of the graph."""
        return [
            (u, v)
            for u, v, d in self.edges(data=True)
            if d["type"] == EdgeType.DISJUNCTIVE
        ]

    @functools.cached_property
    def conjunctive_edges(self) -> list[tuple[str, str]]:
        """Returns the conjunctive edges of the graph."""
        return [
            (u, v)
            for u, v, d in self.edges(data=True)
            if d["type"] == EdgeType.CONJUNCTIVE
        ]

    @classmethod
    def from_job_shop_instance(cls, instance: JobShopInstance) -> nx.DiGraph:
        """Creates the disjunctive graph of the instance."""
        disjunctive_graph = cls(name=instance.name)

        # Adding source and sink nodes
        disjunctive_graph.add_node(
            "S", duration=0, machine_id=-1, job_id=-1, position=-1
        )
        disjunctive_graph.add_node(
            "T", duration=0, machine_id=-1, job_id=-1, position=-1
        )

        disjunctive_graph = cls.add_conjuctive_edges(
            instance, disjunctive_graph
        )
        disjunctive_graph = cls.add_disjunctive_edges(
            instance, disjunctive_graph
        )

        return disjunctive_graph

    def plot(
        self,
        figsize: tuple[float, float] = (12, 8),
        node_size: int = 1600,
        title: Optional[str] = None,
        layout: Optional[Layout] = None,
        edge_width: int = 2,
        font_size: int = 10,
        arrow_size: int = 35,
        alpha=0.95,
        node_font_color: str = "white",
        color_map: str = "Dark2_r",
        draw_disjunctive_edges: bool = True,
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
                from networkx.drawing.nx_agraph import (
                    graphviz_layout,
                )

                layout = functools.partial(
                    graphviz_layout, prog="dot", args="-Grankdir=LR"
                )
            except ImportError:
                warnings.warn(
                    "Could not import graphviz_layout. "
                    + "Using spring_layout instead."
                )
                layout = nx.spring_layout

        temp_graph = self.copy()
        # Remove disjunctive edges to get a better layout
        temp_graph.remove_edges_from(
            [
                (u, v)
                for u, v, d in self.edges(data=True)
                if d["type"] == EdgeType.DISJUNCTIVE
            ]
        )
        pos = layout(temp_graph)

        # Draw nodes
        # ----------
        node_colors = [
            node.get("machine_id", -1) for node in temp_graph.nodes.values()
        ]

        nx.draw_networkx_nodes(
            self,
            pos,
            node_size=node_size,
            node_color=node_colors,
            alpha=alpha,
            cmap=matplotlib.colormaps.get_cmap(color_map),
        )

        # Draw edges
        # ----------
        conjunctive_edges = [
            (u, v)
            for u, v, d in self.edges(data=True)
            if d["type"] == EdgeType.CONJUNCTIVE
        ]
        disjunctive_edges = [
            (u, v)
            for u, v, d in self.edges(data=True)
            if d["type"] == EdgeType.DISJUNCTIVE
        ]

        nx.draw_networkx_edges(
            self,
            pos,
            edgelist=conjunctive_edges,
            width=edge_width,
            edge_color="black",
            arrowsize=arrow_size,
        )

        if draw_disjunctive_edges:
            nx.draw_networkx_edges(
                self,
                pos,
                edgelist=disjunctive_edges,
                width=edge_width,
                edge_color="red",
                arrowsize=arrow_size,
            )

        # Draw node labels
        # ----------------
        durations = list(nx.get_node_attributes(self, "duration").values())[2:]
        nodes = list(self.nodes.keys())[2:]

        labels = {}
        labels["S"] = "S"
        labels["T"] = "T"
        for node, machine, duration in zip(nodes, node_colors[2:], durations):
            labels[node] = f"m={machine}\nd={duration}"

        nx.draw_networkx_labels(
            self,
            pos,
            labels=labels,
            font_color=node_font_color,
            font_size=font_size,
            font_family="sans-serif",
        )

        # Final touches
        # -------------
        plt.axis("off")
        plt.tight_layout()
        # Create a legend to indicate the meaning of the edge colors
        conjunctive_patch = matplotlib.patches.Patch(
            color="black", label="conjunctive edges"
        )
        disjunctive_patch = matplotlib.patches.Patch(
            color="red", label="disjunctive edges"
        )

        # Add to the legend the meaning of m and d
        text = "m = machine_id\nd = duration"
        extra = matplotlib.patches.Rectangle(
            (0, 0),
            1,
            1,
            fc="w",
            fill=False,
            edgecolor="none",
            linewidth=0,
            label=text,
        )
        plt.legend(
            handles=[conjunctive_patch, disjunctive_patch, extra],
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0.0,
        )
        return plt.gcf()
