from __future__ import annotations

import collections

import networkx as nx


def get_n_machines(graph: nx.DiGraph) -> int:
    """Returns the number of machines from the graph."""
    machines = set()
    for _, node_data in graph.nodes(data=True):
        machines.add(node_data["machine_id"])
    return len(machines)


def get_n_jobs(graph: nx.DiGraph) -> int:
    """Returns the number of jobs from the graph."""
    jobs = set()
    for _, node_data in graph.nodes(data=True):
        jobs.add(node_data["job_id"])
    return len(jobs)


def get_total_processing_time(graph: nx.DiGraph) -> float:
    """Returns the total processing time of the graph."""
    total_processing_time = 0
    for _, node_data in graph.nodes(data=True):
        total_processing_time += node_data["duration"]
    return total_processing_time


def get_job_loads(graph: nx.DiGraph) -> dict[int, float]:
    """Returns the load of each job."""
    job_loads = collections.defaultdict(float)
    for _, node_data in graph.nodes(data=True):
        job_loads[node_data["job_id"]] += node_data["duration"]
    return job_loads


def get_machine_loads(graph: nx.DiGraph) -> dict[int, float]:
    """Returns the load of each machine."""
    machine_loads = collections.defaultdict(float)
    for _, node_data in graph.nodes(data=True):
        machine_loads[node_data["machine_id"]] += node_data["duration"]
    return machine_loads


def get_min_makespan(graph: nx.DiGraph) -> float:
    """Cumputes the cumulative processing time of each job and return the 
    maximum.
    """
    job_loads = get_job_loads(graph)
    return max(job_loads.values())


def normalize_optimum(optimum: float, graph: nx.DiGraph) -> float:
    """Normalizes the optimum by the minimum processing time for solving the
    problem.
    """
    return optimum / get_min_makespan(graph) - 1


def max_graph_duration(graph: nx.DiGraph) -> float:
    """Calculates the maximum operation time across the graph."""
    max_duration = 0
    for _, node_data in graph.nodes(data=True):
        max_duration = max(max_duration, node_data["duration"])
    return max_duration


def max_job_durations(graph: nx.DiGraph) -> float:
    """Calculates the maximum operation time across each job."""
    max_durations = collections.defaultdict(float)
    for _, node_data in graph.nodes(data=True):
        duration = node_data["duration"]
        job_id = node_data["job_id"]
        max_durations[node_data["job_id"]] = max(max_durations[job_id],
                                                 duration)
    return max_durations


def max_machine_durations(graph: nx.DiGraph) -> float:
    """Calculates the maximum operation time across each machine."""
    max_durations = collections.defaultdict(float)
    for _, node_data in graph.nodes(data=True):
        duration = node_data["duration"]
        machine_id = node_data["machine_id"]
        max_durations[machine_id] = max(max_durations[machine_id], duration)
    return max_durations
