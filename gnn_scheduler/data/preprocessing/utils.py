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


def normalize_optimum(optimum: float, graph: nx.DiGraph) -> float:
    """Normalizes the optimum by the total processing time of the graph."""
    return optimum / get_total_processing_time(graph) - 1
