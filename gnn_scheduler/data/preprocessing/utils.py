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
