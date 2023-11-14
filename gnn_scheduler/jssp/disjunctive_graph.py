import networkx as nx
import matplotlib.pyplot as plt

from gnn_scheduler.jssp import JobShopInstance


def create_disjunctive_graph(jssp: JobShopInstance) -> nx.DiGraph:
    """Returns a networkx.DiGraph object representing the disjunctive graph.
    
    A disjunctive graph is a directed graph where each node represents an
    operation. There are two types of edges in the graph:
    
    - Conjunctive arcs: an arc from operation A to operation B means that
        operation B cannot start before operation A finishes.
    - Disjunctive arcs: an edge from operation A to operation B means that
        both operations are on the same machine and that they cannot be
        executed in parallel.
    
    Args:
        jssp (JobShopInstance): The job-shop scheduling problem instance.
    
    Returns:
        nx.DiGraph: The disjunctive graph.
    """

    # Create the graph
    disjunctive_graph = nx.DiGraph()
    

    return disjunctive_graph


def plot_disjunctive_graph(disjunctive_graph: nx.DiGraph,
                           figsize=(12, 8),
                           node_size=1000,
                           layout=nx.spring_layout) -> plt.Figure:
    """
    Returns a matplotlib.Figure object representing the disjunctive graph.
    
    Parameters:
    - disjunctive_graph (nx.DiGraph): The disjunctive graph to be plotted.
    - figsize (tuple): Size of the figure (width, height).
    - node_size (int): Size of the nodes.
    - layout (function): NetworkX layout function for positioning nodes.
    """
    # Set up the plot
    plt.figure(figsize=figsize)
    plt.title('Disjunctive Graph Visualization')

    # Change node_id to node name
    mapping = {node_id: node["name"] for node_id, node in disjunctive_graph.nodes(data=True)}
    disjunctive_graph = nx.relabel_nodes(disjunctive_graph, mapping)

    # Choose a layout for the graph
    pos = layout(disjunctive_graph)

    # Draw nodes
    nx.draw_networkx_nodes(disjunctive_graph,
                           pos,
                           node_size=node_size,
                           node_color="lightblue")

    # Draw edges
    # Conjunctive arcs (solid line)
    conjunctive_edges = [(u, v) for (u, v, d) in disjunctive_graph.edges(data=True)
                         if d["type"] == "conjuctive"]
    nx.draw_networkx_edges(disjunctive_graph,
                           pos,
                           edgelist=conjunctive_edges,
                           width=2,
                           edge_color="black")
    print(conjunctive_edges)
    # Disjunctive arcs (dashed line)
    disjunctive_edges = [(u, v) for (u, v, d) in disjunctive_graph.edges(data=True)
                         if d["type"] == "disjunctive"]

    nx.draw_networkx_edges(disjunctive_graph,
                           pos,
                           edgelist=disjunctive_edges,
                           width=2, style="dashed",
                           edge_color="red")

    # Add labels
    nx.draw_networkx_labels(disjunctive_graph,
                            pos,
                            font_size=12,
                            font_family="sans-serif")

    # Customize the plot
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    from gnn_scheduler.jssp import Operation

    # Example usage (2 jobs and 2 machines)
    jssp_instance_ = JobShopInstance(
        n_jobs=2,
        n_machines=2,
        operations=[
            Operation(job=0, machine=0, duration=3),
            Operation(job=0, machine=1, duration=2),
            Operation(job=1, machine=0, duration=2),
            Operation(job=1, machine=1, duration=3),
        ]
    )

    disjuntive_graph = create_disjunctive_graph(jssp_instance_)
    print(disjuntive_graph.edges(data=True))
    plot_disjunctive_graph(disjuntive_graph)
