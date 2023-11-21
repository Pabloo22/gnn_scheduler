from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import torch
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.utils as pyg_utils
import networkx as nx

from gnn_scheduler.jssp import (load_metadata, 
                                load_from_benchmark, 
                                JobShopInstance,
                                )



class OptimumPreduction(InMemoryDataset):
    """Dataset with job-shop modelled as a disjunctive graph and their
       optimum value associated as the target.

    Args:
        InMemoryDataset (_type_): _description_
    """
    
    def __init__(self,
                 root: os.PathLike | str | bytes,
                 transform: Optional[callable] = None,
                 pre_transform: Optional[callable] = None,
                 max_jobs: int = 20,
                 max_machines: int = 10,
    ) -> None:
        super().__init__(root, transform, pre_transform)
        data_file_path = self.processed_paths[0]
        self.data, self.slices = torch.load(data_file_path)
        self.max_jobs = max_jobs
        self.max_machines = max_machines
    
    @property
    def processed_file_names(self) -> list[str]:
        return [f"one_shot_jssp_{self.max_jobs}_{self.max_machines}.pt"]
    
    def _process_instance(self, instance: dict, metadata: list[dict]) -> Data:
        instance_name = instance["name"]
        instance = load_from_benchmark(instance_name, 
                                       self.root,
                                       metatdata=metadata)
        return self._process_instance_helper(instance)
    
    def _process_instance_helper(self, instance: JobShopInstance) -> Data:
        
        digraph = instance.disjunctive_graph
        n_nodes = digraph.number_of_nodes()
        n_edges = digraph.number_of_edges()
        edge_index = torch.zeros((2, n_edges), dtype=torch.long)
        edge_attr = torch.zeros((n_edges, 1), dtype=torch.float)
        node_attr = torch.zeros((n_nodes, 1), dtype=torch.float)
        target = torch.tensor([instance.optimum], dtype=torch.float)
        
        for i, (u, v) in enumerate(digraph.edges):
            edge_index[0, i] = u
            edge_index[1, i] = v
            edge_attr[i] = digraph.edges[u, v]["duration"]
            
        
    
    def process(self) -> None:
        metadata = load_metadata(self.root)
        instances = []
        for instance in metadata:
            if instance["jobs"] > self.max_jobs:
                continue
            if instance["machines"] > self.max_machines:
                continue
            instances.append(instance)
        data_list = [self._process_instance(instance, metadata) 
                     for instance in instances]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    