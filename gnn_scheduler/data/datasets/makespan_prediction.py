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


class MakespanPrediction(InMemoryDataset):
    """Dataset with job-shop modelled as a disjunctive graph and their
       optimum value associated as the target.

    Args:
        InMemoryDataset (_type_): _description_
    """
    
    