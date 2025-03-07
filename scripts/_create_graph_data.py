from gnn_scheduler.data import JobShopDataset


# Add JobShopData to PyTorch's safe globals to fix the deserialization warning

if __name__ == "__main__":
    dataset = JobShopDataset(num_chunks=20, force_reload=False)
    data = dataset[0]
    print(data)
    print(len(dataset))
    print(dataset[10])
