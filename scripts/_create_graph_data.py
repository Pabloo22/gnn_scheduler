from gnn_scheduler.data import JobShopDataset


if __name__ == "__main__":
    dataset = JobShopDataset()
    dataset.process()
    print("Done")
