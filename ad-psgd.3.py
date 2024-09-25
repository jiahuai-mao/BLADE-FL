# adpsgd_distributed.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_data_loader(rank, size, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(
        './data', train=True, download=(rank == 0), transform=transform)
    dist.barrier()  # Ensure dataset is downloaded before proceeding
    subset_size = len(dataset) // size
    indices = list(range(rank * subset_size, (rank + 1) * subset_size))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True)
    return loader


def run_async(rank, size):
    torch.manual_seed(42 + rank)
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 5
    batch_size = 64

    # Get data loader for this process
    train_loader = get_data_loader(rank, size, batch_size)

    # Communication topology: ring topology
    prev_rank = (rank - 1) % size
    next_rank = (rank + 1) % size

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Local training step
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Asynchronous communication step
            with torch.no_grad():
                for param in model.parameters():
                    # Non-blocking send and receive
                    recv_tensor = torch.zeros_like(param.data)
                    req_send = dist.isend(tensor=param.data, dst=next_rank)
                    req_recv = dist.irecv(tensor=recv_tensor, src=prev_rank)
                    # Wait for communication to complete
                    req_send.wait()
                    req_recv.wait()
                    # Update parameters using received data
                    alpha = 0.5  # Mixing parameter
                    param.data = alpha * recv_tensor + (1 - alpha) * param.data

        if rank == 0:
            print(f'Epoch {epoch + 1} completed on rank {rank}')


def main():
    size = 4  # Number of processes/nodes
    processes = []

    mp.set_start_method('spawn')

    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run_async))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
