import multiprocessing as mp
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import copy
import time

# Set random seeds
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

num_clients = 4
num_nodes_list = [2, 2, 2, 2]
batch_size = 16
num_rounds = 5  # Reduced number of rounds for testing

# Transform and Dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

total_samples = len(train_dataset)
fraction = 0.26
num_samples = int(total_samples * fraction)
total_indices = list(range(total_samples))
random.shuffle(total_indices)

# Split indices among clients
client_indices = []
num_samples_per_client = len(total_indices) // num_clients
for i in range(num_clients):
    start_idx = i * num_samples_per_client
    end_idx = (i + 1) * num_samples_per_client
    subset_indices = total_indices[start_idx:end_idx]
    client_indices.append(subset_indices)

test_dataset_full = Subset(
    train_dataset, total_indices[:int(total_samples * fraction)]
)


def customize_topology():
    topology = []
    for i in range(num_clients):
        topology.append(
            [(i - 1 + num_clients) % num_clients, (i + 1) % num_clients]
        )
    return topology


def create_iid_splits(indices, num_nodes):
    random.shuffle(indices)
    node_indices = np.array_split(indices, num_nodes)
    return [list(idx) for idx in node_indices]


class ComplexCNN(nn.Module):
    def __init__(self, input_channel=1, num_classes=10):
        super(ComplexCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.Tanh(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Node(mp.Process):
    def __init__(self, node_id, data_indices, dataset, global_model_state_dict, aggregator_queue):
        super(Node, self).__init__()
        self.node_id = node_id
        self.data_indices = data_indices
        self.dataset = dataset
        self.global_model_state_dict = global_model_state_dict
        self.aggregator_queue = aggregator_queue

    def run(self):
        print(f"{self.node_id} starting local training.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        data_loader = DataLoader(
            Subset(self.dataset, self.data_indices), batch_size=batch_size, shuffle=True, num_workers=0
        )

        # Recreate the model and load state dict
        local_model = ComplexCNN().to(device)
        local_model.load_state_dict(self.global_model_state_dict)
        local_model.train()

        optimizer = torch.optim.SGD(
            local_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = local_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            break  # Process one batch for testing

        # Extract gradients
        gradients = [
            param.grad.clone().cpu() for param in local_model.parameters()
        ]
        self.aggregator_queue.put(gradients)
        print(f"{self.node_id} sent gradients to aggregator.")
        del local_model
        torch.cuda.empty_cache()


class Client:
    def __init__(
        self,
        client_id,
        node_indices_list,
        dataset,
        num_clients,
        clients_list,
        joint_clients,
        global_model=None,
    ):
        self.client_id = client_id
        self.node_indices_list = node_indices_list
        self.dataset = dataset
        self.global_model = global_model
        self.num_clients = num_clients
        self.received_models_q = mp.Queue()
        self.aggregator_queue = mp.Queue()
        self.clients_list = clients_list
        self.joint_clients = joint_clients

    def aggregate_and_update(self, collected_gradients):
        aggregated_gradients = [
            torch.zeros_like(param) for param in self.global_model.parameters()
        ]
        num_nodes = len(collected_gradients)
        for gradients in collected_gradients:
            for idx, grad in enumerate(gradients):
                aggregated_gradients[idx] += grad
        self.aggregated_gradients = [
            grad / num_nodes for grad in aggregated_gradients
        ]
        with torch.no_grad():
            for param, grad in zip(self.global_model.parameters(), self.aggregated_gradients):
                param -= 0.01 * grad  # Learning rate of 0.01

    def communicate(self):
        for client_id in self.joint_clients:
            if client_id != self.client_id:
                target_client = self.clients_list[client_id]
                target_client.received_models_q.put(
                    copy.deepcopy(self.global_model.state_dict())
                )
                print(
                    f"Client {self.client_id} sent model parameters to Client {client_id}."
                )

    def average_models(self):
        num_models_to_receive = len(self.joint_clients) - 1
        received_state_dicts = []
        while len(received_state_dicts) < num_models_to_receive:
            if not self.received_models_q.empty():
                model_state_dict = self.received_models_q.get()
                received_state_dicts.append(model_state_dict)
            else:
                time.sleep(0.1)
        received_state_dicts.append(self.global_model.state_dict())
        param_keys = received_state_dicts[0].keys()
        averaged_state_dict = {}
        for key in param_keys:
            params = [state_dict[key] for state_dict in received_state_dicts]
            stacked_params = torch.stack(params, dim=0)
            averaged_param = torch.mean(stacked_params, dim=0)
            averaged_state_dict[key] = averaged_param
        self.global_model.load_state_dict(averaged_state_dict)


def test_global_model(global_model, test_loader):
    global_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model.to(device)
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = global_model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    average_loss = total_loss / total
    return accuracy, average_loss


if __name__ == "__main__":
    mp.set_start_method('spawn')
    clients_global_models = [ComplexCNN() for _ in range(num_clients)]
    joint_clients = customize_topology()

    # Prepare test loader
    test_loader = DataLoader(
        test_dataset_full, batch_size=batch_size, shuffle=False, num_workers=0
    )

    clients_list = []

    for i in range(num_clients):
        client = Client(
            i,
            node_indices_list=[],
            dataset=train_dataset,
            num_clients=num_clients,
            clients_list=[],
            joint_clients=joint_clients[i],
            global_model=clients_global_models[i],
        )
        clients_list.append(client)

    # Update clients_list in each client
    for client in clients_list:
        client.clients_list = clients_list

    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1} ===")
        processes = []

        # Start node processes for each client
        for client in clients_list:
            c_indices = client_indices[client.client_id]
            node_indices_list = create_iid_splits(
                c_indices, num_nodes_list[client.client_id]
            )
            client.node_indices_list = node_indices_list
            global_model_state_dict = client.global_model.state_dict()
            for idx, node_indices in enumerate(node_indices_list):
                node = Node(
                    node_id=f"Client{client.client_id}_Node{idx}",
                    data_indices=node_indices,
                    dataset=client.dataset,
                    global_model_state_dict=global_model_state_dict,
                    aggregator_queue=client.aggregator_queue
                )
                processes.append(node)

        # Start all node processes
        for process in processes:
            process.start()

        # Wait for all node processes to finish
        for process in processes:
            process.join()

        # Aggregate gradients and update global models
        for client in clients_list:
            collected_gradients = []
            while not client.aggregator_queue.empty():
                gradients = client.aggregator_queue.get()
                collected_gradients.append(gradients)
            client.aggregate_and_update(collected_gradients)
            print(f"Client {client.client_id} has updated the global model.")

        # Communication and model averaging
        for client in clients_list:
            client.communicate()

        for client in clients_list:
            client.average_models()
            print(
                f"Client {client.client_id} updated its model by averaging received models.")

        # Evaluate the global model (using the first client's model)
        global_model = clients_list[0].global_model
        accuracy, avg_loss = test_global_model(global_model, test_loader)
        print(
            f"Round {round_num + 1}: Test Accuracy: {accuracy*100:.2f}%, Test Loss: {avg_loss:.4f}"
        )

        # Clear processes list and free up memory
        del processes[:]
        torch.cuda.empty_cache()
