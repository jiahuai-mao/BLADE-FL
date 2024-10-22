# # # from queue import Queue
# # # import time
# # # import random
# # # import numpy as np
# # # from torch.utils.data import DataLoader, Subset
# # # import threading
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # import torchvision
# # # import torchvision.transforms as transforms
# # # import copy

# # # # Set device to GPU if available
# # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # # # Define the BasicBlock for ResNet


# # # class BasicBlock(nn.Module):
# # #     expansion = 1

# # #     def __init__(self, in_planes, planes, stride=1):
# # #         super(BasicBlock, self).__init__()
# # #         self.conv1 = nn.Conv2d(
# # #             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
# # #         )
# # #         self.bn1 = nn.BatchNorm2d(planes)
# # #         self.relu = nn.ReLU(inplace=True)
# # #         self.conv2 = nn.Conv2d(
# # #             planes, planes, kernel_size=3, stride=1, padding=1, bias=False
# # #         )
# # #         self.bn2 = nn.BatchNorm2d(planes)

# # #         self.shortcut = nn.Sequential()
# # #         if stride != 1 or in_planes != self.expansion * planes:
# # #             self.shortcut = nn.Sequential(
# # #                 nn.Conv2d(
# # #                     in_planes,
# # #                     self.expansion * planes,
# # #                     kernel_size=1,
# # #                     stride=stride,
# # #                     bias=False,
# # #                 ),
# # #                 nn.BatchNorm2d(self.expansion * planes),
# # #             )

# # #     def forward(self, x):
# # #         out = self.relu(self.bn1(self.conv1(x)))
# # #         out = self.bn2(self.conv2(out))
# # #         out += self.shortcut(x)
# # #         out = self.relu(out)
# # #         return out

# # # # Define the ResNet class


# # # class ResNet(nn.Module):
# # #     def __init__(self, block, num_blocks, num_classes=10):
# # #         super(ResNet, self).__init__()
# # #         self.in_planes = 16

# # #         self.conv1 = nn.Conv2d(
# # #             3, 16, kernel_size=3, stride=1, padding=1, bias=False
# # #         )
# # #         self.bn1 = nn.BatchNorm2d(16)
# # #         self.relu = nn.ReLU(inplace=True)

# # #         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
# # #         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
# # #         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

# # #         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
# # #         self.linear = nn.Linear(64 * block.expansion, num_classes)

# # #     def _make_layer(self, block, planes, num_blocks, stride):
# # #         # First block can have stride
# # #         strides = [stride] + [1] * (num_blocks - 1)
# # #         layers = []
# # #         for stride in strides:
# # #             layers.append(block(self.in_planes, planes, stride))
# # #             self.in_planes = planes * block.expansion
# # #         return nn.Sequential(*layers)

# # #     def forward(self, x):
# # #         out = self.relu(self.bn1(self.conv1(x)))  # Initial convolution
# # #         out = self.layer1(out)
# # #         out = self.layer2(out)
# # #         out = self.layer3(out)
# # #         out = self.avgpool(out)  # Global average pooling
# # #         out = out.view(out.size(0), -1)
# # #         out = self.linear(out)
# # #         return out

# # # # Function to create ResNet-20


# # # def resnet20():
# # #     return ResNet(BasicBlock, [3, 3, 3])

# # # # Function to simulate loading model parameters


# # # def load_model_parameters(model_id):
# # #     # For demonstration, initialize a model with random parameters
# # #     model = resnet20()
# # #     for param in model.parameters():
# # #         param.data = torch.randn_like(param)
# # #     return model.state_dict()


# # # # Number of models to aggregate
# # # num_models = 10

# # # # Collect all state_dicts
# # # state_dicts = []
# # # for i in range(num_models):
# # #     state_dict = load_model_parameters(i)
# # #     state_dicts.append(state_dict)

# # # # Assign weights to each model (example: equal weights)
# # # weights = [1.0 / num_models] * num_models

# # # # Initialize an empty state_dict for the aggregated model
# # # aggregated_state_dict = {}
# # # for key in state_dicts[0].keys():
# # #     # Ensure that the aggregated_state_dict tensors are float32
# # #     aggregated_state_dict[key] = torch.zeros_like(
# # #         state_dicts[0][key], dtype=torch.float32)

# # # # Aggregate parameters
# # # for key in state_dicts[0].keys():
# # #     # Check if the parameter is of floating point type
# # #     if torch.is_floating_point(state_dicts[0][key]):
# # #         # Initialize the aggregated parameter
# # #         aggregated_state_dict[key] = torch.zeros_like(
# # #             state_dicts[0][key], dtype=torch.float32)
# # #         # Sum weighted parameters
# # #         for state_dict, weight in zip(state_dicts, weights):
# # #             param = state_dict[key].to(torch.float32)
# # #             aggregated_state_dict[key] += weight * param
# # #     else:
# # #         # For integer parameters (e.g., running counts), copy from the first model
# # #         aggregated_state_dict[key] = state_dicts[0][key]

# # # # Load aggregated parameters into a new model
# # # aggregated_model = resnet20()
# # # aggregated_model.load_state_dict(aggregated_state_dict)
# # # aggregated_model.to(device)

# # # # Prepare CIFAR-10 test dataset and loader
# # # transform_test = transforms.Compose([
# # #     transforms.ToTensor(),
# # #     transforms.Normalize((0.494, 0.485, 0.450), (0.246, 0.242, 0.261)),
# # # ])

# # # test_dataset = torchvision.datasets.CIFAR10(
# # #     root='./data', train=False, download=True, transform=transform_test
# # # )
# # # test_loader = torch.utils.data.DataLoader(
# # #     test_dataset, batch_size=100, shuffle=False, num_workers=2
# # # )

# # # # Evaluate the aggregated model


# # # def evaluate_model(model, test_loader):
# # #     model.eval()
# # #     correct = 0
# # #     total = 0
# # #     criterion = nn.CrossEntropyLoss()
# # #     test_loss = 0
# # #     with torch.no_grad():
# # #         for inputs, targets in test_loader:
# # #             inputs, targets = inputs.to(device), targets.to(device)
# # #             outputs = model(inputs)
# # #             loss = criterion(outputs, targets)
# # #             test_loss += loss.item() * targets.size(0)
# # #             _, predicted = outputs.max(1)
# # #             total += targets.size(0)
# # #             correct += predicted.eq(targets).sum().item()
# # #     accuracy = 100.0 * correct / total
# # #     avg_loss = test_loss / total
# # #     print(f'Aggregated Model Accuracy: {accuracy:.2f}%')
# # #     print(f'Aggregated Model Loss: {avg_loss:.4f}')


# # # evaluate_model(aggregated_model, test_loader)


# # # seed = 0
# # # random.seed(seed)
# # # np.random.seed(seed)
# # # torch.manual_seed(seed)
# # # torch.cuda.manual_seed(seed)

# # # num_clients = 4
# # # num_nodes_list = [6, 6, 6, 6]

# # # transform = transforms.Compose(
# # #     [
# # #         transforms.Resize((32, 32)),
# # #         transforms.ToTensor(),
# # #         # transforms.Normalize((0.5,), (0.5,))
# # #         transforms.Normalize((0.1307,), (0.3081,))
# # #     ]
# # # )

# # # train_dataset = torchvision.datasets.FashionMNIST(
# # #     root="./data", train=True, download=True, transform=transform
# # # )


# # # # Run the simulation for 10 rounds
# # # num_rounds = 200
# # # total_samples = len(train_dataset)
# # # fraction = 0.3  # Change to 0.3 for 30%
# # # num_samples = int(total_samples * fraction)
# # # total_indices = list(range(total_samples))

# # # client_indices = []
# # # for _ in range(num_clients):
# # #     subset_indices = total_indices[:total_samples]
# # #     client_indices.append(subset_indices)

# # # random.shuffle(total_indices)
# # # test_dataset_full = Subset(
# # #     train_dataset, total_indices[:int(total_samples*0.3)])


# # # def customize_topology():
# # #     topology = list()
# # #     for i in range(num_clients):
# # #         topology.append(
# # #             [(i - 1 + num_clients) % num_clients,
# # #              (i + 1 + num_clients) % num_clients]
# # #         )
# # #     return topology


# # # def create_iid_splits(dataset, indices, num_nodes):
# # #     node_indices = list()
# # #     for _ in range(num_nodes):
# # #         random.shuffle(indices)
# # #         subset_indices = indices[:num_samples]
# # #         node_indices.append(subset_indices)
# # #     return node_indices


# # # def create_noniid_splits(dataset, indices, num_nodes):
# # #     """
# # #     Splits the data among nodes in a non-IID manner.
# # #     Each node gets data from certain classes only.
# # #     """
# # #     labels = np.array(dataset.targets)[indices]
# # #     classes = np.unique(labels)
# # #     np.random.shuffle(classes)
# # #     class_splits = np.array_split(classes, num_nodes)
# # #     node_indices = []
# # #     for class_split in class_splits:
# # #         idx = [i for i, label in zip(indices, labels) if label in class_split]
# # #         node_indices.append(idx)
# # #     return node_indices


# # # class ComplexCNN(nn.Module):
# # #     def __init__(self, input_channel=1, num_classes=10):
# # #         super(ComplexCNN, self).__init__()
# # #         self.features = nn.Sequential(
# # #             nn.Conv2d(input_channel, 32, kernel_size=3, padding=1),
# # #             nn.Tanh(),
# # #             nn.MaxPool2d(kernel_size=2),
# # #             nn.Conv2d(32, 64, kernel_size=3, padding=1),
# # #             nn.Tanh(),
# # #             nn.MaxPool2d(kernel_size=2),
# # #             nn.Conv2d(64, 128, kernel_size=3, padding=1),
# # #             nn.Tanh(),
# # #             # nn.MaxPool2d(kernel_size=2),
# # #         )

# # #         self.classifier = nn.Sequential(
# # #             nn.Linear(128 * 8 * 8, 512),
# # #             nn.Tanh(),
# # #             nn.Linear(512, num_classes),
# # #         )

# # #     def forward(self, x):
# # #         x = self.features(x)
# # #         # print("111 ",x.shape)
# # #         x = torch.flatten(x, 1)
# # #         # print("222 ",x.shape)
# # #         x = self.classifier(x)
# # #         return x


# # # class Node(threading.Thread):
# # #     """
# # #     Represents a node in the network.
# # #     Each node performs local training and computes gradients.
# # #     """

# # #     def __init__(self, node_id, data_indices, dataset, global_model, aggregator_queue):
# # #         threading.Thread.__init__(self)
# # #         self.node_id = node_id
# # #         self.data_loader = DataLoader(
# # #             Subset(dataset, data_indices), batch_size=128, shuffle=True
# # #         )
# # #         self.local_model = copy.deepcopy(global_model).cuda()
# # #         self.criterion = nn.CrossEntropyLoss()
# # #         self.gradients = None  # Placeholder for storing gradients
# # #         # Queue to send gradients to aggregator
# # #         self.aggregator_queue = aggregator_queue

# # #     def run(self):
# # #         print(f"{self.node_id} starting local training.")
# # #         # Perform forward and backward passes to compute gradients
# # #         self.local_model.train()
# # #         for data, target in self.data_loader:
# # #             data = data.cuda()
# # #             target = target.cuda()
# # #             output = self.local_model(data)
# # #             loss = self.criterion(output, target)
# # #             loss.backward()
# # #         # Extract gradients
# # #         self.gradients = [
# # #             param.grad.clone() for param in self.local_model.cpu().parameters()
# # #         ]
# # #         # Send gradients to aggregator
# # #         self.aggregator_queue.put(self.gradients)
# # #         print(f"{self.node_id} sent gradients to aggregator.")


# # # class Client(threading.Thread):
# # #     """
# # #     Represents a client that manages its nodes and communicates with other clients.
# # #     """

# # #     def __init__(
# # #         self,
# # #         client_id,
# # #         node_indices_list,
# # #         dataset,
# # #         num_clients,
# # #         clients_list,
# # #         joint_clients,
# # #         global_model=None,
# # #     ):
# # #         threading.Thread.__init__(self)
# # #         self.client_id = client_id
# # #         self.nodes = []
# # #         self.dataset = dataset
# # #         # Initialize global model if not provided
# # #         self.global_model = global_model
# # #         self.num_clients = num_clients
# # #         self.received_models = Queue()
# # #         self.received_models_q = Queue()
# # #         self.aggregator_queue = Queue()  # Queue to collect gradients from nodes
# # #         self.clients_list = clients_list  # Reference to other clients
# # #         self.joint_clients = joint_clients
# # #         self.optimizer = torch.optim.SGD(
# # #             self.global_model.parameters(), lr=0.003, momentum=0.9, weight_decay=5e-4
# # #         )
# # #         # Initialize nodes
# # #         for i, node_indices in enumerate(node_indices_list):
# # #             node = Node(
# # #                 f"Client{client_id}_Node{i}",
# # #                 node_indices,
# # #                 dataset,
# # #                 self.global_model,
# # #                 self.aggregator_queue,
# # #             )
# # #             self.nodes.append(node)

# # #     def run(self):
# # #         print(f"Client {self.client_id} starting.")
# # #         # Start all nodes under this client
# # #         for node in self.nodes:
# # #             node.start()
# # #         # Collect gradients from nodes
# # #         collected_gradients = []
# # #         for _ in self.nodes:
# # #             gradients = self.aggregator_queue.get()
# # #             collected_gradients.append(gradients)
# # #         # Wait for all nodes to complete
# # #         for node in self.nodes:
# # #             node.join()
# # #         print(f"Client {self.client_id} collected all gradients.")
# # #         # Aggregate gradients and update the global model
# # #         self.aggregate_and_update(collected_gradients)
# # #         print(f"Client {self.client_id} has updated the global model.")
# # #         # Communicate updated model parameters to other clients
# # #         self.communicate()
# # #         print(f"Client {self.client_id} finished communication.")
# # #         # Average the received models to update the local model
# # #         self.average_models()
# # #         print(
# # #             f"Client {self.client_id} updated its model by averaging received models."
# # #         )

# # #     def aggregate_and_update(self, collected_gradients):
# # #         """
# # #         Aggregates gradients from all nodes and updates the global model.
# # #         """
# # #         # Initialize aggregated gradients
# # #         aggregated_gradients = [
# # #             torch.zeros_like(param) for param in self.global_model.parameters()
# # #         ]
# # #         num_nodes = len(collected_gradients)
# # #         # Sum gradients from all nodes
# # #         for gradients in collected_gradients:
# # #             for idx, grad in enumerate(gradients):
# # #                 aggregated_gradients[idx] += grad
# # #         # Average the gradients
# # #         self.aggregated_gradients = [
# # #             grad / num_nodes for grad in aggregated_gradients]

# # #     def communicate(self):
# # #         """
# # #         Sends the updated model parameters to all other clients.
# # #         """
# # #         for client_id in range(self.num_clients):
# # #             if (client_id in self.joint_clients) and (client_id != self.client_id):
# # #                 target_client = self.clients_list[client_id]
# # #                 # Send (copy) the global model parameters to the target client
# # #                 target_client.received_models_q.put(
# # #                     copy.deepcopy(self.global_model))
# # #                 print(
# # #                     f"Client {self.client_id} sent model parameters to Client {client_id}."
# # #                 )

# # #                 # update the parameter if receive others'
# # #                 if self.received_models_q.qsize() != 0:
# # #                     tmp_received_models = []
# # #                     for _ in range(self.received_models_q.qsize()):
# # #                         tmp_model = self.received_models_q.get()
# # #                         tmp_received_models.append(tmp_model)
# # #                         self.received_models.put(tmp_model)
# # #                     tmp_received_models.append(self.global_model)
# # #                     state_dicts = [model.state_dict()
# # #                                    for model in tmp_received_models]
# # #                     # Get keys from the state_dict
# # #                     param_keys = state_dicts[0].keys()
# # #                     # Initialize new state_dict for averaged parameters
# # #                     averaged_state_dict = {}
# # #                     for key in param_keys:
# # #                         # Sum parameters from all models
# # #                         params = [state_dict[key]
# # #                                   for state_dict in state_dicts]
# # #                         # Stack parameters and compute mean
# # #                         stacked_params = torch.stack(params, dim=0)
# # #                         averaged_param = torch.mean(stacked_params, dim=0)
# # #                         averaged_state_dict[key] = averaged_param
# # #                     # Load averaged parameters into the global model
# # #                     self.global_model.load_state_dict(averaged_state_dict)

# # #                 print(
# # #                     "target client ",
# # #                     target_client.client_id,
# # #                     target_client.received_models.qsize(),
# # #                     target_client.received_models_q.qsize(),
# # #                     len(target_client.joint_clients),
# # #                 )

# # #     def average_models(self):
# # #         """
# # #         Averages the received models to update the local global model.
# # #         """
# # #         while True:
# # #             if (self.received_models.qsize() + self.received_models_q.qsize()) == len(
# # #                 self.joint_clients
# # #             ):
# # #                 break
# # #         if self.received_models.qsize() == len(self.joint_clients):
# # #             self.global_model.train()
# # #             for param, grad in zip(
# # #                 self.global_model.parameters(), self.aggregated_gradients
# # #             ):
# # #                 # param -= 0.001 * grad  # Update rule with learning rate 0.01
# # #                 param.grad = grad
# # #             self.optimizer.step()
# # #             print(f"client {self.client_id}, averge model 1")
# # #         else:
# # #             for _ in range(self.received_models_q.qsize()):
# # #                 self.received_models.put(self.received_models_q.get())
# # #             print(
# # #                 f"Client {self.client_id} received {self.received_models.qsize()} model parameters."
# # #             )
# # #             # Include own model in averaging
# # #             self.received_models.put(self.global_model)
# # #             # Average parameters using state_dict
# # #             # Collect state_dicts from all models
# # #             state_dicts = [
# # #                 self.received_models.get().state_dict()
# # #                 for i in range(self.received_models.qsize())
# # #             ]
# # #             # Get keys from the state_dict
# # #             param_keys = state_dicts[0].keys()
# # #             # Initialize new state_dict for averaged parameters
# # #             averaged_state_dict = {}
# # #             for key in param_keys:
# # #                 # Sum parameters from all models
# # #                 params = [state_dict[key] for state_dict in state_dicts]
# # #                 # Stack parameters and compute mean
# # #                 stacked_params = torch.stack(params, dim=0)
# # #                 averaged_param = torch.mean(stacked_params, dim=0)
# # #                 averaged_state_dict[key] = averaged_param
# # #             # Load averaged parameters into the global model
# # #             self.global_model.load_state_dict(averaged_state_dict)
# # #             # Clear received models for the next round
# # #             self.global_model.train()
# # #             for param, grad in zip(
# # #                 self.global_model.parameters(), self.aggregated_gradients
# # #             ):
# # #                 # param -= 0.001 * grad  # Update rule with learning rate 0.01
# # #                 param.grad = grad
# # #             self.optimizer.step()
# # #             print(f"client {self.client_id}, averge model 2")


# # # # Define test function
# # # def test_global_model(global_model, test_loader):
# # #     global_model.eval()
# # #     correct = 0
# # #     total = 0
# # #     total_loss = 0.0
# # #     criterion = nn.CrossEntropyLoss()
# # #     with torch.no_grad():
# # #         for data, target in test_loader:
# # #             output = global_model(data)
# # #             loss = criterion(output, target)
# # #             total_loss += loss.item() * data.size(0)
# # #             _, predicted = torch.max(output.data, 1)
# # #             total += target.size(0)
# # #             correct += (predicted == target).sum().item()
# # #     accuracy = correct / total
# # #     average_loss = total_loss / total
# # #     return accuracy, average_loss


# # # # Initialize clients' global models (None at the start)
# # # clients_global_models = [ComplexCNN() for _ in range(num_clients)]
# # # joint_clients = customize_topology()
# # # # Prepare test loader
# # # test_loader = DataLoader(test_dataset_full, batch_size=32, shuffle=False)


# # # best_acc, best_round = 0.0, 0
# # # accuracy_list = []
# # # loss_list = []

# # # for round_num in range(num_rounds):
# # #     print(f"\n=== Round {round_num + 1} ===")
# # #     # Create a list to hold clients for this round
# # #     clients_list = []
# # #     # Create clients and their nodes for this round
# # #     for i, c_indices in enumerate(client_indices):
# # #         node_indices_list = create_iid_splits(
# # #             train_dataset, c_indices, num_nodes_list[i]
# # #         )
# # #         # Use the previous global_model if exists
# # #         global_model = clients_global_models[i]
# # #         client = Client(
# # #             i,
# # #             node_indices_list,
# # #             train_dataset,
# # #             num_clients,
# # #             clients_list,
# # #             joint_clients[i],
# # #             global_model=global_model,
# # #         )
# # #         clients_list.append(client)
# # #     # Update clients_list in each client
# # #     for client in clients_list:
# # #         client.clients_list = clients_list
# # #     # Start all clients
# # #     for client in clients_list:
# # #         client.start()
# # #     # Wait for all clients to complete
# # #     for client in clients_list:
# # #         client.join()
# # #     # Store the updated global models for the next round
# # #     clients_global_models = [client.global_model for client in clients_list]
# # #     global_model = clients_global_models[0]
# # #     accuracy, avg_loss = test_global_model(global_model, test_loader)
# # #     accuracy_list.append(accuracy)
# # #     loss_list.append(avg_loss)
# # #     if accuracy >= best_acc:
# # #         best_acc = accuracy
# # #         best_round = round_num
# # #     print(
# # #         f"Round {round_num + 1}: Test Accuracy: {accuracy*100:.2f}%, Test Loss: {avg_loss:.4f}"
# # #     )

# # # with open("./results/res_{}.txt", "wt") as f:
# # #     for i in range(len(accuracy_list)):
# # #         acc = accuracy_list[i]
# # #         loss = loss_list[i]
# # #         f.write("round {}, acc {}, loss {}, best acc {}, best round {}".format(
# # #             i, acc, loss, best_acc, best_round))

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import matplotlib.pyplot as plt

# # # Define the model


# # class DeepModel(nn.Module):
# #     def __init__(self, input_size, hidden_sizes, output_size):
# #         super(DeepModel, self).__init__()
# #         layers = []
# #         in_size = input_size
# #         for h_size in hidden_sizes:
# #             layers.append(nn.Linear(in_size, h_size))
# #             layers.append(nn.ReLU())
# #             in_size = h_size
# #         layers.append(nn.Linear(in_size, output_size))
# #         self.network = nn.Sequential(*layers)

# #     def forward(self, x):
# #         return self.network(x)


# # # Sample data
# # torch.manual_seed(0)  # For reproducibility
# # input_size = 1
# # output_size = 1
# # num_samples = 100

# # x = torch.linspace(-5, 5, num_samples).unsqueeze(1)  # Shape: (num_samples, 1)
# # y = 2 * x + 1 + torch.randn_like(x) * 0.5  # Shape: (num_samples, 1)

# # # Instantiate the model and loss function
# # hidden_sizes = [64, 64]
# # model = DeepModel(input_size, hidden_sizes, output_size)
# # criterion = nn.MSELoss()

# # # Hyperparameters for Adam optimizer
# # lr = 0.01       # Learning rate
# # beta1 = 0.9     # Exponential decay rate for the first moment estimates
# # beta2 = 0.999   # Exponential decay rate for the second moment estimates
# # epsilon = 1e-8  # Small value to prevent division by zero

# # # Initialize moment vectors and time step
# # m = {}
# # v = {}
# # t = 0  # Time step

# # for name, param in model.named_parameters():
# #     m[name] = torch.zeros_like(param)
# #     v[name] = torch.zeros_like(param)

# # # Training loop
# # num_epochs = 1000
# # for epoch in range(num_epochs):
# #     # Forward pass
# #     outputs = model(x)
# #     loss = criterion(outputs, y)

# #     # Zero the gradients
# #     model.zero_grad()

# #     # Backward pass
# #     loss.backward()

# #     # Increment time step
# #     t += 1

# #     # Update parameters manually using Adam optimizer
# #     for name, param in model.named_parameters():
# #         if param.grad is not None:
# #             grad = param.grad

# #             # Update first moment estimate
# #             m[name] = beta1 * m[name] + (1 - beta1) * grad

# #             # Update second moment estimate
# #             v[name] = beta2 * v[name] + (1 - beta2) * grad * grad

# #             # Compute bias-corrected first moment estimate
# #             m_hat = m[name] / (1 - beta1 ** t)

# #             # Compute bias-corrected second moment estimate
# #             v_hat = v[name] / (1 - beta2 ** t)

# #             # Update parameters
# #             param.data = param.data - lr * m_hat / \
# #                 (torch.sqrt(v_hat) + epsilon)

# #     # Print loss every 100 epochs
# #     if (epoch + 1) % 100 == 0:
# #         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# # # Visualize the results
# # model.eval()
# # with torch.no_grad():
# #     predicted = model(x).detach().numpy()

# # plt.figure(figsize=(8, 6))
# # plt.plot(x.numpy(), y.numpy(), 'ro', label='Original data')
# # plt.plot(x.numpy(), predicted, 'b-', label='Fitted line')
# # plt.legend()
# # plt.show()


# import threading
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Subset
# import numpy as np
# import random
# import copy
# import time
# from queue import Queue
# import os
# from tqdm import tqdm
# from demo.demoloader.vgg_bn import vgg19_bn, vgg11_bn

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)


# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.Resize((32, 32)),
#     transforms.Grayscale(num_output_channels=1),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,)),
# ])

# train_dataset = torchvision.datasets.FashionMNIST(
#     root="./data", train=True, download=True, transform=transform
# )
# test_dataset = torchvision.datasets.FashionMNIST(
#     root="./data", train=False, download=True, transform=transform
# )


# class VGG16(nn.Module):
#     def __init__(self, input_channel=1, output_size=10):
#         super(VGG16, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(input_channel, 4, kernel_size=3, padding=1),
#             nn.Tanh(),
#             nn.Conv2d(4, 4, kernel_size=3, padding=1),
#             nn.Tanh(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(4, 8, kernel_size=3, padding=1),
#             nn.Tanh(),
#             nn.Conv2d(8, 8, kernel_size=3, padding=1),
#             nn.Tanh(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(8, 16, kernel_size=3, padding=1),
#             nn.Tanh(),
#             nn.Conv2d(16, 16, kernel_size=3, padding=1),
#             nn.Tanh(),
#             nn.Conv2d(16, 16, kernel_size=3, padding=1),
#             nn.Tanh(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.Tanh(),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.Tanh(),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.Tanh(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.Tanh(),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.Tanh(),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.Tanh()
#             # nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(32 * 32 * 32, 1024),
#             nn.Tanh(),
#             nn.Dropout(),
#             nn.Linear(1024, 1024),
#             nn.Tanh(),
#             nn.Dropout(),
#             nn.Linear(1024, output_size)
#             # nn.Softmax(dim=1)
#         )

#     def forward(self, x):
#         # x = x.reshape([-1, 1, 7, 8])
#         # print("x = ", x.shape)
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         # print("x.view = ", x.shape, "linear = ", 32*32*32)
#         x = self.classifier(x)
#         return x


# def test_global_model(global_model, test_loader):
#     global_model.eval()
#     correct = 0
#     total = 0
#     total_loss = 0.0
#     criterion = nn.CrossEntropyLoss()
#     with torch.no_grad():
#         for data, target in tqdm(test_loader):
#             data, target = data.cuda(), target.cuda()
#             output = global_model(data)
#             loss = criterion(output, target)
#             total_loss += loss.item() * data.size(0)
#             _, predicted = torch.max(output.data, 1)
#             # print(predicted)
#             total += target.size(0)
#             correct += (predicted == target).sum().item()
#     accuracy = correct / total
#     average_loss = total_loss / total

#     return accuracy, average_loss


# # Run the simulation for 10 rounds
# num_rounds = 1000
# batch_size = 1024
# accumulation_steps = 1

# if __name__ == "__main__":
#     model = VGG16()
#     # model = vgg19_bn(input_channel=1, num_classes=10)
#     print(model)
#     data_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False)
#     # optimizer = torch.optim.SGD(
#     #     model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
#     # )
#     # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#     criterion = nn.CrossEntropyLoss()
#     model = model.cuda()

#     for epoch in range(num_rounds):
#         correct = 0
#         total = 0
#         total_loss = 0.0
#         model.train()
#         optimizer.zero_grad()
#         for i, (data, target) in enumerate(tqdm(data_loader)):
#             # print("data shape = ", data.shape)
#             data = data.cuda()
#             target = target.cuda()

#             output = model(data)
#             loss = criterion(output, target)

#             # total_loss += loss.item()
#             # _, predicted = output.max(1)
#             # total += target.size(0)
#             # correct += predicted.eq(target).sum().item()

#             loss = loss/accumulation_steps
#             loss.backward()
#             # if (epoch+1) % 10 == 0 or epoch == 0:
#             #     for param in model.parameters():
#             #         print(param[0], "\n", param.grad[0])
#             #         break
#             if ((i+1) % accumulation_steps) or (i+1 == data_loader.__len__) == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()

#         # accuracy, average_loss = correct / total, total_loss / total
#         accuracy, average_loss = 0.0, 0.0

#         test_accuracy, test_avg_loss = test_global_model(
#             model, test_loader)
#         # test_accuracy, test_avg_loss = 0.0, 0.0

#         print(
#             f"epoch {epoch}, ///train acc: {accuracy}, train loss: {average_loss}, //// test acc = {test_accuracy}, test loss = {test_avg_loss}")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict


def split_data_non_iid(dataset, N, alpha=0.5):
    """
    Splits the dataset into N nodes with non-iid distribution using a Dirichlet distribution.

    Args:
        dataset (pd.DataFrame): The dataset to be split.
        N (int): The number of nodes to split the data into.
        alpha (float): The Dirichlet distribution parameter controlling non-iid-ness.

    Returns:
        dict: A dictionary where keys are node indices and values are datasets for each node.
    """
    # Extract features and labels
    X = dataset.drop(columns=['label'])
    y = dataset['label']

    # Get unique classes in the dataset
    classes = y.unique()
    class_indices = {c: np.where(y == c)[0] for c in classes}

    # Split indices using Dirichlet distribution
    node_data_indices = defaultdict(list)
    for c in classes:
        # Use Dirichlet distribution to generate proportions for each node
        proportions = np.random.dirichlet([alpha] * N)
        class_size = len(class_indices[c])
        # Shuffle indices for each class
        shuffled_indices = np.random.permutation(class_indices[c])
        # Allocate indices to nodes based on generated proportions
        start_idx = 0
        for i in range(N):
            num_samples = int(proportions[i] * class_size)
            node_data_indices[i].extend(
                shuffled_indices[start_idx:start_idx + num_samples])
            start_idx += num_samples

    # Create datasets for each node
    node_datasets = {}
    for i in range(N):
        indices = node_data_indices[i]
        node_datasets[i] = dataset.iloc[indices]

    return node_datasets


# Example usage
if __name__ == "__main__":
    # Create a synthetic dataset for demonstration
    data = {
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'label': np.random.randint(0, 5, 1000)  # Assume 5 classes
    }
    dataset = pd.DataFrame(data)

    # Split the dataset into 4 nodes with non-iid distribution controlled by alpha
    N = 4
    alpha = 0.3
    node_datasets = split_data_non_iid(dataset, N, alpha)

    # Print the number of samples in each node
    for node, data in node_datasets.items():
        print(f"Node {node}: {len(data)} samples")
