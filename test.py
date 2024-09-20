import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the BasicBlock for ResNet


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Define the ResNet class


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # First block can have stride
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))  # Initial convolution
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)  # Global average pooling
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Function to create ResNet-20


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

# Function to simulate loading model parameters


def load_model_parameters(model_id):
    # For demonstration, initialize a model with random parameters
    model = resnet20()
    for param in model.parameters():
        param.data = torch.randn_like(param)
    return model.state_dict()


# Number of models to aggregate
num_models = 10

# Collect all state_dicts
state_dicts = []
for i in range(num_models):
    state_dict = load_model_parameters(i)
    state_dicts.append(state_dict)

# Assign weights to each model (example: equal weights)
weights = [1.0 / num_models] * num_models

# Initialize an empty state_dict for the aggregated model
aggregated_state_dict = {}
for key in state_dicts[0].keys():
    # Ensure that the aggregated_state_dict tensors are float32
    aggregated_state_dict[key] = torch.zeros_like(
        state_dicts[0][key], dtype=torch.float32)

# Aggregate parameters
for key in state_dicts[0].keys():
    # Check if the parameter is of floating point type
    if torch.is_floating_point(state_dicts[0][key]):
        # Initialize the aggregated parameter
        aggregated_state_dict[key] = torch.zeros_like(
            state_dicts[0][key], dtype=torch.float32)
        # Sum weighted parameters
        for state_dict, weight in zip(state_dicts, weights):
            param = state_dict[key].to(torch.float32)
            aggregated_state_dict[key] += weight * param
    else:
        # For integer parameters (e.g., running counts), copy from the first model
        aggregated_state_dict[key] = state_dicts[0][key]

# Load aggregated parameters into a new model
aggregated_model = resnet20()
aggregated_model.load_state_dict(aggregated_state_dict)
aggregated_model.to(device)

# Prepare CIFAR-10 test dataset and loader
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.494, 0.485, 0.450), (0.246, 0.242, 0.261)),
])

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=100, shuffle=False, num_workers=2
)

# Evaluate the aggregated model


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100.0 * correct / total
    avg_loss = test_loss / total
    print(f'Aggregated Model Accuracy: {accuracy:.2f}%')
    print(f'Aggregated Model Loss: {avg_loss:.4f}')


evaluate_model(aggregated_model, test_loader)
