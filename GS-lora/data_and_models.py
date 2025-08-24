# data_and_models.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# 💡 FIX: Import the missing function from lora_utils.py
from lora_utils import setup_lora_training

class SplitCIFAR100(Dataset):
    def __init__(self, cifar100_dataset, class_indices, task_id):
        self.cifar100_dataset = cifar100_dataset
        self.class_map = {old_idx: new_idx for new_idx, old_idx in enumerate(class_indices)}
        self.task_id = task_id
        self.indices = []
        for i, (_, label) in enumerate(self.cifar100_dataset):
            if label in self.class_map:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, label = self.cifar100_dataset[original_idx]
        new_label = self.class_map[label]
        return image, new_label, self.task_id

def get_cifar100_dataloaders(num_tasks=10, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    classes_per_task = 100 // num_tasks
    all_classes = list(range(100))
    
    task_dataloaders = []
    for task_id in range(num_tasks):
        start_class = task_id * classes_per_task
        end_class = (task_id + 1) * classes_per_task
        class_indices = all_classes[start_class:end_class]
        
        train_task_dataset = SplitCIFAR100(train_dataset, class_indices, task_id)
        test_task_dataset = SplitCIFAR100(test_dataset, class_indices, task_id)
        
        train_loader = DataLoader(train_task_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_task_dataset, batch_size=batch_size, shuffle=False)
        
        task_dataloaders.append((train_loader, test_loader))
    return task_dataloaders

class MultiHeadLoRAModel(nn.Module):
    def __init__(self, num_tasks, num_classes_per_task=10, rank=8, alpha=16):
        super().__init__()
        # Each task gets its own ResNet with its own LoRA adapters
        self.tasks_models = nn.ModuleList()
        for _ in range(num_tasks):
            task_model = models.resnet18(weights='IMAGENET1K_V1')
            task_model.fc = nn.Linear(task_model.fc.in_features, num_classes_per_task)
            # Setup this task's model for LoRA training
            task_model = setup_lora_training(task_model, rank=rank, alpha=alpha)
            self.tasks_models.append(task_model)

    def forward(self, x, task_id):
        # Select the correct model for the given task_id
        return self.tasks_models[task_id](x)