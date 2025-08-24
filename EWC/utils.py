import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class SplitCIFAR100(Dataset):
    """
    Custom Dataset class for splitting CIFAR-100 into tasks.
    """
    def __init__(self, cifar100_dataset, class_indices):
        self.cifar100_dataset = cifar100_dataset
        self.class_map = {old_idx: new_idx for new_idx, old_idx in enumerate(class_indices)}
        
        self.indices = []
        for i, (_, label) in enumerate(cifar100_dataset):
            if label in self.class_map:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, label = self.cifar100_dataset[original_idx]
        new_label = self.class_map[label]
        return image, new_label

def get_cifar100_dataloaders(num_tasks=10, batch_size=32):
    """
    Prepares the Split CIFAR-100 dataloaders for all tasks.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Load full datasets
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Split classes into tasks
    classes_per_task = 100 // num_tasks
    all_classes = list(range(100))
    
    task_dataloaders = []
    for task_id in range(num_tasks):
        start_class = task_id * classes_per_task
        end_class = (task_id + 1) * classes_per_task
        class_indices = all_classes[start_class:end_class]
        
        # Create task-specific datasets
        train_task_dataset = SplitCIFAR100(train_dataset, class_indices)
        test_task_dataset = SplitCIFAR100(test_dataset, class_indices)
        
        # Create dataloaders
        train_loader = DataLoader(train_task_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_task_dataset, batch_size=batch_size, shuffle=False)
        
        task_dataloaders.append((train_loader, test_loader))
        
    return task_dataloaders