import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class SplitCIFAR100(Dataset):
    def __init__(self, cifar100_dataset, class_indices):
        self.cifar100_dataset = cifar100_dataset
        self.class_map = {old_idx: new_idx for new_idx, old_idx in enumerate(class_indices)}
        self.indices = []
        for i, (_, label) in enumerate(cifar100_dataset):
            if label in self.class_map:
                self.indices.append(i)
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, label = self.cifar100_dataset[original_idx]
        new_label = self.class_map[label]
        return image, new_label

def get_cifar100_dataloaders(num_tasks=10, batch_size=32):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    classes_per_task = 100 // num_tasks
    task_dataloaders = []
    for task_id in range(num_tasks):
        start_class, end_class = task_id * classes_per_task, (task_id + 1) * classes_per_task
        class_indices = list(range(100))[start_class:end_class]
        train_task_dataset = SplitCIFAR100(train_dataset, class_indices)
        test_task_dataset = SplitCIFAR100(test_dataset, class_indices)
        train_loader = DataLoader(train_task_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_task_dataset, batch_size=batch_size, shuffle=False)
        task_dataloaders.append((train_loader, test_loader))
    return task_dataloaders

class EWC(nn.Module):
    def __init__(self, model, device, lambda_ewc=1000.0):
        super(EWC, self).__init__()
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.fisher_matrices = {}
        self.optimal_params = {}

    def calculate_penalty(self, task_ids_to_penalize):
        """Calculates penalty for a specific list of task IDs."""
        penalty = 0
        for tid in task_ids_to_penalize:
            if tid in self.fisher_matrices:
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        fisher = self.fisher_matrices[tid][n].to(self.device)
                        opt_param = self.optimal_params[tid][n].to(self.device)
                        penalty += (fisher * (p - opt_param) ** 2).sum()
        return self.lambda_ewc * penalty

    def on_task_end(self, task_id, train_loader, criterion):
        print(f"Calculating Fisher Information Matrix for Task {task_id + 1}...")
        self.model.eval()
        self.optimal_params[task_id] = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        # Use a temporary optimizer to avoid interfering with the main one
        temp_optimizer = optim.SGD(self.model.parameters(), lr=0.1)

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            temp_optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().clone() ** 2
        
        num_samples = len(train_loader.dataset)
        for n in fisher:
            fisher[n] /= num_samples
        self.fisher_matrices[task_id] = fisher
        self.model.train()
        print("Fisher Matrix calculation complete.")

# --- Evaluation Function ---
def evaluate(model, test_loaders, device):
    model.eval()
    accuracies = []
    for loader in test_loaders:
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracies.append(100 * correct / total)
    return accuracies
    
# --- 💡 New Unlearning Function ---
def unlearn_task_ewc(model, cl_strategy, task_to_forget, dataloader_forget, device, unlearn_epochs=5, unlearn_lr=1e-5, retain_lambda=5000.0):
    print(f"\n--- Starting Unlearning Process for Task {task_to_forget + 1} ---")
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=unlearn_lr)
    criterion = nn.CrossEntropyLoss()
    
    task_ids_to_retain = [tid for tid in cl_strategy.fisher_matrices.keys() if tid != task_to_forget]
    
    for epoch in range(unlearn_epochs):
        pbar = tqdm(dataloader_forget, desc=f"Unlearning Epoch {epoch+1}/{unlearn_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # 1. Forget Loss: We want to MAXIMIZE this, so we take its negative
            outputs = model(images)
            loss_forget = -criterion(outputs, labels) # Gradient ASCENT
            
            # 2. Retain Loss: The EWC penalty for all other tasks
            loss_retain = cl_strategy.calculate_penalty(task_ids_to_retain)
            
            # Combine the losses
            total_loss = loss_forget + retain_lambda * loss_retain
            
            total_loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=total_loss.item())

def main():
    NUM_TASKS = 5
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Standard Continual Learning Phase
    print("--- Phase 1: Continual Learning ---")
    task_dataloaders = get_cifar100_dataloaders(num_tasks=NUM_TASKS, batch_size=128)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100 // NUM_TASKS)
    model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    cl_strategy = EWC(model, DEVICE, lambda_ewc=5000.0)

    for task_id in range(NUM_TASKS):
        print(f"\n--- Training on Task {task_id + 1}/{NUM_TASKS} ---")
        train_loader, _ = task_dataloaders[task_id]
        for epoch in range(10): # Shorter training for demonstration
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                if task_id > 0:
                    loss += cl_strategy.calculate_penalty(range(task_id))
                loss.backward()
                optimizer.step()
        cl_strategy.on_task_end(task_id, train_loader, criterion)

    # 2. Evaluation Before Unlearning
    print("\n--- Phase 2: Accuracies BEFORE Unlearning ---")
    test_loaders = [loader for _, loader in task_dataloaders]
    before_accuracies = evaluate(model, test_loaders, DEVICE)
    for i, acc in enumerate(before_accuracies):
        print(f"Task {i+1} Accuracy: {acc:.2f}%")
    print(f"Average Accuracy: {np.mean(before_accuracies):.2f}%")

    # 3. Unlearning Phase
    task_to_forget = 1
    forget_train_loader, _ = task_dataloaders[task_to_forget]
    unlearn_task_ewc(model, cl_strategy, task_to_forget, forget_train_loader, DEVICE)

    # 4. Evaluation After Unlearning
    print(f"\n--- Phase 3: Accuracies AFTER Unlearning Task {task_to_forget + 1} ---")
    after_accuracies = evaluate(model, test_loaders, DEVICE)
    for i, acc in enumerate(after_accuracies):
        status = " (FORGOTTEN)" if i == task_to_forget else " (RETAINED)"
        change = after_accuracies[i] - before_accuracies[i]
        print(f"Task {i+1} Accuracy: {acc:.2f}% {status} | Change: {change:+.2f}%")
    
    retained_acc = np.mean([acc for i, acc in enumerate(after_accuracies) if i != task_to_forget])
    print(f"Average Accuracy on Retained Tasks: {retained_acc:.2f}%")

if __name__ == "__main__":
    main()