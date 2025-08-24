import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import numpy as np

from utils import get_cifar100_dataloaders
from strategies import ExperienceReplay

class MultiHeadResNet(nn.Module):
    def __init__(self, num_tasks, num_classes_per_task):
        super().__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleList([
            nn.Linear(in_features, num_classes_per_task) for _ in range(num_tasks)
        ])
    def forward(self, x, task_id):
        features = self.backbone(x)
        output = self.heads[task_id](features)
        return output

# --- Evaluation Function ---
def evaluate(model, test_loader, device, task_id):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, task_id=task_id)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    NUM_TASKS = 10
    CLASSES_PER_TASK = 10
    BATCH_SIZE = 64
    EPOCHS_PER_TASK = 10
    LR = 0.01

    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    task_dataloaders = get_cifar100_dataloaders(num_tasks=NUM_TASKS, batch_size=BATCH_SIZE)

    model = MultiHeadResNet(num_tasks=NUM_TASKS, num_classes_per_task=CLASSES_PER_TASK).to(DEVICE)
    
    # --- Strategy Setup ---
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    cl_strategy = ExperienceReplay(buffer_size_per_task=50)

    # --- Training Loop ---
    results = []
    for task_id in range(NUM_TASKS):
        train_loader, _ = task_dataloaders[task_id]
        print(f"\n--- Training on Task {task_id + 1}/{NUM_TASKS} ---")
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        for epoch in range(EPOCHS_PER_TASK):
            model.train()
            pbar = tqdm(train_loader)
            for images, labels in pbar:
                images, labels = images.to(device=DEVICE), labels.to(device=DEVICE)
                optimizer.zero_grad()
                
                outputs = model(images, task_id=task_id)
                loss = criterion(outputs, labels)
                
                if task_id > 0:
                    re_batch = cl_strategy.get_rehearsal_batch(images.size(0))
                    if re_batch is not None:
                        re_images, re_labels, re_task_ids = re_batch
                        re_images, re_labels = re_images.to(DEVICE), re_labels.to(DEVICE)
                        
                        for t_id in range(task_id):
                            mask = (re_task_ids == t_id)
                            if mask.sum() > 1:
                                re_outputs = model(re_images[mask], task_id=t_id)
                                loss += criterion(re_outputs, re_labels[mask])

                loss.backward()
                optimizer.step()
                pbar.set_description(f"Epoch {epoch+1}/{EPOCHS_PER_TASK} | Loss: {loss.item():.4f}")
            
            scheduler.step()
        
        cl_strategy.on_task_end(task_id, train_loader)

        # --- Evaluation ---
        task_accuracies = []
        for i in range(task_id + 1):
            _, test_loader = task_dataloaders[i]
            acc = evaluate(model, test_loader, DEVICE, task_id=i)
            task_accuracies.append(acc)
        results.append(task_accuracies)

        print(f"Accuracies after Task {task_id + 1}: {['{:.2f}'.format(acc) for acc in task_accuracies]}")
        avg_acc = np.mean(task_accuracies)
        print(f"Average Accuracy: {avg_acc:.2f}%")
        
    # --- Final Results Analysis ---
    final_accuracies = results[-1]
    avg_final_accuracy = np.mean(final_accuracies)
    
    forgetting = 0
    if NUM_TASKS > 1:
        for i in range(NUM_TASKS - 1):
            max_acc_i = max([res[i] for res in results if len(res) > i])
            final_acc_i = final_accuracies[i]
            forgetting += (max_acc_i - final_acc_i)
        avg_forgetting = forgetting / (NUM_TASKS - 1)
    else:
        avg_forgetting = 0

    print("\n--- Final Report ---")
    print(f"Average Accuracy across all tasks at the end: {avg_final_accuracy:.2f}%")
    print(f"Average Forgetting: {avg_forgetting:.2f}%")

if __name__ == "__main__":
    main()