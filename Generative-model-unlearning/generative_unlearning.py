import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import numpy as np
import copy

from data_setup import get_cifar100_dataloaders
from generative_models import GenerativeReplay

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
def evaluate(model, test_loaders, device, tasks_to_evaluate):
    model.eval()
    accuracies = []
    for task_id in tasks_to_evaluate:
        loader = test_loaders[task_id]
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, task_id=task_id)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracies.append((task_id, 100 * correct / total))
    return accuracies

def main():
    NUM_TASKS = 5
    CLASSES_PER_TASK = 100 // NUM_TASKS
    BATCH_SIZE = 64
    EPOCHS_PER_TASK = 10
    LR = 0.01

    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    task_dataloaders = get_cifar100_dataloaders(num_tasks=NUM_TASKS, batch_size=BATCH_SIZE)

    solver_model = MultiHeadResNet(num_tasks=NUM_TASKS, num_classes_per_task=CLASSES_PER_TASK).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    cl_strategy = GenerativeReplay(device=DEVICE, num_tasks=NUM_TASKS, generator_epochs=5)

    # --- Phase 1: Continual Learning ---
    print("--- Phase 1: Continual Learning ---")
    for task_id in range(NUM_TASKS):
        train_loader, _ = task_dataloaders[task_id]
        print(f"\n--- Training Solver on Task {task_id + 1}/{NUM_TASKS} ---")
        
        # This is better for CL as each task is a distinct distribution.
        optimizer = optim.Adam(solver_model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        for epoch in range(EPOCHS_PER_TASK):
            solver_model.train()
            pbar = tqdm(train_loader, desc=f"Solver Epoch {epoch+1}/{EPOCHS_PER_TASK}")
            for images, labels in pbar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                
                loss = 0
                outputs = solver_model(images, task_id=task_id)
                loss += criterion(outputs, labels)
                
                if task_id > 0:
                    re_batch = cl_strategy.get_rehearsal_batch(task_id, images.size(0))
                    if re_batch is not None:
                        re_images, re_labels, re_task_ids = re_batch
                        re_images, re_labels = re_images.to(DEVICE), re_labels.to(DEVICE)
                        
                        for t_id in range(task_id):
                            mask = (re_task_ids == t_id)
                            if mask.sum() > 1:
                                re_outputs = solver_model(re_images[mask], task_id=t_id)
                                loss += criterion(re_outputs, re_labels[mask])

                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
            scheduler.step()
        
        cl_strategy.on_task_end(task_id, train_loader)

    # --- Phase 2: Evaluation Before Unlearning ---
    print("\n--- Phase 2: Accuracies BEFORE Unlearning ---")
    test_loaders = [loader for _, loader in task_dataloaders]
    before_accuracies = evaluate(solver_model, test_loaders, DEVICE, tasks_to_evaluate=range(NUM_TASKS))
    before_avg_acc = np.mean([acc for _, acc in before_accuracies])
    for i, acc in before_accuracies:
        print(f"Task {i+1} Accuracy: {acc:.2f}%")
    print(f"Average Accuracy: {before_avg_acc:.2f}%")
    
    # --- Phase 3: Unlearning a Task ---
    task_to_forget = 2 
    print(f"\n--- Phase 3: 'Unlearning' Task {task_to_forget + 1} by excluding its generator ---")
    
    unlearned_strategy = copy.deepcopy(cl_strategy)
    
    unlearned_strategy.generators[task_to_forget] = None 
    
    print("Performing a short fine-tuning on remaining tasks...")
    optimizer = optim.Adam(solver_model.parameters(), lr=LR/10)
    for epoch in range(3): # A few epochs of fine-tuning
        solver_model.train()
        pbar = tqdm(range(50), desc=f"Unlearning Epoch {epoch+1}/3")
        for _ in pbar:
            optimizer.zero_grad()
            # Rehearse from the UNLEARNED strategy
            re_batch = unlearned_strategy.get_rehearsal_batch(NUM_TASKS, BATCH_SIZE)
            if re_batch is not None:
                re_images, re_labels, re_task_ids = re_batch
                re_images, re_labels = re_images.to(DEVICE), re_labels.to(DEVICE)
                
                loss = 0
                for t_id in range(NUM_TASKS):
                    if t_id == task_to_forget: continue # Skip forgotten task
                    mask = (re_task_ids == t_id)
                    if mask.sum() > 1:
                        re_outputs = solver_model(re_images[mask], task_id=t_id)
                        loss += criterion(re_outputs, re_labels[mask])
                
                if loss != 0:
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item())

    # --- Phase 4: Evaluation After Unlearning ---
    print(f"\n--- Phase 4: Accuracies AFTER Unlearning Task {task_to_forget + 1} ---")
    after_accuracies = evaluate(solver_model, test_loaders, DEVICE, tasks_to_evaluate=range(NUM_TASKS))
    after_avg_acc = np.mean([acc for _, acc in after_accuracies])
    
    before_acc_map = dict(before_accuracies)
    for i, acc in after_accuracies:
        status = " (FORGOTTEN)" if i == task_to_forget else " (RETAINED)"
        change = acc - before_acc_map[i]
        print(f"Task {i+1} Accuracy: {acc:.2f}% {status} | Change: {change:+.2f}%")
    
    retained_acc = np.mean([acc for i, acc in after_accuracies if i != task_to_forget])
    print(f"\nBefore Unlearning Average Accuracy: {before_avg_acc:.2f}%")
    print(f"After Unlearning Average Accuracy: {after_avg_acc:.2f}%")
    print(f"Average Accuracy on Retained Tasks: {retained_acc:.2f}%")


if __name__ == "__main__":
    main()