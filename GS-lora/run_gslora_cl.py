# run_gslora_cl.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from data_and_models import get_cifar100_dataloaders, MultiHeadLoRAModel
from lora_utils import get_gradient_sparse_mask, apply_gradient_mask

def evaluate(model, test_loaders, device):
    model.eval()
    task_accuracies = []
    for task_id, loader in enumerate(test_loaders):
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels, _ in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, task_id=task_id)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        task_accuracies.append(100 * correct / total)
    return task_accuracies

def main():
    # --- Config ---
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    NUM_TASKS = 5  # Reduced for speed
    EPOCHS_PER_TASK = 5 # Reduced for speed
    LR = 0.005
    LORA_RANK = 4
    SPARSITY = 0.8 # Keep only top 20% of gradients

    print(f"Using device: {DEVICE}")

    # --- Data ---
    task_dataloaders = get_cifar100_dataloaders(num_tasks=NUM_TASKS, batch_size=64)

    # --- Model ---
    # Create one model with a separate LoRA adapter for each task
    model = MultiHeadLoRAModel(num_tasks=NUM_TASKS, rank=LORA_RANK).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    # --- Continual Learning Loop ---
    test_loaders = [loader for _, loader in task_dataloaders]
    results = []

    for task_id in range(NUM_TASKS):
        print(f"\n--- Training on Task {task_id + 1}/{NUM_TASKS} ---")
        train_loader, _ = task_dataloaders[task_id]
        
        # Select the specific model for the current task
        task_model = model.tasks_models[task_id]

        # --- GS-LoRA Step: Get mask for the current task's adapter ---
        masks = get_gradient_sparse_mask(task_model, train_loader, DEVICE, sparsity=SPARSITY)

        optimizer = optim.Adam([p for p in task_model.parameters() if p.requires_grad], lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(EPOCHS_PER_TASK):
            task_model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_PER_TASK}")
            for images, labels, _ in pbar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = task_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                apply_gradient_mask(task_model, masks)
                
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
            scheduler.step()
        
        # --- Evaluation after each task ---
        accuracies = evaluate(model, test_loaders[:task_id+1], DEVICE)
        results.append(accuracies)
        print(f"Accuracies after Task {task_id + 1}: {['{:.2f}'.format(acc) for acc in accuracies]}")

    # --- Final Report ---
    final_accuracies = results[-1]
    avg_acc = np.mean(final_accuracies)
    forgetting = 0
    for i in range(len(final_accuracies) - 1):
        forgetting += max(res[i] for res in results if len(res) > i) - final_accuracies[i]
    avg_forgetting = forgetting / (len(final_accuracies) - 1) if len(final_accuracies) > 1 else 0

    print(f"\n--- Final Report ---")
    print(f"Final Average Accuracy: {avg_acc:.2f}%")
    print(f"Average Forgetting: {avg_forgetting:.2f}%")

if __name__ == "__main__":
    main()