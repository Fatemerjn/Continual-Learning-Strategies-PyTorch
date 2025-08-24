import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import numpy as np

# Import our custom modules
from utils import get_cifar100_dataloaders
from strategies import EWC #, PackNet, GEM

def evaluate(model, test_loader, device):
    """
    Evaluates the model on a given test loader.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    # --- Configuration ---
    NUM_TASKS = 10
    BATCH_SIZE = 64
    EPOCHS_PER_TASK = 5
    LR = 0.01
    STRATEGY = 'EWC'  # Choose from 'Finetune', 'EWC'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running on device: {DEVICE}")
    print(f"Using strategy: {STRATEGY}")

    # --- Data Loading ---
    task_dataloaders = get_cifar100_dataloaders(num_tasks=NUM_TASKS, batch_size=BATCH_SIZE)

    # --- Model Setup ---
    # We use a ResNet18 but modify the final layer for 10 classes per task
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10) # 10 classes per task
    model.to(DEVICE)

    # --- Strategy Setup ---
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    cl_strategy = None
    if STRATEGY == 'EWC':
        cl_strategy = EWC(model, optimizer, criterion, DEVICE, lambda_ewc=5000.0)

    # --- Training Loop ---
    results = [] # To store accuracies on all tasks after each task is learned

    for task_id in range(NUM_TASKS):
        train_loader, _ = task_dataloaders[task_id]
        print(f"\n--- Training on Task {task_id + 1}/{NUM_TASKS} ---")

        for epoch in range(EPOCHS_PER_TASK):
            model.train()
            pbar = tqdm(train_loader)
            for images, labels in pbar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Add penalty from CL strategy if it exists
                if cl_strategy and task_id > 0:
                    penalty = cl_strategy.penalty(task_id)
                    loss += penalty

                loss.backward()
                optimizer.step()
                
                pbar.set_description(f"Epoch {epoch+1}/{EPOCHS_PER_TASK} | Loss: {loss.item():.4f}")
        
        # --- End of Task ---
        if cl_strategy:
            cl_strategy.on_task_end(task_id, train_loader)

        # --- Evaluation ---
        task_accuracies = []
        for i in range(task_id + 1):
            _, test_loader = task_dataloaders[i]
            acc = evaluate(model, test_loader, DEVICE)
            task_accuracies.append(acc)
        results.append(task_accuracies)

        print(f"Accuracies after Task {task_id + 1}: {['{:.2f}'.format(acc) for acc in task_accuracies]}")
        avg_acc = np.mean(task_accuracies)
        print(f"Average Accuracy: {avg_acc:.2f}%")

    # --- Final Results Analysis ---
    final_accuracies = results[-1]
    avg_final_accuracy = np.mean(final_accuracies)
    
    # Calculate forgetting
    forgetting = 0
    for i in range(NUM_TASKS - 1):
        # Max accuracy on task i
        max_acc_i = max([res[i] for res in results if len(res) > i])
        # Accuracy on task i after the last task
        final_acc_i = final_accuracies[i]
        forgetting += (max_acc_i - final_acc_i)
    
    avg_forgetting = forgetting / (NUM_TASKS - 1)

    print("\n--- Final Report ---")
    print(f"Average Accuracy across all tasks at the end: {avg_final_accuracy:.2f}%")
    print(f"Average Forgetting: {avg_forgetting:.2f}%")


if __name__ == "__main__":
    main()