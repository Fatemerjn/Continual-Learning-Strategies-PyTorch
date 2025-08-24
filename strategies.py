import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import copy

class EWC(nn.Module):
    """
    Elastic Weight Consolidation (EWC) strategy.
    """
    def __init__(self, model, optimizer, criterion, device, lambda_ewc=1000.0):
        super(EWC, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.fisher_matrices = {}
        self.optimal_params = {}

    def penalty(self, task_id):
        penalty = 0
        for tid in range(task_id):
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    # Access saved fisher and optimal params for the specific task
                    fisher = self.fisher_matrices[tid][n]
                    opt_param = self.optimal_params[tid][n]
                    penalty += (fisher * (p - opt_param) ** 2).sum()
        return penalty

    def on_task_end(self, task_id, train_loader):
        # 1. Store optimal parameters
        self.optimal_params[task_id] = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

        # 2. Calculate Fisher Information Matrix (diagonal approximation)
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
        
        # Average Fisher
        num_samples = len(train_loader.dataset)
        for n in fisher:
            fisher[n] /= num_samples
        
        self.fisher_matrices[task_id] = fisher
        self.model.train()

# NOTE: PackNet and GEM are more complex to implement fully in a simple script.
# The following are conceptual representations. A full implementation would
# require deeper integration with the model and optimizer.

class PackNet:
    """
    Conceptual implementation of PackNet.
    Requires model-level modification and a special optimizer.
    """
    def __init__(self, model):
        self.model = model
        self.masks = {}
        # ... more setup needed

    def on_task_end(self, task_id, prune_ratio=0.5):
        # 1. Prune the network based on weight magnitude
        # 2. Create and store a binary mask of pruned weights
        # 3. Freeze the pruned weights (they are now part of the mask)
        print(f"[PackNet] Task {task_id} finished. Pruning network.")

class GEM:
    """
    Conceptual implementation of Gradient Episodic Memory (GEM).
    """
    def __init__(self, model, margin=0.5):
        self.model = model
        self.margin = margin
        self.memory_buffer = []
        # ... more setup needed for buffer management

    def project_gradient(self, current_grad, past_grads):
        # Logic to project the current gradient if it conflicts with past gradients
        # This is the core of GEM and is computationally intensive.
        return current_grad