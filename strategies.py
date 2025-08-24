import torch
import torch.nn as nn

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
                    fisher = self.fisher_matrices[tid][n].to(self.device)
                    opt_param = self.optimal_params[tid][n].to(self.device)
                    penalty += (fisher * (p - opt_param) ** 2).sum()
        return self.lambda_ewc * penalty

    def on_task_end(self, task_id, train_loader):
        print(f"Calculating Fisher Information Matrix for Task {task_id + 1}...")
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
                    fisher[n] += p.grad.detach().clone() ** 2
        
        # Average Fisher over number of samples
        num_samples = len(train_loader.dataset)
        for n in fisher:
            fisher[n] /= num_samples
        
        self.fisher_matrices[task_id] = fisher
        self.model.train()
        print("Fisher Matrix calculation complete.")