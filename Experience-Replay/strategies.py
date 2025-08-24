import torch
import torch.nn as nn
import random

class ExperienceReplay(nn.Module):
    def __init__(self, buffer_size_per_task=200):
        super(ExperienceReplay, self).__init__()
        self.buffer_size_per_task = buffer_size_per_task
        self.buffer = []

    def get_rehearsal_batch(self, batch_size):
        if not self.buffer:
            return None
        
        num_samples_to_replay = min(batch_size, len(self.buffer))
        rehearsal_samples = random.sample(self.buffer, num_samples_to_replay)
        
        images, labels, task_ids = zip(*rehearsal_samples)
        
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        task_ids = torch.tensor(task_ids, dtype=torch.long)
        
        return images, labels, task_ids

    def on_task_end(self, task_id, train_loader):
        print(f"Updating memory buffer after Task {task_id + 1}...")
        
        num_samples_to_add = self.buffer_size_per_task
        samples_added = 0
        
        max_samples_per_task = self.buffer_size_per_task
        
        new_buffer = []
        task_counts = {i: 0 for i in range(task_id + 1)}
        
        for sample in self.buffer:
            t_id = sample[2]
            if task_counts[t_id] < max_samples_per_task:
                new_buffer.append(sample)
                task_counts[t_id] += 1
        self.buffer = new_buffer
        
        for images, labels in train_loader:
            for i in range(len(images)):
                if samples_added < num_samples_to_add:
                    self.buffer.append((images[i], labels[i].item(), task_id))
                    samples_added += 1
                else:
                    break
            if samples_added >= num_samples_to_add:
                break
        
        print(f"Buffer size is now: {len(self.buffer)}")