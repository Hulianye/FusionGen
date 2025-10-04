import os
import random
import math
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.init as init
import datetime
from torch.utils.data import DataLoader, TensorDataset
from model.model import FusionGen

class EEGBaseProcessor:
    """Base functionality for EEG data processing"""
    
    def __init__(self, subject_num=9, train_num=15, seed_range=range(3)):
        """Initialize base EEG processor with configuration parameters"""
        self.subject_num = subject_num
        self.train_num = train_num
        self.seed_range = seed_range
        self.all_best_history = np.zeros((len(seed_range), subject_num))
        
        # Set up device (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def add_noise_to_data(self, data, noise_mod_val=2):
        """Add noise to EEG data using the same approach as data_noise_f in data_augment.py"""
        new_data = []
        
        for i in range(data.shape[0]):
            # Calculate standard deviation of the current sample
            stddev_t = np.std(data[i])
            
            # Generate random noise with mean 0 and normalized to [-0.5, 0.5]
            rand_t = np.random.rand(data[i].shape[0], data[i].shape[1])
            rand_t = rand_t - 0.5
            
            # Scale noise by standard deviation and modulation value
            to_add_t = rand_t * stddev_t / noise_mod_val
            
            # Add noise to the original data
            data_t = data[i] + to_add_t
            new_data.append(data_t)
        
        # Convert to numpy array and reshape
        new_data_ar = np.array(new_data).reshape(data.shape)
        return new_data_ar

    def set_random_seed(self, seed):
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def prepare_data_loader(self, data, labels, subjects=None, shuffle=False, batch_size=64):
        """Prepare DataLoader for training or testing"""
        # Convert data to tensors
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Create dataset based on whether subjects are provided
        if subjects is not None:
            subjects_tensor = torch.tensor(subjects, dtype=torch.int64)
            dataset = TensorDataset(data_tensor, labels_tensor, subjects_tensor)
        else:
            dataset = TensorDataset(data_tensor, labels_tensor)
        
        # Create DataLoader
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def initialize_result_file(self, summary_file, subject_num):
        """Initialize result file and write header"""
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        
        with open(summary_file, 'w') as f_sum:
            header = f"{'Method':<15}" + "".join([f"{f'Sub{i}':>8}" for i in range(1, subject_num+1)]) + f"{'Avg':>10}\n"
            f_sum.write(header)

    def train_denoising_model(self, train_data, train_data_noise, train_labels, train_subjects, epochs=100, batch_size=64, shuffle=True):
        """Train the UNet2D denoising model"""
        # Prepare data loader
        train_dataset = TensorDataset(
            torch.tensor(train_data, dtype=torch.float32),
            torch.tensor(train_data_noise, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long),
            torch.tensor(train_subjects, dtype=torch.int64)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        
        # Initialize model, loss function, and optimizer
        model = FusionGen(train_data.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Training loop
        for epoch in range(epochs):
            running_loss = 0
            for data, data_noise, _, _ in train_loader:
                data = data.to(self.device)
                data_noise = data_noise.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data_noise)
                loss = criterion(outputs, data)
                
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
            
            print(f'test:Loss: {running_loss / len(train_loader):.4f}')
        
        return model

    def save_results_to_file(self, summary_file, seed, best_history, subject_num):
        """Save results to the summary file"""
        best_avg = np.mean(best_history)
        
        with open(summary_file, 'a') as f_sum:
            row = f"Seed_{seed:<12}" + "".join([f"{best_history[i]:>8.2f}%" for i in range(subject_num)]) + f"{best_avg:>10.2f}%\n"
            f_sum.write(row)
        
        return best_avg

    def save_final_average(self, summary_file, all_best_history):
        """Save the final average across all seeds"""
        final_total_avg = np.mean(all_best_history)
        with open(summary_file, 'a') as f_sum:
            f_sum.write(f"\n{'Final Avg':<15}" + f"{final_total_avg:>10.2f}%")
        
        return final_total_avg

    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        torch.cuda.empty_cache()

    def augment_data_with_model(self, model, train_loader, device, alpha=0.2, include_original=True):
        """Perform data augmentation using the trained model"""
        # Initialize augmented data lists
        train_data_augment = []
        label_augment = []
        subjects_augment = []
        
        # Set model to evaluation mode
        model.eval()
        
        with torch.no_grad():
            for data, label, subjects_batch in train_loader:
                data = data.to(device)
                out = model.fusedforward(data, label, alpha=alpha)
                
                # Add original and/or augmented data to the lists
                if include_original:
                    train_data_augment.append(data.cpu().numpy())
                    label_augment.append(label.cpu().numpy())
                    subjects_augment.append(subjects_batch.cpu().numpy())
                
                train_data_augment.append(out.cpu().numpy())
                label_augment.append(label.cpu().numpy())
                subjects_augment.append(subjects_batch.cpu().numpy())
        
        # Concatenate augmented data
        train_data_augment = np.concatenate(train_data_augment, axis=0)
        label_augment = np.concatenate(label_augment, axis=0)
        subjects_augment = np.concatenate(subjects_augment, axis=0)
        
        return train_data_augment, label_augment, subjects_augment