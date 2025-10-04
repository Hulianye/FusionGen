import os
import random
import math
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.init as init
import datetime
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from data.dataset001 import *
from torch.utils.data import DataLoader, TensorDataset
from Utils.data_augment import data_aug
from Utils.CSP_LDA import CSP_LDA
from model.model import FusionGen
from FusionGen_BaseProcessor import EEGBaseProcessor

class EEGSingleSubjectProcessor(EEGBaseProcessor):
    def __init__(self, subject_num=9, train_num=14, seed_range=range(3)):
        """Initialize EEG single subject processor with configuration parameters"""
        super().__init__(subject_num, train_num, seed_range)
        self.summary_file = f'log/SingleSubject/BNCI2014001_{train_num}_FusionGen.txt'
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(self.summary_file), exist_ok=True)

    def initialize_result_file(self):
        """Initialize result file and write header"""
        super().initialize_result_file(self.summary_file, self.subject_num)

    def train_denoising_model(self, train_data, train_data_noise, train_labels, train_subjects, epochs=100):
        """Train the UNet2D denoising model"""
        return super().train_denoising_model(
            train_data, 
            train_data_noise, 
            train_labels, 
            train_subjects, 
            epochs=epochs, 
            batch_size=64, 
            shuffle=True
        )

    def process_subject(self, model, data, labels, subjects, sessions, idx_subject, batch_size=64):
        """Process a single subject for single subject evaluation"""
        # Get training and testing data
        Train, Test = getdata(
            data, labels, subjects, sessions,
            session_selected=None,
            subject_selected=[idx_subject],
            val_subect_selected=idx_subject,
            train_num=self.train_num
        )
        
        # Apply EA preprocessing
        subject_mask = (Train.subject == idx_subject)
        Train.data[subject_mask], refEA, numEA = EA(Train.data[subject_mask])
        
        Test.data, _, _ = EA(Test.data)
        
        # Prepare data loaders
        train_loader = self.prepare_data_loader(
            Train.data, Train.label, Train.subject, shuffle=False, batch_size=batch_size
        )
        
        # Prepare reference data for the current subject
        ref_data = Train.data[subject_mask]
        ref_labels = Train.label[subject_mask]
        
        # Encode reference data using the model
        ref_data_tensor = torch.tensor(ref_data, dtype=torch.float32).to(self.device)
        ref_labels_tensor = torch.tensor(ref_labels, dtype=torch.long).to(self.device)
        model.encode_reference(ref_data_tensor, ref_labels_tensor)
        model.eval()
        
        # Initialize augmented data lists
        train_data_augment = []
        label_augment = []
        subjects_augment = []
        
        # Perform data augmentation using the model
        with torch.no_grad():
            for data, label, subjects_batch in train_loader:
                data = data.to(self.device)
                out = model.fusedforward(data, label, alpha=0.3)
                
            # Use the shared method for data augmentation
            train_data_augment, label_augment, subjects_augment = self.augment_data_with_model(
                model, train_loader, self.device, alpha=0.3, include_original=True)
        
        # Create augmented training data object
        Train_augment = DATA(train_data_augment, label_augment, subjects_augment)
        
        # Evaluate using CSP-LDA
        best = CSP_LDA(Train_augment, Test, 'LDA')
        print(f"best_avg: {best:.2f}%\n")
        
        print(f"Subject: {idx_subject}, best: {best:.2f}%\n")
        
        return best

    def run_single_subject_evaluation(self):
        """Run the complete single subject evaluation pipeline"""
        # Initialize result file
        self.initialize_result_file()
        
        # Import dataset
        Data, Labels, Subjects, Sessions = import_data_BNCI2014_001()
        
        # Process each seed
        for seed_idx, SEED in enumerate(self.seed_range):
            # Set random seed
            self.set_random_seed(SEED)
            
            # Initialize model and best history for this seed
            model = None
            best_history = []
            
            # Process each subject individually
            for idx_subject in range(1, self.subject_num + 1):
                # Get training data for the current subject
                Train, _ = getdata(
                    Data, Labels, Subjects, Sessions,
                    session_selected=None,
                    subject_selected=[idx_subject],
                    val_subect_selected=None,
                    train_num=self.train_num
                )
                
                # Apply EA preprocessing
                subject_mask = (Train.subject == idx_subject)
                Train.data[subject_mask], refEA, numEA = EA(Train.data[subject_mask])
                
                # Add noise to training data using our custom function
                data_noise = self.add_noise_to_data(Train.data)
                
                # Train denoising model
                model = self.train_denoising_model(
                    Train.data,
                    data_noise,
                    Train.label,
                    Train.subject,
                    epochs=50
                )
                
                # Process the subject with the trained model
                best = self.process_subject(
                    model, Data, Labels, Subjects, Sessions, idx_subject
                )
                best_history.append(best)
            
            # Record results for this seed
            best_avg = np.mean(best_history)
            self.all_best_history[seed_idx] = best_history
            
            # Write results to file immediately after each seed
            with open(self.summary_file, 'a') as f_sum:
                row = f"Seed_{SEED:<12}" + "".join([f"{best_history[i]:>8.2f}%" for i in range(self.subject_num)]) + f"{best_avg:>10.2f}%\n"
                f_sum.write(row)
            
            # Clear GPU cache
            self.clear_gpu_cache()
        
        # Calculate final average across all seeds
        self.save_final_average(self.summary_file, self.all_best_history)

if __name__ == '__main__':
    # Initialize and run the EEG single subject processor
    processor = EEGSingleSubjectProcessor(subject_num=9, train_num=10, seed_range=range(10))
    processor.run_single_subject_evaluation()