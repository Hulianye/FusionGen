import os
import random
import math
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.init as init
from matplotlib import pyplot as plt
from data.dataset001 import *
from torch.utils.data import DataLoader, TensorDataset
import datetime
from scipy.interpolate import interp1d
from Utils.CSP_LDA import *
from model.model import FusionGen
from FusionGen_BaseProcessor import EEGBaseProcessor

class EEGDataProcessor(EEGBaseProcessor):
    def __init__(self, subject_num=9, train_num=15, seed_range=range(3)):
        """Initialize EEG data processor with configuration parameters"""
        super().__init__(subject_num, train_num, seed_range)
        self.summary_file = f'log/CrossSubject/BNCI2014001_{train_num}_FusionGen.txt'
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(self.summary_file), exist_ok=True)

    def initialize_result_file(self):
        """Initialize result file and write header"""
        super().initialize_result_file(self.summary_file, self.subject_num)

    def train_denoising_model(self, train_data, train_data_noise, train_labels, train_subjects):
        """Train the UNet2D denoising model"""
        return super().train_denoising_model(
            train_data, 
            train_data_noise, 
            train_labels, 
            train_subjects, 
            epochs=40, 
            batch_size=64, 
            shuffle=False
        )

    def process_subject(self, model, data, labels, subjects, sessions, idx_subject):
        """Process a single subject for cross-subject evaluation"""
        # Get training and testing data
        Train, Test = getdata(
            data, labels, subjects, sessions,
            session_selected=None,
            subject_selected=range(1, self.subject_num+1),
            val_subect_selected=idx_subject,
            train_num=self.train_num
        )
        
        # Apply EA preprocessing
        for f_subject in range(1, self.subject_num+1):
            subject_mask = (Train.subject == f_subject)
            Train.data[subject_mask], refEA, numEA = EA(Train.data[subject_mask])
        
        Test.data, _, _ = EA(Test.data)
        
        # Prepare data loaders
        train_loader = self.prepare_data_loader(
            Train.data, Train.label, Train.subject, shuffle=False
        )
        
        test_loader = self.prepare_data_loader(
            Test.data, Test.label, Test.subject, shuffle=True
        )
        
        # Prepare reference data for the current subject
        subject_mask = (Train.subject == idx_subject)
        ref_data = Train.data[subject_mask]
        ref_labels = Train.label[subject_mask]
        ref_subjects = Train.subject[subject_mask]
        
        # Initialize augmented data lists
        train_data_augment = [ref_data]
        label_augment = [ref_labels]
        subjects_augment = [ref_subjects]
        
        # Encode reference data using the model
        ref_data_tensor = torch.tensor(ref_data, dtype=torch.float32).to(self.device)
        ref_labels_tensor = torch.tensor(ref_labels, dtype=torch.long).to(self.device)
        model.encode_reference(ref_data_tensor, ref_labels_tensor)
        model.eval()
        
        # Start timing for this subject processing
        start_time = time.time()
        
        # Perform data augmentation using the model
        with torch.no_grad():
            for data, label, subjects_batch in train_loader:
                data = data.to(self.device)
                out = model.fusedforward(data, label, alpha=0.2)
                
                # Add original and augmented data to the lists
                train_data_augment.append(data.cpu().numpy())
                train_data_augment.append(out.cpu().numpy())
                label_augment.append(label.cpu().numpy())
                subjects_augment.append(subjects_batch.cpu().numpy())
                label_augment.append(label.cpu().numpy())
                subjects_augment.append(subjects_batch.cpu().numpy())
        
        # Concatenate augmented data
        train_data_augment = np.concatenate(train_data_augment, axis=0)
        label_augment = np.concatenate(label_augment, axis=0)
        subjects_augment = np.concatenate(subjects_augment, axis=0)
        
        # Create augmented training data object
        Train_augment = DATA(train_data_augment, label_augment, subjects_augment)
        
        # Evaluate using CSP-LDA
        best = CSP_LDA(Train_augment, Test, 'LDA')
        print(f"best_avg: {best:.2f}%\n")
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Subject {idx_subject} time: {elapsed_time:.2f}s")
        
        return best, elapsed_time

    def run_cross_subject_evaluation(self):
        """Run the complete cross-subject evaluation pipeline"""
        # Initialize result file
        self.initialize_result_file()
        
        # Import dataset
        Data, Labels, Subjects, Sessions = import_data_BNCI2014_001()
        
        # Process each seed
        for seed_idx, SEED in enumerate(self.seed_range):
            # Set random seed
            self.set_random_seed(SEED)
            
            # Get training and testing data for all subjects
            Train_all, Test_all = getdata(
                Data, Labels, Subjects, Sessions,
                session_selected=None,
                subject_selected=range(1, self.subject_num+1),
                val_subect_selected=None,
                train_num=self.train_num
            )
            
            # Apply EA preprocessing
            for f_subject in range(1, self.subject_num+1):
                subject_mask = (Train_all.subject == f_subject)
                Train_all.data[subject_mask], refEA, numEA = EA(Train_all.data[subject_mask])
            
            Test_all.data, _, _ = EA(Test_all.data)
            
            # Add noise to training data using our custom function
            data_noise = self.add_noise_to_data(Train_all.data)
            
            # Train denoising model
            model = self.train_denoising_model(
                Train_all.data,
                data_noise,
                Train_all.label,
                Train_all.subject
            )
            
            best_history = []
            time_records = []
            
            # Process each subject individually
            for idx_subject in range(1, self.subject_num+1):
                best, elapsed_time = self.process_subject(
                    model, Data, Labels, Subjects, Sessions, idx_subject
                )
                best_history.append(best)
                time_records.append(elapsed_time)
            
            # Calculate and print average time for this seed
            avg_time = np.mean(time_records)
            print(f"\nSeed {SEED} average time: {avg_time:.2f}s\n")
            
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
        final_total_avg = np.mean(self.all_best_history)
        with open(self.summary_file, 'a') as f_sum:
            f_sum.write(f"\n{'Final Avg':<15}" + f"{final_total_avg:>10.2f}%")

if __name__ == '__main__':
    # Initialize and run the EEG data processor
    processor = EEGDataProcessor(subject_num=9, train_num=10, seed_range=range(10))
    processor.run_cross_subject_evaluation()