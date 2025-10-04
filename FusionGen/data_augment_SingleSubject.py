import os
import random
import numpy as np
import torch
import datetime
from typing import List, Tuple, Dict, Any, Optional, Union

from data.dataset001 import import_data_BNCI2014_001, getdata
from data.dataset002 import download_data_BNCI2014_002
from data.datasetZhou2016 import import_data_Zhou2016
from Utils.data_augment import data_aug
from Utils.CSP_LDA import CSP_LDA


# Make sure the log directory exists
def ensure_log_directory(log_path: str) -> None:
    """Ensure the log directory exists."""
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)


class SingleSubjectAugmentationEvaluator:
    """
    Evaluator for single-subject EEG data augmentation techniques.
    This class provides a framework to evaluate the performance of different
    data augmentation methods on single-subject EEG classification tasks.
    """
    
    def __init__(self, dataset: str = 'BNCI2014001', subject_num: int = 9, train_num: int = 20, 
                 seed: int = 0):
        """
        Initialize the evaluator with configuration parameters.
        
        Args:
            dataset: Dataset name (default: 'BNCI2014001')
            subject_num: Number of subjects in the dataset (default: 9)
            train_num: Number of training samples per subject (default: 20)
            seed: Random seed for reproducibility (default: 0)

        """
        self.dataset = dataset
        self.subject_num = subject_num
        self.train_num = train_num
        self.seed = seed
        self.subjects_to_evaluate = range(1, subject_num + 1)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize summary file
        self.summary_file = f'log/SingleSubject/{dataset}_{train_num}_augment.txt'
        self._create_summary_file()
    
    def _create_summary_file(self) -> None:
        """Create the summary file and write the header."""
        try:
            # Ensure log directory exists
            ensure_log_directory(self.summary_file)
            
            # Write header to summary file
            with open(self.summary_file, 'w') as f_sum:
                # Add timestamp to the header
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f_sum.write(f"# Single-subject Augmentation Evaluation - {timestamp}\n")
                f_sum.write(f"# Dataset: {self.dataset}, Subjects: {self.subject_num}, Training samples: {self.train_num}\n")
                
                # Write column headers
                header = f"{'Method':<15}" + "".join([f"{f'Sub{i}':>8}" for i in range(1, self.subject_num + 1)]) + f"{'Avg':>10}"
                f_sum.write(header + "\n")
            
            print(f"Summary file created: {self.summary_file}")
            print(f"Configuration: Dataset={self.dataset}, Subjects={self.subject_num}, Training samples={self.train_num}")
        except Exception as e:
            print(f"Error creating summary file: {e}")
    
    def _set_random_seed(self, seed_value: int) -> None:
        """Set random seed for all libraries to ensure reproducibility."""
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
    
    
    def evaluate_method(self, flag_aug: str, data: np.ndarray, labels: np.ndarray, 
                        subjects: np.ndarray, sessions: np.ndarray) -> Tuple[List[float], float]:
        """
        Evaluate a specific augmentation method.
        
        Args:
            flag_aug: Augmentation method to evaluate
            data: Input EEG data with shape (samples, channels, time)
            labels: Corresponding labels
            subjects: Subject IDs
            sessions: Session IDs
        
        Returns:
            Tuple containing list of best accuracy for each subject and average accuracy
        """
        best_history: List[float] = []
        
        
        for idx_subject in self.subjects_to_evaluate:
            try:
                # Set random seed for reproducibility
                self._set_random_seed(self.seed)
                
                # Select current subject only
                subject_selected = [idx_subject]
                
                # Load and prepare data for the current subject
                Train, Test = getdata(data, labels, subjects, sessions,
                                     session_selected=None,
                                     subject_selected=subject_selected,
                                     val_subect_selected=idx_subject,
                                     train_num=self.train_num)
                
                if Train is None or Test is None:
                    print(f"Skipping subject {idx_subject} due to data loading failure")
                    best_history.append(0.0)
                    continue
                
                # Apply data augmentation if specified
                if flag_aug != 'None':
                    try:
                        # Generate augmented data
                        inputs, ag_labels, ag_subjects = data_aug(
                            Train.data, Train.label, Train.subject,
                            target=None, flag_aug=flag_aug, dataset=self.dataset)
                        
                        # Concatenate original and augmented data
                        Train.data = np.concatenate((Train.data, inputs), axis=0)
                        Train.label = np.concatenate((Train.label, ag_labels), axis=0)
                        Train.subject = np.concatenate((Train.subject, ag_subjects), axis=0)
                    except Exception as aug_error:
                        print(f"Augmentation error for subject {idx_subject} with method {flag_aug}: {aug_error}")
                        # Continue with original data if augmentation fails
                
                # Evaluate using CSP-LDA
                # Using LDA as the classifier type as in the original code
                best = CSP_LDA(Train, Test, 'LDA')
                best_history.append(best)
                print(f"Subject: {idx_subject}, best: {best:.2f}%")
                
            except Exception as e:
                print(f"Error processing subject {idx_subject}: {e}")
                # Append a default value to maintain list length
                best_history.append(0.0)
        
        # Calculate average accuracy (ignoring zero values from errors)
        valid_scores = [score for score in best_history if score > 0]
        best_avg = np.mean(valid_scores) if valid_scores else 0.0
        
        # Write results to summary file
        try:
            row = f"{flag_aug:<15}"
            for result in best_history:
                row += f"{result:>8.2f}%"
            row += f"{best_avg:>10.2f}%"
            
            with open(self.summary_file, 'a') as f_sum:
                f_sum.write(row + "\n")
        except Exception as e:
            print(f"Error writing to summary file: {e}")
        
        print(f"best_avg: {best_avg:.2f}%")
        return best_history, best_avg
    
    def run_evaluation(self, method_list: Optional[List[str]] = None) -> None:
        """
        Run evaluation for a list of augmentation methods.
        
        Args:
            method_list: List of augmentation methods to evaluate
        """
        if method_list is None:
            method_list = ['None', 'Noise', 'Scale', 'Flip', 'Cutdown_Resize', 'Fshift', 'CR', 'DWTaug', 'HHTaug']
        
        # Import dataset
        try:
            if self.dataset == 'BNCI2014001':
                data, labels, subjects, sessions = import_data_BNCI2014_001()
            elif self.dataset == 'BNCI2014002':
                data, labels, subjects, sessions = download_data_BNCI2014_002()
            elif self.dataset == 'Zhou2016':
                data, labels, subjects, sessions = import_data_Zhou2016()
            else:
                print(f"Warning: Unknown dataset '{self.dataset}', using BNCI2014001 instead")
                data, labels, subjects, sessions = import_data_BNCI2014_001()
                
            print(f"Dataset loaded: {self.dataset} with {data.shape[0]} samples")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
        
        # Evaluate each method
        for flag_aug in method_list:
            print(f"\nEvaluating method: {flag_aug}")
            try:
                self.evaluate_method(flag_aug, data, labels, subjects, sessions)
            except Exception as e:
                print(f"Error evaluating method {flag_aug}: {e}")
                continue


if __name__ == '__main__':
    # Configuration parameters
    config = {
        'dataset': 'BNCI2014001',
        'subject_num': 9,
        'train_num': 10,
        'seed': 0,
        'methods': ['None', 'Noise', 'Scale', 'Flip', 'Cutdown_Resize', 'Fshift', 'CR', 'DWTaug', 'HHTaug']
    }
    
    try:
        # Initialize evaluator
        print(f"Initializing evaluator with configuration: {config}")
        evaluator = SingleSubjectAugmentationEvaluator(
            dataset=config['dataset'],
            subject_num=config['subject_num'],
            train_num=config['train_num'],
            seed=config['seed']
        )
        
        # Run evaluation
        print("Starting single-subject augmentation evaluation...")
        evaluator.run_evaluation(config['methods'])
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()