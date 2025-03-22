import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from dataclasses import dataclass
import random
from torch.utils.data import Dataset, DataLoader

from final_project_models import UniversityMLP, ApplicantMLP

@dataclass
class FacultyParams:
    """Parameters for each faculty"""
    name: str
    utility_vector: np.ndarray  # Hidden vector that determines student success
    capacity: int  # Number of spots available (can be infinite)

@dataclass
class SupplierParams:
    """Parameters for each preparation supplier"""
    name: str
    diff_vector: np.ndarray  # How this supplier modifies student features

class UniversityEnvironment:
    def __init__(
        self,
        n_features: int = 8,
        n_faculties: int = 5,
        n_suppliers: int = 20,
        noise_range: Tuple[float, float] = (0,0),
        enable_university_supplier: bool = False
    ):
        self.n_features = n_features
        self.n_faculties = n_faculties
        self.n_suppliers = n_suppliers
        self.noise_range = noise_range
        self.enable_university_supplier = enable_university_supplier
        self.university_applicants = set()  # Track applicants who chose university as supplier
        
        # Initialize faculties with random utility vectors
        # Initialize faculties with normalized random utility vectors
        self.faculties = [
            FacultyParams(
                name=f"faculty_{i}",  # Using the predefined faculty names
                utility_vector=self._create_normalized_vector(n_features),
                capacity=np.inf  # As per description, infinite capacity
            )
            for i in range(n_faculties)
        ] 
        
        # Initialize suppliers with random modification vectors
        self.suppliers = [
          SupplierParams(
              name=f"Supplier_{i}",
              diff_vector=np.array([
                  30 if j == idx1 else 20 if j == idx2 else -30 if j == idx3 else 0 if j == idx4 else 0
                  for j in range(n_features)
              ]),
          )
          for i in range(n_suppliers)
          for idx1, idx2, idx3, idx4 in [np.random.choice(n_features, size=4, replace=False)]
        ]
        
        self.past_applicants_df = None
        self.current_applicants_df = None

    def _create_normalized_vector(self, size: int) -> np.ndarray:
        """
        Create a normalized random vector of given size.
        Normalization ensures ||vector|| = 1
        """
        # Create vector with some high and some low values
        vector = np.random.uniform(0, 0.2, size)  # Base small values
        
        # Randomly select ~40% of elements to be higher values
        high_value_indices = np.random.choice(size, size=2, replace=False)
        vector[high_value_indices] = np.random.uniform(0.6, 1, size=len(high_value_indices))
        
        # Normalize to sum to 1 while preserving relative differences
        return vector / np.sum(vector)
    
    def _generate_truncated_normal_features(self, n_samples: int) -> np.ndarray:
        """
        Generate features using truncated normal distribution between 55 and 100.
        Uses mean at center of range (77.5) and std that makes the distribution fit well in the range.
        """
        # Generate features with normal distribution between 0 and 100
        features = np.random.normal(55, 40, (n_samples, self.n_features))
        features = np.clip(features, 0, 100)
        # features = np.random.uniform(0, 100, (n_samples, self.n_features))
        
        return features
    
    def generate_past_applicants(
        self,
        n_applicants: int = 100
    ) -> pd.DataFrame:
        """Generate dataset of past applicants with their outcomes"""
        # Generate random feature vectors
        features = self._generate_truncated_normal_features(n_applicants)
        
        # Randomly assign faculty for each applicant
        df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(self.n_features)])
        df['assigned_faculty'] = np.random.randint(0, self.n_faculties, n_applicants)
        
        # Calculate grade only for assigned faculty
        faculty_vectors = np.array([f.utility_vector for f in self.faculties])
        grades = np.zeros(n_applicants)
        # Get faculty vectors for each applicant based on their assigned faculty
        faculty_vectors_per_applicant = faculty_vectors[df['assigned_faculty']]
        
        # Calculate base grades using matrix multiplication
        base_grades = np.sum(features * faculty_vectors_per_applicant, axis=1)
        
        # Generate noise for all applicants at once
        noise = np.random.uniform(*self.noise_range, size=n_applicants)
        
        # Calculate final grades
        grades = base_grades + noise
            
        df['final_grade'] = grades
        self.past_applicants_df = df
        return df

    def generate_current_applicants(
        self,
        n_applicants: int = 100
    ) -> pd.DataFrame:
        """Generate dataset of current applicants"""
        # Generate random feature vectors
        features = self._generate_truncated_normal_features(n_applicants)
        
        # Create DataFrame
        feature_cols = [f"feature_{i}" for i in range(self.n_features)]
        df = pd.DataFrame(features, columns=feature_cols)
        
        # Add desired faculty (random)
        df['desired_faculty'] = np.random.randint(0, self.n_faculties, n_applicants)
        
        self.current_applicants_df = df
        return df
    
    def reconstruct_original_features(
        self,
        modified_features: np.ndarray,
        desired_faculties: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct approximate original features using group-wise mean vectors.
        
        Args:
            modified_features: Modified feature vectors of shape (n_students, n_features)
            desired_faculties: Array of desired faculty indices for each student
            
        Returns:
            Reconstructed original feature vectors
        """
        # Calculate global mean vector
        global_mean = np.mean(modified_features, axis=0)
        
        # Initialize reconstructed features array
        reconstructed_features = np.zeros_like(modified_features)
        
        # Process each faculty group
        for faculty in range(self.n_faculties):
            # Get indices of students who desire this faculty
            faculty_mask = desired_faculties == faculty
            if not np.any(faculty_mask):
                continue
                
            # Calculate mean vector for this faculty group
            faculty_mean = np.mean(modified_features[faculty_mask], axis=0)
            
            # Calculate proportion vector (avoiding division by zero)
            proportion_vector = np.ones_like(global_mean)
            non_zero_mask = global_mean != 0
            proportion_vector[non_zero_mask] = faculty_mean[non_zero_mask] / global_mean[non_zero_mask]
            
            # Apply inverse proportion to reconstruct original features
            reconstructed_features[faculty_mask] = modified_features[faculty_mask] / proportion_vector
            
        return reconstructed_features

    def assign_applicants_to_faculties_with_reconstruction(
        self,
        model: UniversityMLP,
        modified_features: np.ndarray,
        desired_faculties: np.ndarray
    ) -> np.ndarray:
        """
        Assign applicants to faculties using reconstructed original features.
        
        Args:
            model: Trained UniversityMLP model
            modified_features: Modified features of current applicants
            desired_faculties: Array of desired faculty indices
            
        Returns:
            Array of assigned faculty indices
        """
        # First reconstruct the approximate original features
        reconstructed_features = self.reconstruct_original_features(modified_features, desired_faculties)
        
        # Use reconstructed features for prediction
        model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(reconstructed_features)
            predicted_grades = model(features_tensor)
            chosen_faculties = torch.argmax(predicted_grades, dim=1).numpy()
        
        return chosen_faculties
    
    def train_applicant_model(
        self,
        past_data: pd.DataFrame = None,
        verbose: bool = False
    ) -> ApplicantMLP:
        """
        Train applicant model on past data
        
        Args:
            past_data: Optional past data to train on. If None, uses self.past_applicants_df
            verbose: Whether to print training progress
            
        Returns:
            Trained ApplicantMLP model
        """
        if past_data is None:
            past_data = self.past_applicants_df
        
        if past_data is None:
            raise ValueError("No past data available. Generate past applicants first.")
        
        # Create and train applicant's MLP model
        feature_cols = [f"feature_{i}" for i in range(self.n_features)]
        X_train = torch.FloatTensor(past_data[feature_cols].values)
        y_train = torch.LongTensor(past_data['assigned_faculty'].values)
        
        model = ApplicantMLP(self.n_features, self.n_faculties)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Reduced learning rate
        
        # Train the model
        model.train()
        
        # Keep track of losses for progress reporting
        recent_losses = []
        if verbose:
            print(f"Training applicant model...")
        
        for epoch in range(500):  
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss_value = loss.item()
            
            # Store recent losses
            recent_losses.append(loss_value)
            if len(recent_losses) > 5:
                recent_losses.pop(0)
            
            # Only print progress occasionally or when loss improves significantly
            if verbose:
                if epoch < 5:
                    print(f"Epoch {epoch}: loss = {loss_value:.6f}")
                elif epoch % 50 == 0 or (len(recent_losses) >= 5 and loss_value < sum(recent_losses[:-1])/4):
                    avg_previous = sum(recent_losses[:-1])/len(recent_losses[:-1]) if len(recent_losses) > 1 else float('inf')
                    improvement = (avg_previous - loss_value) / avg_previous * 100 if avg_previous > 0 else 0
                    print(f"Epoch {epoch}: loss = {loss_value:.6f} (improved by {improvement:.2f}%)")
            
            loss.backward()
            optimizer.step()
        
        if verbose:
            print(f"Applicant model training complete. Final loss: {loss_value:.6f}")
        return model

    def choose_supplier_for_applicant(
        self,
        applicant_features: np.ndarray,
        desired_faculty: int,
        applicant_model: ApplicantMLP = None,
        applicant_id: int = None,
        with_university_supplier: bool = False
    ) -> Tuple[int, np.ndarray]:
        """Modified to include university as supplier option"""
        applicant_model.eval()
        best_probability = -1
        best_supplier_idx = -1
        best_modified_features = None
        
        original_features = torch.FloatTensor(applicant_features).unsqueeze(0)
        
        with torch.no_grad():
            # Try each supplier
            for i, supplier in enumerate(self.suppliers):
                modified_features_unclipped = original_features + torch.FloatTensor(supplier.diff_vector)
                modified_features = np.clip(modified_features_unclipped, 0, 100)
                
                probabilities = applicant_model(modified_features)
                prob_desired = probabilities[0, int(desired_faculty)].item()
                
                if prob_desired > best_probability:
                    best_probability = prob_desired
                    best_supplier_idx = i
                    best_modified_features = modified_features.squeeze(0).numpy()
        
        # Check if university supplier should be used
        if with_university_supplier and self.enable_university_supplier and (best_probability < 0.5 or best_supplier_idx == -1):
            if applicant_id is not None:
                self.university_applicants.add(applicant_id)
            return (-1, applicant_features)  # -1 indicates university supplier
        
        if best_supplier_idx == -1:
            return (-1, applicant_features)
        
        return (best_supplier_idx, best_modified_features)
    
    def assign_applicants_to_faculties_fully_exposed(
        self,
        model: UniversityMLP,
        current_applicants_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use trained model to make faculty recommendations for current applicants.

        Args:
            model: Trained UniversityMLP model
            current_applicants_features: Modified features of current applicants (n_applicants x n_features)

        Returns:
            Tuple of (chosen_faculties, predicted_grades)
            - chosen_faculties: Array of faculty indices chosen for each applicant
            - predicted_grades: Array of predicted grades for each applicant across all faculties
        """
        model.eval()
        with torch.no_grad():
            current_features = torch.FloatTensor(current_applicants_features)
            predicted_grades = model(current_features)

            # Choose best faculty for each applicant based on predicted grades
            chosen_faculties = torch.argmax(predicted_grades, dim=1).numpy()

        return chosen_faculties, predicted_grades.numpy()
        
    def choose_supplier_for_applicant_fully_exposed(
        self,
        applicant_features: np.ndarray,
        desired_faculty: int,
        trained_model: UniversityMLP
    ) -> Tuple[int, np.ndarray]:
        """
        Choose the best supplier for an applicant based on direct access to the university model.

        The function finds the supplier that maximizes the predicted grade for the desired faculty.

        Args:
            applicant_features: The current features of the applicant
            desired_faculty: The faculty index the applicant wants to get into
            trained_model: A trained UniversityMLP model

        Returns:
            Tuple of (chosen_supplier_idx, modified_features)
        """
        best_supplier_idx = -1
        best_modified_features = applicant_features.copy()
        best_score = float('-inf')

        # Get the baseline score without modifications
        trained_model.eval()
        with torch.no_grad():
            baseline_features = torch.FloatTensor(applicant_features).unsqueeze(0)
            baseline_predictions = trained_model(baseline_features)
            # Convert desired_faculty to int to ensure proper indexing
            desired_faculty_idx = int(desired_faculty)
            baseline_score = baseline_predictions[0, desired_faculty_idx].item()
        
        # Only make changes if they improve the score for the desired faculty
        best_score = baseline_score

        # Iterate over suppliers to find the one that maximizes the predicted grade for desired faculty
        for i, supplier in enumerate(self.suppliers):
            # Apply supplier's modifications to features
            modified_features = applicant_features + supplier.diff_vector
            modified_features = np.clip(modified_features, 0, 100)  # Ensure within valid range

            # Get the predicted grades for the modified features
            with torch.no_grad():
                mod_features_tensor = torch.FloatTensor(modified_features).unsqueeze(0)
                predictions = trained_model(mod_features_tensor)
                score_for_desired = predictions[0, desired_faculty_idx].item()
            
            # Check if this supplier improves the score for the desired faculty
            if score_for_desired > best_score:
                best_score = score_for_desired
                best_supplier_idx = i
                best_modified_features = modified_features.copy()
        
        return best_supplier_idx, best_modified_features
    
    
    def recommend(
        self,
        student_features: np.ndarray,
        recommended_faculties: np.ndarray
    ) -> np.ndarray:
        """Calculate final grades for students given their features and recommended faculties
        
        Args:
            student_features: Features matrix of shape (n_students, n_features)
            recommended_faculties: Array of faculty indices of shape (n_students,)
            
        Returns:
            Array of final grades of shape (n_students,)
        """
        # Get utility vectors for all recommended faculties
        faculty_vectors = np.array([self.faculties[f].utility_vector for f in recommended_faculties])
        
        # Calculate base grades using batch matrix multiplication
        base_grades = np.sum(student_features * faculty_vectors, axis=1)
        
        # Generate noise for all students at once
        noise = np.random.uniform(*self.noise_range, size=len(student_features))
        
        return base_grades + noise

    def train_university_model(
        self,
        past_data: pd.DataFrame = None,
        verbose: bool = False
    ) -> UniversityMLP:
        """
        Train university model on past data.
        
        Args:
            past_data: Optional past data to train on. If None, uses self.past_applicants_df
            verbose: Whether to print training progress
            
        Returns:
            Trained UniversityMLP model
        """
        if past_data is None:
            past_data = self.past_applicants_df
        
        if past_data is None:
            raise ValueError("No past data available. Generate past applicants first.")
        
        # Prepare training data
        feature_cols = [f"feature_{i}" for i in range(self.n_features)]
        X_train = torch.FloatTensor(past_data[feature_cols].values)
        
        # Create and train university model
        model = UniversityMLP(self.n_features, self.n_faculties)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Custom loss function that only considers the assigned faculty's grade
        def custom_loss(predictions, targets, assigned_faculties):
            batch_size = predictions.size(0)
            indices = torch.arange(batch_size)
            predicted_assigned_grades = predictions[indices, assigned_faculties]
            return torch.mean((predicted_assigned_grades - targets) ** 2)
        
        # Train the model
        model.train()
        batch_size = 128
        n_epochs = 100
        
        # Keep track of losses for progress reporting
        recent_losses = []
        if verbose:
            print(f"Training university model...")
        
        for epoch in range(n_epochs):
            epoch_losses = []
            # Process in batches
            permutation = torch.randperm(len(X_train))
            for i in range(0, len(X_train), batch_size):
                indices = permutation[i:i + batch_size]
                batch_x = X_train[indices]
                batch_y = torch.FloatTensor(past_data['final_grade'].values[indices])
                batch_assigned = torch.LongTensor(past_data['assigned_faculty'].values[indices])
                
                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = custom_loss(predictions, batch_y, batch_assigned)
                epoch_losses.append(loss.item())
                
                loss.backward()
                optimizer.step()
            
            # Calculate average loss for this epoch
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            
            # Store recent losses
            recent_losses.append(avg_loss)
            if len(recent_losses) > 5:
                recent_losses.pop(0)
            
            # Only print progress occasionally or when loss improves significantly
            if verbose:
                if epoch < 5:
                    print(f"Epoch {epoch}: loss = {avg_loss:.6f}")
                elif epoch % 10 == 0 or (len(recent_losses) >= 5 and avg_loss < sum(recent_losses[:-1])/4):
                    avg_previous = sum(recent_losses[:-1])/len(recent_losses[:-1]) if len(recent_losses) > 1 else float('inf')
                    improvement = (avg_previous - avg_loss) / avg_previous * 100 if avg_previous > 0 else 0
                    print(f"Epoch {epoch}: loss = {avg_loss:.6f} (improved by {improvement:.2f}%)")
        
        if verbose:
            print(f"University model training complete. Final loss: {avg_loss:.6f}")
        return model

    def assign_applicants_to_faculties(
        self,
        model: UniversityMLP,
        current_applicants_features: np.ndarray,
        desired_faculties: np.ndarray = None,
        applicant_ids: np.ndarray = None,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Modified to handle university supplier cases
        
        Args:
            model: Trained UniversityMLP model
            current_applicants_features: Features of current applicants
            desired_faculties: Optional array of desired faculty indices
            applicant_ids: Optional array of applicant IDs
            verbose: Whether to print detailed assignment information
            
        Returns:
            Array of assigned faculty indices
        """
        model.eval()
        with torch.no_grad():
            current_features = torch.FloatTensor(current_applicants_features)
            predicted_grades = model(current_features)
            
            # Initialize assignments with best predicted faculty
            chosen_faculties = torch.argmax(predicted_grades, dim=1).numpy()
            
            # Handle university supplier cases if enabled
            if self.enable_university_supplier and desired_faculties is not None and applicant_ids is not None:
                for i, (pred_grades, desired_faculty, applicant_id) in enumerate(
                    zip(predicted_grades, desired_faculties, applicant_ids)
                ):
                    if applicant_id in self.university_applicants:
                        if verbose:
                            print(f'Applicant {applicant_id} is a university applicant')
                        # If predicted grade for desired faculty is >= 55, assign to desired faculty
                        if pred_grades[desired_faculty] >= 55:
                            if verbose:
                                print(f'Applicant {applicant_id} is assigned to desired faculty {desired_faculty}')
                            chosen_faculties[i] = desired_faculty
        
        return chosen_faculties 