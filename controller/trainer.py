# controller/trainer.py

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import List
# Ensure this import path is correct based on your file structure
from loss.pinn_loss import physics_loss 

def train_model(
    model: Module,
    train_loader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    epochs: int,
    use_pinn: bool = True,
    lambda_phy: float = 0.1,
    mu: float = 0.001
) -> List[float]:
    """
    Executes the training loop and returns the loss for each epoch.

    Args:
        model (Module): The PyTorch model defined for learning (e.g., an LSTM).
                        Expected to take (batch_size, sequence_length, 1) and output
                        (batch_size, sequence_length, 1) when using PINN loss.
        train_loader (DataLoader): DataLoader for the training data.
        criterion (Module): The loss function for observed data (e.g., MSELoss).
                            Expected to compare (batch_size, sequence_length, 1) outputs
                            with (batch_size, sequence_length, 1) targets.
        optimizer (Optimizer): The optimizer for updating model parameters (e.g., Adam).
        epochs (int): Number of training epochs.
        use_pinn (bool, optional): Whether to add the PINN physics term to the loss.
                                   Defaults to True.
        lambda_phy (float, optional): Weight for the PINN term. Defaults to 0.1.
        mu (float, optional): Parameter mu for the mean-reverting GBM model.
                              Defaults to 0.001.

    Returns:
        List[float]: A list of average losses for each epoch.
    """
    losses: List[float] = []
    
    for epoch in range(epochs):
        model.train() # Set the model to training mode
        epoch_loss: float = 0.0
        
        for X_batch, y_batch in train_loader:
            # Preprocess X_batch: Add a feature dimension if not present.
            # Assuming X_batch from DataLoader is (batch_size, sequence_length)
            # Make it (batch_size, sequence_length, 1) for the LSTM input
            if X_batch.dim() == 2:
                X_batch = X_batch.unsqueeze(-1)
            
            # Ensure y_batch matches the model's output shape for criterion
            # Assuming y_batch from DataLoader is (batch_size, sequence_length)
            # Make it (batch_size, sequence_length, 1) for comparison
            if y_batch.dim() == 2:
                y_batch = y_batch.unsqueeze(-1)
            
            optimizer.zero_grad() # Clear gradients from previous step

            # Forward pass: Get model predictions
            output: Tensor = model(X_batch) # output shape is now (batch, seq_len, 1)

            # Calculate data-driven loss
            loss_data: Tensor = criterion(output, y_batch)
            
            # Calculate total loss, potentially including PINN term
            if use_pinn:
                # physics_loss expects model output to be differentiable with respect to X_batch
                # which means model should output (batch, seq_len, 1) given (batch, seq_len, 1) input
                loss_phy: Tensor = physics_loss(model, X_batch, mu=mu)
                loss: Tensor = loss_data + lambda_phy * loss_phy
            else:
                loss: Tensor = loss_data
            
            # Backpropagation
            loss.backward()
            optimizer.step() # Update model parameters

            epoch_loss += loss.item() # Accumulate loss for the epoch

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")

    return losses