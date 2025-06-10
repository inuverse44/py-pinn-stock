# loss/pinn_loss.py

import torch
from torch import Tensor
from torch.nn import Module

def physics_loss(model: Module, X_batch: Tensor, mu: float = 0.001) -> Tensor:
    """
    Calculates the physics-informed neural network (PINN) loss term
    based on the deterministic Geometric Brownian Motion (GBM) equation: dS/dt = mu * S.

    This function prepares the input for the model, computes the predicted stock price (S_pred),
    calculates the time derivative of S_pred (dS_dt) using automatic differentiation,
    and then computes the mean squared error of the residual (dS_dt - mu * S_pred).

    Args:
        model (Module): The PyTorch neural network model that predicts S(t).
                        It is expected to take a tensor of shape (batch_size, sequence_length, 1)
                        (representing time 't' for each step in a sequence)
                        and output a tensor of shape (batch_size, sequence_length, 1)
                        (representing S(t) for each 't').
        X_batch (Tensor): The input batch of time values.
                          Expected initial shape: (batch_size, sequence_length, 1).
                          This function ensures it is correctly prepared for `torch.autograd.grad`.
        mu (float): The drift coefficient in the GBM equation. Defaults to 0.001.

    Returns:
        Tensor: The mean squared error of the physics-informed residual.
                This loss term enforces the physical constraint.

    Raises:
        RuntimeError: If the shape of S_pred after the model call is not as expected
                      for the derivative calculation.
    """
    # Ensure X_batch requires gradients for differentiation.
    # We clone and detach to avoid modifying the original tensor if it's used elsewhere,
    # and then set requires_grad_ to True for differentiation.
    # X_batch is expected to be (batch_size, sequence_length, 1) here.
    X_batch = X_batch.detach().clone().requires_grad_(True)
    
    # Ensure data type is float, as models typically operate on float tensors.
    # This also helps if the input data was, e.g., long type.
    if X_batch.dtype != torch.float32: # Adjust to torch.float64 if your model uses double precision
        X_batch = X_batch.to(torch.float32)

    # Debug prints to confirm final input shape and properties
    print(f"physics_loss: X_batch final shape for model input: {X_batch.shape}")
    print(f"physics_loss: X_batch dtype: {X_batch.dtype}")
    print(f"physics_loss: X_batch device: {X_batch.device}")
    
    # Pass the preprocessed time input to the neural network model.
    # S_pred is expected to represent S(t) for each time step in the sequence.
    # With the recommended model change, S_pred will be (batch_size, sequence_length, 1).
    S_pred: Tensor = model(X_batch)
    
    # Debug print for model output shape
    print(f"physics_loss: S_pred shape after model call: {S_pred.shape}")

    # Validate S_pred shape for derivative calculation.
    # It must be (batch_size, sequence_length, 1) to match X_batch for element-wise gradient.
    if S_pred.dim() != 3 or S_pred.shape[-1] != 1:
         raise RuntimeError(
             f"physics_loss: Model output S_pred has an unexpected shape: {S_pred.shape}. "
             "Expected (batch_size, sequence_length, 1) for dS/dt calculation based on PINN setup."
         )

    print("physics_loss: torch.autograd started for dS/dt calculation...")
    # Calculate dS/dt using automatic differentiation.
    # We compute the gradient of S_pred with respect to X_batch (time).
    # Since S_pred is a tensor of shape (batch, seq_len, 1) and X_batch is also
    # (batch, seq_len, 1), the gradient dS_dt will have the same shape.
    dS_dt: Tensor = torch.autograd.grad(
        outputs=S_pred,
        inputs=X_batch,
        grad_outputs=torch.ones_like(S_pred), # Must match outputs shape
        create_graph=True,  # Needed to compute higher-order gradients if necessary
        retain_graph=True,  # Needed if you plan to call .grad() multiple times on the same graph
        only_inputs=True    # Only return gradients for inputs
    )[0] # torch.autograd.grad returns a tuple, we take the first element (gradient of X_batch)
    print("physics_loss: torch.autograd finished for dS/dt.")
    print(f"physics_loss: dS_dt shape: {dS_dt.shape}")
    print(f"physics_loss: dS_dt dtype: {dS_dt.dtype}")

    # Calculate the residual of the GBM equation: (dS/dt - mu * S)
    # The equation is enforced across all time steps in the sequence for each batch.
    residual: Tensor = dS_dt - mu * S_pred
    
    # Compute the mean squared error of the residual.
    # This is the physics-informed loss term.
    loss_phy = torch.mean(residual ** 2)
    print("physics_loss: Physics loss calculation is done.")
    return loss_phy