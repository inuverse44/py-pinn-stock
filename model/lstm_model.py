import torch
from torch import Tensor
from torch.nn import Module, LSTM, Linear

class LSTMModel(Module): # Replace YourModel with your actual model class name
    def __init__(self, input_size: int = 1, hidden_size: int = 32, output_size: int = 1):
        super().__init__()
        self.lstm = LSTM(input_size, hidden_size, batch_first=True)
        # The FC layer should map the hidden_size at each time step to output_size
        self.fc = Linear(hidden_size, output_size) 

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: (batch, time_step, input_size) - Each time_step in input_size is time 't'.
        :return: (batch, time_step, output_size) - Predicted S(t) for each time_step.
        """
        # lstm_out shape: (batch, time_step, hidden_size)
        lstm_out, _ = self.lstm(x) 
        
        # Apply the linear layer to the last dimension (hidden_size)
        # to map it to output_size (which is 1 for S(t)).
        # Reshape for Linear layer: (batch * time_step, hidden_size)
        # Then reshape back to: (batch, time_step, output_size)
        batch_size, time_step, _ = lstm_out.shape
        out: Tensor = self.fc(lstm_out.reshape(-1, lstm_out.size(2))) 
        out = out.reshape(batch_size, time_step, -1) # This will be (batch, time_step, 1)
        
        return out