from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class PyNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float, random_seed: int | None = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Fully connected layers with Xavier initalization
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.fc1(x))
        
        logits = self.fc2(hidden)

        return F.softmax(logits, dim=-1)
        
    

    def reinforce_update(self, state: torch.Tensor, action: int, advantage: float) -> torch.Tensor:
        
        probs = self.forward(state)

        # Log-probability of the action that was actually taken
        log_prob = torch.log(probs[action] + 1e-10)
        loss = - log_prob * advantage

        self.optimizer.zero_grad() 
        loss.backward() 
        self.optimizer.step()

        return probs.detach() # Detaching to avoid tracking gradients in the next forward pass