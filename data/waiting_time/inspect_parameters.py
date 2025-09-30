import torch
import torch.nn as nn
import json

# Define the RNN classes that are needed to load the models
class RNN(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.num_layers = parameter['num_layers']
        self.hidden_size = parameter['hidden_size']
        self.num_stacked_layers = self.num_layers
        self.bidirectional = 2 if parameter['bidirectional'] else 1
        self.rnn = nn.GRU(parameter['input_size'], self.hidden_size, self.num_layers, batch_first=True, bidirectional=parameter['bidirectional'], dropout=parameter['dropout'])
        self.fc = nn.Linear(self.bidirectional * self.hidden_size, parameter['output_size'])

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.bidirectional * self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class RNN2(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.num_layers = parameter['num_layers']
        self.hidden_size = parameter['hidden_size']
        self.num_stacked_layers = self.num_layers
        self.bidirectional = 2 if parameter['bidirectional'] else 1
        self.rnn = nn.GRU(parameter['input_size'], self.hidden_size, self.num_layers, batch_first=True, bidirectional=parameter['bidirectional'], dropout=parameter['dropout'])
        self.fc1 = nn.Linear(self.bidirectional * self.hidden_size, 1)
        self.fc2 = nn.Linear(2, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.bidirectional * self.num_layers, batch_size, self.hidden_size).to(x.device)
        # Note: This forward method references model1 which won't be available in this context
        # We'll just focus on the parameter structure
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc1(out)
        return out

# Load the model parameter files
model_path = "prediction_model/"

# Load food model parameters with weights_only=False to allow custom classes
food_para1 = torch.load(model_path + 'food/' + 'model1.pt', map_location=torch.device('cpu'), weights_only=False)
food_para2 = torch.load(model_path + 'food/' + 'newModel2.pt', map_location=torch.device('cpu'), weights_only=False)

# Load site model parameters
site_para1 = torch.load(model_path + 'sites/' + 'model1.pt', map_location=torch.device('cpu'), weights_only=False)
site_para2 = torch.load(model_path + 'sites/' + 'newModel2.pt', map_location=torch.device('cpu'), weights_only=False)

print("=== FOOD MODEL PARAMETERS ===")
print("\nFood Model 1 (model1.pt):")
for key, value in food_para1.items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for subkey, subvalue in value.items():
            if isinstance(subvalue, torch.Tensor):
                print(f"    {subkey}: tensor with shape {subvalue.shape}")
            else:
                print(f"    {subkey}: {subvalue}")
    elif isinstance(value, torch.Tensor):
        print(f"  {key}: tensor with shape {value.shape}")
    else:
        print(f"  {key}: {value}")

print("\nFood Model 2 (newModel2.pt):")
for key, value in food_para2.items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for subkey, subvalue in value.items():
            if isinstance(subvalue, torch.Tensor):
                print(f"    {subkey}: tensor with shape {subvalue.shape}")
            else:
                print(f"    {subkey}: {subvalue}")
    elif isinstance(value, torch.Tensor):
        print(f"  {key}: tensor with shape {value.shape}")
    else:
        print(f"  {key}: {value}")

print("\n=== SITE MODEL PARAMETERS ===")
print("\nSite Model 1 (model1.pt):")
for key, value in site_para1.items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for subkey, subvalue in value.items():
            if isinstance(subvalue, torch.Tensor):
                print(f"    {subkey}: tensor with shape {subvalue.shape}")
            else:
                print(f"    {subkey}: {subvalue}")
    elif isinstance(value, torch.Tensor):
        print(f"  {key}: tensor with shape {value.shape}")
    else:
        print(f"  {key}: {value}")

print("\nSite Model 2 (newModel2.pt):")
for key, value in site_para2.items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for subkey, subvalue in value.items():
            if isinstance(subvalue, torch.Tensor):
                print(f"    {subkey}: tensor with shape {subvalue.shape}")
            else:
                print(f"    {subkey}: {subvalue}")
    elif isinstance(value, torch.Tensor):
        print(f"  {key}: tensor with shape {value.shape}")
    else:
        print(f"  {key}: {value}")

# Check if there are model configuration parameters
print("\n=== MODEL CONFIGURATION PARAMETERS ===")
print("Looking for configuration parameters that define model architecture...")

# Check for common parameter names
config_keys = ['num_layers', 'hidden_size', 'input_size', 'output_size', 'bidirectional', 'dropout', 'batch_size']
for model_name, model_data in [("Food Model 1", food_para1), ("Food Model 2", food_para2), 
                               ("Site Model 1", site_para1), ("Site Model 2", site_para2)]:
    print(f"\n{model_name}:")
    for key in config_keys:
        if key in model_data:
            print(f"  {key}: {model_data[key]}")
        else:
            # Check if it's in a sub-dictionary
            for subkey, subvalue in model_data.items():
                if isinstance(subvalue, dict) and key in subvalue:
                    print(f"  {key}: {subvalue[key]} (found in {subkey})")
