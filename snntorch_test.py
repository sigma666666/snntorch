import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import matplotlib.pyplot as plt  # For visualization

# Hyperparameters
num_steps = 25  # Number of time steps to simulate
batch_size = 1  # Processing one sample at a time
beta = 0.5  # Neuron decay rate - controls the memory of the neuron
spike_grad = surrogate.fast_sigmoid()  # Surrogate gradient for backpropagation

# Create the network architecture
print("Creating neural network...")
net = nn.Sequential(
    nn.Conv2d(1, 8, 5),  # First convolutional layer: 1 input channel, 8 output channels, 5x5 kernel
    nn.MaxPool2d(2),     # Reduces spatial dimensions by half
    snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),  # Leaky integrate-and-fire neuron
    nn.Conv2d(8, 16, 5), # Second convolutional layer: 8 input channels, 16 output channels
    nn.MaxPool2d(2),     # Second pooling layer
    snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),  # Second leaky neuron layer
    nn.Flatten(),        # Flatten for fully connected layer
    nn.Linear(16 * 4 * 4, 10),  # Fully connected layer mapping to 10 output classes
    snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True)  # Output neuron layer
)
print("Network created successfully!")

# Generate random input data
print("\nGenerating input data...")
data_in = torch.rand(num_steps, batch_size, 1, 28, 28)  # Random input data simulating MNIST dimensions
print(f"Input data shape: {data_in.shape}")

# Initialize lists to store results
spike_recording = []  # Record spikes over time
membrane_recording = []  # Record membrane potentials

# Reset network states
print("\nResetting network states...")
utils.reset(net)
print("Network states reset successfully!")

# Run simulation
print("\nRunning simulation...")
with torch.no_grad():  # Disable gradient computation for inference
    for step in range(num_steps):
        spike, state = net(data_in[step])
        spike_recording.append(spike)
        membrane_recording.append(state)
print("Simulation completed!")

# Convert recordings to tensors for easier analysis
spike_recording = torch.stack(spike_recording)
membrane_recording = torch.stack(membrane_recording)

# Print shape information
print("\nOutput shapes:")
print(f"Spike recording shape: {spike_recording.shape}")
print(f"Membrane potential recording shape: {membrane_recording.shape}")

# Visualize results
print("\nCreating visualizations...")
plt.figure(figsize=(15, 5))

# Plot spike activity
plt.subplot(121)
plt.imshow(spike_recording.squeeze().cpu().numpy(), cmap='hot', aspect='auto')
plt.colorbar(label='Spike Activity')
plt.xlabel('Neuron Index')
plt.ylabel('Time Step')
plt.title('Spike Activity Over Time')

# Plot membrane potentials
plt.subplot(122)
plt.imshow(membrane_recording.squeeze().cpu().numpy(), cmap='viridis', aspect='auto')
plt.colorbar(label='Membrane Potential')
plt.xlabel('Neuron Index')
plt.ylabel('Time Step')
plt.title('Membrane Potentials Over Time')

plt.tight_layout()
plt.show()

print("\nInstallation verification complete!")
print("If you can see the network creation messages, simulation results, and plots, snntorch is working correctly!")

# Print some statistics
print("\nSummary Statistics:")
print(f"Average spike rate: {spike_recording.mean().item():.4f}")
print(f"Maximum membrane potential: {membrane_recording.max().item():.4f}")
print(f"Minimum membrane potential: {membrane_recording.min().item():.4f}")