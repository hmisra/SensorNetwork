#!/usr/bin/env python
# Script to generate the SensorAugmentor workflow diagram

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.patheffects as path_effects

# Set the style and figure size
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 8))

# Define colors
colors = {
    'input': '#3498db',  # blue
    'process': '#2ecc71',  # green
    'output': '#e74c3c',  # red
    'background': '#f8f9fa',
    'arrow': '#7f8c8d',  # gray
    'text': '#2c3e50',  # dark blue
    'compare': '#9b59b6',  # purple
    'background_box': '#ecf0f1',  # light gray
    'border': '#bdc3c7'  # medium gray
}

# Create the main axis
ax = plt.gca()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Function to add a box with text
def add_box(x, y, width, height, label, color, text_color=colors['text'], fontsize=10, alpha=1.0):
    rect = Rectangle((x, y), width, height, facecolor=color, edgecolor=colors['border'], 
                     alpha=alpha, linewidth=1.5, zorder=1)
    ax.add_patch(rect)
    
    # Add text with a slight shadow effect for better visibility
    text = ax.text(x + width/2, y + height/2, label, ha='center', va='center', 
                  color=text_color, fontsize=fontsize, fontweight='bold', zorder=2)
    text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    
    return rect

# Function to add an arrow
def add_arrow(start, end, color=colors['arrow'], width=0.5, style='-|>', zorder=0):
    arrow = FancyArrowPatch(start, end, arrowstyle=style, color=color, 
                           connectionstyle='arc3,rad=0.1', linewidth=width, zorder=zorder)
    ax.add_patch(arrow)
    return arrow

# Add title
plt.text(50, 95, 'SensorAugmentor Workflow', ha='center', fontsize=16, 
         fontweight='bold', color=colors['text'])

# Add a large background box
background = add_box(5, 10, 90, 80, '', colors['background_box'], alpha=0.5)

# Add input boxes
lq_sensor = add_box(10, 70, 20, 10, 'Low-Quality\nSensor Data', colors['input'])
hq_sensor = add_box(10, 40, 20, 10, 'High-Quality\nSensor Data', colors['input'])
actuator = add_box(10, 20, 20, 10, 'Actuator\nCommands', colors['input'])

# Add process boxes
encoder = add_box(40, 60, 20, 10, 'Encoder', colors['process'])
latent = add_box(40, 50, 20, 10, 'Latent Space', colors['process'])
decoder = add_box(40, 40, 20, 10, 'Decoder', colors['process'])
act_head = add_box(40, 20, 20, 10, 'Actuator Head', colors['process'])

# Add output boxes
hq_recon = add_box(70, 40, 20, 10, 'Reconstructed\nHQ Signal', colors['output'])
act_pred = add_box(70, 20, 20, 10, 'Predicted\nCommands', colors['output'])

# Add comparison box
loss = add_box(55, 30, 15, 8, 'Loss\nFunction', colors['compare'])

# Add arrows from inputs to process
add_arrow((30, 75), (40, 65), width=1.5)  # LQ sensor to encoder
add_arrow((30, 45), (40, 45), width=1.5)  # HQ sensor to decoder
add_arrow((30, 25), (65, 30), width=1.5)  # Actuator to loss

# Add arrows within process
add_arrow((50, 60), (50, 50), width=1.5)  # Encoder to latent
add_arrow((50, 50), (50, 40), width=1.5)  # Latent to decoder
add_arrow((50, 50), (40, 25), width=1.5)  # Latent to actuator head

# Add arrows to outputs
add_arrow((60, 45), (70, 45), width=1.5)  # Decoder to reconstructed
add_arrow((60, 25), (70, 25), width=1.5)  # Actuator head to predicted

# Add arrows to loss
add_arrow((70, 35), (65, 30), width=1.5)  # Reconstructed to loss
add_arrow((70, 30), (65, 30), width=1.5)  # Predicted to loss

# Add labels for different stages
plt.text(20, 85, 'Input Stage', ha='center', fontsize=12, fontweight='bold', 
        color=colors['text'])
plt.text(50, 85, 'Processing Stage', ha='center', fontsize=12, fontweight='bold', 
        color=colors['text'])
plt.text(80, 85, 'Output Stage', ha='center', fontsize=12, fontweight='bold', 
        color=colors['text'])

# Add explanation notes
plt.text(15, 55, 'Training phase only', ha='center', fontsize=8, color=colors['text'], 
        style='italic')
plt.text(35, 30, 'Shared weights', ha='center', fontsize=8, color=colors['text'], 
        style='italic')
plt.text(55, 35, 'Combined losses', ha='center', fontsize=8, color=colors['text'], 
        style='italic')

# Save the figure
plt.tight_layout()
plt.savefig('docs/images/sensor_augmentor_workflow.png', dpi=300, bbox_inches='tight')
print("Workflow diagram generated and saved to docs/images/sensor_augmentor_workflow.png") 