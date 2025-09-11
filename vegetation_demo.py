#!/usr/bin/env python3
"""
Demonstration script for vegetation classification plotting
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

# Import our vegetation plotting module
from vegetation_plot import class_name_map, class_colors

def create_color_legend_demo():
    """
    Create a demonstration plot showing the color mapping for vegetation classes
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Remove axes for cleaner look
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(class_colors))
    ax.axis('off')
    
    # Create color swatches for each vegetation class
    y_pos = 0
    for class_code, color in class_colors.items():
        # Draw color rectangle
        rect = mpatches.Rectangle((1, y_pos), 2, 0.8, 
                                facecolor=color, 
                                edgecolor='black', 
                                linewidth=1)
        ax.add_patch(rect)
        
        # Add class code
        ax.text(0.5, y_pos + 0.4, class_code, 
               fontsize=12, fontweight='bold', 
               verticalalignment='center')
        
        # Add full class name
        ax.text(3.5, y_pos + 0.4, class_name_map[class_code], 
               fontsize=11, verticalalignment='center')
        
        # Add color hex code
        ax.text(8, y_pos + 0.4, color, 
               fontsize=10, verticalalignment='center',
               fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray'))
        
        y_pos += 1
    
    # Add title
    ax.text(5, len(class_colors) + 0.5, 
           'Vegetation Classification Color Mapping', 
           fontsize=16, fontweight='bold', 
           horizontalalignment='center')
    
    # Add column headers
    ax.text(0.5, -0.5, 'Code', fontsize=12, fontweight='bold')
    ax.text(2, -0.5, 'Color', fontsize=12, fontweight='bold')
    ax.text(3.5, -0.5, 'Class Name', fontsize=12, fontweight='bold')
    ax.text(8, -0.5, 'Hex Value', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax

def create_sample_vegetation_plot():
    """
    Create a sample plot demonstrating how the colors would look in a map
    """
    
    # Create sample data points for demonstration
    np.random.seed(42)  # For reproducible results
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate random points for each vegetation class
    for i, (class_code, color) in enumerate(class_colors.items()):
        # Generate some random points
        n_points = np.random.randint(5, 15)  # Random number of points per class
        x = np.random.uniform(i, i+1, n_points)  # Spread horizontally
        y = np.random.uniform(0, 10, n_points)   # Random vertical distribution
        
        # Plot points with class color
        ax.scatter(x, y, c=color, s=100, 
                  label=class_name_map[class_code],
                  alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
             fontsize=10, title='Vegetation Classes',
             title_fontsize=12)
    
    ax.set_title('Sample Vegetation Distribution Map', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Geographic Area (example)', fontsize=12)
    ax.set_ylabel('Elevation/Location (example)', fontsize=12)
    
    # Remove some clutter
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

if __name__ == "__main__":
    print("Creating vegetation classification color demonstration...")
    
    # Create and show the color legend
    fig1, ax1 = create_color_legend_demo()
    plt.figure(fig1)
    plt.savefig('/home/runner/work/2024.3.18/2024.3.18/vegetation_color_legend.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Color legend saved as 'vegetation_color_legend.png'")
    
    # Create and show sample vegetation plot
    fig2, ax2 = create_sample_vegetation_plot()
    plt.figure(fig2)
    plt.savefig('/home/runner/work/2024.3.18/2024.3.18/sample_vegetation_map.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Sample map saved as 'sample_vegetation_map.png'")
    
    print("\nColor mapping summary:")
    print("=" * 50)
    for code, name in class_name_map.items():
        color_desc = {
            "NCF": "Dark green",
            "EBF": "Medium green", 
            "DBF": "Light green",
            "PCF": "Teal",
            "MF": "Pale green",
            "AM": "Light yellow",
            "DPF": "Purple (placeholder)",
            "AS": "Yellow",
            "DVSG": "Orange",
            "SDSG": "Peach"
        }
        print(f"{code}: {name} -> {color_desc[code]} ({class_colors[code]})")