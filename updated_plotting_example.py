#!/usr/bin/env python3
"""
Updated plotting example showing integration with existing GeoPythonRegionCheck workflow
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

# Define the class name mapping dictionary with vegetation classes
class_name_map = {
    "NCF": "Natural Conifer Forest (NCF)",
    "EBF": "Evergreen Broadleaved Forest (EBF)", 
    "DBF": "Deciduous Broadleaf Forest (DBF)",
    "PCF": "Planted Conifer Forest (PCF)",
    "MF": "Mixed Forest (MF)",
    "AM": "Alpine Meadow (AM)",
    "DPF": "Degraded Planted Forest (DPF)",
    "AS": "Alpine Shrub (AS)",
    "DVSG": "Dry Valley Shrub and Grass (DVSG)",
    "SDSG": "SubAlpine Deciduous Shrub and Grass (SDSG)"
}

# Define specific colors for each vegetation class according to the legend
class_colors = {
    "NCF": "#006400",    # Dark green for Natural Conifer Forest
    "EBF": "#228B22",    # Medium green for Evergreen Broadleaved Forest  
    "DBF": "#90EE90",    # Light green for Deciduous Broadleaf Forest
    "PCF": "#008080",    # Teal for Planted Conifer Forest
    "MF": "#98FB98",     # Pale green for Mixed Forest
    "AM": "#FFFFE0",     # Light yellow for Alpine Meadow
    "DPF": "#800080",    # Purple for Degraded Planted Forest (placeholder)
    "AS": "#FFFF00",     # Yellow for Alpine Shrub
    "DVSG": "#FFA500",   # Orange for Dry Valley Shrub and Grass
    "SDSG": "#FFDAB9"    # Peach for SubAlpine Deciduous Shrub and Grass
}

def plot_watersheds_stations_with_vegetation_legend(watersheds=None, stations=None):
    """
    Enhanced plotting function that matches the GeoPythonRegionCheck workflow
    but includes the vegetation classification legend.
    
    This function can be used as a drop-in replacement for the existing plotting code.
    """
    
    # Create figure with larger size to accommodate legend
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Plot watersheds if available
    if watersheds is not None:
        watersheds.plot(ax=ax, color='white', edgecolor='black', alpha=1, linewidth=1)
        print("Plotted watersheds")
    
    # Plot stations if available
    if stations is not None:
        if 'counts' in stations.columns:
            # Use existing counts column for coloring (preserving original functionality)
            stations.plot(ax=ax, column='counts', cmap='Reds', legend=True, 
                         markersize=35, alpha=0.8)
            print("Plotted stations with counts-based coloring")
        else:
            # Plot with uniform color if no counts column
            stations.plot(ax=ax, color='blue', markersize=35, alpha=0.8)
            print("Plotted stations with uniform coloring")
    
    # Create vegetation class legend (even if no vegetation data is present yet)
    legend_elements = []
    for class_code, color in class_colors.items():
        legend_elements.append(Patch(facecolor=color, 
                                   edgecolor='black',
                                   linewidth=0.5,
                                   label=class_name_map[class_code]))
    
    # Add vegetation class legend
    vegetation_legend = ax.legend(handles=legend_elements, 
                                loc='center left', 
                                bbox_to_anchor=(1.02, 0.5),
                                fontsize=11,
                                title='Vegetation Classes\n(Available for Future Use)',
                                title_fontsize=13,
                                frameon=True,
                                fancybox=True,
                                shadow=True,
                                framealpha=0.9)
    
    # Customize legend appearance
    vegetation_legend.get_frame().set_facecolor('white')
    vegetation_legend.get_frame().set_edgecolor('gray')
    
    # Set enhanced title and labels
    ax.set_title('Watersheds, Stations and Vegetation Classification Legend', 
                fontsize=18, fontweight='bold', pad=25)
    
    # Add subtle labels if geographic extent is available
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    
    # Remove tick labels for cleaner appearance (common in GIS plots)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add grid for better reference
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    return fig, ax

def demonstrate_with_sample_data():
    """
    Create a demonstration using sample data that mimics the structure
    of the GeoPythonRegionCheck workflow.
    """
    
    print("Creating demonstration with sample data...")
    print("=" * 50)
    
    # This would normally be your actual data loading:
    # watersheds = gpd.read_file('path_to_watersheds.shp') 
    # stations = gpd.read_file('path_to_stations.shp')
    
    # For demonstration, we'll create mock data
    # In reality, you would use your actual GeoPandas DataFrames
    
    # Call the enhanced plotting function
    fig, ax = plot_watersheds_stations_with_vegetation_legend()
    
    # Add informational text to the plot
    ax.text(0.02, 0.98, 
           "Updated Plotting Function\n" +
           "• Preserves existing watershed/station plotting\n" +
           "• Adds vegetation class color legend\n" +
           "• Ready for vegetation data integration\n" +
           "• Follows original GeoPythonRegionCheck style",
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor='lightblue', 
                    alpha=0.8))
    
    # Save the plot
    plt.savefig('/home/runner/work/2024.3.18/2024.3.18/enhanced_watershed_plot.png', 
                dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax

def print_integration_instructions():
    """
    Print instructions for integrating the updated code into the notebook.
    """
    
    print("\n" + "="*60)
    print("INTEGRATION INSTRUCTIONS FOR GeoPythonRegionCheck.ipynb")
    print("="*60)
    
    print("""
To update your existing GeoPythonRegionCheck.ipynb:

1. REPLACE the existing plotting cell that contains:
   ```
   station_Relative_Error_010.plot(ax=ax, column='counts', cmap='Reds', legend=True, markersize=35)
   ```
   
2. WITH the enhanced version that includes:
   • The class_name_map dictionary
   • The class_colors dictionary  
   • Enhanced legend with vegetation classes
   • Improved plot styling
   
3. The updated code maintains backward compatibility:
   • Still plots watersheds and stations as before
   • Still uses the 'counts' column for station coloring
   • Adds vegetation class legend for future use
   
4. When you have actual vegetation data:
   • Add a 'vegetation_class' column to your GeoDataFrame
   • Use the class codes: NCF, EBF, DBF, PCF, MF, AM, DPF, AS, DVSG, SDSG
   • The colors will automatically match the legend

5. Color specifications match requirements:
   • Natural Conifer Forest (NCF): Dark green (#006400)
   • Evergreen Broadleaved Forest (EBF): Medium green (#228B22)
   • Deciduous Broadleaf Forest (DBF): Light green (#90EE90)
   • Planted Conifer Forest (PCF): Teal (#008080)
   • Mixed Forest (MF): Pale green (#98FB98)
   • Alpine Meadow (AM): Light yellow (#FFFFE0)
   • Degraded Planted Forest (DPF): Purple (#800080) [placeholder]
   • Alpine Shrub (AS): Yellow (#FFFF00)
   • Dry Valley Shrub and Grass (DVSG): Orange (#FFA500)
   • SubAlpine Deciduous Shrub and Grass (SDSG): Peach (#FFDAB9)
""")

if __name__ == "__main__":
    print("Enhanced GeoPythonRegionCheck Integration Example")
    print("=" * 55)
    
    # Demonstrate the enhanced plotting
    fig, ax = demonstrate_with_sample_data()
    
    # Print integration instructions
    print_integration_instructions()
    
    print("\nFiles created:")
    print("• enhanced_watershed_plot.png - Demonstration plot")
    print("• vegetation_plot.py - Reusable plotting functions") 
    print("• vegetation_demo.py - Color demonstrations")
    print("• This file - Integration example")
    
    print(f"\nScript completed successfully!")
    print("Enhanced plotting ready for integration into GeoPythonRegionCheck.ipynb")