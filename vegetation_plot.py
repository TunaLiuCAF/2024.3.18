#!/usr/bin/env python3
"""
Vegetation Classification Plotting Script

This script creates plots with specific color mapping for vegetation classes
according to the legend requirements.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
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

def create_vegetation_plot(gdf, class_column='vegetation_class', title='Vegetation Classification'):
    """
    Create a plot with vegetation classes using the specified color scheme.
    
    Parameters:
    gdf (GeoDataFrame): GeoDataFrame containing vegetation data
    class_column (str): Column name containing vegetation class codes
    title (str): Title for the plot
    
    Returns:
    matplotlib figure and axis objects
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Plot each vegetation class with its specific color
    for class_code, color in class_colors.items():
        # Filter data for this class
        class_data = gdf[gdf[class_column] == class_code]
        
        if not class_data.empty:
            class_data.plot(ax=ax, 
                          color=color, 
                          label=class_name_map[class_code],
                          alpha=0.8,
                          edgecolor='black',
                          linewidth=0.5)
    
    # Create custom legend
    legend_elements = []
    for class_code, color in class_colors.items():
        legend_elements.append(Patch(facecolor=color, 
                                   label=class_name_map[class_code]))
    
    # Add legend
    ax.legend(handles=legend_elements, 
              loc='center left', 
              bbox_to_anchor=(1, 0.5),
              fontsize=10,
              frameon=True,
              fancybox=True,
              shadow=True)
    
    # Set title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Remove axes ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    return fig, ax

def update_existing_plot(watersheds_gdf, stations_gdf=None, vegetation_gdf=None):
    """
    Update the existing watershed and station plot to include vegetation classes.
    
    Parameters:
    watersheds_gdf (GeoDataFrame): Watershed boundaries
    stations_gdf (GeoDataFrame): Station locations (optional)
    vegetation_gdf (GeoDataFrame): Vegetation classification data (optional)
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Plot watersheds as background
    if watersheds_gdf is not None:
        watersheds_gdf.plot(ax=ax, 
                           color='white', 
                           edgecolor='black', 
                           alpha=0.75, 
                           linewidth=1)
    
    # Plot vegetation classes if available
    if vegetation_gdf is not None and 'vegetation_class' in vegetation_gdf.columns:
        for class_code, color in class_colors.items():
            class_data = vegetation_gdf[vegetation_gdf['vegetation_class'] == class_code]
            if not class_data.empty:
                class_data.plot(ax=ax, 
                              color=color, 
                              alpha=0.8,
                              edgecolor='none')
    
    # Plot stations if available
    if stations_gdf is not None:
        if 'counts' in stations_gdf.columns:
            # Use existing counts column for coloring
            stations_gdf.plot(ax=ax, 
                            column='counts', 
                            cmap='Reds', 
                            legend=True, 
                            markersize=35,
                            alpha=0.8)
        else:
            # Plot with uniform color
            stations_gdf.plot(ax=ax, 
                            color='blue', 
                            markersize=15, 
                            alpha=0.8,
                            edgecolor='white',
                            linewidth=1)
    
    # Create legend for vegetation classes
    if vegetation_gdf is not None:
        legend_elements = []
        for class_code, color in class_colors.items():
            if class_code in vegetation_gdf['vegetation_class'].values:
                legend_elements.append(Patch(facecolor=color, 
                                           label=class_name_map[class_code]))
        
        if legend_elements:
            ax.legend(handles=legend_elements, 
                      loc='center left', 
                      bbox_to_anchor=(1, 0.5),
                      fontsize=10,
                      title='Vegetation Classes',
                      title_fontsize=12,
                      frameon=True,
                      fancybox=True,
                      shadow=True)
    
    # Set title and labels
    ax.set_title('Watersheds, Stations and Vegetation Classification', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Remove axes ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax

# Example usage and testing
if __name__ == "__main__":
    print("Vegetation Classification Plotting Script")
    print("=" * 50)
    print("Available vegetation classes:")
    for code, name in class_name_map.items():
        print(f"  {code}: {name}")
        print(f"      Color: {class_colors[code]}")
    
    print("\nColor mapping implemented according to legend requirements:")
    print("- Natural Conifer Forest (NCF): Dark green")
    print("- Evergreen Broadleaved Forest (EBF): Medium green") 
    print("- Deciduous Broadleaf Forest (DBF): Light green")
    print("- Planted Conifer Forest (PCF): Teal")
    print("- Mixed Forest (MF): Pale green")
    print("- Alpine Meadow (AM): Light yellow")
    print("- Degraded Planted Forest (DPF): Purple (placeholder)")
    print("- Alpine Shrub (AS): Yellow")
    print("- Dry Valley Shrub and Grass (DVSG): Orange")
    print("- SubAlpine Deciduous Shrub and Grass (SDSG): Peach")