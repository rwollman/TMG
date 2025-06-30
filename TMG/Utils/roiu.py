"""
ROI Selection utilities for TMG.

This module provides interactive tools for selecting regions of interest (ROIs)
in spatial data visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.path import Path


class RoiSelector:
    """
    An optimized class to plot a sample of points and draw multiple polygonal
    ROIs to classify the full dataset.
    
    This interactive tool allows users to:
    - View a spatial scatter plot of points with colors
    - Draw polygonal regions of interest by clicking
    - Automatically downsample large datasets for performance
    - Generate classification assignments for all points
    
    Parameters
    ----------
    x_points : array-like
        X coordinates of all points
    y_points : array-like
        Y coordinates of all points
    c_values : array-like
        Color values for plotting points (e.g., cell types)
    display_threshold : int, default 1000
        Maximum number of points to display for performance.
        If dataset is larger, it will be randomly downsampled for visualization.
        
    Attributes
    ----------
    points : ndarray
        Combined XY coordinates of all points
    completed_polygons : list
        List of completed polygon vertices
    final_output : ndarray or None
        Final classification vector (0=background, 1,2,3...=polygon assignments)
        
    Examples
    --------
    >>> # Basic usage with TMG data
    >>> XY = TMG.Layers[0].get_XY()
    >>> Type = TMG.Layers[0].Type
    >>> roi_selector = RoiSelector(XY[:,0], XY[:,1], Type, display_threshold=10000)
    >>> # Interactive drawing occurs...
    >>> # After finishing, get the results:
    >>> section_assignments = roi_selector.final_output
    """
    
    def __init__(self, x_points, y_points, c_values, display_threshold=1000):
        self.points = np.c_[x_points, y_points]
        self.c_values = c_values
        self.plot_indices = None  # Will store the indices of points we plot
        self.completed_polygons = []
        self.current_poly_vertices = []
        self.final_output = None

        # Cycle through colors for each new polygon
        self.color_cycle = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        self.current_color_index = 0

        self.fig, self.ax = plt.subplots(figsize=(10, 20))
        
        # --- Optimization: Downsample points for plotting if necessary ---
        num_points = len(self.points)
        if num_points > display_threshold:
            print(f"Dataset has {num_points} points. Displaying a random sample of {display_threshold} for performance.")
            self.plot_indices = np.random.choice(num_points, display_threshold, replace=False)
            x_to_plot = self.points[self.plot_indices, 0]
            y_to_plot = self.points[self.plot_indices, 1]
            c_to_plot = self.c_values[self.plot_indices]
        else:
            print(f"Displaying all {num_points} points.")
            self.plot_indices = np.arange(num_points)  # All indices
            x_to_plot = self.points[:, 0]
            y_to_plot = self.points[:, 1]
            c_to_plot = self.c_values

        self.scatter_plot = self.ax.scatter(x_to_plot, y_to_plot, c=c_to_plot, cmap='tab20', zorder=1, s=3)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # --- Add Buttons ---
        ax_button_next = self.fig.add_axes([0.7, 0.01, 0.15, 0.06])
        self.btn_next = Button(ax_button_next, 'Next Polygon')
        self.btn_next.on_clicked(self.next_polygon)

        ax_button_finish = self.fig.add_axes([0.5, 0.01, 0.15, 0.06])
        self.btn_finish = Button(ax_button_finish, 'Finish & Analyze')
        self.btn_finish.on_clicked(self.finish_analysis)
        
        print("Click to start drawing. Click 'Next Polygon' to complete the current one and start a new one.")
        print("Click 'Finish & Analyze' when you are done drawing all polygons.")

    def on_click(self, event):
        """Handles mouse clicks to draw the polygon."""
        if event.inaxes != self.ax: 
            return
        if event.button == 1:
            color = self.color_cycle[self.current_color_index % len(self.color_cycle)]
            self.current_poly_vertices.append((event.xdata, event.ydata))
            if len(self.current_poly_vertices) > 1:
                self.ax.plot([self.current_poly_vertices[-2][0], self.current_poly_vertices[-1][0]],
                             [self.current_poly_vertices[-2][1], self.current_poly_vertices[-1][1]],
                             color=color, linestyle='-', marker='o', markersize=3, zorder=2)
            else:
                self.ax.plot(event.xdata, event.ydata, color=color, marker='o', markersize=3, zorder=2)
            self.fig.canvas.draw()

    def next_polygon(self, event):
        """Finalizes the current polygon and prepares for the next one."""
        if len(self.current_poly_vertices) < 3:
            print("A polygon must have at least 3 vertices. Current polygon cancelled.")
            return

        color = self.color_cycle[self.current_color_index % len(self.color_cycle)]
        self.ax.plot([self.current_poly_vertices[-1][0], self.current_poly_vertices[0][0]],
                     [self.current_poly_vertices[-1][1], self.current_poly_vertices[0][1]],
                     color=color, linestyle='-', marker='o', markersize=3, zorder=2)
        self.fig.canvas.draw()
        
        self.completed_polygons.append(self.current_poly_vertices)
        self.current_poly_vertices = []
        self.current_color_index += 1
        print(f"Polygon {self.current_color_index} saved. Ready to draw polygon {self.current_color_index + 1}.")
        
    def get_polygon_indices(self):
        """
        Calculates which polygon each point from the *entire dataset* falls into.
        
        Returns
        -------
        ndarray
            Classification vector where 0=background, 1,2,3...=polygon assignments
        """
        if len(self.current_poly_vertices) >= 3 and self.current_poly_vertices not in self.completed_polygons:
            print("Note: The polygon being drawn was automatically finalized.")
            self.completed_polygons.append(self.current_poly_vertices)

        L = np.zeros(len(self.points), dtype=int)
        for i, poly_verts in enumerate(self.completed_polygons):
            poly_path = Path(poly_verts)
            # Calculation is performed on ALL points
            is_inside = poly_path.contains_points(self.points)
            L[(is_inside) & (L == 0)] = i + 1
        
        self.final_output = L
        return L

    def finish_analysis(self, event):
        """Calculates and prints the final output vector for all points."""
        output_vector_full = self.get_polygon_indices()  # Analysis on all points
        
        print("\n--- Analysis Complete ---")
        print(f"Output vector L has been calculated for all {len(self.points)} points.")
        print(f"Found {len(self.completed_polygons)} polygons.")
        
        # To see the full vector, you would print it here (can be long)
        # print("Output vector L:", output_vector_full)

        # Update the colors of the *displayed* scatter plot
        colors_for_plot = output_vector_full[self.plot_indices]
        self.scatter_plot.set_color(plt.cm.tab20(colors_for_plot / np.max(output_vector_full, initial=1)))
        
        self.ax.set_title("Analysis Complete")
        self.fig.canvas.draw()


# --- Example Usage ---

def example_usage_with_large_dataset():
    """
    Example showing how to use RoiSelector with a large synthetic dataset.
    
    This function demonstrates the basic workflow:
    1. Generate or load spatial data
    2. Create the interactive selector
    3. Draw polygons interactively
    4. Extract results
    """
    # 1. Generate a large number of sample points
    np.random.seed(42)
    num_points = 50000
    x = np.random.rand(num_points)
    y = np.random.rand(num_points)
    c = np.random.rand(num_points)  # Original color value for plotting

    # 2. Create the interactive selector
    # It will automatically downsample for the display
    roi_selector = RoiSelector(x, y, c, display_threshold=1000)
    
    # 3. Interactive drawing occurs here...
    # User clicks to draw polygons, then clicks "Finish & Analyze"
    
    # 4. After analysis is complete, extract results:
    # section_assignments = roi_selector.final_output
    # This can be used with TMG.split_into_sections(section_assignments)


def example_usage_with_tmg(TMG, display_threshold=10000):
    """
    Example showing how to use RoiSelector with TMG data.
    
    Parameters
    ----------
    TMG : TissueMultiGraph
        The TMG object containing spatial data
    display_threshold : int
        Maximum points to display for performance
        
    Returns
    -------
    RoiSelector
        The interactive selector object. After interaction, use
        roi_selector.final_output to get section assignments.
        
    Examples
    --------
    >>> # Create ROI selector for interactive section selection
    >>> roi_selector = example_usage_with_tmg(TMG, display_threshold=10000)
    >>> # ... interactive drawing occurs ...
    >>> # After finishing, create section names and split:
    >>> section_base = TMG.Layers[0].unqS[0][:-3]  # Remove last 3 chars
    >>> section_names = [f"{section_base}{assignment:02d}" for assignment in roi_selector.final_output]
    >>> TMG.split_into_sections(section_names)
    """
    # Get spatial coordinates and cell types from TMG
    XY = TMG.Layers[0].get_XY()
    Type = TMG.Layers[0].Type
    
    # Create the interactive selector
    roi_selector = RoiSelector(XY[:, 0], XY[:, 1], Type, display_threshold=display_threshold)
    
    return roi_selector 