"""Viz module manages all TissueMultiGraph vizualization needs

The module contains two elements: View and the Panel class hierarchy. 
View acts as:
#. connection to specific TMG container,
#. manage figure related stuff (size, save/load, etc)  
#. container of different Panels. 

The Panel hierarchy is where the specific graphics are created. 
The base class (Panel) is an abstract class with an abstract plot method users have to overload: 

All Panels include a few important properties: 
#. V : a pointer to the View that this Panel belogs to. This also acts as the link to TMG (self.V.TMG)
#. Data : a dict container for any data required for Viz, 
#. ax
# either given as input during __init__ 

A specific type of Panel that is important in the hierarchy is Map which is dedicated to spatial plotting of a TMG section. 

Other panels (Scatter, Historgram, LogLogPlot) etc are there for convinent so that View maintains 
"""

"""
TODO: 
1. Finish with legends and pie/wedge legens. 
2. clean dependencies
3. move UMAP color to coloru
4. move ploting from geomu to powerplots (?)
"""

from matplotlib.gridspec import GridSpec, SubplotSpec
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from matplotlib.patches import Wedge, Circle
from matplotlib.patches import Rectangle as pRectangle
import colorcet as cc
import math
import umap
import xycmap

import abc

import seaborn as sns

from scipy import optimize
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering
from scipy.spatial.distance import jensenshannon, pdist, squareform

from TMG.Utils import geomu
from TMG.Utils import coloru 
from TMG.Analysis.TissueGraph import *

"""
TODO: 
* Add scale-bar to Map and all subclasses (pixel size)
* Add colorbar to Colorpleth (but not to RandomColorpleth...)
* Fix Circle-legend w/o wedges 
* Fix TypeMap to the new non-stype format
* Add RegionMaps that shows regions as boundaries and cells as types. 
* In TMG - add isozones / region geometry by merging polygons using type vector (use accessory function in geomu)
"""

class View:
    def __init__(self, TMG, name=None, figsize=(11,11),facecolor='white'):
        
        # each view needs a unique name
        self.name = name
        
        # link to the TMG used to create the view
        self.TMG = TMG
        self.lims = {'x' : np.array([0, 11.4]),
                     'y' : np.array([0,8]),
                     'z' : np.array([0,13.2])}
        
        # list of all panels
        self.Panels = list()
        
        self.figsize = figsize
        self.fig = plt.figure(figsize = self.figsize)
        self.fig.patch.set_facecolor(facecolor)

        
    def show(self):
        """
        plot all panels. 
        """
        # Add all panels to figure
        for i in range(len(self.Panels)): 
            # Check if this panel requires polar axes
            if hasattr(self.Panels[i], 'requires_polar') and self.Panels[i].requires_polar:
                # Create polar axes for PolarPanel subclasses
                if isinstance(self.Panels[i].pos, SubplotSpec):
                    ax = self.fig.add_subplot(self.Panels[i].pos, projection='polar')
                else: 
                    ax = self.fig.add_axes(self.Panels[i].pos, projection='polar')
            else:
                # Create regular axes for standard panels
                if isinstance(self.Panels[i].pos, SubplotSpec):
                    ax = self.fig.add_subplot(self.Panels[i].pos)
                else: 
                    ax = self.fig.add_axes(self.Panels[i].pos)
            
            self.Panels[i].ax = ax
            plt.sca(ax)
            self.Panels[i].plot()

class BasisView(View):
    def __init__(self,TMG,section = None, basis = np.arange(24), qntl = (0.025,0.975),colormaps="jet",subplot_layout = [1,1],**kwargs):
        figsize = kwargs.get('figsize',(15,10))
        super().__init__(TMG,name = "View basis",figsize=figsize)
        self.fig.patch.set_facecolor(kwargs.get('facecolor','white'))
        # decide the subplot layout, that keep a 2x3 design
        # this will only run if the subplot_layout is smaller then needed 
        while np.prod(subplot_layout)<len(basis):
            subplot_layout[1] += 1
            if np.prod(subplot_layout)>=len(basis): 
                break
            if subplot_layout[1]/subplot_layout[0] > 1.5:
                subplot_layout[0] += 1

        # set up subpanels
        gs = self.fig.add_gridspec(subplot_layout[0],subplot_layout[1],wspace = 0.01,hspace = 0.01) 
        # add 24 Colorpleths
        feature_mat = self.TMG.Layers[0].get_feature_mat(section = section)
        feature_mat = feature_mat[:,basis]
        for i in range(len(basis)):
            P = Colorpleth(feature_mat[:,i],V = self,section = section,geom_type='voronoi',colormaps = colormaps,pos = gs[i],**kwargs,qntl = qntl)

class MultiSectionBasisView(View): 
    def __init__(self,TMG, basis = None, qntl = (0.025,0.975),clim = None, colormaps="jet",subplot_layout = [1,1],**kwargs):

        if basis is None: 
            raise ValueError("Please provide basis index to plot")

        figsize = kwargs.get('figsize',(15,10))
        super().__init__(TMG,name = "View basis",figsize=figsize)
        self.fig.patch.set_facecolor(kwargs.get('facecolor','white'))
        # decide the subplot layout, that keep a 2x3 design
        # this will only run if the subplot_layout is smaller then needed 
        while np.prod(subplot_layout)<len(TMG.unqS):
            subplot_layout[1] += 1
            if np.prod(subplot_layout)>=len(TMG.unqS): 
                break
            if subplot_layout[1]/subplot_layout[0] > 1.5:
                subplot_layout[0] += 1
        
        # set up subpanels
        gs = self.fig.add_gridspec(subplot_layout[0],subplot_layout[1],wspace = 0.01,hspace = 0.01) 

        sec_feature_mats = list()
        for i,section in enumerate(TMG.unqS):
            f_mat = self.TMG.Layers[0].get_feature_mat(section = section)
            sec_feature_mats.append(f_mat[:,basis])

        # Calculate the quantiles for the concatenated feature matrix
        if clim is None: 
            all_feature_mat = np.concatenate(sec_feature_mats, axis=0)
            clim = (np.quantile(all_feature_mat, qntl[0]), np.quantile(all_feature_mat, qntl[1]))

        for i,section in enumerate(TMG.unqS):
            P = Colorpleth(sec_feature_mats[i],V = self,section = section,geom_type='voronoi',colormaps = colormaps,pos = gs[i],**kwargs,clim = clim)

    

class MultiSectionMapView(View):
    """
    A view that creates a figure with multiple sections displayed in a grid layout.
    Each section gets its own subplot showing the specified map type.
    """
    def __init__(self, TMG, sections=None, level_type="cell", map_type="type", 
                 figsize='infer', n_columns=4, facecolor='white', **kwargs):
        """
        Create a multi-section map view.

        Parameters
        ----------
        TMG : TissueMultiGraph
            The TissueMultiGraph object containing the data.
        sections : list, optional
            A list of sections to display. If None, defaults to all sections in TMG.
        level_type : str, default "cell"
            The level of geometry to plot (e.g., "cell").
        map_type : str, default "type"
            The type of map to create (e.g., "type", "colorpleth", "random").
        figsize : tuple or 'infer', default 'infer'
            The size of the overall figure. If 'infer', calculates based on grid size.
        n_columns : int, default 4
            The number of columns in the plot grid.
        facecolor : str, default 'white'
            The face color of the figure.
        **kwargs : dict
            Additional keyword arguments passed to the individual map panels.
        """
        
        # Determine sections to plot
        if sections is None:
            sections = TMG.unqS
        
        # Calculate grid layout
        self.sections = sections
        self.level_type = level_type
        self.map_type = map_type
        self.n_columns = n_columns
        self.n_rows = math.ceil(len(sections) / n_columns)
        self.kwargs = kwargs
        
        # Calculate figure size
        if figsize == 'infer':
            figsize = (self.n_columns * 5, self.n_rows * 3.5)
        
        # Initialize the View
        super().__init__(TMG, f"Multi-Section {map_type.title()} Maps", figsize, facecolor)
        
        # Create grid spec for subplots
        gs = self.fig.add_gridspec(self.n_rows, self.n_columns, wspace=0.01, hspace=0.01)
        
        # Create panels for each section
        for i, section in enumerate(sections):
            if i >= self.n_rows * self.n_columns:
                break  # Don't exceed grid size
                
            row = i // self.n_columns
            col = i % self.n_columns
            
            # Create the appropriate map panel for this section
            if map_type == 'colorpleth':
                val_to_map = TMG.Layers[0].get_feature_mat(section=section)[:, kwargs.get('basis', 0)]
                geom_type = TMG.layer_to_geom_type_mapping[level_type]
                panel = Colorpleth(val_to_map, geom_type=geom_type, V=self, section=section, 
                                 name=f"{section}_colorpleth", pos=gs[row, col], **kwargs)
            elif map_type == 'type':
                geom_type = TMG.layer_to_geom_type_mapping[level_type]
                panel = TypeMap(geom_type, V=self, section=section, 
                              name=f"{section}_type", pos=gs[row, col], **kwargs)
            elif map_type == 'random':
                geom_type = TMG.layer_to_geom_type_mapping[level_type]
                panel = RandomColorMap(geom_type, V=self, section=section,
                                     name=f"{section}_random", pos=gs[row, col], **kwargs)
            else:
                raise ValueError(f"Unsupported map_type: {map_type}")
    
    def show(self):
        """Display the multi-section view."""
        super().show()
        
        # Clean up axes for all panels
        for panel in self.Panels:
            if panel.ax is not None:
                panel.ax.set_xticks([])
                panel.ax.set_yticks([])
                panel.ax.set_title(panel.name.split('_')[0], fontsize=8)  # Use section name as title

class SingleMapView(View):
    """
    A view that has only a single map in it. 
    map_type is one of 'type', 'colorpleth', 'random', 'type_with_boundaries' (supply two levels in level_type)
    """
    def __init__(self, TMG, section = None,level_type = "cell",map_type = "type", val_to_map = None, figsize=(11, 11),**kwargs):
        super().__init__(TMG, "Map", figsize)

        if len(TMG.unqS)==1 and section is None: 
            section = TMG.unqS[0]
        
        # Auto-calculate view limits based on actual XY data for the section
        # Only do this if xlim and ylim are not explicitly provided in kwargs
        if 'xlim' not in kwargs or 'ylim' not in kwargs:
            try:
                # Get XY coordinates for the specified section
                xy_data = TMG.Layers[0].get_XY(section=section)
                xlim_calc, ylim_calc = geomu.calculate_xy_limits(xy_data)
                if xlim_calc is not None and ylim_calc is not None:
                    # Update the view limits if not provided in kwargs
                    if 'xlim' not in kwargs:
                        self.lims['x'] = xlim_calc
                    if 'ylim' not in kwargs:
                        self.lims['y'] = ylim_calc
            except Exception as e:
                # If auto-calculation fails, keep default limits and warn
                print(f"Warning: Could not auto-calculate limits from XY data: {e}")
                print("Using default hardcoded limits.")
      
        if map_type == 'type':
            geom_type = TMG.layer_to_geom_type_mapping[level_type]
            Pmap = TypeMap(geom_type,V=self,section = section,**kwargs)
        elif map_type == 'colorpleth': 
            geom_type = TMG.layer_to_geom_type_mapping[level_type]
            if val_to_map is None: 
                raise ValueError("Must provide val_to_map if map_type is colorpleth")
            Pmap = Colorpleth(val_to_map,geom_type=geom_type,V=self,section = section,**kwargs)
        elif map_type == 'random':
            geom_type = TMG.layer_to_geom_type_mapping[level_type]
            Pmap = RandomColorMap(geom_type,V=self,section=section,**kwargs)
        elif map_type == 'type_with_boundaries':
            poly_geom = TMG.layer_to_geom_type_mapping[level_type[0]]
            bound_geom = TMG.layer_to_geom_type_mapping[level_type[1]]
            Pmap = TypeWithBoundaries(V=self, section=section, poly_geom=poly_geom, boundaries_geom=bound_geom,**kwargs)
        else: 
            raise ValueError(f"value {map_type} is not a recognized map_type")

class CellAndNeighborhoodView(View):
    """
    A view that displays two type maps side by side:
    - Left panel: cell layer type map
    - Right panel: cell_neighborhoods layer type map
    """
    def __init__(self, TMG, section=None, figsize=(22, 11), **kwargs):
        """
        Create a dual type map view with cell and cell_neighborhoods layers.

        Parameters
        ----------
        TMG : TissueMultiGraph
            The TissueMultiGraph object containing the data.
        section : str, optional
            The section to display. If None, uses the first section in TMG.
        figsize : tuple, default (22, 11)
            The size of the overall figure (width, height).
        **kwargs : dict
            Additional keyword arguments passed to the TypeMap panels.
        """
        super().__init__(TMG, "Cell and Neighborhood Type Maps", figsize)
        
        # Set default section to be the first in unqS if not specified
        if section is None and len(TMG.unqS) > 0:
            section = TMG.unqS[0]
        
        # Calculate view limits based on cell layer XY data
        # Auto-calculate view limits if not provided in kwargs
        if 'xlim' not in kwargs or 'ylim' not in kwargs:
            try:
                xy_data = TMG.Layers[0].get_XY(section=section)
                xlim_calc, ylim_calc = geomu.calculate_xy_limits(xy_data)
                if xlim_calc is not None and ylim_calc is not None:
                    if 'xlim' not in kwargs:
                        self.lims['x'] = xlim_calc
                    if 'ylim' not in kwargs:
                        self.lims['y'] = ylim_calc
            except Exception as e:
                print(f"Warning: Could not auto-calculate limits from XY data: {e}")
                print("Using default hardcoded limits.")
        
        # Create grid spec for two panels side by side
        gs = self.fig.add_gridspec(1, 2, wspace=0.02)
        
        # Create left panel for cell layer
        cell_geom_type = TMG.layer_to_geom_type_mapping["cell"]
        TypeMap(cell_geom_type, V=self, section=section, 
                name="Cell Layer", pos=gs[0], **kwargs)
        
        # Create right panel for cell_neighborhoods layer  
        neighborhoods_geom_type = TMG.layer_to_geom_type_mapping["cell_neighborhoods"]
        TypeMap(neighborhoods_geom_type, V=self, section=section,
                name="Cell Neighborhoods Layer", pos=gs[1], **kwargs)


class MultiSectionScatterView(View):
    """
    A view that generates scatter plots for multiple sections in a grid layout.
    Each scatter plot shows spatial coordinates colored by feature values.
    """
    def __init__(self, TMG, sections=None, layer=None, bit=None, n_columns=4, 
                 facecolor='black', cmap='jet', global_contrast=True, 
                 quantile=[0.05, 0.95], vmin=None, vmax=None, sort_order=True, s=0.01):
        """
        Create a multi-section scatter view.

        Parameters
        ----------
        TMG : TissueMultiGraph
            The TMG object containing the data to be plotted.
        sections : list, optional
            The sections to be plotted. If None, all unique sections in TMG will be plotted.
        layer : str, optional
            The layer of the TMG object to be used. If None, 'X' will be used.
        bit : str or int, optional
            The feature/bit to be used for coloring. If None, the first feature will be used.
        n_columns : int, default 4
            The number of columns in the plot grid.
        facecolor : str, default 'black'
            The face color of the plot.
        cmap : str, default 'jet'
            The color map to be used for the scatter plots.
        global_contrast : bool, default True
            Whether to use global contrast for the color scale. If False, local contrast will be used.
        quantile : list, default [0.05, 0.95]
            The percentiles to be used for calculating the vmin and vmax for the color scale.
        vmin, vmax : float, optional
            Explicit min/max values for color scaling. If None, calculated from quantiles.
        sort_order : bool, default True
            Whether to sort points by color value (helps with visualization).
        s : float, default 0.01
            Size of scatter plot points.
        """
        
        # Store original X data to restore later
        self.original_X = TMG.Layers[0].adata.X.copy()
        
        # Handle layer switching
        if layer is not None:
            TMG.Layers[0].adata.X = TMG.Layers[0].adata.layers[layer].copy()
            self.layer = layer
        else:
            self.layer = 'X'
        
        # Handle bit selection
        if bit is None:
            bit = TMG.Layers[0].adata.var.index[0]
        elif isinstance(bit, int):
            bit = TMG.Layers[0].adata.var.index[bit]
        else:
            if bit not in TMG.Layers[0].adata.var.index:
                raise ValueError(f"{bit} is not in the adata.var index.")
        
        self.bit = bit
        
        # Set text color based on face color
        if facecolor == 'black':
            textcolor = 'white'
        else:
            textcolor = 'black'
        
        # Determine sections to plot
        if sections is None:
            sections = TMG.unqS
        
        # Calculate grid layout
        self.sections = sections
        self.n_columns = n_columns
        self.n_rows = math.ceil(len(sections) / n_columns)
        
        # Calculate figure size
        figsize = (5 * n_columns, 3.5 * self.n_rows)
        
        # Initialize the View
        super().__init__(TMG, f"Multi-Section Scatter: {self.layer} {bit}", figsize, facecolor)
        
        # Calculate global contrast values if needed
        if global_contrast:
            C_global = TMG.Layers[0].get_feature_mat()[:, np.isin(TMG.Layers[0].adata.var.index, [bit])]
            if vmin is None:
                vmin = np.quantile(C_global, quantile[0])
            if vmax is None:
                vmax = np.quantile(C_global, quantile[1])
            
            # Set figure title with global contrast info
            self.fig.suptitle(f"{self.layer} {bit} vmin{vmin:.2f}|vmax{vmax:.2f}", 
                            color=textcolor)
        else:
            self.fig.suptitle(f"{self.layer} {bit}", color=textcolor)
        
        # Create grid spec for subplots
        gs = self.fig.add_gridspec(self.n_rows, self.n_columns, wspace=0.01, hspace=0.01)
        
        # Create panels for each section
        for i, section in enumerate(sections):
            if i >= self.n_rows * self.n_columns:
                break  # Don't exceed grid size
                
            row = i // self.n_columns
            col = i % self.n_columns
            
            # Create scatter panel for this section
            panel = MultiSectionScatter(
                V=self, section=section, layer=self.layer, bit=bit,
                cmap=cmap, global_contrast=global_contrast, quantile=quantile,
                vmin=vmin, vmax=vmax, sort_order=sort_order, s=s,
                textcolor=textcolor, name=f"scatter_{section}", pos=gs[row, col]
            )
    
    def __del__(self):
        """Restore original X data when view is destroyed."""
        try:
            if hasattr(self, 'original_X') and hasattr(self, 'TMG'):
                self.TMG.Layers[0].adata.X = self.original_X
        except:
            pass  # Ignore errors during cleanup

class UMAPwithSpatialMapView(View):
    """
    TODO: 
    1. add support multi-section
    2. allow coloring by cell types taken from TMG. 
    """
    def __init__(self,TMG,section = None, qntl = (0,1),clp_embed = (0,1),**kwargs):
        super().__init__(TMG,name = "UMAP with spatial map",figsize=(16,8))

        # add two subplots
        gs = self.fig.add_gridspec(1,2,wspace = 0.01)
        basis = self.TMG.Layers[0].get_feature_mat(section = section)

        for i in range(basis.shape[1]):
            lmts = np.percentile(basis[:,i],np.array(qntl)*100)
            basis[:,i] = np.clip(basis[:,i],lmts[0],lmts[1])
        
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(basis)
        self.ump1 = np.clip(embedding[:,0],np.quantile(embedding[:,0],clp_embed[0]),np.quantile(embedding[:,0],clp_embed[1]))
        self.ump2 = np.clip(embedding[:,1],np.quantile(embedding[:,1],clp_embed[0]),np.quantile(embedding[:,1],clp_embed[1]))

        corner_colors = kwargs.get('corner_colors',["#d10f7c", "#53e6de", "#f4fc4e", "#4e99fc"]) #magenta, cyan, yellow, blue
        self.cmap = xycmap.custom_xycmap(corner_colors = corner_colors, n = (100, 100))



        colors = xycmap.bivariate_color(sx=self.ump1, sy=self.ump2, cmap=self.cmap)
        clr = np.vstack(colors)
        clr[clr>1]=1

        # add the scatter plot
        Pumap = Scatter(self,self.ump1,self.ump2,c = clr,s=0.1,name = 'umaps',pos = gs[0],
                        xlabel = 'UMAP-1',ylabel = 'UMAP-2',xtick = [],ytick=[],**kwargs)
        Pmap = Map(V = self,section = section,rgb_faces = clr,pos = gs[1],name = 'map with UMAP colors',**kwargs)

    def show(self):
        super().show()
         # add legend 
        self.legend_ax = self.fig.add_axes([0.15, 0.15, 0.05, 0.1])
        self.legend_ax = xycmap.bivariate_legend(ax=self.legend_ax , sx=self.ump1, sy=self.ump2, cmap=self.cmap)
        self.legend_ax.set_xticks([])
        self.legend_ax.set_yticks([])


class DapiValueDistributionsView(View):
    def __init__(self,TMG,figsize = (16,8),min_dapi_line = None,max_dapi_line = None):
        super().__init__(TMG,name = "Dapi per section violin",figsize = figsize)
        if self.TMG.Nsections > 1:
            P = Violin(V = self, cat_values = TMG.Layers[0].Section,num_values = TMG.Layers[0].adata.obs['dapi'],xlabel='Section',ylabel='Dapi')
            self.min_dapi_line=min_dapi_line
            self.max_dapi_line=max_dapi_line
        else: 
            P = Histogram(V = self,values_to_count = TMG.Layers[0].adata.obs['dapi'],xlabel='Dapi')
            self.min_dapi_line=min_dapi_line
            self.max_dapi_line=max_dapi_line

    def show(self): 
        super().show()
        if self.TMG.Nsections > 1:
            if self.min_dapi_line is not None: 
                self.Panels[0].ax.axhline(self.min_dapi_line)
            if self.max_dapi_line is not None: 
                self.Panels[0].ax.axhline(self.max_dapi_line)
        else: 
            if self.min_dapi_line is not None: 
                self.Panels[0].ax.axvline(self.min_dapi_line)
            if self.max_dapi_line is not None: 
                self.Panels[0].ax.axvline(self.max_dapi_line)

class SumValueDistributionsView(View):
    def __init__(self,TMG,figsize = (16,8),min_sum_line = None,max_sum_line = None):
        super().__init__(TMG,name = "Log10 Sum per section violin",figsize = figsize)
        num_values = TMG.Layers[0].adata.X.sum(1)
        num_values[num_values<0] = 0
        num_values = np.log10(num_values+1)
        min_sum_line = np.log10(min_sum_line+1)
        max_sum_line = np.log10(max_sum_line+1)
        self.min_sum_line=min_sum_line
        self.max_sum_line=max_sum_line
        
        if self.TMG.Nsections > 1:
            P = Violin(V = self, cat_values = TMG.Layers[0].Section,num_values = num_values,xlabel='Section',ylabel='Sum')
        else: 
            P = Histogram(V = self,values_to_count = num_values,xlabel='Sum')

    def show(self): 
        super().show()
        if self.TMG.Nsections > 1:
            if self.min_sum_line is not None: 
                self.Panels[0].ax.axhline(self.min_sum_line)
            if self.max_sum_line is not None: 
                self.Panels[0].ax.axhline(self.max_sum_line)
        else: 
            if self.min_sum_line is not None: 
                self.Panels[0].ax.axvline(self.min_sum_line)
            if self.max_sum_line is not None: 
                self.Panels[0].ax.axvline(self.max_sum_line)

class ValueDistributionsView(View):
    def __init__(self,TMG,num_values,log=False,title='',figsize = (16,8),min_line = None,max_line = None):
        super().__init__(TMG,name = title+" per section violin",figsize = figsize)
        if log:
            title = 'Log10 '+title
            num_values[num_values<0] = 0
            num_values = np.log10(num_values+1)
            if not isinstance(min_line,type(None)):
                min_line = np.log10(min_line+1)
            if not isinstance(max_line,type(None)):
                max_line = np.log10(max_line+1)
        self.min_line=min_line
        self.max_line=max_line
        
        if self.TMG.Nsections > 1:
            P = Violin(V = self, cat_values = TMG.Layers[0].Section,num_values = num_values,xlabel='Section',ylabel=title)
        else: 
            P = Histogram(V = self,values_to_count = num_values,xlabel=title)

    def show(self): 
        super().show()
        if self.TMG.Nsections > 1:
            if self.min_line is not None: 
                self.Panels[0].ax.axhline(self.min_line)
            if self.max_line is not None: 
                self.Panels[0].ax.axhline(self.max_line)
        else: 
            if self.min_line is not None: 
                self.Panels[0].ax.axvline(self.min_line)
            if self.max_line is not None: 
                self.Panels[0].ax.axvline(self.max_line)


""" class IsoZonesView(View):
    # A view with two panels: a isozone colorpleth with node_size and log-log hist
    # simple overloading of colorpleth, only different is that we're getting values from TMG 
    # since TMG is only defined after init we pass the Data during set_view instead. 

    def __init__(self, section=0, name="isozones", pos=(0,0,1,1)):
        values_to_map = None
        super().__init__(values_to_map, section=section, name=name, pos=pos)
    def set_view(self):
        self.Data['values_to_map'] = np.log10(self.V.TMG.Layers[1].node_size)
        super().set_view() """



class LocalCellDensityView(View):
    """
    Creates a map of local cell density of a given section (or None for all)
    local density is estimated using k-nearest neighbors (default k=10)
    """
    def __init__(self,TMG,figsize = (16,8),section = None, qntl = (0,1),k = 10,**kwargs):
        super().__init__(TMG, "local cell density", figsize)

        # calculate cell densities
        ind = self.TMG.Layers[0].SG.neighborhood(order = k)

        # add two subplots
        gs = self.fig.add_gridspec(1,2,wspace = 0.01)

        XY = TMG.Layers[0].get_XY(section = section)

        local_dens = geomu.local_density(XY,k=k)
        
        Pmap = Colorpleth(local_dens,V=self,section = section,qntl = qntl,**kwargs)


class CellCellInteractionView(View):
    """
    A view that creates a cell-cell interaction chord diagram for a specific section.
    Automatically calculates interaction frequencies and displays them using pycirclize.
    """
    def __init__(self, TMG, section=None, figsize=(12, 12), **kwargs):
        super().__init__(TMG, "Cell-Cell Interactions", figsize)
        
        # Set default section to be the first in unqS, useful if there is only single section
        if section is None:
            section = self.TMG.unqS[0]
        
        # Create the cell-cell interaction panel
        interaction_panel = CellCellInteractions(
            V=self, 
            section=section, 
            pos=(0, 0, 1, 1),  # Full figure
            **kwargs
        )


class Panel(metaclass=abc.ABCMeta):
    def __init__(self, V = None, name = None, pos=(0,0,1,1)):
        if not isinstance(V,View):
            raise ValueError(f"The value of parameter V must be a View object, instead it was {type(V)}")
        self.V = V
        self.name = name
        self.pos = pos
        self.ax = None
        self.Data = {}
        self.clrmp = None

        # Add the Panel into the View it is linked to. 
        self.V.Panels.append(self)

    @abc.abstractmethod
    def plot(self):
        """
        actual plotting happens here, overload in subclasses! 
        """
        pass

class LayeredMap(Panel):
    """
    """
    def __init__(self, V = None, section = None, name = None, 
                       pos=(0,0,1,1), geom_list_to_plot=dict(), **kwargs):
        super().__init__(V,name, pos)
         # set default section to be the first in unqS, useful is there is only single section
        if section is None:
            section = self.V.TMG.unqS[0]    

        self.xlim = kwargs.get('xlim',None)
        self.ylim = kwargs.get('ylim',None)
        self.rotation = kwargs.get('rotation',None)
        self.axis_off = kwargs.get('axis_off', True)  # Default to axis off

        # section information: which section in TMG are we plotting
        self.section = section
        # Verify that each item in geom_list_to_plot is a dictionary with the required keys
        for item in geom_list_to_plot:
            if not isinstance(item, dict):
                raise ValueError("Each item in geom_list_to_plot must be a dictionary.")
            required_keys = {'geom', 'plot_type','rgb_faces', 'rgb_edges'}
            if not required_keys.issubset(item.keys()):
                missing_keys = required_keys - item.keys()
                raise ValueError(f"Missing keys in geom_list_to_plot item: {missing_keys}")
            
        self.geom_list_to_plot=geom_list_to_plot

    def plot(self):
        """
        The LayeredMap plotting function runs through the list of dicts and plots 
        the geometrycollection (points, lines, polygons) requested 
        """
        for geom_to_plot in self.geom_list_to_plot: 
            if geom_to_plot["plot_type"] == "points":
                self.sizes = None # TODO - make sure to fix self.sizes to something real...
                self.ax = geomu.plot_point_collection(geom_to_plot["geom"],
                                            self.sizes,
                                            rgb_faces = geom_to_plot["rgb_faces"],
                                            rgb_edges = geom_to_plot["rgb_edges"], 
                                            ax = self.ax,
                                            xlm = self.xlim,ylm = self.ylim,
                                            rotation = self.rotation,
                                            axis_off = self.axis_off)
                
            elif geom_to_plot["plot_type"] == "lines":
                self.ax = geomu.plot_polygon_boundaries(geom_to_plot["geom"],
                                              rgb_edges = geom_to_plot["rgb_edges"],
                                              linewidths=geom_to_plot["linewidths"], 
                                              inward_offset = geom_to_plot["inward_offset"],
                                              ax=self.ax,
                                              xlm = self.xlim,ylm = self.ylim,
                                              rotation = self.rotation,
                                              axis_off = self.axis_off)
                
            elif geom_to_plot["plot_type"] == "polygons": 
                self.ax = geomu.plot_polygon_collection(geom_to_plot["geom"],
                                            rgb_faces = geom_to_plot["rgb_faces"], 
                                            ax = self.ax,
                                            xlm = self.xlim,ylm = self.ylim,
                                            rotation = self.rotation,
                                            axis_off = self.axis_off)    
            else: 
                raise ValueError(f"Geometry plotting type: {geom_to_plot['plot_type']} not supported")


class Map(Panel):
    """
    Basic map plotting functionality. 
    This panels can plots single type of geometry from TMG based on use supplied rgb (Nx3 array) for faces and/or edges
    It does not calcualte any of the rgb maps. 
    """
    def __init__(self, V = None, section = None, name = None, 
                       pos=(0,0,1,1), geom_type = "voronoi",rgb_faces = None,
                       rgb_edges = None, **kwargs):
        super().__init__(V,name, pos)
        
        # set default section to be the first in unqS, useful is there is only single section
        if section is None:
            section = self.V.TMG.unqS[0]

        # section information: which section in TMG are we plotting
        self.section = section
        
        # Make sure one color was provided
        assert rgb_edges is not None or rgb_faces is not None,"To plot either pleaes provide RGB array (nx3) for either edges or faces "
        self.rgb_edges = rgb_edges
        self.rgb_faces = rgb_faces
        self.geom_type = geom_type

        # get limits
        if self.V.TMG.unqS.count(self.section) == 0:
            raise ValueError(f"section {self.section} is not found in TMG.unqS {self.V.TMG.unqS}")
        else: 
            section_ix = self.V.TMG.unqS.index(self.section)

        self.xlim = kwargs.get('xlim',V.lims['x'])
        self.ylim = kwargs.get('ylim',V.lims['y'])
        self.rotation = kwargs.get('rotation',None)
        self.axis_off = kwargs.get('axis_off', True)  # Default to axis off

        # get the geom collection saved in appropriate TMG Geom
        self.geom_collection = self.V.TMG.Geoms[section_ix][geom_type].verts

        return
        
    def plot(self):
        """
        plot a map (list of shapely points in colors) 
        """
        if self.geom_type == "points":
            self.sizes = None # TODO - make sure to fix self.sizes to something real...
            geomu.plot_point_collection(self.geom_collection,
                                        self.sizes,
                                        rgb_faces = self.rgb_faces,
                                        rgb_edges = self.rgb_edges, 
                                        ax = self.ax,
                                        xlm = self.xlim,ylm = self.ylim,
                                        rotation = self.rotation,
                                        axis_off = self.axis_off)
        else: 
            geomu.plot_polygon_collection(self.geom_collection,
                                          rgb_faces = self.rgb_faces,
                                          ax = self.ax,
                                          xlm = self.xlim,ylm = self.ylim,
                                          rotation = self.rotation,
                                          axis_off = self.axis_off)
        

class TypeMap(Map):
    """
    plots types of given geom using UMAP color projection of types
    """
    def __init__(self, geom_type, V = None,section = None, name = None, 
                                      pos = (0,0,1,1), color_assign_method = 'taxonomy',**kwargs):
        super().__init__(V = V,section = section, geom_type = geom_type, name = name, pos = pos,rgb_faces=[1,1,1],**kwargs)
        
        # find layer and tax ids
        layer = self.V.TMG.geom_to_layer_type_mapping[geom_type]
        layer_ix = self.V.TMG.find_layer_by_name(layer)
        tax_ix = self.V.TMG.layer_taxonomy_mapping[layer_ix]

        # get data and rename for simplicity
        self.Data['data'] = self.V.TMG.Taxonomies[tax_ix].feature_mat
        self.Data['tax'] = self.V.TMG.Taxonomies[tax_ix].Type
        self.Data['type'] = self.V.TMG.Layers[layer_ix].Type[self.V.TMG.Layers[layer_ix].Section==section]
        types = self.Data['type']
        target = self.Data['tax']
        _, target_index = np.unique(target, return_index=True)
        data =  self.Data['data']

        # get colors and assign for each unit (cell, isozone, regions)
        if color_assign_method == 'taxonomy': 
            self.rgb_by_type = self.V.TMG.Taxonomies[tax_ix].RGB
        elif color_assign_method == 'supervised_umap': 
            self.rgb_by_type = coloru.type_color_using_supervized_umap(data,target_index)
        elif color_assign_method == 'linkage':
            metric = kwargs.get("metric","cosine")
            self.cmap_names = kwargs.get('colormaps', "hot")
            self.cmap = coloru.merge_colormaps(self.cmap_names,range=(0.05,1))
            self.rgb_by_type = coloru.type_color_using_linkage(data,self.cmap,metric = metric)
        self.rgb_faces = self.rgb_by_type[types.astype(int),:]

class TypeWithBoundaries(LayeredMap): 

    def __init__(self, V=None, section=None, name=None, pos=(0, 0, 1, 1), poly_geom = None,  
                 boundaries_geom = None, **kwargs):
        
        geom_list_to_plot = [{'geom' : None, 'plot_type' : None,'rgb_faces' : None, 'rgb_edges' : None}]
        super().__init__(V, section, name, pos, geom_list_to_plot,**kwargs)
        self.geom_list_to_plot=[None]*2
        section_ix = self.V.TMG.unqS.index(section)

        # collect infomraiton about polygon layer (Types)
        layer = self.V.TMG.geom_to_layer_type_mapping[poly_geom]
        layer_ix = self.V.TMG.find_layer_by_name(layer)
        tax_ix = self.V.TMG.layer_taxonomy_mapping[layer_ix]

        self.rgb_by_type = self.V.TMG.Taxonomies[tax_ix].RGB
        poly_types = self.V.TMG.Layers[layer_ix].Type[self.V.TMG.Layers[layer_ix].Section==section]
        self.rgb = self.rgb_by_type[poly_types,:]
        
        self.geom_list_to_plot[0]={'geom' : self.V.TMG.Geoms[section_ix][poly_geom].polys, 
                              'plot_type' : 'polygons',
                              'rgb_faces' : self.rgb, 
                              'rgb_edges' : None}
        
        # collect infomraiton about polygon layer (Types)

        bound = self.V.TMG.geom_to_layer_type_mapping[boundaries_geom]
        bound_layer_ix = self.V.TMG.find_layer_by_name(bound)
        bound_tax_ix = self.V.TMG.layer_taxonomy_mapping[bound_layer_ix]
        
        linewidths = kwargs.get("linewidths",2)
        inward_offset = kwargs.get("inward_offset",0)

        types = self.V.TMG.Layers[bound_layer_ix].Type[self.V.TMG.Layers[bound_layer_ix].Section==section]
        self.geom_list_to_plot[1]={'geom' : self.V.TMG.Geoms[section_ix][boundaries_geom].polys, 
                              'plot_type' : 'lines',
                              'rgb_faces' : None, 
                              'rgb_edges' : self.V.TMG.Taxonomies[bound_tax_ix].RGB[types,:],
                              'linewidths' : linewidths,
                              'inward_offset' : inward_offset}


class RandomColorMap(Map):
    """
    plots types of given geom using UMAP color projection of types
    """
    def __init__(self, geom_type, V = None,cmap_list = ['jet'],section = None, name = None, 
                                      pos = (0,0,1,1), **kwargs):
        super().__init__(V = V,section = section, geom_type = geom_type, name = name, pos = pos,rgb_faces=[1,1,1],**kwargs)
        
        # set colormap
        self.cmap_names = kwargs.get('colormaps', "jet")
        self.cmap = coloru.merge_colormaps(self.cmap_names)
        # find layer and tax ids
        layer = self.V.TMG.geom_to_layer_type_mapping[geom_type]
        layer_ix = self.V.TMG.find_layer_by_name(layer)
        self.rgb_faces = self.cmap(np.linspace(0,1,self.V.TMG.N[layer_ix]))
        self.rgb_faces = self.rgb_faces[np.random.permutation(self.V.TMG.N[layer_ix]),:]
       
class Colorpleth(Map):
    """
    Show a user-provided vector color coded. 
    It will choose the geomtry (voronoi,isozones,regions) from the size of the use provided values_to_map vector. 
    """
    def __init__(self, values_to_map,geom_type = None,V = None,section = None, name = None, 
                                      pos = (0,0,1,1), qntl = (0,1), clim = None, **kwargs):
        # if geom_type is None chose it based on size of input vector
        if geom_type is None:         
            raise ValueError("Must supply geom_type to colorpleth")
        
        super().__init__(V = V,section = section, name = name, pos = pos,rgb_faces=[1,1,1],geom_type=geom_type,**kwargs)
        self.Data['values_to_map'] = values_to_map
        
        self.cmap_names = kwargs.get('colormaps', "hot")
        self.colorbar = kwargs.get('colorbar', False)
        self.cmap = coloru.merge_colormaps(self.cmap_names)
        
        # Create the rgb_faces using self.clrmp and self.Data['values_to_map']
        scalar_mapping = self.Data['values_to_map']

        if clim is None:
            clim = (np.quantile(scalar_mapping, qntl[0]), np.quantile(scalar_mapping, qntl[1]))

        # Rescale scalar_mapping based on clim
        scalar_mapping = (scalar_mapping - clim[0]) / (clim[1] - clim[0])
        scalar_mapping = np.clip(scalar_mapping, 0, 1)
        
        self.rgb_faces = self.cmap(scalar_mapping)
        
 
                 
    
""" class LegendWithCircles(Panel): 
    def __init__(self, map_panel, name=None, pos=(0,0,1,1), scale=300, **kwargs):
        super().__init__(name=name, pos=pos, **kwargs)
        self.map_panel = map_panel
        self.scale = scale
        
    def set_view(self, count_type='index'):
        lvl = self.map_panel.lvl
        if count_type == 'index':
            if lvl == 0:
                cell_mapped_index = np.arange(self.V.TMG.Layers[lvl].N)
            else:
                cell_mapped_index = self.V.TMG.Layers[lvl].Upstream
            unq, cnt = np.unique(cell_mapped_index, return_counts=True)
        elif count_type == 'type':
            # cell_mapped_types = self.map_panel.Data['values_to_map']
            cell_mapped_types = self.V.TMG.Layers[lvl].Type 
            unq, cnt = np.unique(cell_mapped_types, return_counts=True)
        self.sz = cnt.astype('float')
        self.sz = self.sz/self.sz.mean()*self.scale

        if count_type == 'index':
            layout = self.V.TMG.Layers[lvl].FG.layout_fruchterman_reingold()
        elif count_type == 'type':
            # groupby (not implemented yet)
            raise ValueError("not implemented")
            _TG = self.V.TMG.Layers[lvl]
            sumTG = _TG.contract_graph(_TG.Type) # summarize the current graph by type...
            layout = sumTG.FG.layout_fruchterman_reingold()

        xy = np.array(layout.coords)
        xy[:,0] = xy[:,0]-xy[:,0].min()
        xy[:,1] = xy[:,1]-xy[:,1].min()
        xy[:,0] = xy[:,0]/xy[:,0].max()
        xy[:,1] = xy[:,1]/xy[:,1].max()
        self.xy = xy

        self.clr = self.map_panel.basecolors[self.V.TMG.Layers[lvl].Type.astype(int)]
        self.clrmp = self.map_panel.clrmp # xxx
    
    def plot(self): 
        # plt.sca(self.ax)
        self.ax.scatter(
                    x=self.xy[:,0], 
                    y=self.xy[:,1],
                    c=self.clr,
                    s=self.sz, 
                    )
        self.ax.set_xticks([], [])
        self.ax.set_yticks([], [])
        
class LegendWithCirclesAndWedges(LegendWithCircles):
    def __init__(self, map_panel, cell_map_panel, name=None, pos=(0,0,1,1), **kwargs):
        super().__init__(map_panel, name=name, pos=pos, **kwargs)
        self.cell_map_panel = cell_map_panel
                               
    def plot(self): 

        # get fractions and sort by type2
        feature_mat = self.V.TMG.Layers[self.map_panel.lvl].feature_mat
        # ordr = np.argsort(self.cell_map_panel.type2)
        # feature_mat = feature_mat[:,ordr]
        sum_of_rows = feature_mat.sum(axis=1)
        feature_mat = feature_mat / sum_of_rows[:, None]
        n_smpl, n_ftrs = feature_mat.shape 
            
        # scale between radi in points to xy that was normed to 0-1
        scale_factor = self.V.fig.dpi * self.V.figsize[0] 
            
        xy = self.xy*scale_factor
        radi = np.sqrt(self.sz/np.pi) * self.scale/100
        cdf_in_angles = np.cumsum(np.hstack((np.zeros((n_smpl,1)), feature_mat)), axis=1)*360

        wedges = list()
        wedge_width = 0.66
        region_lvl = self.map_panel.lvl
        region_types = self.V.TMG.Layers[region_lvl].Type.astype(int)
        cell_lvl = self.cell_map_panel.lvl
        cell_types = self.V.TMG.Layers[cell_lvl].Type.astype(int)

        for i in range(n_smpl):
            for j in range(n_ftrs):
                w = Wedge((xy[i,0],xy[i,1]), radi[i], cdf_in_angles[i,j], 
                           cdf_in_angles[i,j+1],
                           width=wedge_width*radi[i], 
                           facecolor=self.cell_map_panel.basecolors[cell_types[j]]
                           )
                c = Circle((xy[i,0],xy[i,1]), wedge_width*radi[i], 
                            facecolor=self.map_panel.basecolors[region_types[i]],
                            # facecolor=self.map_panel.clr[i,:], 
                            fill=True) 
                wedges.append(w)
                wedges.append(c)

        p = PatchCollection(wedges, match_original=True)
        self.ax.add_collection(p)

        margins = 0.05
        self.ax.set_xlim(-margins*scale_factor,(1+margins)*scale_factor)
        self.ax.set_ylim(-margins*scale_factor,(1+margins)*scale_factor)
        self.ax.set_xticks([])
        self.ax.set_yticks([])




  commented section was used to plot Cond Entropy Vs resolution, not clear if that is a panel
    
        # add a panel with conditional entropy
        if hasattr(self.TMG.Layers[0], 'cond_entropy_df') and self.lvl==1:
            EntropyCalcsL1 = self.TMG.Layers[0].cond_entropy_df
            fig = plt.figure()
         
            ax1 = plt.gca()
            yopt = self.TMG.cond_entropy[1]
            xopt = self.TMG.Ntypes[1]
            ax1.plot(EntropyCalcsL1['Ntypes'],EntropyCalcsL1['Entropy'])
            ylm = ax1.get_ylim()
            ax1.plot([xopt,xopt],[ylm[0], yopt],'r--',linewidth=1)
            ax1.set_xlabel('# of types',fontsize=18)
            ax1.set_ylabel('H (Map | Type)',fontsize=18)
            fig = plt.gcf()
            left, bottom, width, height = [0.6, 0.55, 0.25, 0.25]
            ax2 = fig.add_axes([left, bottom, width, height])
            ax2.semilogx(EntropyCalcsL1['Ntypes'],EntropyCalcsL1['Entropy'])
            ylm = ax2.get_ylim()
            ax2.plot([xopt,xopt],[ylm[0], yopt],'r--',linewidth=1)
            
            fig = plt.figure()
            unq,cnt = np.unique(self.TMG.Layers[0].Type,return_counts=True)
            plt.hist(cnt,bins=15);
            plt.title("Cells per type")
            plt.xlabel("# Cells in a type")
            plt.ylabel("# of Types")


class RegionMap(CellMap): 
    def __init__(self,name = "region map"):
        super().__init__(TMG,name = name)
        self.lvl = 2
    
    def set_view(self):
        super().set_view()
        
    def plot(self,V1 = None,**kwargs):
        super().plot(**kwargs)
        if V1 is None:
            return
        # add another panel with piechart markers
        # start new figure (to calc size factor)
        fig = plt.figure(figsize = self.figsize)
        self.figs.append(fig)
        ax = plt.gca()
            
        # get fractions and sort by type2
        feature_type_mat = self.TMG.Layers[self.lvl].feature_type_mat
        ordr = np.argsort(V1.type2)
        feature_type_mat = feature_type_mat[:,ordr]
        sum_of_rows = feature_type_mat.sum(axis=1)
        feature_type_mat = feature_type_mat / sum_of_rows[:, None]
            
        # scale between radi in points to xy that was normed to 0-1
        scale_factor = fig.dpi * self.figsize[0]
            
        xy = self.xy*scale_factor
        radi = np.sqrt(self.sz/np.pi)
        cdf_in_angles = np.cumsum(np.hstack((np.zeros((feature_type_mat.shape[0],1)),feature_type_mat)),axis=1)*360

        wedges = list()
        wedge_width = 0.66
        for i in range(feature_type_mat.shape[0]):
            for j in range(feature_type_mat.shape[1]):
                w = Wedge((xy[i,0],xy[i,1]), radi[i], cdf_in_angles[i,j], 
                           cdf_in_angles[i,j+1],width = wedge_width*radi[i], facecolor = V1.clr[ordr[j],:])
                c = Circle((xy[i,0],xy[i,1]),wedge_width*radi[i],facecolor = self.clr[i,:],fill = True) 
                wedges.append(w)
                wedges.append(c)


        p = PatchCollection(wedges,match_original=True)
        ax.add_collection(p)

        margins = 0.05
        ax.set_xlim(-margins*scale_factor,(1+margins)*scale_factor)
        ax.set_ylim(-margins*scale_factor,(1+margins)*scale_factor)
        ax.set_xticks([])
        ax.set_yticks([])

next section could be used to show mapping between regions and cell
        
        fig = plt.figure(figsize=(10,10))
        self.figs.append(fig)
        region_cell_types = self.TMG.Layers[2].feature_type_mat
        row_sums = region_cell_types.sum(axis=1)
        row_sums = row_sums[:,None]
        region_cell_types_nrm=region_cell_types/row_sums
        g = sns.clustermap(region_cell_types_nrm,method="ward", cmap="mako",col_colors=V1.clr,row_colors = self.clr)
        g.ax_heatmap.set_xticks(list())
        g.ax_heatmap.set_yticks(list())
        self.figs.append(fig)
"""



class Scatter(Panel):
    def __init__(self,V,x,y,c = [0.25, 0.3, 0.95] , s = 0.1,name = 'scatter',pos = (0,0,1,1), **kwargs):
        super().__init__(V,name=name, pos=pos)
        self.Data['x'] = x
        self.Data['y'] = y
        self.Data['c'] = c
        self.Data['s'] = s
        self.xtick = kwargs.get('xtick',None)
        self.ytick = kwargs.get('ytick',None)
        self.xlabel = kwargs.get('xlabel',None)
        self.ylabel = kwargs.get('ylabel',None)
        self.label_font_size = kwargs.get('label_font_size',20)
       
    def plot(self):
        self.ax.scatter(x=self.Data['x'],y=self.Data['y'],c=self.Data['c'],s = self.Data['s'])
        if self.xtick is not None: 
            self.ax.set_xticks(self.xtick)
        if self.ytick is not None: 
            self.ax.set_yticks(self.ytick)
        self.ax.set_xlabel(self.xlabel,fontsize = self.label_font_size)
        self.ax.set_ylabel(self.ylabel,fontsize = self.label_font_size)

class Violin(Panel):
    def __init__(self,V = None,cat_values = None, num_values = None,name = 'violin',pos=(0,0,1,1),**kwargs):
        super().__init__(V,name=name, pos=pos)
        if cat_values is None:
            raise ValueError("Must supply categorical variable")
        if num_values is None: 
            raise ValueError("Must supply numerical variable")

        self.Data['cat_values'] = cat_values
        self.Data['num_values'] = num_values
        self.xlabel = kwargs.get('xlabel',None)
        self.ylabel = kwargs.get('ylabel',None)

    def plot(self):
        df = pd.DataFrame({'cat' : self.Data['cat_values'],
                           'num' : self.Data['num_values']})
        sns.violinplot(ax = self.ax,data = df,x = "cat", y = "num")
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

    
        

class Histogram(Panel):
    def __init__(self, V = None,values_to_count = None, name='hist', pos=(0,0,1,1), n_bins=50, **kwargs):
        if values_to_count is None: 
            raise ValueError("Must supply values to show their distribution")
        super().__init__(V = V,name=name, pos=pos)
        self.n_bins = n_bins 
        self.Data['values_to_count'] = values_to_count
        self.xlabel = kwargs.get('xlabel',None)

    def plot(self): 
        self.ax.hist(self.Data['values_to_count'], bins=self.n_bins)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_yticks([])

class LogLogPlot(Histogram):
    def __init__(self, V = None,values_to_count = None, name="loglog", pos=(0,0,1,1), n_bins=50, trend_line = None,**kwargs):
        super().__init__(values_to_count, name=name, pos=pos, n_bins=n_bins, **kwargs)
        
        mx_sz = self.Data['values_to_count'].max()
        bins = np.logspace(0, np.ceil(np.log10(mx_sz)), self.n_bins+1)

        # Calculate histogram
        hist = np.histogram(self.Data['values_to_count'], bins=bins)
        # normalize by bin width
        hist_norm = hist[0]/hist[0].sum()

        ix = hist_norm>0
        x=(bins[0:-1]+bins[1:])/2
        self.log_size=np.log10(x[ix])
        self.log_freq = np.log10(hist_norm[ix])

        def piecewise_linear(x, x0, y0, k1, k2):
            return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

        self.trend_line = trend_line
        if trend_line is None:
            pass
        elif self.trend_line == "piecewise_linear":
            self.p, e = optimize.curve_fit(piecewise_linear, self.log_size, self.log_freq)
            self.exponents = self.p[2:4]
        elif self.trend_line == "linear": 
            self.p = np.polyfit(self.log_size, self.log_freq, 1)
            self.exponents = self.p[0]
            pass
        
    def plot(self):
        def piecewise_linear(x, x0, y0, k1, k2):
            return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
        
        # plot it!
        self.ax.plot(10**self.log_size, 10**self.log_freq,'.')
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlabel('Value',fontsize=18)
        self.ax.set_ylabel('Freq',fontsize=18)
        if self.trend_line == "piecewise_linear": 
            self.ax.plot(10**self.log_size, 10**piecewise_linear(self.log_size, *self.p),'r-')
            self.ax.set_title(f"Exponents: {self.exponents[0]:.2f} {self.exponents[1]:.2f}")
        elif self.trend_line == "linear": 
            self.ax.plot(10**self.log_size, 10**(self.p[0]*self.log_size+self.p[1]),'r-')
            self.ax.set_title(f"Exponent: {self.exponents[0]:.2f}")


class Zoom(Panel):
    def __init__(self, panel_to_zoom, zoom_coords=np.array([0,0,1,1]), name=None, pos=(0,0,1,1)):
        f = type(panel_to_zoom) # Type of the Panel
        self.ZP = f()
        
        for attr in dir(panel_to_zoom):
            try:
                setattr(self.ZP , attr, getattr(panel_to_zoom, attr))
            except:
                print(f"cannot copy attribute: {attr}")

        self.ZP.ax = None
        self.pos = pos
        self.name = name
        self.zoom_coords = zoom_coords
        self.panel_to_zoom = panel_to_zoom

        
    def plot(self):
        # add boundary around zoomed area
        self.panel_to_zoom.ax.add_patch(pRectangle((self.zoom_coords[0], self.zoom_coords[1]),
                                   self.zoom_coords[2], self.zoom_coords[3],
                                   fc ='none', 
                                   ec ="w",
                                   lw = 3))
        self.ZP.ax = self.ax
        self.ZP.plot()
        self.ax.set_xlim(self.zoom_coords[0],self.zoom_coords[0]+self.zoom_coords[2])
        self.ax.set_ylim(self.zoom_coords[1],self.zoom_coords[1]+self.zoom_coords[3])



class MultiSectionScatter(Panel):
    """
    Panel that creates a scatter plot for a single section showing spatial coordinates 
    colored by a feature value.
    """
    def __init__(self, V, section, layer, bit, cmap='jet', global_contrast=True, 
                 quantile=[0.05, 0.95], vmin=None, vmax=None, sort_order=True, s=0.01, 
                 textcolor='white', name=None, pos=(0,0,1,1)):
        super().__init__(V, name or f"scatter_{section}", pos)
        
        self.section = section
        self.layer = layer
        self.bit = bit
        self.cmap = cmap
        self.global_contrast = global_contrast
        self.quantile = quantile
        self.vmin = vmin
        self.vmax = vmax
        self.sort_order = sort_order
        self.s = s
        self.textcolor = textcolor
        
        # Get data for this section
        self.XY = self.V.TMG.Layers[0].get_XY(section=section)
        self.C = self.V.TMG.Layers[0].get_feature_mat(section=section)[:, 
                 np.isin(self.V.TMG.Layers[0].adata.var.index, [bit])].ravel()
        
        # Calculate local contrast if needed
        if not global_contrast:
            if vmin is None:
                self.vmin = np.quantile(self.C, quantile[0])
            if vmax is None:
                self.vmax = np.quantile(self.C, quantile[1])
    
    def plot(self):
        """Plot the scatter plot for this section."""
        if self.sort_order:
            order = np.argsort(self.C)
        else:
            order = np.array(range(self.C.shape[0]))
        
        scatter = self.ax.scatter(self.XY[order, 0], self.XY[order, 1], 
                                c=self.C[order], s=self.s, cmap=self.cmap, 
                                vmin=self.vmin, vmax=self.vmax)
        
        # Set title
        if not self.global_contrast:
            title = f"{self.section} {self.vmin:.2f}|{self.vmax:.2f}"
        else:
            title = f"{self.section}"
        
        self.ax.set_title(title, color=self.textcolor, fontsize=8)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.axis('off')




class PolarPanel(Panel):
    """
    Base class for panels that require polar axes instead of regular matplotlib axes.
    
    This class handles the creation and management of polar axes, which are required
    by certain visualization libraries like pycirclize. It properly integrates with
    the Panel system by creating a polar subplot within the Panel's position.
    """
    def __init__(self, V=None, name=None, pos=(0,0,1,1)):
        # Initialize the base Panel first
        super().__init__(V, name, pos)
        
        # We'll create the polar axes during the View.show() process
        # The ax will be set to a PolarAxes object instead of regular Axes
        self.requires_polar = True
    
    def create_polar_axes(self, fig, pos):
        """
        Create polar axes for this panel.
        
        This method should be called by the View during the show() process
        to create the appropriate polar axes.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to add the polar axes to
        pos : tuple or SubplotSpec
            Position specification for the axes
            
        Returns
        -------
        matplotlib.projections.polar.PolarAxes
            The created polar axes
        """
        if isinstance(pos, SubplotSpec):
            ax = fig.add_subplot(pos, projection='polar')
        else:
            ax = fig.add_axes(pos, projection='polar')
        return ax


class CellCellInteractions(PolarPanel):
    """
    Creates a chord diagram showing cell-cell interaction frequencies between different cell types.
    Uses pycirclize to generate the visualization with colors matching the TMG taxonomy.
    
    This Panel properly uses polar axes as required by pycirclize, inheriting from PolarPanel
    to handle the polar axes creation correctly.
    """
    def __init__(self, V=None, section=None, name="Cell-Cell Interactions", pos=(0,0,1,1), 
                 min_interactions=5, max_cell_types=20, space=3, **kwargs):
        super().__init__(V, name, pos)
        
        # Set default section to be the first in unqS, useful if there is only single section
        if section is None:
            section = self.V.TMG.unqS[0]
        
        self.section = section
        self.min_interactions = min_interactions  # Minimum interactions to include a cell type
        self.max_cell_types = max_cell_types      # Maximum cell types to show for clarity
        self.space = space                        # Space between sectors in degrees
        
        # Store additional pycirclize parameters
        self.label_kws = kwargs.get('label_kws', {'r': 110, 'size': 10, 'orientation': 'vertical'})
        self.link_kws = kwargs.get('link_kws', {'ec': 'gray', 'lw': 0.3, 'alpha': 0.7})
        
        # Calculate frequency matrix and extract cell type information
        self._calculate_interaction_matrix()
        self._extract_cell_type_colors()
        
    def _calculate_interaction_matrix(self):
        """Calculate the cell-cell interaction frequency matrix using TMG's built-in method."""
        print(f"🔍 Calculating interaction matrix for section {self.section}")
        
        # Find the cell layer (layer 0 typically contains cells)
        cell_layer = self.V.TMG.Layers[0]
        
        # Get the taxonomy for proper type mapping
        cell_layer_idx = 0  # Assuming cells are in layer 0
        taxonomy_idx = self.V.TMG.layer_taxonomy_mapping.get(cell_layer_idx, 0)
        taxonomy = self.V.TMG.Taxonomies[taxonomy_idx]
        
        try:
            # Use the built-in type_neighbor_frequency method
            freq_matrix, types_used = cell_layer.type_neighbor_frequency(taxonomy=taxonomy)
            
            # If we're filtering by section, we need to recalculate for just that section
            if self.section is not None:
                # Get section-specific edges by filtering the spatial graph
                edge_list = cell_layer.spatial_edge_list
                section_mask = cell_layer.Section == self.section
                
                if section_mask.sum() == 0:
                    raise ValueError(f"No cells found in section {self.section}")
                
                # Filter edges to only include those within the specified section
                section_indices = np.where(section_mask)[0]
                edge_mask = np.isin(edge_list, section_indices).all(axis=1)
                section_edges = edge_list[edge_mask]
                
                if len(section_edges) == 0:
                    print("Warning: No spatial edges found within the specified section")
                    # Create empty matrix
                    freq_matrix = np.zeros_like(freq_matrix)
                else:
                    # Recalculate frequency matrix for section-specific edges
                    node_types = cell_layer.Type
                    type_pairs = np.column_stack([node_types[section_edges[:, 0]], 
                                                node_types[section_edges[:, 1]]])
                    
                    # Reset frequency matrix and recalculate
                    freq_matrix = np.zeros_like(freq_matrix)
                    np.add.at(freq_matrix, (type_pairs[:, 0], type_pairs[:, 1]), 1)
            
            self.Data['freq_matrix'] = freq_matrix
            self.unique_types = types_used
            
            print(f"✓ Calculated interaction matrix: {freq_matrix.shape} "
                  f"with {np.count_nonzero(freq_matrix)} non-zero interactions")
            
        except Exception as e:
            print(f"Error using type_neighbor_frequency method: {e}")
            raise ValueError(f"Could not calculate interaction matrix: {e}")
    
    def _extract_cell_type_colors(self):
        """Extract cell type names and RGB colors from TMG taxonomy."""
        # Find the appropriate taxonomy
        cell_layer_idx = 0  # Assuming cells are in layer 0
        taxonomy_idx = self.V.TMG.layer_taxonomy_mapping.get(cell_layer_idx, 0)
        taxonomy = self.V.TMG.Taxonomies[taxonomy_idx]
        
        # Extract cell type names and colors
        self.Data['cell_type_names'] = []
        self.Data['cell_type_colors'] = {}
        
        for i, type_id in enumerate(self.unique_types):
            try:
                # Get the actual cell type name from taxonomy
                cell_name = taxonomy.Type[type_id]
                self.Data['cell_type_names'].append(cell_name)
                
                # Get RGB color and convert to matplotlib format (0-1 range)
                rgb_values = taxonomy.RGB[type_id]
                
                # Handle different RGB data formats
                if hasattr(rgb_values, '__len__') and len(rgb_values) >= 3:
                    # Convert from 0-255 to 0-1 range if needed
                    if np.max(rgb_values) > 1:
                        color_tuple = tuple(c/255.0 for c in rgb_values[:3])
                    else:
                        color_tuple = tuple(rgb_values[:3])
                else:
                    # Fallback to gray if RGB extraction fails
                    color_tuple = (0.5, 0.5, 0.5)
                
                self.Data['cell_type_colors'][cell_name] = color_tuple
                
            except (IndexError, KeyError, AttributeError):
                # Fallback naming and coloring
                cell_name = f"Type_{type_id}"
                self.Data['cell_type_names'].append(cell_name)
                # Use a default color scheme
                color_val = i / len(self.unique_types)
                self.Data['cell_type_colors'][cell_name] = plt.cm.Set3(color_val)[:3]
    
    def plot(self):
        """Create and plot the chord diagram using pycirclize with proper polar axes."""
        print("🔍 CellCellInteractions.plot() called")
        
        try:
            import pandas as pd
            from pycirclize import Circos
            print("✓ Successfully imported pycirclize and dependencies")
            
            # Verify we have polar axes
            from matplotlib.projections.polar import PolarAxes
            if not isinstance(self.ax, PolarAxes):
                raise ValueError(f"Expected PolarAxes, got {type(self.ax)}. "
                               "Make sure the View properly creates polar axes for PolarPanel subclasses.")
            
            # Get the data
            freq_matrix = self.Data['freq_matrix']
            cell_type_names = self.Data['cell_type_names']
            cell_type_colors = self.Data['cell_type_colors']
            
            # Print basic statistics
            print(f"Matrix shape: {freq_matrix.shape}")
            print(f"Matrix sum: {freq_matrix.sum()}")
            print(f"Non-zero elements: {np.count_nonzero(freq_matrix)}")
            print(f"Cell types: {len(cell_type_names)}")
            
            # Filter out cell types with very few interactions
            row_sums = freq_matrix.sum(axis=1)
            col_sums = freq_matrix.sum(axis=0)
            total_interactions = row_sums + col_sums
            
            # Keep only cell types with sufficient interactions
            keep_mask = total_interactions >= self.min_interactions
            
            if keep_mask.sum() == 0:
                print("Warning: No cell types meet minimum interaction threshold. Using all types.")
                keep_mask = np.ones(len(cell_type_names), dtype=bool)
            
            # Further limit to top interacting cell types if too many
            if keep_mask.sum() > self.max_cell_types:
                keep_indices = np.where(keep_mask)[0]
                top_indices = keep_indices[np.argsort(total_interactions[keep_indices])[-self.max_cell_types:]]
                keep_mask = np.zeros(len(cell_type_names), dtype=bool)
                keep_mask[top_indices] = True
            
            # Filter data
            filtered_matrix = freq_matrix[np.ix_(keep_mask, keep_mask)]
            filtered_names = [cell_type_names[i] for i in range(len(cell_type_names)) if keep_mask[i]]
            filtered_colors = {name: cell_type_colors[name] for name in filtered_names}
            
            print(f"Using {len(filtered_names)} cell types after filtering")
            print(f"Filtered matrix sum: {filtered_matrix.sum()}")
            
            # Check if we have any interactions to plot
            if filtered_matrix.sum() == 0:
                self._plot_no_data_message("No interactions found after filtering")
                return
            
            # Create DataFrame for pycirclize
            matrix_df = pd.DataFrame(filtered_matrix, index=filtered_names, columns=filtered_names)
            print(f"Created DataFrame: {matrix_df.shape}")
            
            # Now we can use pycirclize properly with our polar axes
            print("Creating chord diagram using Circos.chord_diagram and plotting on provided polar axes...")
            
            # Use the chord_diagram static method to create the Circos object properly
            circos = Circos.chord_diagram(
                matrix_df,
                space=self.space,
                cmap=filtered_colors,
                label_kws=self.label_kws,
                link_kws=self.link_kws
            )
            
            # Plot on our provided polar axes
            circos.plotfig(ax=self.ax)
            
            print("✓ Successfully created chord diagram on polar axes")
            
            # Add title and summary statistics
            total_interactions = filtered_matrix.sum()
            n_types = len(filtered_names)
            
            
            print("✓ CellCellInteractions.plot() completed successfully")
                
        except ImportError:
            print("Error: pycirclize not available. Please install with: pip install pycirclize")
            self._plot_error_message('pycirclize not available\nPlease install: pip install pycirclize')
            
        except Exception as e:
            print(f"Unexpected error in chord diagram creation: {e}")
            import traceback
            traceback.print_exc()
            self._plot_error_message(f'Chord Diagram Error:\n{str(e)}')
    
    def _plot_no_data_message(self, message):
        """Plot a message when no data is available."""
        # Clear the polar axes and add text
        self.ax.clear()
        self.ax.text(0.5, 0.5, message,
                   ha='center', va='center', transform=self.ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   fontsize=12)
        # Hide polar axes elements
        self.ax.set_rticks([])
        self.ax.set_thetagrids([])
    
    def _plot_error_message(self, message):
        """Plot an error message."""
        # Clear the polar axes and add text
        self.ax.clear()
        self.ax.text(0.5, 0.5, message, 
                   ha='center', va='center', transform=self.ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                   fontsize=10)
        # Hide polar axes elements
        self.ax.set_rticks([])
        self.ax.set_thetagrids([])


