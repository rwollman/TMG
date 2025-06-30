"""
Minimal fileu module for TMG package.
Handles logging, anndata loading, and polygon I/O operations.
"""
import os
import logging
import numpy as np
import anndata
import shapely.wkt


def update_user(message, level=20, logger=None, verbose=False):
    """
    Send string messages to logger with various levels of importance.
    
    :param message: Message to log
    :type message: str
    :param level: Logging level (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR)
    :type level: int
    :param logger: Logger name or logger object
    :type logger: str or logging.Logger
    :param verbose: If True, also print to console
    :type verbose: bool
    """
    if verbose:
        print(message)
    
    if isinstance(logger, str):
        log = logging.getLogger(logger)
    elif logger is None:
        log = logging.getLogger('Update_User')
    else:
        log = logger
        
    log.log(level, message)


def load(path='', file_type='anndata', model_type='', fname='', logger='FileU'):
    """
    Load files - simplified version supporting only anndata and mask loading.
    
    :param path: Path to file directory
    :type path: str
    :param file_type: Type of file to load ('anndata' or 'mask')
    :type file_type: str
    :param fname: Direct filename if provided
    :type fname: str
    :param logger: Logger name
    :type logger: str
    :return: Loaded data
    """
    if fname:
        filename = fname
    else:
        # Simple filename generation for the specific cases TMG uses
        if file_type == 'anndata':
            filename = os.path.join(path, 'anndata.h5ad')
        elif file_type == 'mask':
            filename = os.path.join(path, f'{model_type}_mask.pt')
        else:
            raise ValueError(f'Unsupported file type: {file_type}')
    
    if not os.path.exists(filename):
        update_user(f'File does not exist: {filename}', level=30, logger=logger)
        return None
    
    update_user(f"Loading {os.path.basename(filename)}", level=20, logger=logger)
    
    if file_type == 'anndata':
        return anndata.read_h5ad(filename)
    elif file_type == 'mask':
        import torch
        return torch.load(filename)
    else:
        raise ValueError(f'Unsupported file type: {file_type}')


def save_polygon_list(polys, fname):
    """
    Save a list of polygon geometries to WKT format.
    
    :param polys: List of polygon objects
    :type polys: list
    :param fname: Output filename
    :type fname: str
    """
    with open(fname, 'w') as f:
        for poly in polys:
            f.write(poly.wkt + '\n')


def load_polygon_list(fname):
    """
    Load polygon geometries from WKT format file.
    
    :param fname: Input filename
    :type fname: str
    :return: List of polygon objects
    :rtype: list
    """
    polys = []
    with open(fname, 'r') as f:
        for line in f:
            poly = shapely.wkt.loads(line.strip())
            polys.append(poly)
    return polys