import time
import jpype
import jpype.imports
from jpype.types import *
from datetime import timedelta
from .tonal_tracker import TonalTracker


def dt_tonals_tracking(filename, start_s, stop_s, graph_ret=False, *args):
    """
    Equivalent Python function to `dtTonalsTracking` in MATLAB.
    
    Args:
        filename (str): The filename or list of filenames.
        start_s (float): Start time in seconds.
        stop_s (float): Stop time in seconds.
        
    Returns:
        tonals (list): List of tonal objects if required.
        subgraphs (list): List of subgraph objects if required.
    """

    # Start the stopwatch
    stopwatch_start = time.time()
    subgraphs = None
    
    # Initialize the Java TonalTracker with provided arguments
    tt = TonalTracker(filename, start_s, stop_s, *args)
    
    # Process the file
    tt.processFile()  # Assuming this processes the tonal data
    
    # Retrieve subgraphs if required
    if graph_ret:
        subgraphs = tt.getGraphs()
    graph_s = time.time() - stopwatch_start  # Time for graph processing
    
    stopwatch_start = time.time()
    tonals = tt.getTonals()
    discarded_count = tt.getDiscardedCount()
    disambiguate_s = time.time() - stopwatch_start  # Time for disambiguation

    # Print summary of results
    elapsed_s = graph_s + disambiguate_s
    processed_s = tt.getEndTime() - tt.getStartTime()

    def to_time(seconds):
        """Convert seconds to HH:MM:SS format."""
        return str(timedelta(seconds=seconds))

    # Summary output
    print(f'Detected {tonals.size()} tonals, rejected {discarded_count} shorter than {tt.thr.minlen_s} s')
    print('Timing statistics')
    print('function\tduration\tx Realtime')
    print(f'graph gen\t{to_time(graph_s)}\t{processed_s / graph_s:.2f}')
    print(f'tonal gen\t{to_time(disambiguate_s)}\t{processed_s / disambiguate_s:.2f}')
    print(f'Overall\t\t{to_time(elapsed_s)}\t{processed_s / elapsed_s:.2f}')

    return tonals, subgraphs
