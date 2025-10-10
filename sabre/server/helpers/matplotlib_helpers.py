"""
Matplotlib helper utilities.

Provides helpers for creating and managing matplotlib figures.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MatplotlibToImage:
    """
    Context manager for creating matplotlib figures.

    Creates a figure with specified size and dpi, and automatically
    captures it as an image that gets displayed to the user.

    Example:
        with matplotlib_to_image(figsize=(28.0, 18.0), dpi=130) as fig:
            plt.plot([1, 2, 3], [4, 5, 6])
            plt.title("My Plot", fontsize=28)
    """

    def __init__(self, figsize=(28.0, 18.0), dpi=130):
        """
        Initialize context manager.

        Args:
            figsize: Figure size in inches (width, height)
            dpi: Resolution in dots per inch
        """
        self.figsize = figsize
        self.dpi = dpi

    def __enter__(self):
        """Enter context - create and return figure."""
        import matplotlib

        matplotlib.use("Agg")  # Ensure non-interactive backend

        from matplotlib import pyplot as plt

        plt.figure(figsize=self.figsize, dpi=self.dpi)
        return plt.gcf()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - figure will be auto-captured by runtime."""
        # Don't close the figure - let the runtime's auto-capture handle it
        # The runtime will capture all open figures after execution
        return False


def matplotlib_to_image(figsize=(28.0, 18.0), dpi=130):
    """
    Create a context manager for matplotlib figures.

    Use this when you want explicit control over figure creation.
    Figures created with this context manager will be automatically
    captured and displayed to the user.

    Example:
        with matplotlib_to_image(figsize=(28.0, 18.0), dpi=130) as fig:
            plt.plot([1, 2, 3], [4, 5, 6])
            plt.title("My Plot", fontsize=28)
            plt.xlabel("X Label", fontsize=24)
            plt.ylabel("Y Label", fontsize=24)

    Args:
        figsize: Figure size in inches (width, height). Default (28.0, 18.0)
        dpi: Resolution in dots per inch. Default 130

    Returns:
        Context manager that yields a matplotlib Figure object
    """
    return MatplotlibToImage(figsize=figsize, dpi=dpi)


def generate_graph_image(x_y_data_dict: Dict[str, Any], title: str, x_label: str, y_label: str) -> str:
    """
    Generate a simple line plot from x/y data.

    Creates a publication-quality line plot with large fonts suitable
    for display. The figure is automatically captured and shown to the user.

    Example:
        generate_graph_image(
            x_y_data_dict={"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]},
            title="My Graph Title",
            x_label="X Label",
            y_label="Y Label"
        )

    Args:
        x_y_data_dict: Dictionary with 'x' and 'y' keys containing data lists
        title: Graph title
        x_label: Label for x-axis
        y_label: Label for y-axis

    Returns:
        Status message (figure is auto-captured and displayed)
    """
    import matplotlib

    matplotlib.use("Agg")  # Ensure non-interactive backend

    from matplotlib import pyplot as plt

    # Extract data
    data_dict = {}

    if isinstance(x_y_data_dict, list):
        # If just a list, use indices as x
        list_data = [float(item) for item in x_y_data_dict if isinstance(item, (int, float))]
        data_dict["x"] = list(range(len(list_data)))
        data_dict["y"] = list_data
    elif isinstance(x_y_data_dict, dict) and "dates" in x_y_data_dict and "prices" in x_y_data_dict:
        # Handle date/price format
        data_dict["x"] = [str(timestamp) for timestamp in x_y_data_dict["dates"]]
        data_dict["y"] = x_y_data_dict["prices"]
    elif "x" not in x_y_data_dict:
        # Use dict keys/values as x/y
        data_dict["x"] = list(x_y_data_dict.keys())
        data_dict["y"] = list(x_y_data_dict.values())
    else:
        # Use provided x/y
        data_dict = x_y_data_dict

    # Create figure with large size
    plt.figure(figsize=(28.0, 18.0))
    plt.plot(data_dict["x"], data_dict["y"])
    plt.title(title, fontsize=28)
    plt.xlabel(x_label, fontsize=24)
    plt.ylabel(y_label, fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, alpha=0.3)

    # Figure will be auto-captured by runtime's _capture_matplotlib_figures()
    logger.info(f"Generated graph: {title}")

    return f"Generated graph: {title}"
