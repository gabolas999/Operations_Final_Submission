# container_plot.py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random as rnd




def draw_container(ax, index, total_width, IorE=1, W_c=1, width=20, height=8.6):
    """
    Draw a single container in a grid layout.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes to draw on.
    index : int
        1-based index of the container (1, 2, 3, ...).
    total_width : int
        Layout parameter: effective containers per row = total_width - 1.
        Example: total_width=5 -> max 4 containers per row.
    width : float
        Container width (default 20).
    height : float
        Container height (default 8.6).
    IorE : int
        1 = green container (import), 2 = orange container (export).
    """
    color_green = "#00A63C"
    color_green_dark = "#006E28"

    color_orange = "#FF6F00"
    color_orange_dark = "#B23E00"

    

    # Effective number of columns per row, following your specification:
    # total_width = 5  -> max 4 containers in a row
    max_cols = max(1, total_width - 1)

    # Convert 1-based index -> 0-based for row/col computation
    idx0 = index - 1
    row = idx0 // max_cols
    col = idx0 % max_cols

    # Compute coordinates

    x = col * width
    y = row * height 

    # Choose colors based on IorE flag
    if IorE == 1:
        face = color_green
        edge = color_green_dark
    elif IorE == 2:
        face = color_orange
        edge = color_orange_dark
    else:
        # Fallback neutral color if flag is unexpected
        face = "#CCCCCC"
        edge = "#888888"

    rect = Rectangle(
        (x, y),
        width*W_c,
        height,
        facecolor=face,
        edgecolor=edge,
        linewidth=2
    )
    ax.add_patch(rect)


def draw_capacity(ax, total_height, total_width, width=20, height=8.6):
    """
    Draw a grey bounding box showing the full capacity of the ship.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes to draw on.
    total_height : int
        Number of container rows.
    total_width : int
        Layout parameter: effective slots per row = total_width - 1.
    width : float
        Base container slot width.
    height : float
        Container height.
    """
    # Effective number of columns per row (same as in draw_container)
    max_cols = max(1, total_width - 1)

    margin = width / 10

    # Total width and height spanned by the capacity grid
    total_w = max_cols * width
    total_h = total_height * height

    # Grey color for the capacity box
    color_gray = "#888888"


    lw = 3.5

    # Left vertical line
    ax.plot([-margin*1.1, -margin*1.1], [-margin, total_h], color=color_gray, linewidth=lw)

    # Bottom horizontal line
    ax.plot([-margin, total_w+margin], [-margin, -margin], color=color_gray, linewidth=lw)

    # Right vertical line
    ax.plot([total_w+margin, total_w+margin], [-margin, total_h], color=color_gray, linewidth=lw)


def main():
    rnd.seed(0)
    # Create a clean, white figure with no axes
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    # Remove all axes, ticks, and spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


    draw_capacity(ax, total_height=6, total_width=8)

    i = 1
    while i < 49:

        IorE = rnd.randint(1, 2)
        # IorE = 2
        W_c  = rnd.randint(1, 2)
        W_c  = 1

        # print(f"Drawing container {i}: IorE={IorE}, W_c={W_c}")
        draw_container(ax, i, total_width=8, IorE=IorE, W_c=W_c)

        # Skip the next plotting index if W_c == 2
        if W_c == 2:
            i += 2       # uses two spots
        else:
            i += 1       # uses one spot
            

    # Adjust view limits to fit container nicely
    ax.set_xlim(-5, 160)
    ax.set_ylim(-10, 120)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("container_example.png", dpi=300)



if __name__ == "__main__":
    main()
