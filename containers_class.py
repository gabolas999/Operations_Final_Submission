



import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class ContainerPlotter:
    """
    Minimal wrapper class for draw_container() and draw_capacity().
    Logic, colours, and geometry remain EXACTLY as in the original functions.
    """

    def __init__(self, width=20, x=0, y=0):
        # Colors (exact same values)
        self.color_green = "#00A63C"
        self.color_green_dark = "#006E28"

        self.color_orange = "#FF6F00"
        self.color_orange_dark = "#B23E00"

        self.color_gray = "#888888"

        self.width = width
        self.height = width * 8.6 / 20

        self.starting_x = x
        self.starting_y = y

    # ------------------------------------------------------------------
    def draw_container(self, ax, index, total_width, IorE=1, W_c=1 ):
        """
        Draw a single container in a grid layout.
        IDENTICAL to your original function.
        """

        # Effective number of columns per row
        max_cols = max(1, total_width - 1)

        idx0 = index - 1
        row = idx0 // max_cols
        col = idx0 % max_cols

        x = self.starting_x + col * self.width
        y = self.starting_y + row * self.height

        # Colours
        if IorE == 1:
            face = self.color_green
            edge = self.color_green_dark
        elif IorE == 2:
            face = self.color_orange
            edge = self.color_orange_dark
        else:
            face = "#CCCCCC"
            edge = "#888888"

        rect = Rectangle(
            (x, y),
            self.width * W_c,
            self.height,
            facecolor=face,
            edgecolor=edge,
            linewidth=2
        )
        ax.add_patch(rect)

    # ------------------------------------------------------------------
    def draw_capacity(self, ax, total_height, total_width):
        """
        Draw the grey bounding box around the full capacity.
        IDENTICAL to your original function.
        """
        max_cols = max(1, total_width - 1)

        margin = self.width / 10

        total_w = max_cols * self.width
        total_h = total_height * self.height

        color_gray = self.color_gray
        lw = 3.5

        ax.plot(
            [self.starting_x - margin * 1.1, self.starting_x - margin * 1.1],
            [self.starting_y - margin, self.starting_y + total_h],
            color=color_gray, linewidth=lw
        )

        ax.plot(
            [self.starting_x - margin, self.starting_x + total_w + margin],
            [self.starting_y - margin, self.starting_y - margin],
            color=color_gray, linewidth=lw
        )

        ax.plot(
            [self.starting_x + total_w + margin, self.starting_x + total_w + margin],
            [self.starting_y - margin, self.starting_y + total_h],
            color=color_gray, linewidth=lw
        )


if __name__ == "__main__":

    plotter = ContainerPlotter()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw capacity box
    plotter.draw_capacity(ax, total_height=5, total_width=5)

    # Draw some containers
    import random as rnd
    i = 1
    while i <= 15:
        W_c = rnd.randint(1, 2)
        W_c = 1
        IorE = rnd.randint(1, 2)
        plotter.draw_container(ax, index=i, total_width=5, IorE=IorE, W_c=W_c)
        i += W_c


    ax.set_xlim(-5, 160)
    ax.set_ylim(-10, 120)
    ax.set_aspect("equal")
    plt.show()
