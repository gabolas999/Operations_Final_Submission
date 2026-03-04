



import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class ContainerPlotter:
    """
    Minimal wrapper class for draw_container() and draw_capacity().
    Logic, colours, and geometry remain EXACTLY as in the original functions.
    """

    def __init__(self, width=20, x=22, y=22, sign_x=1, sign_y=1):
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

        self.sign_x = sign_x
        self.sign_y = sign_y

    # ------------------------------------------------------------------
    def draw_container(self, ax, index, total_height, total_width, IorE=1, W_c=1 ):
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

        if self.sign_x == -1:
            x = self.starting_x - (col + 1) * self.width
        
        if self.sign_y == -1:
            y = self.starting_y - total_height * self.height + (row - 1) * self.height

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
            linewidth=1.8
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
        lw = 3

        if self.sign_y == 1:
            ax.plot(
                [self.starting_x - self.sign_x * margin * 1.1, self.starting_x - self.sign_x * margin * 1.1],
                [self.starting_y - margin, self.starting_y + total_h],
                color=color_gray, linewidth=lw
            )

            ax.plot(
                [self.starting_x - self.sign_x * margin, self.starting_x + self.sign_x * total_w + self.sign_x * margin],
                [self.starting_y - margin, self.starting_y - margin],
                color=color_gray, linewidth=lw
            )

            ax.plot(
                [self.starting_x + self.sign_x * total_w + self.sign_x * margin, self.starting_x + self.sign_x * total_w + self.sign_x * margin],
                [self.starting_y - margin, self.starting_y + total_h],
                color=color_gray, linewidth=lw
            )



        elif self.sign_y == -1:
            # Left vertical line (flipped vertically)
            ax.plot(
                [self.starting_x - self.sign_x * margin * 1.1, 
                self.starting_x - self.sign_x * margin * 1.1],
                [self.starting_y - total_h - self.height - margin, 
                self.starting_y - self.height - margin],
                color=color_gray, linewidth=lw
            )


            # Bottom horizontal line (flipped vertically)
            ax.plot(
                [self.starting_x - self.sign_x * margin,
                self.starting_x + self.sign_x * total_w + self.sign_x * margin],
                [self.starting_y - total_h - self.height - margin, self.starting_y - total_h - self.height- margin],
                color=color_gray, linewidth=lw
            )

            # Right vertical line (flipped vertically)
            ax.plot(
                [self.starting_x + self.sign_x * total_w + self.sign_x * margin, 
                self.starting_x + self.sign_x * total_w + self.sign_x * margin],
                [self.starting_y - total_h - self.height - margin, 
                self.starting_y - self.height - margin],
                color=color_gray, linewidth=lw
            )

if __name__ == "__main__":

    plotter = ContainerPlotter(sign_x=1, sign_y=-1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw capacity box
    plotter.draw_capacity(ax, total_height=4, total_width=5)

    # Draw some containers
    import random as rnd
    rnd.seed(0)
    i = 1
    while i <= 12:
        W_c = rnd.randint(1, 2)
        W_c = 1
        IorE = rnd.randint(1, 2)
        plotter.draw_container(ax, index=i, total_height=4,total_width=5, IorE=IorE, W_c=W_c)
        i += W_c


    plt.plot(0, 0, marker="o")  # Dummy plot to fix autoscaling
    ax.set_xlim(-160, 160)
    ax.set_ylim(-120, 120)
    ax.set_aspect("equal")
    plt.show()
