# 3rd party
import matplotlib
import matplotlib.pyplot as plt
from ipywidgets import widgets
import numpy as np


class DrawingPad:
    """ Interactive drawing pad controlled by left mouse button.
    Implements a grid of cells, where each cell consists of
    (Hc, Wc) pixels. One cell is the smallest interaction unit,
    i.e. all pixels in the cell are either on or off. The real
    data matrix size is thus (Hg * Hc, Wg * Wc).

    Args:
        shape_grid (tuple): Shape of the grid, shape (Hg, Wg).
        shape_cell (tuple): # pixels per grid cell, shape (Hc, Wc).
    """

    def __init__(self, shape_grid, shape_cell):
        # Data matrix shape.
        self._shape_cell = np.array(shape_cell)
        h, w = np.array(shape_grid) * self._shape_cell

        # Vis. and mouse control flags.
        self._initialized = False
        self._down = False

        # Data.
        self._data = np.zeros((h, w), dtype=np.float32)

        # Create figure, figure manager, configure axes.
        fig = plt.figure(figsize=(1, w / h), dpi=h)
        self._figmngr = plt.get_current_fig_manager()
        self._ax = fig.gca()
        self._ax.set_xlim(0, w)
        self._ax.set_ylim(h, 0)
        self._ax.set_xticks([])
        self._ax.set_yticks([])

        # Image canvas object.
        self._img_obj = self._ax.imshow(self._data, cmap='gray')

        # Mouse callbacks.
        plt.connect('motion_notify_event', self._on_move)
        plt.connect('button_press_event', self._on_press)
        plt.connect('button_release_event', self._on_release)

        # Reset button.
        rbutton = widgets.Button(description='reset')
        rbutton.on_click(self._on_reset_button_click)
        display(rbutton)

    @property
    def data(self):
        """ Data matrix getter. """
        return self._data

    @property
    def grid(self):
        """ Getter. """
        return self._data[::self._shape_cell[0], ::self._shape_cell[1]]

    def _reset(self):
        """ Sets the underlying data matrix to 0, initializes the object. """
        self._data[:] = 0.
        self._initialized = False
        self._img_obj.set_data(self._data)
        self._figmngr.canvas.draw_idle()

    def __getitem__(self, key):
        return self._data[key[0], key[1]]

    def __setitem__(self, key, val):
        r, c = np.array(key) // self._shape_cell
        ch, cw = self._shape_cell
        self._data[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = val

    def _on_move(self, event):
        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))

        # Only draw if left button pressed.
        if self._down and not self[y, x]:
            self[y, x] = 1.
            self._img_obj.set_data(self._data)
            self._figmngr.canvas.draw_idle()

            # Due to pyplot bug, self._img_obj must be set twice.
            if not self._initialized:
                self._initialized = True
                self._img_obj = self._ax.imshow(self._data, cmap='gray')

    def _on_press(self, event):
        self._down = True

    def _on_release(self, event):
        self._down = False

    def _on_reset_button_click(self, b):
        self._reset()

def visualize_convolution():
    fig1 = plt.figure(figsize=(8, 4))

    x = np.array([2,2,2,2,2,2,2,10,10,10,10,10,1,1,1,1,1,1,1,1,5,5,5,5,5])
    h = np.array([-1,0,1])
    y = np.convolve(a=x,v=h,mode='same')

    for i in range(-1,-1+y.shape[0]):
        fig1.clear()

        plt.subplot(1,2,1)
        markerline, stemlines, baseline = plt.stem(x, linefmt=':', markerfmt="*", label="input")
        plt.setp(stemlines, color='r', linewidth=2)
        plt.stem(range(i,i+3), h[::-1], label="filter")
        plt.title("input signal and filter")
        plt.legend()

        plt.subplot(1,2,2)
        plt.stem(y[0:(i+2)], label="result")
        plt.title("result")
        plt.legend()

        plt.suptitle("convolution visualization")

        fig1.canvas.draw()
        user_input = input()

        if (user_input == "q" or user_input == "Q"):
            break

def accuracy(x, y):
    """ Accuracy.

    Args:
        x (torch.Tensor of float32): Predictions (logits), shape (B, C), B is
            batch size, C is num classes.
        y (torch.Tensor of int64): GT labels, shape (B, ),
            B = {b: b \in {0 .. C-1}}.

    Returns:
        Accuracy, in [0, 1].
    """
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    return np.mean(np.argmax(x, axis=1) == y)

def load_blackwhite_image(image_name):
    image = np.mean(plt.imread(image_name), axis=2)
    return image
