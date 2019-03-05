# 3rd party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn import datasets

# Python std
import itertools

EPS = 1e-8

def load_ds_iris(sep_l=True, sep_w=True, pet_l=True, pet_w=True,
                 setosa=True, versicolor=True, virginica=True):
    """ Loads the iris dataset [1]. The function arguments select which
    features and classes will be included in the dataset.

    [1] https://en.wikipedia.org/wiki/Iris_flower_data_set

    Args:
        sep_l (bool): Include "sepal length" feature.
        sep_w (bool): Include "sepal width" feature.
        pet_l (bool): Include "petal length" feature.
        pet_w (bool): Include "petal width" feature.
        setosa (bool): Include "setosa" class, 50 samples.
        versicolor (bool): Include "versicolor" class, 50 samples.
        virginica (bool): Include "virginica" class, 50 samples.

    Returns:
        data (np.array of float64): Data, shape (N, D), N depends on `setosa`,
            `versicolor`, `virginica` (each gives 50 samples), D depends on
            `sep_l`, `sep_w`, `pet_l`, `pet_w` (each gives 1 feature).
        labels (np.array of int64): Labels, shape (N, ).
    """

    # Load ds.
    d, l = datasets.load_iris(return_X_y=True)
    data = np.empty((0, 4))
    labels = np.empty((0, ), dtype=np.int64)

    # Get classes.
    for idx, c in enumerate([setosa, versicolor, virginica]):
        if c:
            data = np.concatenate([data, d[l == idx]], axis=0)
            labels = np.concatenate([labels, l[l == idx]], axis=0)

    # Get features.
    feats_incl = []
    for idx, f in enumerate([sep_l, sep_w, pet_l, pet_w]):
        if f:
            feats_incl.append(idx)
    data = data[:, feats_incl]

    return data, labels


def scatter2d_multiclass(data, labels, fig=None, fig_size=None, color_map=None,
                         marker_map=None, legend=True, legend_map=None,
                         grid=False, show=False, aspect_equal=False):
    """ Plots the 2D scatter plot for multiple classes.

    Args:
        data (np.array of float): Data, shape (N, 2), N is # of samples of
            (x, y) coordinates.
        labels (np.array of int): Class labels, shape (N, )
        fig (plt.Figure): The Figure to plot to. If None, new Figure will be
            created.
        fig_size (tuple): Figure size.
        color_map (dict): Mapping of classes inds to string color codes.
            If None, each class is assigned different color automatically.
        maerker_map (dict): Mapping of classes inds to to string markers.
        legend (bool): Whetehr to print a legend.
        legend_map (dict): Mapping of classes inds to str class labels.
            If None, the int inds are uased as labels.
        grid (bool): Whether to show a grid.
        show (bool): Whether to show the plot.
        aspect_equal (bool): Whether to equalize the aspect ratio for the axes.

    Returns:
        plt.Figure
    """
    # Check dims.
    labels = labels.flatten()
    assert(data.ndim == 2 and data.shape[1] == 2)
    assert(data.shape[0] == labels.shape[0])

    # Get classes.
    classes = np.unique(labels)

    # Prepare class colors.
    colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
    if color_map is None:
        color_map = {}
        for cl in classes:
            color_map[cl] = next(colors)
    # assert(np.all(sorted(list(color_map.keys())) == classes))
    assert(np.all([cl in color_map.keys() for cl in classes]))

    # Prepare class markers.
    markers = itertools.cycle(['o', 'x', '+', '*', 'D', 'p', 's'])
    if marker_map is None:
        marker_map = {}
        for cl in classes:
            marker_map[cl] = next(markers)
    # assert (np.all(sorted(list(marker_map.keys())) == classes))
    assert (np.all([cl in marker_map.keys() for cl in classes]))

    # Prepare legend labels.
    if legend_map is None:
        legend_map = {}
        for cl in classes:
            legend_map[cl] = cl
    assert(np.all(sorted(list(legend_map.keys())) == classes))

    # Plots
    if fig is None:
        fig, _ = plt.subplots(1, 1, figsize=fig_size)
    ax = fig.gca()
    for cl in classes:
        ax.plot(data[:, 0][labels == cl], data[:, 1][labels == cl],
                linestyle='', marker=marker_map[cl], color=color_map[cl],
                label=legend_map[cl])

    if aspect_equal:
        ax.set_aspect('equal', adjustable='datalim')

    if legend:
        ax.legend()
    if grid:
        ax.grid()
    if show:
        fig.show()

    return fig


def vis_lin_decision_boundary(line_params, fig, color='k', linestyle='-',
                              keep_axes_lims=True, show=False):
    """ Plots a linear decision boundary spanning the whole plot.

    Args:
        line_params (np.array of float): 2D line parameters (a, b, c) as in
            linear equation: ax + by + c = 0
        fig (plt.Figure): Figure to plot to.
        color (str): Color as str code.
        linestyle (str): Linestyle.
        keep_axes_lims (bool): Whether to adjust the axes limits to match
            the span of the decision boundary line. If 'False' an the line
            happens not to intersect the rectangle given by the current `fig`s
            axes limits, the boundary line will not be seen.
        show (bool): Whether to show the plot.

    Returns:
        plt.Figure
    """
    ax = fig.gca()
    xlim = np.array(ax.get_xlim())
    ylim = np.array(ax.get_ylim())

    ab = line_params[:2]
    xaxis = np.array([1., 0.])
    yaxis = np.array([0., 1.])

    # Depending on the slope of the line, find x or y coordinates.
    if ab.dot(xaxis) >= ab.dot(yaxis):
        x = xlim
        y = line_y(line_params, x)
    else:
        y = ylim
        x = line_x(line_params, y)

    ax.plot(x, y, linestyle=linestyle, color=color)

    if keep_axes_lims:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    if show:
        fig.show()

    return fig


def line_y(line_params, x):
    """ Returns the value "y" in implicit linear equation ax + by + c = 0.

    Args:
        line_params (np.array of float): Line parameters (a, b, c).
        x (np.array of float): x coordinates, shape (N, ).

    Returns:
        y (np.array of float): y coordinates, shape (N, ).
    """
    a, b, c = line_params
    b = (b, b + EPS * (-1., 1.)[b >= 0.])[np.isclose(b, 0., atol=EPS)]
    return -(a / b) * x - (c / b)


def line_x(line_params, y):
    """ Returns the value "x" in implicit linear equation ax + by + c = 0.

    Args:
        line_params (np.array of float): Line parameters (a, b, c).
        y (np.array of float): y coordinates, shape (N, ).

    Returns:
        y (np.array of float): x coordinates, shape (N, ).
    """
    a, b, c = line_params
    a = (a, a + EPS * (-1., 1.)[a >= 0.])[np.isclose(a, 0., atol=EPS)]
    return -(b / a) * y - (c / a)

def load_ds1(center=False):
    """ Loads the iris dataset with features "sepal length" as x,
    "sepal width" as y and with classes "setosa" and "versicolor".

    Args:
        center (bool): Whether to translate the data such that the mean
            of ceontrids of the classes would be (0, 0).

    Returns:
        data (np.array of float64): Data, shape (100, 2)
        labels (np.array of int64): Labels, shape (100, )
    """
    data, labels = \
        load_ds_iris(sep_l=True, sep_w=True, pet_l=False, pet_w=False,
                              setosa=True, versicolor=True, virginica=False)
    labels[labels == 0] = -1

    if center:
        c1 = np.mean(data[labels == -1], axis=0)
        c2 = np.mean(data[labels == 1], axis=0)
        centr = (c1 + c2) * 0.5
        data -= centr

    return data, labels

def vis_ds1(data, labels, fig=None):
    """ Visualizes dataset1 - iris ds consisting of classes
    "setosa", "versicolor" and features "sepal length" and "sepal width".

    Args:
        data (np.array of float64): Data, shape (100, 2)
        labels (np.array of int64): Labels, shape (100, )

    Returns:
        plt.Figure
    """
    if data.shape[1] == 3:
        data = data[:, 1:]

    fig = scatter2d_multiclass(data, labels, fig=fig,
                                        color_map={-1: 'r', 1: 'g'},
                                        marker_map={-1: 'x', 1: 'o'},
                                        legend_map={-1: 'setosa',
                                                     1: 'versicolor'},
                                        show=False, aspect_equal=True)
    return fig


def vis_ds1_pred(data, labels_gt, labels_pred, fig=None):
    """ Visualizes the dataset, where the GT classes are denoted by "x" and
    "o" markers which are colored according to predicted labels.

    Args:
        data (np.array): Dataset, shape (N, D).
        labels_gt (np.array): GT labels, shape (N, ).
        labels_pred (np.array): Predicted labels, shape (N, )
        fig (plt.Figure): Figure to plot to. If None, new one is created.

    Returns:
        plf.Figure: Figure.
    """
    if data.shape[1] == 3:
        data = data[:, 1:]

    inds_c = labels_gt == labels_pred
    inds_w = labels_gt != labels_pred

    if np.any(inds_c):
        fig = scatter2d_multiclass(
            data[inds_c], labels_gt[inds_c], fig=fig,
            color_map={-1: 'r', 1: 'g'}, marker_map={-1: 'x', 1: 'o'},
            show=False, legend=False, aspect_equal=True)

    if np.any(inds_w):
        fig = scatter2d_multiclass(
            data[inds_w], labels_gt[inds_w], fig=fig,
            color_map={-1: 'g', 1: 'r'}, marker_map={-1: 'x', 1: 'o'},
            show=False, legend=False, aspect_equal=True)

    aux = [Line2D([0], [0], color='r', marker='s', linestyle=''),
           Line2D([0], [0], color='g', marker='s', linestyle=''),
           Line2D([0], [0], color='k', marker='x', linestyle=''),
           Line2D([0], [0], color='k', marker='o', linestyle='')]

    fig.gca().legend(aux, ['setosa (pred)', 'versicolor (pred)',
                           'setosa (GT)', 'versicolor (GT)'])

    return fig


def vis_ds1_dec_bound(data, labels_gt, labels_pred, w, fig=None):
    """ Visualizes the predictions together with a decision boundary.

    Args:
        data (np.array): Dataset, shape (N, D).
        labels_gt (np.array): GT labels, shape (N, ).
        labels_pred (np.array): Predicted labels, shape (N, )
        w (np.array): Weights, either (w1, w2) or (bias, w1, w2).
        fig (plt.Figure): Figure to plot to. If None, new one is created.

    Returns:
        plt.Figure: Figure.
    """
    if w.shape[0] == 2:
        w = np.append(w, 0.)
    else:
        w = np.roll(w, -1)

    fig = vis_ds1_pred(data, labels_gt, labels_pred, fig=fig)
    fig = vis_lin_decision_boundary(w, fig=fig)
    return fig


def vis_learn_curve(history, iters_per_sample=1):
    """ Visualizes the learning curve.

    Args:
        history (np.array): Metric values over iterations, shape (N, ).
        iters_per_sample (int): Period at which the metric was being saved.
    """
    history = np.array(history)
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(history.shape[0]) * iters_per_sample, history * 100.)
    ax.set_xlabel('# iterations')
    ax.set_ylabel('accuracy [%]')
    fig.show()


class PerceptronStepVisualizer:
    """ Perceptron learning algorithm step visualizer. Upon `step()` is called,
    the visualizers gradually shows the dataset, selected sample and current
    decision boundary with its normal, and then the new decision boundary with
    its normal.

    Args:
        data (np.array): Dataset, shape (N, D).
        labels (np.array): GT labels, shape (N, ).
    """
    def __init__(self, data, labels):
        self._fig, self._ax = plt.subplots(1, 1)
        self._data = data
        self._labels = labels

    def step(self, sample, w_old, w_new, labels_pred):
        """
        Args:
            sample (np.array): Data sample, shape (2, ) or (3, ).
            w_old (np.array): Previous weights, shape (2, ) or (3, ).
            w_new (np.array): New weights, shape (2, ) or (3, ).
            labels_pred (np.array): Predicted labels for the whole dataset,
                shape (N, ).
        """
        self._reset()

        w_old = np.append(w_old, 0.) if w_old.shape[0] == 2 \
            else np.roll(w_old, -1)
        w_new = np.append(w_new, 0.) if w_new.shape[0] == 2 \
            else np.roll(w_new, -1)

        vis_ds1_pred(self._data, self._labels, labels_pred, fig=self._fig)
        # vis_ds1(self._data, self._labels, labels fig=self._fig)
        vis_lin_decision_boundary(w_old, self._fig, show=False)

        # Plot decision boundary normal as an arrow.
        n_old = w_old[:2]
        self._ax.arrow(0., 0., n_old[0], n_old[1], head_width=0.1,
                 length_includes_head=True, color='b')
        self._ax.text(n_old[0], n_old[1], 'n_old', color='b')
        self._fig.canvas.draw()
        input()

        # Highlight selected sample and draw an arrow pointing to it.
        s = sample if sample.shape[0] == 2 else sample[1:]
        self._ax.plot(s[0], s[1], marker='x', color='k')
        self._ax.arrow(0., 0., s[0], s[1], head_width=0.1,
                       length_includes_head=True, color='k')
        self._fig.canvas.draw()
        input()

        # Draw new decision boundary and its normal.
        n_new = w_new[:2]
        self._ax.arrow(0., 0., n_new[0], n_new[1], head_width=0.1,
                       length_includes_head=True, linestyle='--', color='b')
        self._ax.text(n_new[0], n_new[1], 'n_new', color='b')
        vis_lin_decision_boundary(w_new, self._fig, linestyle='--',
                                           show=False)
        self._fig.canvas.draw()
        input()

    def _reset(self):
        """ Resets the figure content.
        """
        self._fig.clear()
        self._ax = self._fig.add_subplot(1, 1, 1)
