import numpy as np
from matplotlib import pyplot as plt

def draw(points, **scatter_params):
    """
    Displays the points contained in the matrix passed as parameter on a plot.
    Each column represents a dimension.
    """
    if scatter_params is None:
        scatter_params = {label: 'Data points'}
    X = points[:,0] / points[:,2]
    Y = points[:,1] / points[:,2]

    # Plot the points
    plt.scatter(X, Y, **scatter_params)


def draw_ellipse(points, transformation):
    """
    Displays original and transformed points
    """    
    transformed_points = transformation @ points
    draw(points.T, label="Original points")
    draw(transformed_points.T, label="Transformed points", s=1)

    # Plot the predicted ellipse
    # - Generate unit circle
    angles = np.linspace(0, 2 * np.pi, 1000).reshape(1000, 1)
    circle = np.hstack([np.cos(angles), np.sin(angles), np.ones(angles.shape)]).T
    # - Apply affine transformation, i.e inverse of predicted transformation, to get ellipse
    predicted_ellipse = np.linalg.inv(transformation) @ circle
    predicted_ellipse = predicted_ellipse / predicted_ellipse[2]
    plt.plot(predicted_ellipse[0, :], predicted_ellipse[1, :], '1r', label='Predicted Ellipse')
    plt.legend()
    plt.axis('scaled')
    
def get_points_and_transformation(N=500):
    DIM = 2

    # Generate random points on the unit circle by sampling uniform angles
    theta = np.random.uniform(0, 2*np.pi, (N,1))
    eps_noise = 0.2 * np.random.normal(size=[N,1])
    circle = np.hstack([np.cos(theta), np.sin(theta)])

    # Stretch and rotate circle to an ellipse with random linear tranformation
    B = np.random.randint(-3, 3, (DIM, DIM))
    noisy_ellipse = circle.dot(B) + eps_noise

    # Extract x coords and y coords of the ellipse as column vectors
    X = noisy_ellipse[:,0:1]
    Y = noisy_ellipse[:,1:]

    # Plot the noisy data
    #plt.scatter(X, Y, label='Data Points')
    return noisy_ellipse, B

def get_ellipse(transformation):
    # Plot the original ellipse from which the data was generated
    phi = np.linspace(0, 2*np.pi, 1000).reshape((1000,1))
    c = np.hstack([np.cos(phi), np.sin(phi)])
    ground_truth_ellipse = c.dot(transformation)
    #plt.plot(ground_truth_ellipse[:,0], ground_truth_ellipse[:,1], 'k--', label='Generating Ellipse')
    return ground_truth_ellipse
#    return ground_truth_
