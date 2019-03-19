import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotC(fracts, nDims, trials):
    plt.figure()
    domain = np.round(np.linspace(2,200,25)).astype(int)/nDims
    plt.plot(domain,fracts)
    plt.xlabel("N/D")
    plt.ylabel("C(N,D)")
    plt.title("Fraction of convergences per {} trials as a function of N".format(trials))

def plot3Dscatter(X):
    prefix = "original" if X.shape[1]<3 else "transformed"
    if X.shape[1]<3:
        Z = np.zeros((X.shape[0],1))
        X = np.hstack([X, Z])
        
    fig = plt.figure()
    ax  = Axes3D(fig)
    point_size = 1000
    colors_vec = ["red","blue","blue","red"]
    ax.scatter(list(X[:,0]), list(X[:,1]), list(X[:,2]), s=point_size, c=colors_vec)
    ax.set_zlim3d(0,2)

    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_zlabel("z")
    plt.title("XOR problem {} data points".format(prefix))


'''Plotting helper for SVM exercise'''
def plot(X,Y,clf,show=True,dataOnly=False):
    
    plt.figure()
    # plot data points
    X1 = X[Y==0]
    X2 = X[Y==1]
    Y1 = Y[Y==0]
    Y2 = Y[Y==1]
    class1 = plt.scatter(X1[:, 0], X1[:, 1], zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)
    class2 = plt.scatter(X2[:, 0], X2[:, 1], zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)
    if not dataOnly:
        # get the range of data
        x_min = X[:, 0].min() 
        x_max = X[:, 0].max() 
        y_min = X[:, 1].min() 
        y_max = X[:, 1].max() 

        # sample the data space
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

        # apply the model for each point
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)

        # plot the partitioned space
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        
        # plot hyperplanes
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                    linestyles=['--', '-', '--'], levels=[-1, 0, 1], alpha=0.5)
        
        # plot support vectors
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                edgecolors='g', s=100, linewidth=1)
    if dataOnly:
        plt.title('Data Set')
    else:
        if clf.kernel == 'rbf':
            plt.title('Decision Boundary and Margins, C={}, gamma={}'.format(clf.C,clf.gamma)) 
        elif clf.kernel == 'poly':
            plt.title('Decision Boundary and Margins, C={}, degree={}'.format(clf.C,clf.degree)) 
        else:
            plt.title('Decision Boundary and Margins, C={}'.format(clf.C)) 
        
    plt.legend((class1,class2),('Claas A','Class B'),scatterpoints=1,
           loc='upper left',
           ncol=3,
           fontsize=8)
    if show:
        plt.show()
        
'''Plotting Heatmap for CV results'''
def plot_cv_result(grid_val,grid_search_c,grid_search_gamma):
    plt.figure(figsize=(8,10))
    plt.imshow(grid_val)
    plt.colorbar()
    plt.xticks(np.arange(len(grid_search_gamma)), grid_search_gamma, rotation=20)
    plt.yticks(np.arange(len(grid_search_c)), grid_search_c, rotation=20)
    plt.xlabel('Gamma')
    plt.ylabel('C')
    plt.title('Val Accuracy for different Gamma and C')
    plt.show()
