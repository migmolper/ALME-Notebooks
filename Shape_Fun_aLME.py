#!/opt/anaconda3/bin/python3

from __future__ import print_function
from ipywidgets import (
    interactive,
    IntSlider,
    FloatSlider,
    HBox,
    Layout,
)
import ipywidgets as widgets
from IPython.display import display
from IPython.display import Image


import sys
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab


"""
  Shape functions based in :
  " Local maximum-entropy approximation schemes : a seamless 
  bridge between finite elements and meshfree methods "
  by M.Arroyo and M.Ortiz, 2006.

  Here we employ the same nomenclature as in the paper. With the single
  different of the "l" variable wich represents the distances between the
  evaluation point and the neighborhood nodes.

  List of functions :
  - LME_lambda_NR
  - LME_fa
  - LME_p
  - LME_r
  - LME_J
  - LME_dp
"""


def LME_compute_beta(Gamma, DeltaX):
    """
        Initial values of beta
    """
    Beta = Gamma / (DeltaX * DeltaX)

    return Beta


def LME_compute_R(Gamma, DeltaX):

    TOL_LME = 10e-10

    R = DeltaX * np.sqrt(-np.log(TOL_LME) / Gamma)

    return R


def LME_update_metric_tensor_Kumar(G, DeltaF):
    """
        Introduce some deformation...
    """
    DeltaF_inv = np.linalg.inv(DeltaF)
    DeltaF_inv_tran = np.transpose(DeltaF_inv)
    aux1 = np.matmul(G, DeltaF_inv)
    G = np.matmul(DeltaF_inv_tran, aux1)

    return G


def LME_update_metric_tensor_Molinos(F):
    """
        Introduce some deformation...
    """
    F_inv = np.linalg.inv(F)
    F_inv_tran = np.transpose(F_inv)
    b_m1 = np.matmul(F_inv_tran, F_inv)

    return b_m1


def LME_compute_metric_tensor_U(F):
    """
    """
    F_tran = np.transpose(F)
    C = np.matmul(F_tran, F)
    Eigen_val_C, Eigen_vec_C = np.linalg.eig(C)

    Lambda = np.sqrt(Eigen_val_C)

    i_1 = Lambda[0] + Lambda[1]
    i_2 = Lambda[0] * Lambda[1]
    i_3 = Lambda[0] * Lambda[1]

    D = i_1 * i_2 - i_3

    U = (1 / D) * (-np.matmul(C, C) + (i_1 * i_1 - i_2) * C + i_1 * i_3 * np.eye(2))

    return U


def LME_r(l, p):
    """
    Get the gradient "r" (dim x 1) of the function log(Z) = 0.
    Input parameters :
    -> l : Matrix with the distances to the 
    neighborhood nodes (neighborhood x dim).
    -> p : Shape function value in the
    neighborhood nodes (1 x neighborhood).
    """

    # Definition of some parameters
    N_a = np.shape(l)[0]
    N_dim = 2

    # Value of the gradient
    r = np.zeros([N_dim])

    # Fill ''r''
    for i in range(0, N_dim):
        for a in range(0, N_a):
            r[i] += p[a] * l[a][i]

    # Return the value of the gradient
    return r


def LME_J(l, p, r):
    """
    Get the Hessian "J" (dim x dim) of the function log(Z) = 0.
    Input parameters :
    -> l : Matrix with the distances to the
    neighborhood nodes (neighborhood x dim).
    -> p : Shape function value in the
    neighborhood nodes (neighborhood x 1).
    -> r : Gradient of log(Z) (dim x 1).
    """
    # Definition of some parameters
    N_a = np.shape(l)[0]
    N_dim = 2

    # Allocate Hessian */
    J = np.zeros([N_dim, N_dim])

    # Fill the Hessian
    # Get the first component of the Hessian
    for a in range(0, N_a):
        J = J + p[a] * np.tensordot(l[a, :], l[a, :], axes=0)
    # Get the second value of the Hessian
    J = J - np.tensordot(r, r, axes=0)

    # Return the value of the Hessian
    return J


def LME_p(X_I, X_p, Beta, R, F):
    """
    Get the lagrange multipliers "lambda" (1 x dim) for the LME 
    shape function. The numerical method for that is the Newton-Rapson.
    
    Input parameters :
    -> l : Matrix with the distances to the
    neighborhood nodes (neighborhood x dim).
    -> lambda : Initial value of the
    lagrange multipliers (1 x dim).
    -> Beta : Tunning parameter (scalar).
    -> h : Grid spacing (scalar).
    -> TOL_zero : Tolerance for Newton-Rapson.
    """

    # Definition of some parameters */
    MaxIter = 10
    Ndim = 2
    NumIter = 0  # Iterator counter
    # Value of the norm
    norm_r = 10.0
    norm_r0 = 10.0
    Relative_norm = 10
    TOL_NR = 1e-12  # tolerance

    Lambda = np.zeros([2])

    l = np.zeros_like(X_I)
    for a in range(0, l.shape[0]):
        l[a, :] = X_p - X_I[a, :]

    # Definition of some parameters
    N_a = np.shape(l)[0]

    # Vector with the values of the shape-function in the nodes
    p = np.zeros([N_a])

    G = np.matmul(np.linalg.inv(F.transpose()), np.linalg.inv(F))

    # Start with the Newton-Rapson method
    Convergence = False
    while Convergence == False:

        # Compute p
        Z = 0
        for a in range(0, N_a):
            la = l[a, :]
            distance = np.sqrt(np.matmul(la, np.matmul(G, la)))
            if distance < R:
                p[a] = np.exp(-Beta * distance * distance + np.dot(la, Lambda))
                Z += p[a]

        p[:] *= 1.0 / Z

        # Compute grad(log(Z)) and its norm
        r = LME_r(l, p)
        norm_r = np.linalg.norm(r)

        # Get a reference value of the tolerance
        if NumIter == 0:
            norm_r = norm_r0
        # compute the relative error
        Relative_norm = norm_r / norm_r0

        # Update the number of iterations
        if (Relative_norm < TOL_NR) or (NumIter >= MaxIter):
            if NumIter >= MaxIter:
                print(
                    "The Maximum number of interations (",
                    MaxIter,
                    ") was exceded :",
                    NumIter,
                    "/",
                )
            Convergence = True

        else:
            NumIter += 1

            # Get the Hessian of log(Z) and update it with +||r||*I
            # according with Dennis M.Kochmann et al. 2019 (CMAME)
            J = LME_J(l, p, r)
            J = J + norm_r * np.eye(2)

            # Check the conditioning number of the modified Hessian
            rcond = np.linalg.cond(J)
            if rcond > 10:
                print("The Hessian is near to singular matrix ", "Iter :", NumIter)

            # Get the increment of lambda
            D_Lambda = np.linalg.solve(J, r)

            # Update the value of lambda
            for i in range(0, Ndim):
                Lambda[i] -= D_Lambda[i]

    # Return the shape function
    return p


def LME_dp(X_I, X_p, Beta, R, F):
    """
    Get the lagrange multipliers "lambda" (1 x dim) for the LME 
    shape function. The numerical method for that is the Newton-Rapson.
    
    Input parameters :
    -> l : Matrix with the distances to the
    neighborhood nodes (neighborhood x dim).
    -> lambda : Initial value of the
    lagrange multipliers (1 x dim).
    -> Beta : Tunning parameter (scalar).
    -> h : Grid spacing (scalar).
    -> TOL_zero : Tolerance for Newton-Rapson.
    """

    # Definition of some parameters */
    MaxIter = 10
    Ndim = 2
    NumIter = 0  # Iterator counter
    # Value of the norm
    norm_r = 10.0
    norm_r0 = 10.0
    Relative_norm = 10
    TOL_NR = 1e-12  # tolerance

    Lambda = np.zeros([2])

    l = np.zeros_like(X_I)
    for a in range(0, l.shape[0]):
        l[a, :] = X_I[a, :] - X_p

    # Definition of some parameters
    N_a = np.shape(l)[0]

    # Vector with the values of the shape-function in the nodes
    p = np.zeros([N_a])

    G = np.matmul(np.linalg.inv(F.transpose()), np.linalg.inv(F))

    # Start with the Newton-Rapson method
    Convergence = False
    while Convergence == False:

        # Compute p
        Z = 0
        for a in range(0, N_a):
            la = l[a, :]
            distance = np.sqrt(np.matmul(la, np.matmul(G, la)))
            if distance < R:
                p[a] = np.exp(-Beta * distance * distance + np.dot(la, Lambda))
            Z += p[a]

        p[:] *= 1.0 / Z

        # Compute grad(log(Z)) and its norm
        r = LME_r(l, p)
        norm_r = np.linalg.norm(r)

        # Get a reference value of the tolerance
        if NumIter == 0:
            norm_r = norm_r0
        # compute the relative error
        Relative_norm = norm_r / norm_r0

        # Update the number of iterations
        if (Relative_norm < TOL_NR) or (NumIter >= MaxIter):
            if NumIter >= MaxIter:
                print(
                    "The Maximum number of interations (",
                    MaxIter,
                    ") was exceded :",
                    NumIter,
                    "/",
                )
            Convergence = True

        else:
            NumIter += 1

            # Get the Hessian of log(Z) and update it with +||r||*I
            # according with Dennis M.Kochmann et al. 2019 (CMAME)
            J = LME_J(l, p, r) 
            J = J + norm_r * np.eye(2)

            # Check the conditioning number of the modified Hessian
            rcond = np.linalg.cond(J)
            if rcond > 10:
                print("The Hessian is near to singular matrix ", "Iter :", NumIter)

            # Get the increment of lambda
            D_Lambda = np.linalg.solve(J, r)

            # Update the value of lambda
            for i in range(0, Ndim):
                Lambda[i] -= D_Lambda[i]

    # Compute gradient
    dp = np.zeros([N_a, 2])
    for a in range(0, N_a):
        dp[a, :] = - p[a] * np.matmul(np.linalg.inv(J),l[a, :]) 

    # Return the shape function
    return dp


def create_mesh(x0, DeltaX):
    X_I = np.array(
        [
            ######################
            x0[0] + -0.5 * DeltaX,  # x
            x0[1] + 0.0,  # y
            x0[0] + 0.5 * DeltaX,  # x
            x0[1] + 0.0,  # y
            x0[0] + -1.5 * DeltaX,  # x
            x0[1] + 0.0,  # y
            ######################
            x0[0] + 0.0,  # x
            x0[1] + 0.5 * DeltaX * np.sqrt(3),  # y
            x0[0] + DeltaX,  # x
            x0[1] + 0.5 * DeltaX * np.sqrt(3),  # y
            x0[0] + -DeltaX,  # x
            x0[1] + 0.5 * DeltaX * np.sqrt(3),  # y
            ######################
            x0[0] + 0.0,  # x
            x0[1] + -0.5 * DeltaX * np.sqrt(3),  # y
            x0[0] + DeltaX,  # x
            x0[1] + -0.5 * DeltaX * np.sqrt(3),  # y
            x0[0] + -DeltaX,  # x
            x0[1] + -0.5 * DeltaX * np.sqrt(3),  # y
            ######################
            x0[0] + 0.5 * DeltaX,  # x
            x0[1] + DeltaX * np.sqrt(3),  # y
            x0[0] + -0.5 * DeltaX,  # x
            x0[1] + DeltaX * np.sqrt(3),  # y
            x0[0] + -1.5 * DeltaX,  # x
            x0[1] + DeltaX * np.sqrt(3),  # y
            ######################
        ]
    )
    X_I = np.resize(X_I, (int(0.5 * np.size(X_I)), 2))

    return X_I


def automatic_create_mesh(N, DeltaX):

    X_I = create_mesh(np.array([0, 0]), DeltaX)

    for i in range(-N, N + 1, 1):
        for j in range(-N, N + 1, 1):
            if (i == 0 and j == 0) == False:
                x0 = np.array([i * (DeltaX * 3), j * (DeltaX * 2 * np.sqrt(3))])
                X_I_aux = create_mesh(x0, DeltaX)
                X_I = np.concatenate((X_I, X_I_aux))

    return X_I


def print_N_LME(X_p, Gamma, F, N, DeltaX, Plot_opt, Scale_value):

    # Define mesh
    X_I = automatic_create_mesh(N, DeltaX)

    # Compute shape function
    Beta = LME_compute_beta(Gamma, DeltaX)
    R = LME_compute_R(Gamma, DeltaX)

    p = LME_p(X_I, X_p, Beta, R, F)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(min(X_I[:, 0]), max(X_I[:, 0]))
    ax.set_ylim(min(X_I[:, 1]), max(X_I[:, 1]))
    ax.set_zlim(0, max(p))
    my_cmap = plt.get_cmap("jet")
    if Plot_opt == 0:
        surface = ax.plot_trisurf(X_I[:, 0], X_I[:, 1], p, cmap=my_cmap)
        fig.colorbar(surface, shrink=0.5, aspect=5)
    elif Plot_opt == 1:
        ax.scatter3D(X_I[:, 0], X_I[:, 1], p, c=p, cmap=my_cmap)

    return ax, fig


def print_dNdx_LME(X_p, Gamma, F, N, DeltaX, Plot_opt, Scale_value):

    # Define mesh
    X_I = automatic_create_mesh(N, DeltaX)

    # Compute shape function
    Beta = LME_compute_beta(Gamma, DeltaX)
    R = LME_compute_R(Gamma, DeltaX)

    print(X_I.shape)

    print(Beta)

    dp = LME_dp(X_I, X_p, Beta, R, F)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(min(X_I[:, 0]), max(X_I[:, 0]))
    ax.set_ylim(min(X_I[:, 1]), max(X_I[:, 1]))
    my_cmap = plt.get_cmap("jet")
    if Plot_opt == 0:
        ax.plot_trisurf(X_I[:, 0], X_I[:, 1], dp[:, 0], cmap=my_cmap)
    elif Plot_opt == 1:
        ax.scatter3D(X_I[:, 0], X_I[:, 1], dp[:, 0], c=dp[:, 0], cmap=my_cmap)

    return ax, fig


def print_dNdy_LME(X_p, Gamma, F, N, DeltaX, Plot_opt, Scale_value):

    # Define mesh
    X_I = automatic_create_mesh(N, DeltaX)

    # Compute shape function
    Beta = LME_compute_beta(Gamma, DeltaX)
    R = LME_compute_R(Gamma, DeltaX)

    print(X_I.shape)

    print(Beta)

    dp = LME_dp(X_I, X_p, Beta, R, F)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(min(X_I[:, 0]), max(X_I[:, 0]))
    ax.set_ylim(min(X_I[:, 1]), max(X_I[:, 1]))
    my_cmap = plt.get_cmap("jet")
    if Plot_opt == 0:
        ax.plot_trisurf(X_I[:, 0], X_I[:, 1], dp[:, 1], cmap=my_cmap)
    elif Plot_opt == 1:
        ax.scatter3D(X_I[:, 0], X_I[:, 1], dp[:, 1], c=dp[:, 1], cmap=my_cmap)

    return ax, fig


def animate_LME():
    F = np.eye(2, 2)
    gamma = 0.1
    N = 10
    DeltaX = 1

    positions = np.arange(-30, 30, 0.1)

    for a in range(0, 600):

        xp = np.array([positions[a], 0.0])
        ax, fig = print_N_LME(xp, gamma, F, N, DeltaX, 0, 1)
        plt.tight_layout()
        fig.savefig("Frame_%i.png" % (a))
        plt.close(fig)


def animate_ALME():
    F = np.eye(2, 2)
    F[0, 1] = 1.0
    gamma = 0.1
    N = 5
    DeltaX = 1

    positions = np.arange(-30, 30, 0.1)

    for a in range(0, 600):

        xp = np.array([positions[a], 0.0])
        ax, fig = print_N_LME(xp, gamma, F, N, DeltaX, 0, 1)
        plt.tight_layout()
        fig.savefig("Frame_%i.png" % (a))
        plt.close(fig)


def LME_interactive(gamma):
    F = np.eye(2, 2)
    N = 5
    DeltaX = 1
    Plot_opt = 0

    X_p = np.array([0.333333, 0.0])

    # Define mesh
    X_I = automatic_create_mesh(N, DeltaX)

    # Compute shape function
    Beta = LME_compute_beta(gamma, DeltaX)
    R = LME_compute_R(gamma, DeltaX)

    p = LME_p(X_I, X_p, Beta, R, F)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(min(X_I[:, 0]), max(X_I[:, 0]))
    ax.set_ylim(min(X_I[:, 1]), max(X_I[:, 1]))
    ax.set_zlim(0, max(p))
    my_cmap = plt.get_cmap("jet")
    if Plot_opt == 0:
        surface = ax.plot_trisurf(X_I[:, 0], X_I[:, 1], p, cmap=my_cmap)
        fig.colorbar(surface, shrink=0.5, aspect=5)
    elif Plot_opt == 1:
        ax.scatter3D(X_I[:, 0], X_I[:, 1], p, c=p, cmap=my_cmap)

    fig.tight_layout()


def ALME_interactive(x_p, y_p, gamma, shear, Plot_opt):
    F = np.eye(2, 2)
    F[0, 1] = shear
    N = 2
    DeltaX = 0.5

    X_p = np.array([x_p, y_p])

    # Define mesh
    X_I = automatic_create_mesh(N, DeltaX)

    # Compute shape function
    Beta = LME_compute_beta(gamma, DeltaX)
    R = LME_compute_R(gamma, DeltaX)

    p = LME_p(X_I, X_p, Beta, R, F)
    dp = LME_dp(X_I, X_p, Beta, R, F)

    # Twice as wide as it is tall.
    fig = plt.figure(figsize=(15, 5))
    my_cmap = plt.get_cmap("jet")

    # ---- First subplot
    ax_N = fig.add_subplot(1, 3, 1, projection="3d")
    ax_N.set_title("N (beta = %f)" % (Beta), fontsize=20)
    ax_N.set_xlim(min(X_I[:, 0]), max(X_I[:, 0]))
    ax_N.set_ylim(min(X_I[:, 1]), max(X_I[:, 1]))

    if Plot_opt == 0:
        ax_N.plot_trisurf(X_I[:, 0], X_I[:, 1], p, cmap=my_cmap)
    elif Plot_opt == 1:
        ax_N.scatter3D(X_I[:, 0], X_I[:, 1], p, c=p, cmap=my_cmap)

    # ---- Second subplot
    ax_dNdx = fig.add_subplot(1, 3, 2, projection="3d")
    ax_dNdx.set_title("dNdx", fontsize=20)
    ax_dNdx.set_xlim(min(X_I[:, 0]), max(X_I[:, 0]))
    ax_dNdx.set_ylim(min(X_I[:, 1]), max(X_I[:, 1]))

    if Plot_opt == 0:
        ax_dNdx.plot_trisurf(X_I[:, 0], X_I[:, 1], dp[:, 0], cmap=my_cmap)
    elif Plot_opt == 1:
        ax_dNdx.scatter3D(X_I[:, 0], X_I[:, 1], dp[:, 0], c=dp[:, 0], cmap=my_cmap)

    # ---- Third subplot
    ax_dNdy = fig.add_subplot(1, 3, 3, projection="3d")
    ax_dNdy.set_title("dNdy", fontsize=20)
    ax_dNdy.set_xlim(min(X_I[:, 0]), max(X_I[:, 0]))
    ax_dNdy.set_ylim(min(X_I[:, 1]), max(X_I[:, 1]))

    if Plot_opt == 0:
        ax_dNdy.plot_trisurf(X_I[:, 0], X_I[:, 1], dp[:, 1], cmap=my_cmap)
    elif Plot_opt == 1:
        ax_dNdy.scatter3D(X_I[:, 0], X_I[:, 1], dp[:, 0], c=dp[:, 1], cmap=my_cmap)

    fig.tight_layout()
