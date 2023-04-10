# by Hussein Nassar (nassarh@missouri.edu)
# you are free to use and modify for research and education purposes with
# proper citation and attribution.
# for commercial use, bugs and questions, contact the author.

import numpy as np
import numpy.linalg as la


#########################################
# Tools to construct intersection of three spheres
#########################################
def w(x, y, x0, y0, z0, sgn=1):
    # Find intersection z of three spheres centered at 0, x and y
    # and of radius vectors z0, z0-x0 and z0-y0
    # (x,y,z) is right-handed if sgn > 0

    # Define basis (xs,ys,n) dual to (x,y,n)
    # n unitary orthogonal to (x,y)
    # (x,y,n) is right handed
    n = np.cross(x, y)
    J = la.norm(n)

    # Check if basis is degenerate
    if J == 0.0:
        raise ValueError("basis degenerate - angle too small")

    n = n / J
    xs = np.cross(y, n) / J
    ys = np.cross(n, x) / J

    # Define solution coordinates
    z1 = np.dot(z0, x0)
    z2 = np.dot(z0, y0)

    z3squared = (
        la.norm(z0) ** 2
        - la.norm(z1 * xs) ** 2
        - la.norm(z2 * ys) ** 2
        - 2 * z1 * z2 * np.dot(xs, ys)
    )

    # Check if solution exists
    if z3squared < 0:
        raise ValueError(
            "no solution - angle too large (well, it could be too small too)"
        )

    z3 = np.sqrt(z3squared)

    # Build intersection with given orientation (sgn)
    z = z1 * xs + z2 * ys + np.sign(sgn) * z3 * n

    return z


#########################################
# Cauchy's (initial value) problem under periodic boundary conditions
#########################################
def onestep(
    u,
    v,
    u0,
    v0,
    w0,
    rot=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    sgn=0,
    roll=1,
):
    # Apply the three spheres intersection for one dent
    # Takes into account periodic bc through rot
    # roll = 1 => u and v are principal folds
    #       -1 => u and v are diagonal faint creases

    # Setup:
    #
    #  rot-1 <-      ********V***************U********      -> rot
    #                *             * * *             *
    #                *          *    *    *          *
    #                *      W-V      W      W-U      *      turn = 1
    #                *    *          *          *    *
    #                * *             *             * *
    #                ********V***************U********

    if roll == 1:
        # use U and V to build W
        ww = w(u, v, u0, v0, w0, sgn=sgn)
        uu = ww - v
        vv = ww - u
    else:
        # use W-U and W-V to build W
        ww = w(np.dot(rot, u), v, w0 - v0, w0 - u0, w0, sgn=sgn)
        uu = ww - v
        ww = w(u, np.dot(v, rot), w0 - v0, w0 - u0, w0, sgn=sgn)
        vv = ww - u

    return uu, vv, ww


def manysteps(
    u,
    v,
    u0,
    v0,
    w0,
    cells,
    rot=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    sgnop=1
):
    # Iterate onestep for many cells
    # Use sgnop to alternate between the two possible periodic foldings:
    # sgnop = 1  -> mountain (w) valley (wb)
    # sgnop = -1 -> mountain (w) mountain (wb)
    # flags if construction was interrupted

    # Initialization
    U = np.zeros((2 * cells + 1, 3))
    V = np.zeros((2 * cells + 1, 3))
    W = np.zeros((2 * cells, 3))

    U[0, :] = u
    V[0, :] = v

    # Be optimistic
    flag = 0

    # Start with the first w0
    turn = 0

    # period of w0
    period = np.size(w0, 0)

    # Start with a right-handed dent
    sgn = 1

    for n in range(0, 2 * cells):
        try:
            # Two steps for each row of new nodes
            z0 = w0[turn, :]
            turn = (turn + 1) % period

            # step 1: build new w
            uu, vv, ww = onestep(U[n, :], V[n, :], u0, v0, z0, rot=rot, sgn=sgn, roll=1)
            W[n, :] = ww

            # step 2: build new u and v
            uu, vv, ww = onestep(uu, vv, u0, v0, z0, rot=rot, sgn=-sgn, roll=-1)
            if sgn * np.dot(np.cross(uu, vv), ww) < 0.0:
                raise ValueError("basis reverted - paper in self-contact")
            else:
                U[n + 1, :] = uu
                V[n + 1, :] = vv

            # Next time

            # change into a valley or keep it a mountain
            sgn = -sgnop * sgn

        # Stop construction when the paper tears or self-contact
        except ValueError as e:
            print(e)
            print("after " + str(n) + " iterations")
            # make sure to keep a whole number of unit cells
            if n % 2:
                V = V[:n, :]
                U = U[:n, :]
                W = W[: n - 1, :]
            else:
                V = V[: n + 1, :]
                U = U[: n + 1, :]
                W = W[:n, :]
            # You can't always get what you want - MJ
            flag = 1
            break

    return U, V, W, flag


def integrate(
    U,
    V,
    W,
    N,
    rot=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    per=False,
):
    # Construct surface of revolution from fields U and V and rotation rot
    # N is the number of unit cells to be constructed along a parallel
    # per = False : first and last meridian are not identical
    # per = True  : first and last meridian are identical

    # Initialization
    c = np.shape(U)[0]
    X = np.zeros((2 * N + 1, c))
    Y = np.zeros((2 * N + 1, c))
    Z = np.zeros((2 * N + 1, c))

    # Integrate in the meridian direction
    for j in range(1, c):
        [X[0, j], Y[0, j], Z[0, j]] = [X[0, j - 1], Y[0, j - 1], Z[0, j - 1]] + W[
            j - 1, :
        ]

    # Initialize current (dynamic) meridian
    UU = np.copy(U).T
    VV = np.copy(V).T

    # Integrate in the parallel direction
    for i in range(1, N + 1):
        # Build next meridian (going through the midline of the cells)
        [X[2 * i - 1, :], Y[2 * i - 1, :], Z[2 * i - 1, :]] = [
            X[2 * i - 2, :],
            Y[2 * i - 2, :],
            Z[2 * i - 2, :],
        ] + UU
        # Update V
        VV = np.dot(rot, VV)
        # Build next meridian (going through the cells boundaries)
        [X[2 * i, :], Y[2 * i, :], Z[2 * i, :]] = [
            X[2 * i - 1, :],
            Y[2 * i - 1, :],
            Z[2 * i - 1, :],
        ] - VV
        # Update U
        UU = np.dot(rot, UU)

    if per:
        # Make sure ends meet
        [X[2 * N, :], Y[2 * N, :], Z[2 * N, :]] = [X[0, :], Y[0, :], Z[0, :]]

    return X, Y, Z


#########################################
# Tools to define initial conditions
#########################################
def zigzag(
    theta,
    u0=np.array([1.0, 0.0, 0.0]),
    v0=np.array([-1.0, 0.0, 0.0]),
    w0=np.array([1.0, 0.0, 0.0]),
    sgnop=1,
):
    # Defines a uniform boundary condition
    # characterized by a single dent (u,v) with an opening theta
    # parallel direction is (1,0,0)
    # tangent plane is normal to (0,0,1)
    # flags if initial condition is not feasible

    # this is ok ...
    u = la.norm(u0) * np.array([np.sin(theta / 2), np.cos(theta / 2), 0])
    v = la.norm(v0) * np.array([-np.sin(theta / 2), np.cos(theta / 2), 0])

    # ... but let's adjust the parallel direction and the tangent plane

    # Initialize unit cell
    U, V, W, flag = manysteps(u, v, u0, v0, w0, np.size(w0, 0), sgnop=sgnop)

    # Test if initial condition is feasible
    if flag:
        return u0, v0, flag

    # Define current, normalized, local basis
    t1 = u - v
    t1 = t1 / la.norm(t1)

    t2 = W[0, :] + W[1, :]
    t2 = t2 / la.norm(t2)

    n = np.cross(t1, t2)
    n = n / la.norm(n)

    # Complete the basis (t1, ?, n)
    t1p = np.cross(n, t1)

    # Adjust the dent
    u = np.array([np.dot(u, t1), np.dot(u, t1p), np.dot(u, n)])
    v = np.array([np.dot(v, t1), np.dot(v, t1p), np.dot(v, n)])

    return u, v, 0


def zigcircle(
    theta,
    beta,
    N=10,
    u0=np.array([1.0, 0.0, 0.0]),
    v0=np.array([-1.0, 0.0, 0.0]),
    w0=np.array([1.0, 0.0, 0.0]),
    sgnop=1,
):
    # Defines an axisymmetric boundary condition
    # characterized by a single dent (u,v) with an opening theta
    # N = nb of unit cells per meridian
    # beta = inclination of the meridian with respect to the axis of symmetry
    # returns the dent (u,v) and a rotation rot definining the symmetry
    # parallel is tangent to (1,0,0) at (0,0,0)
    # axis of symmetry is parallel to (0,1,0)
    # flags if initial condition is not feasible

    # Initialize dent
    u, v, flag = zigzag(theta, u0=u0, v0=v0, w0=w0, sgnop=sgnop)

    # Define local basis
    t1 = np.array([1.0, 0.0, 0.0])

    # Rotate dent by beta around t1
    u = (
        np.cos(beta) * (u - np.dot(u, t1) * t1)
        + np.sin(beta) * np.cross(t1, u)
        + np.dot(u, t1) * t1
    )
    v = (
        np.cos(beta) * (v - np.dot(v, t1) * t1)
        + np.sin(beta) * np.cross(t1, v)
        + np.dot(v, t1) * t1
    )

    # Define the symmetry of the pattern: rotation by 2pi/N about (0.,1.,0.)
    c = np.cos(2 * np.pi / N)
    s = np.sin(2 * np.pi / N)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    return u, v, rot, flag


# Test stuff locally
# if __name__ == "__main__":
# test code
# ...
