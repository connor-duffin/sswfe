import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

from swe import ShallowTwo


def compute_errors(nx):
    control = {"nx": nx, "dt": 0.0025, "theta": 1}
    swe = ShallowTwo(control)

    t = 0.
    nt_max = 10_000
    for i in range(nt_max):
        du_prev = swe.du.copy(deepcopy=True)
        u_prev, h_prev = du_prev.split()

        t += swe.dt
        swe.solve(t)

        u, h = swe.du.split()
        reached_steady_state = swe.steady_state(u, u_prev)
        if reached_steady_state:
            print(f"steady state reached in {i + 1} timesteps")
            break

    if (i + 1) == nt_max:
        print("steady state not necessarily reached")

    u_error = fe.errornorm(swe.u_exact, u)
    h_error = fe.errornorm(swe.h_exact, h)
    return u_error, h_error


if __name__ == "__main__":
    nx = [5, 10, 20]
    u_errors, h_errors = [], []
    for n in nx:
        u_error, h_error = compute_errors(nx=n)
        u_errors.append(u_error)
        h_errors.append(h_error)

    dx = np.array([1 / n for n in nx])
    u_errors = np.array(u_errors)
    h_errors = np.array(h_errors)

    # check rates of convergence
    from scipy.stats import linregress
    print(linregress(np.log(dx), np.log(u_errors)))
    print(linregress(np.log(dx), np.log(h_errors)))

    # and plot rates of convergence
    plt.loglog(dx, u_errors, ".-", label=r"$\Vert u_{\mathrm{exact}} - u \Vert_2$")
    plt.loglog(dx, h_errors, ".-", label=r"$\Vert h_{\mathrm{exact}} - h \Vert_2$")
    plt.xlabel(r"$\Delta x$")
    plt.ylabel(r"$L^2$ error")
    plt.legend()
    plt.savefig("figures/mms-convergence.png", dpi=600)
    plt.close()
