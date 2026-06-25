import numpy as np

class Sloan:
    """Class for computing the Sloan and Smith 1966 analytic solution.

    This class computes analytic solutions to a Hunt-II-like case, extended to
    the case of thick conducting walls as derived in
    Sloan D M and Smith P 1966 J. Appl. Math. Mech. 46 (7) 439-443.
    The solution has been reformulated, eliminating hyperbolic functions which
    explode as Hartmann number and Fourier iterations increase.

    In addition to being applicable to the Hunt-II case with thick walls, in
    the limit of wall thickness or wall conductivity tending to zero the
    Shercliff solution for perfectly insulating Hartmann and side walls,
    originally derived in
    Shercliff J A 1953 Math. Proc. Camb. Phil. Soc. 49 136-44,
    is recovered.

    The magnetic field is applied in the y direction, with Hartmann walls of
    length 2b and side walls of length 2a such that -b<x<b, -a<y<a. Flow is in
    the positive z direction, and the analytic solution gives the velocity and
    induced magnetic field z-components, with x and y components of both equal
    to zero, as well as the pressure drop in the z direction.

    Note that this solver may be unable to obtain solutions in some regimes,
    particularly observed for Ha>~500. This is due to the solution including
    many hyperbolic functions which tend to infinity as their arguments
    increase.
    """

    def __init__(
        self,
        Ha,
        a,
        b,
        t_w,
        truncation,
        x,
        y,
        dyn_visc,
        conductivity_f,
        conductivity_w,
        permeability,
    ):
        """Constructs the necessary attributes for the HuntII object.

        Parameters
        ----------
        Ha : int or float
            Hartmann Number.
        a : int or float
            Side wall half-height (y-axis).
        b : int or float
            Hartmann wall half-width (x-axis).
        t_w : int or float
            Hartmann wall thickness (x-axis).
        truncation : int
            Number of iterations after which the Fourier series is truncated.
        x : float or 1D numpy.ndarray
            x-axis points (-b<=x<=b, order ascending).
        y : float or 1D numpy.ndarray
            y-axis points (-a-t_w<=y<=a+t_w, order ascending).
        dyn_visc : float
            Dynamic viscosity.
        conductivity_f : float
            Fluid conductivity.
        conductivity_w : float
            Hartmann wall conductivity
        permeability : float
            Fluid permeability.


        Returns
        -------
        None.

        """

        # Above: should change y to run from -(a+t_w) to (a+t_w),
        # and implement B_2 solution

        self.Ha = Ha
        self.a = a
        self.b = b
        self.q = 1 + t_w / a  # a+t_w=a*q, so q=(a+t_w)/a = 1+t_w/a
        self.truncation = truncation
        if isinstance(x, (float, np.floating)) and isinstance(
            x, (float, np.floating)
        ):
            self.x = [x]
            self.y = [y]
            self.single_point = True
        elif type(x) is not np.ndarray or type(y) is not np.ndarray:
            raise Exception(
                "Either both x and y must be numpy arrays, "
                "or both must be floats"
            )
        else:
            self.x = x
            self.y = y
            self.single_point = False
        self.dyn_visc = dyn_visc
        self.conductivity_w = conductivity_w
        self.conductivity_f = conductivity_f
        self.permeability = permeability

        self.r = self.b / self.a
        self.xXi = [x_val / self.a for x_val in self.x]
        self.yEta = [y_val / self.a for y_val in self.y]
        self.xyShape = (len(self.x), len(self.y))

        self._scaling_constraint = None
        self._solved = False

    def analytic_solve(self):
        """Calculate the nondimensional velocity, magnetic field,
        flow rate and average velocity."""
        n_list = list(range(0, self.truncation))
        w = np.zeros(self.xyShape)
        B = np.zeros(self.xyShape)
        Q = 0

        for n in n_list:
            # calculate term in w (velocity) equation
            # calculate term in B (induced magnetic field) equation
            # add terms to w and B equations

            # calculate common terms
            a_n = self._an(n)
            k_n = self._kn(n, a_n)
            alpha_n = self._alphan(a_n)
            beta_n = self._betan(a_n)

            wXi_n = np.zeros(self.xyShape[0])
            wEta_n = np.zeros(self.xyShape[1])

            BXi_n = np.zeros(self.xyShape[0])
            BEta_n = np.zeros(self.xyShape[1])

            xi_arr = np.asarray(self.xXi)
            wXi_n = self._XiComponent(a_n, xi_arr)
            BXi_n = wXi_n

            # compute yEta constant parts
            D_n = self._Dn(a_n, beta_n)
            E_n = self._En(a_n, alpha_n)
            P_n = self._Pn(D_n, E_n, alpha_n, beta_n)
            Eta_n_solid = self._Etan_solid(D_n, E_n, alpha_n, beta_n, a_n)

            eta_arr = np.asarray(self.yEta)
            wEta_n = self._wEtaComponent(
                eta_arr, D_n, E_n, alpha_n, beta_n, P_n
            )
            BEta_n = self._BEtaComponent(
                eta_arr,
                D_n,
                E_n,
                alpha_n,
                beta_n,
                P_n,
                a_n,
                Eta_n_solid,
            )

            # compute iteration
            w_n = k_n * np.outer(wEta_n, wXi_n)
            B_n = k_n * np.outer(BEta_n, BXi_n)

            # add to series
            w += w_n
            B += B_n

            # calculate Q_k

            Q_Eta_component = self._wEtaIntegral(
                D_n, E_n, alpha_n, beta_n, P_n
            )
            Q_Xi_component = self._wXiIntegral(a_n)
            Q_k = (4 * k_n / a_n) * Q_Eta_component * Q_Xi_component

            Q += Q_k

        if self.single_point:
            w = float(w)
            B = float(B)

        self.w = w
        self.B = B
        self.Q = Q
        self.average_velocity = self.Q / (4 * self.r)
        self._solved = True

    def _an(self, n):
        a_n = (n + 0.5) * np.pi / self.r
        return a_n

    def _kn(self, n, a_n):
        k_n = 2 * ((-1) ** n) / (self.r * a_n**3)
        return k_n

    def _alphan(self, a_n):
        alpha_n = 2 * a_n**2 / (self.Ha + np.sqrt(self.Ha**2 + 4 * a_n**2))
        return alpha_n

    def _betan(self, a_n):
        beta_n = -self.Ha - self._alphan(a_n)
        return beta_n

    def _XiComponent(self, a_n, xi_arr):
        return np.cos(a_n * xi_arr)

    def _tanh_cancels(self, dummy):
        # If walls are perfectly conducting, tanh terms cancel
        return 1

    def _Dn(self, a_n, beta_n):
        if self.conductivity_w is np.inf:
            sigma_f = 0  # terms in sigma_f are negligible
            sigma_w = (
                1  # terms in sigma_w dominate and then all cancel in ratios
            )
            tanh = (
                self._tanh_cancels
            )  # and the tanh functions also cancel in ratios
        else:
            sigma_f = self.conductivity_f
            sigma_w = self.conductivity_w
            tanh = np.tanh

        D_n = sigma_f * a_n * (
            np.exp(2 * beta_n) - 1
        ) + sigma_w * beta_n * tanh(a_n * (self.q - 1)) * (
            np.exp(2 * beta_n) + 1
        )
        return D_n

    def _En(self, a_n, alpha_n):
        if self.conductivity_w is np.inf:
            sigma_f = 0  # terms in sigma_f are negligible
            sigma_w = (
                1  # terms in sigma_w dominate and then all cancel in ratios
            )
            tanh = (
                self._tanh_cancels
            )  # and the tanh functions also cancel in ratios
        else:
            sigma_f = self.conductivity_f
            sigma_w = self.conductivity_w
            tanh = np.tanh

        E_n = sigma_f * a_n * (
            1 - np.exp(-2 * alpha_n)
        ) + sigma_w * alpha_n * tanh(a_n * (self.q - 1)) * (
            1 + np.exp(-2 * alpha_n)
        )
        return E_n

    def _Pn(self, D_n, E_n, alpha_n, beta_n):
        P_n = 0.5 * (
            D_n * (1 + np.exp(-2 * alpha_n)) - E_n * (1 + np.exp(2 * beta_n))
        )
        return P_n

    def _Etan_solid(self, D_n, E_n, alpha_n, beta_n, a_n):
        numer = 0.5 * (
            D_n * (np.exp(-2 * alpha_n) - 1) + E_n * (np.exp(2 * beta_n) - 1)
        )
        solid_factor = numer / np.sinh(a_n * (1 - self.q))
        return solid_factor

    def _wEtaComponent(self, eta_arr, D_n, E_n, alpha_n, beta_n, P_n):
        fluid_mask = (eta_arr >= -1) & (eta_arr <= 1)
        solid_mask = ((eta_arr > 1) & (eta_arr <= self.q)) | (
            (eta_arr >= -self.q) & (eta_arr < -1)
        )

        if not np.all(fluid_mask | solid_mask):
            bad = eta_arr[~(fluid_mask | solid_mask)]
            raise ValueError(f"eta values outside defined domain: {bad}")

        numer = np.zeros_like(eta_arr, dtype=np.float64)
        wEtaComponent = np.zeros_like(eta_arr, dtype=np.float64)

        if np.any(fluid_mask):
            # fluid
            numer = (D_n / 2) * (
                np.exp(-alpha_n * (1 - eta_arr[fluid_mask]))
                + np.exp(-alpha_n * (1 + eta_arr[fluid_mask]))
            ) - (E_n / 2) * (
                np.exp(beta_n * (1 - eta_arr[fluid_mask]))
                + np.exp(beta_n * (1 + eta_arr[fluid_mask]))
            )
            wEtaComponent[fluid_mask] = 1 - (numer / P_n)

        if np.any(solid_mask):
            # solid
            wEtaComponent[solid_mask] = 0

        return wEtaComponent

    def _BEtaComponent(
        self,
        eta_arr,
        D_n,
        E_n,
        alpha_n,
        beta_n,
        P_n,
        a_n,
        Eta_n_solid,
    ):
        fluid_mask = (eta_arr >= -1) & (eta_arr <= 1)
        solid_top_mask = (eta_arr > 1) & (eta_arr <= self.q)
        solid_bot_mask = (eta_arr >= -self.q) & (eta_arr < -1)

        if not np.all(fluid_mask | solid_top_mask | solid_bot_mask):
            bad = eta_arr[~(fluid_mask | solid_top_mask | solid_bot_mask)]
            raise ValueError(f"eta values outside defined domain: {bad}")

        numer = np.zeros_like(eta_arr, dtype=np.float64)
        pm = np.zeros_like(eta_arr, dtype=np.float64)

        if np.any(fluid_mask):
            # fluid
            numer[fluid_mask] = (D_n / 2) * (
                -np.exp(-alpha_n * (1 - eta_arr[fluid_mask]))
                + np.exp(-alpha_n * (1 + eta_arr[fluid_mask]))
            ) + (E_n / 2) * (
                -np.exp(beta_n * (1 - eta_arr[fluid_mask]))
                + np.exp(beta_n * (1 + eta_arr[fluid_mask]))
            )

        if np.any(solid_top_mask | solid_bot_mask):
            if np.any(solid_top_mask):
                pm[solid_top_mask] = -1  # h_n
            if np.any(solid_bot_mask):
                pm[solid_bot_mask] = 1  # s_n

            numer[solid_top_mask | solid_bot_mask] = Eta_n_solid * np.sinh(
                a_n
                * (
                    eta_arr[solid_top_mask | solid_bot_mask]
                    + (pm[solid_top_mask | solid_bot_mask] * self.q)
                )
            )

        BEtaComponent = numer / P_n
        return BEtaComponent

    def _wEtaIntegral(self, D_n, E_n, alpha_n, beta_n, P_n):
        numer = 0.5 * (
            (D_n / alpha_n) * (1 - np.exp(-2 * alpha_n))
            + (E_n / beta_n) * (np.exp(2 * beta_n) - 1)
        )
        wEtaIntegral = 1 - (numer / P_n)
        return wEtaIntegral

    def _wXiIntegral(self, a_n):
        wXiIntegral = np.sin(a_n * self.r)
        return wXiIntegral

    def set_scaled_pressure_grad(self, scaled_pressure_grad):
        """Set a dimensional pressure gradient to
        constrain scaling of analytic solution.

        Parameters
        ----------
        scaled_pressure_grad : float
            Dimensional pressure grad (-dp/dz), e.g. in simulation units.

        Raises
        ------
        Exception
            If analytic_solve has not been run
        Exception
            If scaled_average_velocity has been set
        """
        if not self._solved:
            raise Exception(
                "Must run analytic_solve before"
                "scaled results can be obtained."
            )
        if self._scaling_constraint:
            raise Exception(
                "Cannot set scaled_pressure_grad if "
                "scaled_average_velocity is already set"
            )
        self._scaling_constraint = "Set Pressure Gradient"
        self.scaled_pressure_grad = scaled_pressure_grad
        self.scaled_average_velocity = (
            self.average_velocity
            * scaled_pressure_grad
            * (self.a**2 / self.dyn_visc)
        )

    def set_scaled_average_velocity(self, scaled_average_velocity):
        """Set a dimensional average velocity
        to constrain scaling of analytic solution.

        Parameters
        ----------
        scaled_average_velocity : float
            Dimensional average velocity, e.g. in simulation units.

        Raises
        ------
        Exception
            If analytic_solve has not been run
        Exception
            If scaled_pressure_grad has been set
        """

        if not self._solved:
            raise Exception(
                "Must run analytic_solve before"
                "scaled results can be obtained."
            )

        if self._scaling_constraint:
            raise Exception(
                "Cannot set scaled_average_velocity if "
                "scaled_pressure_grad is already set"
            )
        self._scaling_constraint = "Set Average Velocity"
        self.scaled_average_velocity = scaled_average_velocity
        self.scaled_pressure_grad = (
            self.scaled_average_velocity / self.average_velocity
        ) * (self.dyn_visc / self.a**2)

    def calculate_scaled_solution(self):
        """Calculate scaled solution a previously set scaling constraint.

        Raises
        ------
        Exception
            If analytic_solve has not been run
        Exception
            If scaling constraint not set
        """
        if not self._solved:
            raise Exception(
                "Must run analytic_solve before"
                "scaled results can be obtained."
            )
        if not self._scaling_constraint:
            raise Exception(
                "Cannot calculate scaled fields until either "
                "set_scaled_average_velocity "
                "or set_scaled_pressure_grad has been called."
            )
        self._calculate_scaled_velocity()
        self._calculate_scaled_B_field()
        self._calculate_scaled_H_field()

    def _calculate_scaled_velocity(self):
        self.scaled_velocity_z = (
            self.w * self.scaled_pressure_grad * (self.a**2) / self.dyn_visc
        )

        # unsure about this

    def _calculate_scaled_B_field(self):
        self.scaled_H_field_z = (
            self.B
            * self.scaled_pressure_grad
            * self.a**2
            * np.sqrt(self.conductivity_f / self.dyn_visc)
        )

    # unsure about this
    def _calculate_scaled_H_field(self):
        self.scaled_B_field_z = self.scaled_H_field_z / self.permeability


def makeXYVectors(N_x, N_y, a, b):
    x = np.linspace(-b, b, N_x)
    y = np.linspace(-a, a, N_y)

    return x, y
