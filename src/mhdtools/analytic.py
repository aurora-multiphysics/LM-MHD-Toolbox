import numpy as np
from numpy import cosh, sinh, tanh


class HuntII:
    """Class for computing the Hunt-II analytic solution.

    This class computes analytic solutions to the Hunt-II case, originally
    derived in Hunt J C R 1965 J. Fluid Mech. 21 577-90, using the formulation
    detailed in Appendix A of
    Ni M J, Munipalli R, Huang P, Morley N B and Abdou M A 2007
    J. Comput. Phys. 227 205-28.

    In addition to being applicable to the Hunt-II case, in the limit of dB=0
    this solution generalises to the Shercliff solution for perfectly
    insulating Hartmann and side walls, originally derived in
    Shercliff J A 1953 Math. Proc. Camb. Phil. Soc. 49 136-44

    The magnetic field is applied in the y direction, with Hartmann walls of
    length 2b and side walls of length 2a such that -b<x<b, -a<y<a. Flow is in
    the positive z direction, and the analytic solution gives the velocity and
    induced magnetic field z-components, with x and y components of both equal
    to zero, as well as the pressure drop in the z direction.
    """

    def __init__(
        self,
        Ha,
        a,
        b,
        dB,
        num_k_points,
        x,
        y,
        dyn_visc,
        conductivity,
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
        dB : int, float or np.inf
            Hartmann wall conduction ratio, (t_w*sigma_w)/(a*sigma),
            where t_w is wall thickness, sigma_w is wall conductivity,
            and sigma is fluid conductivity.
        num_k_points : int
            Number of Fourier iterations.
        x : float or 1D numpy.ndarray
            x-axis points (-b<=x<=b, order ascending).
        y : float or 1D numpy.ndarray
            y-axis points (-a<=y<=a, order ascending).
        dyn_visc : float
            Dynamic viscosity.
        conductivity : float
            Fluid conductivity.
        permeability : float
            Fluid permeability.


        Returns
        -------
        None.

        """

        self.Ha = Ha
        self.a = a
        self.b = b
        self.dB = dB
        self.num_k_points = num_k_points
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
        self.conductivity = conductivity
        self.permeability = permeability

        self.l_ratio = self.b / self.a
        self.xXi = [x_val / self.a for x_val in self.x]
        self.yEta = [y_val / self.a for y_val in self.y]
        self.xyShape = (len(self.x), len(self.y))

        self._scaling_constraint = None
        self._solved = False

    def analytic_solve(self):
        """Calculate the nondimensional velocity, magnetic field,
        flow rate and average velocity."""
        k_list = list(range(0, self.num_k_points))
        v = np.zeros(self.xyShape)
        h = np.zeros(self.xyShape)
        Q = 0

        for k in k_list:
            # calculate term in V equation
            # calculate term in H equation
            # add terms to V and H equations

            # calculate common terms
            alpha_k = self._alphak(k)
            N_k = self._Nk(alpha_k)
            r1_k = self._rk(k, 1, alpha_k)
            r2_k = self._rk(k, 2, alpha_k)

            vXi_k = np.zeros(self.xyShape[0])
            vEta_k = np.zeros(self.xyShape[1])

            hXi_k = np.zeros(self.xyShape[0])
            hEta_k = np.zeros(self.xyShape[1])

            # compute xXi constant part
            vXi_constant = self._vXiConstant(k, alpha_k)

            # compute xXi variable part
            xi_arr = np.asarray(self.xXi)
            vXi_k = self._vXiComponent(vXi_constant, alpha_k, xi_arr)
            hXi_k = vXi_k

            # compute yEta constant parts
            term2_k_constant = self._term2kConstant(r1_k, r2_k, N_k)
            term3_k_constant = self._term3kConstant(r1_k, r2_k, N_k)

            # compute yEta variable parts
            eta_arr = np.asarray(self.yEta)

            v2 = self._v2k(r1_k, eta_arr, term2_k_constant)
            v3 = self._v3k(r2_k, eta_arr, term3_k_constant)
            vEta_k = self._vEtak(v2, v3)

            h2 = self._h2k(r1_k, eta_arr, term2_k_constant)
            h3 = self._h3k(r2_k, eta_arr, term3_k_constant)
            hEta_k = self._hEtak(h2, h3)

            # compute outer product of orthogonal components
            v_k = np.outer(vXi_k, vEta_k)
            h_k = np.outer(hXi_k, hEta_k)

            v += v_k
            h += h_k

            # calculate Q_k
            vXi_integral = self._vXiIntegral(alpha_k)
            v2_k_integral = self._v2kIntegral(r1_k)
            v3_k_integral = self._v3kIntegral(r2_k)
            Q_Xi_component = vXi_constant * vXi_integral
            Q_v2 = term2_k_constant * v2_k_integral
            Q_v3 = term3_k_constant * v3_k_integral
            Q_Eta_component = self._QEtak(Q_v2, Q_v3)
            Q_k = Q_Xi_component * Q_Eta_component

            Q += Q_k

        v = np.transpose(v)
        h = np.transpose(h)

        if self.single_point:
            v = float(v)
            h = float(h)

        self.v = v
        self.h = h
        self.Q = Q
        self.average_velocity = self.Q / (4 * self.l_ratio)
        self._solved = True

    def _alphak(self, k):
        alpha_k = (k + 0.5) * np.pi / self.l_ratio
        return alpha_k

    def _Nk(self, alpha_k):
        N_k = np.sqrt(self.Ha**2 + 4 * alpha_k**2)
        return N_k

    def _rk(self, k, pm, alpha_k):

        if pm == 1:
            m = 1
        elif pm == 2:
            m = -1
        else:
            raise ValueError("pm must be 1 or 2")

        r_k = 0.5 * (m * self.Ha + np.sqrt(self.Ha**2 + 4 * alpha_k**2))
        return r_k

    def _vXiConstant(self, k, alpha_k):
        vXi_constant = 2 * ((-1) ** k) / (self.l_ratio * alpha_k**3)
        return vXi_constant

    def _vXiComponent(self, vXi_constant, alpha_k, xi_val):
        vXi_variable = np.cos(alpha_k * xi_val)
        vXi_component = vXi_constant * vXi_variable
        return vXi_component

    def _vEtak(self, v2, v3):
        vEta_component_k = 1 - v2 - v3
        return vEta_component_k

    def _hEtak(self, h2, h3):
        hEta_component_k = h2 - h3
        return hEta_component_k

    def _term2kConstant(self, r1_k, r2_k, N_k):
        dB = self.dB
        if dB == np.inf:
            term2_k_constant = r2_k / (N_k * (1 + np.exp(-2 * r1_k)))
            return term2_k_constant
        else:
            num = (
                dB * r2_k + ((1 - np.exp(-2 * r2_k)) / (1 + np.exp(-2 * r2_k)))
            ) / 2
            den = (((1 + np.exp(-2 * r1_k)) / 2) * dB * N_k) + (
                (1 + np.exp(-2 * (r1_k + r2_k))) / (1 + np.exp(-2 * r2_k))
            )
            term2_k_constant = num / den
            return term2_k_constant

    def _v2k(self, r1_k, eta_val, term2_k_constant):
        v2_k_variable = np.exp(-r1_k * (1 - eta_val)) + np.exp(
            -r1_k * (1 + eta_val)
        )
        v2_k = term2_k_constant * v2_k_variable
        return v2_k

    def _h2k(self, r1_k, eta_val, term2_k_constant):
        h2_k_variable = np.exp(-r1_k * (1 - eta_val)) - np.exp(
            -r1_k * (1 + eta_val)
        )
        h2_k = term2_k_constant * h2_k_variable
        return h2_k

    def _term3kConstant(self, r1_k, r2_k, N_k):
        dB = self.dB
        if dB == np.inf:
            term3_k_constant = r1_k / (N_k * (1 + np.exp(-2 * r2_k)))
            return term3_k_constant
        else:
            num = (
                dB * r1_k + ((1 - np.exp(-2 * r1_k)) / (1 + np.exp(-2 * r1_k)))
            ) / 2
            den = (((1 + np.exp(-2 * r2_k)) / 2) * dB * N_k) + (
                (1 + np.exp(-2 * (r1_k + r2_k))) / (1 + np.exp(-2 * r1_k))
            )
            term3_k_constant = num / den
            return term3_k_constant

    def _v3k(self, r2_k, eta_val, term3_k_constant):
        v3_k_variable = np.exp(-r2_k * (1 - eta_val)) + np.exp(
            -r2_k * (1 + eta_val)
        )
        v3_k = term3_k_constant * v3_k_variable
        return v3_k

    def _h3k(self, r2_k, eta_val, term3_k_constant):
        h3_k_variable = np.exp(-r2_k * (1 - eta_val)) - np.exp(
            -r2_k * (1 + eta_val)
        )
        h3_k = term3_k_constant * h3_k_variable
        return h3_k

    def _vXiIntegral(self, alpha_k):
        v_Xi_integral = (2 / alpha_k) * np.sin(alpha_k * self.l_ratio)
        return v_Xi_integral

    def _v2kIntegral(self, r1_k):
        v2_k_integral = (2 / r1_k) * (1 - np.exp(-2 * r1_k))
        return v2_k_integral

    def _v3kIntegral(self, r2_k):
        v3_k_integral = (2 / r2_k) * (1 - np.exp(-2 * r2_k))
        return v3_k_integral

    def _QEtak(self, Q_v2, Q_v3):
        Q_Eta_component_k = 2 - Q_v2 - Q_v3
        return Q_Eta_component_k

    def set_scaled_pressure_drop(self, scaled_pressure_drop):
        """Set a dimensional pressure drop
        to constrain scaling of analytic solution.

        Parameters
        ----------
        scaled_pressure_drop : float
            Dimensional pressure drop (-dp/dz), e.g. in simulation units.

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
                "Cannot set scaled_pressure_drop if "
                "scaled_average_velocity is already set"
            )
        self._scaling_constraint = "Set Pressure Drop"
        self.scaled_pressure_drop = scaled_pressure_drop
        self.scaled_average_velocity = (
            self.average_velocity
            * scaled_pressure_drop
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
            If scaled_pressure_drop has been set
        """
        if not self._solved:
            raise Exception(
                "Must run analytic_solve before"
                "scaled results can be obtained."
            )

        if self._scaling_constraint:
            raise Exception(
                "Cannot set scaled_average_velocity if "
                "scaled_pressure_drop is already set"
            )
        self._scaling_constraint = "Set Average Velocity"
        self.scaled_average_velocity = scaled_average_velocity
        self.scaled_pressure_drop = (
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
                "or set_scaled_pressure_drop has been called."
            )
        self._calculate_scaled_velocity()
        self._calculate_scaled_H_field()
        self._calculate_scaled_B_field()

    def _calculate_scaled_velocity(self):
        self.scaled_velocity_z = (
            self.v * self.scaled_pressure_drop * (self.a**2) / self.dyn_visc
        )

    def _calculate_scaled_H_field(self):
        self.scaled_H_field_z = (
            self.h
            * self.scaled_pressure_drop
            * self.a**2
            * np.sqrt(self.conductivity / self.dyn_visc)
        )

    def _calculate_scaled_B_field(self):
        self.scaled_B_field_z = self.scaled_H_field_z * self.permeability


class Sloan:
    """Class for computing the Sloan and Smith 1966 analytic solution,
    rewritten in a stable exponential form.

    This class computes analytic solutions to a Hunt-II-like case, extended to
    the case of thick conducting walls as derived in
    Sloan D M and Smith P 1966 J. Appl. Math. Mech. 46 (7) 439-443.
    The solution has been reformulated, eliminating hyperbolic functions which
    explode as Hartmann number and Fourier iterations increase.

    In addition to thick wall cases, this solution reproduces the Hunt-II
    solution for either thin walls or perfectly conducting walls, as derived in
    Hunt J C R 1965 J. Fluid Mech. 21 577-90, and also in
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
            w_n = k_n * np.outer(wXi_n, wEta_n)
            B_n = k_n * np.outer(BXi_n, BEta_n)

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

        w = np.transpose(w)
        B = np.transpose(B)

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
            - (E_n / beta_n) * (np.exp(2 * beta_n) - 1)
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

    def _calculate_scaled_velocity(self):
        self.scaled_velocity_z = (
            self.w * self.scaled_pressure_grad * (self.a**2) / self.dyn_visc
        )

    def _calculate_scaled_B_field(self):
        self.scaled_B_field_z = (
            self.B
            * self.scaled_pressure_grad
            * self.a**2
            * np.sqrt(self.conductivity_f / self.dyn_visc)
            * self.permeability
        )


class Sloan_66_original:
    """Class for computing the Sloan and Smith 1966 analytic solution.
    This class is included as an implementation of the original form of the
    solution, however the exponential form implemented in the Sloan class
    provides a stable solution beyond the limited range of Hartmann numbers
    for which this class works.

    This class computes analytic solutions to a Hunt-II-like case, extended to
    the case of thick conducting walls as derived in
    Sloan D M and Smith P 1966 J. Appl. Math. Mech. 46 (7) 439-443.

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
            y-axis points (-a<=y<=a, order ascending).
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
            # # calculate term in V equation
            # # calculate term in H equation
            # # add terms to V and H equations

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
            Q_n = self._Qn(D_n, E_n, alpha_n, beta_n)
            Eta_n_solid = self._Etan_solid(D_n, E_n, alpha_n, beta_n, a_n)

            eta_arr = np.asarray(self.yEta)
            wEta_n = self._wEtaComponent(
                eta_arr, D_n, E_n, alpha_n, beta_n, Q_n
            )
            BEta_n = self._BEtaComponent(
                eta_arr,
                D_n,
                E_n,
                alpha_n,
                beta_n,
                Q_n,
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
                D_n, E_n, alpha_n, beta_n, Q_n
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
        alpha_n = 0.5 * (-self.Ha + np.sqrt(self.Ha**2 + 4 * a_n**2))
        return alpha_n

    def _betan(self, a_n):
        beta_n = 0.5 * (-self.Ha - np.sqrt(self.Ha**2 + 4 * a_n**2))
        return beta_n

    def _XiComponent(self, a_n, xi_arr):
        return np.cos(a_n * xi_arr)

    def _Dn(self, a_n, beta_n):
        if self.conductivity_w is np.inf:
            D_n = None
        else:
            sigma_1 = self.conductivity_f
            sigma_2 = self.conductivity_w
            D_n = sigma_1 * a_n * sinh(beta_n) - sigma_2 * beta_n * tanh(
                a_n * (1 - self.q)
            ) * cosh(beta_n)
        return D_n

    def _En(self, a_n, alpha_n):
        if self.conductivity_w is np.inf:
            E_n = None
        else:
            sigma_1 = self.conductivity_f
            sigma_2 = self.conductivity_w
            E_n = sigma_1 * a_n * sinh(alpha_n) - sigma_2 * alpha_n * tanh(
                a_n * (1 - self.q)
            ) * cosh(alpha_n)
        return E_n

    def _Qn(self, D_n, E_n, alpha_n, beta_n):
        if self.conductivity_w is np.inf:
            Q_n = (beta_n - alpha_n) * cosh(alpha_n) * cosh(beta_n)
        else:
            Q_n = D_n * cosh(alpha_n) - E_n * cosh(beta_n)
        return Q_n

    def _Etan_solid(self, D_n, E_n, alpha_n, beta_n, a_n):
        if self.conductivity_w is np.inf:
            numer = alpha_n * sinh(beta_n) * cosh(alpha_n) - beta_n * sinh(
                alpha_n
            ) * cosh(beta_n)
        else:
            numer = E_n * sinh(beta_n) - D_n * sinh(alpha_n)
        solid_factor = numer / sinh(a_n * (1 - self.q))
        return solid_factor

    def _wEtaComponent(self, eta_arr, D_n, E_n, alpha_n, beta_n, Q_n):
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
            if self.conductivity_w is np.inf:
                numer = beta_n * cosh(beta_n) * cosh(
                    alpha_n * eta_arr[fluid_mask]
                ) - alpha_n * cosh(alpha_n) * cosh(
                    beta_n * eta_arr[fluid_mask]
                )
            else:
                numer = D_n * cosh(alpha_n * eta_arr[fluid_mask]) - E_n * cosh(
                    beta_n * eta_arr[fluid_mask]
                )
            wEtaComponent[fluid_mask] = 1 - (numer / Q_n)

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
        Q_n,
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
            if self.conductivity_w is np.inf:
                numer[fluid_mask] = alpha_n * cosh(alpha_n) * sinh(
                    beta_n * eta_arr[fluid_mask]
                ) - beta_n * cosh(beta_n) * sinh(alpha_n * eta_arr[fluid_mask])
            else:
                numer[fluid_mask] = E_n * sinh(
                    beta_n * eta_arr[fluid_mask]
                ) - D_n * sinh(alpha_n * eta_arr[fluid_mask])

        if np.any(solid_top_mask | solid_bot_mask):
            if np.any(solid_top_mask):
                pm[solid_top_mask] = -1  # h_n
            if np.any(solid_bot_mask):
                pm[solid_bot_mask] = 1  # s_n

            numer[solid_top_mask | solid_bot_mask] = Eta_n_solid * sinh(
                a_n
                * (
                    eta_arr[solid_top_mask | solid_bot_mask]
                    + (pm[solid_top_mask | solid_bot_mask] * self.q)
                )
            )

        BEtaComponent = numer / Q_n
        return BEtaComponent

    def _wEtaIntegral(self, D_n, E_n, alpha_n, beta_n, Q_n):
        numer = ((D_n / alpha_n) * sinh(alpha_n)) - (
            (E_n / beta_n) * sinh(beta_n)
        )
        wEtaIntegral = 1 - (numer / Q_n)
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

    def _calculate_scaled_velocity(self):
        self.scaled_velocity_z = (
            self.w * self.scaled_pressure_grad * (self.a**2) / self.dyn_visc
        )

    def _calculate_scaled_B_field(self):
        self.scaled_B_field_z = (
            self.B
            * self.scaled_pressure_grad
            * self.a**2
            * np.sqrt(self.conductivity_f / self.dyn_visc)
            * self.permeability
        )


def makeXYVectors(N_x, N_y, a, b):
    x = np.linspace(-b, b, N_x)
    y = np.linspace(-a, a, N_y)

    return x, y
