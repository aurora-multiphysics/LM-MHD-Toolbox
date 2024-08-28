import numpy as np


class HuntII:
    """Class for computing the Hunt-II analytic solution.

    This class computes analytic solutions to the Hunt-II case, originally derived in
    Hunt J C R 1965 J. Fluid Mech. 21 577-90, using the formulation detailed in
    Appendix A of Ni M J, Munipalli R, Huang P, Morley N B and Abdou M A 2007
    J. Comput. Phys. 227 205-28.

    In addition to being applicable to the Hunt-II case, in the limit of dB=0
    this solution generalises to the Shercliff solution for perfectly insulating
    Hartmann and side walls, originally derived in Shercliff J A 1953
    Math. Proc. Camb. Phil. Soc. 49 136-44

    The magnetic field is applied in the y direction, with Hartmann walls of length 2b
    and side walls of length 2a such that -b<x<b, -a<y<a. Flow is in the positive z
    direction, and the analytic solution gives the velocity and induced magnetic field
    z-components, with x and y components of both equal to zero, as well as the
    pressure drop in the z direction.
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
        x : list or numpy array
            x-axis points (-b<=x<=b, order ascending).
        y : list or numpy array
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
        # replace these later with a "grid" argument
        # or instead reorganise this to be single x, y points
        self.x = x
        self.y = y
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
            for n in range(len(self.xXi)):
                xi_val = self.xXi[n]
                vXi_k[n] = self._vXiComponent(vXi_constant, alpha_k, xi_val)
            hXi_k = vXi_k

            # compute yEta constant parts
            term2_k_constant = self._term2kConstant(r1_k, r2_k, N_k)
            term3_k_constant = self._term3kConstant(r1_k, r2_k, N_k)

            # compute yEta variable parts
            for n in range(len(self.yEta)):
                eta_val = self.yEta[n]

                v2 = self._v2k(r1_k, eta_val, term2_k_constant)
                v3 = self._v3k(r2_k, eta_val, term3_k_constant)
                vEta_k[n] = self._vEtak(v2, v3)

                h2 = self._h2k(r1_k, eta_val, term2_k_constant)
                h3 = self._h3k(r2_k, eta_val, term3_k_constant)
                hEta_k[n] = self._hEtak(h2, h3)

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
            num = (dB * r2_k + ((1 - np.exp(-2 * r2_k)) / (1 + np.exp(-2 * r2_k)))) / 2
            den = (((1 + np.exp(-2 * r1_k)) / 2) * dB * N_k) + (
                (1 + np.exp(-2 * (r1_k + r2_k))) / (1 + np.exp(-2 * r2_k))
            )
            term2_k_constant = num / den
            return term2_k_constant

    def _v2k(self, r1_k, eta_val, term2_k_constant):
        v2_k_variable = np.exp(-r1_k * (1 - eta_val)) + np.exp(-r1_k * (1 + eta_val))
        v2_k = term2_k_constant * v2_k_variable
        return v2_k

    def _h2k(self, r1_k, eta_val, term2_k_constant):
        h2_k_variable = np.exp(-r1_k * (1 - eta_val)) - np.exp(-r1_k * (1 + eta_val))
        h2_k = term2_k_constant * h2_k_variable
        return h2_k

    def _term3kConstant(self, r1_k, r2_k, N_k):
        dB = self.dB
        if dB == np.inf:
            term3_k_constant = r1_k / (N_k * (1 + np.exp(-2 * r2_k)))
            return term3_k_constant
        else:
            num = (dB * r1_k + ((1 - np.exp(-2 * r1_k)) / (1 + np.exp(-2 * r1_k)))) / 2
            den = (((1 + np.exp(-2 * r2_k)) / 2) * dB * N_k) + (
                (1 + np.exp(-2 * (r1_k + r2_k))) / (1 + np.exp(-2 * r1_k))
            )
            term3_k_constant = num / den
            return term3_k_constant

    def _v3k(self, r2_k, eta_val, term3_k_constant):
        v3_k_variable = np.exp(-r2_k * (1 - eta_val)) + np.exp(-r2_k * (1 + eta_val))
        v3_k = term3_k_constant * v3_k_variable
        return v3_k

    def _h3k(self, r2_k, eta_val, term3_k_constant):
        h3_k_variable = np.exp(-r2_k * (1 - eta_val)) - np.exp(-r2_k * (1 + eta_val))
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
        """Set a dimensional pressure drop to constrain scaling of analytic solution.

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
                "Must run analytic_solve before scaled results can be obtained."
            )
        if self._scaling_constraint:
            raise Exception(
                "Cannot set scaled_pressure_drop if "
                "scaled_average_velocity is already set"
            )
        self._scaling_constraint = "Set Pressure Drop"
        self.scaled_pressure_drop = scaled_pressure_drop
        self.scaled_average_velocity = (
            self.average_velocity * scaled_pressure_drop * (self.a**2 / self.dyn_visc)
        )

    def set_scaled_average_velocity(self, scaled_average_velocity):
        """Set a dimensional average velocity to constrain scaling of analytic solution.

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
                "Must run analytic_solve before scaled results can be obtained."
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
                "Must run analytic_solve before scaled results can be obtained."
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


def makeXYVectors(N_x, N_y, a, b):
    x = np.linspace(-b, b, N_x)
    y = np.linspace(-a, a, N_y)

    return x, y
