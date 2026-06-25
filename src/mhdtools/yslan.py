import numpy as np

def _chk_var(name, value, orEqual=False):
    if orEqual:
        assert value >= 0, f'{name} must >= 0, {value}'
    else:
        assert value > 0, f'{name} must > 0, {value}'
    return value

# Follow Appendix A of [Ni et. al, JCP 2007]
class Hunt2:
    def __init__(self, Ha, dB, a=1, b=1, debug=False):
        _chk_var('Ha', Ha, True)
        _chk_var('dB', dB, True)
        _chk_var('a', a)
        _chk_var('b', b)

        self.Ha = Ha

        self.a = a   # 2a = dist(A-A) between side walls
        self.b = b   # 2b = dist(B-B) between hartmann walls
        self.ell = b / a

        bigNum = 1e20
        if np.isinf(dB) or dB > bigNum:
            dB = np.inf
            self.perfect = True
        else:
            self.perfect = False
        self.dB = dB # conductivity at B-B


    def _comp_xi(self, x): # [-1, 1]
        return x / self.a
    def _comp_eta(self, y): # [-ell, ell]
        return y / self.a

    def _comp_alpha_k(self, k):
        return (k + 0.5) * np.pi / self.ell

    def _comp_N(self, alpha_k): # eq 73
        return np.sqrt(self.Ha*self.Ha + 4*alpha_k*alpha_k)

    def _comp_rk(self, pm, N_k):
        return 0.5*(pm * self.Ha + N_k)

    # exp: only the exp part, leave scale for the later limiting cases
    def _comp_exp_num_termL(self, rk):
        return (1.0 - np.exp(-2*rk)) / (1.0 + np.exp(-2*rk))

    def _comp_exp_num_termR(self, rk, sgn, eta):
        return ( np.exp(-rk*(1.0-eta)) + sgn * np.exp(-rk*(1.0+eta)) ) / 2.0

    def _comp_exp_den_termL(self, rk):
        return (1.0 + np.exp(-2*rk)) / 2.0

    def _comp_exp_den_termR(self, rk_a, rk_b): # rk_b goes to denumerator
        return (1.0 + np.exp(-2.0*(rk_a+rk_b))) / (1.0 + np.exp(-2.0*rk_b))

    def _comp_VH2(self, N_k, alpha_k, r1k, r2k, sgn, eta): # H is basically V but with minus  sgn
        num_R = self._comp_exp_num_termR(r1k, sgn, eta)
        den_L = self._comp_exp_den_termL(r1k)

        if self.perfect:
            return r2k * num_R / (N_k * den_L)
        else:
            num_L = self._comp_exp_num_termL(r2k)
            den_R = self._comp_exp_den_termR(r1k, r2k)
            return (self.dB * r2k + num_L) * num_R / (self.dB * N_k * den_L + den_R)

    def _comp_VH3(self, N_k, alpha_k, r1k, r2k, sgn, eta): # this is basically V2 but swap r1k and r2k
        num_R = self._comp_exp_num_termR(r2k, sgn, eta)
        den_L = self._comp_exp_den_termL(r2k)

        if self.perfect:
            return r1k * num_R / (N_k * den_L)
        else:
            num_L = self._comp_exp_num_termL(r1k)
            den_R = self._comp_exp_den_termR(r2k, r1k)
            return (self.dB * r1k + num_L) * num_R / (self.dB * N_k * den_L + den_R)

    def _comp_cosXi(self, k, alpha_k, xi):
        return 2.0 * np.power(-1, k) * np.cos(alpha_k*xi) / (self.ell * np.power(alpha_k,3))


    def _comp_vk(self, N_k, alpha_k, r1k, r2k, xi_k, eta):
        sgn = 1.0
        V2 = self._comp_VH2(N_k, alpha_k, r1k, r2k, sgn, eta)
        V3 = self._comp_VH3(N_k, alpha_k, r1k, r2k, sgn, eta)
        return xi_k * (1.0 - V2 - V3)
        
    def _comp_hk(self, N_k, alpha_k, r1k, r2k, xi_k, eta):
        sgn = -1.0
        H2 = self._comp_VH2(N_k, alpha_k, r1k, r2k, sgn, eta)
        H3 = self._comp_VH3(N_k, alpha_k, r1k, r2k, sgn, eta)
        return xi_k * (H2 - H3)

    def V(self, X, Y, num_k = 100):
        xi = X / self.a
        eta = Y / self.a
        V = np.zeros_like(xi)
        for k in range(num_k):
            alpha_k = self._comp_alpha_k(k)
            N_k = self._comp_N(alpha_k)
            r1k = self._comp_rk( 1, N_k)
            r2k = self._comp_rk(-1, N_k)
            xi_k = self._comp_cosXi(k, alpha_k, xi)

            vk = self._comp_vk(N_k, alpha_k, r1k, r2k, xi_k, eta)
            V += vk
        return V

    def H(self, X, Y, num_k = 100):
        xi = X / self.a
        eta = Y / self.a
        H = np.zeros_like(xi)
        for k in range(num_k):
            alpha_k = self._comp_alpha_k(k)
            N_k = self._comp_N(alpha_k)
            r1k = self._comp_rk( 1, N_k)
            r2k = self._comp_rk(-1, N_k)
            xi_k = self._comp_cosXi(k, alpha_k, xi)

            hk = self._comp_hk(N_k, alpha_k, r1k, r2k, xi_k, eta)
            H += hk
        return  H

    def _VH(self, xx, yy, num_k = 100): # fast tensor prod of 1d arrs
        xi = xx / self.a
        eta = yy / self.a

        sz = (len(yy), len(xx))
        V = np.zeros(sz)
        H = np.zeros(sz)
        for k in range(num_k):
            alpha_k = self._comp_alpha_k(k)
            N_k = self._comp_N(alpha_k)
            r1k = self._comp_rk( 1, N_k)
            r2k = self._comp_rk(-1, N_k)
            xi_k = self._comp_cosXi(k, alpha_k, xi)

            v_eta_k = self._comp_vk(N_k, alpha_k, r1k, r2k, 1.0, eta)
            h_eta_k = self._comp_hk(N_k, alpha_k, r1k, r2k, 1.0, eta)
            vk = np.outer(v_eta_k, xi_k)
            hk = np.outer(h_eta_k, xi_k)
            V += vk
            H += hk
        return V, H

    def VH(self, X, Y, num_k = 100, tensor=True):
        if tensor:
            assert isinstance(X, np.ndarray) and X.ndim == 1 \
               and isinstance(Y, np.ndarray) and Y.ndim == 1, 'invalid size'
            return self._VH(X, Y, num_k)
        else:
            X = np.array(X)
            Y = np.array(Y)
            xi = X / self.a
            eta = Y / self.a
            
            if isinstance(X, np.ndarray) and X.size > 1:
                V = np.zeros_like(xi)
                H = np.zeros_like(xi)
            elif isinstance(Y, np.ndarray) and Y.size > 1:
                V = np.zeros_like(eta)
                H = np.zeros_like(eta)
            else:
                V, H = 0.0, 0.0

            for k in range(num_k):
                alpha_k = self._comp_alpha_k(k)
                N_k = self._comp_N(alpha_k)
                r1k = self._comp_rk( 1, N_k)
                r2k = self._comp_rk(-1, N_k)
                xi_k = self._comp_cosXi(k, alpha_k, xi)

                vk = self._comp_vk(N_k, alpha_k, r1k, r2k, 1, eta) * xi_k
                hk = self._comp_hk(N_k, alpha_k, r1k, r2k, 1, eta) * xi_k 
                V += vk
                H += hk

            return V, H


class Sloan:
    def __init__(self, Ha, sigma_f, sigma_w, tw, a=1, b=1, debug=False):
        _chk_var('Ha', Ha, True)
        _chk_var('a', a)
        _chk_var('b', b)
        _chk_var('tw', tw)
        _chk_var('sigma_f', sigma_f)
        _chk_var('sigma_w', sigma_w, True)

        self.Ha = Ha
        self.a = a   # 2a = dist(A-A) between side walls
        self.b = b   # 2b = dist(B-B) between hartman walls
        self.ell = b / a
        self.debug = debug

        bigNum = 1e20
        if np.isinf(sigma_w) or sigma_w > bigNum:
            sigma_w = np.inf
            self.perfect = True
        else:
            self.perfect = False
        self.sigma_f = sigma_f # fluid electrical conductivity
        self.sigma_w = sigma_w # solid (wall) inside B

        c = sigma_w * tw / (sigma_f * a) # wall conducting ratio of B
        self.tw = tw # thickness of wall (in unit)

        bigNum = 1e20
        if np.isinf(c) or c > bigNum:
            c = np.inf
            self.perfect = True
        else:
            self.perfect = False
        self.c = c

        # paper's notation
        self.r = self.ell
        self.q = 1.0 + tw / a

        # perfectly conducting
        if self.perfect:
            self.sigma_f = 0.0
            self.sigma_w = 1.0 # conveniently, this works :)

    def _comp_xi(self, x): # [-1, 1]
        return x / self.a
    def _comp_eta(self, y): # [-ell, ell]
        return y / self.a

    def _comp_an(self, n):
        return (0.5 + n) * np.pi / self.r

    def _comp_kn(self, n, a_n):
        return 2.0 * np.power(-1, n) / (self.r * np.power(a_n,3))

    def _comp_alphabeta(self, pm, n, a_n):
        # original
        #return 0.5 * (- self.Ha + pm * np.sqrt(self.Ha*self.Ha + 4.0*a_n*a_n))
        # avoid cancelation error
        return (2.0 * a_n*a_n) / (self.Ha + pm * np.sqrt(self.Ha*self.Ha + 4.0*a_n*a_n))

    def _comp_Tn(self, a_n):
        return np.tanh(a_n * (1.0 - self.q))

    # exp(beta) * D_n
    def _comp_Dbar(self, a_n, beta_n, T_n):
        e2beta = np.exp(2*beta_n)
        term1 = self.sigma_f * a_n * (e2beta - 1.0)
        term2 = self.sigma_w * beta_n * T_n * (e2beta + 1.0)
        return 0.5 * (term1 - term2)

    # exp(-alpha_n) * E_n
    def _comp_Ebar(self, a_n, alpha_n, T_n):
        em2alph = np.exp(-2*alpha_n)
        term1 = self.sigma_f * a_n * (1.0 - em2alph)
        term2 = self.sigma_w * alpha_n * T_n * (1.0 + em2alph)
        return 0.5 * (term1 - term2)

    # exp(-alpha + beta) * denominator
    # denominator = D cosh(alpha) - E cosh(beta)
    def _comp_Qbar(self, Dbar_n, Ebar_n, alpha_n, beta_n):
        em2alph = np.exp(-2*alpha_n)
        e2beta = np.exp(2*beta_n)
        return 0.5 * ( Dbar_n*(1.0 + em2alph) - Ebar_n*(1.0 + e2beta) )

    # numerator of w term, scaled by exp(-alpha + beta)
    def _comp_Nwbar(self, alpha_n, beta_n, Dbar_n, Ebar_n, eta):
        term1 = np.exp(-alpha_n * (1.0 - eta)) \
              + np.exp(-alpha_n * (1.0 + eta))
        term2 = np.exp(beta_n * (1.0 + eta)) \
              + np.exp(beta_n * (1.0 - eta))
        return 0.5 * (Dbar_n * term1 - Ebar_n * term2)


    # numerator of B term, scaled by exp(-alpha + beta)
    def _comp_NBbar(self, alpha_n, beta_n, Dbar_n, Ebar_n, eta):
        term1 = np.exp(beta_n * (1.0 + eta)) \
              - np.exp(beta_n * (1.0 - eta))
        term2 = np.exp(-alpha_n * (1.0 - eta)) \
              - np.exp(-alpha_n * (1.0 + eta))
        return 0.5 * (Ebar_n * term1 - Dbar_n * term2)

    def _comp_Xi(self, a_n, xi):
        return np.cos(a_n * xi)

    # V = sum_n { k_n * (1 - Eta_n) * Xi_n}
    # Eta_n is a big fraction term
    # Xi_n is cos (a_n xi)
    def _VH(self, xx, yy, num_n = 100): # fast tensor prod of 1d arrs

        xi = xx / self.a 
        eta = yy / self.a
            
        sz = (len(yy), len(xx))
        V = np.zeros(sz)
        H = np.zeros(sz)
        for n in range(num_n):
            a_n = self._comp_an(n)
            k_n = self._comp_kn(n, a_n)

            alpha_n = self._comp_alphabeta(1, n, a_n)
            beta_n = self._comp_alphabeta(-1, n, a_n)
            T_n = self._comp_Tn(a_n)

            Dbar_n = self._comp_Dbar(a_n, beta_n, T_n)
            Ebar_n = self._comp_Ebar(a_n, alpha_n, T_n)
 
            Qbar_n = self._comp_Qbar(Dbar_n, Ebar_n, alpha_n, beta_n)
            Nwbar_n = self._comp_Nwbar(alpha_n, beta_n, Dbar_n, Ebar_n, eta)
            NBbar_n = self._comp_NBbar(alpha_n, beta_n, Dbar_n, Ebar_n, eta)
            if self.debug:
                print(n,a_n,self.Ha,alpha_n,beta_n,Dbar_n,Ebar_n,Qbar_n,Nwbar_n)

            wEta_n = 1.0 - Nwbar_n / Qbar_n
            bEta_n = NBbar_n / Qbar_n
            Xi_n = self._comp_Xi(a_n, xi)

            vn = k_n * np.outer(wEta_n, Xi_n)
            bn = k_n * np.outer(bEta_n, Xi_n)
            V += vn
            H += bn
        return V, H

    def VH(self, X, Y, num_n = 100, tensor=True):
        if tensor:
            assert isinstance(X, np.ndarray) and X.ndim == 1 \
               and isinstance(Y, np.ndarray) and Y.ndim == 1, 'invalid size'
            return self._VH(X, Y, num_n)
        else:
            X = np.array(X)
            Y = np.array(Y)
            xi = X / self.a
            eta = Y / self.a
            
            if isinstance(X, np.ndarray) and X.size > 1:
                V = np.zeros_like(xi)
                H = np.zeros_like(xi)
            elif isinstance(Y, np.ndarray) and Y.size > 1:
                V = np.zeros_like(eta)
                H = np.zeros_like(eta)
            else:
                V, H = 0.0, 0.0

            for n in range(num_n):
                a_n = self._comp_an(n)
                k_n = self._comp_kn(n, a_n)
                    
                alpha_n = self._comp_alphabeta(1, n, a_n)
                beta_n = self._comp_alphabeta(-1, n, a_n)
                T_n = self._comp_Tn(a_n)

                Dbar_n = self._comp_Dbar(a_n, beta_n, T_n)
                Ebar_n = self._comp_Ebar(a_n, alpha_n, T_n)

                Qbar_n = self._comp_Qbar(Dbar_n, Ebar_n, alpha_n, beta_n)
                Nwbar_n = self._comp_Nwbar(alpha_n, beta_n, Dbar_n, Ebar_n, eta)
                NBbar_n = self._comp_NBbar(alpha_n, beta_n, Dbar_n, Ebar_n, eta)

                wEta_n = 1.0 - Nwbar_n / Qbar_n
                bEta_n = NBbar_n / Qbar_n
                Xi_n = self._comp_Xi(a_n, xi)

                vn = k_n * wEta_n * Xi_n
                bn = k_n * bEta_n * Xi_n
                V += vn
                H += bn

            return V, H
        



