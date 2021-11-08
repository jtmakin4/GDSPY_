import numpy as np
from scipy.special import ellipk as K
from scipy.optimize import fsolve as solve


class CPW_calculator: # todo функция дерьма, переделать
    c = 2.99792458 * 1e8
    e0 = 8.85418 * 1e-12

    def __init__(self, s, w, t=0.1, h=503, e=10.2,
                 ctype='closed'):  # in and out in micrometers, others in meters
        s = s*1e-6
        w = w*1e-6
        t = t*1e-6
        h = h*1e-6
        self.t = t
        d = 2 * self.t / np.pi
        a = s / 2.
        b = s / 2. + w
        w1 = a + d / 2 - d * np.log(d / a) / 2 + 3 / 2 * d * np.log(2) - d / 2 * np.log((a + b) / (b - a))
        w2 = b - d / 2 + d * np.log(d / a) / 2 - 3 / 2 * d * np.log(2) + d / 2 * np.log((a + b) / (b - a))
        self.s = 2 * w1
        self.w = w2 - w1

        self.h1 = h
        self.h3 = h
        self.e1 = e
        self.type = ctype
        self.b = 2. * self.w + self.s
        if ctype == 'opened':  # Coplanar Waveguide With Ground
            self.e2 = 1.
            self.h2 = 1e14
            self.h4 = 1e14
        elif ctype == 'closed':  # CPW sandwiched between two  dielectric substrates
            self.e2 = e
            self.h2 = h
            self.h4 = h
        else:
            print('type err')

    # calculating functions (from article)
    def _k34(self, h):
        return np.tanh(np.pi * self.s / (4. * h)) / np.tanh(np.pi * self.b / (4. * h))

    def _k34_(self, h):
        return np.sqrt(1. - self._k34(h) ** 2)

    def _k12(self, h):
        return np.sinh(np.pi * self.s / (4. * h)) / np.sinh(np.pi * self.b / (4. * h))

    def _k12_(self, h):
        return np.sqrt(1. - self._k12(h) ** 2)

    def _frac34(self, h):
        return K(self._k34(h)) / K(self._k34_(h))

    def _frac12(self, h):
        return K(self._k12(h)) / K(self._k12_(h))

    def eps_eff(self):
        return 1. + self._frac12(self.h1) / (self._frac34(self.h3) + self._frac34(self.h4)) * (self.e1 - 1.) + \
               self._frac12(self.h2) / (self._frac34(self.h3) + self._frac34(self.h4)) * (self.e2 - 1.)

    def z0(self):
        return 60. * np.pi / np.sqrt(self.eps_eff()) / (self._frac34(self.h3) + self._frac34(self.h4))


def calc_line(b, z=50.):
    """
        Finds S for 50 Om CPW.

    :param z:
    :param b: Num (micrometers).
        Total width of CPW
    :return: S (micrometers).
    """
    b = b * 1e-6

    def func(s):
        w = (b - s) / 2.
        res = CPW_calculator(s, w, h=503, e=10.2, ctype='closed')
        return res.z0() - z

    s = round(solve(func, x0=np.array([1e-4]), xtol=1e-9)[0] * 1e6, 3)
    w = (b - s) / 2
    return s, w
if __name__ == '__main__':
    res = CPW_calculator(s=1000, w=1000, h=500, e=10, ctype='opened')
    # res.z0()
    print(res.z0())

