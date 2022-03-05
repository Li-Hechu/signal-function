"""
本模块用于生成基础的连续时间信号
主要有正弦信号，指数信号，抽样信号，阶跃信号，斜变信号，冲激信号
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as cp
import sympy as sp
from scipy import integrate
from scipy import signal


class Signal:
    """
    信号处理的一些基本函数
    函数参数全部支持“数字+参数”的形式，如：“3-t”等，更加灵活，参数较少
    对于参数为函数的内置方法，其参数必须是函数，使用lambda匿名函数创建
    请注意，该模块里面x轴坐标的取值间隔必须小于0.01
    """

    def __init__(self):
        """
        Signal模块包含了基础信号函数
        使用 from Signal import sa 来导入此模块
        """
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        sp.init_printing()

    def upstep(self, t):
        """
        阶跃信号
        t：x轴坐标
        例：
        y = signal.upstep(3+t)
        返回值为np.array数组
        """
        return np.where(t > 0, 1, 0)

    def ramp(self, t):
        """
        斜变信号
        t：x轴坐标
        例：
        y = 2*signal.ramp(1-t)
        """
        return np.array(t * self.upstep(t=t))

    def Sa(self, t):
        """
        抽样信号
        t：x轴坐标
        例：
        y = 4*signal.Sa(3+t)
        """
        return np.array(np.sin(t) / t)

    def impulse(self, t, *, offset=0):
        """
        冲激函数
        t：x轴坐标，为np.array类型
        offset：偏移量，正数表示向左，负数表示向右，可选
        """
        y = np.zeros(len(t))
        index = int((offset - t[0]) / 0.01) - 1
        y[index] = 100.000000000045
        plt.ylim(bottom=-2, top=5)
        return y

    def even_dis(self, t, fun):
        """
        信号偶分解
        t：x轴坐标，为np.array类型
        fun：需要分解的函数
        例：
        f1 = lambda x: 3*signal.ramp(x+3) - 6*signal.ramp(x+1)
        y = signal.even(t,f1)
        返回值为np.array类型数组
        """
        f = lambda x: (fun(x) + fun(-x)) * 0.5
        return f(t)

    def odd_dis(self, t, fun):
        """
        信号奇分解
        t：x轴坐标，为np.array类型
        fun：需要分解的函数
        例：
        f1 = lambda x: 3*signal.ramp(x+3) - 6*signal.ramp(x+1)
        y = signal.odd(t,f1)
        """
        f = lambda x: (fun(x) - fun(-x)) * 0.5
        return f(t)

    def convolution(self, t, fun1, fun2):
        """
        信号卷积积分
        t：x轴坐标，为np.array类型
        fun1：卷积的第一个函数，为匿名函数
        fun2：卷积的第二个函数，为匿名函数
        请注意，在使用signal模块函数的时候，由于返回是数组而不是函数，所以这个时候
        实际参数应当为函数。
        例：
        f1 = lambda x: signal.upstep(x-1) - 2*signal.upstep(x-2)
        f2 = lambda x: np.exp(x)**(-2)
        y = signal.convolution(t, f1, f2)
        返回值为np.array类型数组
        """
        ans = []
        for i in range(len(t)):
            f = lambda tau: fun1(tau) * fun2(t[i] - tau)
            ans.append(cp.integrate.simps(f(t), t))  # 使用辛普森积分法，对孤立的点积分
        return np.array(ans)

    def correlate(self, s, m, f1, f2):
        """
        相关性计算
        s：原序列x轴的值，为ndarray类型
        m：相关序列的k值范围，应当为整数数组
        f1,f2：两个函数，应当为函数类型
        返回值为相关序列
        """
        ans = []
        for k in range(len(m)):
            a = 0
            for i in range(len(s)):
                a += f1(i) * f2(i - k)
            ans.append(a)
        return np.array(ans) / len(s)

    def psd(self, lim, r):
        """
        功率谱密度
        lim：x轴的范围
        r：相关序列
        返回值为频率（横坐标）和功率（纵坐标）
        """
        ans = []
        for w in lim:
            a = 0
            for k in range(len(r)):
                a += r[k] * np.exp(-w * k * 1j)
            ans.append(a)
        return lim / (2 * np.pi), np.abs(np.array(ans)) / len(r)

    def zi(self, t, B: list, ics0: list, ics1: list):
        """
        零输入响应
        t：x轴坐标，为np.array类型
        li：导数前的系数，为list类型，从左往右导数阶数依次降低
        ics0：初始条件中自变量的值，从左往右导数阶数依次降低
        ics1：初始条件中在该自变量的值下，函数的值，与ics0相对应，从左往右导数阶数依次降低
        返回值为np.array类型数组
        """
        f = sp.Function("f")
        x = sp.Symbol("x")
        count = len(B)
        ode = 0
        for i in range(count):
            ode += B[i] * f(x).diff(x, count - i - 1)
        sol = sp.dsolve(ode)
        symbol = sol.free_symbols - {x}
        eqs = [sol.rhs.diff(x, count - i - 2).subs(x, ics0[i]) - ics1[i] for i in range(count - 1)]
        sol_params = sp.solve(eqs, symbol)
        ans = sol.rhs.subs(sol_params)
        g = sp.lambdify(x, ans, "numpy")
        return g(t) * self.upstep(t)

    def zs(self, t, A: list, B: list, fun):
        """
        零状态响应
        t：x轴坐标，为np.array类型
        A：传输算子分子系数，从左到右指数依次降低
        B：传输算子分母系数，从左到右指数依次降低
        fun：输入函数
        """
        f1 = lambda x: self.h(x, A, B)
        re = self.convolution(t, fun, f1)
        return re * self.upstep(t)

    def lism(self, t, A: list, B: list, isc0: list, isc1: list, fun):
        """
        全响应函数
        t：x轴坐标，为np.array类型
        A：传输算子分子系数，从左到右指数依次降低
        B：传输算子分母系数，从左到右指数依次降低
        ics0：初始条件中自变量的值，从左往右导数阶数依次降低
        ics1：初始条件中在该自变量的值下，函数的值，与ics0相对应，从左往右导数阶数依次降低
        fun：输入函数
        """
        y1 = self.zi(t, B, isc0, isc1)
        y2 = self.zs(t, A, B, fun)
        return y1 + y2

    def h(self, t, A: list, B: list):
        """
        单位冲激响应
        A：传输算子分子系数，从左往右指数依次降低
        B：传输算子分母系数，从左往右指数依次降低
        """
        sys = (A, B)
        x, y = signal.impulse(sys, T=t)
        return y * self.upstep(t)

    def g(self, t, A: list, B: list):
        """
        单位阶跃响应
        A：传输算子分子系数，从左往右指数依次降低
        B：传输算子分母系数，从左往右指数依次降低
        """
        H = (A, B)
        x, y = signal.step(H, T=t)
        return y * self.upstep(t)

    def sgn(self, t):
        """
        符号函数
        t：x轴坐标，为np.array类型
        返回值为np.array类型数组
        """
        return -self.upstep(-t) + self.upstep(t)

    def single_index_fft(self, a, w):
        """
        单边因果指数信号傅里叶变换
        单边因果指数信号：y = exp(-at)*upstep(t)
        a：指数，大于0
        w：x轴坐标，为np.array类型
        返回值为幅度谱和相位谱
        """
        Fjw = np.abs(1.0 / (a + complex(0, 1) * w))
        fi = -np.arctan(w / a)
        return Fjw, fi

    def inverse_single_index_fft(self, a, w):
        """
        单边反因果指数信号傅里叶变换
        单边反因果指数信号：y = exp(at)*upstep(t)
        a：指数，大于0
        w：x轴坐标，为np.array类型
        返回值为幅度谱和相位谱
        """
        Fjw = np.abs(1.0 / (a - complex(0, 1) * w))
        fi = np.arctan(w / a)
        return Fjw, fi

    def double_index_fft(self, a, w):
        """
        双边指数函数傅里叶变换
        双边指数函数信号：y = exp(-a*abs(t))
        a：指数，大于0
        w：x轴坐标，为np.array类型
        返回值为幅度谱和相位谱
        """
        Fjw = (2 * a) / (a ** 2 + w ** 2)
        fi = np.zeros(len(w))
        return Fjw, fi

    def sgn_fft(self, w):
        """
        符号函数傅里叶变换
        w：x轴坐标，为np.array类型
        返回值为幅度谱和相位谱
        """
        Fjw = 2 / np.abs(w)
        fi = np.pi / 2 * self.sgn(w)
        return Fjw, fi


sa = Signal()
