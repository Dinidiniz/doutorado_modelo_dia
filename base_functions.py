#!/usr/bin/python3
# -*- coding: utf-8 -*-


class environment_functions:
    
    u_h = None
    u_e = None
    u_v = None
    f_v = None
    v_t = None
    k = None
    infected_imigration = None
    gamma = None
    delta = None
    
    def __init__(self, o_multiplier = 1, s_a_multiplier = 1, a_multiplier = 1,
                p_hm_multiplier = 1, PDR_multiplier = 1, p_mh_multiplier = 1,
                u_h_multiplier = 1, u_e_multiplier = 1, u_v_multiplier = 1,
                f_v_multiplier = 1, d_multiplier = 1, v_t_multiplier = 1, k_multiplier = 1,
                infected_imigration_multiplier = 1, gamma_multiplier = 1,
                delta_multiplier = 1, incubation_multiplier = 1):
        self.o_constant = o_multiplier * 0.00856
        self.s_a_constant = s_a_multiplier * -0.00599
        self.a_constant = a_multiplier * 0.000202 
        self.p_hm_constant = p_hm_multiplier * 0.000491
        self.PDR_constant = PDR_multiplier * 0.0000665
        self.p_mh_constant = p_mh_multiplier * 0.000849
        self.u_h = u_h_multiplier * 0.00001869013
        self.u_e = u_e_multiplier * 0.01
        self.u_v = u_v_multiplier * 0.048184
        self.f_v = f_v_multiplier * 0.5
        self.v_t = v_t_multiplier * 0.11
        self.k = k_multiplier * 2
        self.gamma = gamma_multiplier * (1./5)
        self.infected_imigration = infected_imigration_multiplier * 0.00001
        self.d_constant = d_multiplier  * -0.37017772
        self.delta = delta_multiplier * (1./5.9)
        self.incubation_constant = incubation_multiplier * 0.0000665

    def o(self, T):
        min_value = 14.58
        max_value = 34.61
        T[T < min_value] = min_value
        T[T > max_value] = max_value
        return self.o_constant * T * (T - min_value) * ((max_value - T) ** (1./2))
    
    
    def i(self, T):
        min_value = 10.68
        max_value = 45.90
        T[T < min_value] = min_value
        T[T > max_value] = max_value
        return self.incubation_constant * T * (T - min_value) * ((max_value - T) ** (1./2))

    def s_a(self, T):
        min_value = 13.56
        max_value = 38.29
        T[T < min_value] = min_value
        T[T > max_value] = max_value
        return self.s_a_constant * (T - min_value) * (T - max_value)

    def a(self, T):
        min_value = 13.35
        max_value = 40.08
        T[T < min_value] = min_value
        T[T > max_value] = max_value
        return self.a_constant * T * (T - min_value) * ((max_value - T) ** (1./2))

    def p_hm(self, T):
        min_value = 12.22
        max_value = 37.46
        T[T < min_value] = min_value
        T[T > max_value] = max_value
        return self.p_hm_constant * T * (T - min_value) * ((max_value - T) ** (1./2))

    def PDR(self, T):
        min_value = 10.68
        max_value = 45.90
        T[T < min_value] = min_value
        T[T > max_value] = max_value
        return self.PDR_constant * T * (T - min_value) * ((max_value - T) ** (1./2))

    def p_mh(self, T):
        min_value = 17.05 #17 para 15
        max_value = 35.83
        T[T < min_value] = min_value
        T[T > max_value] = max_value
        return self.p_mh_constant * T * (T - min_value) * ((max_value - T) ** (1./2))
    
    def d(self, R):
        #min_value = 0
        #max_value = 1
        #R[R < min_value] = min_value
        #R[R > max_value] = max_value
        #return variable_peh *(-0.22358793 * R **2 + 0.32969868 * R + 0.46623609)
        rain = self.d_constant * (R - 0 ) * (R - 1)
        if rain < 0.0001:
            return 0
        return rain