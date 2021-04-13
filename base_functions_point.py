#!/usr/bin/python3
# -*- coding: utf-8 -*-
from theano import tensor as tt


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
    
    def __init__(self, o_multiplier, s_a_multiplier, a_multiplier,
                p_hm_multiplier, PDR_multiplier, p_mh_multiplier,
                u_h_multiplier, u_e_multiplier, u_v_multiplier,
                f_v_multiplier, d_multiplier, v_t_multiplier, k_multiplier,
                infected_imigration_multiplier, gamma_multiplier,
                delta_multiplier = 1, incubation_multiplier = 1):
        self.o_constant = o_multiplier * 0.00856
        self.s_a_constant = s_a_multiplier * -0.00599
        self.a_constant = a_multiplier * 0.000202 
        self.p_hm_constant = p_hm_multiplier * 0.000491
        self.PDR_constant = PDR_multiplier * 0.0000665
        self.p_mh_constant = p_mh_multiplier * 0.000849
        self.d_constant = d_multiplier  * -0.37017772
        self.incubation_constant = incubation_multiplier * 0.0000665
        
    def u_h(self, constant = 1):
        return constant * 0.00001869013
    
    def u_e(self, constant = 1):
        return constant * 0.01
    
    def u_v(self, constant = 1):
        return constant * 0.048184
    
    def f_v(self, constant = 1):
        return constant * 0.5
    
    def v_t(self, constant = 1):
        return constant * 0.11
    
    def k(self, constant = 1):
        return constant * 2
    
    def gamma(self, constant = 1):
        return constant * (1./5)
    
    def infected_imigration(self, constant = 1):
        return constant * 0.00001
    
    def delta(self, constant = 1):
        return constant * (1./5.9)

    def o(self, T, constant = 1):
        min_value = 14.58
        max_value = 34.61
        answer = constant * self.o_constant * T * (T - min_value) * ((max_value - T) ** (1./2))
        answer_clipped = answer.clip(0,1000000000000000000)
        return answer_clipped
    
    
    def i(self, T, constant = 1):
        min_value = 10.68
        max_value = 45.90
        answer = constant * self.incubation_constant * T * (T - min_value) * ((max_value - T) ** (1./2))
        answer_clipped = answer.clip(0,1000000000000000000)
        return answer_clipped

    def s_a(self, T, constant = 1):
        min_value = 13.56
        max_value = 38.29
        answer = constant * self.s_a_constant * (T - min_value) * (T - max_value)
        answer_clipped = answer.clip(0,1000000000000000000)
        return answer_clipped

    def a(self, T, constant = 1):
        min_value = 13.35
        max_value = 40.08
        answer = constant * self.a_constant * T * (T - min_value) * ((max_value - T) ** (1./2))
        answer_clipped = answer.clip(0,1000000000000000000)
        return answer_clipped

    def p_hm(self, T, constant = 1):
        min_value = 12.22
        max_value = 37.46
        answer = constant * self.p_hm_constant * T * (T - min_value) * ((max_value - T) ** (1./2))
        answer_clipped = answer.clip(0,1000000000000000000)
        return answer_clipped

    def PDR(self, T, constant = 1):
        min_value = 10.68
        max_value = 45.90
        answer = constant * self.PDR_constant * T * (T - min_value) * ((max_value - T) ** (1./2))
        answer_clipped = answer.clip(0,1000000000000000000)
        return answer_clipped

    def p_mh(self, T, constant = 1):
        min_value = 17.05 #17 para 15
        max_value = 35.83
        answer = constant * self.p_mh_constant * T * (T - min_value) * ((max_value - T) ** (1./2))
        answer_clipped = answer.clip(0,1000000000000000000)
        return answer_clipped
    
    def d(self, R, constant = 1):
        #min_value = 0
        #max_value = 1
        #R[R < min_value] = min_value
        #R[R > max_value] = max_value
        #return variable_peh *(-0.22358793 * R **2 + 0.32969868 * R + 0.46623609)
        answer = constant * self.d_constant * (R - 0 ) * (R - 1)
        answer_clipped = answer.clip(0,1000000000000000000)
        return answer_clipped