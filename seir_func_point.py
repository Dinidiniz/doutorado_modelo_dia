#!/usr/bin/python3
# -*- coding: utf-8 -*-

import base_functions_point as base_functions
import map_functions_point as map_functions
import numpy as np
import os
from scipy.integrate import odeint
from theano import tensor as tt
import theano

class seir:
    
    def __init__(self, 
                 precipitation_folder = "/home/leon/Doutorado/SIR/maps/precipitation-mean",
                 temp_day_folder = "/home/leon/Doutorado/SIR/maps/LST_Day",
                 temp_night_folder = "/home/leon/Doutorado/SIR/maps/Temp_Night",
                 city = "rj", 
                 variables_multiplier = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
        if len(variables_multiplier) < 2:
            variables_multiplier = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            
        o_multiplier, s_a_multiplier, a_multiplier, p_hm_multiplier, PDR_multiplier, p_mh_multiplier, u_h_multiplier, u_e_multiplier, u_v_multiplier, f_v_multiplier, d_multiplier, v_t_multiplier, k_multiplier, infected_imigration_multiplier, gamma_multiplier, delta_multiplier, incubation_multiplier = variables_multiplier
            
        self.env_func = base_functions.environment_functions(o_multiplier, s_a_multiplier, a_multiplier, p_hm_multiplier, PDR_multiplier, p_mh_multiplier, u_h_multiplier, u_e_multiplier, u_v_multiplier, f_v_multiplier, d_multiplier, v_t_multiplier,
                                                            k_multiplier, infected_imigration_multiplier, 
                                                            gamma_multiplier,
                                                            delta_multiplier, incubation_multiplier)
        self.itemp = map_functions.get_temp_interpolate(
            temp_day_folder = temp_day_folder,
            temp_night_folder = temp_night_folder,
            city = city
        )
        self.theano_itemp = theano.shared(
            np.array([float(self.itemp(x)) for x in range(2000)])
        )
        self.i_precipitation = map_functions.get_rainfall_interpolate(
            folder = precipitation_folder,
            city = city
        )
        self.theano_precipitation = theano.shared(
            np.array([float(self.i_precipitation(x)) for x in range(2000)])
        )
        self.total_population = map_functions.pop_dict[city]
        self.variables_multiplier = np.array(variables_multiplier)
        self.initial_state = np.array([[1000, 0, 0.99, 0, 0.0, 0.7, 0, 0.0000001, 0.3]])
        self.city = city
        
    def derivate(self, estado, tempo, args):
        '''
        estado: variavel do estado em que se encontra a simulação
        tempo: tempo
        args: alpha e outras variáveis para calcular o estado
        '''
        
        #estado = np.reshape(estado,(8,17,7))
        # female_probabilidade= 0.5
        tempo_truncate = tt.cast(tt.floor(tempo), 'int32')
        tempo_diff = tempo - tempo_truncate
        T = (1 - tempo_diff) * self.theano_itemp[tempo_truncate] + tempo_diff * self.theano_itemp[tempo_truncate + 1] # 30
        R = (1 - tempo_diff) * self.theano_precipitation[tempo_truncate] +  tempo_diff * self.theano_precipitation[tempo_truncate + 1]

        SEV = estado[0]
        IEV = estado[1]
        
        SV = estado[2]
        EV = estado[3] 
        IV = estado[4] 

        SH = estado[5]
        EH = estado[6]
        IH = estado[7]
        RH = estado[8] 
        
        Incidence = estado[9]

        NV = 2*(SV + IV + EV)
        NH = SH + IH + RH + EH

        # Vector part
        diff0 = self.env_func.o(T, constant = args[0]) * SV + (1 - self.env_func.v_t(constant = args[11])) * self.env_func.o(T, constant = args[0]) *  IV  -\
        (self.env_func.u_e(constant = args[7]) + self.env_func.d(R, constant = args[10])) * SEV 
        diff1 = self.env_func.v_t(constant = args[11]) * self.env_func.o(T, constant = args[0]) * IV -\
        (self.env_func.u_e(constant = args[7]) + self.env_func.d(R, constant = args[10])) * IEV

        #ADULTS VECTOR
        diff2 = self.env_func.d(R, constant = args[10]) * self.env_func.f_v(constant = args[9]) * self.env_func.s_a(T, constant = args[1]) * SEV * (1 - NV/(self.env_func.k(constant = args[12]) * NH)) -\
        (self.env_func.a(T, constant = args[2]) * self.env_func.p_hm(T, constant = args[3]) * (IH + self.env_func.infected_imigration(constant = args[13])*NH)/(NH) + self.env_func.u_v(constant = args[8])) * SV
        
        diff3 =  (self.env_func.a(T, constant = args[2]) * self.env_func.p_hm(T, constant = args[3]) * (IH + self.env_func.infected_imigration(constant = args[13])*NH)/(NH)) * SV - (self.env_func.i(T) +  self.env_func.u_v(constant = args[8])) * EV
        
        diff4 =  self.env_func.d(R, constant = args[10]) * self.env_func.f_v(constant = args[9]) * self.env_func.s_a(T, constant = args[1]) * IEV * (1 - NV/(self.env_func.k(constant = args[12]) * NH)) + self.env_func.i(T) * EV - self.env_func.u_v(constant = args[8]) * IV


        #Human part
        diff5 = - self.env_func.a(T, constant = args[2]) * self.env_func.p_mh(T, constant = args[5]) * IV * (SH / (NH)) + self.env_func.u_h(constant = args[6]) * NH - self.env_func.u_h(constant = args[6]) * SH  #parÂmetro para suavizar
        
        diff6 = self.env_func.a(T, constant = args[2]) * self.env_func.p_mh(T, constant = args[5]) * IV * (SH / (NH))  - (self.env_func.delta(constant = args[15]) + self.env_func.u_h(constant = args[6])) * EH
        
        diff7 = self.env_func.delta(constant = args[15]) * EH  - (self.env_func.gamma(constant = args[14]) + self.env_func.u_h(constant = args[6])) * IH
        
        diff8 = self.env_func.gamma(constant = args[14]) * IH - self.env_func.u_h(constant = args[6]) * RH

        #Incidencia
        diff9 = self.env_func.delta(constant = args[15]) * EH - Incidence
        
        return [diff0,diff1,diff2,diff3,diff4,diff5,diff6,diff7,diff8,diff9]
    
    
    def update_parameters(self, 
                          parameters=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         initial_state = [[1000, 0, 1, 0, 0.0, 1, 0, 0, 0]]):
        self.initial_state = initial_state
        o_multiplier, s_a_multiplier, a_multiplier, p_hm_multiplier, PDR_multiplier, p_mh_multiplier, u_h_multiplier, u_e_multiplier, u_v_multiplier, f_v_multiplier, d_multiplier, v_t_multiplier, k_multiplier, infected_imigration_multiplier, gamma_multiplier, delta_multiplier, incubation_multiplier = parameters
        self.variables_multiplier = parameters
        self.env_func = base_functions.environment_functions(o_multiplier, s_a_multiplier, a_multiplier, p_hm_multiplier, PDR_multiplier, p_mh_multiplier, u_h_multiplier, u_e_multiplier, u_v_multiplier, f_v_multiplier, d_multiplier, v_t_multiplier,
                                                            k_multiplier, infected_imigration_multiplier, 
                                                            gamma_multiplier,
                                                            delta_multiplier, incubation_multiplier)
        
        if not isinstance(self.variables_multiplier, np.ndarray):
            self.variables_multiplier = np.array(self.variables_multiplier)
        if not isinstance(self.initial_state, np.ndarray):
            self.initial_state = np.array(self.initial_state)
        
    
    def emulate(self, initial_state = None, time_stamp = 1000):
        
        if initial_state == None:
            initial_state = self.initial_state
            
        
            
            
            
        initial_list = initial_state[0].tolist()
        initial_list.append(0)
        initial_state = np.array(initial_list)
        
        initial_state = initial_state * self.total_population
        
        saved_file = "." + "/results_seir_point/{}/{}-{}-{}".format(
            str(self.city),
            str(initial_state.round(decimals=4).flatten()),
            str(time_stamp),
            str(self.variables_multiplier.round(decimals=4).flatten())
        ).replace(".","").replace("[","").replace("\n","").replace("]","") + ".npy"
        
        if (os.path.exists(saved_file)):
            return np.load(saved_file)
        else:
            sol =  odeint(self.derivate, initial_state.flatten(), 
                          range(time_stamp),
                          hmin = 0.00000000000000001,
                         h0=0.0000000000000000001)
            np.save(saved_file, sol)
        return sol