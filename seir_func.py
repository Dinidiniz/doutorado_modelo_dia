#!/usr/bin/python3
# -*- coding: utf-8 -*-

import base_functions
import map_functions
import numpy as np
import os
from scipy.integrate import odeint

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
        self.i_precipitation = map_functions.get_rainfall_interpolate(
            folder = precipitation_folder,
            city = city
        )
        self.total_population = map_functions.pop_dict[city]
        self.variables_multiplier = np.array(variables_multiplier)
        self.initial_state = np.array([[1000, 0, 0.99, 0, 0.0, 0.7, 0, 0.0000001, 0.3, 0]])
        self.city = city
        
    def derivate(self, estado, tempo, *args):
        '''
        estado: variavel do estado em que se encontra a simulação
        tempo: tempo
        args: alpha e outras variáveis para calcular o estado
        '''
        
        if len(args) > 1:
            print(args)
            self.update_parameters(
                          parameters=args[1:]
            )
        
        #estado = np.reshape(estado,(8,17,7))
        female_probabilidade= 0.5
        
        T = self.itemp(tempo) #self.env_func.T(tempo) # 30
        R = self.i_precipitation(tempo)

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
        diff0 = self.env_func.o(T) * self.env_func.f_v * SV + (1 - self.env_func.v_t) * self.env_func.o(T) * self.env_func.f_v *  IV  -\
        (self.env_func.u_e + self.env_func.d(R)) * SEV 
        diff1 = self.env_func.v_t * self.env_func.o(T) * self.env_func.f_v * IV -\
        (self.env_func.u_e + self.env_func.d(R)) * IEV

        #ADULTS VECTOR
        diff2 = self.env_func.d(R) * self.env_func.f_v * self.env_func.s_a(T) * SEV * (1 - NV/(self.env_func.k * NH)) -\
        (self.env_func.a(T) * self.env_func.p_hm(T) * (IH + self.env_func.infected_imigration*NH)/(NH) + self.env_func.u_v) * SV
        
        diff3 =  (self.env_func.a(T) * self.env_func.p_hm(T) * (IH + self.env_func.infected_imigration*NH)/(NH)) * SV - (self.env_func.i(T) +  self.env_func.u_v) * EV
        
        diff4 =  self.env_func.d(R) * self.env_func.f_v * self.env_func.s_a(T) * IEV * (1 - NV/(self.env_func.k * NH)) + self.env_func.i(T) * EV - self.env_func.u_v * IV


        #Human part
        diff5 = - self.env_func.a(T) * self.env_func.p_mh(T) * IV * (SH / (NH)) + self.env_func.u_h * NH - self.env_func.u_h * SH  #parÂmetro para suavizar
        
        diff6 = self.env_func.a(T) * self.env_func.p_mh(T) * IV * (SH / (NH))  - (self.env_func.delta + self.env_func.u_h) * EH
        
        diff7 = self.env_func.delta * EH  - (self.env_func.gamma + self.env_func.u_h) * IH
        
        diff8 = self.env_func.gamma * IH - self.env_func.u_h * RH

        #Incidencia
        diff9 = self.env_func.delta * EH - Incidence
        
        return np.array([diff0,diff1,diff2,diff3,diff4,diff5,diff6,diff7,diff8,diff9])
    
    
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
        
        if initial_state != None:
            self.initial_state = np.array(initial_state)
        
        initial_state = self.initial_state * self.total_population
        
        saved_file = "." + "/results_seir/{}/{}-{}-{}".format(
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