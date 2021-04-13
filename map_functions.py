#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import requests
import pandas as pd
import io
import os 
import rasterio
from rasterio.mask import mask
from scipy import interpolate
import math

bounds_dict = {
    "curitiba": [ -49.392920, -25.349078, -49.179762, -25.619437 ],
    "rj": [-43.09, -22.74 , -43.79, -23.08],
    "foz": [-54.600009, -25.445518, -54.498446, -25.597800],
    "fortaleza": [-38.443224, -3.898599, -38.689268, -3.629847],
    "bh": [-44.091244, -19.815407, -43.875275, -20.015823],
    "poa": [-51.336471, -29.826417, -51.000050, -30.158798 ]
}

turism_dict = {
    "rj": 982538/(365./8),
    "foz": 725077/(365./8),
    "curitiba": 725077/(365./8),
    "fortaleza": 95786/(365./8),
    "bh": 56230/(365./8),
    "poa": 653622/(365./8)
}

geocodes = {
    "rj": 3304557,
    "curitiba": 4106902,
    "foz": 4108304,
    "fortaleza": 2304400,
    "bh": 3106200,
    "poa": 4314902
}

data_type = {
    "rj": "data_iniSE",
    "curitiba": "data_iniSE",
    "foz": "data_iniSE",
    "fortaleza": "data_iniSE",
    "bh": "data_iniSE",
    "poa": "data_iniSE"
}

pop_dict = {
    "rj": 6320446,
    "fortaleza": 2452185,
    "foz": 256088,
    "poa": 1409351
}

def get_map_files_in_order(folder):
    dates_and_files = []
    for filename in os.listdir(folder):
        date = '_'.join(filename.split(".")[0].split("_")[-4:-1])
        dates_and_files.append([date,filename])
    dates_and_files.sort()
    return dates_and_files

def get_RJ_map(folder, map_in_folder, city = "rj"):
    new_bounds = bounds_dict[city]
    bounds = np.array(new_bounds)
    geoms = [{'type': 'Polygon', 'coordinates': [[(bounds[0], bounds[1]), (bounds[0], bounds[3]), 
                                                  (bounds[2], bounds[3]), (bounds[2], bounds[1]), (bounds[0], bounds[1])]]}]
    
    ds2 = rasterio.open(os.path.join(folder, map_in_folder))

    # load the raster, mask it by the polygon and crop it
    with ds2 as src:
        out_image, out_transform = mask(src, geoms, crop=True, all_touched=True)

    return out_image[0]

def get_rainfall_interpolate(folder = "/home/leon/Doutorado/SIR/maps/precipitation-mean",
                            city = "rj"):
    '''
    Get precipitation data as interpolate
    '''
    if (os.path.exists("list_of_maps_precipitation_mean_day_{0}.npy".format(city))):
        list_of_maps = np.load("list_of_maps_precipitation_mean_day_{0}.npy".format(city))
    else:
        maps_in_order = get_map_files_in_order(folder)
        list_of_maps = []
        for x in range(len(maps_in_order)):
            map_temperatura = get_RJ_map(folder, maps_in_order[x][1])/100
            list_of_maps.append(np.nanmean(map_temperatura))
        np.save("list_of_maps_precipitation_mean_{0}.npy".format(city), list_of_maps)

    #print(list_of_maps)
    ml = list(list_of_maps)
    for i in range(100): #PUT SAME DATA 10 TIMES
        ml = ml + list(list_of_maps)
    ml = np.array(ml)
    #print(len(ml))
    return interpolate.interp1d(np.arange(0.0, len(ml)*8, 8),ml, axis=0)


def get_cases(city = "rj"):
    if (os.path.exists("cases/casos_dia_{0}.npy".format(city))):
        return np.load("cases/casos_dia_{0}.npy".format(city))
    else:
        url="https://info.dengue.mat.br/api/alertcity/?geocode={0}&disease=dengue&format=csv&ew_start=0&ey_start=2010&ew_end=01&ey_end=2021".format(geocodes[city])
        s=requests.get(url).content
        c=pd.read_csv(io.StringIO(s.decode('utf-8')))
        c = c.groupby(data_type[city]).sum() # data ou iniSE??
        casos_dia = []
        first = True
        for row in c.itertuples(index=False):
            if first:
                first = False
                continue
            for i in range(7):
                casos_dia.append(row[4]/7)
        np.save('cases/casos_dia_{0}.npy'.format(city), casos_dia)
        return casos_dia


def get_temp_interpolate(temp_day_folder = "/home/leon/Doutorado/SIR/maps/LST_Day",
                         temp_night_folder = "/home/leon/Doutorado/SIR/maps/Temp_Night",
                        city = "rj"):
    '''
    Get temperature data as interpolate
    '''
    if (os.path.exists("list_of_maps_temperature_day_{0}.npy".format(city))):
        list_of_maps = np.load("list_of_maps_temperature_day_{0}.npy".format(city))
    else:
        maps_in_order = get_map_files_in_order(temp_day_folder)
        maps_in_order_2 = get_map_files_in_order(temp_night_folder)
        list_of_maps = []
        for x in range(len(maps_in_order)):
            map_temperatura = np.nanmean(get_RJ_map(temp_day_folder, maps_in_order[x][1]))*0.02 - 273.15
            map_temperatura_2 = np.nanmean(get_RJ_map(temp_night_folder, maps_in_order_2[x][1]))
            list_of_maps.append((map_temperatura + map_temperatura_2)/2)
        np.save("list_of_maps_temperature_day_{0}.npy".format(city), list_of_maps)


    for n in range(len(list_of_maps)):
        if math.isnan(list_of_maps[n]):
            list_of_maps[n] = (list_of_maps[n-1] + list_of_maps[n+1])/2
    #print(list_of_maps)
    ml = list(list_of_maps)
    for i in range(100): #PUT SAME DATA 10 TIMES
        ml = ml + list(list_of_maps)
    ml = np.array(ml)

    return interpolate.interp1d(np.arange(0.0, len(ml)*8, 8),ml, axis=0)