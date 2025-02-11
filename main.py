import model
import numpy as np
from gurobipy import GRB
import folium
from folium.plugins import TimestampedGeoJson
import os
import math
import matplotlib.pyplot as plt
import datetime
import geopandas as gpd
import random

center_lat = 43.220034483305525
center_lon = -2.021473511525546

import requests
import pandas as pd
import os

def fetch_pv_data(lat, lon):
    url = "https://re.jrc.ec.europa.eu/pvgis5/seriescalc.php"

    # Set API parameters
    params = {
        "lat": lat,
        "lon": lon,
        "raddatabase": "PVGIS-SARAH3",  # Radiation database
        "usehorizon": 1,  # Use horizon data (boolean value)
        "outputformat": "json",  # Output format
        "pvcalculation": 1,  # Basic PV calculation
        "peakpower": 0.4,  # Peak power (kW)
        "loss": 14,  # System losses (%)
        "angle": 35,  # Tilt angle (degrees)
        "index": 0,  # Tracking system (0 = fixed)
        "optimalangles": 1,  # Optimal tilt angle
        "start": "2023",  # Start time (YYYYMMDD:HHMM)
        "end": "2023",  # End time (YYYYMMDD:HHMM)
        "timeinterval": 1  # Time interval (minutes)
    }

    # Make API request
    response = requests.get(url, params=params)

    # Check if response was successful
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()
        # Create a Pandas DataFrame
        df = pd.DataFrame({
            "Time": [hour["time"] for hour in data["outputs"]["hourly"]],
            "G(i)": [hour["G(i)"] for hour in data["outputs"]["hourly"]],
            "H_sun": [hour["H_sun"] for hour in data["outputs"]["hourly"]],
            "T2m": [hour["T2m"] for hour in data["outputs"]["hourly"]],
            "WS10m": [hour["WS10m"] for hour in data["outputs"]["hourly"]],
            "Int": [hour["Int"] for hour in data["outputs"]["hourly"]],
            "PV Power": [hour["P"] for hour in data["outputs"]["hourly"]]
        })
        return df
    else:
        print("Error:", response.status_code)
        print("Response content:")
        print(response.content)
        return None

def save_pv_data(df, filename):
    curr_dir = os.getcwd()
    curr_dir = os.path.join(curr_dir, filename)
    df.to_csv(curr_dir, index=False)
    print("Data saved to", filename)

# Example usage:
#lat = 43.22493388170683
#lon = -2.0197695759229872
#df = fetch_pv_data(lat, lon)
#if df is not None:
#    save_pv_data(df, "pv_data.csv")

#def generate_inputs():
#def generate_inputs(num_generators, num_users, num_storage, 
#                    generators_power, user_requirements):
def generate_inputs(num_generators, num_users, num_storage, 
                    start_dt, user_requirements):
    
    """
    # Define the number of nodes
    num_generators = 3
    num_users = 7
    num_storage = 3
    """
    #resistivity_factor = 2.82*10**-8/0.0005 # Ohm/m 
    resistivity_factor = 2.5*10**-4 # Ohm/m (rho/A [Ohm*m/m2])
    
    # Define the size of the square in kilometers
    square_size_km = 1
    # Convert the square size from kilometers to degrees
    square_size_deg = square_size_km / 111  # approximate conversion factor


    # Save the map as an HTML file
    curr_dir = os.getcwd()
    shp_path = os.path.join(curr_dir,'Gipuzkoa_Shapefile','Persons_Powerreq_PVcapacity')
    # Read the shapefile into a GeoDataFrame
    Persons_Powerreq_PVcapacity_gdf = gpd.read_file(shp_path)
    centroids = Persons_Powerreq_PVcapacity_gdf['geometry'].centroid

    # Select random centroids for generators
    random.seed(41)  # Set the seed value
    generator_positions = random.sample(list(centroids), num_generators)
    user_positions = random.sample([c for c in centroids if c not in generator_positions], num_users)

    generator_positions = np.array([(point.y, point.x) for point in generator_positions])
    user_positions = np.array([(point.y, point.x) for point in user_positions])
  
    storage_positions = np.zeros((num_storage, 2))
    for i in range(num_storage):
        # Randomly select a generator
        generator_index = i
        # Generate a random angle
        np.random.seed(41)  # Set the seed value
        #angle = random.uniform(0, 2 * math.pi)
        angle = np.random.uniform(0, 2 * np.pi)
        # Calculate the x and y coordinates of the storage node
        storage_positions[i, 0] = generator_positions[generator_index, 0] + 100 * math.cos(angle) / 111111  # convert meters to degrees
        storage_positions[i, 1] = generator_positions[generator_index, 1] + 100 * math.sin(angle) / 111111  # convert meters to degrees

    """
    for ig in range(num_generators):
        lat, lon = generator_positions[ig]
        df = fetch_pv_data(lat, lon)
        if df is not None:
            save_pv_data(df, "pv_data_" + str(ig) + ".csv")
    """

    generators_power = []
    for ig in range(num_generators):
        lat, lon = generator_positions[ig]
        df = fetch_pv_data(lat, lon)
        if df is None:
            print(f"No data for generator {ig} at position ({lat}, {lon})")
        else:
            df['Time'] = pd.to_datetime(df['Time'], format='%Y%m%d:%H%M')
            # Calculate the absolute difference between each timestamp and the given timestamp
            diff = np.abs(df['Time'] - start_dt)
            # Find the index of the row with the smallest absolute difference
            idx = np.argmin(diff)

            df_24h = df.iloc[idx:idx+24]
            hourly_power = df_24h['PV Power'].values
            #print('hourly_power')
            #print(hourly_power)
            # Append the hourly power to the generator_power list
            generators_power.append(hourly_power)
            save_pv_data(df, "pv_data_" + str(ig) + ".csv")

    # Convert the generator_power list to a numpy array
    generators_power = np.array(generators_power)
    print(np.shape(generators_power))
    # Plot the signals
    time = np.arange(0, 24, 1)
    fig, ax = plt.subplots()
    for i in range(num_generators):
        ax.plot(time, generators_power[i], label=f'Generator {i+1}', color=f'C{i}',linewidth=3)
    for i in range(num_users):
        ax.plot(time, user_requirements[i], label=f'User {i+1}', color=f'C{i+num_generators}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Signal Amplitude')
    ax.legend()
    plt.show()
    fig.savefig('signals.png')


    # Define the maximum capacity for each storage node
    storage_power_capacity = np.random.uniform(2000, 4000, num_storage)
    stored_energy = np.random.uniform(0, 5000, num_storage)
    storage_Emax = np.random.uniform(10000, 12000, num_storage) # Wh
    time_power = 1

    print()
    print('Inputs generated')
    print()
    input_data = {
        'num_generators': num_generators,
        'num_users': num_users,
        'num_storage': num_storage,
        'generator_positions': generator_positions.flatten(),
        'user_positions': user_positions.flatten(),
        'storage_positions': storage_positions.flatten(),
        'generator_power': generators_power,
        'user_requirements': user_requirements,
        'storage_power_capacity': storage_power_capacity,
        'stored_energy': stored_energy,
        'storage_Emax': storage_Emax,
        'time_power': time_power,
        'resistivity_factor': resistivity_factor
    }

    return input_data

def generate_and_plot_signals():
    # Define parameters
    num_generators = 3
    num_users = 7
    max_amplitudes_generators = [12000, 10000, 10000]
    #max_amplitudes_users = np.random.uniform(3000, 4000, size=num_users)
    max_amplitudes_users = np.random.uniform(100, 200, size=num_users)
    time = np.arange(0, 24, 1)
    sigma = 1  # standard deviation of the Gaussian
    mu = np.pi  # mean of the Gaussian
    
    # Generate generators signals
    generators = np.zeros((num_generators, len(time)))
    noise_std = 500
    for i in range(num_generators):
        #generators[i] = max_amplitudes_generators[i] * abs(np.sin(2 * np.pi * frequency * time))
        # Add low-frequency noise
        frequency = 1/24.
        # Generate Gaussian function
        x = 2 * np.pi * frequency * time
        generators[i] = max_amplitudes_generators[i]*np.exp(-(x - mu)**2 / (2 * sigma**2))
        generators[i] += noise_std * np.random.normal(size=len(time))
        generators[i] = np.where(generators[i] < 0, 0, generators[i])

    # Generate users requirements signals
    users_requirements = np.zeros((num_users, len(time)))
    noise_std = 2
    for i in range(num_users):
        frequency = 1/24.
        # Generate Gaussian function
        x = 2 * np.pi * frequency * time
        users_requirements[i] = max_amplitudes_users[i]*np.exp(-(x - mu)**2 / (2 * sigma**2))
        users_requirements[i] += noise_std * np.random.normal(size=len(time))
        users_requirements[i] = np.where(users_requirements[i] < 0, 0, users_requirements[i])

    # Plot the signals
    fig, ax = plt.subplots()
    for i in range(num_generators):
        ax.plot(time, generators[i], label=f'Generator {i+1}', color=f'C{i}',linewidth=3)
    for i in range(num_users):
        ax.plot(time, users_requirements[i], label=f'User {i+1}', color=f'C{i+num_generators}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Signal Amplitude')
    ax.legend()
    plt.show()

    return generators, users_requirements

def perturb_input(input_data, perturbation_factor=0.1):
    # Create a copy of the input data to avoid modifying the original values
    perturbed_data = input_data.copy()

    # Perturb the generator power
    #for key in ['generator_power', 'storage_power_capacity', 'stored_energy', 'storage_Emax']:
    #for key in ['generator_power', 'stored_energy']:
    for key in ['generator_power', 'storage_power_capacity', 'stored_energy', 'storage_Emax']:
        if key in perturbed_data:
            perturbation = np.random.uniform(-perturbation_factor, perturbation_factor, size=perturbed_data[key].shape)
            perturbed_data[key] = np.multiply(perturbed_data[key], (1 + perturbation))
            diff = np.subtract(perturbed_data[key], input_data[key])
            #print(f'Variation in {key}:', np.max(diff))

    return perturbed_data


def get_opacity_for_charging_state(charging_state):
    return charging_state / 100


def main():
    # Define the number of nodes
    num_generators = 3
    num_users = 7
    num_storage = 3
    # Define the center of the 2km2 square in Andoain
    center_lat = 43.2169548061656
    center_lon = -2.02130803900803

    #start_time = datetime.datetime(2017, 6, 2, 0, 0, 0)
    #start_time = datetime.datetime.now()
    # Midnight:
    start_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    #time_step = datetime.timedelta(minutes=15)  
    time_step = datetime.timedelta(hours=1)  

    scaling_factor = 0.8

    generators_power, user_pow_requ = generate_and_plot_signals()

    # Create a Folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # Generate inputs
    #start_datetime = datetime.datetime.now()
    start_datetime = datetime.datetime(2022, 7, 1, 0, 10, 0)
    print('start_datetime:', start_datetime)
    #input_data = generate_inputs(num_generators, num_users, num_storage,
    #                         generators_power user_pow_requ)
    input_data = generate_inputs(num_generators, num_users, num_storage,
                                 start_datetime, user_pow_requ)



    # Create a list to store the 24 results
    solutions = []
    all_features = []
    max_generators_power = max(max(row) for row in generators_power)
    excess_power = []


    generators_power = input_data['generator_power']
    for ii in range(len(generators_power[0])):
        generator_power = generators_power[:, ii][:3]
        user_requirements = user_pow_requ[:, ii]

        current_time = start_time + ii * time_step

        # Generate the inputs
        #input_data = generate_inputs()
        #input_data = generate_inputs(generator_power, user_requirements)
        input_data['generator_power'] = generator_power
        input_data['user_requirements'] = user_requirements
        #input_data['stored_energy'] = stored_E
        # Define the maximum capacity for each storage node
        # Call the define_model function with the required arguments
        model_instance, x_GU, x_GS, x_SU, power_GU, power_GS, power_SU, power_loss_GU, power_loss_GS, power_loss_SU, stored_E = model.define_model(input_data)

        # Set the inputs
        # Solve the model
        model_instance.optimize()

        num_generators = input_data['num_generators']
        num_users = input_data['num_users']
        num_storage = input_data['num_storage']
        generator_positions = input_data['generator_positions'].reshape((num_generators, 2))
        user_positions = input_data['user_positions'].reshape((num_users, 2))
        storage_positions = input_data['storage_positions'].reshape((num_storage, 2))
        generator_power = input_data['generator_power']
        user_requirements = input_data['user_requirements']
        storage_power_capacity = input_data['storage_power_capacity']
        #stored_energy = input_data['stored_energy']
        storage_Emax = input_data['storage_Emax']
        #time_power = input_data['time_power']

        # Check if the problem is infeasible
        if model_instance.status == GRB.INFEASIBLE:
            print()
            print("The model is infeasible")
            print()
            solutions.append([ii,'Infeasible'])
            solved = False
            while(solved==False):
                print()
                print('WARNING: model is being re-defined and executed!')
                print()
                perturbed_data = perturb_input(input_data, 0.3)
                model_instance, x_GU, x_GS, x_SU, power_GU, power_GS, power_SU, power_loss_GU, power_loss_GS, power_loss_SU, stored_E = model.define_model(perturbed_data)
                model_instance.optimize()
                num_generators = perturbed_data['num_generators']
                num_users = perturbed_data['num_users']
                num_storage = perturbed_data['num_storage']
                generator_positions = perturbed_data['generator_positions'].reshape((num_generators, 2))
                user_positions = perturbed_data['user_positions'].reshape((num_users, 2))
                storage_positions = perturbed_data['storage_positions'].reshape((num_storage, 2))
                generator_power = perturbed_data['generator_power']
                user_requirements = perturbed_data['user_requirements']
                storage_power_capacity = perturbed_data['storage_power_capacity']
                #stored_energy = perturbed_data['stored_energy']
                storage_Emax = perturbed_data['storage_Emax']
                #time_power = perturbed_data['time_power']
                if (model_instance.status == GRB.OPTIMAL):
                    solved = True
        else:
            print()
            print("The model is feasible")
            print()
            solutions.append([ii,'Feasible'])

        # Update the stored energy
        #input_data['stored_energy'] = stored_E
        stored_E_values = [stored_E[j].x for j in range(num_storage)]
        #input_data['stored_energy'] = stored_E 
        input_data['stored_energy'] = stored_E_values 
        #print()
        #print('Stored Energy: ')
        #print(stored_E_values)


        #                    "times": [current_time.isoformat(), (current_time + time_step).isoformat()],
        points_lines = [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            'coordinates': [float(generator_positions[j, 1]), float(generator_positions[j, 0])]
                        },
                        "properties": {
                            "times": [current_time.isoformat()], 
                            'icon': 'circle', 
                            'iconstyle': {'color': 'orange'},
                            'popup': f'Generated Power: {generator_power[j]/1000:.2f} kW'        
                        }
                    }
                    for j in range(num_generators)
                    ]   
        all_features.extend(points_lines)
        points_lines = [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            'coordinates': [float(user_positions[j, 1]), float(user_positions[j, 0])]
                        },
                        "properties": {
                            "times": [current_time.isoformat()], 
                            'icon': 'circle', 
                            'iconstyle': {'color': 'red'},
                            'popup': f'Requested Power: {user_requirements[j]/1000:.2f} kW'        
                        }
                    }
                    for j in range(num_users)
                    ] 
        all_features.extend(points_lines)
        #'iconstyle': {'color': get_color_for_charging_state(100*stored_E[j].x/storage_Emax[j])},
        points_lines = [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            'coordinates': [float(storage_positions[j, 1]), float(storage_positions[j, 0])]
                        },
                        "properties": {
                            "times": [current_time.isoformat()], 
                            'icon': 'circle',                           
                            'iconstyle': {'color': 'green',
                                        'radius': 5 + (15 * (100*stored_E[j].x/storage_Emax[j]) / 100),  # Calculate radius based on charging state
                                        'fillOpacity': get_opacity_for_charging_state(100*stored_E[j].x/storage_Emax[j])
                                        },
                            'popup': f'Storage Power capacity: {storage_power_capacity[j]/1000:.2f} kW <br> Charging state={100*stored_E[j].x/storage_Emax[j]:.2f}% ({stored_E[j].x:.2f} of {storage_Emax[j]:.2f})'      
                        }
                    }
                    for j in range(num_storage)
                    ] 
        all_features.extend(points_lines)

        # Lines G-U
        points_lines = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    'coordinates': [[float(generator_positions[i, 1]), float(generator_positions[i, 0])],
                                    [float(user_positions[j, 1]), float(user_positions[j, 0])]]
                },
                "properties": {
                    "times": [current_time.isoformat(), current_time.isoformat()],                     
                    "style": {
                            "color": 'black',
                            "weight": 50* power_GU[i, j].x * x_GU[i, j].x  / (scaling_factor*max_generators_power),
                            },
                    'popup': f'Transmitted Power: {power_GU[i, j].x*x_GU[i, j].x/1000.:.2f} kW\nDissipated Power: {power_loss_GU[i, j].x*x_GU[i, j].x/1000:.2e} '      
                }                                   
            }
            for i in range(len(generator_positions))
            for j in range(len(user_positions))
            if x_GU[i,j].x > 0.5
        ]
        all_features.extend(points_lines)        


        points_lines = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    'coordinates': [[float(generator_positions[i, 1]), float(generator_positions[i, 0])],
                                    [float(storage_positions[j, 1]), float(storage_positions[j, 0])]]
                },
                "properties": {
                    "times": [current_time.isoformat(), current_time.isoformat()],                    
                    "style": {
                            "color": 'blue',
                            "weight": 50*power_GS[i, j].x * x_GS[i, j].x / (scaling_factor*max_generators_power) 
                            },
                    'popup': f'Transmitted Power: {power_GS[i, j].x*x_GS[i, j].x/1000.:.2f} kW'       
                }
            }
            for i in range(len(generator_positions))
            for j in range(len(storage_positions))
            if x_GS[i,j].x > 0.5
        ]
        all_features.extend(points_lines) 


        points_lines = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    'coordinates': [[float(storage_positions[i, 1]), float(storage_positions[i, 0])],
                                    [float(user_positions[j, 1]), float(user_positions[j, 0])]]
                },
                "properties": {
                    "times": [current_time.isoformat(), current_time.isoformat()],                      
                    "style": {
                            "color": 'green',
                            "weight": 50*power_SU[i, j].x * x_SU[i, j].x / (scaling_factor*max_generators_power)
                            },
                    'popup': f'Transmitted Power: {power_SU[i, j].x*x_SU[i, j].x/1000.:.2f} kW'         
                }
            }
            for i in range(len(storage_positions))
            for j in range(len(user_positions))
            if x_SU[i,j].x > 0.5
        ]
        all_features.extend(points_lines) 


        # Create a Folium map
        m1 = folium.Map(location=[center_lat, center_lon], zoom_start=14)
        # Add markers for generators, storage devices, and users
        for i in range(num_generators):
            total_power = sum(power_GS[i, j].x * x_GS[i, j].x for j in range(num_storage)) + sum(power_GU[i, j].x * x_GU[i, j].x for j in range(num_users))
            folium.Marker([generator_positions[i, 0], generator_positions[i, 1]], popup=f'Generator {i}\nTotal Power: {total_power}', icon=folium.Icon(color='orange')).add_to(m1)
        for i in range(num_storage):
            total_power = sum(power_SU[i, j].x * x_SU[i, j].x for j in range(num_users))
            folium.Marker([storage_positions[i, 0], storage_positions[i, 1]], popup=f'Storage {i}\nTotal Power: {total_power}', icon=folium.Icon(color='blue')).add_to(m1)
        for i in range(num_users):
            total_power = sum(power_GU[j, i].x * x_GU[j, i].x for j in range(num_generators)) + sum(power_SU[j, i].x*x_SU[j, i].x for j in range(num_storage))
            #folium.Marker([user_positions[i, 0], user_positions[i, 1]], popup=f'User {i}\nTotal Power: {total_power}', icon=folium.Icon(color='red')).add_to(m1)
            folium.Marker([user_positions[i, 0], user_positions[i, 1]], popup=f'User {i}\nRequested power: {user_requirements[i]/1000:.2f} kW', icon=folium.Icon(color='red')).add_to(m1)


        # Add lines for transmission lines
        for i in range(num_generators):
            for j in range(num_users):
                power = power_GU[i, j].x * x_GU[i, j].x 
                dissipated_power = power_loss_GU[i, j].x * x_GU[i, j].x 
                folium.PolyLine([[generator_positions[i, 0], generator_positions[i, 1]], [user_positions[j, 0], user_positions[j, 1]]], tooltip=f'Transmission Line\nPower: {power}\nDissipated Power: {dissipated_power:.2e}', color='black', weight= 50* power / (scaling_factor*max_generators_power)).add_to(m1)
        #print()
        for i in range(num_generators):
            for j in range(num_storage):
                power = power_GS[i, j].x
                dissipated_power = power_loss_GS[i, j].x * x_GS[i, j].x
                #folium.PolyLine([[generator_positions[i, 0], generator_positions[i, 1]], [storage_positions[j, 0], storage_positions[j, 1]]], tooltip=f'Transmission Line\nPower: {power}\nDissipated Power: {dissipated_power:.2e}', color='yellow', weight= 60*x_GS[i, j].x).add_to(m1)
                folium.PolyLine([[generator_positions[i, 0], generator_positions[i, 1]], [storage_positions[j, 0], storage_positions[j, 1]]], tooltip=f'Transmission Line\nPower: {power}\nDissipated Power: {dissipated_power:.2e}', color='blue', weight= 50*x_GS[i, j].x*power/(scaling_factor*max_generators_power)).add_to(m1)
        #print()
        for i in range(num_storage):
            for j in range(num_users):
                power = power_SU[i, j].x
                dissipated_power = power_loss_SU[i, j].x * x_SU[i, j].x
                #print(i,j,x_SU[i, j], power)
                folium.PolyLine([[storage_positions[i, 0], storage_positions[i, 1]], [user_positions[j, 0], user_positions[j, 1]]], tooltip=f'Transmission Line\nPower: {power}\nDissipated Power: {dissipated_power}', color='black', weight= 50* power / (scaling_factor*max_generators_power)).add_to(m1)

        # Save the map as an HTML file
        curr_dir = os.getcwd()
        name = current_time.isoformat()
        name = name.replace(':', '-')
        print(name)
        curr_dir = os.path.join(curr_dir, "map_" + str(ii) + '_' + name + ".html")
        #print(curr_dir)
        m1.save( curr_dir )

    

        # Calculate total generated power
        total_generated_power = sum(generator_power)

        # Calculate power delivered to users
        power_delivered_to_users = sum(power_GU[i, j].x * x_GU[i, j].x for i in range(len(generator_positions)) for j in range(len(user_positions)))

        # Calculate power delivered to storage
        power_delivered_to_storage = sum(power_GS[i, j].x * x_GS[i, j].x for i in range(len(generator_positions)) for j in range(len(storage_positions)))

        # Calculate excess power
        excess_power.append(total_generated_power - power_delivered_to_users - power_delivered_to_storage)



    TimestampedGeoJson(
    {"type": "FeatureCollection", 
    "features": all_features
    },
    period="PT1H",  # display each feature for x minutes
    add_last_point=False,
    auto_play=False,
    loop=False,
    max_speed=5,  # decrease the max speed to make the simulation faster
    loop_button=True,
    date_options="YYYY/MM/DD HH:MM:SS",
    time_slider_drag_update=True,
    duration='PT50M',
    ).add_to(m)

    # Save the map as an HTML file
    curr_dir = os.getcwd()
    curr_dir = os.path.join(curr_dir, "map.html")
    m.save(curr_dir)
    print(curr_dir)

    for ii in solutions:
        print(ii)
  

if __name__ == '__main__':
    main()  