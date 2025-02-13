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
    area_m2 = Persons_Powerreq_PVcapacity_gdf['area_m2']

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
        storage_positions[i, 0] = generator_positions[generator_index, 0] + 200 * math.cos(angle) / 111111  # convert meters to degrees
        storage_positions[i, 1] = generator_positions[generator_index, 1] + 200 * math.sin(angle) / 111111  # convert meters to degrees

    generators_power = []
    for ig in range(num_generators):
        lat, lon = generator_positions[ig]
        df = fetch_pv_data(lat, lon)
        area_m2_ig = area_m2.iloc[ig]
        if df is None:
            print(f"No data for generator {ig} at position ({lat}, {lon})")
        else:
            df['Time'] = pd.to_datetime(df['Time'], format='%Y%m%d:%H%M')
            # Calculate the absolute difference between each timestamp and the given timestamp
            diff = np.abs(df['Time'] - start_dt)
            # Find the index of the row with the smallest absolute difference
            idx = np.argmin(diff)

            df_24h = df.iloc[idx:idx+24]
            hourly_power = area_m2_ig * 0.75 * df_24h['PV Power'].values # 75% of surface covered
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
    storage_power_capacity = np.random.uniform(5000, 10000, num_storage)
    stored_energy = np.random.uniform(30000, 40000, num_storage)
    storage_Emax = np.random.uniform(40000, 50000, num_storage) # Wh
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

def generate_user_requirements():
    # Define parameters
    num_users = 7
    max_amplitudes_users = np.random.uniform(1000, 3000, size=num_users)
    time = np.arange(0, 24, 1)
    sigma = 1  # standard deviation of the Gaussian
    mu = np.pi  # mean of the Gaussian
    
    # Generate users requirements signals
    users_requirements = np.zeros((num_users, len(time)))
    noise_std = 100
    for i in range(num_users):
        frequency = 1/24.
        # Generate Gaussian function
        x = time
        np.random.seed(42)
        mu = 12 + np.random.normal(-3, 3)  # mean of the Gaussian with small noise
        users_requirements[i] = max_amplitudes_users[i]*np.exp(-(x - mu)**2 / (2 * sigma**2))

        #x = 2 * np.pi * frequency * time
        #mu = np.pi + np.random.normal(-0.8, 0.8)  # mean of the Gaussian with small noise
        #users_requirements[i] = max_amplitudes_users[i]*np.exp(-(x - mu)**2 / (2 * sigma**2))
        users_requirements[i] += noise_std * np.random.normal(size=len(time))
        users_requirements[i] = np.where(users_requirements[i] < 0, 0, users_requirements[i])
        inverted_gauss = np.ones(len(time))
        inverted_gauss = np.max(users_requirements[i])*inverted_gauss
        users_requirements[i] = inverted_gauss - users_requirements[i]
    return users_requirements

def get_opacity_for_charging_state(charging_state):
    return charging_state / 100


def generate_plot(generated_power, delivered_power_to_storage, delivered_power_to_users, stored_energies):
    hours = [str(i) for i in range(24)]
    variables = {
        'Gen. power (kW)': generated_power,
        'Power to users (kW)': delivered_power_to_users,
        'Power to storage (kW)': delivered_power_to_storage,
        'Stored energy (kWh)': stored_energies
    }

    x = np.arange(24)  # the label locations
    width = 0.25  # the width of the bars

    fig, axs = plt.subplots(len(variables), 1, figsize=(6, 8), layout='constrained')

    colormaps = ['Greens', 'Reds', 'Blues', 'Greys']

    for i, (attribute, measurement) in enumerate(variables.items()):
        ax = axs[i]
        for j, community in enumerate(measurement):
            rects = ax.bar(x + j * width, community, width, label=f'Community {j+1}', color=plt.cm.get_cmap(colormaps[i])(0.5 + j * 0.25))
        ax.set_ylabel(attribute, fontweight='bold')
        #ax.set_title(attribute)
        ax.set_xticks(x + width)
        ax.set_xticklabels(hours)
        ax.legend(loc='upper left', ncols=3)
        ax.set_ylim(0, 1.2*np.max(np.max(measurement, axis=1)))
        if i == len(variables) - 1:
            ax.set_xlabel('Time (hours)', fontweight='bold')

    curr_dir = os.getcwd()
    plot_file = os.path.join(curr_dir, "plot_panel.png")
    plt.savefig(plot_file, dpi=300)
    return plot_file

def main():
    # Define the number of nodes
    num_generators = 3
    num_users = 7
    num_storage = 3
    # Define the center of the 2km2 square in Andoain
    center_lat = 43.2169548061656
    center_lon = -2.02130803900803

    scaling_factor = 0.8

    user_pow_requ = generate_user_requirements()

    # Create a Folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # Generate inputs
    # Midnight:
    #start_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    #time_step = datetime.timedelta(minutes=15)  
    time_step = datetime.timedelta(hours=1) 
    start_datetime = datetime.datetime(2022, 7, 1, 0, 10, 0)
    print('start_datetime:', start_datetime)
    #input_data = generate_inputs(num_generators, num_users, num_storage,
    #                         generators_power user_pow_requ)
    input_data = generate_inputs(num_generators, num_users, num_storage,
                                 start_datetime, user_pow_requ)

    # Create a list to store the 24 results
    solutions = []
    all_features = []
    excess_power = []

    generators_power = input_data['generator_power']
    max_generators_power = max(max(row) for row in generators_power)

    Delivered_power_to_storage_hour = np.zeros((generators_power.shape[0], generators_power.shape[1]))
    Delivered_power_to_users_hour = np.zeros((generators_power.shape[0], generators_power.shape[1]))
    Energy_stored_hour = np.zeros((generators_power.shape[0], generators_power.shape[1]))

    for ii in range(len(generators_power[0])):
        generator_power = generators_power[:, ii][:3]
        user_requirements = user_pow_requ[:, ii]

        #current_time = start_time + ii * time_step
        current_time = start_datetime + ii * time_step

        print(generator_power)
        print(user_requirements)
        print(input_data['stored_energy'])
        try:
            print(input_data['stored_energy'].x)
        except:
            pass

        # Generate the inputs
        input_data['generator_power'] = generator_power
        input_data['user_requirements'] = user_requirements
        #input_data['stored_energy'] = stored_E
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
        else:
            print()
            print("The model is feasible")
            print()
            solutions.append([ii,'Feasible'])

        # Update the stored energy
        #input_data['stored_energy'] = stored_E
        stored_E_values = [stored_E[j].x for j in range(num_storage)]
        input_data['stored_energy'] = stored_E_values 
        #input_data['stored_energy'] = stored_E 

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
                                        'radius': 5 + (10 * (stored_E[j].x/storage_Emax[j])),  # Calculate radius based on charging state
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
                            "weight": 30* power_GU[i, j].x * x_GU[i, j].x  / (scaling_factor*max_generators_power),
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
                            "weight": 30*power_GS[i, j].x * x_GS[i, j].x / (scaling_factor*max_generators_power) 
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
                            "weight": 30*power_SU[i, j].x * x_SU[i, j].x / (scaling_factor*max_generators_power)
                            },
                    'popup': f'Transmitted Power: {power_SU[i, j].x*x_SU[i, j].x/1000.:.2f} kW'         
                }
            }
            for i in range(len(storage_positions))
            for j in range(len(user_positions))
            if x_SU[i,j].x > 0.5
        ]
        all_features.extend(points_lines) 

        # Calculate total generated power
        total_generated_power = sum(generator_power)

        # Calculate power delivered to users
        power_delivered_to_users = sum(power_GU[i, j].x * x_GU[i, j].x for i in range(len(generator_positions)) for j in range(len(user_positions)))

        # Calculate power delivered to storage
        power_delivered_to_storage = sum(power_GS[i, j].x * x_GS[i, j].x for i in range(len(generator_positions)) for j in range(len(storage_positions)))

        # Calculate excess power
        excess_power.append(total_generated_power - power_delivered_to_users - power_delivered_to_storage)

        delivered_power_to_storage = np.zeros(num_generators)
        delivered_power_to_users = np.zeros(num_generators)
        for jj in range(num_generators):
            delivered_power_to_storage[jj] = sum(power_GS[jj, j].x * x_GS[jj, j].x for j in range(num_storage))
            delivered_power_to_users[jj] = sum(power_GU[jj, j].x * x_GU[jj, j].x for j in range(num_users)) 


        Delivered_power_to_storage_hour[:,ii] = delivered_power_to_storage
        Delivered_power_to_users_hour[:,ii] = delivered_power_to_users
        Energy_stored_hour[:,ii] = stored_E_values 
        
    plot_dir = generate_plot(generators_power, Delivered_power_to_storage_hour, Delivered_power_to_users_hour, Energy_stored_hour, )

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


    from folium import raster_layers
    raster_layers.ImageOverlay(
        image=plot_dir,
        bounds=[[43.2292500962932, -1.979357144462435], [43.19312849807213, -1.94]],
        opacity=0.5,
        interactive=False,
        cross_origin=False,
        zindex=1,
        image_size=[100, 300]  # smaller image size
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