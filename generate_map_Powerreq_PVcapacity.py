import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import folium
import branca.colormap as cm
import os

def process_data(andoain_gdf, selected, aver_power, aver_energy):
    selected['centroid'] = selected['geometry'].centroid
    geometries = []
    areas_m2_values = []
    seccion_values = []
    for index, row in andoain_gdf.iterrows():
        seccion = row['Seccion']
        areas_persons = selected[selected.geometry.within(row['geometry'])]
        selected_projected = areas_persons.to_crs(epsg=3857)  # transform to Web Mercator
        areas_m2 = abs(selected_projected.geometry.area)
        geometries.extend(areas_persons.geometry)
        areas_m2_values.extend(areas_m2)
        seccion_values.extend([seccion] * len(areas_m2))

    new_selected = gpd.GeoDataFrame({'Seccion': seccion_values, 'area_m2': areas_m2_values}, geometry=geometries)

    new_selected['fraction'] = 0
    for index, row in new_selected.iterrows():
        row['fraction'] = row['area_m2'] / new_selected[new_selected['Seccion'] == row['Seccion']]['area_m2'].sum()
        total_persons_seccion = andoain_gdf[andoain_gdf['Seccion'] == row['Seccion']]['Total']
        total_persons_seccion = total_persons_seccion.iloc[0]
        #print(row['Seccion'], row['fraction'], total_persons_seccion)
        persons = int(round(row['fraction']*total_persons_seccion))
        new_selected.at[index, 'persons'] = int(persons)
        new_selected.at[index, 'required_power'] = round(pd.to_numeric(persons * aver_power), 3)
        new_selected.at[index, 'available_energy'] = round(pd.to_numeric(row['area_m2'] * aver_energy), 3)
        new_selected.at[index, 'area_m2'] = round(pd.to_numeric(row['area_m2']), 3)
        new_selected.at[index, 'fraction'] = round(pd.to_numeric(row['fraction']), 3)
        #print(new_selected[['fraction']][:20])
    #print((new_selected['fraction'] == 0).all())
    return new_selected

def create_map(selected,new_selected):
    m = folium.Map(location=[selected.geometry.centroid.y.mean(), selected.geometry.centroid.x.mean()], zoom_start=14)

    new_selected.crs = selected.crs
    new_selected = new_selected.to_crs(epsg=4326)

    # Define the color scale
    min_value = new_selected['persons'].min() 
    max_value = new_selected['persons'].max() 

    # Create a colormap
    colormap1 = cm.LinearColormap(['green', 'orange', 'red'], vmin=min_value, vmax=max_value)

    def get_color(feature):
        value = feature['properties']['persons']
        return colormap1(value)

    folium.GeoJson(
        new_selected.__geo_interface__,
        name='Power Required',
        style_function=lambda feature: {
            'fillColor': get_color(feature),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.5
        },
        tooltip=folium.features.GeoJsonTooltip(fields=['persons','required_power', 'area_m2'],
                                            aliases=['Persons ','Power Required (kW): ','area (m2) '],
                                            labels=True)
    ).add_to(m)                                          

    # Define the color scale
    min_value = new_selected['available_energy'].min()
    max_value = new_selected['available_energy'].max()

    # Create a colormap
    colormap2 = cm.LinearColormap(['yellow', 'red'], vmin=min_value, vmax=max_value)

    def get_color2(feature):
        value = feature['properties']['available_energy']
        return colormap2(value)

    folium.GeoJson(
        new_selected.__geo_interface__,
        name='Available Energy',
        style_function=lambda feature: {
            'fillColor': get_color2(feature),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.5
        },
        tooltip=folium.features.GeoJsonTooltip(fields=['available_energy','area_m2'],
                                            labels=True,
                                            aliases=['PV Energy capacity (kWh/day):',
                                                        'area (m2):'])                         
                                                        
    ).add_to(m)

    # Add the layer control
    folium.LayerControl().add_to(m)

    return m

def main():

    aver_power = 2.41888 # kW

    # Daily Output (kWh) = Wattage (W) x Hours of Sunlight x Efficiency
    # Daily Output (kWh) = 300 W x 5 hours x 0.2 (assuming a 20% efficiency) = 0.3 kWh
    # 0.3 kWh x 30 = 9 kWh per month,
    # 9 kWh x 12 = 108 kWh per year.


    aver_energy =  0.3 # kWh/m2
    gen_dir = os.getcwd()
    #curr_dir = os.path.join(curr_dir, "map.html")
    #gen_dir = "C:/Users/gfotidellaf/Desktop/E-grids/"
    #dir_shp = gen_dir  + "manual_selection"
    #andoain_gdf = gpd.read_file(r"C:\Users\gfotidellaf\Desktop\E-grids\census_data\geojson\sections_gipuzkoa_demographic_both.geojson")

    dir_shp = os.path.join(gen_dir, "manual_selection")
    andoain_gdf = gpd.read_file(os.path.join(gen_dir, "census_data", "geojson", "sections_gipuzkoa_demographic_both.geojson"))
    selected = gpd.read_file(dir_shp)
    new_selected = process_data(andoain_gdf, selected, aver_power, aver_energy)
    new_selected.to_file(os.path.join(gen_dir,'Gipuzkoa_Shapefile','Persons_Powerreq_PVcapacity','persons_Powerreq_PVcapacity.shp'))
    #print(new_selected[['fraction']][:20])
    m = create_map(selected, new_selected)
    #m.save('C:\\Users\\gfotidellaf\\Desktop\\E-grids\\map.html')
    m.save(os.path.join(gen_dir, "map_Powerreq_PVcapacity.html"))

if __name__ == '__main__':
    main()