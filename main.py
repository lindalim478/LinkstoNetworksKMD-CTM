import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from shapely import wkt

from data_preparation import addNodeGeom, filter_data_for_corridors
from ctm import apply_ctm
from koopman import run_koopman_modes, check_stability

def main():
    # 1) Load Node & Link data
    nodes = pd.read_csv('data/sf_nodes.csv')
    links = pd.read_csv('data/sf_links.csv')

    # 2) Enrich link geometry
    links = addNodeGeom(links, nodes)
    links.dropna(subset=['ref_lat','ref_long','nref_lat','nref_long'], inplace=True)
    links['geometry'] = links.apply(
        lambda row: LineString([(row['ref_long'], row['ref_lat']),
                                (row['nref_long'], row['nref_lat'])]), axis=1
    )
    links_gdf = gpd.GeoDataFrame(links, geometry='geometry', crs="EPSG:4326")

    # 3) Load corridor definitions
    downtown_df = pd.read_excel("data/downtown.xlsx")
    downtown_df["geometry"] = downtown_df["geometry"].apply(wkt.loads)
    downtown_gdf = gpd.GeoDataFrame(downtown_df, geometry="geometry", crs="EPSG:4326")
    if "LENGTH(feet)" in downtown_gdf.columns:
        link_lengths_downtown = downtown_gdf["LENGTH(feet)"].fillna(0).values
        link_positions_downtown = np.cumsum(link_lengths_downtown) - (link_lengths_downtown / 2)
    else:
        link_positions_downtown = np.arange(len(downtown_gdf))

    mid_df = pd.read_excel("data/mid.xlsx")
    mid_df["geometry"] = mid_df["geometry"].apply(wkt.loads)
    mid_gdf = gpd.GeoDataFrame(mid_df, geometry="geometry", crs="EPSG:4326")

    outer_df = pd.read_excel("data/outer2.xlsx")
    outer_df["geometry"] = outer_df["geometry"].apply(wkt.loads)
    outer_gdf = gpd.GeoDataFrame(outer_df, geometry="geometry", crs="EPSG:4326")

    # 4) Example: node definitions (dummy link IDs)
    nodes_definition = {
        'Hwy101_Hwy880': {'incoming': [945459409, 23716147, 947277948], 'outgoing': [945459411]},
        'Hwy280_Hwy880': {'incoming': [28431231], 'outgoing': [1111849260, 783188341]},
        'Hwy237_Hwy101': {'incoming': [743932518], 'outgoing': [759432794, 23753427]},
    }

    # 5) Filter data for corridors
    fspeeds, fflows, fdens = filter_data_for_corridors(
        speeds_file="data/avg_speeds.csv",
        flow_file="data/links_flow.csv",
        density_file="data/links_density.csv",
        downtown_gdf=downtown_gdf,
        mid_gdf=mid_gdf,
        outer_gdf=outer_gdf
    )

    # 6) Apply CTM
    updated_speeds_dict = apply_ctm(
        downtown_gdf=downtown_gdf,
        mid_gdf=mid_gdf,
        outer_gdf=outer_gdf,
        nodes=nodes_definition,
        speeds_file=fspeeds,
        flow_file=fflows,
        density_file=fdens,
        dt=30,
        propagation_steps=3,
        verbose=True
    )

    # 7) Run Koopman on, e.g., the original speeds file for the downtown corridor
    A_downtown, eigvals_downtown = run_koopman_modes(
        gdf_loop=downtown_gdf,
        link_positions_feet=link_positions_downtown,
        csv_file="data/avg_speeds.csv",
        loop_label="Downtown_Original",
        skip_cols=1,
        delay=5,
        delt=15,
        mode_range=(1,10)
    )

    # 8) Check stability
    check_stability(eigvals_downtown, loop_label="Downtown Original")

    print("All done.")

if __name__ == "__main__":
    main()
