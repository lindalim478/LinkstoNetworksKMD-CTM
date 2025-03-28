import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely import wkt

def addNodeGeom(links_df, nodes_df):
    """
    Merge node coordinates into the links DataFrame to create geometry.
    """
    def map_node_geometry(ldf, linkcol, ndf, nodecol):
        return ldf[linkcol].map(ndf.set_index('NODE_ID')[nodecol])
    
    colnamelist = [
        ['ref_lat', 'REF_IN_ID', 'LAT'],
        ['ref_long', 'REF_IN_ID', 'LON'],
        ['nref_lat', 'NREF_IN_ID', 'LAT'],
        ['nref_long', 'NREF_IN_ID', 'LON']
    ]
    for new_col, link_id_col, node_coord_col in colnamelist:
        links_df[new_col] = map_node_geometry(links_df, link_id_col, nodes_df, node_coord_col)
    return links_df


def filter_data_for_corridors(
    speeds_file, flow_file, density_file, 
    downtown_gdf, mid_gdf, outer_gdf, 
    output_prefix="filtered_"
):
    """
    Filters speeds, flows, and density files for only the links in the given corridors.
    """
    print("Pre-filtering data for CTM...")
    all_link_ids = set(
        list(downtown_gdf["LINK_ID"]) +
        list(mid_gdf["LINK_ID"]) +
        list(outer_gdf["LINK_ID"])
    )
    speeds_df = pd.read_csv(speeds_file)
    flows_df = pd.read_csv(flow_file)
    density_df = pd.read_csv(density_file)

    filtered_speeds = speeds_df[speeds_df["link_id"].isin(all_link_ids)]
    filtered_flows = flows_df[flows_df["link_id"].isin(all_link_ids)]
    filtered_densities = density_df[density_df["link_id"].isin(all_link_ids)]

    fspeeds = f"{output_prefix}{speeds_file}"
    fflows  = f"{output_prefix}{flow_file}"
    fdens   = f"{output_prefix}{density_file}"

    filtered_speeds.to_csv(fspeeds, index=False)
    filtered_flows.to_csv(fflows, index=False)
    filtered_densities.to_csv(fdens, index=False)

    print(f"Filtered speeds:   {fspeeds}   ({len(filtered_speeds)} links)")
    print(f"Filtered flows:    {fflows}    ({len(filtered_flows)} links)")
    print(f"Filtered density:  {fdens}    ({len(filtered_densities)} links)")

    return fspeeds, fflows, fdens

def project_and_sort(gdf, sortby='cx', ascending=True):
    """
    Re-project a GeoDataFrame to EPSG:26910, compute centroids (cx, cy),
    then sort by a chosen column in ascending/descending order.
    """
    if gdf.empty:
        return []
    gdf_proj = gdf.to_crs(epsg=26910)
    gdf_proj['cx'] = gdf_proj.geometry.centroid.x
    gdf_proj['cy'] = gdf_proj.geometry.centroid.y
    out = gdf_proj.sort_values(by=sortby, ascending=ascending)
    return out['LINK_ID'].tolist()
