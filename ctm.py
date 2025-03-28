import numpy as np
import pandas as pd

def propagate_upstream(
    link_id, upstream_links, current_flows, current_densities, current_speeds,
    free_flow_speeds, link_props, scale_factor, dt
):
    """
    Propagates congestion upstream using exponential decay with distance,
    retrieving each link's jam density from 'link_props' if present.
    """
    if link_id not in upstream_links:
        return

    links_to_update = upstream_links[link_id]
    current_length = link_props[link_id].get("LENGTH(meters)", 100) if link_id in link_props else 100
    accumulated_distance = 0
    prop_factor = scale_factor

    for up_link in links_to_update:
        if up_link in link_props and up_link in current_flows and up_link in current_densities:
            up_length   = link_props[up_link].get("LENGTH(meters)", 100)
            jam_density = link_props[up_link].get("JAM_DENSITY", 150)

            link_distance       = (current_length + up_length) / 2
            accumulated_distance += link_distance
            decay_constant       = 1000
            distance_factor      = np.exp(-accumulated_distance / decay_constant)
            prop_factor          = scale_factor * distance_factor

            if prop_factor < 0.05:
                break

            original_flow = current_flows[up_link]
            reduced_flow  = original_flow * (1 - (1 - scale_factor) * prop_factor)
            current_flows[up_link] = reduced_flow

            length_km = link_props[up_link].get("LENGTH(meters)", 100)/1000
            lanes     = link_props[up_link].get("NUM_PHYS_LANES", 1)
            trapped_vehicles  = (original_flow - reduced_flow) * dt
            density_increase  = trapped_vehicles / (length_km * lanes)
            current_densities[up_link] += density_increase

            if up_link in free_flow_speeds and up_link in current_speeds:
                free_flow = free_flow_speeds[up_link]
                if current_densities[up_link] < jam_density:
                    new_speed = free_flow * (1 - current_densities[up_link] / jam_density)
                    current_speeds[up_link] = max(0.1, new_speed)
                else:
                    current_speeds[up_link] = 0.1

            current_length = up_length


def propagate_downstream(
    link_id, downstream_links, current_flows, current_densities, current_speeds,
    free_flow_speeds, link_props, dt
):
    """
    Propagates changes downstream, retrieving jam density from 'link_props'
    if present for each link.
    """
    if link_id not in downstream_links or link_id not in current_flows:
        return

    links_to_update = downstream_links[link_id]
    current_flow    = current_flows[link_id]
    current_length  = link_props[link_id].get("LENGTH(meters)", 100) if link_id in link_props else 100
    accumulated_distance = 0
    prop_factor = 1.0

    for down_link in links_to_update:
        if down_link in link_props and down_link in current_flows and down_link in current_densities:
            down_length   = link_props[down_link].get("LENGTH(meters)", 100)
            jam_density   = link_props[down_link].get("JAM_DENSITY", 150)

            link_distance = (current_length + down_length) / 2
            accumulated_distance += link_distance
            decay_constant = 1500
            distance_factor = np.exp(-accumulated_distance / decay_constant)
            prop_factor = distance_factor

            if prop_factor < 0.05:
                break

            original_down_flow = current_flows[down_link]
            current_flows[down_link] = (
                original_down_flow * (1 - prop_factor) + current_flow * prop_factor
            )

            if down_link in free_flow_speeds and down_link in current_densities:
                free_flow = free_flow_speeds[down_link]
                if free_flow > 0:
                    q = current_flows[down_link]
                    k_j = jam_density
                    v_f = free_flow

                    # fundamental diagram approach
                    discriminant = max(0, 1 - 4*q / (k_j * v_f))
                    new_density = k_j * (1 - np.sqrt(discriminant)) / 2

                    current_densities[down_link] = (
                        current_densities[down_link] * (1 - prop_factor) + new_density * prop_factor
                    )

                    if current_densities[down_link] < jam_density:
                        new_speed = v_f * (1 - current_densities[down_link] / jam_density)
                        current_speeds[down_link] = max(0.1, new_speed)
                    else:
                        current_speeds[down_link] = 0.1

            current_length = down_length


def apply_ctm(
    downtown_gdf, mid_gdf, outer_gdf, nodes,
    speeds_file="filtered_avg_speeds.csv",
    flow_file="filtered_links_flow.csv",
    density_file="filtered_links_density.csv",
    dt=30,
    process_nodes=None,
    time_range=None,
    propagation_steps=3,
    verbose=True
):
    """
    Apply Cell Transmission Model with node constraints, using link-specific jam density if available.
    """
    if verbose:
        print("Applying CTM with link-specific jam densities (if present)...")

    speeds_df  = pd.read_csv(speeds_file)
    flows_df   = pd.read_csv(flow_file)
    density_df = pd.read_csv(density_file)

    speeds_df.set_index("link_id", inplace=True)
    flows_df.set_index("link_id", inplace=True)
    density_df.set_index("link_id", inplace=True)

    # Skip the first column of times if needed
    time_cols = speeds_df.columns[1:]

    if time_range is not None:
        start_idx, end_idx = time_range
        time_cols = time_cols[start_idx:end_idx]

    # Collate link properties
    all_links = pd.concat([
        downtown_gdf[["LINK_ID", "LENGTH(meters)", "CAPACITY(veh/hour)", "NUM_PHYS_LANES", "SPEED_KPH"]],
        mid_gdf[["LINK_ID", "LENGTH(meters)", "CAPACITY(veh/hour)", "NUM_PHYS_LANES", "SPEED_KPH"]],
        outer_gdf[["LINK_ID", "LENGTH(meters)", "CAPACITY(veh/hour)", "NUM_PHYS_LANES", "SPEED_KPH"]]
    ])
    link_props = all_links.set_index("LINK_ID").to_dict(orient="index")

    if process_nodes is not None:
        nodes_to_process = {k: nodes[k] for k in process_nodes if k in nodes}
    else:
        nodes_to_process = nodes

    # Connectivity
    def create_connectivity(links_list):
        upstream = {}
        downstream = {}
        for i in range(len(links_list)):
            link_id = links_list[i]
            if i > 0:
                upstream[link_id] = links_list[max(0, i - propagation_steps): i]
            else:
                upstream[link_id] = []
            if i < len(links_list) - 1:
                downstream[link_id] = links_list[i+1 : min(len(links_list), i+1+propagation_steps)]
            else:
                downstream[link_id] = []
        return upstream, downstream

    downtown_links = downtown_gdf["LINK_ID"].tolist()
    mid_links      = mid_gdf["LINK_ID"].tolist()
    outer_links    = outer_gdf["LINK_ID"].tolist()

    dt_up, dt_down = create_connectivity(downtown_links)
    mid_up, mid_down = create_connectivity(mid_links)
    ot_up,  ot_down  = create_connectivity(outer_links)

    upstream_links   = {**dt_up, **mid_up, **ot_up}
    downstream_links = {**dt_down, **mid_down, **ot_down}

    if verbose:
        print(f"Connectivity: {len(upstream_links)} links with upstream/downstream definitions")

    # Precompute free-flow speeds
    free_flow_speeds = {}
    for lid, props in link_props.items():
        kph = props.get("SPEED_KPH", 100)
        free_flow_speeds[lid] = kph / 3.6

    updated_speeds    = speeds_df.copy()
    updated_flows     = flows_df.copy()
    updated_densities = density_df.copy()

    directly_affected_links = set()
    for node_name, node_data in nodes.items():
        directly_affected_links.update(node_data['incoming'])
        directly_affected_links.update(node_data['outgoing'])

    propagation_links = set()
    for link_id in directly_affected_links:
        if link_id in upstream_links:
            propagation_links.update(upstream_links[link_id])
        if link_id in downstream_links:
            propagation_links.update(downstream_links[link_id])

    all_affected_links = directly_affected_links.union(propagation_links)

    if verbose:
        print(f"Affected links: {len(directly_affected_links)} direct, {len(propagation_links)} propagation")

    fine_speeds    = {lid: [] for lid in speeds_df.index if lid in all_affected_links}
    fine_flows     = {lid: [] for lid in flows_df.index if lid in all_affected_links}
    fine_densities = {lid: [] for lid in density_df.index if lid in all_affected_links}

    for node_name, node_data in nodes_to_process.items():
        if verbose:
            print(f"Node: {node_name}")
        incoming_links = node_data['incoming']
        outgoing_links = node_data['outgoing']

        valid_incoming = [lk for lk in incoming_links if lk in link_props]
        valid_outgoing = [lk for lk in outgoing_links if lk in link_props]
        if not (valid_incoming and valid_outgoing):
            if verbose:
                print(f"  Missing link props for node {node_name}, skipping.")
            continue

        # Precompute for incoming
        incoming_props = {}
        for lk in valid_incoming:
            props = link_props[lk]
            incoming_props[lk] = {
                'length': props.get("LENGTH(meters)", 100) / 1000,
                'lanes': props.get("NUM_PHYS_LANES", 1),
                'free_flow': free_flow_speeds.get(lk, 27.78)
            }

        # Precompute for outgoing
        outgoing_props = {}
        for lk in valid_outgoing:
            props = link_props[lk]
            outgoing_props[lk] = {
                'capacity': props.get("CAPACITY(veh/hour)", 2000) / 3600,
                'lanes': props.get("NUM_PHYS_LANES", 1),
                'length': props.get("LENGTH(meters)", 100)
            }

        total_out_capacity = sum(
            outgoing_props[lk]['capacity'] * outgoing_props[lk]['lanes']
            for lk in outgoing_props
        )
        out_demand_share = {}
        if total_out_capacity > 0:
            for lk in outgoing_props:
                cap   = outgoing_props[lk]['capacity']
                lanes = outgoing_props[lk]['lanes']
                out_demand_share[lk] = (cap * lanes) / total_out_capacity

        for t_idx, time_col in enumerate(time_cols):
            if verbose and t_idx % 10 == 0:
                print(f"  Time: {time_col}")
            steps = int(15*60 / dt)

            current_flows = {lk: flows_df.loc[lk, time_col] for lk in all_affected_links if lk in flows_df.index}
            current_dens  = {lk: density_df.loc[lk, time_col] for lk in all_affected_links if lk in density_df.index}
            current_spd   = {lk: speeds_df.loc[lk, time_col] for lk in all_affected_links if lk in speeds_df.index}

            # Initialize fine-resolution recording
            for lk in all_affected_links:
                if lk in current_flows:    fine_flows[lk].append(current_flows[lk])
                if lk in current_dens:     fine_densities[lk].append(current_dens[lk])
                if lk in current_spd:      fine_speeds[lk].append(current_spd[lk])

            # CTM each 30-second sub-step
            for stp in range(1, steps):
                total_demand = sum(current_flows[lk]*dt for lk in valid_incoming if lk in current_flows)
                total_supply = 0
                for lk in valid_outgoing:
                    if lk in outgoing_props and lk in current_dens:
                        jam_density = link_props[lk].get("JAM_DENSITY", 150)
                        capacity    = outgoing_props[lk]['capacity']
                        lanes       = outgoing_props[lk]['lanes']
                        length_m    = outgoing_props[lk]['length']
                        density     = current_dens[lk]
                        remain_dens = jam_density - density
                        remain_cap  = remain_dens * (length_m / 1000) * lanes
                        link_supply = min(capacity*dt*lanes, remain_cap)
                        total_supply += link_supply

                if total_demand > total_supply and total_supply > 0:
                    scale_factor = total_supply / total_demand
                    for lk in valid_incoming:
                        if lk in current_flows and lk in current_dens and lk in incoming_props:
                            original_flow = current_flows[lk]
                            reduced_flow  = original_flow * scale_factor
                            current_flows[lk] = reduced_flow
                            length_km = incoming_props[lk]['length']
                            lanes     = incoming_props[lk]['lanes']
                            free_flow = incoming_props[lk]['free_flow']
                            trapped   = (original_flow - reduced_flow) * dt
                            dens_incr = trapped / (length_km*lanes)
                            jam_density = link_props[lk].get("JAM_DENSITY", 150)
                            current_dens[lk] += dens_incr
                            if current_dens[lk] < jam_density:
                                new_spd = free_flow*(1 - current_dens[lk]/jam_density)
                                current_spd[lk] = max(0.1, new_spd)
                            else:
                                current_spd[lk] = 0.1
                            propagate_upstream(
                                lk, upstream_links, current_flows, current_dens, current_spd,
                                free_flow_speeds, link_props, scale_factor, dt
                            )

                    for lk in valid_outgoing:
                        if lk in out_demand_share and lk in current_flows:
                            current_flows[lk] = total_supply * out_demand_share[lk] / dt
                            propagate_downstream(
                                lk, downstream_links, current_flows, current_dens, current_spd,
                                free_flow_speeds, link_props, dt
                            )

                # Record updated states
                for lk in all_affected_links:
                    if lk in current_flows:    fine_flows[lk].append(current_flows[lk])
                    if lk in current_dens:     fine_densities[lk].append(current_dens[lk])
                    if lk in current_spd:      fine_speeds[lk].append(current_spd[lk])

    # Re-aggregate 30-second data to 15-minute intervals
    intervals_per_period = int((15*60) / dt)
    for lk in fine_speeds:
        if len(fine_speeds[lk]) > 0:
            num_periods = len(fine_speeds[lk]) // intervals_per_period
            for period in range(num_periods):
                if period < len(time_cols):
                    col = time_cols[period]
                    start = period*intervals_per_period
                    end   = (period+1)*intervals_per_period
                    updated_speeds.loc[lk, col]    = np.mean(fine_speeds[lk][start:end])
                    updated_flows.loc[lk, col]     = np.mean(fine_flows[lk][start:end])
                    updated_densities.loc[lk, col] = np.mean(fine_densities[lk][start:end])

    updated_speeds.to_csv("updated_speeds.csv")
    updated_flows.to_csv("updated_flows.csv")
    updated_densities.to_csv("updated_densities.csv")
    if verbose:
        print("CTM complete. Updated files saved.")

    # Separate corridor DataFrame outputs
    dt_spd = updated_speeds.loc[updated_speeds.index.intersection(downtown_links)]
    mid_spd = updated_speeds.loc[updated_speeds.index.intersection(mid_links)]
    ot_spd  = updated_speeds.loc[updated_speeds.index.intersection(outer_links)]
    return {
        "downtown": dt_spd,
        "mid": mid_spd,
        "outer": ot_spd
    }

