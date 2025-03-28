import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import your Hankel-DMD and CTM utilities
from ctm import apply_ctm
from koopman_H_DMD import H_DMD

def koopman_ctm_iterative_forecast(
    downtown_gdf, mid_gdf, outer_gdf, nodes,
    speeds_file="filtered_avg_speeds.csv",
    flow_file="filtered_links_flow.csv", 
    density_file="filtered_links_density.csv",
    delay=8,
    prediction_horizon=4,
    start_time_idx=48,
    dt=30,
    propagation_steps=3,
    time_col_start_idx=0,
    window_size=48,
    verbose=True
):
    """
    Iterative Koopman forecasting with CTM constraints.
    Rolling window approach:
      1) For each forecast step:
         - Collect last 'window_size' snapshots
         - Build Koopman model (Hankel-DMD)
         - Predict one step
         - Apply CTM to enforce physical consistency
      2) Move forward in time
    """
    if verbose:
        print(f"Starting iterative Koopman-CTM forecasting with {delay}-lag embedding")
        print(f"Using a {window_size//4}-hour ({window_size} snapshots) sliding window")
    
    speeds_df = pd.read_csv(speeds_file)
    flows_df = pd.read_csv(flow_file)
    density_df = pd.read_csv(density_file)

    for df in [speeds_df, flows_df, density_df]:
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
            if verbose:
                print("Dropped 'Unnamed: 0' column")

    speeds_df.set_index("link_id", inplace=True)
    flows_df.set_index("link_id", inplace=True)
    density_df.set_index("link_id", inplace=True)

    time_cols = speeds_df.columns[time_col_start_idx:]

    if start_time_idx < window_size - 1:
        raise ValueError(f"start_time_idx must be at least {window_size-1} to get a {window_size//4}-hour window")

    max_prediction_idx = len(time_cols) - 1
    if start_time_idx + prediction_horizon > max_prediction_idx:
        original_horizon = prediction_horizon
        prediction_horizon = max_prediction_idx - start_time_idx
        print(f"Warning: Reduced prediction_horizon from {original_horizon} to {prediction_horizon} due to data limit")

    updated_speeds = speeds_df.copy()
    forecasted_speeds = speeds_df.copy()

    downtown_links = downtown_gdf["LINK_ID"].tolist()
    mid_links = mid_gdf["LINK_ID"].tolist()
    outer_links = outer_gdf["LINK_ID"].tolist()

    if verbose:
        print(f"Applying CTM to historical data up to time {time_cols[start_time_idx]}")

    # Apply CTM historically (0..start_time_idx)
    historical_time_range = (0, start_time_idx + 1)
    updated_speeds_dict = apply_ctm(
        downtown_gdf=downtown_gdf,
        mid_gdf=mid_gdf,
        outer_gdf=outer_gdf,
        nodes=nodes,
        speeds_file=speeds_file,
        flow_file=flow_file,
        density_file=density_file,
        dt=dt,
        time_range=historical_time_range,
        propagation_steps=propagation_steps,
        verbose=verbose
    )

    downtown_historical = updated_speeds_dict["downtown"]
    mid_historical = updated_speeds_dict["mid"]
    outer_historical = updated_speeds_dict["outer"]

    downtown_forecasts = []
    mid_forecasts = []
    outer_forecasts = []

    for step in range(prediction_horizon):
        current_time_idx = start_time_idx + step
        if current_time_idx + 1 >= len(time_cols):
            print(f"Warning: Reached the end of available time columns at step {step}")
            break

        if verbose:
            print(f"\n--- Forecast step {step+1}/{prediction_horizon} (time={time_cols[current_time_idx]}) ---")

        for corridor_name, corridor_data, corridor_links in [
            ("downtown", downtown_historical, downtown_links),
            ("mid", mid_historical, mid_links),
            ("outer", outer_historical, outer_links)
        ]:
            if verbose:
                print(f"\nProcessing {corridor_name} corridor")

            window_start = max(0, current_time_idx - window_size + 1)
            window_end = current_time_idx + 1
            window_cols = time_cols[window_start:window_end]

            if verbose:
                print(f"Window columns: {len(window_cols)} from {window_cols[0]} to {window_cols[-1]}")

            corridor_speeds = corridor_data.loc[corridor_data.index.intersection(corridor_links)]
            window_data = corridor_speeds[window_cols].values
            data_mean = np.mean(window_data, axis=1, keepdims=True)
            window_data_centered = window_data - data_mean

            A, eigvals, Modes, bo, X1, X2, H = H_DMD(window_data_centered, delay=delay)
            link_ids = corridor_speeds.index.tolist()
            num_links = len(link_ids)

            full_state = np.zeros((delay * num_links, 1))
            for i, col in enumerate(window_cols[-delay:]):
                start_idx = i * num_links
                end_idx = (i+1) * num_links
                speeds_slice = corridor_speeds[col].values.reshape(-1, 1)
                full_state[start_idx:end_idx] = speeds_slice

            expanded_mean = np.vstack([data_mean for _ in range(delay)])
            centered_state = full_state - expanded_mean

            from numpy.linalg import pinv
            if Modes.shape[0] != centered_state.shape[0]:
                if Modes.shape[0] > centered_state.shape[0]:
                    pad_size = Modes.shape[0] - centered_state.shape[0]
                    centered_state = np.vstack([centered_state, np.zeros((pad_size, 1))])
                else:
                    centered_state = centered_state[:Modes.shape[0]]

            koopman_state = pinv(Modes) @ centered_state
            if A.shape[1] != koopman_state.shape[0]:
                print("ERROR: A dimension mismatch. Skipping corridor.")
                continue

            next_koopman_state = A @ koopman_state
            next_embedded_state = Modes @ next_koopman_state
            next_centered_speeds = next_embedded_state[-num_links:]
            next_speeds = next_centered_speeds.flatten() + data_mean.flatten()
            predicted_speeds_dict = dict(zip(link_ids, next_speeds))

            if corridor_name == "downtown":
                downtown_forecasts.append(predicted_speeds_dict)
            elif corridor_name == "mid":
                mid_forecasts.append(predicted_speeds_dict)
            else:
                outer_forecasts.append(predicted_speeds_dict)

            if current_time_idx + 1 < len(time_cols):
                next_time_col = time_cols[current_time_idx + 1]
                for lid, spd in predicted_speeds_dict.items():
                    forecasted_speeds.loc[lid, next_time_col] = float(np.real(spd))

                if step < prediction_horizon - 1:
                    corridor_historical_new = corridor_data.copy()
                    for lid, spd in predicted_speeds_dict.items():
                        corridor_historical_new.loc[lid, next_time_col] = float(np.real(spd))
                    
                    if corridor_name == "downtown":
                        downtown_historical = corridor_historical_new
                    elif corridor_name == "mid":
                        mid_historical = corridor_historical_new
                    else:
                        outer_historical = corridor_historical_new

        if step < prediction_horizon - 1 and current_time_idx + 1 < len(time_cols):
            temp_speeds_file = "temp_forecast_speeds.csv"
            temp_flow_file = "temp_forecast_flows.csv"
            temp_density_file = "temp_forecast_densities.csv"

            temp_speeds = forecasted_speeds.copy()
            temp_speeds.reset_index(inplace=True)
            temp_speeds.to_csv(temp_speeds_file, index=False)

            temp_flows = flows_df.copy()
            temp_flows.reset_index(inplace=True)
            temp_flows.to_csv(temp_flow_file, index=False)

            temp_density = density_df.copy()
            temp_density.reset_index(inplace=True)
            temp_density.to_csv(temp_density_file, index=False)

            ctm_time_start = window_start
            ctm_time_end = current_time_idx + 2

            if verbose:
                print(f"\nApplying CTM to verify consistency for time_col {time_cols[current_time_idx+1]}")

            updated_dict = apply_ctm(
                downtown_gdf=downtown_gdf,
                mid_gdf=mid_gdf,
                outer_gdf=outer_gdf,
                nodes=nodes,
                speeds_file=temp_speeds_file,
                flow_file=temp_flow_file,
                density_file=temp_density_file,
                dt=dt,
                time_range=(ctm_time_start, ctm_time_end),
                propagation_steps=propagation_steps,
                verbose=False
            )

            downtown_historical = updated_dict["downtown"]
            mid_historical = updated_dict["mid"]
            outer_historical = updated_dict["outer"]

            next_time_col = time_cols[current_time_idx + 1]
            for lid in downtown_historical.index:
                forecasted_speeds.loc[lid, next_time_col] = downtown_historical.loc[lid, next_time_col]
            for lid in mid_historical.index:
                forecasted_speeds.loc[lid, next_time_col] = mid_historical.loc[lid, next_time_col]
            for lid in outer_historical.index:
                forecasted_speeds.loc[lid, next_time_col] = outer_historical.loc[lid, next_time_col]

    forecasted_speeds.to_csv("koopman_ctm_forecasted_speeds.csv")

    if verbose and len(downtown_links) > 0:
        print("\nSample of forecasted values:")
        sample_links = downtown_links[:5]
        forecast_cols = time_cols[start_time_idx+1 : min(start_time_idx+prediction_horizon+1, len(time_cols))]
        print(f"{'Link ID':<10} " + " ".join(f"{c:<8}" for c in forecast_cols))
        for lid in sample_links:
            if lid in forecasted_speeds.index:
                vals = [f"{forecasted_speeds.loc[lid, c]:.2f}" for c in forecast_cols]
                print(f"{lid:<10} " + " ".join(vals))

    if verbose:
        print("\nKoopman-CTM forecasting completed!")
        print("Forecasted speeds saved to koopman_ctm_forecasted_speeds.csv")

    return {
        "downtown_forecasts": downtown_forecasts,
        "mid_forecasts": mid_forecasts,
        "outer_forecasts": outer_forecasts,
        "forecasted_speeds": forecasted_speeds
    }

def analyze_koopman_forecast_accuracy(actual_speeds_dict, forecasted_speeds_dict):
    """
    Computes TMAE, SMAE, and speed histograms for each corridor.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np
    import os

    def ms_to_mph(spd):
        return spd * 2.23694

    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 3, figure=fig)
    metrics_dict = {}

    for idx, corridor in enumerate(['downtown', 'mid', 'outer']):
        if corridor not in actual_speeds_dict or corridor not in forecasted_speeds_dict:
            print(f"No data for corridor: {corridor}")
            continue
        
        actual_ms = actual_speeds_dict[corridor]
        forecast_ms = forecasted_speeds_dict[corridor]
        
        actual_mph = ms_to_mph(actual_ms)
        forecast_mph = ms_to_mph(forecast_ms)
        
        error = forecast_mph - actual_mph
        abs_error = np.abs(error)

        # TMAE, SMAE
        TMAE = abs_error.mean(axis=0)
        SMAE = abs_error.mean(axis=1)
        AvgTMAE = TMAE.mean()
        AvgSMAE = SMAE.mean()

        # MAE, RMSE (overall)
        MAE = abs_error.values.flatten().mean()
        RMSE = np.sqrt((abs_error.values.flatten()**2).mean())

        metrics_dict[corridor] = {
            "AvgTMAE": AvgTMAE,
            "AvgSMAE": AvgSMAE,
            "MAE": MAE,
            "RMSE": RMSE
        }

        ax1 = fig.add_subplot(gs[0, idx])
        ax1.plot(SMAE.values, color='blue', linewidth=2)
        ax1.axhline(y=AvgSMAE, color='magenta', linestyle='-', linewidth=2)
        ax1.set_title(f"{corridor.capitalize()} SMAE")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1, idx])
        ax2.plot(TMAE.values, color='blue', linewidth=2)
        ax2.axhline(y=AvgTMAE, color='magenta', linestyle='-', linewidth=2)
        ax2.set_title(f"{corridor.capitalize()} TMAE")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[2, idx])
        actual_vals = actual_mph.values.flatten()
        forecast_vals = forecast_mph.values.flatten()
        valid_mask = (~np.isnan(actual_vals) & ~np.isnan(forecast_vals) & (actual_vals>=0) & (forecast_vals>=0))
        bins = np.linspace(0, 75, 40)
        ax3.hist(actual_vals[valid_mask], bins=bins, density=True, alpha=0.7, color='blue', label='Data')
        ax3.hist(forecast_vals[valid_mask], bins=bins, density=True, alpha=0.7, color='orange', label='Forecast')
        ax3.set_xlim(0, 75)
        ax3.set_title(f"{corridor.capitalize()} Speed Histogram")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = "forecast_analysis"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/all_corridors_forecast_analysis_nooutliers.png", dpi=300)

    print("\nForecast Analysis:")
    for corridor, m in metrics_dict.items():
        print(f"\n{corridor.capitalize()} Corridor:")
        print(f" TMAE: {m['AvgTMAE']:.2f} mph")
        print(f" SMAE: {m['AvgSMAE']:.2f} mph")
        print(f" MAE:  {m['MAE']:.2f} mph")
        print(f" RMSE: {m['RMSE']:.2f} mph")

    return metrics_dict
