# LinkstoNetworksKMD-CTM

1. Data Preparation/Preprocessing
   - Reading CSV/Excel files  
   - Creating link geometries
   - Filter data to relevent corridors
   - Align Link IDs between speed dataset and City Network layout file
   - Remove NaN entries

2. Cell Transmission Model (CTM) 
   - Uses node-based constraints  
   - Link-Specific Jam Density
   - Propagates congestion upstream/downstream with exponential distance decay

3. Koopman Decomposition H_DMD
   - Hankel-DMD approach  
   - Visualize modes  
   - Check eigenvalue stability

4. Iterative Koopman Forecasting + CTM Correction
   - Builds Koopman models from a rolling window of traffic snapshots
   - Predicts traffic speeds step-by-step
   - Applies CTM after each step to ensure physical consistency

5. Forecast Accuracy Analysis
   - Computes TMAE (Time-Mean Absolute Error) and SMAE (Space-Mean Absolute Error)
   - Generates comparative histograms of actual vs. forecasted speeds

> Note: Actual speed/flow/density values and link IDs are replaced with dummy data for Mobiliti data privacy.

## File Structure

- data_preparation.py: Functions for merging node geometry, filtering corridor data, sorting geodata, etc.  
- ctm.py: Main CTM logic (apply_ctm) plus `propagate_upstream` / `propagate_downstream`.  
- koopman.py: Hankel-DMD methods, Koopman modes, eigenvalue stability checks.  
- main.py: Ties everything together, loads data, runs filtering, CTM, and Koopman.  
- data/: Example CSV/Excel files (with dummy or masked data).
