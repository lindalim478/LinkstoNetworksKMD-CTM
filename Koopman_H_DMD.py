import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, eig, pinv

def H_DMD(X, delay):
    """
    Perform Hankel-DMD for time-delay embedding and compute Koopman operator.
    Returns A, eigenvalues, modes, initial condition, X1, X2, HankelMatrix.
    """
    n, m = X.shape
    H_rows = delay * n
    H_cols = m - delay + 1 
    H = np.zeros((H_rows, H_cols))
    for k in range(delay):
        H[k*n : (k+1)*n, :] = X[:, k : k+H_cols]
    X1 = H[:, :-1]
    X2 = H[:, 1:]
    U, s, Vh = svd(X1, full_matrices=False)
    V = Vh.conj().T
    S_inv = np.diag(1.0 / s)
    A = U.conj().T @ X2 @ V @ S_inv
    eigvals, Y = eig(A)
    Modes = U @ Y
    bo = pinv(Modes) @ X1[:, 0]
    return A, eigvals, Modes, bo, X1, X2, H


def run_koopman_modes(
    gdf_loop,
    link_positions_feet,
    csv_file="avg_speeds.csv",
    loop_label="Downtown",
    skip_cols=1,
    delay=7,
    delt=15,
    mode_range=(1,10)
):
    """
    Runs Hankel-DMD on a speed dataset to extract Koopman modes/eigenvalues,
    and saves 3D plots of modes over position & time.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # for 3D plots

    df_vel = pd.read_csv(csv_file)
    df_vel.set_index("link_id", inplace=True)
    all_ids = gdf_loop["LINK_ID"].unique()
    valid_ids = df_vel.index.intersection(all_ids)
    missing = set(all_ids) - set(valid_ids)
    if missing:
        print(f"{loop_label}: Missing {len(missing)} link IDs in {csv_file}. Skipping them...")
        missing_df = pd.DataFrame(list(missing), columns=["missing_link_id"])
        out_csv = f"{loop_label}_missing_link_ids.csv"
        missing_df.to_csv(out_csv, index=False)
        print(f"Missing IDs saved to {out_csv}")

    df_vel = df_vel.loc[valid_ids]
    sub_loop = gdf_loop[gdf_loop["LINK_ID"].isin(valid_ids)].copy()
    df_vel = df_vel.reindex(sub_loop["LINK_ID"])

    time_cols = df_vel.columns[skip_cols:]
    Data = df_vel[time_cols].values.astype(float)*3.28084
    Data = np.nan_to_num(Data, nan=0.0, posinf=0.0, neginf=0.0)

    Data_mean = np.mean(Data, axis=1, keepdims=True)
    Data_centered = Data - Data_mean

    A, eigvals, Modes, bo, X1, X2, H = H_DMD(Data_centered, delay)

    omega = np.log(eigvals) / delt
    Freal = np.imag(omega) / (2*np.pi)
    T = (1.0 / Freal) / 60.0
    sidx = np.argsort(-T)
    eigvals = eigvals[sidx]
    Modes  = Modes[:, sidx]
    bo     = bo[sidx]
    omega  = omega[sidx]
    T      = T[sidx]

    n_links, n_times = Data.shape
    time_array = np.arange(n_times)*delt
    time_hours = time_array/60.0
    link_positions_miles = link_positions_feet / 5280.0

    X_mesh, Y_mesh = np.meshgrid(time_hours, link_positions_miles)
    out_dir = f"KMD_{loop_label}"
    os.makedirs(out_dir, exist_ok=True)

    m1, m2 = mode_range
    for i in range(m1-1, m2):
        psi_t = np.exp(omega[i]*time_array)*bo[i]
        psi_space = Modes[:n_links, i]
        psi_mode = psi_space[:, np.newaxis]*psi_t[np.newaxis, :]
        Z = np.real(psi_mode)

        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(
            X_mesh, Y_mesh, Z,
            cmap='viridis',
            linewidth=0, rcount=200, ccount=200
        )
        ax.set_title(f"{loop_label} Mode #{i+1}, Period={T[i]:.2f}h", fontsize=14)
        ax.set_xlabel("Time [Hr]")
        ax.set_ylabel("Position [Miles]")
        ax.set_zlabel("Amplitude")
        cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
        cbar.set_label("Feet/Second")
        ax.view_init(elev=30, azim=-60)

        fname = f"{loop_label}_Mode_{i+1}.jpg"
        plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
        plt.close(fig)


    np.save(os.path.join(out_dir, "A_matrix.npy"), A)
    np.save(os.path.join(out_dir, "eigvals.npy"), eigvals)
    np.save(os.path.join(out_dir, "Modes.npy"), Modes)
    np.save(os.path.join(out_dir, "bo.npy"), bo)
    np.save(os.path.join(out_dir, "X1.npy"), X1)
    np.save(os.path.join(out_dir, "data_mean.npy"), Data_mean)
    link_ids = sub_loop["LINK_ID"].values
    np.save(os.path.join(out_dir, "link_ids.npy"), link_ids)

    print(f"Koopman done for {loop_label}. Results in {out_dir}")
    return A, eigvals


def check_stability(eigvals, loop_label="Loop"):
    """
    Plots Koopman eigenvalues and counts how many are outside the unit circle (unstable).
    """
    import matplotlib.pyplot as plt

    magnitudes = np.abs(eigvals)
    unstable_indices = magnitudes > 1
    num_unstable = unstable_indices.sum()

    plt.figure(figsize=(7,7))
    plt.scatter(eigvals.real, eigvals.imag, color='b', label=f'{loop_label} Eigenvalues')
    plt.scatter(eigvals.real[unstable_indices], eigvals.imag[unstable_indices], color='r', label='Unstable')
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=1)
    circle = plt.Circle((0,0), 1, color='r', fill=False, linestyle='dashed', linewidth=2)
    plt.gca().add_patch(circle)
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axis('equal')
    plt.title(f"{loop_label}: Koopman Eigenvalues")
    plt.legend()
    plt.grid(True, linestyle="dotted")
    plt.show()

    print(f"{loop_label}: {num_unstable} unstable eigenvalues")
    return num_unstable == 0
