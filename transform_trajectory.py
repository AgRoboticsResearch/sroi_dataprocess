#!/usr/bin/env python3
"""
Transform Camera Trajectory Script

This script transforms camera trajectories from ORB-SLAM format to an initial camera frame
with X-forward coordinate system and saves the result with timestamps.

Usage:
    python transform_trajectory.py <folder_path>
    
The script expects to find 'times.txt' and 'CameraTrajectory.txt' in the specified folder.
Output will be saved as 'CameraTrajectoryTransformed.txt' in the same folder.

Author: Generated for SROI ROS Bag Utilities
Date: 2025-07-25
"""

import numpy as np
import os
import sys
import argparse
from pathlib import Path


def load_trajectory_data(folder_path):
    """
    Load trajectory, frame ledger and timestamp data from folder.

    Expects the REPAIRED generation (repair_trajectory.py): trajectory rows
    == frame count, CameraTrajectory_idx.txt is the 3-col ledger
    (<frame> <state> <seg_len>). Timestamps are taken through the ledger's
    frame ids — never by row position.

    Returns:
        tuple: (trajectory_matrices, timestamps)
    Raises:
        ValueError on any count/format mismatch (caller reports and fails).
    """
    time_path = os.path.join(folder_path, "times.txt")
    traj_path = os.path.join(folder_path, "CameraTrajectory.txt")
    idx_path = os.path.join(folder_path, "CameraTrajectory_idx.txt")

    for p, name in ((time_path, "times.txt"), (traj_path, "CameraTrajectory.txt"),
                    (idx_path, "CameraTrajectory_idx.txt")):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} not found in {folder_path} (run SLAM + repair first)")

    timestamps = np.loadtxt(time_path, delimiter=" ", ndmin=1)
    n = len(timestamps)

    traj = np.loadtxt(traj_path, delimiter=" ")
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    if traj.shape[1] != 12:
        raise ValueError(f"CameraTrajectory.txt has {traj.shape[1]} columns, expected 12")
    if traj.shape[0] != n:
        raise ValueError(
            f"trajectory rows ({traj.shape[0]}) != frames ({n}) — "
            "repair_trajectory.py has not run (or SLAM output is stale). Run it first.")

    idx_lines = [l for l in open(idx_path).read().splitlines() if l.strip() and not l.startswith("#")]
    if len(idx_lines) != n:
        raise ValueError(f"ledger rows ({len(idx_lines)}) != frames ({n}) — rerun repair_trajectory.py")
    rows = [l.split() for l in idx_lines]
    if any(len(r) != 3 for r in rows):
        raise ValueError("ledger is 1-column (raw SLAM generation) — run repair_trajectory.py first")
    frames = np.array([int(r[0]) for r in rows])
    if not np.array_equal(frames, np.arange(n)):
        raise ValueError("ledger frame ids are not 0..N-1 — rerun repair_trajectory.py")

    # Cross-check: NaN rows in the trajectory must sit exactly on the ledger's
    # invalid rows — ties the holes in the data to the verdicts in the ledger.
    traj2d = np.loadtxt(traj_path, delimiter=" ", ndmin=2)
    nan_rows = np.isnan(traj2d).any(axis=1)
    invalid_rows = np.array([r[1] == "invalid" for r in rows])
    if not np.array_equal(nan_rows, invalid_rows):
        raise ValueError("NaN rows do not match ledger 'invalid' rows — rerun repair_trajectory.py")

    traj = traj.reshape(-1, 3, 4)
    append_row = np.zeros((traj.shape[0], 1, 4))
    append_row[:, 0, 3] = 1
    traj = np.concatenate((traj, append_row), axis=1)

    return traj, timestamps[frames]


def transform_trajectory(traj):
    """
    Transform trajectory to initial camera frame with X-forward coordinate system.
    
    Args:
        traj (np.ndarray): Input trajectory as 4x4 transformation matrices
        
    Returns:
        np.ndarray: Transformed trajectory matrices
    """
    # ORB-SLAM already normalizes its world origin to the first keyframe, so
    # poses arrive relative to the initial camera frame by construction — no
    # re-basing needed here (the previous inverse of traj[0] was dead code).

    # Transformation from ORB-SLAM camera frame to X-forward frame
    # ORB-SLAM: X-right, Y-down, Z-forward
    # Target:   X-forward, Y-left, Z-up
    T_conv = np.array([
        [ 0, 0, 1, 0],  # Target X = ORB Z (forward)
        [-1, 0, 0, 0],  # Target Y = -ORB X (right -> left)
        [ 0,-1, 0, 0],  # Target Z = -ORB Y (down -> up)
        [ 0, 0, 0, 1]
    ])
    T_conv_inv = np.linalg.inv(T_conv)
    
    # Transform all poses to be relative to initial camera frame with new coordinate system
    traj_transformed = np.zeros_like(traj)
    
    for i in range(len(traj)):
        # Current pose in world coordinates (ORB-SLAM frame)
        T_old = traj[i]
        
        # Apply similarity transform: T_new = T_conv * T_old * T_conv^-1
        T_new = T_conv @ T_old @ T_conv_inv
        
        traj_transformed[i] = T_new
    
    return traj_transformed


def save_transformed_trajectory(traj_transformed, timestamps, output_path):
    """
    Save transformed trajectory with timestamps and proper header.
    
    Args:
        traj_transformed (np.ndarray): Transformed trajectory matrices
        timestamps (np.ndarray): Corresponding timestamps
        output_path (str): Output file path
    """
    # Create header
    header = [
        "# Transformed Camera Trajectory",
        "# Generated by transform_trajectory.py",
        "# Coordinate system: X-forward, Y-left, Z-up relative to initial camera frame",
        "# Format: timestamp r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz",
        "# Where:",
        "#   timestamp: frame timestamp",
        "#   r11 r12 r13 tx: first row of 3x4 transformation matrix",
        "#   r21 r22 r23 ty: second row of 3x4 transformation matrix", 
        "#   r31 r32 r33 tz: third row of 3x4 transformation matrix",
        "#"
    ]
    
    # Convert transformation matrices to timestamp + 3x4 matrix format (same as CameraTrajectory.txt)
    output_data = []
    
    for i in range(len(traj_transformed)):
        # Extract 3x4 transformation matrix (rotation + translation)
        transform_3x4 = traj_transformed[i, :3, :]  # Get first 3 rows (3x4 matrix)
        transform_flat = transform_3x4.flatten()  # Flatten to 12 elements
        
        # Format: timestamp + 12 elements of 3x4 matrix
        line_data = [timestamps[i]] + transform_flat.tolist()
        output_data.append(line_data)
    
    # Save to file
    with open(output_path, 'w') as f:
        # Write header
        for line in header:
            f.write(line + '\n')
        
        # Write data
        for line_data in output_data:
            formatted_line = ' '.join([f'{val:.6f}' for val in line_data])
            f.write(formatted_line + '\n')


def process_folder(folder_path, output_path=None):
    """Process a single folder containing trajectory data."""
    if not output_path:
        output_path = folder_path / "CameraTrajectoryTransformed.txt"
    
    try:
        # Check if files exist to avoid error spam when recursive
        if not (folder_path / "times.txt").exists() or not (folder_path / "CameraTrajectory.txt").exists():
            return False

        print(f"\n--- Processing: {folder_path} ---")
        print(f"Loading trajectory data from: {folder_path}")
        traj, timestamps = load_trajectory_data(str(folder_path))
        print(f"Loaded {len(traj)} trajectory poses and {len(timestamps)} timestamps")

        print("Transforming trajectory to initial camera frame...")
        traj_transformed = transform_trajectory(traj)
        
        print(f"Saving transformed trajectory to: {output_path}")
        save_transformed_trajectory(traj_transformed, timestamps, str(output_path))
        
        print("Transformation completed successfully!")
        print(f"\nTrajectory Statistics:")
        print(f"Original frame - Position range:")
        print(f"  X: [{traj[:, 0, 3].min():.3f}, {traj[:, 0, 3].max():.3f}]")
        print(f"  Y: [{traj[:, 1, 3].min():.3f}, {traj[:, 1, 3].max():.3f}]") 
        print(f"  Z: [{traj[:, 2, 3].min():.3f}, {traj[:, 2, 3].max():.3f}]")
        print(f"Transformed frame - Position range:")
        print(f"  X: [{traj_transformed[:, 0, 3].min():.3f}, {traj_transformed[:, 0, 3].max():.3f}]")
        print(f"  Y: [{traj_transformed[:, 1, 3].min():.3f}, {traj_transformed[:, 1, 3].max():.3f}]")
        print(f"  Z: [{traj_transformed[:, 2, 3].min():.3f}, {traj_transformed[:, 2, 3].max():.3f}]")
        return True
    except Exception as e:
        print(f"Error processing trajectory in {folder_path}: {e}")
        return False


def main():
    """Main function to process command line arguments and run transformation."""
    parser = argparse.ArgumentParser(
        description="Transform camera trajectory to initial camera frame with X-forward coordinate system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python transform_trajectory.py /path/to/data/folder
    python transform_trajectory.py ./postproc/segment_1/
        """
    )
    
    parser.add_argument(
        'folder_path',
        type=str,
        help='Path to folder containing times.txt and CameraTrajectory.txt'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: CameraTrajectoryTransformed.txt in input folder)'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively search for folders containing times.txt and CameraTrajectory.txt'
    )
    
    args = parser.parse_args()
    
    # Validate input folder
    folder_path = Path(args.folder_path)
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        sys.exit(1)
    
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory")
        sys.exit(1)
    
    # Process recursive or single folder
    if args.recursive:
        print(f"Searching recursively in {folder_path}...")
        processed_count = failed_count = 0
        for root, dirs, files in os.walk(folder_path):
            current_dir = Path(root)
            if (current_dir / "times.txt").exists() and (current_dir / "CameraTrajectory.txt").exists():
                out_path = Path(args.output) if args.output else None
                if process_folder(current_dir, out_path):
                    processed_count += 1
                else:
                    failed_count += 1
        print(f"\nFinished. Processed {processed_count} folders, {failed_count} failed.")
        sys.exit(1 if failed_count else 0)
    else:
        out_path = Path(args.output) if args.output else None

        if not (folder_path / "times.txt").exists() or not (folder_path / "CameraTrajectory.txt").exists():
            print(f"Error: Required files 'times.txt' and/or 'CameraTrajectory.txt' not found in '{folder_path}'")
            print(f"To search subdirectories, use the --recursive flag.")
            sys.exit(1)

        sys.exit(0 if process_folder(folder_path, out_path) else 1)


if __name__ == "__main__":
    main()
