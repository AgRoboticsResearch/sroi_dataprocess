#!/usr/bin/env python3
"""
Create ORB-SLAM YAML configuration files based on extracted camera information.
This script reads the `camera_info_{left,right}.json` files from a session/episode
folder and generates a customized ORB-SLAM YAML configuration file.
"""

import json
import numpy as np
import argparse
import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_camera_info(output_folder: str) -> Dict[str, Any]:
    """Load camera information from JSON files"""
    camera_info = {}
    
    # Load left camera info
    left_json_path = os.path.join(output_folder, "camera_info_left.json")
    if os.path.exists(left_json_path):
        with open(left_json_path, 'r') as f:
            camera_info['left'] = json.load(f)
    else:
        raise FileNotFoundError(f"Left camera info not found: {left_json_path}")
    
    # Load right camera info
    right_json_path = os.path.join(output_folder, "camera_info_right.json")
    if os.path.exists(right_json_path):
        with open(right_json_path, 'r') as f:
            camera_info['right'] = json.load(f)
    else:
        raise FileNotFoundError(f"Right camera info not found: {right_json_path}")
    
    # Load color camera info if available (for RealSense)
    color_json_path = os.path.join(output_folder, "camera_info_color.json")
    if os.path.exists(color_json_path):
        with open(color_json_path, 'r') as f:
            camera_info['color'] = json.load(f)
    
    return camera_info


def calculate_stereo_transform(left_P: np.ndarray, right_P: np.ndarray) -> np.ndarray:
    """Calculate stereo transformation matrix from projection matrices"""
    # Extract baseline from projection matrices
    # For stereo cameras, right camera's P matrix has the baseline in P[0,3]
    # baseline = -P_right[0,3] / P_right[0,0]
    baseline = -(right_P[0, 3] - left_P[0, 3]) / left_P[0, 0]
    
    # Create transformation matrix (right camera relative to left camera)
    T_c1_c2 = np.eye(4, dtype=np.float32)
    T_c1_c2[0, 3] = baseline  # Translation in x-direction
    
    return T_c1_c2


def load_opencv_yaml(file_path: str) -> Dict[str, Any]:
    """Load OpenCV YAML file handling special directives and matrix format"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove the %YAML:1.0 directive if present
    if content.startswith('%YAML:1.0'):
        content = content.replace('%YAML:1.0\n', '')
    
    # Use unsafe loader to handle OpenCV matrix format
    try:
        config = yaml.load(content, Loader=yaml.UnsafeLoader)
    except:
        # Fallback to safe loader if unsafe fails
        config = yaml.safe_load(content)
    
    return config


def save_opencv_yaml(config: Dict[str, Any], file_path: str):
    """Save configuration as OpenCV-compatible YAML"""
    with open(file_path, 'w') as f:
        # Write the YAML directive first
        f.write('%YAML:1.0\n\n')
        
        # Process each key-value pair
        for key, value in config.items():
            if key == 'Stereo.T_c1_c2' and isinstance(value, dict):
                # Special handling for OpenCV matrix
                f.write(f'{key}: !!opencv-matrix\n')
                f.write(f'  rows: {value["rows"]}\n')
                f.write(f'  cols: {value["cols"]}\n')
                f.write(f'  dt: {value["dt"]}\n')
                f.write(f'  data: {value["data"]}\n')
            else:
                # Regular YAML formatting
                if isinstance(value, str):
                    f.write(f'{key}: "{value}"\n')
                else:
                    f.write(f'{key}: {value}\n')
        f.write('\n')


def create_orb_slam_yaml(camera_info: Dict[str, Any], template_path: str, output_path: str):
    """Create ORB-SLAM YAML configuration file"""
    
    # Load template YAML with custom loader to handle OpenCV YAML format
    config = load_opencv_yaml(template_path)
    
    # Extract camera parameters from left camera (Camera1)
    left_K = np.array(camera_info['left']['K']).reshape(3, 3)
    left_P = np.array(camera_info['left']['P']).reshape(3, 4)
    right_P = np.array(camera_info['right']['P']).reshape(3, 4)
    
    # Extract intrinsic parameters
    fx = left_K[0, 0]
    fy = left_K[1, 1]
    cx = left_K[0, 2]
    cy = left_K[1, 2]
    
    # Extract distortion parameters
    left_D = camera_info['left']['D']
    
    # Get image dimensions
    width = camera_info['left']['width']
    height = camera_info['left']['height']
    
    # Update Camera1 parameters (left camera)
    config['Camera1.fx'] = float(fx)
    config['Camera1.fy'] = float(fy)
    config['Camera1.cx'] = float(cx)
    config['Camera1.cy'] = float(cy)
    
    # Set distortion parameters (assuming we use the left camera's distortion)
    if len(left_D) >= 4:
        config['Camera1.k1'] = float(left_D[0]) if len(left_D) > 0 else 0.0
        config['Camera1.k2'] = float(left_D[1]) if len(left_D) > 1 else 0.0
        config['Camera1.p1'] = float(left_D[2]) if len(left_D) > 2 else 0.0
        config['Camera1.p2'] = float(left_D[3]) if len(left_D) > 3 else 0.0
    
    # For stereo cameras, right camera parameters are typically the same as left
    # (assuming rectified stereo pair)
    config['Camera2.fx'] = float(fx)
    config['Camera2.fy'] = float(fy)
    config['Camera2.cx'] = float(cx)
    config['Camera2.cy'] = float(cy)
    
    # Set right camera distortion (typically same as left for rectified pairs)
    if len(left_D) >= 4:
        config['Camera2.k1'] = float(left_D[0]) if len(left_D) > 0 else 0.0
        config['Camera2.k2'] = float(left_D[1]) if len(left_D) > 1 else 0.0
        config['Camera2.p1'] = float(left_D[2]) if len(left_D) > 2 else 0.0
        config['Camera2.p2'] = float(left_D[3]) if len(left_D) > 3 else 0.0
    
    # Update image resolution
    config['Camera.width'] = int(width)
    config['Camera.height'] = int(height)
    
    # Calculate and update stereo transformation
    T_c1_c2 = calculate_stereo_transform(left_P, right_P)
    
    # Convert to the format expected by ORB-SLAM
    stereo_transform = {
        'rows': 4,
        'cols': 4,
        'dt': 'f',
        'data': T_c1_c2.flatten().tolist()
    }
    
    config['Stereo.T_c1_c2'] = stereo_transform
    
    # Save the updated YAML file using custom OpenCV YAML format
    save_opencv_yaml(config, output_path)
    
    print(f"ORB-SLAM YAML configuration saved to: {output_path}")
    print(f"Camera parameters:")
    print(f"  fx: {fx:.8f}, fy: {fy:.8f}")
    print(f"  cx: {cx:.8f}, cy: {cy:.8f}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Baseline: {T_c1_c2[0, 3]:.8f}")


def main():
    parser = argparse.ArgumentParser(description="Create ORB-SLAM YAML from extracted camera info")
    parser.add_argument("output_folder", help="Session/episode folder holding camera_info_{left,right}.json")
    parser.add_argument("camera_type", choices=["realsense_d435i"], 
                       help="Camera type (currently only realsense_d435i is supported)")
    # Default to the templates shipped in this repo (resolved relative to this
    # file, so it works wherever the repo is checked out). Pass --template_dir
    # only to override with a different rig's template dir.
    parser.add_argument("--template_dir",
                       default=str(Path(__file__).resolve().parent / "orb_slam_yaml"),
                       help="Directory containing template YAML files "
                            "(default: <repo>/orb_slam_yaml)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.output_folder):
        raise FileNotFoundError(f"Output folder not found: {args.output_folder}")
    
    # Determine template file based on camera type
    if args.camera_type == "realsense_d435i":
        template_file = "RealSense_D435i.yaml"
    else:
        raise ValueError(f"Unsupported camera type: {args.camera_type}")
    
    template_path = os.path.join(args.template_dir, template_file)
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    # Load camera information
    print(f"Loading camera info from: {args.output_folder}")
    camera_info = load_camera_info(args.output_folder)
    
    # Create output YAML path
    output_yaml_path = os.path.join(args.output_folder, f"orb_slam_{args.camera_type}.yaml")
    
    # Create ORB-SLAM YAML configuration
    create_orb_slam_yaml(camera_info, template_path, output_yaml_path)


if __name__ == "__main__":
    main()