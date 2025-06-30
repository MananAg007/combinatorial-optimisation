#!/usr/bin/env python3
"""
Convert FrontierCO TSP instances to DIFUSCO format.

This script reads TSP instances in TSPLIB format and converts them to the format expected by DIFUSCO,
which is a space-separated list of coordinates followed by "output" and the optimal tour.

The script uses pre-solved solutions from the FrontierCO dataset.
"""

import os
import argparse
import numpy as np
import tqdm


def read_tsplib_instance(filepath):
    """Read a TSP instance in TSPLIB format."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find where node coordinates begin
    coord_section_idx = -1
    for i, line in enumerate(lines):
        if "NODE_COORD_SECTION" in line:
            coord_section_idx = i
            break
    
    if coord_section_idx == -1:
        raise ValueError(f"Could not find NODE_COORD_SECTION in {filepath}")
    
    # Extract coordinates
    coordinates = []
    for line in lines[coord_section_idx + 1:]:
        if "EOF" in line:
            break
        parts = line.strip().split()
        if len(parts) >= 3:  # node_id, x, y
            coordinates.append((float(parts[1]), float(parts[2])))
    
    return np.array(coordinates)


def read_tsplib_solution(filepath):
    """Read a TSP solution in TSPLIB format."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find where the tour begins
    tour_section_idx = -1
    for i, line in enumerate(lines):
        if "TOUR_SECTION" in line:
            tour_section_idx = i
            break
    
    if tour_section_idx == -1:
        raise ValueError(f"Could not find TOUR_SECTION in {filepath}")
    
    # Extract tour
    tour = []
    for line in lines[tour_section_idx + 1:]:
        if "EOF" in line or "-1" in line:
            break
        node_id = line.strip()
        if node_id and node_id.isdigit():
            tour.append(int(node_id))
    
    return tour


def convert_instance(instance_file, solution_file, output_file):
    """Convert a single TSP instance and its solution to DIFUSCO format."""
    coordinates = read_tsplib_instance(instance_file)
    tour = read_tsplib_solution(solution_file)
    
    # Format as expected by DIFUSCO
    coord_str = " ".join(f"{x} {y}" for x, y in coordinates)
    
    # Convert from 1-indexed to 0-indexed and back for consistency with DIFUSCO's expectations
    # DIFUSCO expects 1-indexed tours in its output format
    tour_zero_indexed = [node_id - 1 for node_id in tour]
    tour_str = " ".join(str(node_idx + 1) for node_idx in tour_zero_indexed)
    
    # Add the first node at the end to complete the cycle if not already there
    if tour[0] != tour[-1]:
        tour_str += f" {tour[0]}"
    
    # Write to output file in DIFUSCO format
    with open(output_file, 'w') as f:
        f.write(f"{coord_str} output {tour_str}\n")


def main():
    parser = argparse.ArgumentParser(description='Convert TSP instances from FrontierCO to DIFUSCO format')
    parser.add_argument('--instance_dir', type=str, required=True, help='Directory containing FrontierCO TSP instances')
    parser.add_argument('--solution_dir', type=str, required=True, help='Directory containing FrontierCO TSP solutions')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save DIFUSCO-formatted instances')
    parser.add_argument('--output_file', type=str, help='Single file to combine all instances (optional)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all TSP files in input directory
    tsp_files = [f for f in os.listdir(args.instance_dir) if f.endswith('.tsp')]
    
    if args.output_file:
        # Process all instances and write to a single file
        combined_output_path = os.path.join(args.output_dir, args.output_file)
        with open(combined_output_path, 'w') as out_file:
            for tsp_file in tqdm.tqdm(tsp_files):
                instance_path = os.path.join(args.instance_dir, tsp_file)
                solution_path = os.path.join(args.solution_dir, f"{tsp_file}.sol")
                
                if not os.path.exists(solution_path):
                    print(f"Solution file not found for {tsp_file}, skipping...")
                    continue
                
                coordinates = read_tsplib_instance(instance_path)
                tour = read_tsplib_solution(solution_path)
                
                # Format as expected by DIFUSCO
                coord_str = " ".join(f"{x} {y}" for x, y in coordinates)
                
                # Convert from 1-indexed to 0-indexed and back for consistency with DIFUSCO's expectations
                tour_zero_indexed = [node_id - 1 for node_id in tour]
                tour_str = " ".join(str(node_idx + 1) for node_idx in tour_zero_indexed)
                
                # Add the first node at the end to complete the cycle if not already there
                if tour[0] != tour[-1]:
                    tour_str += f" {tour[0]}"
                
                out_file.write(f"{coord_str} output {tour_str}\n")
        
        print(f"Combined instances written to {combined_output_path}")
    else:
        # Process each instance individually
        for tsp_file in tqdm.tqdm(tsp_files):
            instance_path = os.path.join(args.instance_dir, tsp_file)
            solution_path = os.path.join(args.solution_dir, f"{tsp_file}.sol")
            
            if not os.path.exists(solution_path):
                print(f"Solution file not found for {tsp_file}, skipping...")
                continue
            
            output_path = os.path.join(args.output_dir, f"{os.path.splitext(tsp_file)[0]}.txt")
            convert_instance(instance_path, solution_path, output_path)
        
        print(f"Individual instances written to {args.output_dir}")


if __name__ == "__main__":
    main() 