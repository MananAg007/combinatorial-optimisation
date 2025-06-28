# TSP Instance Conversion for DIFUSCO

This document explains how to convert TSP instances from FrontierCO format to DIFUSCO format.

## Format Differences

1. **FrontierCO Format**: Uses TSPLIB format with metadata headers and node coordinates.
   ```
   NAME : instance_1000_1
   COMMENT : Generated TSP instance N=1000, seed=1
   TYPE : TSP
   DIMENSION : 1000
   EDGE_WEIGHT_TYPE : EUC_2D
   NODE_COORD_SECTION
   1 374540 950714
   2 731993 598658
   ...
   ```

   The solutions are also provided in TSPLIB format:
   ```
   NAME : instance_1000_1.22600139.tour
   COMMENT : Length = 22600139
   COMMENT : Found by LKH-3 [Keld Helsgaun] Fri Jun 27 06:40:49 2025
   TYPE : TOUR
   DIMENSION : 1000
   TOUR_SECTION
   12
   329
   483
   ...
   ```

2. **DIFUSCO Format**: Uses a space-separated list of coordinates followed by "output" and the optimal tour.
   ```
   x1 y1 x2 y2 ... xn yn output 1 3 2 5 4 1
   ```

## Conversion Script

The `convert_tsp_instances.py` script converts TSP instances from FrontierCO format to DIFUSCO format, using the pre-solved optimal tours.

### Prerequisites

The script requires:
- Python 3.7+
- NumPy
- tqdm

### Usage

```bash
python convert_tsp_instances.py \
  --instance_dir /path/to/frontierco/instances \
  --solution_dir /path/to/frontierco/solutions \
  --output_dir /path/to/output/directory \
  --output_file tsp1000_frontierco.txt
```

Arguments:
- `--instance_dir`: Directory containing FrontierCO TSP instances
- `--solution_dir`: Directory containing FrontierCO TSP solutions
- `--output_dir`: Directory to save DIFUSCO-formatted instances
- `--output_file`: (Optional) Single file to combine all instances

### Process

1. The script reads each TSP file in the input directory.
2. It extracts the coordinates from the TSPLIB instance format.
3. It reads the corresponding optimal tour from the solution file.
4. It formats the data as expected by DIFUSCO and writes it to the output.

### Example

For the slurm script in `difusco.sh`, we:
1. Create a directory for converted data
2. Convert all FrontierCO instances to a single file in DIFUSCO format
3. Use this file for training the DIFUSCO model

## Notes

- The script handles the indexing conversion between the formats. FrontierCO uses 1-indexed node IDs, and DIFUSCO also expects 1-indexed tours.
- If a solution file is not found for a given instance, the script will skip that instance. 