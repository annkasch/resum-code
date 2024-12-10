#!/bin/bash

# Directory containing the files (change if needed)
#TARGET_DIR="/global/cfs/projectdirs/legend/users/aschuetz/analysis/legend-multi-fidelity-surrogate-model/adaptive_sampling/out/LF/v1.7/neutron-inputs-v1.7"

# Loop through all matching files in the directory
#for file in "${TARGET_DIR}"/neutron-inputs2-design0_*_0003_v1.7.dat; do
    # Extract the new filename by replacing "inputs2" with "inputs"
 #   new_file=$(echo "$file" | sed 's/inputs2/inputs/')

    # Rename the file
  #  mv "$file" "$new_file"

    # Print a message for each rename
   # echo "Renamed: $file -> $new_file"
#done

#echo "Renaming completed."


TARGET_DIR="/global/cfs/projectdirs/legend/users/aschuetz/simulation/out/low_fidelity/Neutron-Simulation-LF-v1.4/csv/tier3_2"

for file in "${TARGET_DIR}"/neutron-sim-LF-v1.4*tier2.csv; do

    # Replace v1.4 with v1.7
    updated_path=$(echo "$file" | sed 's/tier2/tier3/g')

    # Output the updated path

    echo "Updated  Path: $updated_path"
    mv $file $updated_path
done