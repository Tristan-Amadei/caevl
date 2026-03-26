#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory_path> <total_images>"
    exit 1
fi

# Assign arguments to variables
directory_path=$1
total_images=$2

# Check if the provided directory path is valid
if [ ! -d "$directory_path" ]; then
    echo "Error: Directory $directory_path does not exist."
    exit 1
fi

# Check if the total_images is a non-zero number
if ! [[ "$total_images" =~ ^[0-9]+$ ]] || [ "$total_images" -eq 0 ]; then
    echo "Error: total_images must be a non-zero number."
    exit 1
fi

# Record the initial number of files in the directory
initial_file_count=$(find "$directory_path" -type f | wc -l)

# Adjust the total number of images to account for the initial files
adjusted_total_images=$((total_images - initial_file_count))

if [ "$adjusted_total_images" -le 0 ]; then
    echo "The directory already contains $initial_file_count files. No additional files need to be created."
    exit 0
fi

# Record the start time
start_time=$(date +%s)

# Function to count the number of files and calculate the percentage
calculate_percentage() {
    current_file_count=$(find "$directory_path" -type f | wc -l)
    file_count=$((current_file_count - initial_file_count))
    percentage=$(echo "scale=2; ($file_count * 100 / $adjusted_total_images)" | bc -l)

    # Calculate the elapsed time
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    elapsed_hours=$((elapsed_time / 3600))
    elapsed_minutes=$(((elapsed_time - $elapsed_hours * 3600) / 60))
    elapsed_seconds=$((elapsed_time % 60))

    if [ "$file_count" -ne 0 ]; then
        avg_time_per_file=$(echo "scale=4; $elapsed_time / $file_count" | bc -l)
        remaining_files=$((adjusted_total_images - file_count))
        estimated_remaining_hours=$(echo "scale=0; ($remaining_files * $avg_time_per_file) / 3600" | bc)
        estimated_remaining_minutes=$(echo "scale=0; ($remaining_files * $avg_time_per_file - $estimated_remaining_hours * 3600) / 60" | bc)
        estimated_remaining_seconds=$(echo "scale=0; ($remaining_files * $avg_time_per_file) % 60" | bc)
    else
        avg_time_per_file=0
        estimated_remaining_hours=0
        estimated_remaining_minutes=0
        estimated_remaining_seconds=0
    fi

    echo "$file_count images have been generated since the start, out of $adjusted_total_images. This represents $percentage% of all images to be generated."
    echo "Time elapsed: ${elapsed_hours}h ${elapsed_minutes}m ${elapsed_seconds}s"
    echo "On average, an image takes ${avg_time_per_file}s to be generated"
    echo "Estimated time remaining: ${estimated_remaining_hours}h ${estimated_remaining_minutes}m ${estimated_remaining_seconds}s"
    echo ""
    echo "In total, $current_file_count files have been generated, out of a total of $total_images."
}

# Initial calculation and display
calculate_percentage

# Loop until percentage reaches 100%
while (( $(echo "$percentage < 100" | bc -l) )); do
    sleep 15
    clear
    calculate_percentage
done

echo "Reached 100% or more."
