#!/bin/bash

# Script to copy combined videos from all experimental combinations
# Skips raw camera files (hand_camera.mp4, varied_camera_1.mp4, varied_camera_2.mp4)

# All combinations to process
MODELS=("pi0_droid" "pi05_droid")
TASKS=("open_drawer" "open_cabinet" "open_oven")
STATUSES=("success" "failure")
DATES=("time_denoise_0.4" "time_denoise_0.2" "2026-01-16")

# Output directory
OUTPUT_DIR="results_mp4"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Copying combined videos to '$OUTPUT_DIR'..."
echo "Skipping: hand_camera.mp4, varied_camera_1.mp4, varied_camera_2.mp4"
echo ""

TOTAL_COPIED=0
TOTAL_SKIPPED=0

# Iterate through all combinations
for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        for status in "${STATUSES[@]}"; do
            for date in "${DATES[@]}"; do
                INPUT_PATH="results/${task}/${model}/${status}/${date}"
                
                # Check if path exists
                if [ ! -d "$INPUT_PATH" ]; then
                    echo "Skipping (path does not exist): $INPUT_PATH"
                    continue
                fi
                
                echo "Processing: $INPUT_PATH"
                
                # Find all mp4 files, excluding the raw camera files
                while IFS= read -r file; do
                    FILENAME=$(basename "$file")
                    
                    # Skip raw camera files
                    if [[ "$FILENAME" == "hand_camera.mp4" ]] || \
                       [[ "$FILENAME" == "varied_camera_1.mp4" ]] || \
                       [[ "$FILENAME" == "varied_camera_2.mp4" ]]; then
                        continue
                    fi
                    
                    # Copy the combined video
                    echo "  Copying: $FILENAME"
                    cp "$file" "$OUTPUT_DIR/"
                    TOTAL_COPIED=$((TOTAL_COPIED + 1))
                    
                done < <(find "$INPUT_PATH" -type f -name "*.mp4")
                
            done
        done
    done
done

echo ""
echo "=============================================="
echo "COMPLETE: Copied $TOTAL_COPIED combined video(s) to '$OUTPUT_DIR'"
echo "=============================================="
