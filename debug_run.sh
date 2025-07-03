#!/bin/bash
# Debug run script for media-to-text with enhanced logging and memory settings

# Set OpenMP environment variable to prevent conflicts with PyTorch
export KMP_DUPLICATE_LIB_OK=TRUE

# Memory management environment variables to help prevent segmentation faults
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# If installed as a package
if command -v media-to-text &> /dev/null; then
    media-to-text --debug "$@"
else
    # If running from source
    python -m media_to_text --debug "$@"
fi
