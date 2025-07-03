#!/bin/bash
# Run script for media-to-text with environment variables set correctly

# Set OpenMP environment variable to prevent conflicts with PyTorch
export KMP_DUPLICATE_LIB_OK=TRUE

# If installed as a package
if command -v media-to-text &> /dev/null; then
    media-to-text "$@"
else
    # If running from source
    python -m media_to_text "$@"
fi
