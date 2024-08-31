#!/bin/bash

# File containing the list of packages
REQUIREMENTS_FILE="build/requirements.txt"

# Array to hold problematic packages
PROBLEMATIC_PACKAGES=()

# Function to install packages
install_package() {
    local package=$1
    python3 -m pip install "$package"
    if [ $? -ne 0 ]; then
        PROBLEMATIC_PACKAGES+=("$package")
    fi
}

# Read the requirements file and install each package
while IFS= read -r package || [ -n "$package" ]; do
    install_package "$package"
done < "$REQUIREMENTS_FILE"

# Print out problematic packages if any
if [ ${#PROBLEMATIC_PACKAGES[@]} -eq 0 ]; then
    echo "All packages installed successfully."
else
    echo "The following packages failed to install:"
    for pkg in "${PROBLEMATIC_PACKAGES[@]}"; do
        echo "$pkg"
    done
fi