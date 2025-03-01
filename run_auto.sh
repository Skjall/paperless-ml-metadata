#!/bin/bash

# This script runs the paperless-ml tool in automatic mode
# Useful for cron jobs or scheduled tasks

# Change to the directory containing this script
cd "$(dirname "$0")"

# Make sure AUTO_PROCESSING is set to True
if grep -q "AUTO_PROCESSING=False" .env; then
    echo "Setting AUTO_PROCESSING=True in .env for this run"
    # Backup .env
    cp .env .env.bak
    # Replace AUTO_PROCESSING=False with AUTO_PROCESSING=True
    sed -i 's/AUTO_PROCESSING=False/AUTO_PROCESSING=True/g' .env
    RESTORE_ENV=true
else
    RESTORE_ENV=false
fi

# Run the paperless-ml container in auto mode
echo "Running Paperless ML in automatic mode"
docker-compose run --rm paperless-ml

# Restore the .env file if needed
if [ "$RESTORE_ENV" = true ]; then
    echo "Restoring original .env file"
    mv .env.bak .env
fi

echo "Paperless ML automatic run completed"