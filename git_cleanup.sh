#!/bin/bash

# Set the warning squelch to avoid the filter-branch warning
export FILTER_BRANCH_SQUELCH_WARNING=1

# First, stash any unstaged changes
echo "Stashing any unstaged changes..."
git stash

# Remove files from Git history
echo "Cleaning Git history..."
git filter-branch --force --index-filter \
  "git rm -rf --cached --ignore-unmatch \
  *.mp4 \
  *.mp3 \
  *.zip \
  *.jpg \
  *.png \
  *.wav \
  luma_generated_videos/* \
  **/scene_frames/* \
  **/lora_training_data/* \
  **/*_output/* \
  demand-io-base*.json \
  .env" \
  --prune-empty --tag-name-filter cat -- --all

# Clean up refs and remove old files
echo "Cleaning up refs..."
git for-each-ref --format="delete %(refname)" refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Optionally restore stashed changes
echo "Restoring stashed changes..."
git stash pop

echo "Cleanup complete! Now you can force push with:"
echo "git push origin main --force" 