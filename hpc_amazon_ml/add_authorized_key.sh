#!/bin/bash
# Run this on your Mac or any machine that already has passwordless 'ssh hpc' access
# This will authorize your Windows machine to connect directly via passwordless SSH!

WINDOWS_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDzIcYAVGJhWcMheh1CQ4bHxi72Nq9nOMSmfodlIAZlY ce1240901_windows"

echo "Adding Windows public key to cluster ~/.ssh/authorized_keys..."
ssh hpc "mkdir -p ~/.ssh && echo '${WINDOWS_KEY}' >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"

echo "Verifying..."
ssh hpc "tail -n 1 ~/.ssh/authorized_keys"
echo "Done! You can now run 'ssh hpc' directly from your Windows machine without passwords."
