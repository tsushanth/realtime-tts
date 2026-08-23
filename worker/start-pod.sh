#!/bin/bash
set -e

# RunPod injects the account's public key as $PUBLIC_KEY on Pods — wire it into
# authorized_keys and start sshd so RunPod's runtime/status reporting (and manual
# debugging) works, matching the pattern their own docs specify for custom images.
mkdir -p ~/.ssh
if [ -n "$PUBLIC_KEY" ]; then
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys
fi
ssh-keygen -A
service ssh start

exec python3 -u /app/server.py
