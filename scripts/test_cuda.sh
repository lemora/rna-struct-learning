#!/bin/bash

echo "Checking for libcuda.so..."
if ls /usr/lib*/libcuda.so* &>/dev/null || find / -name "libcuda.so*" 2>/dev/null | grep -q libcuda; then
    echo "libcuda.so found: NVIDIA driver is installed."
else
    echo "libcuda.so not found: CUDA likely unavailable."
fi

echo "Checking NVIDIA driver version..."
cat /proc/driver/nvidia/version 2>/dev/null || echo "Could not read NVIDIA driver version"

echo "Checking for GPU via lspci..."
if command -v lspci &>/dev/null; then
    lspci | grep -i nvidia || echo "No NVIDIA GPU visible in lspci"
else
    echo "Cannot check GPU model: lspci not available"
fi