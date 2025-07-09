#!/bin/bash
# Comprehensive fix for Intel GPU driver conflicts on Compute Canada
# This must be sourced BEFORE running any Python scripts

echo "🔧 Applying Intel GPU driver conflict fixes..."

# 1. CRITICAL: Completely remove Intel GPU libraries from LD_LIBRARY_PATH
echo "Removing Intel libraries from LD_LIBRARY_PATH..."
if [ -n "$LD_LIBRARY_PATH" ]; then
    # Remove any paths containing intel, oneapi, level-zero
    export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v -i intel | grep -v -i oneapi | grep -v -i level-zero | grep -v -i ze_loader | tr '\n' ':' | sed 's/:*$//')
    echo "Cleaned LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
fi

# 2. Disable Intel GPU and oneAPI completely
export ONEAPI_DEVICE_SELECTOR="cuda:*"
export INTEL_DEVICE_SELECTOR="none"
export DISABLE_INTEL_GPU=1
export ONEAPI_ROOT=""
export SYCL_DEVICE_FILTER="cuda"
export LEVEL_ZERO_DEBUG=1

# 3. Force CUDA only environment
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HIP_VISIBLE_DEVICES=""

# 4. PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export TORCH_USE_CUDA_DSA=1

# 5. Disable problematic libraries
export DISABLE_LEVEL_ZERO=1
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1

# 6. Threading optimizations
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 7. Unload Intel GPU modules if loaded
if command -v module >/dev/null 2>&1; then
    echo "Checking for Intel GPU modules..."
    module list 2>&1 | grep -i intel && {
        echo "Unloading Intel GPU modules..."
        module unload intel 2>/dev/null || true
        module unload oneapi 2>/dev/null || true
        module unload level-zero 2>/dev/null || true
    }
fi

# 8. Check for problematic libraries and hide them
PROBLEMATIC_LIBS=("libze_loader.so" "libze_loader.so.1" "liblevel-zero.so")
for lib in "${PROBLEMATIC_LIBS[@]}"; do
    if ldconfig -p | grep -q "$lib"; then
        echo "Warning: $lib found in system libraries"
        # Try to mask it by setting a dummy path
        export LEVEL_ZERO_LOADER_PATH="/dev/null"
    fi
done

# 9. Create a clean Python environment launcher
cat > run_python_clean.sh << 'EOF'
#!/bin/bash
# Clean Python launcher that ensures Intel GPU libraries are not loaded

# Apply all environment fixes
source quick_fix_intel_gpu.sh

# Run Python with clean environment
exec python "$@"
EOF

chmod +x run_python_clean.sh

echo "✅ Intel GPU fixes applied successfully!"
echo "💡 Usage:"
echo "   1. Source this script: source quick_fix_intel_gpu.sh"
echo "   2. Test PyTorch: python -c 'import torch; print(torch.cuda.is_available())'"
echo "   3. Or use clean launcher: ./run_python_clean.sh your_script.py"
echo ""
echo "Current environment:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  ONEAPI_DEVICE_SELECTOR: $ONEAPI_DEVICE_SELECTOR"
echo "  LD_LIBRARY_PATH length: $(echo $LD_LIBRARY_PATH | wc -c) characters" 