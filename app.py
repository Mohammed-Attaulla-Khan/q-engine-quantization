import streamlit as st
import numpy as np
import pandas as pd
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Q-Engine | Model Compression", page_icon="🧠", layout="wide")

st.title("🧠 Q-Engine: LLM Quantization & Memory Compression Simulator")
st.caption("A systems-level engineering tool simulating Absolute Maximum (Absmax) INT8 Quantization for Deep Learning memory footprint reduction.")

st.markdown("""
---
### ⚙️ Why Quantization Matters (Infrastructure Perspective)
Large AI models use 32-bit floating-point numbers (FP32) for weights, demanding massive VRAM bandwidth. 
By compressing weights into 8-bit integers (INT8), we **reduce memory footprint by 75%** and speed up compute bound operations, at the cost of slight mathematical precision loss (Quantization Error).
---
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Configure Tensor Layer")
matrix_size = st.sidebar.slider("Matrix Size (N x N)", min_value=1024, max_value=8192, value=4096, step=1024)
st.sidebar.markdown(f"**Total Weights:** {matrix_size * matrix_size:,}")

# --- QUANTIZATION LOGIC ---
def generate_fp32_tensor(size):
    # Simulate a layer of neural network weights (Normal distribution)
    return np.random.randn(size, size).astype(np.float32)

def quantize_to_int8(tensor):
    # Absmax Quantization Formula: Scale = 127 / Max(Abs(Weights))
    absmax = np.max(np.abs(tensor))
    scale = 127.0 / absmax
    
    # Multiply by scale and round to nearest integer
    int8_tensor = np.round(tensor * scale).astype(np.int8)
    return int8_tensor, scale

def dequantize_to_fp32(int8_tensor, scale):
    # Convert back to float to measure how much data we lost
    return (int8_tensor / scale).astype(np.float32)

# --- DASHBOARD UI ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Original FP32 Tensor (Standard)")
    st.write("Loading raw 32-bit neural network weights...")
    
    # Generate weights
    fp32_weights = generate_fp32_tensor(matrix_size)
    fp32_memory_mb = fp32_weights.nbytes / (1024 * 1024)
    
    st.metric("VRAM Footprint (FP32)", f"{fp32_memory_mb:.2f} MB", "High Bandwidth Constraint")
    
    # Show sample of the raw weights
    st.write("Sample Weights (Raw Floats):")
    st.dataframe(pd.DataFrame(fp32_weights[:5, :5]))

with col2:
    st.subheader("2️⃣ Compressed INT8 Tensor (Quantized)")
    st.write("Click below to run the Absmax INT8 Quantization algorithm.")
    
    if st.button("⚡ Quantize to INT8", type="primary"):
        with st.spinner("Compressing tensor footprint..."):
            time.sleep(0.8) # Simulate compute time
            
        # Run the math
        int8_weights, scale_factor = quantize_to_int8(fp32_weights)
        int8_memory_mb = int8_weights.nbytes / (1024 * 1024)
        
        # Dequantize to calculate error
        reconstructed_weights = dequantize_to_fp32(int8_weights, scale_factor)
        mse_error = np.mean((fp32_weights - reconstructed_weights) ** 2)
        
        st.metric("VRAM Footprint (INT8)", f"{int8_memory_mb:.2f} MB", "-75% Memory Reduction")
        
        # Show compressed weights
        st.write("Sample Weights (Compressed Integers):")
        st.dataframe(pd.DataFrame(int8_weights[:5, :5]))
        
        st.markdown("---")
        st.markdown("### 📊 Infrastructure Impact Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Memory Saved", f"{(fp32_memory_mb - int8_memory_mb):.2f} MB")
        m2.metric("Compression Ratio", "4.0x")
        m3.metric("Quantization Error (MSE)", f"{mse_error:.6f}")
        
        st.info(f"**Systems Note:** Model successfully compressed. Scale factor applied: $S = {scale_factor:.4f}$. The Mean Squared Error (MSE) is negligible, meaning model accuracy is preserved while quadrupling hardware efficiency.")
    else:
        st.info("Awaiting quantization command.")