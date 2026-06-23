import streamlit as st
import numpy as np
import pandas as pd
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Q-Engine | Pro Dashboard", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS: Fixing fonts and matching the DistriSync dark theme ---
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; }
    h1, h2, h3, h4 { color: #38bdf8 !important; font-family: 'Courier New', Courier, monospace; }
    
    /* Force normal text to be standard size and bright white */
    p, label, span, div[data-testid="stMetricLabel"] > div { color: #f8fafc !important; font-size: 16px !important; }
    
    /* Keep metric numbers bright green */
    div[data-testid="stMetricValue"] > div { color: #10b981 !important; }
    
    hr { border-color: #334155; }
    
    /* Custom sleek explanation box */
    .explanation-box { 
        background-color: #1e293b; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 4px solid #38bdf8; 
        color: #f8fafc; 
        font-size: 15px; 
        margin-bottom: 20px; 
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Q-Engine: Model Quantization Simulator")
st.caption("A systems-level engineering tool simulating Absolute Maximum (Absmax) INT8 Quantization for Deep Learning memory footprint reduction.")

# Fixed the giant text by putting it in a strictly controlled CSS box
st.markdown("""
<div class="explanation-box">
    <strong>Infrastructure Perspective:</strong> Large AI models use 32-bit floating-point numbers (FP32) for weights, demanding massive VRAM bandwidth. By compressing weights into 8-bit integers (INT8), we reduce memory footprint by 75% and speed up compute-bound operations, at the cost of negligible mathematical precision loss (Quantization Error).
</div>
""", unsafe_allow_html=True)

# 🎛️ TOP CONTROL PANEL (Moved from Sidebar)
st.markdown("### 🎛️ Tensor Configuration Matrix")
c1, c2 = st.columns(2)
matrix_size = c1.select_slider("Matrix Size (N x N)", options=[1024, 2048, 4096, 8192], value=4096)
c2.metric("Total Parameter Count", f"{matrix_size * matrix_size:,} Weights")

st.markdown("---")

# --- QUANTIZATION LOGIC ---
def generate_fp32_tensor(size):
    return np.random.randn(size, size).astype(np.float32)

def quantize_to_int8(tensor):
    absmax = np.max(np.abs(tensor))
    scale = 127.0 / absmax if absmax != 0 else 1.0
    int8_tensor = np.round(tensor * scale).astype(np.int8)
    return int8_tensor, scale

def dequantize_to_fp32(int8_tensor, scale):
    return (int8_tensor / scale).astype(np.float32)

# --- MAIN DASHBOARD AREA ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 1️⃣ Original FP32 Tensor")
    st.write("Raw 32-bit neural network weights loaded into VRAM.")
    
    fp32_weights = generate_fp32_tensor(matrix_size)
    fp32_memory_mb = fp32_weights.nbytes / (1024 * 1024)
    
    st.metric("VRAM Footprint (FP32)", f"{fp32_memory_mb:.2f} MB", "High Bandwidth Constraint")
    st.dataframe(pd.DataFrame(fp32_weights[:4, :4]))

with col2:
    st.markdown("### 2️⃣ Compressed INT8 Tensor")
    st.write("Execute the Absmax INT8 Quantization algorithm below.")
    
    if st.button("🚀 EXECUTE QUANTIZATION PASS", type="primary", use_container_width=True):
        with st.spinner("Compressing tensor footprint..."):
            time.sleep(0.8) 
            
        int8_weights, scale_factor = quantize_to_int8(fp32_weights)
        int8_memory_mb = int8_weights.nbytes / (1024 * 1024)
        
        reconstructed_weights = dequantize_to_fp32(int8_weights, scale_factor)
        mse_error = np.mean((fp32_weights - reconstructed_weights) ** 2)
        
        st.metric("VRAM Footprint (INT8)", f"{int8_memory_mb:.2f} MB", "-75% Memory Traffic")
        st.dataframe(pd.DataFrame(int8_weights[:4, :4]))
        
        st.markdown("#### 📊 Infrastructure Impact Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Memory Saved", f"{(fp32_memory_mb - int8_memory_mb):.2f} MB")
        m2.metric("Compression", "4.0x Ratio")
        m3.metric("Quantization Error", f"{mse_error:.6f}")
        
        st.info(f"**Systems Note:** Scale factor applied: $S = {scale_factor:.4f}$. The Mean Squared Error (MSE) is strictly bounded, preserving model accuracy while quadrupling hardware efficiency.")
    else:
        st.info("Awaiting quantization command. Click the execute button above.")
