import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import os
import io

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="UCDNet Change Detection",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ UCDNet — Satellite Change Detection")
st.markdown("Compare two **13-band Sentinel-2 patches** to detect man-made changes.")

# ── API URL (env var for deployment, localhost for local) ─────
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ── Load test patches from test.npz ──────────────────────────
@st.cache_data
def load_test_patches():
    base = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(base, '..', 'test.npz')
    data = np.load(npz_path)
    return data['img1'], data['img2'], data['labels']

img1_all, img2_all, labels_all = load_test_patches()
num_patches = img1_all.shape[0]

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("About")
st.sidebar.info("""
**Model:** UCDNet  
**Bands:** 13 (Sentinel-2)  
**Patch Size:** 64×64  
**Cities Trained:** Mumbai & Abu Dhabi  
**Jaccard Score:** 0.70  
**Test Patches:** 53
""")

st.sidebar.title("How to Use")
st.sidebar.markdown("""
1. Choose input mode
2. Select or upload patches
3. Click **Detect Changes**
4. View predicted change mask
""")

# ── Input Mode ────────────────────────────────────────────────
st.subheader("📂 Input Mode")
mode = st.radio(
    "Choose input method:",
    ["Select from test patches", "Upload .npy files manually"],
    horizontal=True
)

def to_rgb(arr):
    rgb = arr[:, :, [3, 2, 1]].astype(np.float32)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-7)
    return rgb

img1_arr = None
img2_arr = None
label_arr = None

# ── Mode 1 — Select from test patches ────────────────────────
if mode == "Select from test patches":
    patch_num = st.slider("Select Patch Number", 0, num_patches - 1, 0)
    img1_arr  = img1_all[patch_num]
    img2_arr  = img2_all[patch_num]
    label_arr = labels_all[patch_num]
    st.success(f"Loaded Patch {patch_num} — shape: {img1_arr.shape}")

# ── Mode 2 — Upload manually ──────────────────────────────────
else:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Image 1 (Older Date)")
        img1_file = st.file_uploader("Upload .npy", type=['npy'], key='img1')
    with col2:
        st.subheader("📅 Image 2 (Later Date)")
        img2_file = st.file_uploader("Upload .npy", type=['npy'], key='img2')

    if img1_file and img2_file:
        img1_arr = np.load(img1_file)
        img2_arr = np.load(img2_file)

# ── Preview ───────────────────────────────────────────────────
if img1_arr is not None and img2_arr is not None:
    st.divider()
    st.subheader("🖼️ Preview (RGB — Bands 4,3,2)")

    prev_col1, prev_col2 = st.columns(2)
    with prev_col1:
        st.image(to_rgb(img1_arr), caption="Image 1 — Older Date",
                 use_column_width=True, clamp=True)
    with prev_col2:
        st.image(to_rgb(img2_arr), caption="Image 2 — Later Date",
                 use_column_width=True, clamp=True)

    # ── Detect Button ─────────────────────────────────────────
    st.divider()
    if st.button("🔍 Detect Changes", type="primary", use_container_width=True):
        with st.spinner("Running inference..."):
            try:
                buf1 = io.BytesIO(); np.save(buf1, img1_arr); buf1.seek(0)
                buf2 = io.BytesIO(); np.save(buf2, img2_arr); buf2.seek(0)

                response = requests.post(
                    f"{API_URL}/predict",
                    files={
                        "img1": ("img1.npy", buf1, "application/octet-stream"),
                        "img2": ("img2.npy", buf2, "application/octet-stream")
                    }
                )
                result = response.json()

                if result['status'] == 'success':
                    mask = np.array(result['change_mask'])
                    st.success("✅ Inference Complete!")

                    # ── Metrics ───────────────────────────────
                    st.subheader("📊 Results")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Change Pixels",  result['change_pixels'])
                    m2.metric("Total Pixels",   result['total_pixels'])
                    m3.metric("Change Ratio",   f"{result['change_ratio_%']}%")

                    # ── Visualization ─────────────────────────
                    st.subheader("🗺️ Change Mask")
                    ncols = 4 if label_arr is not None else 3
                    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

                    axes[0].imshow(to_rgb(img1_arr)); axes[0].set_title("Image 1 (Older)"); axes[0].axis('off')
                    axes[1].imshow(to_rgb(img2_arr)); axes[1].set_title("Image 2 (Later)"); axes[1].axis('off')
                    axes[2].imshow(mask, cmap='gray')
                    axes[2].set_title("Predicted Mask\n(White=Changed)"); axes[2].axis('off')

                    if label_arr is not None:
                        axes[3].imshow(label_arr, cmap='gray')
                        axes[3].set_title("Ground Truth"); axes[3].axis('off')

                    plt.tight_layout()
                    st.pyplot(fig)

                    # ── Download mask ─────────────────────────
                    mask_bytes = io.BytesIO()
                    np.save(mask_bytes, mask); mask_bytes.seek(0)
                    st.download_button(
                        label="⬇️ Download Change Mask (.npy)",
                        data=mask_bytes,
                        file_name="change_mask.npy",
                        mime="application/octet-stream"
                    )

                else:
                    st.error(f"Error: {result['message']}")

            except Exception as e:
                st.error(f"Could not connect to API: {e}")

else:
    st.info("👆 Select a patch or upload files to get started.")