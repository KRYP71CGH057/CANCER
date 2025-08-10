import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import zipfile
import pandas as pd
import json
import hashlib
import imageio
from scipy.ndimage import zoom
from datetime import datetime

# Initialize session state for playback
if 'play' not in st.session_state:
    st.session_state.play = False
if 'current_time' not in st.session_state:
    st.session_state.current_time = 0

# Set page config
st.set_page_config(page_title="MRI & Mask Viewer", layout="wide")

# Helper functions
def is_slice_too_large(slice_data, max_pixels=300000):
    """Check if a slice has more pixels than the allowed maximum."""
    return slice_data.shape[0] * slice_data.shape[1] > max_pixels

def downsample_slice(slice_data, factor=0.5):
    """Downsample slice for display to reduce memory usage"""
    return zoom(slice_data, zoom=factor)

def generate_metadata(img_nii, mask_nii, plane, image_name, mask_name, downsample_factor=0.5):
    """Generate metadata CSV with image info and annotations"""
    img_data = img_nii.get_fdata()
    zooms = img_nii.header.get_zooms()
    orientation = nib.aff2axcodes(img_nii.affine)
    
    metadata = []
    time_points = img_data.shape[3] if img_data.ndim == 4 else 1
    
    if plane == 'axial':
        slices = img_data.shape[2]
        for t in range(time_points):
            for i in range(slices):
                metadata.append({
                    'plane': 'axial',
                    'slice_index': i,
                    'time_index': t,
                    'image_shape_x': img_data.shape[0],
                    'image_shape_y': img_data.shape[1],
                    'image_shape_z': img_data.shape[2],
                    'image_shape_t': img_data.shape[3] if img_data.ndim == 4 else 0,
                    'voxel_size_x': zooms[0],
                    'voxel_size_y': zooms[1],
                    'voxel_size_z': zooms[2],
                    'voxel_size_t': zooms[3] if len(zooms) >= 4 else 0,
                    'orientation': '-'.join(orientation),
                    'image_file': image_name,
                    'mask_file': mask_name,
                    'export_date': datetime.now().isoformat(),
                    'exported_by': st.session_state.username,
                    'role': st.session_state.role,
                    'downsample_factor': downsample_factor,
                    'comment': ''
                })
    elif plane == 'coronal':
        slices = img_data.shape[1]
        for t in range(time_points):
            for i in range(slices):
                metadata.append({
                    'plane': 'coronal',
                    'slice_index': i,
                    'time_index': t,
                    'image_shape_x': img_data.shape[0],
                    'image_shape_y': img_data.shape[1],
                    'image_shape_z': img_data.shape[2],
                    'image_shape_t': img_data.shape[3] if img_data.ndim == 4 else 0,
                    'voxel_size_x': zooms[0],
                    'voxel_size_y': zooms[1],
                    'voxel_size_z': zooms[2],
                    'voxel_size_t': zooms[3] if len(zooms) >= 4 else 0,
                    'orientation': '-'.join(orientation),
                    'image_file': image_name,
                    'mask_file': mask_name,
                    'export_date': datetime.now().isoformat(),
                    'exported_by': st.session_state.username,
                    'role': st.session_state.role,
                    'downsample_factor': downsample_factor,
                    'comment': ''
                })
    elif plane == 'sagittal':
        slices = img_data.shape[0]
        for t in range(time_points):
            for i in range(slices):
                metadata.append({
                    'plane': 'sagittal',
                    'slice_index': i,
                    'time_index': t,
                    'image_shape_x': img_data.shape[0],
                    'image_shape_y': img_data.shape[1],
                    'image_shape_z': img_data.shape[2],
                    'image_shape_t': img_data.shape[3] if img_data.ndim == 4 else 0,
                    'voxel_size_x': zooms[0],
                    'voxel_size_y': zooms[1],
                    'voxel_size_z': zooms[2],
                    'voxel_size_t': zooms[3] if len(zooms) >= 4 else 0,
                    'orientation': '-'.join(orientation),
                    'image_file': image_name,
                    'mask_file': mask_name,
                    'export_date': datetime.now().isoformat(),
                    'exported_by': st.session_state.username,
                    'role': st.session_state.role,
                    'downsample_factor': downsample_factor,
                    'comment': ''
                })
    
    # Add annotations
    metadata_df = pd.DataFrame(metadata)
    annotations = []
    for _, row in metadata_df.iterrows():
        key = (row['plane'], int(row['slice_index']), int(row['time_index']))
        annotations.append(st.session_state.annotations.get(key, ""))
    metadata_df['comment'] = annotations
    
    return metadata_df

st.markdown("""
<style>
    /* Dark gray background */
    .stApp {
        background: linear-gradient(145deg, #1f1f1f, #2d2d2d);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #f5f5f5;
    }

    .card {
        background-color: rgba(40, 40, 40, 0.95);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        margin: 1.5rem 0;
        border: 1px solid #444;
    }

    .section-header {
        color: #ffffff;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.6rem;
        margin-bottom: 2rem;
        font-weight: 600;
        font-size: 1.8rem;
    }

    .info-box {
        background: #2b3442;
        border-left: 6px solid #3b82f6;
        padding: 1.2rem;
        margin: 1.2rem 0;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
        color: #e5e7eb;
    }

    .access-denied {
        background: #3a3a3a;
        border-left: 6px solid #f87171;
        padding: 1.2rem;
        margin: 1.2rem 0;
        border-radius: 10px;
        color: #fef2f2;
    }

    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }

    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #3b82f6;
        padding: 0.8rem 1.2rem;
        font-size: 1rem;
        background-color: rgba(30, 30, 30, 0.9);
        color: #f1f1f1;
    }

    .stTextArea textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }

    .stDownloadButton>button {
        background: linear-gradient(135deg, #6366f1, #3b82f6);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        font-size: 0.95rem;
    }

    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }

    .tooltip .tooltiptext {
        background: #6366f1;
        color: #fff;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }

    .progress-bar {
        height: 8px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        margin: 1rem 0;
    }

    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(to right, #3b82f6, #6366f1);
        width: 0%;
        transition: width 0.5s ease-in-out;
    }

    .stSlider .stMarkdown {
        color: #e0e0e0;
        font-weight: 500;
    }

    .stSlider [data-baseweb="slider"] {
        background: #3b82f6;
    }

    .stFileUploader>label {
        font-weight: 600;
        color: #f0f0f0;
        margin-bottom: 0.5rem;
    }

    .stFileUploader .uploadText {
        color: #cccccc;
        font-style: italic;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #3b82f6, #6366f1);
        color: #fff;
        padding: 1.5rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- AUTH --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Dummy credentials with roles (unchanged)
USERS = {
    "admin": {
        "password": hashlib.sha256("adminpass".encode()).hexdigest(),
        "role": "admin"
    },
    "radiologist": {
        "password": hashlib.sha256("radiopass".encode()).hexdigest(),
        "role": "radiologist"
    },
    "guest": {
        "password": hashlib.sha256("guestpass".encode()).hexdigest(),
        "role": "guest"
    }
}

# Role permission map (unchanged)
ROLE_PERMISSIONS = {
    "admin": {
        "view": True,
        "upload": True,
        "export": True,
        "annotate": True,
    },
    "radiologist": {
        "view": True,
        "upload": True,
        "export": False,
        "annotate": True,
    },
    "guest": {
        "view": True,
        "upload": False,
        "export": False,
        "annotate": False,
    }
}

# Helper functions (unchanged)
def has_permission(action):
    return ROLE_PERMISSIONS.get(st.session_state.get("role", ""), {}).get(action, False)

def show_access_denied(feature="this feature"):
    st.markdown(f"""
    <div class="access-denied">
        🚫 Access Denied: You do not have permission to use <strong>{feature}</strong>.
    </div>
    """, unsafe_allow_html=True)

# Login
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center; color: #ff4b4b; text-shadow: 0 2px 4px rgba(255,75,75,0.3);'>🔒 Login Required</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            st.markdown("<h3 style='text-align: center; color: #1a1a2e; margin-bottom: 1.5rem;'>Login to MRI Viewer</h3>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", use_container_width=True):
                if username in USERS and USERS[username]["password"] == hashlib.sha256(password.encode()).hexdigest():
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = USERS[username]["role"]
                    st.success(f"Welcome, {username} ({st.session_state.role})!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    st.stop()

# Sidebar info
st.sidebar.markdown(f"""
    <div style='padding: 1.5rem; background: linear-gradient(135deg, #ff4b4b, #ff6f61); border-radius: 10px; margin-bottom: 1.5rem; box-shadow: 0 6px 20px rgba(255,75,75,0.3);'>
        <h2 style='color: white; text-align: center; text-shadow: 0 2px 4px rgba(0,0,0,0.3);'>🧠 MRI Viewer</h2>
    </div>
    <div style='background: rgba(255, 255, 255, 0.2); padding: 1rem; border-radius: 10px; backdrop-filter: blur(5px);'>
        <p style='color: #fff; margin-bottom: 1rem;'><strong>👤 User:</strong> {st.session_state.username} ({st.session_state.role})</p>
        <div class='tooltip' style='display: block; text-align: center;'>
            <span style='color: #fff; font-size: 0.9rem;'>Click to log out</span>
            <span class='tooltiptext'>Logout</span>
        </div>
    </div>
""", unsafe_allow_html=True)

if st.sidebar.button("🚪 Logout", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Main app content
st.markdown("<h1 class='section-header'>🧠 MRI + AI Mask Viewer</h1>", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    Upload an MRI scan and its AI mask to view, annotate, and export overlays.
    Supports 2D, 3D, and 4D (time-series) images.
</div>
""", unsafe_allow_html=True)

if has_permission("upload"):
    st.markdown("<h2 class='section-header'>📂 Upload Files</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        image_file = st.file_uploader("Upload MRI image (.nii or .nii.gz)", type=["nii", "nii.gz"])
    with col2:
        mask_file = st.file_uploader("Upload AI mask (.nii or .nii.gz)", type=["nii", "nii.gz"])
else:
    show_access_denied("uploading MRI/Mask files")
    image_file = None
    mask_file = None

if image_file and mask_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_img, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_mask:
        tmp_img.write(image_file.read())
        tmp_img_path = tmp_img.name
        tmp_mask.write(mask_file.read())
        tmp_mask_path = tmp_mask.name
    
    try:
        # Load NIfTI files (unchanged)
        img_nii = nib.load(tmp_img_path)
        mask_nii = nib.load(tmp_mask_path)
        
        # Load data without squeezing
        img_data = img_nii.get_fdata().astype(np.float32)
        mask_data = mask_nii.get_fdata().astype(np.float32)
        
        # Dimension handling (unchanged)
        if img_data.ndim == 2:
            img_data = img_data[..., np.newaxis, np.newaxis]
            mask_data = mask_data[..., np.newaxis, np.newaxis]
        elif img_data.ndim == 3:
            img_data = img_data[..., np.newaxis]
            mask_data = mask_data[..., np.newaxis]
        elif img_data.ndim == 4:
            pass
        else:
            st.error("Unsupported image dimensions. Only 2D, 3D, or 4D images are supported.")
            st.stop()
            
        # Store in session state (unchanged)
        st.session_state.img_data = img_data
        st.session_state.mask_data = mask_data
        
        # Dimension info display
        if img_data.ndim == 4:
            st.markdown(f"<div class='info-box'>📊 4D Image: {img_data.shape[2]} slices × {img_data.shape[3]} time points</div>", unsafe_allow_html=True)
        elif img_data.ndim == 3:
            st.markdown(f"<div class='info-box'>📊 3D Image: {img_data.shape[2]} slices</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='info-box'>📊 2D Image: 1 slice</div>", unsafe_allow_html=True)
        
        # Slice selection (unchanged)
        col1, col2 = st.columns([3, 1])
        with col1:
            if img_data.ndim == 4:
                max_slices = img_data.shape[2]
                slice_index = st.slider("Slice index", 0, max_slices - 1, max_slices // 2)
            elif img_data.ndim == 3:
                max_slices = img_data.shape[2]
                slice_index = st.slider("Slice index", 0, max_slices - 1, max_slices // 2)
            else:
                slice_index = 0
                st.markdown("<div class='info-box'>Only one slice available in 2D images</div>", unsafe_allow_html=True)
        
        # Time point selection
        time_index = 0
        if img_data.ndim == 4:
            max_time = img_data.shape[3]
            if max_time > 1:
                time_index = st.slider("Time index", 0, max_time - 1, 0)
            else:
                st.info("Only one time point available in 4D image.")
        
        # Time series playback toggle
        if img_data.ndim == 4 and img_data.shape[3] > 1:
            st.markdown("<h4 class='section-header'>⏱️ Time Series Playback</h4>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                playback_speed = st.slider("Playback Speed (frames/sec)", 0.5, 5.0, 1.0, 0.1)
            with col2:
                play = st.checkbox("▶️ Play Time Series", key="playback_checkbox")
            
            playback_placeholder = st.empty()
            
            if play:
                st.session_state.play = True
            else:
                st.session_state.play = False
                
            if st.session_state.play:
                st.session_state.current_time = (st.session_state.current_time + 1) % img_data.shape[3]
                time_index = st.session_state.current_time
                display_slice = downsample_slice(img_data[:, :, slice_index, time_index])
                display_mask = downsample_slice(mask_data[:, :, slice_index, time_index])
                
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(display_slice.T, cmap="gray")
                ax[0].set_title(f"MRI Slice (Time: {time_index})")
                ax[0].axis("off")
                ax[1].imshow(display_slice.T, cmap="gray")
                ax[1].imshow(display_mask.T, cmap="Reds", alpha=0.4)
                ax[1].set_title(f"MRI + AI Mask (Time: {time_index})")
                ax[1].axis("off")
                playback_placeholder.pyplot(fig)
                plt.close(fig)
                
                import time
                time.sleep(1/playback_speed)
                st.session_state.current_time = time_index
                st.rerun()
            else:
                st.session_state.current_time = time_index
        
        # Display images (only if not playing)
        if not st.session_state.get('play', False):
            display_slice = downsample_slice(img_data[:, :, slice_index, time_index])
            display_mask = downsample_slice(mask_data[:, :, slice_index, time_index])
            
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(display_slice.T, cmap="gray")
            ax[0].set_title("MRI Slice")
            ax[0].axis("off")
            ax[1].imshow(display_slice.T, cmap="gray")
            ax[1].imshow(display_mask.T, cmap="Reds", alpha=0.4)
            ax[1].set_title("MRI + AI Mask Overlay")
            ax[1].axis("off")
            st.pyplot(fig)
            plt.close(fig)
        
        # Export Section
        st.markdown("<h2 class='section-header'>📦 Batch Overlay Slice Export</h2>", unsafe_allow_html=True)
        
        def export_overlay_slices(img_data, mask_data, plane='axial', out_dir='exports', downsample_factor=0.5):
            os.makedirs(out_dir, exist_ok=True)
            slices = {
                "axial": img_data.shape[2],
                "coronal": img_data.shape[1],
                "sagittal": img_data.shape[0]
            }[plane]
            time_points = img_data.shape[3] if img_data.ndim == 4 else 1
            
            # Status messages
            st.markdown(f"""
            <div class='info-box'>
                Exporting slices with {slices} spatial slices{f' and {time_points} time points' if time_points > 1 else ''}
            </div>
            """, unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total = slices * time_points
            completed = 0
            
            for t in range(time_points):
                for i in range(slices):
                    # Get slice
                    if plane == 'axial':
                        img_slice = img_data[:, :, i, t]
                        mask_slice = mask_data[:, :, i, t]
                    elif plane == 'coronal':
                        img_slice = img_data[:, i, :, t]
                        mask_slice = mask_data[:, i, :, t]
                    else:  # sagittal
                        img_slice = img_data[i, :, :, t]
                        mask_slice = mask_data[i, :, :, t]
                    
                    # Skip very large slices
                    if is_slice_too_large(img_slice):
                        completed += 1
                        progress_bar.progress(completed / total)
                        continue
                    
                    # Downsample
                    img_slice = zoom(img_slice, zoom=downsample_factor)
                    mask_slice = zoom(mask_slice, zoom=downsample_factor)
                    
                    # Normalize and convert to 8-bit
                    img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) if img_slice.max() != img_slice.min() else img_slice
                    img_8bit = (img_norm * 255).astype(np.uint8)
                    
                    # Create overlay
                    overlay = np.zeros((img_8bit.shape[0], img_8bit.shape[1], 3), dtype=np.uint8)
                    overlay[..., 0] = img_8bit
                    overlay[..., 1] = img_8bit
                    overlay[..., 2] = img_8bit
                    
                    # Add red mask overlay
                    overlay[..., 0][mask_slice.T > 0] = 255
                    overlay[..., 1][mask_slice.T > 0] = 0
                    overlay[..., 2][mask_slice.T > 0] = 0
                    
                    # Save with imageio
                    imageio.imwrite(f"{out_dir}/overlay_{plane}_z{i:03d}_t{t:03d}.png", overlay)
                    
                    # Update progress
                    completed += 1
                    progress_bar.progress(completed / total)
            
            status_text.text("✅ Export completed successfully!")
        
        def zip_export_dir(out_dir='exports'):
            zip_path = f"{out_dir}.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for root, _, files in os.walk(out_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file), arcname=file)
            return zip_path
        
        # Export UI
        if img_data.ndim >= 3:
            plane = st.selectbox("Select Plane for Export", ['axial', 'coronal', 'sagittal'])
        else:
            plane = 'axial'
            st.markdown("<div class='info-box'>Only axial view available for 2D images</div>", unsafe_allow_html=True)
        
        if has_permission("export"):
            if st.button("Export All Overlay Slices", use_container_width=True):
                with st.spinner("Exporting slices... This may take a few moments."):
                    export_overlay_slices(img_data, mask_data, plane)
                    
                    # Generate metadata
                    metadata_df = generate_metadata(img_nii, mask_nii, plane, image_file.name, mask_file.name)
                    metadata_path = os.path.join("exports", "metadata.csv")
                    metadata_df.to_csv(metadata_path, index=False)
                    
                    # Create ZIP
                    zip_path = zip_export_dir()
                    
                    # Download button
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            "📥 Download All Overlays (ZIP)", 
                            f, 
                            file_name="overlay_slices.zip", 
                            use_container_width=True
                        )
        else:
            show_access_denied("exporting overlays")
        
        # Annotation Section
        st.markdown("<h2 class='section-header'>📝 Per-Slice Annotation & Export</h2>", unsafe_allow_html=True)
        
        if "annotations" not in st.session_state:
            st.session_state.annotations = {}
        
        # Plane selection
        if img_data.ndim >= 3:
            plane_annot = st.selectbox("Select Plane for Annotation", ['axial', 'coronal', 'sagittal'], key="annot")
        else:
            plane_annot = 'axial'
            st.markdown("<div class='info-box'>Only axial view available for 2D images</div>", unsafe_allow_html=True)
        
        # Calculate available slices
        num_slices = {
            "axial": img_data.shape[2],
            "coronal": img_data.shape[1],
            "sagittal": img_data.shape[0]
        }[plane_annot]
        
        slice_index_annot = st.slider(f"Slice Index ({plane_annot})", 0, num_slices - 1, 0, key="annot_slider")
        
        # Time index for annotation
        time_index_annot = 0
        if img_data.ndim == 4:
            max_time_annot = img_data.shape[3]
            if max_time_annot > 1:
                time_index_annot = st.slider("Time Index", 0, max_time_annot - 1, 0, key="time_annot")
            else:
                st.info("Only one time point available for annotation.")
        
        # Get slice and display
        def get_slice(img, mask, plane, idx, time=0):
            if plane == 'axial':
                return img[:, :, idx, time], mask[:, :, idx, time]
            elif plane == 'coronal':
                return img[:, idx, :, time], mask[:, idx, :, time]
            else:
                return img[idx, :, :, time], mask[idx, :, :, time]

        img_slice, mask_slice = get_slice(img_data, mask_data, plane_annot, slice_index_annot, time_index_annot)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_slice.T, caption=f"{plane_annot.capitalize()} Slice {slice_index_annot} (Time: {time_index_annot})", clamp=True, use_container_width=True)
        with col2:
            st.image(mask_slice.T, caption=f"{plane_annot.capitalize()} Slice {slice_index_annot} (Mask)", clamp=True, use_container_width=True)
        
        # Annotation handling
        default_comment = st.session_state.annotations.get((plane_annot, slice_index_annot, time_index_annot), "")
        comment = st.text_area(f"Annotation for {plane_annot} slice {slice_index_annot}, time {time_index_annot}:", default_comment, height=100)
        
        if has_permission("annotate"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Annotation", use_container_width=True):
                    st.session_state.annotations[(plane_annot, slice_index_annot, time_index_annot)] = comment
                    st.success("Annotation saved!")
            with col2:
                if st.button("📤 Export All Annotations", use_container_width=True):
                    df = pd.DataFrame([
                        {"plane": k[0], "slice_index": k[1], "time_index": k[2], "comment": v}
                        for k, v in st.session_state.annotations.items()
                    ])
                    csv = df.to_csv(index=False).encode("utf-8")
                    json_data = json.dumps(df.to_dict(orient="records"), indent=2)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button("Download CSV", csv, file_name="slice_annotations.csv", use_container_width=True)
                    with col2:
                        st.download_button("Download JSON", json_data, file_name="slice_annotations.json", use_container_width=True)
        else:
            show_access_denied("annotating or exporting annotations")
    
    finally:
        os.unlink(tmp_img_path)
        os.unlink(tmp_mask_path)

elif image_file or mask_file:
    st.warning("Please upload **both** an MRI image and its mask.")