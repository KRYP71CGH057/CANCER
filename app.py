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

st.set_page_config(page_title="MRI & Mask Viewer", layout="wide")

# -------------------- AUTH --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Dummy credentials with roles
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

if not st.session_state.logged_in:
    st.title("🔒 Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USERS and USERS[username]["password"] == hashlib.sha256(password.encode()).hexdigest():
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = USERS[username]["role"]
            st.success(f"Welcome, {username} ({st.session_state.role})!")
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

st.sidebar.markdown(f"👤 **User:** `{st.session_state.username}` ({st.session_state.role})")

if st.sidebar.button("🚪 Logout"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# -------------------- MAIN APP --------------------

st.title("🧠 MRI + AI Mask Viewer")
st.markdown("Upload an MRI scan and its AI mask to view, annotate, and export overlays.")

# Upload files
image_file = st.file_uploader("Upload MRI image (.nii or .nii.gz)", type=["nii", "nii.gz"])
mask_file = st.file_uploader("Upload AI mask (.nii or .nii.gz)", type=["nii", "nii.gz"])

if image_file and mask_file:
    # Save uploaded files temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_img:
        tmp_img.write(image_file.read())
        tmp_img_path = tmp_img.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_mask:
        tmp_mask.write(mask_file.read())
        tmp_mask_path = tmp_mask.name

    try:
        # Load using nibabel
        img_nii = nib.load(tmp_img_path)
        mask_nii = nib.load(tmp_mask_path)

        img_data = img_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        st.session_state.img_data = img_data
        st.session_state.mask_data = mask_data

        # Slice slider
        st.subheader("Select a Slice")
        max_slices = img_data.shape[2]
        slice_index = st.slider("Slice index", 0, max_slices - 1, max_slices // 2)

        # Show selected slice
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(img_data[:, :, slice_index], cmap="gray")
        ax[0].set_title("MRI Slice")
        ax[0].axis("off")

        ax[1].imshow(img_data[:, :, slice_index], cmap="gray")
        ax[1].imshow(mask_data[:, :, slice_index], cmap="Reds", alpha=0.4)
        ax[1].set_title("MRI + AI Mask Overlay")
        ax[1].axis("off")

        st.pyplot(fig)

        # Batch Export Overlay Slices (admin only)
        st.header("📦 Batch Overlay Slice Export")
        def export_overlay_slices(img_data, mask_data, plane='axial', out_dir='exports'):
            os.makedirs(out_dir, exist_ok=True)
            slices = img_data.shape[2] if plane == 'axial' else img_data.shape[1] if plane == 'coronal' else img_data.shape[0]
            for i in range(slices):
                fig, ax = plt.subplots()
                if plane == 'axial':
                    img_slice = img_data[:, :, i]
                    mask_slice = mask_data[:, :, i]
                elif plane == 'coronal':
                    img_slice = img_data[:, i, :]
                    mask_slice = mask_data[:, i, :]
                else:
                    img_slice = img_data[i, :, :]
                    mask_slice = mask_data[i, :, :]
                ax.imshow(img_slice.T, cmap='gray', origin='lower')
                ax.imshow(mask_slice.T, cmap='Reds', alpha=0.5, origin='lower')
                ax.axis('off')
                filename = f"{out_dir}/overlay_{plane}_{i:03d}.png"
                plt.savefig(filename, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

        def zip_export_dir(out_dir='exports'):
            zip_path = f"{out_dir}.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for root, _, files in os.walk(out_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file), arcname=file)
            return zip_path

        plane = st.selectbox("Select Plane for Export", ['axial', 'coronal', 'sagittal'])
        if st.session_state.role == "admin":
            if st.button("Export All Overlay Slices"):
                export_overlay_slices(img_data, mask_data, plane)
                zip_path = zip_export_dir()
                with open(zip_path, "rb") as f:
                    st.download_button("Download All Overlays (ZIP)", f, file_name="overlay_slices.zip")
        else:
            st.info("🛑 Only admins can export overlay slices.")

        # Annotation & Export (admin + radiologist)
        st.header("📝 Per-Slice Annotation & Export")
        if "annotations" not in st.session_state:
            st.session_state.annotations = {}

        plane_annot = st.selectbox("Select Plane for Annotation", ['axial', 'coronal', 'sagittal'], key="annot")
        num_slices = img_data.shape[2] if plane_annot == 'axial' else img_data.shape[1] if plane_annot == 'coronal' else img_data.shape[0]
        slice_index_annot = st.slider(f"Slice Index ({plane_annot})", 0, num_slices - 1, 0, key="annot_slider")

        def get_slice(img, mask, plane, idx):
            if plane == 'axial':
                return img[:, :, idx], mask[:, :, idx]
            elif plane == 'coronal':
                return img[:, idx, :], mask[:, idx, :]
            else:
                return img[idx, :, :], mask[idx, :, :]

        img_slice, mask_slice = get_slice(img_data, mask_data, plane_annot, slice_index_annot)

        st.image(img_slice.T, caption=f"{plane_annot.capitalize()} Slice {slice_index_annot} (Image)", clamp=True, use_container_width=True)
        st.image(mask_slice.T, caption=f"{plane_annot.capitalize()} Slice {slice_index_annot} (Mask)", clamp=True, use_container_width=True)

        default_comment = st.session_state.annotations.get((plane_annot, slice_index_annot), "")
        comment = st.text_area(f"Annotation for {plane_annot} slice {slice_index_annot}:", default_comment)

        if st.session_state.role in ["admin", "radiologist"]:
            if st.button("💾 Save Annotation"):
                st.session_state.annotations[(plane_annot, slice_index_annot)] = comment
                st.success("Annotation saved!")

            if st.button("📤 Export All Annotations"):
                df = pd.DataFrame([
                    {"plane": k[0], "slice_index": k[1], "comment": v}
                    for k, v in st.session_state.annotations.items()
                ])
                csv = df.to_csv(index=False).encode("utf-8")
                json_data = json.dumps(df.to_dict(orient="records"), indent=2)

                st.download_button("Download CSV", csv, file_name="slice_annotations.csv")
                st.download_button("Download JSON", json_data, file_name="slice_annotations.json")
        else:
            st.info("🛑 Guests can view but not save/export annotations.")

    finally:
        os.unlink(tmp_img_path)
        os.unlink(tmp_mask_path)

elif image_file or mask_file:
    st.warning("Please upload **both** an MRI image and its mask.")
