import streamlit as st
import cv2
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image, ImageOps 
import cairosvg 
import zipfile 

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Universal File Converter Pro",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- INITIALIZE MEMORY FOR DOWNLOAD BUG FIX ---
if 'download_ready' not in st.session_state:
    st.session_state.download_ready = False
    st.session_state.file_data = None
    st.session_state.file_name = ""
    st.session_state.mime_type = ""
    st.session_state.button_label = ""

# --- 2. CUSTOM HTML/CSS ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .custom-navbar {
        position: fixed; top: 0; left: 0; width: 100%;
        background-color: #0F172A; color: white; padding: 15px 30px;
        display: flex; justify-content: space-between; align-items: center;
        z-index: 99999; box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .navbar-brand { font-weight: 800; font-size: 1.2rem; letter-spacing: 1px; }
    .navbar-links { font-size: 0.9rem; color: #cbd5e1; font-weight: 500; }
    .navbar-links span { margin-left: 20px; cursor: pointer; }
    .navbar-links span:hover { color: white; }

    .custom-footer {
        position: fixed; bottom: 0; left: 0; width: 100%;
        background-color: #0F172A; color: #64748B; text-align: center;
        padding: 12px 0; font-size: 0.85rem; z-index: 99999;
        border-top: 1px solid #1E293B;
    }
    
    .block-container {
        padding-top: 6rem !important; padding-bottom: 6rem !important;
        max-width: 800px;
    }

    .main-title {
        text-align: center; font-weight: 800; font-size: 2.5rem;
        color: var(--text-color); margin-bottom: 0px;
    }
    .sub-title {
        text-align: center; color: gray; font-size: 1.1rem; margin-bottom: 2rem;
    }

    .stDownloadButton>button {
        border-radius: 8px; font-weight: 600; height: 3.5rem; width: 100%;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white; border: none; margin-top: 10px;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }
    </style>

    <div class="custom-navbar">
        <div class="navbar-brand">⚡ NEXUS</div>
        <div class="navbar-links">
            <span>Home</span><span>API Docs</span><span>Support</span>
        </div>
    </div>
    <div class="custom-footer">&copy; 2026 Nexus Tools. Secure, local, and fast conversions.</div>
""", unsafe_allow_html=True)

# --- 3. CORE LOGIC ---
def process_raster(image_file, target_fmt, mode):
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img) 

    if target_fmt in ["JPG", "JPEG", "PNG", "WEBP", "BMP", "PDF"]:
        if target_fmt in ["JPG", "JPEG", "PDF"] and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG" if target_fmt == "JPG" else target_fmt)
        return buf.getvalue()
        
    img_gray = img.convert("L")
    image = np.array(img_gray)
    
    scale_factor = 1.0 # Default scale
    
    # NEW: HIGH PRECISION MODE FOR QR CODES
    if "High Precision" in mode:
        # Upscale 4x to generate perfectly smooth curves without pixelation
        scale_factor = 4.0
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        # Slight blur to anti-alias the upscaled edges
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        # Otsu's thresholding automatically finds the perfect black/white split
        _, processed = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contour_mode = cv2.CHAIN_APPROX_SIMPLE
        epsilon_factor = 0.0001 # Almost zero error tolerance for perfect circles

    elif "Clean" in mode:
        _, processed = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
        contour_mode = cv2.CHAIN_APPROX_SIMPLE
        epsilon_factor = 0.001 
    elif "Photos" in mode:
        smoothed_raster = cv2.medianBlur(image, 3) 
        processed = cv2.adaptiveThreshold(smoothed_raster, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contour_mode = cv2.CHAIN_APPROX_SIMPLE
        epsilon_factor = 0.005 
    else: 
        processed = cv2.Canny(image, 100, 200)
        contour_mode = cv2.CHAIN_APPROX_SIMPLE
        epsilon_factor = 0.002
        
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_TREE, contour_mode)

    if target_fmt == "DXF":
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        if hierarchy is not None:
            for cnt in contours:
                epsilon = epsilon_factor * cv2.arcLength(cnt, True)
                approx_points = cv2.approxPolyDP(cnt, epsilon, True)
                
                # Divide coordinates by the scale factor to return shape to original size
                points = [((pt[0][0] / scale_factor), -(pt[0][1] / scale_factor)) for pt in approx_points]
                if len(points) >= 3:
                    msp.add_lwpolyline(points, close=True)
        buf = io.StringIO()
        doc.write(buf)
        return buf.getvalue().encode('utf-8')
        
    elif target_fmt == "SVG":
        # Divide canvas size by scale factor
        svg_w, svg_h = int(image.shape[1] / scale_factor), int(image.shape[0] / scale_factor)
        svg_lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}">']
        if hierarchy is not None:
            for cnt in contours:
                epsilon = epsilon_factor * cv2.arcLength(cnt, True)
                approx_points = cv2.approxPolyDP(cnt, epsilon, True)
                
                points = approx_points.reshape(-1, 2)
                if len(points) >= 3:
                    # Divide paths by scale factor
                    path_data = "M " + " L ".join([f"{x / scale_factor},{y / scale_factor}" for x, y in points]) + " Z"
                    fill = "none" if "Outlines" in mode else "black"
                    stroke = "black" if "Outlines" in mode else "none"
                    svg_lines.append(f'<path d="{path_data}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
        svg_lines.append('</svg>')
        return "\n".join(svg_lines).encode('utf-8')

def process_svg(file_bytes, target_fmt, mode):
    if target_fmt == "SVG":
        return file_bytes
    elif target_fmt == "PNG":
        return cairosvg.svg2png(bytestring=file_bytes)
    elif target_fmt == "PDF":
        return cairosvg.svg2pdf(bytestring=file_bytes)
    else:
        png_bytes = cairosvg.svg2png(bytestring=file_bytes, scale=3.0) 
        pseudo_file = io.BytesIO(png_bytes)
        return process_raster(pseudo_file, target_fmt, mode)

def process_dxf(file_bytes, target_fmt):
    doc = ezdxf.read(io.StringIO(file_bytes.decode('utf-8')))
    msp = doc.modelspace()
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp, finalize=True)
    buf = io.BytesIO()
    fig.savefig(buf, format=target_fmt.lower(), dpi=300)
    plt.close(fig)
    return buf.getvalue()

# --- 4. USER INTERFACE ---

st.markdown('<p class="main-title">Nexus Converter</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">A professional suite to convert pixels, vectors, and documents instantly.</p>', unsafe_allow_html=True)

st.markdown("### 1. Select your files")

def reset_download():
    st.session_state.download_ready = False

uploaded_files = st.file_uploader("Drag and drop your files here", type=['png', 'jpg', 'jpeg', 'webp', 'bmp', 'svg', 'dxf'], accept_multiple_files=True, label_visibility="collapsed", on_change=reset_download)

if uploaded_files:
    exts = set([f.name.split('.')[-1].lower() for f in uploaded_files])
    
    with st.container():
        st.success(f"**Ready:** {len(uploaded_files)} file(s) selected for processing.")
    
    st.markdown("---")
    st.markdown("### 2. Configure Output")
    
    col1, col2 = st.columns(2)
    with col1:
        available_formats = ["DXF", "SVG", "PNG", "JPG", "WEBP", "PDF"]
        target_format = st.selectbox("Convert All Files To:", available_formats, on_change=reset_download)
    
    with col2:
        is_vector = target_format in ["DXF", "SVG"]
        vector_mode = st.selectbox(
            "Vector Tracing Style:", 
            ["High Precision (Best for QR & Round Dots)", "Clean Digital Graphics", "Photos / Shadows", "Outlines Only (Edge)"],
            disabled=not is_vector,
            help="Only applies when converting images to vectors.",
            on_change=reset_download
        )
        if target_format == "PDF" and len(uploaded_files) > 1:
            st.info("💡 Images will be combined into a single, multi-page PDF document.")
        elif not is_vector:
            st.info("💡 Standard file conversion will be applied.")

    st.markdown("---")
    st.markdown("### 3. Process Batch")
    
    if st.button("⚙️ Convert Files", use_container_width=True):
        st.session_state.download_ready = False 
        
        with st.status(f"Processing {len(uploaded_files)} files...", expanded=True) as status:
            try:
                # =========================================================
                # SPECIAL CASE: COMBINE MULTIPLE IMAGES INTO A SINGLE PDF
                # =========================================================
                if target_format == "PDF" and len(uploaded_files) > 1:
                    st.write("Combining images into a single multi-page PDF...")
                    images_for_pdf = []
                    
                    for uploaded_file in uploaded_files:
                        file_ext = uploaded_file.name.split('.')[-1].lower()
                        if file_ext in ['png', 'jpg', 'jpeg', 'webp', 'bmp']:
                            img = Image.open(uploaded_file)
                            img = ImageOps.exif_transpose(img)
                            
                            if img.mode in ("RGBA", "P"):
                                img = img.convert("RGB")
                            images_for_pdf.append(img)
                        else:
                            st.warning(f"Skipping {uploaded_file.name}: Only standard images can be stitched into a PDF.")

                    if images_for_pdf:
                        pdf_buffer = io.BytesIO()
                        images_for_pdf[0].save(pdf_buffer, format="PDF", save_all=True, append_images=images_for_pdf[1:])
                        
                        st.session_state.file_data = pdf_buffer.getvalue()
                        st.session_state.file_name = "nexus_combined_document.pdf"
                        st.session_state.mime_type = "application/pdf"
                        st.session_state.button_label = f"💾 Download Combined PDF ({len(images_for_pdf)} pages)"
                        st.session_state.download_ready = True
                        
                        status.update(label="Successfully created combined PDF document!", state="complete", expanded=False)
                        st.balloons()
                    else:
                        status.update(label="No valid images found to combine.", state="error")

                # =========================================================
                # STANDARD CASE: INDIVIDUAL CONVERSION AND ZIP BUNDLING
                # =========================================================
                else:
                    zip_buffer = io.BytesIO()
                    success_count = 0
                    
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for uploaded_file in uploaded_files:
                            file_ext = uploaded_file.name.split('.')[-1].lower()
                            st.write(f"Converting {uploaded_file.name} to {target_format}...")
                            
                            uploaded_file.seek(0)
                            file_bytes = uploaded_file.read()
                            
                            if file_ext == 'svg':
                                output_bytes = process_svg(file_bytes, target_format, vector_mode)
                            elif file_ext == 'dxf':
                                output_bytes = process_dxf(file_bytes, target_format)
                            else:
                                uploaded_file.seek(0)
                                output_bytes = process_raster(uploaded_file, target_format, vector_mode)
                                
                            original_name = uploaded_file.name.rsplit('.', 1)[0]
                            new_filename = f"{original_name}.{target_format.lower()}"
                            
                            zip_file.writestr(new_filename, output_bytes)
                            success_count += 1
                    
                    if len(uploaded_files) == 1:
                        st.session_state.file_data = output_bytes
                        st.session_state.file_name = new_filename
                        st.session_state.mime_type = f"application/{target_format.lower()}"
                        st.session_state.button_label = f"💾 Download {target_format} File"
                    else:
                        st.session_state.file_data = zip_buffer.getvalue()
                        st.session_state.file_name = "nexus_batch_conversion.zip"
                        st.session_state.mime_type = "application/zip"
                        st.session_state.button_label = f"📦 Download ZIP ({success_count} files)"
                    
                    st.session_state.download_ready = True
                    status.update(label=f"Successfully converted {success_count} files!", state="complete", expanded=False)
                    st.balloons() 
                    
            except Exception as e:
                status.update(label="Conversion Failed", state="error", expanded=True)
                st.error(f"Error details: {e}")

    # --- THE DOWNLOAD BUTTON ---
    if st.session_state.download_ready:
        st.download_button(
            label=st.session_state.button_label,
            data=st.session_state.file_data,
            file_name=st.session_state.file_name,
            mime=st.session_state.mime_type
        )
