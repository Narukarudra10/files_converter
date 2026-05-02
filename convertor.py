import streamlit as st
import cv2
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import cairosvg 
import zipfile # NEW: Needed to package multiple files together

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Universal File Converter Pro",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM HTML/CSS (NAVBAR, FOOTER, & STYLING) ---
st.markdown("""
    <style>
    /* Hide Streamlit default header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .custom-navbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #0F172A;
        color: white;
        padding: 15px 30px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        z-index: 99999;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .navbar-brand { 
        font-weight: 800; 
        font-size: 1.2rem; 
        letter-spacing: 1px;
    }
    .navbar-links { 
        font-size: 0.9rem; 
        color: #cbd5e1; 
        font-weight: 500;
    }
    .navbar-links span {
        margin-left: 20px;
        cursor: pointer;
    }
    .navbar-links span:hover {
        color: white;
    }

    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0F172A;
        color: #64748B;
        text-align: center;
        padding: 12px 0;
        font-size: 0.85rem;
        z-index: 99999;
        border-top: 1px solid #1E293B;
    }
    
    .block-container {
        padding-top: 6rem !important;
        padding-bottom: 6rem !important;
        max-width: 800px;
    }

    .main-title {
        text-align: center;
        font-weight: 800;
        font-size: 2.5rem;
        color: var(--text-color); 
        margin-bottom: 0px;
    }
    
    .sub-title {
        text-align: center;
        color: gray;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .stDownloadButton>button {
        border-radius: 8px;
        font-weight: 600;
        height: 3.5rem;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        border: none;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }
    </style>

    <div class="custom-navbar">
        <div class="navbar-brand">⚡ NEXUS</div>
        <div class="navbar-links">
            <span>Home</span>
            <span>API Docs</span>
            <span>Support</span>
        </div>
    </div>

    <div class="custom-footer">
        &copy; 2026 Nexus Tools. Secure, local, and fast conversions.
    </div>
""", unsafe_allow_html=True)

# --- 3. CORE LOGIC ---
def process_raster(image_file, target_fmt, mode):
    if target_fmt in ["JPG", "JPEG", "PNG", "WEBP", "BMP", "PDF"]:
        img = Image.open(image_file)
        if target_fmt in ["JPG", "JPEG", "PDF"] and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG" if target_fmt == "JPG" else target_fmt)
        return buf.getvalue()
        
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    _, processed = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        raise ValueError("No shapes found in the image to convert.")

    all_points = np.vstack(contours)
    x_min, y_min, pixel_w, pixel_h = cv2.boundingRect(all_points)

    target_w_mm = 56.0
    target_h_mm = 88.0
    scale_x = target_w_mm / pixel_w
    scale_y = target_h_mm / pixel_h

    if target_fmt == "DXF":
        doc = ezdxf.new('R2010')
        doc.header['$INSUNITS'] = 4 
        msp = doc.modelspace()
        
        if hierarchy is not None:
            for cnt in contours:
                points = [((pt[0][0] - x_min) * scale_x, -((pt[0][1] - y_min) * scale_y)) for pt in cnt]
                if len(points) >= 3:
                    msp.add_lwpolyline(points, close=True)
        buf = io.StringIO()
        doc.write(buf)
        return buf.getvalue().encode('utf-8')
        
    elif target_fmt == "SVG":
        svg_lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="56mm" height="88mm" viewBox="0 0 {target_w_mm} {target_h_mm}">']
        if hierarchy is not None:
            for cnt in contours:
                if len(cnt) >= 3:
                    path_points = [f"{(pt[0][0] - x_min) * scale_x},{(pt[0][1] - y_min) * scale_y}" for pt in cnt]
                    path_data = "M " + " L ".join(path_points) + " Z"
                    fill = "none" if "Outlines" in mode else "black"
                    stroke = "black" if "Outlines" in mode else "none"
                    svg_lines.append(f'<path d="{path_data}" fill="{fill}" stroke="{stroke}" stroke-width="0.1"/>')
        svg_lines.append('</svg>')
        return "\n".join(svg_lines).encode('utf-8')

def process_svg(file_bytes, target_fmt):
    if target_fmt == "PNG":
        return cairosvg.svg2png(bytestring=file_bytes)
    elif target_fmt == "PDF":
        return cairosvg.svg2pdf(bytestring=file_bytes)
    else:
        raise ValueError("SVGs can currently only be converted to PNG or PDF.")

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
# NEW: accept_multiple_files=True allows you to highlight and upload as many as you want
uploaded_files = st.file_uploader("Drag and drop your files here", type=['png', 'jpg', 'jpeg', 'webp', 'svg', 'dxf'], accept_multiple_files=True, label_visibility="collapsed")

if uploaded_files:
    # Check what types of files were uploaded to adjust the UI
    exts = set([f.name.split('.')[-1].lower() for f in uploaded_files])
    has_svg = 'svg' in exts
    has_dxf = 'dxf' in exts
    
    with st.container():
        st.success(f"**Ready:** {len(uploaded_files)} file(s) selected for batch processing.")
    
    st.markdown("---")
    st.markdown("### 2. Configure Output")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Show a universal list of formats for batch processing
        available_formats = ["DXF", "SVG", "PNG", "JPG", "WEBP", "PDF"]
        target_format = st.selectbox("Convert All Files To:", available_formats, label_visibility="collapsed")
    
    with col2:
        st.info("💡 All selected files will be converted to this target format.")

    st.markdown("---")
    st.markdown("### 3. Process Batch")
    
    if st.button("Convert Files", use_container_width=True, type="primary"):
        with st.status(f"Processing {len(uploaded_files)} files...", expanded=True) as status:
            try:
                # Set up a ZIP file buffer in memory
                zip_buffer = io.BytesIO()
                success_count = 0
                
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for uploaded_file in uploaded_files:
                        file_ext = uploaded_file.name.split('.')[-1].lower()
                        st.write(f"Converting {uploaded_file.name} to {target_format}...")
                        
                        uploaded_file.seek(0)
                        file_bytes = uploaded_file.read()
                        
                        # Route to the correct engine based on the individual file's original type
                        if file_ext == 'svg':
                            output_bytes = process_svg(file_bytes, target_format)
                        elif file_ext == 'dxf':
                            output_bytes = process_dxf(file_bytes, target_format)
                        else:
                            uploaded_file.seek(0)
                            output_bytes = process_raster(uploaded_file, target_format, "Clean")
                            
                        # Figure out the new filename
                        original_name = uploaded_file.name.rsplit('.', 1)[0]
                        new_filename = f"{original_name}.{target_format.lower()}"
                        
                        # Add the converted file to our ZIP package
                        zip_file.writestr(new_filename, output_bytes)
                        success_count += 1
                
                status.update(label=f"Successfully converted {success_count} files!", state="complete", expanded=False)
                st.balloons() 
                
                # If they only uploaded 1 file, download it normally. If more than 1, download the ZIP.
                if len(uploaded_files) == 1:
                    st.download_button(
                        label=f"💾 Download {target_format} File",
                        data=output_bytes,
                        file_name=new_filename,
                        mime=f"application/{target_format.lower()}"
                    )
                else:
                    st.download_button(
                        label=f"📦 Download ZIP ({success_count} files)",
                        data=zip_buffer.getvalue(),
                        file_name="nexus_batch_conversion.zip",
                        mime="application/zip"
                    )
                
            except Exception as e:
                status.update(label="Conversion Failed", state="error", expanded=True)
                st.error(f"Error details: {e}")
