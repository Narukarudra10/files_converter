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

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Universal File Converter Pro",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS (THE PRO LOOK) ---
st.markdown("""
    <style>
    /* Hide Streamlit default header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Background and typography spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }

    /* Style the main title */
    .main-title {
        text-align: center;
        font-weight: 800;
        font-size: 2.5rem;
        color: #1E293B;
        margin-bottom: 0px;
    }
    
    .sub-title {
        text-align: center;
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Primary Action Button (Convert) */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 3rem;
        background-color: #0F172A;
        color: white;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* Success Download Button */
    .stDownloadButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 3.5rem;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3);
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        transform: translateY(-1px);
    }

    /* Styling Expander and Info boxes */
    .stAlert {
        border-radius: 8px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. CORE LOGIC (UNCHANGED) ---
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
    
    if target_fmt == "DXF":
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        if hierarchy is not None:
            for cnt in contours:
                points = [(pt[0][0], -pt[0][1]) for pt in cnt]
                if len(points) >= 3:
                    msp.add_lwpolyline(points, close=True)
        buf = io.StringIO()
        doc.write(buf)
        return buf.getvalue().encode('utf-8')

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

# Header Section
st.markdown('<p class="main-title">⚡ Nexus Converter</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">A professional suite to convert pixels, vectors, and documents instantly.</p>', unsafe_allow_html=True)

# Step 1: Upload
st.markdown("### 1. Select your file")
uploaded_file = st.file_uploader("Drag and drop your file here", type=['png', 'jpg', 'jpeg', 'webp', 'svg', 'dxf'], label_visibility="collapsed")

if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    file_size_kb = uploaded_file.size / 1024
    
    # Show File Info in a clean container
    with st.container():
        st.success(f"**Ready:** {uploaded_file.name} ({file_size_kb:.1f} KB)")
    
    st.markdown("---")
    
    # Step 2: Configuration
    st.markdown("### 2. Configure Output")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if file_ext == 'svg':
            available_formats = ["PNG", "PDF"]
            st.caption("SVG mode active")
        elif file_ext == 'dxf':
            available_formats = ["PNG", "PDF", "SVG"]
            st.caption("CAD mode active")
        else:
            available_formats = ["DXF", "SVG", "PNG", "JPG", "WEBP", "PDF"]
            st.caption("Standard image mode active")
            
        target_format = st.selectbox("Output Format", available_formats, label_visibility="collapsed")
    
    with col2:
        if target_format in ["DXF", "SVG"] and file_ext not in ['dxf', 'svg']:
            st.info("💡 Tracing algorithms will be applied to create vector lines.")
        elif target_format in ["PNG", "JPG", "WEBP"] and file_ext in ['dxf', 'svg']:
            st.info("💡 Mathematical paths will be rendered into a flat pixel image.")
        else:
            st.info("💡 Standard fast-pixel conversion will be applied.")

    st.markdown("---")
    
    # Step 3: Action
    st.markdown("### 3. Process")
    
    if st.button("Convert File"):
        # The loading spinner
        with st.status("Processing your file...", expanded=True) as status:
            try:
                st.write("Initializing engine...")
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                
                st.write(f"Converting {file_ext.upper()} to {target_format}...")
                
                if file_ext == 'svg':
                    output_bytes = process_svg(file_bytes, target_format)
                elif file_ext == 'dxf':
                    output_bytes = process_dxf(file_bytes, target_format)
                else:
                    uploaded_file.seek(0)
                    output_bytes = process_raster(uploaded_file, target_format, "Clean")
                
                mime_type = "application/pdf" if target_format == "PDF" else f"image/{target_format.lower()}"
                if target_format == "DXF": mime_type = "application/dxf"
                
                status.update(label="Conversion Complete!", state="complete", expanded=False)
                
                st.balloons() # Adds a nice celebratory touch for successful conversion
                
                original_name = uploaded_file.name.rsplit('.', 1)[0]
                
                # The big green download button
                st.download_button(
                    label=f"💾 Download {target_format} File",
                    data=output_bytes,
                    file_name=f"{original_name}.{target_format.lower()}",
                    mime=mime_type
                )
                
            except Exception as e:
                status.update(label="Conversion Failed", state="error", expanded=True)
                st.error(f"Error details: {e}")
