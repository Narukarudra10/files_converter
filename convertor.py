import streamlit as st
import cv2
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import cairosvg # NEW: For rendering SVGs

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Universal File Converter", page_icon="🔄", layout="centered")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .stDownloadButton>button { width: 100%; border-radius: 8px; background-color: #4CAF50; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- 1. RASTER TO ANYTHING ---
def process_raster(image_file, target_fmt, mode):
    # Standard format conversion
    if target_fmt in ["JPG", "JPEG", "PNG", "WEBP", "BMP", "PDF"]:
        img = Image.open(image_file)
        if target_fmt in ["JPG", "JPEG", "PDF"] and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG" if target_fmt == "JPG" else target_fmt)
        return buf.getvalue()
        
    # Raster to Vector (Tracing)
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

# --- 2. SVG TO RASTER ---
def process_svg(file_bytes, target_fmt):
    if target_fmt == "PNG":
        return cairosvg.svg2png(bytestring=file_bytes)
    elif target_fmt == "PDF":
        return cairosvg.svg2pdf(bytestring=file_bytes)
    else:
        raise ValueError("SVGs can currently only be converted to PNG or PDF.")

# --- 3. DXF TO RASTER ---
def process_dxf(file_bytes, target_fmt):
    # Read the DXF math
    doc = ezdxf.read(io.StringIO(file_bytes.decode('utf-8')))
    msp = doc.modelspace()
    
    # Use Matplotlib to draw the CAD math onto a blank canvas
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp, finalize=True)
    
    # Save the canvas as an image
    buf = io.BytesIO()
    fig.savefig(buf, format=target_fmt.lower(), dpi=300)
    plt.close(fig)
    return buf.getvalue()

# --- USER INTERFACE ---
st.title("🔄 Ultimate Universal Converter")
st.markdown("Convert Pixels to Vectors, Vectors to Pixels, and everything in between.")

uploaded_file = st.file_uploader("Upload Image, SVG, or DXF", type=['png', 'jpg', 'jpeg', 'webp', 'svg', 'dxf'])

if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    st.markdown("---")
    st.subheader("Conversion Settings")
    
    # Dynamically change available outputs based on input
    if file_ext == 'svg':
        available_formats = ["PNG", "PDF"]
        st.info("💡 SVG detected. You can render this math file into a PNG image or PDF document.")
    elif file_ext == 'dxf':
        available_formats = ["PNG", "PDF", "SVG"]
        st.info("📐 CAD file detected. This will be drawn and rendered as a 2D image.")
    else:
        available_formats = ["DXF", "SVG", "PNG", "JPG", "WEBP", "PDF"]
        st.info("🖼️ Image detected. You can change its format or trace it into a Vector.")
        
    target_format = st.selectbox("Output Format", available_formats)
    
    if st.button("🚀 Convert Now", use_container_width=True):
        with st.spinner("Processing... (Vectors may take a few seconds to draw)"):
            try:
                # Reset file pointer
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                
                # Routing Engine
                if file_ext == 'svg':
                    output_bytes = process_svg(file_bytes, target_format)
                elif file_ext == 'dxf':
                    output_bytes = process_dxf(file_bytes, target_format)
                else:
                    uploaded_file.seek(0)
                    output_bytes = process_raster(uploaded_file, target_format, "Clean")
                
                mime_type = "application/pdf" if target_format == "PDF" else f"image/{target_format.lower()}"
                if target_format == "DXF": mime_type = "application/dxf"
                
                st.success("✅ Conversion successful!")
                
                original_name = uploaded_file.name.rsplit('.', 1)[0]
                st.download_button(
                    label=f"💾 Download {target_format}",
                    data=output_bytes,
                    file_name=f"{original_name}.{target_format.lower()}",
                    mime=mime_type,
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error during conversion: {e}")