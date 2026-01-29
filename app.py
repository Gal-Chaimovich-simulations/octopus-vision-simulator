"""
🐙 Octopus Vision Simulator - Streamlit Web App
Simulate chromatic aberration-based color perception

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.signal import fftconvolve
from PIL import Image
import io

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Octopus Vision Simulator",
    page_icon="🐙",
    layout="wide"
)

# ============================================================
# PHYSICAL PARAMETERS
# ============================================================

WAVELENGTHS = {'red': 650.0, 'green': 550.0, 'blue': 450.0}
FOCAL_LENGTHS = {'red': 52.0, 'green': 50.0, 'blue': 48.0}
Z_POSITIONS = {'z_blue': 48.0, 'z_green': 50.0, 'z_red': 52.0}
GRID_SIZE = 512

# ============================================================
# PSF FUNCTIONS
# ============================================================

def get_aperture_dimensions(aperture_mask):
    """Calculate aperture width and height in pixels"""
    rows = np.any(aperture_mask > 0, axis=1)
    cols = np.any(aperture_mask > 0, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return 0, 0
    
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    
    height = row_indices[-1] - row_indices[0] + 1
    width = col_indices[-1] - col_indices[0] + 1
    
    return width, height


def compute_W20(z_mm, f_mm, aperture_diameter_mm):
    """Compute defocus coefficient W20"""
    defocus_mm = z_mm - f_mm
    radius_mm = aperture_diameter_mm / 2.0
    W20_mm = defocus_mm * (radius_mm ** 2) / (2 * f_mm ** 2)
    return W20_mm, defocus_mm


def create_psf_fft(aperture_mask, wavelength_nm, W20_mm, aperture_diameter_mm):
    """Create PSF using FFT diffraction model"""
    grid_size = aperture_mask.shape[0]
    center = grid_size // 2
    
    wavelength_mm = wavelength_nm * 1e-6
    
    y, x = np.ogrid[-center:grid_size-center, -center:grid_size-center]
    r_pixels = np.sqrt(x**2 + y**2)
    
    aperture_radius_pixels = grid_size / 2.0
    rho = r_pixels / aperture_radius_pixels
    
    k = 2 * np.pi / wavelength_mm
    phase = k * W20_mm * (rho ** 2)
    
    pupil = aperture_mask * np.exp(1j * phase)
    psf = np.abs(fftshift(fft2(pupil))) ** 2
    
    total = np.sum(psf)
    if total > 0:
        psf = psf / total
    else:
        psf = np.zeros((grid_size, grid_size))
        psf[center, center] = 1.0
    
    return psf


def compute_psf(aperture_mask, z_mm, f_mm, wavelength_nm, aperture_diameter_mm):
    """Compute PSF for given parameters"""
    W20_mm, defocus_mm = compute_W20(z_mm, f_mm, aperture_diameter_mm)
    psf = create_psf_fft(aperture_mask, wavelength_nm, W20_mm, aperture_diameter_mm)
    
    blur_info = {
        'W20_mm': W20_mm,
        'defocus_mm': defocus_mm,
        'is_sharp': abs(defocus_mm) < 0.1
    }
    
    return psf, blur_info


def apply_psf_convolution(channel, psf):
    """Apply PSF to channel via convolution"""
    blurred = fftconvolve(channel, psf, mode='same')
    return np.clip(blurred, 0.0, 1.0)

# ============================================================
# APERTURE FUNCTIONS
# ============================================================

def create_circular_aperture(grid_size=512, diameter_px=200):
    """Create circular aperture"""
    aperture = np.zeros((grid_size, grid_size))
    center = grid_size // 2
    y, x = np.ogrid[-center:grid_size-center, -center:grid_size-center]
    radius = diameter_px // 2
    mask = x**2 + y**2 <= radius**2
    aperture[mask] = 1.0
    return aperture


def create_rectangular_aperture(grid_size=512, width_px=400, height_px=20):
    """Create rectangular aperture"""
    aperture = np.zeros((grid_size, grid_size))
    center = grid_size // 2
    
    y_start = center - height_px // 2
    y_end = center + height_px // 2
    x_start = center - width_px // 2
    x_end = center + width_px // 2
    
    aperture[y_start:y_end, x_start:x_end] = 1.0
    return aperture


def create_grating_aperture(grid_size=512, period_px=20, orientation='vertical'):
    """Create grating aperture"""
    aperture = np.zeros((grid_size, grid_size))
    
    if orientation == 'vertical':
        for x in range(grid_size):
            if (x % int(period_px)) < (period_px / 2):
                aperture[:, x] = 1.0
    else:
        for y in range(grid_size):
            if (y % int(period_px)) < (period_px / 2):
                aperture[y, :] = 1.0
    
    return aperture



def create_w_aperture(grid_size=512, width_px=400, height_px=200, thickness_px=20):
    """Create W-shaped aperture (like octopus pupil)"""
    aperture = np.zeros((grid_size, grid_size))
    center = grid_size // 2
    
    half_w = width_px // 2
    half_h = height_px // 2
    half_t = thickness_px // 2
    quarter_w = width_px // 4
    
    # W has 4 diagonal segments:
    # 1. Top-left down to bottom quarter-left
    # 2. Bottom quarter-left up to middle-top
    # 3. Middle-top down to bottom quarter-right
    # 4. Bottom quarter-right up to top-right
    
    # Segment 1: top-left to bottom-left-quarter
    for i in range(height_px):
        t = i / height_px
        y = center - half_h + i
        x = center - half_w + int(t * quarter_w)
        if 0 <= y < grid_size and 0 <= x < grid_size:
            y_s, y_e = max(0, y-half_t), min(grid_size, y+half_t)
            x_s, x_e = max(0, x-half_t), min(grid_size, x+half_t)
            aperture[y_s:y_e, x_s:x_e] = 1.0
    
    # Segment 2: bottom-left-quarter up to middle-top
    for i in range(height_px):
        t = i / height_px
        y = center + half_h - i
        x = center - quarter_w + int(t * quarter_w)
        if 0 <= y < grid_size and 0 <= x < grid_size:
            y_s, y_e = max(0, y-half_t), min(grid_size, y+half_t)
            x_s, x_e = max(0, x-half_t), min(grid_size, x+half_t)
            aperture[y_s:y_e, x_s:x_e] = 1.0
    
    # Segment 3: middle-top down to bottom-right-quarter
    for i in range(height_px):
        t = i / height_px
        y = center - half_h + i
        x = center + int(t * quarter_w)
        if 0 <= y < grid_size and 0 <= x < grid_size:
            y_s, y_e = max(0, y-half_t), min(grid_size, y+half_t)
            x_s, x_e = max(0, x-half_t), min(grid_size, x+half_t)
            aperture[y_s:y_e, x_s:x_e] = 1.0
    
    # Segment 4: bottom-right-quarter up to top-right
    for i in range(height_px):
        t = i / height_px
        y = center + half_h - i
        x = center + quarter_w + int(t * quarter_w)
        if 0 <= y < grid_size and 0 <= x < grid_size:
            y_s, y_e = max(0, y-half_t), min(grid_size, y+half_t)
            x_s, x_e = max(0, x-half_t), min(grid_size, x+half_t)
            aperture[y_s:y_e, x_s:x_e] = 1.0
    
    return aperture

# ============================================================
# SIMULATION FUNCTIONS
# ============================================================

def simulate_at_focal_plane(channels, aperture_mask, z_mm, aperture_diameter_mm):
    """Simulate image at specific focal plane"""
    psfs = {}
    blur_infos = {}
    blurred_channels = {}
    
    for color in ['red', 'green', 'blue']:
        psf, blur_info = compute_psf(
            aperture_mask, z_mm,
            FOCAL_LENGTHS[color],
            WAVELENGTHS[color],
            aperture_diameter_mm
        )
        psfs[color] = psf
        blur_infos[color] = blur_info
        blurred_channels[color] = apply_psf_convolution(channels[color], psf)
    
    rgb_image = np.stack([
        blurred_channels['red'],
        blurred_channels['green'],
        blurred_channels['blue']
    ], axis=2)
    
    return {
        'rgb_image': np.clip(rgb_image, 0.0, 1.0),
        'channels': blurred_channels,
        'psfs': psfs,
        'blur_infos': blur_infos
    }


def run_simulation(channels, aperture_mask, aperture_diameter_mm):
    """Run simulation at all focal planes"""
    results = {}
    
    for plane_name, z_pos in Z_POSITIONS.items():
        results[plane_name] = simulate_at_focal_plane(
            channels, aperture_mask, z_pos, aperture_diameter_mm
        )
    
    return results

# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    st.title("🐙 Octopus Vision Simulator")
    st.markdown("**Simulate chromatic aberration-based color perception**")
    
    # Sidebar
    st.sidebar.header("⚙️ Parameters")
    
    # Aperture diameter
    aperture_diameter_mm = st.sidebar.slider(
        "Aperture Diameter (mm)",
        min_value=1.0, max_value=50.0, value=5.0, step=0.5
    )
    
    pixel_size_um = (aperture_diameter_mm / GRID_SIZE) * 1000
    st.sidebar.info(f"Pixel size: {pixel_size_um:.1f} µm")
    
    # Aperture type
    st.sidebar.header("🔘 Aperture Shape")
    aperture_type = st.sidebar.selectbox(
        "Type",
        ["Circle", "Rectangle", "W-Shape (Octopus)", "Grating", "Upload Custom"]
    )
    
    # Create aperture based on type
    if aperture_type == "Circle":
        diameter_px = st.sidebar.slider("Diameter (pixels)", 50, 500, 200)
        aperture_mask = create_circular_aperture(GRID_SIZE, diameter_px)
        
    elif aperture_type == "Rectangle":
        width_px = st.sidebar.slider("Width (pixels)", 10, 500, 400)
        height_px = st.sidebar.slider("Height (pixels)", 10, 500, 20)
        aperture_mask = create_rectangular_aperture(GRID_SIZE, width_px, height_px)
    
    elif aperture_type == "W-Shape (Octopus)":
        w_width = st.sidebar.slider("W Width (pixels)", 100, 500, 300)
        w_height = st.sidebar.slider("W Height (pixels)", 50, 400, 200)
        w_thickness = st.sidebar.slider("Line Thickness (pixels)", 5, 50, 15)
        aperture_mask = create_w_aperture(GRID_SIZE, w_width, w_height, w_thickness)
        
    elif aperture_type == "Grating":
        period_px = st.sidebar.slider("Period (pixels)", 5, 100, 20)
        orientation = st.sidebar.radio("Orientation", ["vertical", "horizontal"])
        aperture_mask = create_grating_aperture(GRID_SIZE, period_px, orientation)
        
    else:  # Upload Custom
        uploaded_aperture = st.sidebar.file_uploader(
            "Upload aperture (PNG/NPY)", type=['png', 'npy']
        )
        if uploaded_aperture:
            if uploaded_aperture.name.endswith('.npy'):
                aperture_mask = np.load(uploaded_aperture)
            else:
                img = Image.open(uploaded_aperture).convert('L')
                img = img.resize((GRID_SIZE, GRID_SIZE))
                aperture_mask = np.array(img) / 255.0
        else:
            aperture_mask = create_circular_aperture(GRID_SIZE, 200)
    
    # Image upload
    st.sidebar.header("🖼️ Input Image")
    uploaded_image = st.sidebar.file_uploader(
        "Upload image", type=['png', 'jpg', 'jpeg']
    )
    
    # Main content - two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Aperture")
        fig_apt, ax_apt = plt.subplots(figsize=(4, 4))
        ax_apt.imshow(aperture_mask, cmap='gray')
        ax_apt.set_title(f"Aperture ({aperture_type})")
        ax_apt.axis('off')
        st.pyplot(fig_apt)
        plt.close()
        
        w, h = get_aperture_dimensions(aperture_mask)
        st.caption(f"Size: {w} × {h} pixels")
    
    with col2:
        st.subheader("Input Image")
        if uploaded_image:
            img = Image.open(uploaded_image).convert('RGB')
            img = img.resize((GRID_SIZE, GRID_SIZE))
            image_array = np.array(img) / 255.0
            st.image(image_array, caption="Uploaded Image")
        else:
            # Default: white circle
            st.info("No image uploaded. Using default white circle.")
            image_array = np.zeros((GRID_SIZE, GRID_SIZE, 3))
            center = GRID_SIZE // 2
            y, x = np.ogrid[-center:GRID_SIZE-center, -center:GRID_SIZE-center]
            mask = x**2 + y**2 <= 50**2
            image_array[mask, :] = 1.0
            st.image(image_array, caption="Default: White Circle")
    
    # Run simulation button
    if st.button("🚀 Run Simulation", type="primary"):
        
        with st.spinner("Running simulation..."):
            # Prepare channels
            channels = {
                'red': image_array[:, :, 0],
                'green': image_array[:, :, 1],
                'blue': image_array[:, :, 2]
            }
            
            # Run
            results = run_simulation(channels, aperture_mask, aperture_diameter_mm)
        
        st.success("Simulation complete!")
        
        # Display results
        st.header("📊 Results")
        
        # Show all z positions
        cols = st.columns(4)
        
        with cols[0]:
            st.subheader("Original")
            st.image(image_array, caption="All in Focus")
        
        for i, (plane_name, z_pos) in enumerate(Z_POSITIONS.items()):
            with cols[i + 1]:
                result = results[plane_name]
                sharp_colors = [c for c, info in result['blur_infos'].items() if info['is_sharp']]
                sharp_str = ", ".join([c.capitalize() for c in sharp_colors]) if sharp_colors else "None"
                
                st.subheader(f"z = {z_pos} mm")
                st.image(result['rgb_image'], caption=f"{sharp_str} Sharp")
        
        # PSF visualization
        st.header("🔬 Point Spread Functions (PSF)")
        
        fig_psf, axes = plt.subplots(3, 3, figsize=(10, 10))
        
        for i, (plane_name, z_pos) in enumerate(Z_POSITIONS.items()):
            for j, color in enumerate(['red', 'green', 'blue']):
                psf = results[plane_name]['psfs'][color]
                blur_info = results[plane_name]['blur_infos'][color]
                
                # Show center crop
                center = GRID_SIZE // 2
                crop = 50
                psf_crop = psf[center-crop:center+crop, center-crop:center+crop]
                
                axes[i, j].imshow(psf_crop, cmap='hot')
                status = "★ SHARP" if blur_info['is_sharp'] else f"W20={blur_info['W20_mm']*1000:.1f}µm"
                axes[i, j].set_title(f"z={z_pos}mm\n{color.capitalize()}\n{status}", fontsize=9)
                axes[i, j].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig_psf)
        plt.close()
        
        # Download results
        st.header("💾 Download Results")
        
        for plane_name, z_pos in Z_POSITIONS.items():
            rgb_img = results[plane_name]['rgb_image']
            img_pil = Image.fromarray((rgb_img * 255).astype(np.uint8))
            
            buf = io.BytesIO()
            img_pil.save(buf, format='PNG')
            
            st.download_button(
                label=f"Download z={z_pos}mm",
                data=buf.getvalue(),
                file_name=f"octopus_z{z_pos}mm.png",
                mime="image/png"
            )
    
    # Info section
    with st.expander("ℹ️ About"):
        st.markdown("""
        ### How it works
        
        This simulator demonstrates how octopuses might perceive color through **chromatic aberration**.
        
        **Physical model:**
        - Different wavelengths have different focal lengths
        - Red (650nm): f = 52mm
        - Green (550nm): f = 50mm  
        - Blue (450nm): f = 48mm
        
        **PSF calculation:**
        ```
        W20 = (z - f) × (D/2)² / (2 × f²)
        phase = (2π/λ) × W20 × ρ²
        PSF = |FFT(aperture × exp(j × phase))|²
        ```
        
        At each sensor position z, one color is in focus while others are blurred.
        """)

if __name__ == "__main__":
    main()
