import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import euclidean
import colorspacious as cs

st.set_page_config(page_title="Color Scale Perceptual Uniformity Analyzer", layout="wide")

# Helper functions
def interpolate_colors(hex_colors, n):
    """Interpolate colors given in HEX to n colors."""
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_palette", hex_colors)
    interpolated_colors = [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]
    return interpolated_colors

def hex_to_rgb(hex_color):
    """Convert hex color to RGB (0-1 range)."""
    return mcolors.to_rgb(hex_color)

def perceptual_uniformity_test(hex_colors, color_space="CIE Lab"):
    """
    Perform quantitative perceptual uniformity test.
    
    Args:
        hex_colors (list): List of colors in HEX format.
        color_space (str): Color space to use: "CIE Lab" or "CAM02-UCS".
    
    Returns:
        dict: Test results with ΔE differences, average and standard deviation.
    """
    # Convert HEX to RGB
    rgb_colors = np.array([hex_to_rgb(color) for color in hex_colors])
    
    # Map to selected color space
    if color_space == "CIE Lab":
        coords = [cs.cspace_convert(rgb, "sRGB1", "CIELab") for rgb in rgb_colors]
    elif color_space == "CAM02-UCS":
        coords = [cs.cspace_convert(rgb, "sRGB1", "CAM02-UCS") for rgb in rgb_colors]
    else:
        raise ValueError("Color space not supported. Use 'CIE Lab' or 'CAM02-UCS'.")
    
    coords = np.array(coords)
    
    # Calculate ΔE between consecutive colors
    delta_e_values = [euclidean(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
    
    # Analysis of results
    average_delta_e = np.mean(delta_e_values)
    std_dev_delta_e = np.std(delta_e_values)
    
    return {
        "delta_e_values": delta_e_values,
        "average_delta_e": average_delta_e,
        "std_dev_delta_e": std_dev_delta_e,
        "hex_colors": hex_colors[1:]  # Skip first color for alignment
    }

def plot_color_gradient(colors, ax):
    """Plot color gradient."""
    cmap = mcolors.LinearSegmentedColormap.from_list("palette", colors)
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_axis_off()

def create_analysis_plot(palette_results, palette_names, y_max=3):
    """Create analysis plots for multiple palettes in two columns with consistent y-axis."""
    num_palettes = len(palette_results)
    num_cols = 2
    num_rows = (num_palettes + 1) // 2  # Ceiling division
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 3*num_rows))
    
    # Handle single palette case
    if num_palettes == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for idx, (result, name) in enumerate(zip(palette_results, palette_names)):
        ax = axes_flat[idx]
        
        # Plot line
        x = list(range(len(result['delta_e_values'])))
        ax.plot(x, result['delta_e_values'], linewidth=1.5)
        
        # Set consistent y-axis
        ax.set_ylim(0, y_max)
        ax.set_ylabel('Delta E', fontsize=10)
        ax.set_title(f'{name}\n(Avg ΔE: {result["average_delta_e"]:.3f}, Std Dev: {result["std_dev_delta_e"]:.3f})', 
                     fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        
        # Create color gradient above the plot
        divider_height = 0.15
        gradient_ax = ax.inset_axes([0, 1.05, 1, divider_height], transform=ax.transAxes)
        plot_color_gradient(result['hex_colors'], gradient_ax)
    
    # Hide empty subplots if odd number of palettes
    for idx in range(num_palettes, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

# Initialize session state
if 'palettes' not in st.session_state:
    st.session_state.palettes = {}
if 'next_id' not in st.session_state:
    st.session_state.next_id = 1

# Title and description
st.title("🎨 Color Scale Perceptual Uniformity Analyzer")
st.markdown("""
This tool analyzes color scales for perceptual uniformity using the CIE Lab or CAM02-UCS color space.
A perceptually uniform color scale should have relatively constant ΔE values between consecutive colors.
""")

# Sidebar for settings
st.sidebar.header("Settings")
num_colors = st.sidebar.slider("Number of interpolated colors", 10, 500, 255, 5)
color_space = st.sidebar.selectbox("Color Space", ["CIE Lab", "CAM02-UCS"])

# Examples section
st.sidebar.header("📚 Load Examples")
col1, col2 = st.sidebar.columns(2)

if col1.button("Load Viridis (Uniform)", use_container_width=True):
    viridis_colors = ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', 
                      '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725']
    st.session_state.palettes[f"Viridis_{st.session_state.next_id}"] = viridis_colors
    st.session_state.next_id += 1
    st.rerun()

if col2.button("Load Jet (Non-uniform)", use_container_width=True):
    jet_colors = ['#000080', '#0000ff', '#0080ff', '#00ffff', '#80ff80', 
                  '#ffff00', '#ff8000', '#ff0000', '#800000']
    st.session_state.palettes[f"Jet_{st.session_state.next_id}"] = jet_colors
    st.session_state.next_id += 1
    st.rerun()

# Main content
st.header("Add Color Palettes")

# Form to add new palette
with st.form("add_palette_form", clear_on_submit=True):
    palette_name = st.text_input("Palette Name", placeholder="My Custom Palette")
    
    st.write("Enter at least 5 hex colors (one per line):")
    colors_input = st.text_area(
        "Colors (hex codes)",
        height=150,
        placeholder="#7bb5c4\n#9fc1ad\n#d3d3e0\n#8d9bff\n#ff9750\n#ffd900",
        help="Enter hex colors, one per line (e.g., #FF0000)"
    )
    
    submitted = st.form_submit_button("Add Palette", use_container_width=True)
    
    if submitted:
        if not palette_name:
            st.error("Please provide a palette name")
        else:
            colors = [c.strip() for c in colors_input.split('\n') if c.strip()]
            
            # Validate hex colors
            valid_colors = []
            for color in colors:
                if not color.startswith('#'):
                    color = '#' + color
                try:
                    mcolors.to_rgb(color)
                    valid_colors.append(color.upper())
                except ValueError:
                    st.error(f"Invalid hex color: {color}")
                    valid_colors = []
                    break
            
            if len(valid_colors) < 5:
                st.error("Please provide at least 5 valid hex colors")
            elif valid_colors:
                st.session_state.palettes[palette_name] = valid_colors
                st.success(f"Added palette: {palette_name}")
                st.rerun()

# Display current palettes
if st.session_state.palettes:
    st.header("Current Palettes")
    
    # Display palette cards
    for palette_name, colors in list(st.session_state.palettes.items()):
        with st.expander(f"🎨 {palette_name} ({len(colors)} colors)", expanded=False):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Create color preview
                fig_preview, ax = plt.subplots(figsize=(8, 0.5))
                plot_color_gradient(colors, ax)
                st.pyplot(fig_preview)
                plt.close()
                
                # Show colors
                st.write("Colors:", ", ".join(colors))
            
            with col2:
                if st.button("🗑️ Remove", key=f"remove_{palette_name}", use_container_width=True):
                    del st.session_state.palettes[palette_name]
                    st.rerun()
    
    # Analysis section
    st.header("📊 Analysis")
    
    if st.button("🔬 Analyze All Palettes", type="primary", use_container_width=True):
        with st.spinner("Analyzing palettes..."):
            # Analyze all palettes
            results = []
            names = []
            
            # Calculate maximum y value across all palettes
            max_delta_e = 0
            
            for name, colors in st.session_state.palettes.items():
                interpolated = interpolate_colors(colors, num_colors)
                result = perceptual_uniformity_test(interpolated, color_space)
                results.append(result)
                names.append(name)
                
                # Update max delta E
                current_max = max(result['delta_e_values'])
                if current_max > max_delta_e:
                    max_delta_e = current_max
            
            # Determine y-axis range
            y_max = max(3.0, max_delta_e * 1.1)  # At least 3, or 110% of max value
            
            # Display statistics
            st.subheader("Statistics Summary")
            stats_data = []
            for name, result in zip(names, results):
                stats_data.append({
                    "Palette": name,
                    "Average ΔE": f"{result['average_delta_e']:.4f}",
                    "Std Deviation": f"{result['std_dev_delta_e']:.4f}",
                    "Min ΔE": f"{min(result['delta_e_values']):.4f}",
                    "Max ΔE": f"{max(result['delta_e_values']):.4f}"
                })
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
            
            # Create and display plots
            st.subheader("Perceptual Uniformity Analysis")
            st.write(f"**Color Space:** {color_space} | **Y-axis range:** 0 to {y_max:.2f}")
            
            fig = create_analysis_plot(results, names, y_max)
            st.pyplot(fig)
            plt.close()
            
            # Interpretation
            st.info("""
            **💡 Interpretation:**
            - **Lower average ΔE** with **lower standard deviation** indicates better perceptual uniformity
            - A flat line means the color transitions appear equally spaced to the human eye
            - Peaks indicate regions where color changes appear more dramatic
            - Valleys indicate regions where colors appear more similar
            """)

else:
    st.info("👆 Add your first palette above or load an example from the sidebar!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Based on perceptual uniformity testing using CIE Lab and CAM02-UCS color spaces</p>
    <p>Lower standard deviation indicates better perceptual uniformity</p>
</div>
""", unsafe_allow_html=True)