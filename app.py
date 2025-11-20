import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import euclidean
import colorspacious as cs

st.set_page_config(page_title="Color Scale Perceptual Uniformity Analyzer", layout="wide")

# Language dictionaries
TRANSLATIONS = {
    'en': {
        'title': '🎨 Color Scale Perceptual Uniformity Analyzer',
        'description': 'This tool analyzes color scales for perceptual uniformity using the CIE Lab or CAM02-UCS color space.\nA perceptually uniform color scale should have relatively constant ΔE values between consecutive colors.',
        'settings': 'Settings',
        'num_colors': 'Number of interpolated colors',
        'color_space': 'Color Space',
        'load_examples': '📚 Load Examples',
        'load_viridis': 'Load Viridis (Uniform)',
        'load_jet': 'Load Jet (Non-uniform)',
        'add_palettes': 'Add Color Palettes',
        'palette_name': 'Palette Name',
        'palette_name_placeholder': 'My Custom Palette',
        'colors_label': 'Enter at least 5 hex colors (one per line):',
        'colors_placeholder': '#7bb5c4\n#9fc1ad\n#d3d3e0\n#8d9bff\n#ff9750\n#ffd900',
        'colors_help': 'Enter hex colors, one per line (e.g., #FF0000)',
        'add_palette_button': 'Add Palette',
        'error_name': 'Please provide a palette name',
        'error_invalid_color': 'Invalid hex color',
        'error_min_colors': 'Please provide at least 5 valid hex colors',
        'success_added': 'Added palette',
        'current_palettes': 'Current Palettes',
        'colors_count': 'colors',
        'remove_button': '🗑️ Remove',
        'analysis_title': '📊 Analysis',
        'analyze_button': '🔬 Analyze All Palettes',
        'stats_summary': 'Statistics Summary',
        'palette_col': 'Palette',
        'avg_delta': 'Average ΔE',
        'std_dev': 'Std Deviation',
        'min_delta': 'Min ΔE',
        'max_delta': 'Max ΔE',
        'uniformity_analysis': 'Perceptual Uniformity Analysis',
        'color_space_label': 'Color Space',
        'y_axis_range': 'Y-axis range',
        'interpretation_title': '💡 Interpretation:',
        'interpretation_1': '**Lower average ΔE** with **lower standard deviation** indicates better perceptual uniformity',
        'interpretation_2': 'A flat line means the color transitions appear equally spaced to the human eye',
        'interpretation_3': 'Peaks indicate regions where color changes appear more dramatic',
        'interpretation_4': 'Valleys indicate regions where colors appear more similar',
        'footer_1': 'Based on perceptual uniformity testing using CIE Lab and CAM02-UCS color spaces',
        'footer_2': 'Lower standard deviation indicates better perceptual uniformity',
        'add_first_palette': '👆 Add your first palette above or load an example from the sidebar!',
        'language': 'Language',
    },
    'es': {
        'title': '🎨 Analizador de Uniformidad Perceptual de Escalas de Color',
        'description': 'Esta herramienta analiza escalas de color en términos de uniformidad perceptual usando el espacio de color CIE Lab o CAM02-UCS.\nUna escala de color perceptualmente uniforme debe tener valores ΔE relativamente constantes entre colores consecutivos.',
        'settings': 'Configuración',
        'num_colors': 'Número de colores interpolados',
        'color_space': 'Espacio de Color',
        'load_examples': '📚 Cargar Ejemplos',
        'load_viridis': 'Cargar Viridis (Uniforme)',
        'load_jet': 'Cargar Jet (No uniforme)',
        'add_palettes': 'Agregar Paletas de Color',
        'palette_name': 'Nombre de la Paleta',
        'palette_name_placeholder': 'Mi Paleta Personalizada',
        'colors_label': 'Ingrese al menos 5 colores hex (uno por línea):',
        'colors_placeholder': '#7bb5c4\n#9fc1ad\n#d3d3e0\n#8d9bff\n#ff9750\n#ffd900',
        'colors_help': 'Ingrese colores hex, uno por línea (ej., #FF0000)',
        'add_palette_button': 'Agregar Paleta',
        'error_name': 'Por favor proporcione un nombre para la paleta',
        'error_invalid_color': 'Color hex inválido',
        'error_min_colors': 'Por favor proporcione al menos 5 colores hex válidos',
        'success_added': 'Paleta agregada',
        'current_palettes': 'Paletas Actuales',
        'colors_count': 'colores',
        'remove_button': '🗑️ Eliminar',
        'analysis_title': '📊 Análisis',
        'analyze_button': '🔬 Analizar Todas las Paletas',
        'stats_summary': 'Resumen Estadístico',
        'palette_col': 'Paleta',
        'avg_delta': 'ΔE Promedio',
        'std_dev': 'Desviación Estándar',
        'min_delta': 'ΔE Mínimo',
        'max_delta': 'ΔE Máximo',
        'uniformity_analysis': 'Análisis de Uniformidad Perceptual',
        'color_space_label': 'Espacio de Color',
        'y_axis_range': 'Rango del eje Y',
        'interpretation_title': '💡 Interpretación:',
        'interpretation_1': '**ΔE promedio más bajo** con **desviación estándar más baja** indica mejor uniformidad perceptual',
        'interpretation_2': 'Una línea plana significa que las transiciones de color parecen igualmente espaciadas para el ojo humano',
        'interpretation_3': 'Los picos indican regiones donde los cambios de color parecen más dramáticos',
        'interpretation_4': 'Los valles indican regiones donde los colores parecen más similares',
        'footer_1': 'Basado en pruebas de uniformidad perceptual usando espacios de color CIE Lab y CAM02-UCS',
        'footer_2': 'Una desviación estándar más baja indica mejor uniformidad perceptual',
        'add_first_palette': '👆 ¡Agregue su primera paleta arriba o cargue un ejemplo desde la barra lateral!',
        'language': 'Idioma',
    }
}

# Initialize session state for language
if 'language' not in st.session_state:
    st.session_state.language = 'en'

def t(key):
    """Translation helper function"""
    return TRANSLATIONS[st.session_state.language].get(key, key)

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

# Language selector in sidebar
st.sidebar.selectbox(
    t('language'),
    options=['English', 'Español'],
    index=0 if st.session_state.language == 'en' else 1,
    key='lang_selector',
    on_change=lambda: setattr(st.session_state, 'language', 'en' if st.session_state.lang_selector == 'English' else 'es')
)

st.sidebar.markdown("---")

# Title and description
col1, col2 = st.columns([3, 1])
with col1:
    st.title(t('title'))
    st.markdown(t('description'))
with col2:
    try:
        st.image("logo.png", width=200)
    except:
        pass  # Logo not found, continue without it

# Sidebar for settings
st.sidebar.header(t('settings'))
num_colors = st.sidebar.slider(t('num_colors'), 10, 500, 255, 5)
color_space = st.sidebar.selectbox(t('color_space'), ["CIE Lab", "CAM02-UCS"])

# Examples section
st.sidebar.header(t('load_examples'))
col1, col2 = st.sidebar.columns(2)

if col1.button(t('load_viridis'), use_container_width=True):
    viridis_colors = ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', 
                      '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725']
    st.session_state.palettes[f"Viridis_{st.session_state.next_id}"] = viridis_colors
    st.session_state.next_id += 1
    st.rerun()

if col2.button(t('load_jet'), use_container_width=True):
    jet_colors = ['#000080', '#0000ff', '#0080ff', '#00ffff', '#80ff80', 
                  '#ffff00', '#ff8000', '#ff0000', '#800000']
    st.session_state.palettes[f"Jet_{st.session_state.next_id}"] = jet_colors
    st.session_state.next_id += 1
    st.rerun()

# Main content
st.header(t('add_palettes'))

# Form to add new palette
with st.form("add_palette_form", clear_on_submit=True):
    palette_name = st.text_input(t('palette_name'), placeholder=t('palette_name_placeholder'))
    
    st.write(t('colors_label'))
    colors_input = st.text_area(
        "Colors (hex codes)",
        height=150,
        placeholder=t('colors_placeholder'),
        help=t('colors_help')
    )
    
    submitted = st.form_submit_button(t('add_palette_button'), use_container_width=True)
    
    if submitted:
        if not palette_name:
            st.error(t('error_name'))
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
                    st.error(f"{t('error_invalid_color')}: {color}")
                    valid_colors = []
                    break
            
            if len(valid_colors) < 5:
                st.error(t('error_min_colors'))
            elif valid_colors:
                st.session_state.palettes[palette_name] = valid_colors
                st.success(f"{t('success_added')}: {palette_name}")
                st.rerun()

# Display current palettes
if st.session_state.palettes:
    st.header(t('current_palettes'))
    
    # Display palette cards
    for palette_name, colors in list(st.session_state.palettes.items()):
        with st.expander(f"🎨 {palette_name} ({len(colors)} {t('colors_count')})", expanded=False):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Create color preview
                fig_preview, ax = plt.subplots(figsize=(8, 0.5))
                plot_color_gradient(colors, ax)
                st.pyplot(fig_preview)
                plt.close()
                
                # Show colors
                st.write(f"{t('colors_count').capitalize()}:", ", ".join(colors))
            
            with col2:
                if st.button(t('remove_button'), key=f"remove_{palette_name}", use_container_width=True):
                    del st.session_state.palettes[palette_name]
                    st.rerun()
    
    # Analysis section
    st.header(t('analysis_title'))
    
    if st.button(t('analyze_button'), type="primary", use_container_width=True):
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
            st.subheader(t('stats_summary'))
            stats_data = []
            for name, result in zip(names, results):
                stats_data.append({
                    t('palette_col'): name,
                    t('avg_delta'): f"{result['average_delta_e']:.4f}",
                    t('std_dev'): f"{result['std_dev_delta_e']:.4f}",
                    t('min_delta'): f"{min(result['delta_e_values']):.4f}",
                    t('max_delta'): f"{max(result['delta_e_values']):.4f}"
                })
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
            
            # Create and display plots
            st.subheader(t('uniformity_analysis'))
            st.write(f"**{t('color_space_label')}:** {color_space} | **{t('y_axis_range')}:** 0 to {y_max:.2f}")
            
            fig = create_analysis_plot(results, names, y_max)
            st.pyplot(fig)
            plt.close()
            
            # Interpretation
            st.info(f"""
            {t('interpretation_title')}
            - {t('interpretation_1')}
            - {t('interpretation_2')}
            - {t('interpretation_3')}
            - {t('interpretation_4')}
            """)

else:
    st.info(t('add_first_palette'))

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>{t('footer_1')}</p>
    <p>{t('footer_2')}</p>
</div>
""", unsafe_allow_html=True)