import io
import re
import base64
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import euclidean
import colorspacious as cs
from PIL import Image

st.set_page_config(page_title="Color Scale Perceptual Uniformity Analyzer", layout="wide")

# Language dictionaries
TRANSLATIONS = {
    'en': {
        'title': 'Color Scale Perceptual Uniformity Analyzer',
        'description': 'This tool analyzes color scales for perceptual uniformity using the CIE Lab, CAM02-UCS or OKLab color space.\nA perceptually uniform color scale should have relatively constant ΔE values between consecutive colors.',
        'intro_text': 'There are many relevant decisions when building a data visualization. One of them, choosing a scale to map values to colors, is not trivial at all. Since scales function as a kind of "interface" between data and our brain, it is important to select a scale that does not unnecessarily distort that perception. This property is called "perceptual uniformity". [This tool](https://color-analyzer.streamlit.app/) analyzes color scales for perceptual uniformity using the CIE Lab, CAM02-UCS or OKLab color space. A perceptually uniform color scale should have relatively constant ΔE values between consecutive colors.',
        'created_by': 'Created by',
        'settings': 'Settings',
        'num_colors': 'Number of interpolated colors',
        'color_space': 'Color Space',
        'load_examples': '📚 Load Examples',
        'load_viridis': 'Load Viridis (Uniform)',
        'load_jet': 'Load Jet (Non-uniform)',
        'add_palettes': 'Add Color Scales',
        'palette_name': 'Scale Name',
        'palette_name_placeholder': 'My Custom Scale',
        'colors_label': 'Paste hex codes, a table or any text (the #RRGGBB codes are extracted automatically):',
        'colors_textarea_label': 'Colors (hex codes)',
        'colors_placeholder': '#7bb5c4\n#9fc1ad\n#d3d3e0\n#8d9bff\n#ff9750\n#ffd900',
        'colors_help': 'You can paste a table or free text; hex codes (#RGB or #RRGGBB) are detected automatically.',
        'or_upload_image': 'Or upload an image of a color scale:',
        'upload_image': 'Color scale image',
        'upload_image_help': 'The central strip of the image is sampled to extract the scale.',
        'add_palette_button': 'Add Scale',
        'error_name': 'Please provide a scale name',
        'error_invalid_color': 'Invalid hex color',
        'error_min_colors': 'Please provide at least 5 valid hex colors',
        'error_no_hex': 'No hex codes were found in the text',
        'error_image': 'Could not read the image',
        'success_added': 'Added scale',
        'current_palettes': 'Current Scales',
        'colors_count': 'colors',
        'source_image_tag': 'from image',
        'remove_button': '🗑️ Remove scale',
        'download_preview': 'Download preview (PNG)',
        'analysis_title': '📊 Analysis',
        'analyze_button': '🔬 Analyze All Scales',
        'stats_summary': 'Statistics Summary',
        'palette_col': 'Scale',
        'avg_delta': 'Average ΔE',
        'std_dev': 'Std Deviation',
        'min_delta': 'Min ΔE',
        'max_delta': 'Max ΔE',
        'uniformity_analysis': 'Perceptual Uniformity Analysis',
        'color_space_label': 'Color Space',
        'y_axis_range': 'Y-axis range',
        'export_report': '📄 Export report',
        'download_chart': 'Download chart (PNG)',
        'download_html': 'Download report (HTML)',
        'download_pdf': 'Download report (PDF)',
        'report_title': 'Color Scale Perceptual Uniformity Report',
        'generated_on': 'Generated on',
        'interpretation_title': '💡 Interpretation:',
        'interpretation_1': '**Lower average ΔE** with **lower standard deviation** indicates better perceptual uniformity',
        'interpretation_2': 'A flat line means the color transitions appear equally spaced to the human eye',
        'interpretation_3': 'Peaks indicate regions where color changes appear more dramatic',
        'interpretation_4': 'Valleys indicate regions where colors appear more similar',
        'footer_1': 'Based on perceptual uniformity testing using CIE Lab, CAM02-UCS and OKLab color spaces',
        'footer_2': 'Lower standard deviation indicates better perceptual uniformity',
        'add_first_palette': '👆 Add your first scale above or load an example from the sidebar!',
        'language': 'Language',
    },
    'es': {
        'title': 'Analizador de Uniformidad Perceptual de Escalas de Color',
        'description': 'Esta herramienta analiza escalas de color en términos de uniformidad perceptual usando el espacio de color CIE Lab, CAM02-UCS u OKLab.\nUna escala de color perceptualmente uniforme debe tener valores ΔE relativamente constantes entre colores consecutivos.',
        'intro_text': 'Hay muchas decisiones relevantes a la hora de construir una visualización de datos. Una de ellas, la elección de una escala para mapear valores a colores, no es para nada trivial. Dado que las escalas funcionan como una especie de "interface" entre los datos y nuestro cerebro, es importante seleccionar una escala que no distorsione innecesariamente dicha percepción. A esta propiedad se la llama "uniformidad perceptual". [Esta herramienta](https://color-analyzer.streamlit.app/) analiza las escalas de color en busca de uniformidad perceptual utilizando el espacio de color CIE Lab, CAM02-UCS u OKLab. Una escala de color perceptualmente uniforme debe tener valores ΔE relativamente constantes entre colores consecutivos.',
        'created_by': 'Creado por',
        'settings': 'Configuración',
        'num_colors': 'Número de colores interpolados',
        'color_space': 'Espacio de Color',
        'load_examples': '📚 Cargar Ejemplos',
        'load_viridis': 'Cargar Viridis (Uniforme)',
        'load_jet': 'Cargar Jet (No uniforme)',
        'add_palettes': 'Agregar Escalas de Color',
        'palette_name': 'Nombre de la Escala',
        'palette_name_placeholder': 'Mi Escala Personalizada',
        'colors_label': 'Pegá códigos hex, una tabla o cualquier texto (se extraen automáticamente los códigos #RRGGBB):',
        'colors_textarea_label': 'Colores (códigos hex)',
        'colors_placeholder': '#7bb5c4\n#9fc1ad\n#d3d3e0\n#8d9bff\n#ff9750\n#ffd900',
        'colors_help': 'Podés pegar una tabla o texto libre; los códigos hex (#RGB o #RRGGBB) se detectan automáticamente.',
        'or_upload_image': 'O bien, subí una imagen de una escala de color:',
        'upload_image': 'Imagen de la escala de color',
        'upload_image_help': 'Se muestrea la franja central de la imagen para extraer la escala.',
        'add_palette_button': 'Agregar Escala',
        'error_name': 'Por favor proporcione un nombre para la escala',
        'error_invalid_color': 'Color hex inválido',
        'error_min_colors': 'Por favor proporcione al menos 5 colores hex válidos',
        'error_no_hex': 'No se encontraron códigos hex en el texto',
        'error_image': 'No se pudo leer la imagen',
        'success_added': 'Escala agregada',
        'current_palettes': 'Escalas Actuales',
        'colors_count': 'colores',
        'source_image_tag': 'desde imagen',
        'remove_button': '🗑️ Eliminar escala',
        'download_preview': 'Descargar preview (PNG)',
        'analysis_title': '📊 Análisis',
        'analyze_button': '🔬 Analizar Todas las Escalas',
        'stats_summary': 'Resumen Estadístico',
        'palette_col': 'Escala',
        'avg_delta': 'ΔE Promedio',
        'std_dev': 'Desviación Estándar',
        'min_delta': 'ΔE Mínimo',
        'max_delta': 'ΔE Máximo',
        'uniformity_analysis': 'Análisis de Uniformidad Perceptual',
        'color_space_label': 'Espacio de Color',
        'y_axis_range': 'Rango del eje Y',
        'export_report': '📄 Exportar reporte',
        'download_chart': 'Descargar gráfico (PNG)',
        'download_html': 'Descargar reporte (HTML)',
        'download_pdf': 'Descargar reporte (PDF)',
        'report_title': 'Reporte de Uniformidad Perceptual de Escalas de Color',
        'generated_on': 'Generado el',
        'interpretation_title': '💡 Interpretación:',
        'interpretation_1': '**ΔE promedio más bajo** con **desviación estándar más baja** indica mejor uniformidad perceptual',
        'interpretation_2': 'Una línea plana significa que las transiciones de color parecen igualmente espaciadas para el ojo humano',
        'interpretation_3': 'Los picos indican regiones donde los cambios de color parecen más dramáticos',
        'interpretation_4': 'Los valles indican regiones donde los colores parecen más similares',
        'footer_1': 'Basado en pruebas de uniformidad perceptual usando espacios de color CIE Lab, CAM02-UCS y OKLab',
        'footer_2': 'Una desviación estándar más baja indica mejor uniformidad perceptual',
        'add_first_palette': '👆 ¡Agregue su primera escala arriba o cargue un ejemplo desde la barra lateral!',
        'language': 'Idioma',
    }
}

# Initialize session state for language
if 'language' not in st.session_state:
    st.session_state.language = 'es'

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

_HEX_RE = re.compile(r'#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b')

def extract_hex_colors(text):
    """Extract hex color codes from arbitrary pasted text (a table, a list, JSON, etc.).

    Only tokens starting with '#' are matched, to avoid false positives from
    numeric table cells. Short form (#abc) is expanded to #aabbcc.
    """
    colors = []
    for match in _HEX_RE.findall(text or ''):
        if len(match) == 3:
            match = ''.join(c * 2 for c in match)
        colors.append('#' + match.upper())
    return colors

def _srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def srgb_to_oklab(rgb):
    """Convert an sRGB color (0-1 range) to OKLab coordinates."""
    r, g, b = _srgb_to_linear(np.asarray(rgb, dtype=float))
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.array([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ])

def extract_scale_from_image(file, n_samples=16):
    """Extract a color scale from an uploaded image by sampling its central strip."""
    img = Image.open(file).convert("RGB")
    arr = np.asarray(img)
    height, width, _ = arr.shape
    line = arr[height // 2, :, :] if width >= height else arr[:, width // 2, :]
    idx = np.linspace(0, len(line) - 1, n_samples).round().astype(int)
    return [mcolors.to_hex(line[i] / 255.0).upper() for i in idx]

def perceptual_uniformity_test(hex_colors, color_space="CIE Lab"):
    """
    Perform quantitative perceptual uniformity test.

    Args:
        hex_colors (list): List of colors in HEX format.
        color_space (str): Color space to use: "CIE Lab", "CAM02-UCS" or "OKLab".

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
    elif color_space == "OKLab":
        coords = [srgb_to_oklab(rgb) for rgb in rgb_colors]
    else:
        raise ValueError("Color space not supported. Use 'CIE Lab', 'CAM02-UCS' or 'OKLab'.")

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

def fig_to_png_bytes(fig, dpi=150):
    """Render a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def gradient_png_bytes(colors):
    """Render a color scale gradient to PNG bytes."""
    fig, ax = plt.subplots(figsize=(8, 0.6))
    data = None
    try:
        plot_color_gradient(colors, ax)
        data = fig_to_png_bytes(fig, dpi=100)
    finally:
        plt.close(fig)
    return data

def build_html_report(stats_df, analysis_png, previews, color_space):
    """Build a self-contained HTML report with the stats table and charts embedded."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    chart_b64 = base64.b64encode(analysis_png).decode()
    scale_blocks = ""
    for name, png in previews.items():
        b64 = base64.b64encode(png).decode()
        scale_blocks += (
            f'<div class="scale"><h3>{name}</h3>'
            f'<img src="data:image/png;base64,{b64}"></div>'
        )
    return f"""<!doctype html>
<html lang="{st.session_state.language}">
<head>
<meta charset="utf-8">
<title>{t('report_title')}</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: right; }}
  th {{ background: #f0f0f0; }}
  img {{ max-width: 100%; border: 1px solid #eee; }}
  .scale {{ margin: 1rem 0; }}
  .meta {{ color: #555; }}
</style>
</head>
<body>
<h1>{t('report_title')}</h1>
<p class="meta">{t('generated_on')}: {now}<br>{t('color_space_label')}: {color_space}</p>
<h2>{t('stats_summary')}</h2>
{stats_df.to_html(index=False)}
<h2>{t('uniformity_analysis')}</h2>
<img src="data:image/png;base64,{chart_b64}">
<h2>{t('current_palettes')}</h2>
{scale_blocks}
</body>
</html>"""

def build_pdf_report(stats_df, fig, color_space):
    """Build a PDF report: page 1 the stats table, page 2 the analysis chart."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        table_fig, ax = plt.subplots(figsize=(11, max(2.5, 0.5 * len(stats_df) + 1.5)))
        ax.axis("off")
        ax.set_title(
            f"{t('report_title')}\n"
            f"{t('generated_on')}: {datetime.now():%Y-%m-%d %H:%M}   |   "
            f"{t('color_space_label')}: {color_space}",
            fontsize=11, fontweight="bold", pad=20,
        )
        table = ax.table(
            cellText=stats_df.values,
            colLabels=list(stats_df.columns),
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
        pdf.savefig(table_fig, bbox_inches="tight")
        plt.close(table_fig)

        pdf.savefig(fig, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

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

def add_scale(name, colors, source):
    """Register a color scale and invalidate any stale analysis."""
    st.session_state.palettes[name] = {"colors": list(colors), "source": source}
    st.session_state.pop('analysis', None)

def remove_scale(name):
    st.session_state.palettes.pop(name, None)
    st.session_state.pop('analysis', None)

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
    st.markdown("---")
    st.markdown(t('intro_text'))
    st.markdown(f"**{t('created_by')}:** [Germán Rosati](https://gefero.github.io)")
with col2:
    try:
        st.image("logo.png", width=200)
    except Exception:
        pass  # Logo not found, continue without it

# Sidebar for settings
st.sidebar.header(t('settings'))
num_colors = st.sidebar.slider(t('num_colors'), 10, 500, 255, 5)
color_space = st.sidebar.selectbox(t('color_space'), ["CIE Lab", "CAM02-UCS", "OKLab"])

# Examples section
st.sidebar.header(t('load_examples'))
col1, col2 = st.sidebar.columns(2)

if col1.button(t('load_viridis'), use_container_width=True):
    viridis_colors = ['#440154', '#482878', '#3e4989', '#31688e', '#26828e',
                      '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725']
    add_scale(f"Viridis_{st.session_state.next_id}", viridis_colors, "hex")
    st.session_state.next_id += 1
    st.rerun()

if col2.button(t('load_jet'), use_container_width=True):
    # More samples from matplotlib's Jet colormap to better show perceptual non-uniformity
    # Jet is notorious for having uneven perceptual steps, especially in the green-yellow region
    jet_colors = ['#00007F', '#0000C7', '#0000FF', '#0047FF', '#008FFF',
                  '#00D7FF', '#00FFFF', '#2BFFD5', '#56FFAA', '#80FF80',
                  '#AAFF56', '#D5FF2B', '#FFFF00', '#FFD700', '#FFAF00',
                  '#FF8700', '#FF5F00', '#FF0000', '#C70000', '#7F0000']
    add_scale(f"Jet_{st.session_state.next_id}", jet_colors, "hex")
    st.session_state.next_id += 1
    st.rerun()

# Main content
st.header(t('add_palettes'))

# Form to add new scale (from pasted text/table or from an image)
with st.form("add_palette_form", clear_on_submit=True):
    palette_name = st.text_input(t('palette_name'), placeholder=t('palette_name_placeholder'))

    st.write(t('colors_label'))
    colors_input = st.text_area(
        t('colors_textarea_label'),
        height=150,
        placeholder=t('colors_placeholder'),
        help=t('colors_help')
    )

    st.write(t('or_upload_image'))
    image_file = st.file_uploader(
        t('upload_image'),
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        help=t('upload_image_help'),
    )

    submitted = st.form_submit_button(t('add_palette_button'), use_container_width=True)

    if submitted:
        if not palette_name:
            st.error(t('error_name'))
        elif image_file is not None:
            try:
                colors = extract_scale_from_image(image_file)
            except Exception:
                colors = []
                st.error(t('error_image'))
            if len(colors) >= 5:
                add_scale(palette_name, colors, "image")
                st.success(f"{t('success_added')}: {palette_name}")
                st.rerun()
            elif colors:
                st.error(t('error_min_colors'))
        else:
            found = extract_hex_colors(colors_input)
            valid_colors = []
            for color in found:
                try:
                    mcolors.to_rgb(color)
                    valid_colors.append(color.upper())
                except ValueError:
                    continue

            if not valid_colors:
                st.error(t('error_no_hex'))
            elif len(valid_colors) < 5:
                st.error(t('error_min_colors'))
            else:
                add_scale(palette_name, valid_colors, "hex")
                st.success(f"{t('success_added')}: {palette_name}")
                st.rerun()

# Display current scales
if st.session_state.palettes:
    st.header(t('current_palettes'))

    # Display scale cards (no expander: everything visible by default)
    for palette_name, meta in list(st.session_state.palettes.items()):
        colors = meta["colors"]
        source = meta.get("source", "hex")

        with st.container(border=True):
            tag = f" · _{t('source_image_tag')}_" if source == "image" else ""
            st.markdown(f"**{palette_name}** &nbsp;({len(colors)} {t('colors_count')}){tag}")

            # Color preview
            fig_preview, ax = plt.subplots(figsize=(8, 0.5))
            plot_color_gradient(colors, ax)
            st.pyplot(fig_preview)
            plt.close(fig_preview)

            # Show colors
            st.write(f"{t('colors_count').capitalize()}:", ", ".join(colors))

            preview_png = gradient_png_bytes(colors)
            if preview_png:
                st.download_button(
                    t('download_preview'),
                    data=preview_png,
                    file_name=f"{palette_name}_preview.png",
                    mime="image/png",
                    key=f"dl_prev_{palette_name}",
                )

        # "Remove scale" button lives outside the scale card
        _, remove_col = st.columns([4, 1])
        if remove_col.button(t('remove_button'), key=f"remove_{palette_name}", use_container_width=True):
            remove_scale(palette_name)
            st.rerun()

    # Analysis section
    st.header(t('analysis_title'))

    if st.button(t('analyze_button'), type="primary", use_container_width=True):
        with st.spinner("Analyzing scales..."):
            results = []
            names = []
            max_delta_e = 0

            for name, meta in st.session_state.palettes.items():
                interpolated = interpolate_colors(meta["colors"], num_colors)
                result = perceptual_uniformity_test(interpolated, color_space)
                results.append(result)
                names.append(name)

                current_max = max(result['delta_e_values'])
                if current_max > max_delta_e:
                    max_delta_e = current_max

            # Determine y-axis range (OKLab ΔE values live on a much smaller scale)
            if color_space == "OKLab":
                y_max = max(max_delta_e * 1.15, 0.01)
            else:
                y_max = max(3.0, max_delta_e * 1.1)

            st.session_state.analysis = {
                'results': results,
                'names': names,
                'y_max': y_max,
                'color_space': color_space,
            }

    analysis = st.session_state.get('analysis')
    if analysis:
        results = analysis['results']
        names = analysis['names']
        y_max = analysis['y_max']
        analysis_color_space = analysis['color_space']

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
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Create and display plots
        st.subheader(t('uniformity_analysis'))
        st.write(f"**{t('color_space_label')}:** {analysis_color_space} | **{t('y_axis_range')}:** 0 to {y_max:.2f}")

        fig = create_analysis_plot(results, names, y_max)
        st.pyplot(fig)

        # Exports: chart image + full report (HTML and PDF)
        analysis_png = fig_to_png_bytes(fig)
        previews = {
            name: gradient_png_bytes(st.session_state.palettes[name]["colors"])
            for name in names
            if name in st.session_state.palettes
        }

        st.subheader(t('export_report'))
        dc1, dc2, dc3 = st.columns(3)
        dc1.download_button(
            t('download_chart'),
            data=analysis_png,
            file_name="analisis_uniformidad.png",
            mime="image/png",
            use_container_width=True,
        )
        dc2.download_button(
            t('download_html'),
            data=build_html_report(stats_df, analysis_png, previews, analysis_color_space).encode('utf-8'),
            file_name="reporte_uniformidad.html",
            mime="text/html",
            use_container_width=True,
        )
        dc3.download_button(
            t('download_pdf'),
            data=build_pdf_report(stats_df, fig, analysis_color_space),
            file_name="reporte_uniformidad.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        plt.close(fig)

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
