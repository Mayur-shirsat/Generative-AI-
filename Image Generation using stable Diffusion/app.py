# ============================================================
# app.py — Stable Diffusion 4K Image Generator with Custom Dark UI
# ============================================================

# ---------- Import Required Libraries ----------
import os
import io
import base64
import traceback
import torch
import streamlit as st
from PIL import Image

# ============================================================
# Monkey Patch for CLIPTextModel Issue
# ============================================================

def patch_clip_textmodel_init():
    """
    Monkey-patch CLIPTextModel.__init__ to safely remove the 'offload_state_dict' argument
    if it causes a TypeError in newer/older transformer versions.
    """
    try:
        from transformers.models.clip.modeling_clip import CLIPTextModel as _Orig
    except ImportError:
        return

    orig_init = _Orig.__init__

    def new_init(self, *args, **kwargs):
        if "offload_state_dict" in kwargs:
            kwargs.pop("offload_state_dict")
        return orig_init(self, *args, **kwargs)

    _Orig.__init__ = new_init

# Apply patch before importing diffusers pipelines
patch_clip_textmodel_init()

# ============================================================
# Import Stable Diffusion Pipelines
# ============================================================

from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline

# ============================================================
# Hugging Face Cache and Device Setup
# ============================================================

os.environ["HF_HOME"] = r"E:\python3.13"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# ============================================================
# Model Loading (with Streamlit Cache)
# ============================================================

@st.cache_resource
def load_sd_model():
    """Load Stable Diffusion v1.5 model."""
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            safety_checker=None,
            cache_dir=r"E:\\python3.12\\python 3.12"
        )
        pipe = pipe.to(device)
        return pipe
    except Exception as e:
        st.error("Error loading Stable Diffusion model:\n" + "".join(traceback.format_exception_only(type(e), e)))
        st.info("Try upgrading libraries:\n  pip install --upgrade diffusers transformers accelerate safetensors")
        st.stop()

@st.cache_resource
def load_upscaler():
    """Load Stable Diffusion x4 Upscaler model."""
    try:
        upscaler = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=dtype,
            safety_checker=None,
            cache_dir=r"E:\python3.13"
        )
        upscaler = upscaler.to(device)
        return upscaler
    except Exception as e:
        st.error("Error loading Upscaler model:\n" + "".join(traceback.format_exception_only(type(e), e)))
        st.stop()

# Load both models once and reuse
pipe = load_sd_model()
upscaler = load_upscaler()

# ============================================================
# Streamlit Page Setup
# ============================================================

st.set_page_config(page_title="AI 4K Image Generator", layout="wide")

# ============================================================
# Custom CSS Styling (Dark Mode UI)
# ============================================================

st.markdown("""
<style>
/* ---- Global Styles ---- */
body {
    background-color: #0E1117;
    color: #FAFAFA;
    font-family: 'Poppins', sans-serif;
}

/* ---- Title Styling ---- */
h1 {
    text-align: center;
    color: #F8F9FA;
    background: linear-gradient(90deg, #00B4DB, #0083B0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 30px;
}

/* ---- Sidebar Styling ---- */
[data-testid="stSidebar"] {
    background-color: #1B1F24 !important;
    border-right: 1px solid #333;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #E3E3E3 !important;
}

/* ---- Buttons ---- */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #00B4DB, #0083B0);
    color: white;
    border: none;
    padding: 0.6rem 1.5rem;
    border-radius: 12px;
    font-weight: 600;
    transition: all 0.3s ease;
}
div.stButton > button:first-child:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #2193b0, #6dd5ed);
}

/* ---- Image Display ---- */
img {
    border-radius: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    transition: transform 0.3s ease;
}
img:hover {
    transform: scale(1.01);
}

/* ---- Download Link ---- */
a {
    text-decoration: none;
    color: #00B4DB;
    font-weight: 600;
}
a:hover {
    color: #6dd5ed;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Title and Prompt Input
# ============================================================

st.title("🎨 AI 4K Image Generator — Stable Diffusion v1.5")

prompt_text = st.text_area("🧠 Enter a custom text prompt (optional):", "")

# ============================================================
# Sidebar: Theme and Settings
# ============================================================

st.sidebar.header("🎭 Select Theme")

themes = {
    "Cyberpunk": {
        "subject": "futuristic city",
        "style": "digital painting, neon lights, cinematic lighting",
        "time_of_day": "night",
        "colors": ["purple", "blue", "pink"],
        "details": "hovering cars, holograms, rain-soaked streets"
    },
    "Fantasy": {
        "subject": "enchanted forest",
        "style": "fantasy illustration, magical lighting",
        "time_of_day": "dawn",
        "colors": ["green", "gold", "purple"],
        "details": "floating islands, mystical creatures, sparkling rivers"
    },
    "Sci-Fi": {
        "subject": "alien planet",
        "style": "concept art, cinematic lighting",
        "time_of_day": "sunset",
        "colors": ["orange", "red", "blue"],
        "details": "futuristic buildings, alien flora, glowing sky"
    },
    "Custom": {
        "subject": "",
        "style": "",
        "time_of_day": "",
        "colors": [],
        "details": ""
    },
}

selected_theme = st.sidebar.selectbox("Select Theme", list(themes.keys()), index=0)
theme_data = themes[selected_theme]

# ============================================================
# Sidebar: Image Attributes
# ============================================================

st.sidebar.header("📝 Image Attributes")

subject = st.sidebar.text_input("Subject", theme_data["subject"])
style = st.sidebar.text_input("Style", theme_data["style"])
time_of_day = st.sidebar.selectbox(
    "Time of Day",
    ["day", "sunset", "night", "dawn", "dusk"],
    index=["day", "sunset", "night", "dawn", "dusk"].index(theme_data["time_of_day"]) if theme_data["time_of_day"] else 0
)
colors = st.sidebar.multiselect(
    "Colors",
    ["red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "black", "brown"],
    default=theme_data["colors"]
)
details = st.sidebar.text_area("Additional Details", theme_data["details"])

# ============================================================
# Sidebar: Advanced Settings
# ============================================================

st.sidebar.header("⚙️ Advanced Settings")

steps = st.sidebar.slider("Inference Steps", 10, 100, 30, step=5)
guidance_scale = st.sidebar.slider("Guidance Scale", 1.0, 15.0, 7.5, step=0.5)
base_res = st.sidebar.selectbox("Base Resolution", [512, 640, 768], index=0)
use_upscaler = st.sidebar.checkbox("Upscale to 4K", value=True)

# ============================================================
# Image Generation Process
# ============================================================

if st.button("✨ Generate Image"):
    try:
        # ---- Build Prompt ----
        json_prompt = f"{subject}, {style}, {time_of_day}, colors: {', '.join(colors)}, {details}"
        final_prompt = prompt_text.strip() if prompt_text.strip() else json_prompt

        # ---- Generate Base Image ----
        with st.spinner("🎨 Generating base image..."):
            out = pipe(
                prompt=final_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=base_res,
                height=base_res,
            )
            base_image = out.images[0]

        st.image(base_image, caption="🖼️ Base Image", use_column_width=True)

        # ---- Optional 4K Upscaling ----
        if use_upscaler:
            with st.spinner("🔼 Upscaling to 4K..."):
                up = upscaler(
                    prompt=final_prompt,
                    image=base_image,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                )
                upscaled_image = up.images[0]

            st.image(upscaled_image, caption="🌆 Upscaled 4K Image", use_column_width=True)

            # ---- Download Link ----
            buf = io.BytesIO()
            upscaled_image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f'<a href="data:file/png;base64,{b64}" download="4k_image.png">📥 Download 4K Image</a>', unsafe_allow_html=True)
        else:
            buf = io.BytesIO()
            base_image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f'<a href="data:file/png;base64,{b64}" download="image.png">📥 Download Image</a>', unsafe_allow_html=True)

    except Exception as e:
        st.error("❌ Image generation failed.")
        st.text("".join(traceback.format_exception_only(type(e), e)))
        st.info("If `'offload_state_dict'` errors persist, align diffusers and transformers versions.")
