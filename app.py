"""
Streamlit app for AI-powered local portrait enhancement.
Upload an image, choose a goal, and watch the agent decide the enhancement pipeline.
"""

import json

import streamlit as st
from PIL import Image

from src.agent import run_agent_plan
from src.image_analysis import analyze_image
from src.image_processing import apply_pipeline, OPERATION_NAMES

# Available enhancement goals
GOALS = [
    "Professional Headshot",
    "Improve Lighting",
    "Sharpen Face",
    "Social Media Photo",
    "Auto Enhance",
]

# Local model to use with Ollama
MODEL_NAME = "phi3"


def load_image(uploaded_file) -> Image.Image:
    """Load and convert uploaded image to RGB."""
    return Image.open(uploaded_file).convert("RGB")


def format_analysis(analysis: dict) -> str:
    """Format image analysis metrics for display."""
    formatted = []
    for key, value in analysis.items():
        if key == "raw":
            continue
        formatted.append(f"**{key.capitalize()}:** {value}")
    return "\n".join(formatted)


def show_pipeline_explanation(pipeline: list) -> str:
    """Format the enhancement pipeline as a readable list."""
    if not pipeline:
        return "No enhancement steps were selected."
    lines = []
    for step in pipeline:
        label = OPERATION_NAMES.get(step, step)
        lines.append(f"- {label}")
    return "\n".join(lines)


def main():
    st.set_page_config(page_title="AI Photo Enhancer", layout="wide")
    st.title("🎨 AI Portrait Enhancer")
    st.write(
        "Upload a portrait or selfie, choose an enhancement goal, "
        "and the local Ollama agent will suggest a lightweight enhancement pipeline."
    )

    if "enhance_ready" not in st.session_state:
        st.session_state.enhance_ready = False

    with st.sidebar.form(key="enhance_form"):
        st.header("📥 Input")
        uploaded_file = st.file_uploader(
            "Upload a portrait or selfie", type=["jpg", "jpeg", "png"]
        )

        st.markdown("---")
        st.header("🎯 Goal")
        goal = st.selectbox("Choose a goal", GOALS)

        st.markdown("---")
        enhance_button = st.form_submit_button("Enhance image")

    if enhance_button:
        st.session_state.enhance_ready = True

    with st.sidebar:
        st.header("⚙️ Agent Settings")
        st.write("**Local model used for reasoning:**")
        st.code(MODEL_NAME, language="text")
        st.info(
            "Make sure Ollama is running and `"
            + MODEL_NAME
            + "` is pulled locally."
        )

    # Early exit if no image uploaded
    if uploaded_file is None:
        st.session_state.enhance_ready = False
        st.info("👆 Upload an image to see the enhancement workflow.")
        return

    if not st.session_state.enhance_ready:
        st.info("✅ Select a goal and press Enhance to start the workflow.")
        return

    # Load and display original image
    image = load_image(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)

        # Analyze image
        st.subheader("📊 Image Analysis")
        analysis = analyze_image(image)
        st.markdown(format_analysis(analysis))

    # Run agent and generate pipeline
    with st.spinner("🤖 Running the local agent and generating a pipeline..."):
        plan = run_agent_plan(goal, analysis, model=MODEL_NAME)

    # Extract pipeline from plan
    pipeline = plan.get("pipeline", [])

    # Show agent's JSON response
    st.subheader("🔧 Agent Enhancement Plan (JSON)")
    st.code(json.dumps({"pipeline": pipeline}, indent=2), language="json")

    # Apply enhancements
    enhanced = apply_pipeline(image, pipeline)

    # Display enhanced image
    with col2:
        st.subheader("✨ Enhanced Image")
        st.image(enhanced, use_column_width=True)
        st.subheader("📝 What was done")
        st.markdown(show_pipeline_explanation(pipeline))


if __name__ == "__main__":
    main()
