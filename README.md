# AI Photo Enhancer

A lightweight **local** Streamlit web app that orchestrates portrait enhancements using Ollama and OpenCV. The app uses a simple LLM agent running on your machine to decide which lightweight image processing operations to apply.

## Features

- 📸 **Local image upload** – no cloud, no data sent anywhere
- 🤖 **Local LLM agent** – uses Ollama + a small model (phi3) to reason about enhancements
- 🎨 **Lightweight image processing** – OpenCV & Pillow for CPU-friendly operations
- 🏗️ **Clean architecture** – UI layer, agent layer, processing layer clearly separated
- 📊 **Image analysis** – brightness, blur, and noise metrics inform the agent's decisions
- 💾 **Minimal dependencies** – no heavy ML frameworks, no GPU required

## Setup

### Prerequisites

1. **Ollama installed** on macOS (or Linux)
   - Download from [ollama.com](https://ollama.com)
   - Pull a small model:
     ```bash
     ollama pull phi3
     ```

2. **Python 3.10+**

### Installation

1. Clone and navigate to the repo:
   ```bash
   cd ai-photo-enhancer
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## How It Works

1. **Upload** a portrait or selfie image
2. **Choose a goal** (e.g., "Professional Headshot", "Improve Lighting")
3. **Analyze** – the app computes brightness, blur, and noise metrics
4. **Agent reasoning** – Ollama generates a JSON enhancement pipeline based on the goal and metrics
5. **Apply** – the backend runs each enhancement operation sequentially
6. **Display** – see the enhanced image and the JSON plan side-by-side

### Architecture

- **`app.py`** – Streamlit UI and orchestration
- **`src/agent.py`** – LLM prompt generation, Ollama integration, JSON parsing
- **`src/image_analysis.py`** – computes brightness, blur, noise metrics
- **`src/image_processing.py`** – local enhancement operations (brightness, contrast, sharpening, denoising)

## Enhancement Goals

The app includes five predefined enhancement goals:

| Goal | Operations |
|------|------------|
| **Professional Headshot** | brightness, contrast, face sharpening |
| **Improve Lighting** | brightness, contrast adjustment |
| **Sharpen Face** | sharpening, contrast |
| **Social Media Photo** | auto enhancement, face sharpening |
| **Auto Enhance** | brightness, contrast, noise reduction |

The Ollama agent can choose a subset or reorder these, depending on the image analysis.

## Local Model Notes

- **phi3** is lightweight (~4GB) and runs on CPU
- Other options: `gemma` (9B), `qwen` (~7B)
- Keep Ollama running in the background or start it before the app

## Roadmap

- [ ] **Phase 2:** Face detection for targeted sharpening
- [ ] **Phase 3:** Optional support for heavier models (GFPGAN, Real-ESRGAN)
- [ ] **Phase 4:** Batch processing, CLI interface
- [ ] **Phase 5:** Config file for custom pipelines

## Code Philosophy

- **Simple over clever** – straightforward Python functions, minimal abstraction
- **Local first** – everything runs on your machine
- **Educational** – designed to be readable and learnable
- **Extensible** – easy to add new enhancement operations or goals

## License

See LICENSE file.
