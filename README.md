AI Generative Art Lab 🎨✨

A WebGL and TensorFlow-powered creative suite featuring real-time image synthesis, neural style transfer, prompt engineering interface simulation, and procedural canvas generation.

🌟 Features

Neural Style Transfer: Blend content images with arbitrary artistic styles in real time using TensorFlow Hub models (Google Magenta).

Procedural Art Engine: Generate mathematical vector-driven abstract canvases directly through RGB tensor manipulations.

Interactive Prompt Engineering Suite: Adjust synthesis strength, noise seeds, and step resolution dynamically.

High-Resolution Export: Output generated artwork seamlessly as high-quality image formats.

🚀 Quick Start

Prerequisites

Ensure you have Python 3.8+ installed along with pip.

Installation

Clone the repository:

git clone https://github.com/yuT0aa/AI-Generative-Art-Lab
cd ai-generative-art-lab


Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:

pip install tensorflow tensorflow-hub numpy pillow matplotlib


💻 Usage

Run the main script to generate procedural art or perform neural style transfer:

from main import AIGenerativeArtLab

# Initialize the lab
lab = AIGenerativeArtLab()

# 1. Generate Procedural Canvas
procedural_art = lab.generate_procedural_art(width=512, height=512, seed=42)
procedural_art.save("output_procedural.png")

# 2. Apply Neural Style Transfer
stylized_art = lab.generate_style_transfer(
    content_path="content.jpg",
    style_path="style.jpg"
)
stylized_art.save("output_stylized.png")


🛠️ Project Structure

ai-generative-art-lab/
├── main.py              # Main generative pipeline & model loader
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
└── outputs/             # Directory for generated images


📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
