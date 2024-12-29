"""
Main module.
"""
import gradio as gr

from blip import image_captioning
from ui import launch


def main():
    """Main function."""
    launch(image_captioning, inputs=[gr.Image(type="pil"), "text"], outputs="text")


if __name__ == "__main__":
    main()
