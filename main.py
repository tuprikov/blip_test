"""
Main module.
"""
import gradio as gr

from blip import image_captioning
from pytorch import predict
from ui import launch


def main():
    """Main function."""
    #launch(image_captioning, inputs=[gr.Image(type="pil"), "text"], outputs="text")
    launch(predict, inputs=gr.Image(type="pil"), outputs=gr.Label(num_top_classes=3))

if __name__ == "__main__":
    main()
