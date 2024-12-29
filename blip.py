"""
Transformers functions.
"""
from transformers import BlipProcessor, BlipForConditionalGeneration


def image_captioning(image, question: str = None) -> str:
    """Generate a caption or answer a question for an image using the Blip model."""
    # Initialize the processor and model from Hugging Face.
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    # Use the processor to prepare inputs for VQA (image + question).
    inputs = processor(image, question, return_tensors="pt")
    # Generate captions.
    outputs = model.generate(**inputs)
    response = processor.decode(outputs[0], skip_special_tokens=True)

    return response
 