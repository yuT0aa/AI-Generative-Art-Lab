import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

def tensor_to_image(tensor):
    """Converts a TensorFlow tensor into a PIL Image."""
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_and_process_img(path_to_img, max_dim=512):
    """Loads an image and resizes it while maintaining aspect ratio."""
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

class AIGenerativeArtLab:
    def __init__(self):
        # Load pre-trained Magenta Neural Style Transfer model from TensorFlow Hub
        print("Loading TensorFlow Style Transfer Model...")
        self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    def generate_style_transfer(self, content_path, style_path):
        """Applies neural style transfer between a content and style image."""
        content_image = load_and_process_img(content_path)
        style_image = load_and_process_img(style_path)

        # Run style transfer model
        stylized_image = self.model(tf.constant(content_image), tf.constant(style_image))[0]
        return tensor_to_image(stylized_image)

    def generate_procedural_art(self, width=512, height=512, seed=42):
        """Generates procedural abstract art using random TensorFlow operations."""
        tf.random.set_seed(seed)
        
        # Create coordinate grids
        x = tf.linspace(-2.0, 2.0, width)
        y = tf.linspace(-2.0, 2.0, height)
        X, Y = tf.meshgrid(x, y)

        # Mathematical procedural patterns
        R = tf.sqrt(X**2 + Y**2)
        Z1 = tf.sin(R * 10.0) + tf.cos(X * 5.0)
        Z2 = tf.cos(R * 8.0) - tf.sin(Y * 5.0)
        Z3 = tf.sin(X * Y * 12.0)

        # Stack into RGB channels and normalize
        rgb = tf.stack([Z1, Z2, Z3], axis=-1)
        rgb = (rgb - tf.reduce_min(rgb)) / (tf.reduce_max(rgb) - tf.reduce_min(rgb))
        
        return tensor_to_image(rgb[tf.newaxis, :])


# --- Example Usage ---
if __name__ == "__main__":
    lab = AIGenerativeArtLab()

    # 1. Generate Procedural Art Canvas
    print("Generating procedural pattern...")
    procedural_art = lab.generate_procedural_art(width=512, height=512, seed=123)
    procedural_art.save("procedural_canvas.png")

    # 2. Perform Style Transfer (Requires local image paths)
    # result = lab.generate_style_transfer('content.jpg', 'style.jpg')
    # result.save('generated_art.png')

    print("Art generation complete! Output saved as procedural_canvas.png.")