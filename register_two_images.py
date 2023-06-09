from load_images_to_tensor import load_grayscale
from registration import rigid_registration


def register_two_images(path_to_reference, path_to_sample):
    print("Starting registration.")
    ref = load_grayscale(path_to_reference)
    sample = load_grayscale(path_to_sample)
    print("Images loaded")
    return rigid_registration(ref, sample, verbose=True)
