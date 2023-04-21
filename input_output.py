import os
import skimage.io
import torch
import matplotlib.pyplot as plt


def from_folder_to_tensor(path, datatype=torch.float32):
    # get nuber of file, dimensions and prepare empty output tensor
    all_files = os.listdir(path)
    batch_size = len(all_files)
    im = skimage.io.imread(os.path.join(path, all_files[0]))
    height = im.shape[0]
    width = im.shape[1]
    grayscale = len(im.shape) == 2
    if grayscale:
        output_tensor = torch.empty((batch_size, 1, height, width), dtype=datatype)
    else:
        output_tensor = torch.empty((batch_size, im.shape[2], height, width), dtype=datatype)

    orig_shape = im.shape
    # transform images to tensor one by one and add them to prepared tensor
    for i, f in enumerate(all_files):
        im = skimage.io.imread(os.path.join(path, f))
        # check that image has same dimensions as first one
        if im.shape != orig_shape:
            print("ERROR: image dimensions do not match!")
            return 1
        if grayscale:
            output_tensor[i, 0, :, :] = torch.tensor(im, dtype=datatype) / 255
        else:
            im_tensor = torch.tensor(im, dtype=datatype) / 255
            output_tensor[i, :, :, :] = im_tensor.view(im_tensor.shape[2], im_tensor.shape[0], im_tensor.shape[1])
    return output_tensor


def from_image_to_tensor(path, datatype=torch.float32):
    im = skimage.io.imread(path)
    height = im.shape[0]
    width = im.shape[1]
    grayscale = len(im.shape) == 2
    im_tensor = torch.tensor(im, dtype=datatype) / 255

    if grayscale:
        im_tensor = im_tensor.reshape((1, 1, height, width))
    else:
        im_tensor = im_tensor.reshape((1, im.shape[2], height, width))

    return im_tensor


def save_images_from_tensor(tensor, output_folder, filenames=None):
    num_images = tensor.shape[0]
    if filenames is None:
        filenames = ["im" + str(x) + ".jpg" for x in range(num_images)]
    if tensor.shape[0] != len(filenames):
        print("Warning: tensor size does not correspond to filename count. Using default filenames.")
        filenames = ["im" + str(x) + ".jpg" for x in range(num_images)]
    for i in range(tensor.shape[0]):
        skimage.io.imsave(os.path.join(output_folder, filenames[i]), tensor[i, :, :, :])


def display_nth_image_from_tensor(tensor, n=0):
    if tensor.shape[0] < n - 1:
        print("Error: tensor does not have " + str(n) + " images")
        return 1
    tensor_image = tensor[n - 1].view(tensor.shape[2], tensor.shape[3], tensor.shape[1])
    plt.imshow(tensor_image)
    plt.show()
    return 0
