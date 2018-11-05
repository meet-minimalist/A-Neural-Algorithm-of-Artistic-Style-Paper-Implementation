from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

VGG_MEAN = [123.68, 116.779, 103.939]
# RGB format ^

def resize_and_rescale_img(image_path, w, h, output_path_, output_filename):
    # This will resize the image to width x height dimensions and then scale down in the range of [0-1]   
    if os.path.isfile(image_path):
        img = Image.open(image_path)
        img_resized = img.resize(size=(w, h))
        if not os.path.exists(output_path_):
            os.makedirs(output_path_)
        img_resized.save(output_path_ + output_filename)
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, 0)
        img_array = img_array / 255
        
        return img_array
    else:
        print("No image found in given location.")

def post_process_and_display(cnn_output, output_path, output_filename, save_file=True):
    # This will take input_noise of (1, w, h, channels) shapped array taken from tensorflow operation
    # and ultimately displays the image
    
    x = np.squeeze(cnn_output)

    for i in range(x.shape[2]):
        x[:, :, i] -= np.mean(x[:, :, i])
        x[:, :, i] /= (np.std(x[:, :, i]) + 1e-05)
        x[:, :, i] -= np.amin(x[:, :, i])
        x[:, :, i] /= (np.amax(x[:, :, i]) - np.amin(x[:, :, i]))
        x[:, :, i] *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    
    img = Image.fromarray(x, mode='RGB')
    img.show()
    if save_file:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img.save(output_path + output_filename)
    
    return x