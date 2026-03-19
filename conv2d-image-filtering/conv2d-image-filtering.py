import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    image = np.array(image)
    kernel = np.array(kernel)

    if padding > 0:
        image = np.vstack((
                    np.zeros((padding, image.shape[1])), 
                    image, np.zeros((padding, image.shape[1]))))
        image = np.hstack((
                    np.zeros((image.shape[0], padding)),
                    image, np.zeros((image.shape[0], padding))))

    h, w = image.shape
    kh, kw = kernel.shape
    
    output = np.zeros(( (h - kh) // stride + 1, (w - kw) // stride + 1 ))

    h_out, w_out = output.shape
    
    
    for i in range(h_out):
        for j in range(w_out):
            for m in range(kh):
                for n in range(kw):
                        output[i, j] += image[i*stride + m, j*stride+n] * kernel[m, n]

    return output.tolist()
        