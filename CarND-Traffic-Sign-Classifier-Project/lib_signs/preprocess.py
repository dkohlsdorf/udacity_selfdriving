import numpy as np

def to_grayscale(img_stack):
    '''
    Convert a stack of color images to grayscale
    
    :param img_stack: an image stack with n images of size w, h and with 3 color channels: (n, h, w, c)
    :returns: the gray scale image
    '''
    r = img_stack[:, :, :, 0] / 3 
    g = img_stack[:, :, :, 1] / 3 
    b = img_stack[:, :, :, 2] / 3
    return r + g + b


def to_znorm(img_stack):
    '''
    Normalize image stack to standard score
    
    :param img_stack: an image stack with n images of size w, h: (n, w, h)
    :returns: x ~ N(0, 1)
    '''
    n, w, h   = img_stack.shape
    flattened = img_stack.reshape(n, w * h)
    mu        = np.mean(flattened, axis=1).reshape(n, 1)
    std       = np.std(flattened, axis=1).reshape(n, 1) + 1 
    flattened = (flattened - mu) / std
    return flattened.reshape(n, w, h)
