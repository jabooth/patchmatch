import numpy as np
import numpy.ma as ma
from numpy.random import random

def extract_patch(img, pos, size=3):
    pos = np.round(pos)
    if size % 2 != 1:
        raise ValueError("Only odd patch sizes are allowed")
    if size < 3:
        raise ValueError("Patch size has to be 3 or greater")
    img_shape = np.array(img.shape[:2])
    step = (size - 1) / 2
    # grid of index values(s.t. [i, j] is the index into the image
    grid = np.rollaxis(np.indices([size, size]), 0, 3)  - step + pos
    # find patch values that go out of the image bounds
    invalid_under = np.any(grid < 0, axis=-1)
    invalid_over = np.any(grid > np.array(img.shape) - 1, axis=-1)
    # True iff that patch value is inside the image bounds
    legal = ~np.logical_or(invalid_over, invalid_under)
    # numpy slicing is OK if we go over, but we better not go under!
    min_ex = pos - step
    min_ex[min_ex < 0] = 0
    x, y = pos  # for neatness...
    patch_legal_vals = img[min_ex[0]:x+step+1, min_ex[1]:y+step+1]
    patch = np.zeros([size, size])  # the patch we will return
    patch[legal] = patch_legal_vals.ravel()
    return ma.masked_array(patch, mask=~legal)

def compare_patches(a, b):
    return ((a - b) ** 2).sum()

def initial_nnf(a, b):
    # each pixel maps into [0:b_width, 0:b_height]
    nnf = random(a.shape + (2,)) * (np.array(b.shape) - 1)
    # need to correct so that offsets are correct
    grid = np.rollaxis(np.indices(a.shape), 0, 3)
    return nnf - grid

def patchmatch(a, b):
    # build an inital (random) f
    f = initial_nnf(a, b)
    width, height = a.shape
    for x in xrange(1, width):
        for y in xrange(1, height):
            pos = np.array([x, y])
            # Propagate out
            propagate(a, b, pos, f)
            # Random search
            random_search(a, b, pos, f)
    return f

def propagate(a, b, pos, f):
    x, y = pos
    print 'x:{}, y:{}'.format(x, y)
    f_current = f[x, y]
    f_horiz = f[x-1, y]
    f_vert = f[x, y-1]
    current = D(f_current, pos, a, b)
    horiz = D(f_horiz, pos, a, b)
    vert = D(f_vert, pos, a, b)
    options = [(current, f_current), (horiz, f_horiz), (vert, f_vert)]
    options_sorted = sorted(options, key=lambda x: x[0])
    print 'f(x, y):{}, f(x-1, y):{}, f(x, y-1):{}'.format(f_current, f_horiz, f_vert)
    print 'D(f(x, y)):{}, D(f(x-1, y)):{}, D(f(x, y-1)):{}'.format(current, horiz, vert)
    print options_sorted
    f[x, y] = options_sorted[0][1]

def random_search(a, b, pos, f, w=None, alpha=0.5):
    b_shape = np.array(b.shape)
    x, y = pos
    D_0 = D(f[x, y], pos, a, b)
    if w is None:
        w = max(a.shape + b.shape)
    v_0 = f[pos[0], pos[1]]
    i = 0
    radius = w * alpha ** i
    while radius > 1:
        r_min, r_max = clip_to_bounds(b, pos, radius)
        r_size = r_max - r_min
        random_offset = (random(2) * r_size) + r_min
        u_i = v_0 + random_offset
        #print 'u_i: {} (offset {})'.format(u_i, random_offset)
        D_i = D(u_i, pos, a, b)
        if D_i < D_0:
            # this offset is better! save it
            f[x, y] = u_i
            D_0 = D_i
        i += 1
        radius = w * alpha ** i

def clip_to_bounds(img, pos, radius):
    img_shape = np.array(img.shape)
    r_min = np.ones(2) * -radius
    r_max = np.ones(2) * radius
    #print 'r: {}, (x,y) = {}'.format(radius, pos)
    max_ex = pos + radius
    min_ex = pos - radius
    #print 'search area: [{}:{}, {}:{}]'.format(min_ex[0], max_ex[0],
    #                                           min_ex[1], max_ex[1])
    out_of_min_bounds = min_ex < 0
    #print out_of_min_bounds
    out_of_max_bounds = max_ex > img_shape
    #print out_of_max_bounds
    r_min[out_of_min_bounds] = -pos[out_of_min_bounds]
    r_max[out_of_max_bounds] = img_shape[out_of_max_bounds]
    return r_min, r_max

def D(v, pos, a, b):
    patch_a = extract_patch(a, pos)
    patch_b = extract_patch(b, pos + v)
    print patch_a
    print patch_b
    return compare_patches(patch_a, patch_b)


a = random([10, 10])
b = random([5, 5])
f = initial_nnf(a, b)

