import numpy as np
from numpy.random import random


def initial_nnf(a, b):
    # each pixel maps into [0:b_width, 0:b_height]
    nnf = random(a.shape + (2,)) * (np.array(b.shape) - 1)
    # need to correct so that offsets are correct
    grid = np.rollaxis(np.indices(a.shape), 0, 3)
    return nnf - grid


def patchmatch(a, b):
    # build an initial (random) f
    f = initial_nnf(a, b)
    x_max, y_max = a.shape
    # TODO consider boundary case here
    for x in xrange(1, x_max):
        print 'x: {}'.format(x)
        for y in xrange(1, y_max):
            pos = np.array([x, y])
            # Propagate out
            propagate(a, b, pos, f)
            # Random search
            random_search(a, b, pos, f)
    return f


def propagate(a, b, pos, f):
    x, y = pos
    #print 'x:{}, y:{}'.format(x, y)
    f_current = f[x, y]
    f_horiz = f[x-1, y]
    f_vert = f[x, y-1]
    current = d(f_current, pos, a, b)
    horiz = d(f_horiz, pos, a, b)
    vert = d(f_vert, pos, a, b)
    options = [(current, f_current), (horiz, f_horiz), (vert, f_vert)]
    options_sorted = sorted(options, key=lambda x: x[0])
    #print 'f(x, y):{}, f(x-1, y):{}, f(x, y-1):{}'.format(f_current,
    # f_horiz, f_vert)
    #print 'd(f(x, y)):{}, d(f(x-1, y)):{}, d(f(x, y-1)):{}'.format(current,
    #                                                                horiz,
    # vert)
    #print options_sorted
    f[x, y] = options_sorted[0][1]


def random_search(a, b, pos, f, w=None, alpha=0.5):
    b_shape = np.array(b.shape)
    x, y = pos
    d_0 = d(f[x, y], pos, a, b)
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
        d_i = d(u_i, pos, a, b)
        if d_i < d_0:
            # this offset is better! save it
            f[x, y] = u_i
            d_0 = d_i
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


def d(v, pos, a, b):
    patch_a = extract_patch(a, pos)
    patch_b = extract_patch(b, pos + v)
    #print patch_a
    #print patch_b
    return compare_patches(patch_a, patch_b)


def extract_patch(img, pos, size=3):
    if size % 2 != 1 or size < 3:
        raise ValueError("size must be an odd number >=3")
    img_shape = np.array(img.shape[:2])
    pos = np.round(pos)
    step = (size - 1) / 2  # how far we want to step either side pos
    # grid[i, j] = [img_i, img_j], where we want to sample
    grid = np.rollaxis(np.indices([size, size]), 0, 3) - step + pos
    # find patch values that go out of the image bounds
    illegal_under = np.any(grid < 0, axis=-1)
    illegal_over = np.any(grid > img_shape - 1, axis=-1)
    # TODO check this is the right interpretation!
    # legal[i, j] = True iff [img_i, img_j] is inside the image bounds
    legal = ~np.logical_or(illegal_over, illegal_under)
    # numpy slicing is OK if we go over, but we better not go under!
    min_ex = pos - step
    min_ex[min_ex < 0] = 0
    x, y = pos  # for neatness...
    patch_legal_values = img[min_ex[0]:x+step+1, min_ex[1]:y+step+1]
    patch = np.zeros([size, size])  # the patch we will return
    patch[legal] = patch_legal_values.ravel()   # fill the legal values in
    return np.ma.masked_array(patch, mask=~legal)


def compare_patches(a, b):
    sd = (a - b) ** 2
    n_valid = (~sd.mask).sum()
    if n_valid == 0:
        return 0
    n_elements = sd.size
    # return ssd scaled by how many values are invalid
    return sd.sum() * (n_elements * 1.0) / n_valid


def rebuild_img(f, a, b):
    from scipy.ndimage.interpolation import map_coordinates
    grid = np.rollaxis(np.indices(a.shape), 0, 3)
    sampler = (grid + f).reshape([-1, 2]).T
    return map_coordinates(b, sampler).reshape(a.shape)
