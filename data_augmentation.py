import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk


def random_transform(x, y=None,
                     rotation_range=0.,
                     width_shift_range=0.,
                     height_shift_range=0.,
                     shear_range=0.,
                     zoom_range=0.,
                     channel_shift_range=0.,
                     fill_mode='nearest',
                     cval=0.,
                     cvalMask=0.,
                     horizontal_flip=False,
                     vertical_flip=False,
                     rescale=None,
                     spline_warp=False,
                     warp_sigma=0.1,
                     warp_grid_size=3,
                     crop_size=None,
                     rng=None):
    
    # Set random number generator
    if rng is None:
        rng = np.random.RandomState()
    
    # x is a single image, so we don't have batch dimension 
    assert(x.ndim == 3)
    if y is not None:
        assert(y.ndim == 3)
    img_row_index = 1
    img_col_index = 2
    img_channel_index = 0
    
    # Nonlinear spline warping
    if spline_warp:
        warp_field = _gen_warp_field(shape=x.shape[-2:],
                                     sigma=warp_sigma,
                                     grid_size=warp_grid_size,
                                     rng=rng)
        x = _apply_warp(x, warp_field,
                        interpolator=sitk.sitkNearestNeighbor,
                        fill_mode=fill_mode,
                        fill_constant=cval)
        if y is not None:
            y = np.round(_apply_warp(y, warp_field,
                                     interpolator=sitk.sitkNearestNeighbor,
                                     fill_mode=fill_mode,
                                     fill_constant=cvalMask))
    
    # use composition of homographies to generate final transform that needs
    # to be applied
    if np.isscalar(zoom_range):
        zoom_range = [1 - zoom_range, 1 + zoom_range]
    elif len(zoom_range) == 2:
        zoom_range = [zoom_range[0], zoom_range[1]]
    else:
        raise Exception('zoom_range should be a float or '
                        'a tuple or list of two floats. '
                        'Received arg: ', zoom_range)
    
    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = rng.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    
    if rotation_range:
        theta = np.pi / 180 * rng.uniform(-rotation_range, rotation_range)
    else:
        theta = 0
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    if height_shift_range:
        tx = rng.uniform(-height_shift_range, height_shift_range) \
                 * x.shape[img_row_index]
    else:
        tx = 0

    if width_shift_range:
        ty = rng.uniform(-width_shift_range, width_shift_range) \
                 * x.shape[img_col_index]
    else:
        ty = 0

    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    if shear_range:
        shear = rng.uniform(-shear_range, shear_range)
    else:
        shear = 0
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, shear_matrix),
                                     translation_matrix), zoom_matrix)

    h, w = x.shape[img_row_index], x.shape[img_col_index]
    transform_matrix = _transform_matrix_offset_center(transform_matrix, h, w)
    x = _apply_transform_matrix(x, transform_matrix, img_channel_index,
                                fill_mode=fill_mode, cval=cval)
    if y is not None:
        y = _apply_transform_matrix(y, transform_matrix, img_channel_index,
                                    fill_mode=fill_mode, cval=cvalMask)

    if channel_shift_range != 0:
        x = _random_channel_shift(x, channel_shift_range, img_channel_index,
                                  rng=rng)

    if horizontal_flip:
        if rng.random_sample() < 0.5:
            x = _flip_axis(x, img_col_index)
            if y is not None:
                y = _flip_axis(y, img_col_index)

    if vertical_flip:
        if rng.random_sample() < 0.5:
            x = _flip_axis(x, img_row_index)
            if y is not None:
                y = _flip_axis(y, img_row_index)

    # Crop
    crop = list(crop_size) if crop_size else None
    if crop:
        h, w = x.shape[img_row_index], x.shape[img_col_index]

        if crop[0] < h:
            top = rng.randint(h - crop[0])
        else:
            print('Data augmentation: Crop height >= image size')
            top, crop[0] = 0, h
        if crop[1] < w:
            left = rng.randint(w - crop[1])
        else:
            print('Data augmentation: Crop width >= image size')
            left, crop[1] = 0, w

        x = x[:, :, top:top+crop[0], left:left+crop[1]]
        if y is not None:
            y = y[:, :, top:top+crop[0], left:left+crop[1]]

    if y is None:
        return x
    else:
        return x, y 



def _transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x],
                              [0, 1, o_y],
                              [0, 0,   1]])
    reset_matrix = np.array([[1, 0, -o_x],
                             [0, 1, -o_y],
                             [0, 0,    1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def _apply_transform_matrix(x, transform_matrix, channel_index=0,
                            fill_mode='nearest', cval=0.):
    x_ = np.copy(x)
    x_ = np.rollaxis(x_, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(\
                          x_channel, final_affine_matrix, final_offset,
                          order=0, mode=fill_mode, cval=cval)\
                                                           for x_channel in x_]
    x_out = np.stack(channel_images, axis=0)
    x_out = np.rollaxis(x_out, 0, channel_index+1)
    return x_out


def _random_channel_shift(x, intensity, channel_index=0, rng=None):
    x_ = np.copy(x)
    x_ = np.rollaxis(x_, channel_index, 0)
    channel_images = [np.clip(x_channel + \
                              rng.uniform(-intensity, intensity),
                              np.min(x_), np.max(x_))      for x_channel in x_]
    x_out = np.stack(channel_images, axis=0)
    x_out = np.rollaxis(x_out, 0, channel_index+1)
    return x_out


def _flip_axis(x, axis):
    x_ = np.copy(x)
    x_ = np.asarray(x_).swapaxes(axis, 0)
    x_ = x_[::-1, ...]
    x_ = x_.swapaxes(0, axis)
    x_out = x_
    return x_out


def _gen_warp_field(shape, sigma=0.1, grid_size=3, rng=None):
    # Initialize bspline transform
    args = shape+(sitk.sitkFloat32,)
    ref_image = sitk.Image(*args)
    tx = sitk.BSplineTransformInitializer(ref_image, [grid_size, grid_size])

    # Initialize shift in control points:
    # mesh size = number of control points - spline order
    p = sigma * rng.randn(grid_size+3, grid_size+3, 2)

    # Anchor the edges of the image
    p[:, 0, :] = 0
    p[:, -1:, :] = 0
    p[0, :, :] = 0
    p[-1:, :, :] = 0

    # Set bspline transform parameters to the above shifts
    tx.SetParameters(p.flatten())

    # Compute deformation field
    displacement_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_filter.SetReferenceImage(ref_image)
    displacement_field = displacement_filter.Execute(tx)

    return displacement_field


def _pad_image(x, pad_amount, mode='reflect', constant=0.):
    e = pad_amount
    assert(len(x.shape)>=2)
    shape = list(x.shape)
    shape[:2] += 2*e
    if mode == 'constant':
        x_padded = np.ones(shape, dtype=np.float32)*constant
        x_padded[e:-e, e:-e] = x.copy()
    else:
        x_padded = np.zeros(shape, dtype=np.float32)
        x_padded[e:-e, e:-e] = x.copy()

    if mode == 'reflect':
        x_padded[:e, e:-e] = np.flipud(x[:e, :])  # left edge
        x_padded[-e:, e:-e] = np.flipud(x[-e:, :])  # right edge
        x_padded[e:-e, :e] = np.fliplr(x[:, :e])  # top edge
        x_padded[e:-e, -e:] = np.fliplr(x[:, -e:])  # bottom edge
        x_padded[:e, :e] = np.fliplr(np.flipud(x[:e, :e]))  # top-left corner
        x_padded[-e:, :e] = np.fliplr(np.flipud(x[-e:, :e]))  # top-right
        x_padded[:e, -e:] = np.fliplr(np.flipud(x[:e, -e:]))  # bottom-left
        x_padded[-e:, -e:] = np.fliplr(np.flipud(x[-e:, -e:]))  # bottom-right
    elif mode == 'zero' or mode == 'constant':
        pass
    elif mode == 'nearest':
        x_padded[:e, e:-e] = x[[0], :]  # left edge
        x_padded[-e:, e:-e] = x[[-1], :]  # right edge
        x_padded[e:-e, :e] = x[:, [0]]  # top edge
        x_padded[e:-e, -e:] = x[:, [-1]]  # bottom edge
        x_padded[:e, :e] = x[[0], [0]]  # top-left corner
        x_padded[-e:, :e] = x[[-1], [0]]  # top-right corner
        x_padded[:e, -e:] = x[[0], [-1]]  # bottom-left corner
        x_padded[-e:, -e:] = x[[-1], [-1]]  # bottom-right corner
    else:
        raise ValueError("Unsupported padding mode \"{}\"".format(mode))
    return x_padded


def _apply_warp(x, warp_field, fill_mode='reflect',
               interpolator=sitk.sitkLinear,
               fill_constant=0, channel_index=0):
    # Expand deformation field (and later the image), padding for the largest
    # deformation
    warp_field_arr = sitk.GetArrayFromImage(warp_field)
    max_deformation = np.max(np.abs(warp_field_arr))
    pad = np.ceil(max_deformation).astype(np.int32)
    warp_field_padded_arr = _pad_image(warp_field_arr, pad_amount=pad,
                                       mode='nearest')
    warp_field_padded = sitk.GetImageFromArray(warp_field_padded_arr,
                                               isVector=True)

    # Warp x, one filter slice at a time
    x_warped = np.zeros(x.shape, dtype=np.float32)
    warp_filter = sitk.WarpImageFilter()
    warp_filter.SetInterpolator(interpolator)
    warp_filter.SetEdgePaddingValue(np.min(x).astype(np.double))
    x_by_channel = np.rollaxis(x, channel_index, 0)
    for i, channel in enumerate(x_by_channel):
        image_padded = _pad_image(channel, pad_amount=pad, mode=fill_mode,
                                  constant=fill_constant).T
        image_f = sitk.GetImageFromArray(image_padded)
        image_f_warped = warp_filter.Execute(image_f, warp_field_padded)
        image_warped = sitk.GetArrayFromImage(image_f_warped)
        x_warped[i] = image_warped[pad:-pad, pad:-pad].T
    x_warped = np.rollaxis(x_warped, 0, channel_index+1)
    return x_warped
