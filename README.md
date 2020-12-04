# Data tools #

This is a collection of generic tools for loading, manipulating, and creating datasets.

__data_tools__
* __wrap__
    * delayed_view
    * multi_source_array
* __io__
    * data_flow
    * buffered_array_writer
    * h5py_array_writer
    * bcolz_array_writer
    * zarr_array_writer
* __data_augmentation__
    * image_random_transform
    * image_stack_random_transform
* __patches__
    * patch_generator
    * create_dataset

## Installation ##

__Requires__
* numpy
* scipy
* SimpleITK
* h5py
* bcolz

__Setup__
```
python setup.py install --user
```

## Data wrappers ##

In `data_tools.wrap`.

These wrappers allow memory mapped files to be abstracted away as numpy-like arrays which can access data on demand from a file or across multiple files, in sequential or shuffled order. Concatenating the contents of multiple files or shuffling file contents would otherwise require first loading all file data into memory; here, elements of the data are only loaded into memory on demand, when they are accessed.

### Delayed view into an array ###

```python
def delayed_view(arr, shuffle=False, idx_min=None, idx_max=None)
```

Given an array, create a view into that array without preloading the viewed data into memory. Data is loaded as needed when indexing into the delayed_view.

#### Arguments ####
* __arr__ : the source array
* __shuffle__ : randomize data access order within the view
* __idx_min__ : the view into arr starts at this index
* __idx_max__ : the view into arr ends before this index

#### Example ####

Given a (typically memory-mapped) array `some_arr`, a subset of this array can be viewed into by a numpy-array-like object, in shuffled order, as follows:

```python
max_len = len(some_arr)
arr_view = delayed_view(some_arr, shuffle=True, idx_max=int(0.5*max_len))
```

Access of elements in `some_arr` is of course delayed until those elements are indexed in `arr_view`.

### Multi-source array ###

```python
class multi_source_array(delayed_view)
```

Given a list of sources, create an array-like interface that combines the sources. This object allows slicing and iterating over the elements. Data access automatically spans all data sources.

#### Arguments ####
Class initialization uses the following arguments:

```python
def __init__(self, source_list, class_list=None, shuffle=False, maxlen=None)
```

* __source_list__ : list of sources to combine into one source
* __class_list__ : specifies class number for each source; same length as source_list
* __shuffle__ : randomize data access order within and across all sources
* __maxlen__ : the maximum number of elements to take from each source; if shuffle is * __False__, a source is accessed as source[0:maxlen] and if shuffle is True, a source is accessed as shuffle(source)[0:maxlen]

#### Methods ####

```python
shuffle(random_seed)
```

An array can be reshuffled at any time with `shuffle()`. This function optionally takes a random seed as an argument:

```python
get_labels()
```

Retrieve the labels from the unified array. This is especially useful when the array is shuffled -- labels can be retrieved in the same shuffle order.

A label is associated with each source array (see `class_list` argument), thus assuming one class per source array which can be useful for classification datasets.

```python
__len__()
```

Return the length of the unified array when calling `len(obj)` on a multi_source_array `obj`.

#### Examples ####

For some arrays `a1`, `a2`, and `a3`, an array-like object that concatenates the three arrays without preloading them into memory can be create with:

```python
msarr = multi_source_array(source_list=[a1,a2,a3])
```

Access can optionally be shuffled across all arrays:

```python
msarr_shuffled = multi_source_array(source_list=[a1,a2,a3], shuffle=True)
```

An array can be reshuffled at any time with `shuffle()`. This function optionally takes a random seed as an argument:

```python
msarr = multi_source_array(source_list=[a1,a2,a3])
msarr.shuffle()        # Now shuffled.
msarr.shuffle(1234)    # Now shuffled using random seed 1234.
```

Sometimes, it is useful to maintain the same shuffle order for multiple multi_source_array objects. This can be done as follows:

```python
# With some `random_seed`
msarr_1 = multi_source_array(source_list=[a1,a2])
msarr_2 = multi_source_array(source_list=[a2,a3])
msarr_1.shuffle(random_seed)
msarr_2.shuffle(random_seed)

# Alternatively, a more direct hack can be used:
msarr_1 = multi_source_array(source_list=[a1,a2], shuffle=True)
msarr_2 = multi_source_array(source_list=[a2,a3])
msarr_2.index_pairs = msarr_1.index_pairs
```

Especially since data access can be in shuffled order, it may be useful to keep track of labels associated with data elements. One can associate an integer label with any input array. For example, if `a1` and `a2` are both datasets containing examples of class 0 and `a3` contains examples of class 1, one can specify this in `multi_source_array` with a `class_list` like so:

```python
msarr_shuffled = multi_source_array(source_list=[a1,a2,a3],
                                    class_list=[0,0,1],
                                    shuffle=True)
```

The labels from the unified (concatenated) and shuffled array can then be retrieved using `get_labels()`:

```python
msarr_shuffled_labels = msarr_shuffled.get_labels()
```

If one does not specify a `class_list`, then it is assumed to be in increasing sequential order (i.e. `[0,1,2]` in this example).

### Indexing ###

Indexing is numpy-style, using any combination of integers, slices, index lists, ellipsis (only one, as with numpy), and boolean arrays but not non-boolean multi-dimensional arrays. Note that the indexing style is also used on the underlying data sources so those data sources must support the style of indexing used with a multi_source_array object; use simple indexing with integers and slices (eg. obj[0,3:10]) when unsure.

Adding dimensions to the output just by indexing is not supported. This means that unlike with numpy, indexing cannot be done with `None` or `numpy.newaxis`; also, for example, an array `A` with shape (4,5) can be indexed as `A[[0,1]]` and `A[[[0,1]]]` (these are equivalent) but not as `A[[[[0,1]]]]` for which numpy would add a dimension to the output.


## Data loading ##

In `data_tools.io`.

For minibatch-wise model training, a minibatch data generator can be made efficient by preparing minibatches in parallel while a minibatch is being processed on GPU. In the provided generator class, data may be loaded in a separate thread and preprocessed in any number of separate processes (could also be in the main process).

```python
class data_flow(object)
```

Given a list of array-like objects, data from the objects is read in a parallel thread and processed in the same parallel thread or in a set of parallel processes. All objects are iterated in tandem (i.e. for a list data=[A, B, C], a batch of size 1 would be [A[i], B[i], C[i]] for some i).

#### Arguments ####
Class initialization uses the following arguments:

```python
def __init__(self, data, batch_size, nb_io_workers=1, nb_proc_workers=0,
             loop_forever=True, sample_random=False,
             sample_with_replacement=False, sample_weights=None,
             drop_incomplete_batches=False, preprocessor=None, rng=None)
```

* __data__ : A list of data arrays, each of equal length. When yielding a batch,  each element of the batch corresponds to each array in the data list.
* __batch_size__ : The maximum number of elements to yield from each data array in a batch. The actual batch size is the smallest of either this number or the number of elements not yet yielded in the current epoch.
* __nb_io_workers__ : The number of parallel threads to preload data. NOTE that if nb_io_workers > 1, data is loaded asynchronously.
* __nb_proc_workers__ : The number of parallel processes to do preprocessing of data using the _process_batch function. If nb_proc_workers is set to 0, no parallel processes will be launched; instead, any preprocessing will be done in the preload thread and data will have to pass through only one queue rather than two queues. NOTE that if nb_proc_workers > 1, data processing is asynchronous and data will not be yielded in the order that it is loaded!
* __sample_random__ : If True, sample the data in random order.
* __sample_with_replacement__ : If True, sample data with replacement when doing random sampling.
* __sample_weights__ : A list of relative importance weights for each element in the dataset, specifying the relative probability with which that element should be sampled, when using random sampling.
* __drop_incomplete_batches__ : If true, drops batches smaller than the batch size. If the dataset size is not divisible by the batch size, then when sampling without replacement, there is one such batch per epoch.
* __loop_forever__ : If False, stop iteration at the end of an epoch (when all data has been yielded once).
* __preprocessor__ : The preprocessor function to call on a batch. As input, takes a batch of the same arrangement as `data`.
* __rng__ : A numpy random number generator. The rng is used to determine data shuffle order and is used to uniquely seed the numpy RandomState in each parallel process (if any).

#### Methods ####

```python
flow()
```

Returns a data generator that yields minibatches.

```python
__len__()
```

Any data_flow object `obj` has a length attribute that specifies the number of minibatches it can yield with one pass over its data. This can be retrieved, for example, using `len(obj)`.

#### Examples ####

For some set of model inputs `X` and labels `Y`, iterative over batches of pairs from `X` and `Y` in shuffled order:

```python
data_gen = data_flow(data=[X, Y], sample_random=True)
num_batches = len(data_flow)
for i, batch in enumerate(data_flow.flow()):
    print("Yielded batch {} of {}".format(i+1, num_batches))
```

Arbitrary preprocessing can be done with a custom `preprocessor` function. For example all input data is divided by 255.0 in two parallel processes thus:

```python
def preproc_func(batch):
    b0, b1 = batch
    b0 /= 255.
    return b0, b1
    
data_gen = data_flow(data=[X, Y], sample_random=True,
                     preprocessor=preproc_func, nb_proc_workers=2)
num_batches = len(data_flow)
for i, batch in enumerate(data_flow.flow()):
    print("Yielded batch {} of {}".format(i+1, num_batches))
```


## Data augmentation ##

In `data_tools.data_augmentation`.

Data augmentation for 2D images using random image transformations. This code transforms input images alone or jointly input images and their corresponding output images (eg. input images and corresponding segmentation masks).

### Single image transformation ###

Transform a single input image (or input image and target image pair).

```python
def image_random_transform(x, y=None, rotation_range=0., width_shift_range=0.,
                           height_shift_range=0., shear_range=0.,
                           zoom_range=0., intensity_shift_range=0.,
                           fill_mode='nearest', cval_x=0., cval_y=0.,
                           horizontal_flip=False, vertical_flip=False,
                           spline_warp=False, warp_sigma=0.1, warp_grid_size=3,
                           crop_size=None, rng=None)
```

#### Arguments ####
* __x__ : A single 2D input image (ndim=3, channel and 2 spatial dims).
* __y__ : A single output image or mask.
* __rotation_range__ : Positive degree value, specifying the maximum amount to rotate the image in any direction about its center.
* __width_shift_range__ : Float specifying the maximum distance by which to shift the image horizontally, as a fraction of the image's width.
* __height_shift_range__ : Float specifying the maximum distance by which to shift the image vertically, as a fraction of the image's height.
* __shear_range__ : Positive degree value, specifying the maximum horizontal sheer of the image.
* __zoom_range__ : The maximum absolute deviation of the image scale from one. (I.e. zoom_range of 0.2 allows zooming the image to scales within the range [0.8, 1.2]).
* __intensity_shift_range__ : The maximum absolute value by which to shift image intensities up or down.
* __fill_mode__ : Once an image is spatially transformed, fill any empty space with the 'nearest', 'reflect',  or 'constant' strategy. Mode 'nearest' fills the space with the values of the nearest pixels; mode 'reflect' fills the space with a mirror image of the image along its nearest border or corner; 'constant' fills it with the constant value defined in `cval`.
* __cval__ : The constant value with which to fill any empty space in a transformed input image when using `fill_mode='constant'`.
* __cvalMask__ : The constant value with which to fill any empty space in a transformed target image when using `fill_mode='constant'`.
* __horizontal_flip__ : Boolean, whether to randomly flip images horizontally.
* __vertical_flip__ : Boolean, whether to randomly flip images vertically.
* __spline_warp__ : Boolean, whether to apply a b-spline nonlineary warp.
* __warp_sigma__ : Standard deviation of control point jitter in spline warp.
* __warp_grid_size__ : Integer s specifying an a grid with s by s control points.
* __crop_size__ : Tuple specifying the size of random crops taken of transformed images. Crops are always taken from within the transformed image, with no padding.
* __rng__ : A numpy random number generator.

### Image stack transformation ###

Transforms an N-dimensional stack of input images (or input image and target image pairs). Assumes the final two axes are spatial axes (not considering the channel axis).

```python
def image_stack_random_transform(x, *args, y=None, channel_axis=1, **kwargs)
```

Calls `image_random_transform`.


## Data writing ##

In `data_tools.io`.

It is typically most efficient to write data to disk in batches of a constant size -- especially when block-level compression is enabled, in which case it is efficient to write data one block at a time. The following tools buffer data chunks of any size until there is enough for to assemble at least one block and then writes all complete blocks to the target array. If the target array is a memory-mapped file (h5py, bcolz, zarr), then this data is thus written to disk (flushed to ensure writing).

### Generic buffered array writer ###

```python
class buffered_array_writer(object)
```

This is a generic base class. Given an array, data element shape, and batch size, writes data to an array batch-wise. Data can be passed in any number of elements at a time. If the array is an interface to a memory-mapped file, data is thus written batch-wise to the file.

#### Arguments ####
Class initialization uses the following arguments:

```python
def __init__(self, storage_array, data_element_shape, dtype, batch_size,
             length=None)
```

* __storage_array__ : the array to write into
* __data_element_shape__ : shape of one input element
* __batch_size__ : write the data to disk in batches of this size
* __length__ : dataset length (if None, expand it dynamically)

#### Methods ####

```python
flush_buffer()
```
Forces a write of all data remanining in the buffer to the target array.

```python
buffered_write(data)
```
Writes `data` to the target array, first passing the data through the buffer. With any call of this function, `data` can have any number of elements.

### HDF5 buffered array writer ###

```python
class h5py_array_writer(buffered_array_writer)
```

Given a data element shape and batch size, writes data to an HDF5 file batch-wise. Data can be passed in any number of elements at a time. A write is flushed only when the buffer is full, the writer is destroyed, or `flush_buffer()` is called.

#### Arguments ####
Class initialization uses the following arguments:

```python
def __init__(self, data_element_shape, dtype, batch_size, filename,
             array_name, length=None, append=False, kwargs=None)
```

* __data_element_shape__ : shape of one input element
* __batch_size__ : write the data to disk in batches of this size
* __filename__: name of file in which to store data
* __array_name__ : HDF5 array path
* __length__ : dataset length (if None, expand it dynamically)
* __append__ : write files with append mode instead of write mode
* __kwargs__ : dictionary of arguments to pass to h5py on dataset creation (if none, do lzf compression with batch_size chunk size)

#### Methods ####

```python
flush_buffer()
```
Forces a write of all data remanining in the buffer to the target array.

```python
buffered_write(data)
```
Writes `data` to the target array, first passing the data through the buffer. Can be called on `data` with any number of elements.

### Bcolz buffered array writer ###

```python
class bcolz_array_writer(buffered_array_writer)
```

Given a data element shape and batch size, writes data to a bcolz file-set batch-wise. Data can be passed in any number of elements at a time. A write is flushed only when the buffer is full, the writer is destroyed, or `flush_buffer()` is called.

#### Arguments ####
Class initialization uses the following arguments:

```python
def __init__(self, data_element_shape, dtype, batch_size, save_path,
             length=None, append=False, kwargs={})
```

* __data_element_shape__ : shape of one input element
* __batch_size__ : write the data to disk in batches of this size
* __save_path__ : directory to save array in
* __length__ : dataset length (if None, expand it dynamically)
* __append__ : write files with append mode instead of write mode
* __kwargs__ : dictionary of arguments to pass to bcolz on dataset creation (if none, do blosc compression with chunklen determined by the expected array length)

#### Methods ####

```python
flush_buffer()
```
Forces a write of all data remanining in the buffer to the target array.

```python
buffered_write(data)
```
Writes `data` to the target array, first passing the data through the buffer. With any call of this function, `data` can have any number of elements.

### Zarr buffered array writer ###

```python
class zarr_array_writer(buffered_array_writer)
```

Given a data element shape and batch size, writes data to a zarr file-set batch-wise. Data can be passed in any number of elements at a time. A write is flushed only when the buffer is full, the writer is destroyed, or `flush_buffer()` is called.

#### Arguments ####
Class initialization uses the following arguments:

```python
def __init__(self, data_element_shape, dtype, batch_size, filename,
             array_name, length=None, append=False, kwargs=None)
```

* __data_element_shape__ : shape of one input element
* __batch_size__ : write the data to disk in batches of this size
* __filename__ : name of file in which to store data
* __array_name__ : zarr array path
* __length__ : dataset length (if None, expand it dynamically)
* __append__ : write files with append mode instead of write mode
* __kwargs__ : dictionary of arguments to pass to zarr on dataset creation (if none, do blosc lz4 compression with batch_size chunk size)

#### Methods ####

```python
flush_buffer()
```
Forces a write of all data remanining in the buffer to the target array.

```python
buffered_write(data)
```
Writes `data` to the target array, first passing the data through the buffer. With any call of this function, `data` can have any number of elements.


## Patches ##

In `data_tools.patches`.

The following is some support code for generating and saving image patches.

### Patch generator ###

```python
class patch_generator(object)
```

This class creates a generator object which extract 2D patches from a slice or volume. Patches extracted at an edge are mirrored by default (else, zero-padded). Patches are always returned as float32.

#### Arguments ####
Class initialization uses the following arguments:

```python
def __init__(self, patchsize, source, binary_mask=None,
             random_order=False, mirrored=True, max_num=None)
```

* __patchsize__ : edge size of square patches to extract (scalar)
* __source__ : a slice or volume from which to extract patches
* __binary_mask__ : (optional) extract patches only where mask is True
* __random_order__ : randomize the order of patch extraction
* __mirrored__ : at source edges, mirror the patches; else, zero-pad
* __max_num__ : (optional) stop after extracting this number of patches

### Create patch dataset ###

This is a convenience function to extract patches from a stack of images (and optionally, a corresponding stack target classification masks) and save them to a memory-mapped file. For each class, one dataset/array/directory is used.

```python
def create_dataset(save_path, patchsize, volume,
                   mask=None, class_list=None, random_order=True, batchsize=32,
                   file_format='hdf5', kwargs={}, show_progress=False)
```

#### Arguments ####
* __save_path__ : directory to save dataset files/folders in
* __patchsize__ : the size of the square 2D patches
* __volume__ : the stack of input images
* __mask__ : the stack of input masks (not binary)
* __class_list__ : a list of mask values (eg. class_list[0] is the mask value for class 0)
* __random_order__ : randomize patch order
* __batchsize__ : the number of patches to write to disk at a time (affects write speed)
* __file_format__ : 'bcolz', 'hdf5'
* __kwargs__ : a dictionary of arguments to pass to the dataset_writer object corresponding to the file format
* __show_progress__ :show a progressbar.
