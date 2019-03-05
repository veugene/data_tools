import warnings
import numpy as np


class delayed_view(object):
    """
    Given an array, create a view into that array without preloading the viewed
    data into memory. Data is loaded as needed when indexing into the
    delayed_view.
    
    Indexing is numpy-style, using any combination of integers, slices, index
    lists, ellipsis (only one, as with numpy), and boolean arrays but not 
    non-boolean multi-dimensional arrays. Note that the indexing style is also
    used on the underlying data sources so those data sources must support the
    style of indexing used with a multi_source_array object; use simple
    indexing with integers and slices (eg. obj[0,3:10]) when unsure.
    
    Adding dimensions to the output just by indexing is not supported. This
    means that unlike with numpy, indexing cannot be done with `None` or
    `numpy.newaxis`; also, for example, an array A with shape (4,5) can be
    indexed as A[[0,1]] and A[[[0,1]]] (these are equivalent) but not as
    A[[[[0,1]]]] for which numpy would add a dimension to the output.
    
    arr     : the source array
    shuffle : randomize data access order within the view
    idx_min : the view into arr starts at this index
    idx_max : the view into arr ends before this index
    rng     : numpy random number generator
    """
    
    def __init__(self, arr, shuffle=False, idx_min=None, idx_max=None,
                 rng=None):
        self.arr = arr
        self.shuffle = shuffle
        self.idx_min = idx_min
        if idx_min is None:
            self.idx_min = 0
        self.idx_max = idx_max
        if idx_max is None:
            self.idx_max = len(self.arr)
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.num_items = min(self.idx_max, len(arr))-self.idx_min
        assert(self.num_items >= 0)
        self.dtype = self.arr.dtype
        try:
            self.shape = arr.shape
        except AttributeError:
            self.shape = (len(arr),)+np.shape(arr[0])
        self.ndim = len(self.shape)
            
        # Create index list
        self.arr_indices = np.arange(self.idx_min, min(self.idx_max, len(arr)))
        if self.shuffle:
            self.rng.shuffle(self.arr_indices)
            
    def re_shuffle(self, random_seed=None):
        rng = self.rng
        if random_seed is not None:
            rng = np.random.RandomState(random_seed)
        rng.shuffle(self.arr_indices)
    
    def __iter__(self):
        for idx in self.arr_indices:
            idx = int(idx)  # Some libraries don't like np.integer
            yield self.arr[idx]
            
    def _get_element(self, int_key, key_remainder=None):
        if not isinstance(int_key, (int, np.integer)):
            raise IndexError("cannot index with {}".format(type(int_key)))
        idx = self.arr_indices[int_key]
        if key_remainder is not None:
            idx = (idx,)+key_remainder
        idx = int(idx)  # Some libraries don't like np.integer
        return self.arr[idx]
    
    def _get_block(self, values, key_remainder=None):
        item_block = None
        for i, v in enumerate(values):
            # Lists in the aggregate key index in tandem;
            # so, index into those lists (the first list is `values`)
            v_key_remainder = key_remainder
            if isinstance(values, tuple) or isinstance(values, list):
                if key_remainder is not None:
                    broadcasted_key_remainder = ()
                    for k in key_remainder:
                        if hasattr(k, '__len__') and len(k)==np.size(k):
                            broadcasted_key_remainder += (k[i],)
                        else:
                            broadcasted_key_remainder += (k,)
                    v_key_remainder = broadcasted_key_remainder
            
            # Make a single read at an integer index of axis 0
            elem = self._get_element(v, v_key_remainder)
            if item_block is None:
                item_block = np.zeros((len(values),)+elem.shape,
                                      self.dtype)
            item_block[i] = elem
        return item_block
                
    def __getitem__(self, key):
        item = None
        key_remainder = None
        
        # Grab the key for the first dimension, store the remainder
        if hasattr(key, '__len__'):
            if isinstance(key, np.ndarray):
                if key.dtype == np.bool:
                    if key.ndim != self.ndim:
                        raise IndexError("not enough indices, given a boolean "
                                         "index array with shape "
                                         "{}".format(np.shape(key)))
                    key = key.nonzero()
                elif key.ndim > 1:
                    raise IndexError("indexing by non-boolean multidimensional"
                                     " arrays not supported")
                
            # If there are lists in the key, make sure they have the same shape
            key_shapes = []
            for k in key:
                if hasattr(k, '__len__'):
                    key_shapes.append(np.shape(k))
            for s in key_shapes:
                if s!=key_shapes[0]:
                    raise IndexError("shape mismatch: indexing arrays could "
                                     "not be broadcast together with shapes "
                                     ""+" ".join([str(s) for s in key_shapes]))
            if len(key_shapes) > self.ndim:
                # More sublists/subtuples than dimensions in the array
                raise IndexError("too many indices for array")
            
            # If there are iterables in the key, or if the key is a tuple, then
            # each key index corresponds to a separate data dimension (as per
            # Numpy). Otherwise, such as when the key is a list of integers,
            # each index corresponds only to the first data dimension.
            key_remainder = None
            if len(key_shapes) or isinstance(key, tuple):
                key_remainder = tuple(key[1:])
                key = key[0]
            
        # Handle ellipsis
        if key is Ellipsis:
            key = slice(0, self.num_items)
            if key_remainder is not None and len(key_remainder) < self.ndim-1:
                key_remainder = (Ellipsis,)+key_remainder
                        
        # At this point the `key` is only for the first dimension and any keys
        # for other dimensions that may have been passed are in key_remainder
        if isinstance(key, (int, np.integer)):
            item = self._get_element(key, key_remainder)
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.num_items
            stop = min(stop, self.num_items)
            step = key.step if key.step is not None else 1
            item = self._get_block(range(start, stop, step), key_remainder)
        elif hasattr(key, '__len__'):
            item = self._get_block(key, key_remainder)
        else:
            raise IndexError("cannot index with {}".format(type(key)))
        
        return item
        
    def __len__(self):
        return self.num_items


class multi_source_array(delayed_view):
    """
    Given a list of sources, create an array-like interface that combines the
    sources. This object allows slicing and iterating over the elements. Data
    access automatically spans all data sources.
    
    Indexing is numpy-style with the exeption of indexing using non-boolean
    multi-dimensional arrays, as detailed in wrap.delayed_view.
    
    source_list : list of sources to combine into one source
    class_list  : specifies class number for each source; same length as
        source_list
    shuffle     : randomize data access order within and across all sources
    maxlen      : the maximum number of elements to take from each source; if
        shuffle is False, a source is accessed as source[0:maxlen] and if
        shuffle is True, a source is accessed as shuffle(source)[0:maxlen]
    rng         : numpy random number generator
    """
    
    def __init__(self, source_list, class_list=None, shuffle=False,
                 maxlen=None, rng=None):
        self.source_list = source_list
        self.class_list = class_list
        self.shuffle = shuffle
        self.maxlen = maxlen
        if self.maxlen == None:
            self.maxlen = np.inf
        self.num_items = 0
        for source in source_list:
            self.num_items += min(len(source), self.maxlen)
            
        # Ensure that all the data sources contain elements of the same shape
        # and data type
        self.dtype = self.source_list[0].dtype
        self.shape = None
        for i, source in enumerate(source_list):
            try:
                shape = source.shape
            except AttributeError:
                shape = len(source)+np.shape(source[0])
            if self.shape is None:
                self.shape = (self.num_items,)+shape[1:]
            if self.shape[1:]!=shape[1:]:
                # In order, match all dimensions with the same shape, until
                # a match is not found.
                new_shape = self.shape
                for i in range(1, max(min(len(self.shape), len(shape)), 1)):
                    if self.shape[1:i]==shape[1:i]:
                        new_shape = self.shape[:i]
                self.shape = new_shape
            if source.dtype != self.dtype:
                self.dtype = None   # Cannot determine dtype.
        self.ndim = len(self.shape)
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
            
        # Index the data sources
        self.index_pairs = []
        for i, source in enumerate(self.source_list):
            source_indices = np.arange(len(source))
            if self.shuffle:
                self.rng.shuffle(source_indices)
            source_indices = source_indices[:min(len(source), self.maxlen)]
            for j in source_indices:
                self.index_pairs.append((i, j))
        if self.shuffle==True:
            self.rng.shuffle(self.index_pairs)
            
    def re_shuffle(self, random_seed=None):
        rng = self.rng
        if random_seed is not None:
            rng = np.random.RandomState(random_seed)
        rng.shuffle(self.index_pairs)
    
    def get_labels(self):
        labels = []
        for p in self.index_pairs:
            if not self.class_list:
                labels.append(p[0])
            else:
                labels.append(self.class_list[ p[0] ])
        return labels
    
    def __iter__(self):
        for source_num, idx in self.index_pairs:
            yield self.source_list[source_num][idx]
            
    def _get_element(self, int_key, key_remainder=None):
        if not isinstance(int_key, (int, np.integer)):
            raise IndexError("cannot index with {}".format(type(int_key)))
        source_num, idx = self.index_pairs[int_key]
        if key_remainder is not None:
            idx = (idx,)+key_remainder
        idx = int(idx)  # Some libraries don't like np.integer
        return self.source_list[source_num][idx]
