import time
import threading
import multiprocessing
try:
    import queue            # python 3
except ImportError:
    import Queue as queue   # python 2

import numpy as np


class data_flow(object):
    """
    Given a list of array-like objects, data from the objects is read in a
    parallel thread and processed in the same parallel thread or in a set of
    parallel processes. All objects are iterated in tandem (i.e. for a list
    data=[A, B, C], a batch of size 1 would be [A[i], B[i], C[i]] for some i).
    
    data : A list of data arrays, each of equal length. When yielding a batch, 
        each element of the batch corresponds to each array in the data list.
    batch_size : The maximum number of elements to yield from each data array
        in a batch. The actual batch size is the smallest of either this number
        or the number of elements not yet yielded in the current epoch.
    nb_io_workers : The number of parallel threads to preload data. NOTE that
        if nb_io_workers > 1, data is loaded asynchronously.
    nb_proc_workers : The number of parallel processes to do preprocessing of
        data using the _process_batch function. If nb_proc_workers is set to 0,
        no parallel processes will be launched; instead, any preprocessing will
        be done in the preload thread and data will have to pass through only
        one queue rather than two queues. NOTE that if nb_proc_workers > 1,
        data processing is asynchronous and data will not be yielded in the
        order that it is loaded!
    loop_forever : If False, stop iteration at the end of an epoch (when all
        data has been yielded once).
    sample_random : If True, sample the data in random order.
    sample_with_replacement : If True, sample data with replacement when doing
        random sampling.
    sample_weights : A list of relative importance weights for each element in
        the dataset, specifying the relative probability with which that
        element should be sampled, when using random sampling.
    drop_incomplete_batches : If true, drops batches smaller than the batch
        size. If the dataset size is not divisible by the batch size, then when
        sampling without replacement, there is one such batch per epoch.
    preprocessor : The preprocessor function to call on a batch. As input,
        takes a batch of the same arrangement as `data`.
    index_sampler : An iterator that returns array indices according
        to some sampling strategy. By default, uses wrap.index_sampler,
        initialized to do random sampling without replacement.
    rng : A numpy random number generator. The rng is used to determine data
        shuffle order and is used to uniquely seed the numpy RandomState in
        each parallel process (if any).
    """
    
    def __init__(self, data, batch_size, nb_io_workers=1, nb_proc_workers=0,
                 loop_forever=False, sample_random=False,
                 sample_with_replacement=False, sample_weights=None,
                 drop_incomplete_batches=False, preprocessor=None, rng=None):
        self.data = data
        self.batch_size = batch_size
        self.nb_io_workers = nb_io_workers
        if not nb_io_workers>0:
            raise ValueError("nb_io_workers must be 1 or more")
        self.nb_proc_workers = nb_proc_workers
        self.loop_forever = loop_forever
        self.sample_random = sample_random
        self.sample_with_replacement = sample_with_replacement
        self.sample_weights = sample_weights
        self.drop_incomplete_batches = drop_incomplete_batches
        if not sample_with_replacement and np.any(self.sample_weights==0):
            raise ValueError("When sampling without replacement, sample "
                             "weights must never be zero.")
                             
        if preprocessor is not None:
            self._process_batch = preprocessor
        else:
            self._process_batch = lambda x: x   # Do nothing by default
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng
        
        self.num_samples = len(data[0])
        for d in self.data:
            assert(len(d)==self.num_samples)
        
        self.num_batches = self.num_samples//self.batch_size
        if self.num_samples%batch_size > 0 and not drop_incomplete_batches:
            self.num_batches += 1
        
        # Multiprocessing/multithreading objects, queues, and stop event.
        self._stop = None
        self._load_queue = None
        self._proc_queue = None
        self._idx_queue = None
        self._index_thread = None
        self._process_list = []
        self._preload_list = []
            
    def __iter__(self):
        return self.flow()
        
    ''' Generate batches of processed data (output with labels) '''
    def flow(self):
        self._stop = multiprocessing.Event()
        try:
            # Create the queues.
            #   NOTE: these can become corrupt on sub-process termination,
            #   so create them in flow() and let them die with the flow().
            q_size = max(self.nb_io_workers, self.nb_proc_workers)
            self._load_queue = multiprocessing.Queue(q_size)
            if self.nb_proc_workers > 0:
                self._proc_queue = multiprocessing.Queue(q_size)
            else:
                # If there are no worker processes, alias load_queue as
                # proc_queue, allowing data to thus be yielded directly from
                # the load_queue.
                self._proc_queue = self._load_queue
            
            # Start the parallel data processing proccess(es)
            seed_base = self.rng.randint(self.nb_proc_workers, 2**16)
            for i in range(self.nb_proc_workers):
                pseed = seed_base - i
                process_thread = multiprocessing.Process( \
                    target=self._process_subroutine,
                    args=(self._load_queue, self._proc_queue,
                          self._stop, pseed))
                process_thread.daemon = True
                process_thread.start()
                self._process_list.append(process_thread)
                
            # Start the data index provider thread.
            self._idx_queue = queue.Queue(q_size)
            self._index_thread = threading.Thread( \
                target=self._index_provider,
                args=(self._idx_queue, self._stop) )
            self._index_thread.daemon = True
            self._index_thread.start()
                
            # Start the parallel loader thread.
            # (must be started AFTER processes to avoid copying it in fork())
            for i in range(self.nb_io_workers):
                preload_thread = threading.Thread( \
                    target=self._preload_subroutine,
                    args=(self._load_queue, self._idx_queue, self._stop) )
                preload_thread.daemon = True
                preload_thread.start()
                self._preload_list.append(preload_thread)
            
            # Yield batches fetched from the parallel process(es).
            nb_yielded = 0
            while not self._stop.is_set():
                try:
                    if not self.loop_forever and nb_yielded==self.num_batches:
                        self._stop.set()
                        break
                    batch = self._proc_queue.get()
                    yield batch
                    nb_yielded += 1
                except:
                    self._stop.set()
                    raise
        except:
            self._stop.set()
            raise
        finally:
            # Clean up, whether there was an exception or not.
            self._cleanup()
                
    ''' Generate the indices to for each batch of data. '''
    def _index_provider(self, idx_queue, stop):
        while not stop.is_set():
            # Initialize index sampler at start of epoch.
            sampler = iter(\
                index_sampler(array_length=self.num_samples,
                              random=self.sample_random,
                              replacement=self.sample_with_replacement,
                              weights=self.sample_weights,
                              rng=self.rng))
            
            # Loop batchwise over the dataset.
            for b in range(self.num_batches):
                bs = min(self.batch_size,
                            self.num_samples-b*self.batch_size)
                if self.drop_incomplete_batches and bs < self.batch_size:
                    continue
                try:
                    batch_indices = [next(sampler) for _ in range(bs)]
                except:
                    stop.set()
                    raise()
                put_successful = False
                while not put_successful:
                    try:
                        idx_queue.put(batch_indices, timeout=0.001)
                        put_successful = True
                    except queue.Full:
                        put_successful=False
                    except:
                        stop.set()
                        raise
                    if stop.is_set(): return
            if not self.loop_forever:
                return
            
    ''' Preload batches in the background and add them into the load_queue.
        Wait if the queue is full. '''
    def _preload_subroutine(self, load_queue, idx_queue, stop):
        while not stop.is_set():
            batch_indices = None
            while batch_indices is None:
                try:
                    batch_indices = idx_queue.get(timeout=0.001)
                except queue.Empty:
                    pass
                except:
                    stop.set()
                    raise
                if stop.is_set(): return
            # Assuming that if the user chose to have more than one loader
            # thread, data access is known to be threadsafe.
            batch = []
            for d in self.data:
                batch.append([d[int(i)] for i in batch_indices])
            if self.nb_proc_workers==0:
                # If there are no worker processes, preprocess the batch
                # in the loader thread.
                try:
                    batch = self._process_batch(batch)
                except:
                    stop.set()
                    raise
                if stop.is_set(): return
            put_successful = False
            while not put_successful:
                # Poll to allow graceful termination.
                try:
                    load_queue.put(batch, timeout=0.001)
                    put_successful = True
                except queue.Full:
                    put_successful = False
                except:
                    stop.set()
                    raise
                if stop.is_set(): return
                
    ''' Process any loaded batches in the load queue and add them to the
        processed queue -- these are ready to yield. '''
    def _process_subroutine(self, load_queue, proc_queue, stop, seed):
        np.random.seed(seed)
        try:
            while not stop.is_set():
                batch = None
                while batch is None:
                    # Poll to allow graceful termination.
                    try:
                        batch = load_queue.get(timeout=0.001)
                    except queue.Empty:
                        pass
                    except:
                        stop.set()
                        raise
                    if stop.is_set(): return
                try:
                    batch_processed = self._process_batch(batch)
                except:
                    stop.set()
                    raise
                if stop.is_set(): return
                put_successful = False
                while not put_successful:
                    # Poll to allow graceful termination.
                    try:
                        proc_queue.put(batch_processed, timeout=0.001)
                        put_successful = True
                    except queue.Full:
                        put_successful = False
                    except:
                        stop.set()
                        raise
                    if stop.is_set(): return
        except:
            stop.set()
            load_queue.cancel_join_thread()
            proc_queue.cancel_join_thread()
            raise
        
    ''' Set termination event, wait for all threads and processes to exit,
        then close queues. '''
    def _cleanup(self):
        if self._stop is not None:
            self._stop.set()
        if self._index_thread is not None:
            self._index_thread.join()
        for thread in self._preload_list:
            thread.join()
        for process in self._process_list:
            process.join()
        if self.nb_proc_workers and self._proc_queue is not None:
            # If nb_proc_workers==0, proc_queue is just an alias to
            # load_queue
            self._proc_queue.close()
        if self._load_queue is not None:
            self._load_queue.close()
        
        # Clear
        self._stop = None
        self._load_queue = None
        self._proc_queue = None
        self._idx_queue = None
        self._index_thread = None
        self._process_list = []
        self._preload_list = []
        
    def __del__(self):
        self._cleanup()
            
    def __len__(self):
        return self.num_batches
        
        
class index_sampler(object):
    """
    An iterable that generates array indices according to some sampling
    strategy.
    
    array_length : the length of the array to sample from - indicies are
        generated in the range [0, array_length-1].
    random : sample in random order if True.
    replacement : when doing random sampling, sample with replacement if True;
        when this is active, the iterator never stops iterating since it never
        runs out of elements to sample.
    weights : a list of relative importance weights for every index; when 
        normalized, these determine the probability for each element of being
        sampled.
    rng : random number generator
    """
    def __init__(self, array_length, random=True, replacement=False,
                 weights=None, rng=None):
        self.array_length = array_length
        self.random = random
        self.replacement = replacement
        self.weights = weights
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng
        
    def __iter__(self):
        for idx in self._gen_idx():
            yield idx
            
    def _gen_idx(self):   
        if self.random:
            normalized_weights = None
            if self.weights is not None:
                normalized_weights = np.array(self.weights) \
                                     / float(np.sum(self.weights))
            indices = self.rng.choice(range(self.array_length),
                                      size=self.array_length,
                                      replace=self.replacement,
                                      p=normalized_weights)
        else:
            indices = list(range(self.array_length))
        for idx in indices:
            yield idx


class buffered_array_writer(object):
    """
    Given an array, data element shape, and batch size, writes data to an array
    batch-wise. Data can be passed in any number of elements at a time.
    If the array is an interface to a memory-mapped file, data is thus written
    batch-wise to the file.
    
    INPUTS
    storage_array      : the array to write into
    data_element_shape : shape of one input element
    dtype              : numpy data type or data type as a string
    batch_size         : write the data to disk in batches of this size
    length             : dataset length (if None, expand it dynamically)
    """
    
    def __init__(self, storage_array, data_element_shape, dtype, batch_size,
                 length=None):
        self.storage_array = storage_array
        self.data_element_shape = data_element_shape
        self.dtype = dtype
        self.batch_size = batch_size
        self.length = length
        
        self.buffer = np.zeros((batch_size,)+data_element_shape, dtype=dtype)
        self.buffer_ptr = 0
        self.storage_array_ptr = 0
        
    ''' Flush the buffer. '''
    def flush_buffer(self):
        if self.buffer_ptr > 0:
            end = self.storage_array_ptr+self.buffer_ptr
            self.storage_array[self.storage_array_ptr:end] = \
                                                  self.buffer[:self.buffer_ptr]
            self.storage_array_ptr += self.buffer_ptr
            self.buffer_ptr = 0
            
    '''
    Write data to file one buffer-full at a time. Note: data is not written
    until buffer is full.
    '''
    def buffered_write(self, data):
        # Verify data shape 
        if np.shape(data) != self.data_element_shape \
                             and np.shape(data)[1:] != self.data_element_shape:
            raise ValueError("Error: input data has the wrong shape.")
        if np.shape(data) == self.data_element_shape:
            data_len = 1
        elif np.shape(data)[1:] == self.data_element_shape:
            data_len = len(data)
            
        # Stop when data length exceeded
        if self.length is not None and self.length==self.storage_array_ptr:
            raise EOFError("Write aborted: length of input data exceeds "
                           "remaining space.")
            
        # Verify data type
        if data.dtype != self.dtype:
            raise TypeError("Specified dtype '{}' but data has dtype '{}'."
                            "".format(self.dtype, data.dtype))
            
        # Buffer/write
        if data_len == 1:
            data = [data]
        for d in data:
            self.buffer[self.buffer_ptr] = d
            self.buffer_ptr += 1
            
            # Flush buffer when full
            if self.buffer_ptr==self.batch_size:
                self.flush_buffer()
                
        # Flush the buffer when 'length' reached
        if self.length is not None \
                       and self.storage_array_ptr+self.buffer_ptr==self.length:
            self.flush_buffer()
            
    def __len__(self):
        num_elements = len(self.storage_array)+self.buffer_ptr
        return num_elements
            
    def get_shape(self):
        return (len(self),)+self.data_element_shape
    
    def get_element_shape(self):
        return self.data_element_shape
    
    def get_array(self):
        return self.storage_array
        
    def __del__(self):
        self.flush_buffer()


class h5py_array_writer(buffered_array_writer):
    """
    Given a data element shape and batch size, writes data to an HDF5 file
    batch-wise. Data can be passed in any number of elements at a time.
    
    INPUTS
    data_element_shape : shape of one input element
    dtype              : numpy data type or data type as a string
    batch_size         : write the data to disk in batches of this size
    filename           : name of file in which to store data
    array_name         : HDF5 array path
    length             : dataset length (if None, expand it dynamically)
    append             : write files with append mode instead of write mode
    kwargs             : dictionary of arguments to pass to h5py on dataset
                         creation (if none, do lzf compression with
                         batch_size chunk size)
    """
    
    def __init__(self, data_element_shape, dtype, batch_size, filename,
                 array_name, length=None, append=False, kwargs=None):
        import h5py
        super(h5py_array_writer, self).__init__(None, data_element_shape,
                                                dtype, batch_size, length)
        self.filename = filename
        self.array_name = array_name
        self.kwargs = kwargs
        
        # Set up array kwargs
        self.arr_kwargs = {'chunks': (batch_size,)+data_element_shape,
                           'compression': 'lzf',
                           'dtype': dtype}
        if kwargs is not None:
            self.arr_kwargs.update(kwargs)
    
        # Open the file for writing.
        self.file = None
        if append:
            self.write_mode = 'a'
        else:
            self.write_mode = 'w'
        try:
            self.file = h5py.File(filename, self.write_mode)
        except:
            print("Error: failed to open file %s" % filename)
            raise
        
        # Open an array interface (check if the array exists; if not, create it)
        if self.length is None:
            ds_args = (self.array_name, (1,)+self.data_element_shape)
        else:
            ds_args = (self.array_name, (self.length,)+self.data_element_shape)
        try:
            self.storage_array = self.file[self.array_name]
            self.storage_array_ptr = len(self.storage_array)
        except KeyError:
            self.storage_array = self.file.create_dataset( *ds_args,
                               maxshape=(self.length,)+self.data_element_shape,
                               **self.arr_kwargs )
            self.storage_array_ptr = 0
            
    ''' Flush the buffer. Resize the dataset, if needed. '''
    def flush_buffer(self):
        if self.buffer_ptr > 0:
            end = self.storage_array_ptr+self.buffer_ptr
            if self.length is None:
                self.storage_array.resize( (end,)+self.data_element_shape )
            self.storage_array[self.storage_array_ptr:end] = \
                                                  self.buffer[:self.buffer_ptr]
            self.storage_array_ptr += self.buffer_ptr
            self.buffer_ptr = 0
    
    ''' Flush remaining data in the buffer to file and close the file. '''
    def __del__(self):
        self.flush_buffer()
        if self.file is not None:
            self.file.close() 


class bcolz_array_writer(buffered_array_writer):
    """
    Given a data element shape and batch size, writes data to a bcolz file-set
    batch-wise. Data can be passed in any number of elements at a time.
    
    INPUTS
    data_element_shape : shape of one input element
    batch_size         : write the data to disk in batches of this size
    save_path          : directory to save array in
    length             : dataset length (if None, expand it dynamically)
    append             : write files with append mode instead of write mode
    kwargs             : dictionary of arguments to pass to bcolz on dataset 
                         creation (if none, do blosc compression with chunklen
                         determined by the expected array length)
    """
    
    def __init__(self, data_element_shape, dtype, batch_size, save_path,
                 length=None, append=False, kwargs={}):
        import bcolz
        super(bcolz_array_writer, self).__init__(None, data_element_shape,
                                                 dtype, batch_size, length)
        self.save_path = save_path
        self.kwargs = kwargs
        
        # Set up array kwargs
        self.arr_kwargs = {'expectedlen': length,
                           'cparams': bcolz.cparams(clevel=5,
                                                    shuffle=True,
                                                    cname='blosclz'),
                           'dtype': dtype,
                           'rootdir': save_path}
        if kwargs is not None:
            self.arr_kwargs.update(kwargs)
    
        # Create the file-backed array, open for writing.
        # (check if the array exists; if not, create it)
        if append:
            try:
                self.storage_array = bcolz.open(self.save_path, mode='a')
                self.storage_array_ptr = len(self.storage_array)
            except FileNotFoundError:
                append=False
        if not append:
            try:
                self.storage_array = bcolz.zeros(shape=(0,)+data_element_shape,
                                                 mode='w',
                                                 **self.arr_kwargs)
                self.storage_array_ptr = 0
            except:
                print("Error: failed to create file-backed bcolz storage "
                      "array.")
                raise
            
    ''' Flush the buffer. '''
    def flush_buffer(self):
        if self.buffer_ptr > 0:
            self.storage_array.append(self.buffer[:self.buffer_ptr])
            self.storage_array.flush()
            self.storage_array_ptr += self.buffer_ptr
            self.buffer_ptr = 0


class zarr_array_writer(buffered_array_writer):
    """
    Given a data element shape and batch size, writes data to a zarr file
    batch-wise. Data can be passed in any number of elements at a time.
    
    INPUTS
    data_element_shape : shape of one input element
    batch_size         : write the data to disk in batches of this size
    filename           : name of file in which to store data
    array_name         : zarr array path
    length             : dataset length (if None, expand it dynamically)
    append             : write files with append mode instead of write mode
    kwargs             : dictionary of arguments to pass to zarr on dataset
                         creation (if none, do blosc lz4 compression with
                         batch_size chunk size)
    """
    
    def __init__(self, data_element_shape, dtype, batch_size, filename,
                 array_name, length=None, append=False, kwargs=None):
        import zarr
        super(zarr_array_writer, self).__init__(None, data_element_shape,
                                                dtype, batch_size, length)
        self.filename = filename
        self.array_name = array_name
        self.kwargs = kwargs
        
        # Set up array kwargs
        self.arr_kwargs = {'name': array_name,
                           'chunks': (batch_size,)+data_element_shape,
                           'compressor': zarr.Blosc(cname='lz4',
                                                    clevel=5,
                                                    shuffle=1),
                           'dtype': dtype}
        if self.length is None:
            self.arr_kwargs['shape'] = (1,)+self.data_element_shape
        else:
            self.arr_kwargs['shape'] = (self.length,)+self.data_element_shape
        if kwargs is not None:
            self.arr_kwargs.update(kwargs)
    
        # Open the file for writing.
        self.group = None
        if append:
            self.write_mode = 'a'
        else:
            self.write_mode = 'w'
        try:
            self.group = zarr.open_group(filename, self.write_mode)
        except:
            print("Error: failed to open file %s" % filename)
            raise
        
        # Open an array interface (check if the array exists; if not, create it)
        if self.length is None:
            ds_args = (self.array_name, (1,)+self.data_element_shape)
        else:
            ds_args = (self.array_name, (self.length,)+self.data_element_shape)
        try:
            self.storage_array = self.group[self.array_name]
            self.storage_array_ptr = len(self.storage_array)
        except KeyError:
            self.storage_array = self.group.create_dataset(**self.arr_kwargs)
            self.storage_array_ptr = 0
            
    ''' Flush the buffer. Resize the dataset, if needed. '''
    def flush_buffer(self):
        if self.buffer_ptr > 0:
            end = self.storage_array_ptr+self.buffer_ptr
            if self.length is None:
                self.storage_array.resize( (end,)+self.data_element_shape )
            self.storage_array[self.storage_array_ptr:end] = \
                                                  self.buffer[:self.buffer_ptr]
            self.storage_array_ptr += self.buffer_ptr
            self.buffer_ptr = 0
    
    ''' Flush remaining data in the buffer to file and close the file. '''
    def __del__(self):
        self.flush_buffer()
        # Zarr automatically flushes all modifications and does not expose
        # the file handle so the file is not closed in this destructor.
