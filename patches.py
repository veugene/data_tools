import numpy as np
import h5py
from .io import (h5py_array_writer,
                 bcolz_array_writer)


class patch_generator(object):
    """
    Extract 2D patches from a slice or volume. Patches extracted at an edge are
    mirrored by default (else, zero-padded).
    Patches are always returned as float32.
    
    patchsize    : edge size of square patches to extract (scalar)
    source       : a slice or volume from which to extract patches
    binary_mask  : (optional) extract patches only where mask is True
    random_order : randomize the order of patch extraction
    mirrored     : at source edges, mirror the patches; else, zero-pad
    max_num      : (optional) stop after extracting this number of patches
    """
    
    def __init__(self, patchsize, source, binary_mask=None,
                 random_order=False, mirrored=True, max_num=None):
        self.patchsize = patchsize
        self.source = source.astype(np.float32)
        self.mask = binary_mask
        self.random_order = random_order
        self.mirrored = mirrored
        self.max_num = max_num
        
        if len(self.source.shape)==2:
            self.source = self.source[:,:,np.newaxis]
        if self.mask is not None and len(self.mask.shape)==2:
            self.mask = self.mask[:,:,np.newaxis]
            
        if self.mask is not None:
            self.num_patches = (self.mask>0).sum()
        else:
            self.num_patches = np.product(self.source.shape)
        
    def __iter__(self):
        # Create mirror edges and corners (or zero-padding) about image
        new_shape = ( self.source.shape[0]+self.patchsize,
                      self.source.shape[1]+self.patchsize,
                      self.source.shape[2] )
        I = np.zeros(new_shape, dtype=np.float32)
        d1 = self.patchsize//2
        d2 = d1+self.patchsize%2   # in case the patchsize is odd
        I[d1:-d2, d1:-d2, :] = self.source
        if self.mirrored:
            _h = np.fliplr
            _v = np.flipud         
            I[:d1,    d1:-d2, :]=_v(self.source[:d1,  :,    :]) # left
            I[-d2:,   d1:-d2, :]=_v(self.source[-d2:, :,    :]) # right
            I[d1:-d2, :d1,    :]=_h(self.source[:,    :d1,  :]) # top
            I[d1:-d2, -d2:,   :]=_h(self.source[:,    -d2:, :]) # bottom
            I[:d1,    :d1,    :]=_h(_v(self.source[:d1,  :d1,  :])) # top-left
            I[-d2:,   :d1,    :]=_h(_v(self.source[-d2:, :d1,  :])) # top-right
            I[:d1,    -d2:,   :]=_h(_v(self.source[:d1,  -d2:, :])) # bot-left
            I[-d2:,   -d2:,   :]=_h(_v(self.source[-d2:, -d2:, :])) # bot-right
        
        # Extract patches at points in mask (or all points if there is no mask)
        indices = None
        if self.mask is not None:
            indices = np.where(self.mask)
        else:
            indices = np.where(np.ones(self.source.shape, dtype=np.bool))
        num_indices = len(indices[0])
        if self.random_order:
            index_order = np.random.permutation(num_indices)
        else:
            index_order = range(num_indices)
        for i in index_order:
            kp = (indices[0][i], indices[1][i], indices[2][i])
            patch = np.zeros((self.patchsize, self.patchsize),
                             dtype=np.float32)
            patch[:,:] = I[kp[0]:kp[0]+self.patchsize,
                           kp[1]:kp[1]+self.patchsize,
                           kp[2]]

            yield patch
            
    def __len__(self):
        return self.num_patches
            
            
def create_dataset(save_path, patchsize, volume,
                   mask=None, class_list=None, random_order=True, batchsize=32,
                   file_format='hdf5', kwargs={}, show_progress=False):    
    """
    Extract patches and save them to file, with one dataset/array/directory
    per class.
    
    save_path    : directory to save dataset files/folders in
    patchsize    : the size of the square 2D patches
    volume       : the stack of input images
    mask         : the stack of input masks (not binary)
    class_list   : a list of mask values (eg. class_list[0] is the mask value
                   for class 0)
    random_order : randomize patch order
    batchsize    : the number of patches to write to disk at a time (affects
                   write speed)
    file_format  : 'bcolz', 'hdf5'
    kwargs       : a dictionary of arguments to pass to the dataset_writer
                   object corresponding to the file format
    """

    if file_format=='hdf5':
        hdf5_file = h5py.File( save_path, 'w' )
    
    for c in class_list:
        piter_kwargs = {'patchsize': patchsize,
                        'source': volume,
                        'random_order': random_order}
        if mask is not None and class_list is not None:
            binary_mask = mask==c
            piter_kwargs['binary_mask'] = binary_mask
        piter = patch_generator(**piter_kwargs)
        
        if show_progress:
            import progressbar
            print("Working on class %d" % c)
            bar = progressbar.ProgressBar(maxval=len(piter)).start()
        
        if file_format=='hdf5':
            dataset_writer = \
                h5py_array_writer(data_element_shape=(1,patchsize,patchsize),
                                  dtype=np.float32,
                                  batch_size=batchsize,
                                  filename=save_path,
                                  array_name="class_"+str(c),
                                  length=len(piter),
                                  append=True,
                                  kwargs=kwargs )
        elif file_format=='bcolz':
            c_savepath = os.path.join(save_path, "class_"+str(c))
            if not os.path.exists(c_savepath):
                os.makedirs(c_savepath)
            dataset_writer = \
                bcolz_array_writer(data_element_shape=(1,patchsize,patchsize),
                                   dtype=np.float32,
                                   batch_size=batchsize,
                                   save_path=c_savepath,
                                   length=len(piter),
                                   kwargs=kwargs )

        else:
            raise ValueError("Error: unknown file format \'{}\'"
                             "".format(str(file_format)))
        
        for patch in piter:
            dataset_writer.buffered_write(patch[np.newaxis])
            if show_progress:
                bar.update(bar.currval+1)
        
        if show_progress:
            bar.finish()
 
