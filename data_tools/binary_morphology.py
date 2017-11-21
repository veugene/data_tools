import numpy as np
import multiprocessing
import time

def binary_dilation(input_image, spacing, radius, nb_workers=None):
    return binary_operation(input_image, spacing, radius, 'dilation',
                            nb_workers)

def binary_erosion(input_image, spacing, radius, nb_workers=None):
    return binary_operation(input_image, spacing, radius, 'erosion',
                            nb_workers)

def binary_opening(input_image, spacing, radius, nb_workers=None):
    return binary_operation(input_image, spacing, radius, 'opening',
                            nb_workers)

def binary_closing(input_image, spacing, radius, nb_workers=None):
    t = binary_operation(input_image, spacing, radius, 'dilation',
                         nb_workers)
    return binary_operation(t, spacing, radius, 'erosion', nb_workers)

def binary_operation(input_image, spacing, radius, operation,
                     flat_struct=False, nb_workers=None):
    """
    Binary operations in physical unit space.
    
    The scipy implementation of binary_dilation is EXTREMELY slow for large
    structuring elements (fast for small). This here is a naive implementation
    using numpy that is MUCH faster. Also implemented are erosion and opening.
    """
    
    input_is_flat = False
    if len(input_image.shape)==2:
        input_is_flat = True
        input_image = input_image[:,:,np.newaxis]
        spacing = [spacing[0], spacing[1], 1]

    # Create a structuring element for erosion
    # (A boolean voxel-space array defining a ball in physical space)
    s_xdim = int(float(radius)/spacing[0])*2+1
    s_ydim = int(float(radius)/spacing[1])*2+1
    if input_is_flat or flat_struct:
        s_zdim = 1
    else:
        s_zdim = int(float(radius)/spacing[2])*2+1
    structure = np.zeros((s_xdim,s_ydim,s_zdim), dtype=np.bool)
    cp = np.array([s_xdim//2, s_ydim//2, s_zdim//2])    # centerpoint
    for i in range(cp[0]+1):
        for j in range(cp[1]+1):
            for k in range(cp[2]+1):
                pos = np.abs(np.array([i,j,k]) - cp)*spacing
                r = np.sum(pos**2)
                if r <= radius**2:
                    structure[i,         j,         k        ] = True
                    structure[2*cp[0]-i, j,         k        ] = True
                    structure[i,         2*cp[1]-j, k        ] = True
                    structure[i,         j,         2*cp[2]-k] = True
                    structure[2*cp[0]-i, 2*cp[1]-j, k        ] = True
                    structure[2*cp[0]-i, j,         2*cp[2]-k] = True
                    structure[i,         2*cp[1]-j, 2*cp[2]-k] = True
                    structure[2*cp[0]-i, 2*cp[1]-j, 2*cp[2]-k] = True

    # Define the operations (at a given position)
    def operate_on_point(point, input_image, output_image):
        # Index in mask to where to start copying structure.
        start_idx = np.zeros(3, dtype=np.int32)
        start_idx[0] = point[0]-(s_xdim-1)//2
        start_idx[1] = point[1]-(s_ydim-1)//2
        start_idx[2] = point[2]-(s_zdim-1)//2

        # Describes how far out of bounds each index is in either direction.
        overlap = np.zeros((3,2), dtype=np.int32)
        overlap[:,0] = np.abs(start_idx*(start_idx<0))
        overlap[:,1] = start_idx + structure.shape - output_image.shape
        overlap[:,1] *= overlap[:,1]>0

        # Set negative start indices to zero.
        start_idx *= start_idx>0

        # A view into the structuring element showing what fits in the image.
        struct_view = structure[overlap[0,0]:s_xdim-overlap[0,1],
                                overlap[1,0]:s_ydim-overlap[1,1],
                                overlap[2,0]:s_zdim-overlap[2,1]]

        # A view into the output image showing the structuring element's FOV.
        im_view = \
        output_image[start_idx[0]:start_idx[0]+s_xdim-overlap[0,1]-overlap[0,0],
                     start_idx[1]:start_idx[1]+s_ydim-overlap[1,1]-overlap[1,0],
                     start_idx[2]:start_idx[2]+s_zdim-overlap[2,1]-overlap[2,0]]
        
        # A view into the input image showing the structuring element's FOV.
        im_view_input = \
        input_image[start_idx[0]:start_idx[0]+s_xdim-overlap[0,1]-overlap[0,0],
                    start_idx[1]:start_idx[1]+s_ydim-overlap[1,1]-overlap[1,0],
                    start_idx[2]:start_idx[2]+s_zdim-overlap[2,1]-overlap[2,0]]
        
        if operation=='dilation':
            im_view += struct_view
            
        elif operation=='erosion':
            if np.all(im_view_input*struct_view == struct_view):
                half_idx = np.array(im_view.shape)//2
                im_view[half_idx[0], half_idx[1], half_idx[2]] = 1
            
        elif operation=='opening':
            if np.all(im_view_input*struct_view == struct_view):
                im_view += struct_view
                
    # Avoid redundant computation: apply structuring element to an empty mask 
    # at every point where it fits completely in the input mask.
    loc = np.where( input_image )
                
    # Set up the function for multiprocessing
    def process_points(point_list, queue):
        output_image = np.zeros(input_image.shape, dtype=np.bool)
        for point in point_list:
            operate_on_point(point,
                             input_image=input_image,
                             output_image=output_image)
        queue.put(output_image)
    
    # Set up processes
    if nb_workers is None:
        nb_workers = multiprocessing.cpu_count()
    process_list = []
    queue_list = []
    all_points = list(zip(*loc))
    for p in range(nb_workers):
        start = (len(all_points) // nb_workers) * p
        end = (len(all_points) // nb_workers) * (p+1)
        if p==nb_workers:
            end = len(all_points)
        point_list = all_points[start:end]
        queue = multiprocessing.Queue(1)
        process = multiprocessing.Process(target=process_points,
                                          args=(point_list, queue))
        process.daemon = True
        process.start()
        process_list.append(process)
        queue_list.append(queue)
        
    # Wait for processes
    output_image = None
    try:
        for queue in queue_list:
            out = queue.get()
            if output_image is None:
                output_image = out.astype(np.bool)
            else:
                output_image += out.astype(np.bool)
    except:
        raise
    finally:
        for process in process_list:
            if process.is_alive():
                process.terminate()
        for queue in queue_list:
            queue.close()
            
    if input_is_flat:
        output_image = output_image[:,:,0]
        
    return output_image
