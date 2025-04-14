import numpy as np

def ReadRaw(fname, type=0):
    #Read .raw from EMPAD standard format into numpy array
    with open(fname, 'rb') as file:
        dp = np.fromfile(file, np.float32)
    if type == 0:
        sqpix = dp.size/128/130  ##total slice
        pix = int(sqpix**(0.5))  ## scan steps (x,y) are equal

        dp = np.reshape(dp, (pix, pix, 130, 128), order = 'C')
        dp = dp[:, :, 2:126, 2:126]
        where_are_NaNs = np.isnan(dp)   ## look for NAN and set as zero
        dp[where_are_NaNs] = 0
    #    dp = dp[:,:,0:128,:]
        print("dp shape: ", dp.shape)
    elif type == 1:
        sqpix = dp.size/124/124  ##total slice
        pix = int(sqpix**(0.5))  ## scan steps (x,y) are equal

        dp = np.reshape(dp, (pix, pix, 124, 124), order = 'C')
        dp = dp[:, :, :,:]
        where_are_NaNs = np.isnan(dp)   ## look for NAN and set as zero
        dp[where_are_NaNs] = 0
    #    dp = dp[:,:,0:128,:]
        print("dp shape: ", dp.shape)
    file.close()
    
    #Processes out low spurious counts and sets floor at just above zero to avoid errors when taking the logarithm
    #dp = dp*np.array(dp>20, np.int)
    
    dp[dp < 0] = 0
    dp = np.clip(dp, 1e-10, None) ##(min, max)
    return dp



def ApertureSum(matrix, center_x, center_y, radius=0, bkg=(0,0)):
    '''
    Aperture VDF: sums the counts within a circular aperture of a given radius around a given center.
    Args:
        matrix: 4D numpy array
        center_x: int
        center_y: int
        radius: int
    Returns:
        pixel_sum: 2D numpy array
        
        Note: Leaving radius as 0 is equivalent to a point VDF.'''
    #Aperture VDF
    pixel_sum = np.zeros((matrix.shape[0], matrix.shape[1]))
    rows, cols = len(matrix[2]), len(matrix[3])
    count = 0

    for i in range(center_x - radius, center_x + radius + 1):
        for j in range(center_y - radius, center_y + radius + 1):
            if 0 <= i < rows and 0 <= j < cols:
                pixel_sum += matrix[:,:,i,j]
                count += 1
    pixel_sum = pixel_sum
    print("Sum:", count)
    if bkg != (0,0):
        pixel_sum -= AnnularSum(matrix, center_x, center_y, bkg[0], bkg[1])*count
    
    pixel_sum[pixel_sum<0] = 0

    return pixel_sum

def ApertureSumVariance(matrix, center_x, center_y, g=1,m=1,VarB=1, radius=0, bkg=(0,0)):
    '''
    Aperture VDF: sums the counts within a circular aperture of a given radius around a given center.
    Args:
        matrix: 4D numpy array
        center_x: int
        center_y: int
        radius: int
    Returns:
        pixel_sum: 2D numpy array
        
        Note: Leaving radius as 0 is equivalent to a point VDF.'''
    #Aperture VDF
    pixel_sum = np.zeros((matrix.shape[0], matrix.shape[1]))
    rows, cols = len(matrix[2]), len(matrix[3])
    count = 0

    for i in range(center_x - radius, center_x + radius + 1):
        for j in range(center_y - radius, center_y + radius + 1):
            if 0 <= i < rows and 0 <= j < cols:
                pixel_sum += matrix[:,:,i,j]*g*m
                count += 1
    pixel_sum = pixel_sum
    print("Var:", count)
    if bkg != (0,0):
        pixel_sum += AnnularVar(matrix, center_x, center_y,g,m,VarB, bkg[0], bkg[1])*count
    
    pixel_sum[pixel_sum<0] = 0

    return pixel_sum

def AlignByBeam(matrix, x, y, crop=4,reiterate=1):
    '''
    Aligns by the beam position nearest to the defined x, y coordinate.
    Args:   
        matrix: 4D numpy array
        x: int (center x coordinate)
        y: int (center y coordinate)
        reiterate: int (number of times to reiterate the alignment)
    Returns:    
        aligned_matrix: 4D numpy array
    '''
    # Get the shape of the matrix
    matrix_shape = matrix.shape

    # Create a new matrix to store the aligned matrices
    aligned_matrix = np.zeros(matrix.shape, dtype=matrix.dtype)
    count = 0

    # Iterate over each 124x124 matrix
    for k in range(reiterate):
        for i in range(matrix_shape[0]):
            for j in range(matrix_shape[1]):
                # Get the current 124x124 matrix
                if k == 0:
                    current_matrix = matrix[i,j,:,:]
                else:
                    current_matrix = aligned_matrix[i,j,:,:]
                current_submatrix = current_matrix.copy()
                current_submatrix[0:x-crop,0:matrix_shape[3]] = 0
                current_submatrix[0:matrix_shape[2],0:y-crop] = 0
                current_submatrix[x+crop:matrix_shape[2],0:matrix_shape[3]] = 0
                current_submatrix[0:matrix_shape[2],y+crop:matrix_shape[3]] = 0

                # Find the peak closest to the defined x, y coordinate
                peak_x, peak_y = np.unravel_index(np.argmax(current_submatrix), current_submatrix.shape)

                # Calculate the shift needed to align the peak to the defined x, y coordinate
                shift_x = x - peak_x
                shift_y = y - peak_y
                if shift_x or shift_y != 0:
                    count += 1

                # Shift the current matrix
                aligned_matrix[i,j,:,:] = np.roll(current_matrix, (shift_x, shift_y), axis=(0, 1))

    print("number of shifted patterns: ", count)
    return aligned_matrix

def getGridVectors(x0, y0, x2, y2, x1, y1, order=3):
    grid = []
    for i in range(-order, order+1):
        for j in range(-order, order+1):
            vector = (x0 + i * (x1-x0) + j * (x2-x0), y0 + i * (y1-y0) + j * (y2-y0))
            if vector[0] < 1 or vector[0] > 122 or vector[1] < 1 or vector[1] > 122:
                continue
            grid.append(((i,j),vector))
    return grid

def find_nearest_peak(vector, matrix, crop=4):
    '''
    Finds the nearest peak to the defined vector.
    Args:
        vector: tuple
        matrix: 2D numpy array
    Returns:
        max_indices: tuple
    '''
    matrix_shape = matrix.shape
    current_submatrix = matrix.copy()
    #print(vector[0],vector[1])
    current_submatrix[0:vector[1]-crop,0:matrix_shape[0]] = 0
    current_submatrix[0:matrix_shape[0],0:np.array(vector[0])-crop] = 0
    current_submatrix[np.array(vector[1])+crop:matrix_shape[0],0:matrix_shape[1]] = 0
    current_submatrix[0:matrix_shape[0],np.array(vector[0])+crop:matrix_shape[1]] = 0
    max_indices = np.unravel_index(np.argmax(current_submatrix), current_submatrix.shape)
    #print(max_indices)
    #plt.figure()
    #plt.imshow(current_submatrix, cmap='gray', vmin=0, vmax=30000)  
    #plt.show()
    return max_indices
    

def getGridVectors_Adjust(x0, y0, x1, y1, x2, y2, average_after,order=4,crop=2):
    '''
    Generates a grid of vectors up to +/- 5th order.
    Args:
        x0: int (center x coordinate)
        y0: int (center y coordinate)
        x1: int (x coordinate of the first vector)
        y1: int (y coordinate of the first vector)
        x2: int (x coordinate of the second vector)
        y2: int (y coordinate of the second vector)
        average_after: 2D numpy array
    Returns:
        grid2: list of tuples
    '''
    grid2 = []
    for i in range(-order, order+1):
        for j in range(-order, order+1):
            vector = (x0 + i * (x1-x0) + j * (x2-x0), y0 + i * (y1-y0) + j * (y2-y0))
            if vector[0] < 1 or vector[0] > 122 or vector[1] < 1 or vector[1] > 122:
                continue
            # Find the nearest peak to the vector
            peak = find_nearest_peak(vector, average_after,crop=crop)
            if peak[0] < 1 or peak[0] > 122 or peak[1] < 1 or peak[1] > 122:
                continue
            grid2.append(((i,j), peak))
    return grid2

def AnnularSum(matrix, center_x, center_y, ri, ro):
    '''
    Annular VDF: sums the counts within an annular aperture of a given inner and outer radius around a given center.
    Args:
        matrix: 4D numpy array
        center_x: int
        center_y: int
        ri: int
        ro: int
    Returns:
        pixel_sum: 2D numpy array
    '''
    #Annular VDF
    pixel_sum = np.zeros((matrix.shape[0], matrix.shape[1]))
    rows, cols = len(matrix[2]), len(matrix[3])
    count2 = 0

    for i in range(center_x - ro, center_x + ro + 1):
        for j in range(center_y - ro, center_y + ro + 1):
            if 0 <= i < rows and 0 <= j < cols:
                distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if ri <= distance <= ro:
                    pixel_sum += matrix[:,:,i,j]
                    count2 += 1
    return pixel_sum/count2

def AnnularVar(matrix, center_x, center_y,g,m,VarB, ri, ro):
    '''
    Annular VDF: sums the counts within an annular aperture of a given inner and outer radius around a given center.
    Args:
        matrix: 4D numpy array
        center_x: int
        center_y: int
        ri: int
        ro: int
    Returns:
        pixel_sum: 2D numpy array
    '''
    #Annular VDF
    pixel_sum = np.zeros((matrix.shape[0], matrix.shape[1]))
    rows, cols = len(matrix[2]), len(matrix[3])
    count2 = 0

    for i in range(center_x - ro, center_x + ro + 1):
        for j in range(center_y - ro, center_y + ro + 1):
            if 0 <= i < rows and 0 <= j < cols:
                distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if ri <= distance <= ro:
                    pixel_sum += matrix[:,:,i,j]*m*g
                    count2 += 1
    return pixel_sum/count2

def Precession(matrix, center_x, center_y, ri, ro):
    '''
    Precession: sums the counts within an annular aperture of a given inner and outer radius around a given center.
    Args:
        matrix: 4D numpy array
        center_x: int
        center_y: int
        ri: int
        ro: int
    Returns:
        pixel_sum: 2D numpy array
    '''
    #Annular VDF
    pixel_sum = np.zeros((matrix.shape[2], matrix.shape[3]))
    rows, cols = len(matrix[0]), len(matrix[1])
    count = 0

    for i in range(center_x - ro, center_x + ro + 1):
        for j in range(center_y - ro, center_y + ro + 1):
            if 0 <= i < rows and 0 <= j < cols:
                distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if ri <= distance <= ro:
                    pixel_sum += matrix[i,j,:,:]
                    count += 1
    return pixel_sum/count

def Precession_Angled(matrix, center_x, center_y, ri, ro, angle1, angle2):
    '''
    Precession_Angled: sums the counts within an annular aperture of a given inner and outer radius around a given center,
    within the specified angles.
    Args:
        matrix: 4D numpy array
        center_x: int
        center_y: int
        ri: int
        ro: int
        angle1: int
        angle2: int
    Returns:
        pixel_sum: 2D numpy array
    '''
    # Annular VDF
    pixel_sum = np.zeros((matrix.shape[2], matrix.shape[3]))
    rows, cols = len(matrix[0]), len(matrix[1])
    count = 0

    for i in range(center_x - ro, center_x + ro + 1):
        for j in range(center_y - ro, center_y + ro + 1):
            if 0 <= i < rows and 0 <= j < cols:
                distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                angle = np.arctan2(j - center_y, i - center_x) * 180 / np.pi
                if ri <= distance <= ro and angle1 <= angle <= angle2:
                    pixel_sum += matrix[i, j, :, :]
                    count += 1
    return pixel_sum / count

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    max_radius = min(center[0], center[1], data.shape[0] - center[0], data.shape[1] - center[1])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    radialprofile = radialprofile[:max_radius]

    return radialprofile


def get_gvectors(g1, g2, grid):
    g_vectors = []
    for vector in grid:
        g_vector = (vector[0][0]*g1[0] + vector[0][1]*g2[0], vector[0][0]*g1[1] + vector[0][1]*g2[1], vector[0][0]*g1[2] + vector[0][1]*g2[2])
        g_vectors.append(g_vector)
    return g_vectors

def main():
    pass


if __name__ == "__main__":
     
    main()