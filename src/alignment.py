import numpy as np


def alignment_simple(bp, max_pos, mat1, mat2, mode='switch'):
    """
    Reconstructs a column-wise alignment of two windows by
    tracing back through the backpointer matrix

    Modes:
    - 'separate': keep original values in each output
    - 'multiply': elementwise product written into both outputs
    - 'sum': elementwise sum written into both outputs
    - 'switch': swap matched columns between outputs    
    """
    max_i, max_j = max_pos
    
    alignment1 = np.zeros_like(mat1)
    alignment2 = np.zeros_like(mat2)
    
    # traceback from max position
    i, j = max_i, max_j
    positions1 = []
    positions2 = []
    
    while i > 0 and j > 0:
        choice = bp[i, j]
        
        if choice == 0:
            # reset -> terminate
            break
        elif choice == 1:
            # diagonal 
            # match between mat1[i-1] and mat2[j-1]
            positions1.append(i-1)
            positions2.append(j-1)
            i -= 1
            j -= 1
        elif choice == 2:
            # vertical gap
            # mat1[i-1] has no match -> skip in mat1
            i -= 1
        elif choice == 3:
            # horizontal gap
            # mat2[j-1] has no match -> skip in mat2
            j -= 1
        else:
            break
    
    # reverse to get correct order
    positions1 = positions1[::-1]
    positions2 = positions2[::-1]
    
    # fill in the alignments based on mode
    for pos1, pos2 in zip(positions1, positions2):
        if mode == 'multiply':
            # only keep positions where both have activity
            alignment1[:, pos1] = mat1[:, pos1] * mat2[:, pos2]
            alignment2[:, pos2] = mat1[:, pos1] * mat2[:, pos2]
        elif mode == 'separate':
            # keep original values
            alignment1[:, pos1] = mat1[:, pos1]
            alignment2[:, pos2] = mat2[:, pos2]
        elif mode == 'sum':
            # union of activity
            alignment1[:, pos1] = mat1[:, pos1] + mat2[:, pos2]
            alignment2[:, pos2] = mat1[:, pos1] + mat2[:, pos2]
        elif mode == 'switch':
            # put the activity of one in the timeline of the other
            alignment1[:, pos1] = mat2[:, pos2]
            alignment2[:, pos2] = mat1[:, pos1]
    
    return alignment1, alignment2


def alignment_compressed(bp, max_pos, mat1, mat2, mode='switch'):
    """
    Reconstructs a right-aligned column-wise alignment of two windows
    by tracing back through the backpointer matrix

    Modes:
    - 'separate': keep original values in each output
    - 'multiply': elementwise product written into both outputs
    - 'sum': elementwise sum written into both outputs
    - 'switch': swap matched columns between outputs    
    """
    max_i, max_j = max_pos
    
    alignment1 = np.zeros_like(mat1)
    alignment2 = np.zeros_like(mat2)
    
    # traceback from max position
    i, j = max_i, max_j
    row_counter = 0
    col_counter = 0
    
    while i > 0 and j > 0:
        choice = bp[i, j]
        
        if choice == 0:
            # reset -> terminate
            break
        elif choice == 1:
            # diagonal
            # match between mat1[i-1] and mat2[j-1]
            if mode == 'multiply':
                # only keep positions where both have activity
                alignment1[:, row_counter] = mat1[:, i-1] * mat2[:, j-1]
                alignment2[:, col_counter] = mat1[:, i-1] * mat2[:, j-1]
            elif mode == 'separate':
                # keep original values
                alignment1[:, row_counter] = mat1[:, i-1]
                alignment2[:, col_counter] = mat2[:, j-1]
            elif mode == 'sum':
                # union of activity
                alignment1[:, row_counter] = mat1[:, i-1] + mat2[:, j-1]
                alignment2[:, col_counter] = mat1[:, i-1] + mat2[:, j-1]                
            elif mode == 'switch':
                # put the activity of one in the timeline of the other
                alignment1[:, row_counter] = mat2[:, j-1]
                alignment2[:, col_counter] = mat1[:, i-1]
            
            i -= 1
            j -= 1
            row_counter += 1
            col_counter += 1
        elif choice == 2:
            # vertical gap
            # mat1[i-1] has no match -> skip in mat1
            i -= 1
            row_counter += 1
            
        elif choice == 3:
            # horizontal gap
            # mat2[j-1] has no match -> skip in mat2
            j -= 1
            col_counter += 1
        else:
            break
    
    # reverse to get correct order (right-aligned)
    alignment1 = alignment1[:, ::-1]
    alignment2 = alignment2[:, ::-1]
    
    return alignment1, alignment2