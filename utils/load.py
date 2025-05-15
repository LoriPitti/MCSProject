#LOAD .mtx files from folder /matrix

from scipy.io import mmread

def load_matrix(file_path):
    A  = mmread(file_path)
    #convert to CSR for efficency
    return A.tocsr()