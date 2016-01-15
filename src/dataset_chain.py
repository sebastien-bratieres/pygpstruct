import numpy as np
import os

class dataset_chain:

    def __init__(self, task, data_folder, sub_indices = None):
        if (task == 'basenp'):
            n_features_x = 6438;
            n_labels = 3;
            self.load_data_nlp(data_folder, n_labels, sub_indices, n_features_x);
        elif (task == 'chunking'):
            n_features_x = 29764;
            n_labels = 14;
            self.load_data_nlp(data_folder, n_labels, sub_indices, n_features_x);
        elif (task == 'japanesene'):
            n_features_x = 102799;
            n_labels = 17;
            self.load_data_nlp(data_folder, n_labels, sub_indices, n_features_x);
        elif (task == 'segmentation'):
            n_features_x = 1386;
            n_labels = 2;
            self.load_data_nlp(data_folder, n_labels, sub_indices, n_features_x);
        elif (task == 'protein'):
            self.protein_jzhou(data_folder)
            if sub_indices is not None:
                self.trim_dataset(sub_indices)
            #self.protein_data(sub_indices, np.load(data_folder + '/cb513+profile_split1.npy'))
            #self.extract_subset(sub_indices, dataset_chain.protein_unpickle(data_folder))
        elif (task is None): # do nothing; fields will be filled in later
            pass
        else:
            print('Task %s is not configured. Please use one of basenp, chunking, japanesene, segmentation, protein as the value of the task argument (default is basenp).' % task)

    def protein_jzhou(self, data_folder):
        data_jzhou = np.load(data_folder + '/cb513+profile_split1.npy').reshape((514,700,-1))
        self.N = data_jzhou.shape[0] # 514
        self.n_labels = 8 # we're not predicting end of sequence label
        self.object_size = np.empty((self.N), dtype=np.int)
        x_list = [] # will accumulate sparse matrices containing data_jzhou data

        for n in range(self.N):
            non_zero_indices = np.nonzero(data_jzhou[n, :, 21] == 1)[0] # check whether AA is NoSeq
            if non_zero_indices.size == 0: # protein chains #512 and 513 have size 700 (fill out entire row)
                self.object_size[n] = 700
            else:
                self.object_size[n] = non_zero_indices[0]

        self.n_points = self.object_size.sum()
        self.X = [] # initially accumulate everything in lists, then concatenate all # eventually shape like np.empty((self.n_points, 21-0+56-35))
        # we'll keep features 0:21 (AA identity), 35:56 (sequence profile, excluding end of sequence marker at position 56)
        x_indices_extract = list(range(0,21))
        x_indices_extract.extend(list(range(35,56)))
        self.Y = []
        for n in range(self.N):
            self.X.append(data_jzhou[n, :self.object_size[n], x_indices_extract])
            self.Y.append(dataset_chain.one_hot_to_index(data_jzhou[n, :self.object_size[n], range(22,30)].T))
        self.X = np.hstack(self.X).T
        self.define_unaries_binaries()

    def protein_data(sub_indices):
        self.N = len(indices_sub)
        self.n_labels = full_dataset.n_labels
        self.object_size = full_dataset.object_size[indices_sub] # array slicing
        self.n_points = self.object_size.sum()
        self.Y = [full_dataset.Y[i] for i in indices_sub] # list slicing
        
        index_X_list = []
        c = full_dataset.object_size.cumsum()
        for i in indices_sub:
            index_X_list.extend(range(c[i] - full_dataset.object_size[i], c[i]))
        self.X = full_dataset.X[index_X_list, :]
        
        self.N = len(sub_indices)
        self.n_labels = 9
        self.object_size = 700*np.ones((len(sub_indices)))
        self.n_points = self.object_size.sum()
        self.Y = []
        for n in range(self.N):
            self.Y.append()  #WASDOING

            
    def protein_unpickle(data_folder):
        import pickle
        with open(data_folder + '/dataset.pickle', 'rb') as f:
            t = pickle.load(file=f)
        return t
    
    def protein_build_from_disk(data_folder, n_objects):
        # initialize features
        import numpy as np
        import scipy.sparse
        # construct the look-up table of features: look up by AA, and extract a vector of size n_features_per_aa
        features_wang_3 = [[1.4587,0.6962,0.8491],
            [1.2498,0.8031,0.9393],
            [0.8040,0.6428,1.3278],
            [0.9671,0.5056,1.2845],
            [0.6698,1.4103,1.0328],
            [1.2532,0.7660,0.9557],
            [1.4630,0.6267,0.8812],
            [0.4867,0.6879,1.5240],
            [0.9192,1.1220,1.0063],
            [1.0150,1.6853,0.6558],
            [1.3375,1.1946,0.6811],
            [1.1249,0.8516,1.0009],
            [1.3512,1.1681,0.6850],
            [1.0837,1.4331,0.7357],
            [0.5284,0.4394,1.6207],
            [0.7604,0.9021,1.2270],
            [0.6942,1.2628,1.0905],
            [1.1460,1.4842,0.6670],
            [0.9481,1.4696,0.8108],
            [0.9126,1.8529,0.6418],
            [0,0,0]]  # no idea what the last AA should have as features
        features_wang_1 = [ 1.8, -4, -3.5, -3.5, 2.5, -3.5, -3.5, 0.4, -3.2, 4.5, 3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2, 0]
        list_aa = 'ARNDCQEGHILKMFPSTWYV|'
        dict_aa_index = {}
        for (i,s) in enumerate(list_aa):
            dict_aa_index[s]=i
        list_dssp = 'HGEBTS_?'
        n_list_aa = len(list_aa)
        # context window size 15=7+1+7; 
        n_side_context = 7
        n_window_positions = n_side_context + 1 + n_side_context
        # features for each position in context window: single Wang feature, 3 Wang features, 1-of-k indicator for which AA it is
        n_features_per_aa = 1+3+n_list_aa

        n_features_per_window = n_window_positions * n_features_per_aa

        features_dssp = np.empty((n_list_aa, n_features_per_aa))
        for n in range(n_list_aa):
            features_dssp[n,:] = np.array([features_wang_1[n]] + features_wang_3[n] + np.eye((len(list_aa)))[n,:].tolist()) 
        features_dssp = scipy.sparse.csr_matrix(features_dssp)

        # read data from disk and construct features
        result = dataset_chain(None) # constructs empty object
        import glob
        file_names = glob.glob(data_folder + '/*.all')[:n_objects]
        result.N = len(file_names)
        result.n_labels = len(list_dssp)
        result.object_size = np.empty((result.N), dtype=np.int)
        x_list = [] # will accumulate sparse matrices containing X data

        for (n, file_name) in enumerate(file_names):
            result.object_size[n] = len(open(file_name, 'rt').readlines()[0][4:-1])//2 # remove RES: and ,\n; then divide by 2 cos chain is like R,G,T,H,...H,

        result.n_points = result.object_size.sum()
        result.X = scipy.sparse.csr_matrix((result.n_points, n_features_per_window))
        result.Y = []
        result.unaries=[]
        index_X_row = 0

        for (n, file_name) in enumerate(file_names):
            print(n)
            print(file_name)
            lines = open(file_name, 'rt').readlines()
            line_aa = lines[0][4:-2].split(sep=",") # remove RES: and ,\n
            line_dssp = lines[1][5:-2].split(sep=",") # remove DPSS: and ,\n
            assert len(line_aa) == len(line_dssp)
            object_size_current = result.object_size[n] # == len(line_aa) == len(line_dpss)

            temp_y = np.empty((object_size_current), dtype=np.int8)
            for index_sequence in range(object_size_current): # go through sequence
                for offset in range(-n_side_context,+n_side_context+1): # go through possible offsets in the context window
                    if index_sequence+offset in range(object_size_current): # if this position in the window is within the boundaries of the sequence, proceed. 
                        #otherwise leave at 0, not sure this is ideal?
                        result.X[index_X_row,n_features_per_aa*(offset+n_side_context):n_features_per_aa*(offset+1+n_side_context)] = features_dssp[dict_aa_index[line_aa[index_sequence+offset]], :]
                        # so current position (offset 0) is at position n_side_context in window
                        #print(n_features_per_aa*(offset+n_side_context))
                index_X_row = index_X_row + 1
                temp_y[index_sequence] = list_dssp.find(line_dssp[index_sequence])

            result.Y.append(temp_y)
            result.unaries.append(np.zeros((result.object_size[n], result.n_labels), dtype=np.int))

        result.X.eliminate_zeros()
        result.define_unaries_binaries()
        return result

    def define_unaries_binaries(self):
        import numpy as np
        # .unaries has the most basic (assumes no tying) shape compatible with the linear CRF: f~(n, t, y)
        self.unaries = []
        for n in range(self.N):
            self.unaries.append(np.zeros((self.object_size[n], self.n_labels), dtype=np.int))        
            
        f_index_max = 0 # contains largest index in f
        for yt in range(self.n_labels):
            for n in range(self.N):
                for t in range(self.object_size[n]):
                    self.unaries[n][t, yt] = f_index_max
                    f_index_max = f_index_max + 1

        self.binaries = np.arange(f_index_max, f_index_max + self.n_labels**2).reshape((self.n_labels, self.n_labels), order='F')    

    def load_data_nlp(self, data_folder, n_labels, sub_indices, n_features_x):
        if sub_indices is None:
            raise Exception('Need sub_indices array.')
        import scipy
        # .Y list of ndarrays
        self.N = sub_indices.shape[0]
        self.n_labels = n_labels;
        self.Y = []

        # read data from sparse representation in file
        # -----
        # to initialize a sparse matrix X containing (rows = position in sequence, all seq concatenated, cols = feature, value=binaries)
        # first create a large IJV array, with the I indices corresponding to rows in the vstack'ed X
        # if I knew how to append to a COO sparse matrix, I could avoid all this, and just append rows as I go along
        ijv_array_list = []     # ijv_array_list is a list, with one ijv array per sentence
        self.n_points = 0 # how many rows does X have so far?
        self.object_size = np.zeros((self.N), dtype=np.int16) # contains length of sentence n
        for n in range(self.N):
            # set data_X part
            this_x = np.loadtxt(os.path.join(data_folder, str(sub_indices[n]+1) + ".x"), dtype=np.int32) # int32: some tasks have up to 102799 features (japanesene)
            self.object_size[n] = this_x[-1,0] # the last element of the (Matlab, row-ordered) sparse format representation will contain an index into the last row
            this_x[:,[0,1]] -= 1 # from Matlab-indexing to Numpy-indexing: remove 1 from i and j indices
            this_x[:,0] += self.n_points # correct all row indices so that they correspond to a vstack'ed X
            ijv_array_list.append(this_x)
            self.n_points += self.object_size[n]

            # set other parts
            this_y = np.loadtxt(os.path.join(data_folder, str(sub_indices[n]+1) + '.y'), dtype=np.int8) # labels start with 0 in data files
            this_y[this_y < 0] = 0 # remove the three -1 labels found in task JapaneseNE for unknown reason
            self.Y.append(this_y) 

        assert(self.n_points == self.object_size.sum())
        # stack the ijv lists vertically, to create a complete ijv array
        ijv_array=np.vstack(tuple(ijv_array_list))
        # create a sparse matrix from the ijv array (coo_matrix expects input in order vij)
        self.X = scipy.sparse.coo_matrix((ijv_array[:,2],(ijv_array[:,0],ijv_array[:,1])), shape=(self.n_points, n_features_x))

        self.define_unaries_binaries()

    def trim_dataset(self, indices_sub):
        self.N = len(indices_sub)
        self.Y = [self.Y[i] for i in indices_sub] # list slicing
        
        index_X_list = []
        c = self.object_size.cumsum() # working with original object sizes
        for i in indices_sub:
            index_X_list.extend(range(c[i] - self.object_size[i], c[i]))
        self.X = self.X[index_X_list, :]
        self.object_size = self.object_size[indices_sub] # array slicing
        self.n_points = self.object_size.sum()
        self.define_unaries_binaries()
    
    def one_hot_to_index(a): # assume [n, one-hot index]
        result = np.empty(a.shape[0], dtype=np.int32)
        sequence = np.arange(a.shape[1])
        for n in range(a.shape[0]):
            result[n] = a[n].dot(sequence)
        return result
