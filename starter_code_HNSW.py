import faiss
import h5py
import numpy as np
import os
import requests


def evaluate_hnsw():
    # start your code here
    # download data, build index, run query
    
    # with h5py.File('http://ann-benchmarks.com/sift-128-euclidean.hdf5', 'r') as f:
    with h5py.File('sift.h5', 'r') as f:  # after downloading through above link
        print(list(f.keys()))  # we see keys: distances, neighbors, test, train
        train_embeddings = f['train'][:]
        
        # initialize hnsw index on SIFT1M data
        if os.path.exists('hnsw_sift1m.index'):
            print("Loading pretrained HNSW index from disk")
            index = faiss.read_index('hnsw_sift1m.index')
        else:
            print("No HNSW index found, building index")
            M=16  # number of neighbors added to index in each insertion
            efConstruction=200  # for index construction
            efSearch=200  # during search query
            dim = 128  # set dimension
        
            index = faiss.IndexHNSWFlat(dim, M)
            index.hnsw.efConstruction = efConstruction
            index.hnsw.efSearch = efSearch
            index.add(train_embeddings)

            faiss.write_index(index, 'hnsw_sift1m.index')

        # run query on one vector from test data
        test_vector = f['test'][:1]
        print(test_vector.shape)
        results = index.search(test_vector, 10)  # search 10 nearest neighbors
        print(results)
        indices  = results[1][0]
        print("Indices of 10 nearest neighbors:", indices)
    
    # write the indices of the 10 approximate nearest neighbours in output.txt, separated by new line in the same directory
    if not os.path.exists('output.txt'):
        with open('output.txt', 'w') as out_file:
            for idx in indices:
                out_file.write(f"{idx}\n")

if __name__ == "__main__":
    
    evaluate_hnsw()
