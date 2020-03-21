#!/usr/bin/env python
# coding: utf-8

import numpy as np
np.random.seed(0)

# Four different functions are defined in this file.
# For function description, please refer to docstrings of each function.
#
# 1. k_means()
# 2. k_means_repeat()
# 3. optimal_k()
# 4. saving_txt()


def k_means(input_data, num_k):
    """ Naive K-means clustering with num_k clusters on points from input_data.

    Parameters
    ----------

    input_data : numpy array of dimension (m,n)
        The m data points of n dimension to be clustered in the algorithm.
    
    num_k : int
        The number of clusters to form,
        as well as the number of centroids to generate.
    
    Returns
    --------
    centroids: an ordered list of the cluster centroids 
    clusters: a list of num_k lists containing the clustered points from input_data.

    """
    
    
    def generate_clusters(input_data, centroids):
        """Assign points from input_data to clusters centered on centroids."""

        # Clustered points to be put in m independent lists in clusters:
        n_centers = len(centroids)
        clusters = [[] for i in range(n_centers)]

        # Cluster assignment is based on Euclidean distance between each point and the closest centroid.
        for point in input_data:
                eucl_dist = [(ith, np.sqrt(np.dot((point-center),(point-center))))
                                for ith, center in enumerate(centroids)]
                cluster_val = min(eucl_dist,key=lambda x: x[1])[0]
                clusters[cluster_val].append(point)

        return clusters

    
    def update_centroids(centroids, input_clusters):
        """Updating centroids to adjust for new cluster assignments,
        based on mean euclidean distances of points in the same cluster.
        """
        
        centroids = [np.mean(input_clusters[i], axis=0) for i in range(len(centroids))]
        return centroids


    # Initial Set Ups
    n_datapoints = input_data.shape[0] 
    n_dimensions = input_data.shape[1]
    
    # Random num_k points selected as initial centroids (aka. Forgy Method)
    centroids = input_data[np.random.choice(n_datapoints, num_k, replace=False),:]
    
    # Run K-means until convergence criterion is reached. 
    while True:
        pre_centroids = centroids
        clusters = generate_clusters(input_data, centroids)
        centroids = update_centroids(centroids, clusters)
        
        # convergence criterion: if the centers did not change much (by an small threshold value),
        # we are done
        if np.allclose(np.sort(centroids, axis=1), np.sort(pre_centroids, axis=1)):
            break


    return centroids, clusters



def kmeans_repeat(input_data, num_k, num_repeat):
    """Repeating K-means num_repeat times, increasing the chance of picking the optimal clusters.
    
    Note: This function was devised because Naive K-means does not guarantee global optimum.
    The higher the num_k value, the more likely we can achieve the optimal separation for clusters.
    

    Parameters
    ----------

    input_data : numpy array of dimension (m,n)
        The m data points of n dimension to be clustered in the algorithm.
    
    num_k : int
        The number of clusters to form,
        as well as the number of centroids to generate.
        
    num_repeat: int
        The number of times K-means is repeated. 
    
    
    Returns
    --------
    centroids & clusters which are the best results with
    the lowest WISS(Within-Sum-of-Squares, aka Inertia) from num_repeat trial runs
    
    centroids: an ordered list of the cluster centroids 
    clusters: a list of num_k lists containing the clustered points from input_data.
    
    """

    updated_WISS = None
    
    for i in range(num_repeat):
        
        # Some extreme initial random centroid assignment could cause error in k_means()
        # To deal with this, try/except was used.
        try:
            # print(i)
            centroids, clusters = k_means(input_data, num_k)
            WISS = np.sum([np.sum(np.linalg.norm(np.subtract(cluster_c,centroids[j]))) 
                           for j, cluster_c in enumerate(clusters)])

            if updated_WISS is None or updated_WISS > WISS:
                updated_WISS = WISS
                
                global optimized_clustering # To deal with error from optimal_k()
                optimized_clustering = centroids, clusters, updated_WISS
            # print(updated_WISS)
        except:
            # print('error')
            pass
        
    return optimized_clustering
    


def optimal_k(input_data,num_k=7):
    """ Finding optimal K value, i.e. the number of clusters.
    
    Note: There is not a standard way of choosing K for K-means.
    Some common methods include Elbow-plot, Silhouette Score, GAP statistic etc.
    For this assignment, Elbow-plot heuristics was used.
    Since we cannot visually check the "elbow" point every time,
    it was assumed that the "elbow" point occurs when the point is furthest away
    , in terms of orthogonal distance, from the diagonal line connecting
    the first and last points on the Elbow-plot
    
    
    Parameters
    ----------

    input_data : numpy array of dimension (m,n)
        The m data points of n dimension to be clustered in the algorithm.
    
    num_k : int
        The number of clusters to form,
        as well as the number of centroids to generate.
    
    
    Returns
    --------
    k_optimum: the optimal value of K based on Elbow-plot heuristics.
    
    
    """
    
    k_wiss_list = np.zeros(num_k)

    # generate best results for each num_k value
    
    for i in range(num_k):
        
        if i < 1:
            continue
        else:
            k_wiss_list[i] = kmeans_repeat(input_data, i, 100)[2]
    
    
    # Find the "elbow" point using orthogonal distance
    # from first and last points on the plot
    
    coordinates = list(enumerate(k_wiss_list))
    coordinates_short = coordinates[2:-1] #excluding last and first
    coordinates_short
    p1 = np.asarray(coordinates[1])
    p2 = np.asarray(coordinates[-1])
    d = []
    # Generate the list of orthogonal distances for each num_k point on the plot
    for i in range(len(coordinates_short)):
        d.append(np.linalg.norm(np.cross(p2-p1, p1-np.asarray(coordinates_short[i])))/np.linalg.norm(p2-p1))
    k_optimum = d.index(max(d))+2 #account for d starting from 2nd element
    
    
    return k_optimum



def saving_text(input_file, num_k=10):
    """Saves a text file of K-means clusters.
    
    Parameters
    ----------

    input_file : str
        file name of the input data
    
    num_k : int
        The number of clusters to form,
        as well as the number of centroids to generate.
    
    
    Returns
    --------
    A text(.txt) file with title in the format "filename_optimizedK.txt"
    and contains the clusters points for optimizedK.
    
    """
    
    test_file = np.loadtxt(input_file)
    opt_k = optimal_k(test_file, num_k)
    x = kmeans_repeat(test_file, opt_k, 100)[1]
    file_name_no_ext = ('.').join(input_file.split('.')[:-1])
    title = "%s_%s.txt" %(file_name_no_ext,opt_k)
    
    np.savetxt(title,x,fmt='%s')

