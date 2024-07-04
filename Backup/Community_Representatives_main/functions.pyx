
import sys
import matplotlib.pyplot as plt
import igraph as ig
import numpy as np
import sknetwork
from IPython.display import display,SVG,clear_output
from collections import Counter
import os
import graph_tool as gt
import itertools
from graph_tool.all import *
import pyintergraph as pyg
import scipy.sparse as ss
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
cimport cython
cimport numpy as cnp
import numpy as np
cdef extern from "numpy/arrayobject.h":
    pass

cdef extern from "numpy/ufuncobject.h":
    pass
import graph_tool as gt
from graph_tool.all import *
from collections import Counter
import itertools
import random
from cython.view cimport array as cvarray
from libc.math cimport isnan
cdef extern from "math.h":
    double log(double x)
    double sinh(double x)
    double cosh(double x)
    double exp(double x)
    double pow(double x, double y)
    double log1p(double x)
    double sqrt(double x)
    double fabs (double x)
    double lgamma (double x)
    
cdef extern from "<algorithm>" namespace "std" nogil:
    Iter find[Iter, T](Iter first, Iter last, const T& value)
    OutputIter merge[InputIter1, InputIter2, OutputIter] (InputIter1 first1, InputIter1 last1,
                        InputIter2 first2, InputIter2 last2,
                        OutputIter result)
   
    
from libc.stdlib cimport rand, RAND_MAX, srand, malloc, free
from libc.math cimport floor, ceil
from cython.parallel import prange
from libcpp.map cimport map as cpp_map
from libcpp.vector cimport vector as cpp_vector
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport pair as cpp_pair
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as preinc  
from libc.stdint cimport intptr_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef double random_uniform() nogil:
    """
    uniform rng
    """
    cdef double r = rand()
    return r / RAND_MAX

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef long random_int(long a,long b) nogil:
    """
    uniform rng for integer within [a,b]
    """
    b += 1
    return int(a + random_uniform()*(b-a))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef double zero_lgamma(double a):
    """
    log gamma function
    """
    if a == 0.:
        return 0.
    else:
        return lgamma(a)
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef double zero_log(double a):
    """
    log(x), with log(0) set to 0
    """
    if a == 0.:
        return 0.
    else:
        return log(a)
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef double log_choose(long N,long K):
    """
    log(N choose K)
    """
    cdef double Nd = float(N), Kd = float(K)
    return zero_lgamma(Nd+1.) - zero_lgamma(Nd+1.-Kd) - zero_lgamma(Kd+1.)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef double log_multinomial(long N,cpp_vector[double] counts):
    """
    log oof multinomial coefficient, with vector 'counts' giving sizes of bins in the denominator
    """
    cdef double logcoef,c
    logcoef = lgamma(N+1.)
    for c in counts:
        logcoef -= zero_lgamma(c+1.)
    return logcoef
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef unordered_set[long] unique(long[:] arr):
    """
    find unique elements in 'arr'
    """
    cdef long length = arr.shape[0]
    cdef unordered_set[long] uniq
    cdef long i,val
    for i in range(length):
        val = arr[i]
        if (uniq.find(val) == uniq.end()):
            uniq.insert(val)
    return uniq
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef cpp_map[long,double] counter(long[:] arr):
    """
    get counts of unique elements in 'arr' (as with collections.Counter)
    """
    cdef long length = arr.shape[0]
    cdef cpp_map[long,double] counts 
    cdef unordered_set[long] unique_elements
    cdef long i,val
    for i in range(length):
        val = arr[i]
        if (unique_elements.find(val) == unique_elements.end()):
            unique_elements.insert(val)
            counts[val] = 1.
        else:
            counts[val] += 1.   
    return counts

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef long categorical(double[:] ps):
    """
    categorical random draw with probability vector ps
    """
    cdef long i = 0
    cdef double rand = random_uniform()
    cdef double cmf = ps[0]
    
    while cmf < rand:
        cmf += ps[i]
        i += 1
    
    return i

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef unordered_set[long] set_merge(unordered_set[long] s1,unordered_set[long] s2):
    """
    merge sets together
    """
    cdef unordered_set[long] s
    cdef long i
    if s1.size() > s2.size():
        s = s1
        for i in s2:
            s.insert(i)  
    else:
        s = s2
        for i in s1:
            s.insert(i)
        
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef double MI(long[:] part1,long[:] part2):
    """
    mutual information between partitions 'part1' and 'part2'
    """
    
    cdef long i,r,s,N = part1.shape[0],ind 
    cdef double mi,log_omega,w,nu,mu,sum_x2,sum_y2,sum_logx,sum_logy,frac,n = float(N)
    cdef unordered_set[long] uniq_p1 = unique(part1)
    cdef unordered_set[long] uniq_p2 = unique(part2)
    cdef long R = uniq_p1.size(),S = uniq_p2.size()
    cdef double R_fl = float(R), S_fl = float(S)
    cdef double[:,:] cont = cvarray(shape=(R,S), itemsize=sizeof(double), format="d")
    cdef double[:] a = cvarray(shape=(R,), itemsize=sizeof(double), format="d")
    cdef double[:] b = cvarray(shape=(S,), itemsize=sizeof(double), format="d")
    cdef cpp_map[long,long] ind2r
    cdef cpp_map[long,long] ind2s
    cdef double[:] x = cvarray(shape=(R,), itemsize=sizeof(double), format="d")
    cdef double[:] y = cvarray(shape=(S,), itemsize=sizeof(double), format="d")
    
    ind = 0
    for i in uniq_p1:
        ind2r[i] = ind
        ind += 1   
    ind = 0
    for i in uniq_p2:
        ind2s[i] = ind
        ind += 1  
    for r in range(R):
        a[r] = 0.
        for s in range(S):
            b[s] = 0.
            cont[r,s] = 0.
    for i in range(N):
        r = ind2r[part1[i]]
        s = ind2s[part2[i]]
        cont[r,s] += 1.
        a[r] += 1.
        b[s] += 1.
    
    mi = 0.
    for r in range(R):
        for s in range(S):
            mi += (cont[r,s]/n)*zero_log(n*cont[r,s]/a[r]/b[s])
            
    return mi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef double RMI(long[:] part1,long[:] part2):
    """
    reduced mutual information between partitions 'part1' and 'part2'
    computes standard mutual information and subtracts off correction for information in contingency table
    """
    
    cdef long i,r,s,N = part1.shape[0],ind 
    cdef double mi,log_omega,w,nu,mu,sum_x2,sum_y2,sum_logx,sum_logy,frac,n = float(N)
    cdef unordered_set[long] uniq_p1 = unique(part1)
    cdef unordered_set[long] uniq_p2 = unique(part2)
    cdef long R = uniq_p1.size(),S = uniq_p2.size()
    cdef double R_fl = float(R), S_fl = float(S)
    cdef double[:,:] cont = cvarray(shape=(R,S), itemsize=sizeof(double), format="d")
    cdef double[:] a = cvarray(shape=(R,), itemsize=sizeof(double), format="d")
    cdef double[:] b = cvarray(shape=(S,), itemsize=sizeof(double), format="d")
    cdef cpp_map[long,long] ind2r
    cdef cpp_map[long,long] ind2s
    cdef double[:] x = cvarray(shape=(R,), itemsize=sizeof(double), format="d")
    cdef double[:] y = cvarray(shape=(S,), itemsize=sizeof(double), format="d")
    
    ind = 0
    for i in uniq_p1:
        ind2r[i] = ind
        ind += 1   
    ind = 0
    for i in uniq_p2:
        ind2s[i] = ind
        ind += 1  
    for r in range(R):
        a[r] = 0.
        for s in range(S):
            b[s] = 0.
            cont[r,s] = 0.
    for i in range(N):
        r = ind2r[part1[i]]
        s = ind2s[part2[i]]
        cont[r,s] += 1.
        a[r] += 1.
        b[s] += 1.
    
    mi = 0.
    for r in range(R):
        for s in range(S):
            mi += (cont[r,s]/n)*zero_log(n*cont[r,s]/a[r]/b[s])
            
    w = n/(n+0.5*R_fl*S_fl)
    sum_x2 = 0.
    sum_y2 = 0.
    sum_logx = 0.
    sum_logy = 0.
    for r in range(R):
        x[r] = (1.-w)/R_fl + w*a[r]/n
        sum_x2 += x[r]*x[r]
        sum_logx += zero_log(x[r])
    for s in range(S):
        y[s] = (1.-w)/S_fl + w*b[s]/n
        sum_y2 += y[s]*y[s]
        sum_logy += zero_log(y[s])
    nu = (S_fl+1.)/(S_fl*sum_x2) - 1./S_fl
    mu = (R_fl+1.)/(R_fl*sum_y2) - 1./R_fl
    log_omega = (R_fl-1.)*(S_fl-1.)*zero_log(n+0.5*R_fl*S_fl) \
                 + 0.5*(R_fl+nu-2.)*sum_logy + 0.5*(S_fl+mu-2.)*sum_logx \
                 + 0.5*(lgamma(mu*R_fl)+lgamma(nu*S_fl) \
                 - R_fl*(lgamma(S_fl)+lgamma(mu)) - S_fl*(lgamma(R_fl)+lgamma(nu)))

    return mi - log_omega/n

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef double VI(long[:] part1,long[:] part2):
    """
    variation of information between partitions 'part1' and 'part2' 
    """
    
    cdef long i,r,s,N = part1.shape[0],ind 
    cdef double vi,frac,n = float(N)
    cdef unordered_set[long] uniq_p1 = unique(part1)
    cdef unordered_set[long] uniq_p2 = unique(part2)
    cdef long R = uniq_p1.size(),S = uniq_p2.size()
    cdef double[:,:] cont = cvarray(shape=(R,S), itemsize=sizeof(double), format="d")
    cdef double[:] a = cvarray(shape=(R,), itemsize=sizeof(double), format="d")
    cdef double[:] b = cvarray(shape=(S,), itemsize=sizeof(double), format="d")
    cdef cpp_map[long,long] ind2r
    cdef cpp_map[long,long] ind2s
    
    ind = 0
    for i in uniq_p1:
        ind2r[i] = ind
        ind += 1   
    ind = 0
    for i in uniq_p2:
        ind2s[i] = ind
        ind += 1  
    for r in range(R):
        a[r] = 0.
        for s in range(S):
            b[s] = 0.
            cont[r,s] = 0.
    for i in range(N):
        r = ind2r[part1[i]]
        s = ind2s[part2[i]]
        cont[r,s] += 1.
        a[r] += 1.
        b[s] += 1.
    
    vi = 0.
    for r in range(R):
        for s in range(S):
            vi -= (cont[r,s]/n)*(zero_log(cont[r,s]/a[r]) + zero_log(cont[r,s]/b[s]))

    return vi

  
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class clustering_state():
    
    """
    class for state updated during clustering procedure
    """
    
    cdef long num_mcs,n_samples,N,K,Kprior,MItype,mode_info_type
    cdef long[:] mode_labels
    cdef long[:,:] partitions
    cdef double DL,mode_scale,num_mcs_float,N_float,n_samples_float,Lambda
    cdef double[:] H_arr,energies,partition_entropies,num_groups_arr,nps
    cdef double[:,:] CE_matrix,VI_matrix
    cdef unordered_set[long] modes
    cdef cpp_map[long,unordered_set[long]] clusters
    cdef cpp_map[long,double] cluster_sizes
    
    def __init__(self,long[:,:] partitions,double[:] energies,long num_mcs,double Lambda,\
                                 long Kprior,long MItype,long mode_info_type):
        """
        initialize class attributes
        inputs:
            'partitions': S by N vector of partitions
            'energies': length S vector of log probabilities of vectors in 'partitions'
            'num_mcs': number of local MCMC steps to perform for estimating mode of cluster
            'Lambda': linear penalty for K. set to zero for standard description length
            'Kprior': 0 gives no penalty on K; 1 gives linear penalty on K (used in paper); 2 gives log(K) penalty
            'MItype': 0 gives regular mutual information, 1 gives reduced mutual information (used in paper)
            'mode_info_type': 0 gives entropy penalty (used in paper); 1 gives less efficient fixed-length code 
        """

        self.partitions = partitions
        self.energies = energies
        self.num_mcs = num_mcs
        self.n_samples = int(self.partitions.shape[0])
        self.N = int(self.partitions.shape[1])
        self.H_arr = (np.ones(self.n_samples)*-1).astype('float')
        self.CE_matrix = (np.ones((self.n_samples,self.n_samples))*-1).astype('float')
        self.VI_matrix = (np.ones((self.n_samples,self.n_samples))*-1).astype('float')
        self.mode_labels = np.zeros(self.n_samples).astype('int')
        self.partition_entropies = np.array([gt.inference.mutual_information(p,p) for p in np.array(self.partitions).tolist()]).astype('float')
        self.nps = np.array([len(np.unique(p)) for p in np.array(self.partitions).tolist()]).astype('float')
        self.num_mcs_float = float(self.num_mcs)
        self.N_float = float(self.N)
        self.n_samples_float = float(self.n_samples)
        self.num_groups_arr = np.array([len(np.unique(p)) for p in np.array(self.partitions).tolist()]).astype('float')
        self.Lambda = Lambda
        self.Kprior = Kprior
        self.MItype = MItype
        self.mode_info_type = mode_info_type
       
    cdef double CE_w_mat(self,long p1ind,long p2ind):
        """
        compute conditional entropy between partitions with indices 'p1ind' and 'p2ind'
        saves results in matrix 'self.CE_matrix' to avoid redundant computations
        """
        
        cdef double H1,rmi
        if self.CE_matrix[p1ind,p2ind] == -1: 
            if p1ind == p2ind:
                self.CE_matrix[p1ind,p2ind] = 0.
            else:   
                if self.MItype == 0:
                    rmi = MI(self.partitions[p1ind,:],self.partitions[p2ind,:])
                else:
                    rmi = RMI(self.partitions[p1ind,:],self.partitions[p2ind,:])
                H1 = self.partition_entropies[p1ind]
                self.CE_matrix[p1ind,p2ind] = self.N_float*(H1 - rmi)/self.n_samples_float
        return self.CE_matrix[p1ind,p2ind]
    
    cdef double VI_w_mat(self,long p1ind,long p2ind):
        """
        compute variation of information between partitions with indices 'p1ind' and 'p2ind'
        not used in algorithm in paper
        """
        cdef double vi
        if self.VI_matrix[p1ind,p2ind] == -1: 
            vi = VI(self.partitions[p1ind,:],self.partitions[p2ind,:])
            self.VI_matrix[p1ind,p2ind] = vi
        return self.VI_matrix[p1ind,p2ind]
    
    cdef long closest_mode(self,long p1ind,unordered_set[long] mode_choices):
        """
        find the closest mode to partition with index 'p1ind', out of the set 'mode_choices' (which are partition indices)
        """
        cdef long m 
        cdef double min_dist = 1000000000000.,dist
        cdef long best_mode
        if (mode_choices.find(p1ind) != mode_choices.end()):
            return p1ind
        for m in mode_choices:
            dist = self.CE_w_mat(p1ind,m)
            if dist < min_dist:
                min_dist = dist
                best_mode = m      
        return best_mode
    
    cdef double cluster_DL(self,long mode,unordered_set[long] cluster,long approx):
        """
        compute description length contribution from single cluster with mode of index 'mode'
        uses MCMC with 'self.num_mcs' samples to estimate the sum of conditional entropies if approx = 1, else is exact
        """
        cdef long clen = cluster.size()
        cdef double clen_float = float(clen)
        cdef double mean = 0.,res
        cdef long i  
        cdef cpp_vector[long] cluster_vec
        for i in cluster:
            cluster_vec.push_back(i)
        if approx == 1:
            for i in range(self.num_mcs):
                mean += self.CE_w_mat(cluster_vec[i],mode)/self.num_mcs_float   
        else:
            for i in cluster:
                mean += self.CE_w_mat(i,mode)/clen_float
        if self.mode_info_type == 0:
            res = mean*clen_float + self.partition_entropies[mode]*self.N_float/self.n_samples_float
        else:
            res = mean*clen_float + self.N_float/self.n_samples_float*log(self.nps[mode])
        return res
    
    cdef double sizes_DL(self,long KK,cpp_vector[double] sizes):
        """
        compute description length contributions from number and size of clusters
        'KK' is current number of clusters, and 'sizes' is sizes of clusters
        """
        cdef double dl = 0.,s
        cdef double KKD = float(KK)
        for s in sizes:
            dl -= (s/self.n_samples_float)*zero_log(s/self.n_samples_float)
        if self.Kprior == 0:
            return dl
        elif self.Kprior == 1:
            return dl + KKD*self.Lambda
        elif self.Kprior == 2:
            return log(KKD)
    
    cdef long mc_centroid(self,unordered_set[long] cluster):
        """
        estimate mode of 'cluster'
        if size of cluster is less than 'self.num_mcs', computes mode exactly, otherwise estimates with MCMC
        """
        cdef long clen = cluster.size()
        cdef long i,approx
        cdef double min_obj = 1000000000000000.,obj
        cdef long best_centroid
        if clen < self.num_mcs:
            approx = 0
        else:
            approx = 1 
        for i in cluster:
            obj = self.cluster_DL(i,cluster,approx)
            if obj < min_obj:
                min_obj = obj
                best_centroid = i     
        return best_centroid
    
    cdef void initalize_cluster_data(self,long num_init_modes,long initial_runs):
        """
        initializes modes given 'num_init_modes' initial modes K0, and some number of initial runs for which we run K0-medoids
        """
        cdef unordered_set[long] previous_modes
        cdef unordered_set[long] cluster
        cdef unordered_set[double] mode_approx_energies
        cdef long[:] perm_partitions = np.random.permutation(range(self.n_samples)).astype('int')
        cdef double[:] approx_energies = np.array([np.round(e,5) for e in self.energies]).astype('float')
        cdef long index,best,p1ind,old
        cdef unordered_set[long] dummy
         
        self.K = num_init_modes
        
        for index in range(self.n_samples):
            p1ind = perm_partitions[index]
            if (mode_approx_energies.find(self.energies[p1ind]) == mode_approx_energies.end()):
                if (self.modes.size() < num_init_modes):
                    self.modes.insert(p1ind)
                    mode_approx_energies.insert(self.energies[p1ind])

        for p1ind in self.modes:
            self.clusters[p1ind] = dummy 
            
        for p1ind in range(self.n_samples):
            best = self.closest_mode(p1ind,self.modes)
            self.clusters[best].insert(p1ind)
            self.mode_labels[p1ind] = best
        
        for index in range(initial_runs):
         
            previous_modes = self.modes
            for m in previous_modes:
                cluster = self.clusters[m]
                best = self.mc_centroid(cluster)            
                for p1ind in cluster:
                    self.mode_labels[p1ind] = best
                self.clusters.erase(m)
                self.modes.erase(m)
                self.clusters[best] = cluster
                self.modes.insert(best)
                
            for p1ind in range(self.n_samples):
                best = self.closest_mode(p1ind,self.modes)
                old = self.mode_labels[p1ind]
                self.clusters[old].erase(p1ind)
                self.clusters[best].insert(p1ind)
                self.mode_labels[p1ind] = best
         
        for m in self.modes:
            self.cluster_sizes[m] = float(self.clusters[m].size())
            
    cdef void compute_full_DL(self):
        """
        compute full description length of 'clustering_state' object
        """
        cdef cpp_vector[double] sizes
        cdef long m        
        self.DL = 0.
        for m in self.modes:
            sizes.push_back(self.cluster_sizes[m])
            self.DL += self.cluster_DL(m,self.clusters[m],0)
        self.DL += self.sizes_DL(self.K,sizes)
                
    cdef void initialize_mode_scale(self,double dist_percentile,double manual_mode_scale):
        """
        determine scale for which to separate two modes (to avoid them becoming too close together)
        not used in paper algorithm
        """
        cdef long p1,p2,i,num_rands = 10000
        cdef double[:] dists = cvarray(shape=(num_rands,), itemsize=sizeof(double), format="d")
        if dist_percentile != -1.:
            for i in range(num_rands):
                p1 = random_int(0,self.n_samples-1)
                p2 = random_int(0,self.n_samples-1)
                dists[i] = self.VI_w_mat(p1,p2)
            self.mode_scale = np.percentile(np.array(dists),dist_percentile) 
        elif manual_mode_scale != -1.:
            self.mode_scale = manual_mode_scale
            
    cdef long closest_mode_move(self,long i):
        """
        perform move of type 1 for partition indexed 'i' 
        """
       
        cdef long previous_mode = self.mode_labels[i]
        cdef long best_mode,accepted = 0       
        best_mode = self.closest_mode(i,self.modes)
        
        if best_mode != previous_mode:
            self.clusters[best_mode].insert(i)
            self.clusters[previous_mode].erase(i)
            self.mode_labels[i] = best_mode
            self.cluster_sizes[previous_mode] -= 1.
            self.cluster_sizes[best_mode] += 1.
            accepted = 0
      
        return accepted
    
    cdef long merge_move(self):
        """
        perform move type 2 for two random modes
        """
        cdef long accepted = 0,i,mode1,mode2,rand,merged_mode,m
        cdef cpp_vector[long] modes_vector
        cdef unordered_set[long] merged_cluster
        cdef cpp_vector[double] old_sizes
        cdef cpp_vector[double] new_sizes
        cdef double DL_before,DL_after
        
        if self.K == 1:
            return 0

        for m in self.modes:
            modes_vector.push_back(m)
            old_sizes.push_back(self.cluster_sizes[m])

        rand = random_int(0,self.K - 1)
        mode1 = modes_vector[rand]
        rand = random_int(0,self.K - 1)
        mode2 = modes_vector[rand]
        while mode2 == mode1:
            rand = random_int(0,self.K - 1)
            mode2 = modes_vector[rand]
            
        DL_before = self.cluster_DL(mode1,self.clusters[mode1],0) + self.cluster_DL(mode2,self.clusters[mode2],0) \
                        + self.sizes_DL(self.K,old_sizes)
        merged_cluster = set_merge(self.clusters[mode1],self.clusters[mode2])
        merged_mode = self.mc_centroid(merged_cluster)
        for m in self.modes:
            if (m != mode1) and (m != mode2):
                new_sizes.push_back(self.cluster_sizes[m])
        new_sizes.push_back(float(merged_cluster.size()))
        DL_after = self.cluster_DL(merged_mode,merged_cluster,0) + self.sizes_DL(self.K-1,new_sizes)
        
        if (DL_after - DL_before) < 0:
            
            self.clusters.erase(mode1)
            self.clusters.erase(mode2)
            self.modes.erase(mode1)
            self.modes.erase(mode2)
            self.modes.insert(merged_mode)
            self.clusters[merged_mode] = merged_cluster
            for i in merged_cluster:
                self.mode_labels[i] = merged_mode
            self.DL += DL_after - DL_before
            self.K -= 1
            self.cluster_sizes.erase(mode1)
            self.cluster_sizes.erase(mode2)
            self.cluster_sizes[merged_mode] = float(merged_cluster.size())
            accepted = 1
        
        return accepted
    
    cdef long split_move(self,long max_local_moves):
        """
        perform move type 3 for a random mode
        stop local K-medoids if exceeds 'max_local_moves' iterations
        """
        cdef long accepted = 0,i,mode,rand,num_parts,m,index,best
        cdef cpp_vector[long] cluster_vector
        cdef cpp_vector[long] modes_vector
        cdef double DL_before,DL_after
        cdef long new_mode1,new_mode2,old
        cdef unordered_set[long] new_modes
        cdef unordered_set[long] previous_modes
        cdef cpp_map[long,unordered_set[long]] new_clusters
        cdef unordered_set[long] dummy
        cdef unordered_set[long] cluster 
        cdef cpp_vector[double] old_sizes
        cdef cpp_vector[double] new_sizes
        cdef long[:] new_labels = cvarray(shape=(self.n_samples,), itemsize=sizeof(long), format="l")
        
        for m in self.modes:
            modes_vector.push_back(m)
            old_sizes.push_back(self.cluster_sizes[m])
            
        rand = random_int(0,self.K - 1)
        mode = modes_vector[rand]
        cluster = self.clusters[mode]
        num_parts = cluster.size()
        while num_parts < 2:
            rand = random_int(0,self.K - 1)
            mode = modes_vector[rand]
            cluster = self.clusters[mode]
            num_parts = cluster.size()
        
        for i in cluster:
            cluster_vector.push_back(i)
        DL_before = self.cluster_DL(mode,cluster,0)  + self.sizes_DL(self.K,old_sizes) 
        
        rand = random_int(0,num_parts - 1)
        new_mode1 = cluster_vector[rand]
        rand = random_int(0,num_parts - 1)
        new_mode2 = cluster_vector[rand]
        while new_mode2 == new_mode1:
            rand = random_int(0,num_parts - 1)
            new_mode2 = cluster_vector[rand] 
        new_modes.insert(new_mode1)
        new_modes.insert(new_mode2)
        new_clusters[new_mode1] = dummy
        new_clusters[new_mode2] = dummy
        
        for i in cluster:
            best = self.closest_mode(i,new_modes)
            new_clusters[best].insert(i)
            new_labels[i] = best
        
        for index in range(max_local_moves):
            
            previous_modes = new_modes
            for m in previous_modes:
                cluster = new_clusters[m]
                best = self.mc_centroid(cluster)
                for i in cluster:
                    new_labels[i] = best
                new_clusters.erase(m)
                new_modes.erase(m)   
                new_clusters[best] = cluster
                new_modes.insert(best)
            
            for i in cluster:
                best = self.closest_mode(i,new_modes)
                old = new_labels[i]
                new_clusters[old].erase(i)
                new_clusters[best].insert(i)
                new_labels[i] = best
        
        DL_after = 0.
        modes_vector.clear()
        for m in new_modes:
            DL_after += self.cluster_DL(m,new_clusters[m],0)
            modes_vector.push_back(m)
            new_sizes.push_back(float(new_clusters[m].size()))
        for m in self.modes:
            if m != mode:
                new_sizes.push_back(self.cluster_sizes[m])
        DL_after += self.sizes_DL(self.K+1,new_sizes)

        if ((DL_after - DL_before) < 0) and (self.VI_w_mat(modes_vector[0],modes_vector[1]) > self.mode_scale):
            self.modes.erase(mode)
            self.clusters.erase(mode)
            self.K += 1
            self.cluster_sizes.erase(mode)
            for m in new_modes:
                self.modes.insert(m)
                self.clusters[m] = new_clusters[m]
                for i in self.clusters[m]:
                    self.mode_labels[i] = m
                self.cluster_sizes[m] = float(self.clusters[m].size())
            accepted = 1
        
        return accepted

    cdef long merge_split_move(self,long max_local_moves): 
        """
        perform move type 2 for a random pair of modes, then a move of type 3 for this newly merged mode
        stop local K-medoids if exceeds 'max_local_moves' iterations
        """
        cdef long accepted = 0,i,old_mode1,old_mode2,rand,num_parts,m,index,best
        cdef cpp_vector[long] cluster_vector
        cdef cpp_vector[long] modes_vector
        cdef double DL_before,DL_after
        cdef long new_mode1,new_mode2,old
        cdef unordered_set[long] new_modes
        cdef unordered_set[long] previous_modes
        cdef cpp_map[long,unordered_set[long]] new_clusters
        cdef unordered_set[long] dummy
        cdef unordered_set[long] cluster 
        cdef cpp_vector[double] old_sizes
        cdef cpp_vector[double] new_sizes 
        cdef long[:] new_labels = cvarray(shape=(self.n_samples,), itemsize=sizeof(long), format="l")
        
        if self.K == 1:
            return 0
        
        for m in self.modes:
            modes_vector.push_back(m)
            old_sizes.push_back(self.cluster_sizes[m])
        
        rand = random_int(0,self.K - 1)
        old_mode1 = modes_vector[rand]
        rand = random_int(0,self.K - 1)
        old_mode2 = modes_vector[rand]
        while old_mode2 == old_mode1:
            rand = random_int(0,self.K - 1)
            old_mode2 = modes_vector[rand]
        DL_before = self.cluster_DL(old_mode1,self.clusters[old_mode1],0) \
                  + self.cluster_DL(old_mode2,self.clusters[old_mode2],0)\
                  + self.sizes_DL(self.K,old_sizes)
        cluster = set_merge(self.clusters[old_mode1],self.clusters[old_mode2])
        num_parts = cluster.size()
        
        for i in cluster:
            cluster_vector.push_back(i)
        
        rand = random_int(0,num_parts - 1)
        new_mode1 = cluster_vector[rand]
        rand = random_int(0,num_parts - 1)
        new_mode2 = cluster_vector[rand]
        while new_mode2 == new_mode1:
            rand = random_int(0,num_parts - 1)
            new_mode2 = cluster_vector[rand] 
        new_modes.insert(new_mode1)
        new_modes.insert(new_mode2)
        new_clusters[new_mode1] = dummy
        new_clusters[new_mode2] = dummy
        
        for i in cluster:
            best = self.closest_mode(i,new_modes)
            new_clusters[best].insert(i)
            new_labels[i] = best
        
        for index in range(max_local_moves):
            
            previous_modes = new_modes
            for m in previous_modes:
                cluster = new_clusters[m]
                best = self.mc_centroid(cluster)
                for i in cluster:
                    new_labels[i] = best
                new_clusters.erase(m)
                new_modes.erase(m)   
                new_clusters[best] = cluster
                new_modes.insert(best)
            
            for i in cluster:
                best = self.closest_mode(i,new_modes)
                old = new_labels[i]
                new_clusters[old].erase(i)
                new_clusters[best].insert(i)
                new_labels[i] = best
            
        DL_after = 0.
        modes_vector.clear()
        for m in new_modes:
            DL_after += self.cluster_DL(m,new_clusters[m],0)
            new_sizes.push_back(float(new_clusters[m].size()))
            modes_vector.push_back(m)
        for m in self.modes:
            if (m != old_mode1) and (m != old_mode2):
                new_sizes.push_back(self.cluster_sizes[m])
        DL_after += self.sizes_DL(self.K,new_sizes)

        if ((DL_after - DL_before) < 0) and (self.VI_w_mat(modes_vector[0],modes_vector[1]) > self.mode_scale):
            self.modes.erase(old_mode1)
            self.clusters.erase(old_mode1)
            self.cluster_sizes.erase(old_mode1)
            self.modes.erase(old_mode2)
            self.clusters.erase(old_mode2)
            self.cluster_sizes.erase(old_mode2)
            for m in new_modes:
                self.modes.insert(m)
                self.clusters[m] = new_clusters[m]
                for i in self.clusters[m]:
                    self.mode_labels[i] = m
                self.cluster_sizes[m] = float(self.clusters[m].size())
            accepted = 1
        
        return accepted

    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)       
cpdef partition_clustering_cython(long[:,:] partitions,double[:] energies,long num_mcs,long num_init_modes,long initial_runs,\
                               double manual_mode_scale,double dist_percentile,long fix_K,long consecutive_rejects,\
                                 long max_local_moves,double Lambda,long Kprior,long MItype,long mode_info_type):
    """
    run clustering algorithm using 'clustering_state' class defined above
    inputs:
        'fix_K': fix number of clusters at K0 by only performing moves 1 and 4
        'consecutive_rejects': number of failed moves before terminating the algorithm
         all other variables defined in 'clustering_state' class methods
    writes to file 'tmp.txt' in local folder to track progress of algorithm (can comment this out if not needed)
    returns:
        modes (partition indices)
        clusters (sets of partition indices)
        mode_labels (label of mode for each partition, in order of partition index)
    """
    
    cdef long runct = 0,num_rejects = 0,accepted,move,i
    cdef double[:] type_probs
    
    if fix_K == 1:
        type_probs = np.array([0.5,0.,0.,0.5]).astype('float')   
    else:
        type_probs = np.array([0.25,0.25,0.25,0.25]).astype('float')
    f = open('tmp.txt', 'a'); f.write('Starting.... \n'); f.close()
    state = clustering_state(partitions,energies,num_mcs,Lambda,Kprior,MItype,mode_info_type)
    f = open('tmp.txt', 'a'); f.write('A:'+str(state.partition_entropies)+' \n'); f.close()
    state.initalize_cluster_data(num_init_modes,initial_runs)
    f = open('tmp.txt', 'a'); f.write('B:'+str(list(state.modes))+' \n'); f.close()
    state.compute_full_DL()
    f = open('tmp.txt', 'a'); f.write('C:'+str(state.DL)+' \n'); f.close()
    state.initialize_mode_scale(dist_percentile,manual_mode_scale)
    f = open('tmp.txt', 'a'); f.write('D:'+str(state.mode_scale)+' \n'); f.close()
        
    while num_rejects < consecutive_rejects:
        move = categorical(type_probs)
        if move == 0:
            i = random_int(0,state.n_samples-1)
            accepted = state.closest_mode_move(i)
            
        elif move == 1:
            accepted = state.merge_move()
            
        elif move == 2:
            accepted = state.split_move(max_local_moves)
            
        elif move == 3:
            accepted = state.merge_split_move(max_local_moves)
            
        if accepted == 0:
            num_rejects += 1
        else:
            num_rejects = 0
            
        f = open('tmp.txt', 'a'); f.write('Move:'+str(move)+', Acc:'+str(accepted)+\
                                          ', Modes:'+str(list(state.modes))+\
                                          ', Sizes:'+str(dict(state.cluster_sizes))+' \n'); f.close()
        
                 
    return state.modes,state.clusters,state.mode_labels
