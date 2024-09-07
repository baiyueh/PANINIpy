import numpy as np
from scipy.special import loggamma
from collections import Counter

class Network_hubs:
    def __init__(self,name):
        self.name = name

    def logchoose(self,N,K):
        """
        computes log of binomial coefficient
        """
        if (K <= 0) or (N <= 0):
            return 0
        return loggamma(N+1) - loggamma(K+1) - loggamma(N-K+1) 

    def logmultiset(self,N,K):
        """
        computes log of multiset coefficient
        """
        return self.logchoose(N+K-1,K)

    def hubs(self,data, N, degree_list = False, out_degrees = False, weighted = False):
        """
        inputs 
            data: if degree_list is False, input a list of tuples (i,j,weight) for directed edges i --> j; if degree_list is True, input a list of (weighted) degrees; set weights to '1' for unweighted networks
                make sure the indices i and j are in the range [0,N-1] for all edges
            N: number of nodes in network
            degree_list: set to False if data is edge list, set to True if data is list of degrees (can be in- or out-degrees)
            out_degrees: set to False if hubs should be computed using in-degree values, and set to True if hubs should be computed using out-degree values
            weighted: use multigraph encoding (applicable for integer-weighted networks and multigraphs)
            
        outputs 'results', a dictionary of results for the ER and CM encodings
            'results' has the keys 'ER', 'CM', 'AVERAGE', and 'LOUBAR' for the four methods described in the text
            for each model we have the following keys in the results dictionary:
                'hub_nodes': list of node ids corresponding to the hub nodes
                'hub_degrees': list of degrees corresponding to these hub nodes (in same order)
                'description_length' (only available for 'ER' and 'CM'): total final description length
                'compression_ratio' (only available for 'ER' and 'CM'): ratio of description length to description length of corresponding baseline (currently set to max(ER_0,CM_0))
        """
        
        """
        initialize degree list and M
        """
        logmultiset = self.logmultiset
        logchoose = self.logchoose
        if degree_list:
            degrees = np.copy(data)
            M = sum(degrees)
        else:
            M = 0
            degrees = np.zeros(N)
            for e in data:
                i,j,w = e
                if out_degrees:
                    degrees[i] += w
                else:
                    degrees[j] += w
                M += w
        
        """
        sort nodes by degree and compute counts of each unique degree value
        """
        sorted_nodes = np.argsort(degrees)[::-1]
        degree_counts = Counter(degrees)   
        unique_degrees = sorted(list(degree_counts.keys()))[::-1]
        
        """
        compute baseline compression levels with zero hub nodes
        """
        if weighted: 
            ER0 = logmultiset(N**2,M)
            CM0 = logmultiset(N,M) + sum([logmultiset(N,k) for k in degrees])

        else:
            ER0 = logchoose(N*(N-1),M) 
            CM0 = logmultiset(N,M) + sum([logchoose(N-1,k) for k in degrees])
        
        """
        scan over the unique degree values in decreasing order and consider all nodes of that degree value and above as hubs 
        update decription length and optimal number of hubs h_opt if description length decreases for the corresponding encoding
        """
        h_opt_ER,h_opt_CM = 0,0
        dl_opt_ER,dl_opt_CM = ER0,CM0
        h,hub_deg_combs,Mh = 0,0,0
        for k in unique_degrees:
            
            Nk = degree_counts[k]
            h += Nk
            Mh += Nk*k
            if weighted:
                hub_deg_combs += Nk*logmultiset(N,k)      
                dl_ER = np.log(N) + np.log(M) + logchoose(N,h) + logmultiset(h*N,Mh) + logmultiset((N-h)*N,M-Mh)
                dl_CM = np.log(N) + np.log(M) + logchoose(N,h) + logmultiset(h,Mh) + hub_deg_combs + logmultiset((N-h)*N,M-Mh) 
            else:
                hub_deg_combs += Nk*logchoose(N-1,k)
                dl_ER = np.log(N) + np.log(M) + logchoose(N,h) + logchoose(h*(N-1),Mh) + logchoose((N-h)*(N-1),M-Mh)
                dl_CM = np.log(N) + np.log(M) + logchoose(N,h) + logmultiset(h,Mh) + hub_deg_combs + logchoose((N-h)*(N-1),M-Mh) 
            
            if dl_ER < dl_opt_ER:
                dl_opt_ER = dl_ER
                h_opt_ER = h
            if dl_CM < dl_opt_CM:
                dl_opt_CM = dl_CM
                h_opt_CM = h
        
        """
        create results dict
        """
        results = {'ER':{},'CM':{},'AVG':{},'LOUBAR':{}}
        
        results['ER']['hub_nodes'] = sorted_nodes[:h_opt_ER]
        results['ER']['hub_degrees'] = [degrees[i] for i in results['ER']['hub_nodes']]
        results['ER']['description_length'] = dl_opt_ER
        results['ER']['compression_ratio'] = dl_opt_ER/max(ER0,CM0) #can change to divide by ER0 if desired

        results['CM']['hub_nodes'] = sorted_nodes[:h_opt_CM]
        results['CM']['hub_degrees'] = [degrees[i] for i in results['CM']['hub_nodes']]
        results['CM']['description_length'] = dl_opt_CM
        results['CM']['compression_ratio'] = dl_opt_CM/max(ER0,CM0) #can change to divide by CM0 if desired
        
        """
        compute Average and Loubar hubs
        """
        mean = np.mean(degrees)
        results['AVG']['hub_nodes'] = [i for i in range(N) if degrees[i] >= mean]
        results['AVG']['hub_degrees'] = [degrees[i] for i in results['AVG']['hub_nodes']]
        
        maximum = max(degrees)
        percentile = (1-mean/maximum)*100
        threshold = np.percentile(degrees,percentile)
        results['LOUBAR']['hub_nodes'] = [i for i in range(N) if degrees[i] >= threshold]
        results['LOUBAR']['hub_degrees'] = [degrees[i] for i in results['AVG']['hub_nodes']]

        return results 