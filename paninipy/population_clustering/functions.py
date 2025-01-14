from collections import Counter
import numpy as np
from scipy.special import loggamma
import random
import time

def generate_synthetic(S,N,modes,alphas,betas,pis):
    """
    generate synthetic networks from the heterogeneous population model
    uses fast binomial sampling for false positive edges 
    """
    def ind2ij(ind,N):
        i = N - 2 - np.floor(np.sqrt(-8*ind + 4*N*(N-1)-7)/2.0 - 0.5)
        j = ind + i + 1 - N*(N-1)/2 + (N-i)*((N-i)-1)/2
        return int(i),int(j)
    
    K = len(modes)
    NC2 = int(N*(N-1)/2)
    nets,cluster_labels = [],[]
    for t in range(S):
        
        k = np.random.choice(range(K),p=pis)
        Mk = len(modes[k])
        net = set()
        for e in modes[k]:
            if np.random.rand() < alphas[k]: net.add(e)
            
        num_fps = np.random.binomial(NC2-Mk,betas[k])
        while num_fps > 0:
            ind = np.random.randint(NC2-1)
            i,j = ind2ij(ind,N)
            if not((i,j) in modes[k]):
                net.add((i,j))
                num_fps -= 1
        nets.append(net)

        cluster_labels.append(k)
         
    return nets,cluster_labels 

def remap_keys(Dict):
    """
    remap dict keys to first K integers
    """
    sorted_keys = sorted(list(Dict.keys()))
    for i,u in enumerate(sorted_keys):
        Dict[i] = Dict.pop(u)
    return Dict

class MDL_populations():
    
    """
    MDL population clustering class
    
    Inputs:
    edgesets: list of sets. the s-th set contains all the edges (i,j) in the s-th network in the sample (do not include the other direction (j,i) if network is undirected). 
        the order of edgesets within D only matters for contiguous clustering, where we want the edgesets to be in order of the samples in time
    N: number of nodes in each network
    K0: initial number of clusters (for discontiguous clustering, usually K0 = 1 works well; for contiguous clustering it doesn't matter)
    n_fails: number of failed reassign/merge/split/merge-split moves before terminating algorithm
    bipartite: 'None' for unipartite network populations, array [# of nodes of type 1, # of nodes of type 2] otherwise
    directed: boolean indicating whether edgesets contain directed edges
    max_runs: maximum number of allowed moves, regardless of number of fails
    
    Outputs of 'run_sims' (unconstrained description length optimization) and 'dynamic_contiguous' (restriction to contiguous clusters):
    C: dictionary with items (cluster label):(set of indices corresponding to networks in cluster)
    A: dictionary with items (cluster label):(set of edges corresponding to mode of cluster)
    L: inverse compression ratio (description length after clustering)/(description length of naive transmission)
    
    """
    
    def __init__(self, edgesets, N, K0 = 1, n_fails = 100, bipartite = None, directed = False, max_runs = np.inf):
        """
        initialize class attributes
        """
        self.edgesets = edgesets
        self.K0 = K0
        self.n_fails = n_fails
        self.S = len(self.edgesets)
        self.N = N
        self.max_runs = max_runs
        if bipartite is not None:
            self.NC2 = bipartite[0]*bipartite[1]  #bipartite networks only differentiated from unipartite ones through this term
        if directed:
            self.NC2 = self.N*(self.N-1) #directed networks only differentiated from undirected ones through this term
        else:
            self.NC2 = self.N*(self.N-1)/2
        self.C,self.E,self.A = {},{},{}
        self.attmerges,self.attsplits,self.attmergesplits = set(),set(),set()
    
    def initialize_clusters(self):
        """
        initialize K0 random clusters and find their modes as well as the total description length of this configuration
        """
        for i,s in enumerate(np.random.permutation(range(self.S))):
            k = str(i % self.K0)
            if k in self.C: 
                self.C[k].add(s)
            else: 
                self.C[k] = set()
                self.C[k].add(s)
        for k in range(self.K0):
            k = str(k)
            self.E[k] = self.generate_Ek(self.C[k])
            self.A[k] = self.update_mode(self.E[k],len(self.C[k]))
        self.L = sum([self.Lk(self.A[k],self.E[k],len(self.C[k])) for k in self.C])
            
    # def random_key(self): 
    #     """generate random key for new cluster"""
    #     return str(np.random.randint(1000000000000))
    def random_key(self):
        return str(random.randint(0, 1000000000000)) 
    
    def logchoose(self,N,K): 
        """logarithm of binomial coefficient"""
        return loggamma(N+1) - loggamma(N-K+1) - loggamma(K+1)
    
    def logmult(self,Ns): 
        """logarithm of multinomial coefficient with denominator Ns[0]!Ns[1]!..."""
        return loggamma(sum(Ns)+1) - sum(loggamma(i+1) for i in Ns)
    
    def generate_Ek(self,cluster): 
        """
        tally edge counts for networks in cluster
        """
        Ek = {}
        for s in cluster:
            for e in self.edgesets[s]:
                if e in Ek:
                    Ek[e] += 1
                else:
                    Ek[e] = 1             
        return Ek

    def update_mode(self,Ek,Sk):
        """
        generate mode from cluster edge counts by greedily removing least common edges in cluster from mode of all in-cluster edges 
        """
        Ek_vals = set(Ek.values())
        Acomplete = set(Ek.keys())
        
        if (Sk == 1): return Acomplete # return network itself if cluster only has one network
        elif not(Ek): return set() # return empty mode if networks in cluster are empty
        elif (len(Ek_vals) == 1) and (next(iter(Ek_vals)) > 1): return Acomplete # return network itself if networks are all duplicates
        
        Etil = sorted(Ek.items(),key=lambda x:x[1])
        r,tk,fk,Mk,Ak = 0,sum(Ek.values()),0,len(Ek),Acomplete.copy()
        Mmax = len(Ek)
        best_mode,deltaL,deltaL_best = 0,0,0
        while (r < Mmax): 
            
            e,Xij = Etil[r]
            Lafter = self.logchoose(self.NC2,Mk-1) + Sk*np.log(self.S/Sk) + self.logchoose(Sk*(Mk-1),tk-Xij) \
                        + self.logchoose(Sk*(self.NC2-(Mk-1)),fk+Xij)
            Lbefore = self.logchoose(self.NC2,Mk) + Sk*np.log(self.S/Sk) + self.logchoose(Sk*Mk,tk) + self.logchoose(Sk*(self.NC2-Mk),fk)

            Ak.discard(e)
            r += 1
            tk -= Xij
            fk += Xij
            Mk -= 1
                
            deltaL += (Lafter - Lbefore)
            if deltaL < deltaL_best:
                deltaL_best = deltaL
                best_mode = r
        
        Ak = Acomplete.copy()
        for r in range(best_mode): 
            e,Xij = Etil[r]
            Ak.discard(e)
        
        return Ak
            
    def Lk(self,Ak,Ek,Sk):
        """
        cluster description length as function of mode, edge counts, and size of cluster
        """
        if Sk == 0: return 0.
        Mk = len(Ak)
        tk,fk = 0,sum(Ek.values()) 
        for e in Ak:
            if e in Ek:
                tk += Ek[e]
                fk -= Ek[e]
        return self.logchoose(self.NC2,Mk) + Sk*np.log(self.S/Sk) + self.logchoose(Sk*Mk,tk) + self.logchoose(Sk*(self.NC2-Mk),fk)
            
    def move1(self,k=None):
        """
        move type 1: reassign randomly chosen network to best cluster
        """
        ks = list(self.C.keys())
        if k is None:
            k = random.choice(ks)
            
        if len(self.C) == 1: 
            return self.move3() #try splitting if only one cluster
        else:
            
            s = np.random.choice(list(self.C[k]))
            Ek_after = self.E[k].copy()
            for e in self.edgesets[s]: 
                Ek_after[e] -= 1
                if Ek_after[e] == 0: del Ek_after[e]
                
            deltaL1s = {}
            L_kbefore = self.Lk(self.A[k],self.E[k],len(self.C[k]))
            ks = list(self.C.keys())
            for kp in set(ks) - set({k}):
                
                L_kpbefore = self.Lk(self.A[kp],self.E[kp],len(self.C[kp]))
                Ekp_after = self.E[kp].copy()
                for e in self.edgesets[s]: 
                    if e in Ekp_after: Ekp_after[e] += 1
                    else: Ekp_after[e] = 1
                deltaL1s[kp] = self.Lk(self.A[kp],Ekp_after,len(self.C[kp])+1) \
                                    + self.Lk(self.A[k],Ek_after,len(self.C[k])-1) - L_kbefore - L_kpbefore
                
            if min(deltaL1s.values()) < 0:
                
                min_kp = min(deltaL1s, key=deltaL1s.get)
                self.C[k].discard(s)
                self.C[kp].add(s)
                for e in self.edgesets[s]:
                    self.E[k][e] -= 1
                    if self.E[k][e] == 0: self.E[k].pop(e, None)
                    if e in self.E[kp]: self.E[kp][e] += 1
                    else: self.E[kp][e] = 1
                knew,kpnew = self.random_key(),self.random_key()        
                self.C[knew] = self.C.pop(k)
                self.C[kpnew] = self.C.pop(kp)
                self.E[knew] = self.E.pop(k)
                self.E[kpnew] = self.E.pop(kp)
                self.A[knew] = self.A.pop(k)
                self.A[kpnew] = self.A.pop(kp)
                self.A[knew] = self.update_mode(self.E[knew],len(self.C[knew]))
                self.A[kpnew] = self.update_mode(self.E[kpnew],len(self.C[kpnew])) 
                if not(self.C[knew]):
                    del self.C[knew]
                    del self.A[knew]
                    del self.E[knew]
                return True, deltaL1s[min_kp]
            
            else:
                return False, 0
                   
    def move2(self):
        """
        move type 2: merge two randomly chosen clusters
        """
        if len(self.C) == 1: #try splitting if only one cluster
            return self.move3() 
        ks = list(self.C.keys())
        kp,kpp = np.random.choice(ks,size=2,replace=False)
        
        if ((kp,kpp) in self.attmerges) or ((kpp,kp) in self.attmerges): #check if merge already has been tried and failed
            return False,0 
        
        Ek = self.E[kp].copy()
        for e in self.E[kpp]:
            if e in Ek: Ek[e] += self.E[kpp][e]
            else: Ek[e] = self.E[kpp][e]
            
        Skp,Skpp = len(self.C[kp]),len(self.C[kpp])
        Sk = Skp + Skpp
        Ak = self.update_mode(Ek,Sk)
        deltaL2 = self.Lk(Ak,Ek,Sk) - self.Lk(self.A[kp],self.E[kp],Skp) - self.Lk(self.A[kpp],self.E[kpp],Skpp)
        
        if deltaL2 < 0:
            
            k = self.random_key()
            self.C[k] = self.C[kp].union(self.C[kpp])
            del self.C[kp]
            del self.C[kpp]
            self.E[k] = Ek.copy()
            del self.E[kp]
            del self.E[kpp]
            self.A[k] = Ak.copy()
            del self.A[kp]
            del self.A[kpp]
            return True, deltaL2   
        
        else:
            self.attmerges.add((kp,kpp)) #add to attempted merges if move fails
            return False, 0
        
    def move3(self):
        """
        move type 3: split randomly chosen cluster in two and perform K-means type algorithm to get these clusters and modes
        """
        ks = list(self.C.keys())
        k = random.choice(ks)
        
        if len(self.C[k]) == 1: #if only one network in cluster, try move 1 with this cluster
            return self.move1(k) 
        
        if k in self.attsplits: #if split already tried and failed, exit
            return False,0
        
        Sk = len(self.C[k])
        localC = {0:set(),1:set()}
        for i,s in enumerate(np.random.permutation(list(self.C[k]))): 
            localC[i % 2].add(s)
        localS,localE,localA,localL = {0:None,1:None},{0:None,1:None},{0:None,1:None},{0:None,1:None}
        for kl in [0,1]:
            localS[kl] = len(localC[kl])
            localE[kl] = self.generate_Ek(localC[kl])
            localA[kl] = self.update_mode(localE[kl],localS[kl])
            localL[kl] = self.Lk(localA[kl],localE[kl],localS[kl])
            
        #local 2-means type algorithm for identifying clusters C[k] will split into
        movement = True
        num_iters,max_2means = 0,10
        while (movement == True) and (num_iters < max_2means):
            
            movement = False
            to_move = []
            localEafter = {}
            for s in self.C[k]:
                
                for kl in [0,1]: 
                    localEafter[kl] = localE[kl].copy()
                if s in localC[0]: 
                    old,new = 0,1
                else: 
                    old,new = 1,0
                    
                for e in self.edgesets[s]:
                    
                    if e in localEafter[new]:
                        localEafter[new][e] += 1
                    else: 
                        localEafter[new][e] = 1
                    localEafter[old][e] -= 1
                    if localEafter[old][e] == 0: 
                        del localEafter[old][e]
                deltaLmove = self.Lk(localA[new],localEafter[new],len(localC[new])+1) \
                                + self.Lk(localA[old],localEafter[old],len(localC[old])-1) \
                                    - localL[0] - localL[1]
                
                if deltaLmove < 0:
                    to_move.append((s,old,new))
                    movement = True
                    
            for tup in to_move:
                
                s,old,new = tup
                localC[new].add(s)
                localC[old].discard(s)
                localS[new] += 1
                localS[old] -= 1
                
                if localS[old] == 0:
                    return self.move2()
                
                for e in self.edgesets[s]:
                    
                    if e in localE[new]: 
                        localE[new][e] += 1
                    else: 
                        localE[new][e] = 1
                        
                    localE[old][e] -= 1
                    
                    if localE[old][e] == 0:
                        del localE[old][e]
                        
            for kl in [0,1]:
                localA[kl] = self.update_mode(localE[kl],localS[kl])
                localL[kl] = self.Lk(localA[kl],localE[kl],localS[kl])
                
            num_iters += 1

        deltaL3 = self.Lk(localA[0],localE[0],localS[0]) + self.Lk(localA[1],localE[1],localS[1]) - self.Lk(self.A[k],self.E[k],Sk)   
        
        if deltaL3 < 0:
            
            kp,kpp = self.random_key(),self.random_key()
            self.C[kp] = localC[0].copy()
            self.C[kpp] = localC[1].copy()
            del self.C[k]
            self.E[kp] = localE[0].copy()
            self.E[kpp] = localE[1].copy()
            del self.E[k]
            self.A[kp] = localA[0].copy()
            self.A[kpp] = localA[1].copy()
            del self.A[k]
            return True,deltaL3 
        
        else:
            self.attsplits.add(k)
            return False, 0
        
    def move4(self):
        """
        move type 4: merge two randomly chosen clusters then split them (perform moves 2 and 3 in a row)
        """
        if len(self.C) == 1:
            return self.move3() # try split move if only a single cluster exists
        
        ks = list(self.C.keys())
        k1,k2 = np.random.choice(ks,size=2,replace=False)
        if ((k1,k2) in self.attmergesplits) or ((k2,k1) in self.attmergesplits): #check if merge-split combo already tried with these clusters
            return False,0 
        
        Ck = self.C[k1].union(self.C[k2])
        Sk = len(Ck)
        localC = {0:set(),1:set()}
        for i,s in enumerate(np.random.permutation(list(Ck))): localC[i % 2].add(s)
        localS,localE,localA,localL = {0:None,1:None},{0:None,1:None},{0:None,1:None},{0:None,1:None}
        for kl in [0,1]:
            localS[kl] = len(localC[kl])
            localE[kl] = self.generate_Ek(localC[kl])
            localA[kl] = self.update_mode(localE[kl],localS[kl])
            localL[kl] = self.Lk(localA[kl],localE[kl],localS[kl])
            
        movement = True
        num_iters,max_2means = 0,10
        while (movement == True) and (num_iters < max_2means):
            
            movement = False
            to_move = []
            localEafter = {}
            for s in Ck:
                
                for kl in [0,1]: 
                    localEafter[kl] = localE[kl].copy()
                if s in localC[0]: 
                    old,new = 0,1
                    
                else: 
                    old,new = 1,0
                    
                for e in self.edgesets[s]:
                    
                    if e in localEafter[new]: 
                        localEafter[new][e] += 1
                    else: 
                        localEafter[new][e] = 1
                        
                    localEafter[old][e] -= 1
                    if localEafter[old][e] == 0: 
                        del localEafter[old][e]
                        
                deltaLmove = self.Lk(localA[new],localEafter[new],len(localC[new])+1) \
                                + self.Lk(localA[old],localEafter[old],len(localC[old])-1) \
                                    - localL[0] - localL[1]
                if deltaLmove < 0:
                    to_move.append((s,old,new))
                    movement = True
                    
            for tup in to_move:
                
                s,old,new = tup
                localC[new].add(s)
                localC[old].discard(s)
                localS[new] += 1
                localS[old] -= 1
                if localS[old] == 0: #if all networks go into one cluster, try merge move instead
                    return self.move2()
                
                for e in self.edgesets[s]:
                    if e in localE[new]: localE[new][e] += 1
                    else: localE[new][e] = 1
                    localE[old][e] -= 1
                    if localE[old][e] == 0: 
                        del localE[old][e]
                    
            for kl in [0,1]:
                localA[kl] = self.update_mode(localE[kl],localS[kl])
                localL[kl] = self.Lk(localA[kl],localE[kl],localS[kl])
                
            num_iters += 1

        deltaL4 = self.Lk(localA[0],localE[0],localS[0]) + self.Lk(localA[1],localE[1],localS[1]) \
                                        - self.Lk(self.A[k1],self.E[k1],len(self.C[k1])) - self.Lk(self.A[k2],self.E[k2],len(self.C[k2]))
        
        if deltaL4 < 0:
            
            kp,kpp = self.random_key(),self.random_key()
            self.C[kp] = localC[0].copy()
            self.C[kpp] = localC[1].copy()
            del self.C[k1]
            del self.C[k2]
            self.E[kp] = localE[0].copy()
            self.E[kpp] = localE[1].copy()
            del self.E[k1]
            del self.E[k2]
            self.A[kp] = localA[0].copy()
            self.A[kpp] = localA[1].copy()
            del self.A[k1]
            del self.A[k2]
            return True,deltaL4 
        
        else:
            self.attmergesplits.add((k1,k2))
            return False, 0

    def run_sims(self):
        """
        run discontiguous (unconstrained) merge split simulations to identify modes and clusters that minimize the description length
        """
        nf,runs = 0,0
        move_times,move_types = [],[]
        while (nf < self.n_fails) and (runs < self.max_runs):
            
            start = time.time()
            move = np.random.choice([1,2,3,4])
            accepted,deltaL = eval('self.move'+str(move)+'()')
            if accepted: nf = 0
            else: nf += 1
            self.L += deltaL
            runs += 1
            move_times.append(time.time()-start)
            move_types.append(move)
        
        M = sum([len(D) for D in self.edgesets])
        self.L = sum([self.Lk(self.A[k],self.E[k],len(self.C[k])) for k in self.C])
        self.L /= self.logchoose(self.S*self.NC2,M) #return (minimum description length)/(naive code length transmitting all networks separately)
        self.move_times = np.array(move_times)
        self.move_types = np.array(move_types)
    
        return remap_keys(self.C),remap_keys(self.A),self.L
    
    def dynamic_contiguous(self):
        """
        miimize description length while constraining clusters to be contiguous in time (according to order of networks in 'edgesets')
        uses dynamic programming approach for exact optimization, and ignores cluster label entropy terms Sk*log(S/Sk) 
        """
        self.LMDL = {}
        self.clusters = {}
        self.modes = {}
        self.LMDL[-1] = 0
        self.clusters[-1] = {}
        self.modes[-1] = {}
        self.LMDL[0] = self.logchoose(self.NC2,len(self.edgesets[0])) + np.log(self.S)
        key0 = self.random_key()
        self.clusters[0] = {key0:set([0])}.copy()
        self.modes[0] = {key0:self.edgesets[0].copy()}.copy()
        
        start = time.time()
        for j in range(1,self.S):
            
            jkey = self.random_key()
            Lj = self.LMDL[j-1] + self.logchoose(self.NC2,len(self.edgesets[j])) + np.log(self.S)
            Cj = self.clusters[j-1].copy()
            Cj[jkey] = set([j])
            Aj = self.modes[j-1].copy()
            Aj[jkey] = self.edgesets[j].copy()
            localE = Counter(list(Aj[jkey]))
            
            for i in np.arange(j-1,-1,-1):
                
                Lprop = self.LMDL[i-1]
                Cprop = self.clusters[i-1].copy()
                Cprop[jkey] = set(range(i,j+1))
                Aprop = self.modes[i-1].copy()
                for e in self.edgesets[i]: 
                    if e in localE: localE[e] += 1
                    else: localE[e] = 1   
                    
                Aprop[jkey] = self.update_mode(localE,len(Cprop[jkey]))
                Lprop += self.Lk(Aprop[jkey],localE,len(Cprop[jkey])) - len(Cprop[jkey])*np.log(self.S/len(Cprop[jkey])) + np.log(self.S)
                
                if Lprop < Lj:
                    Lj = Lprop
                    Cj = Cprop.copy()
                    Aj = Aprop.copy()
                    
            self.LMDL[j] = Lj
            self.clusters[j] = Cj.copy()
            self.modes[j] = Aj.copy()

        self.C = self.clusters[self.S-1].copy()
        self.A = self.modes[self.S-1].copy()
        M = sum([len(D) for D in self.edgesets]) 
        self.L = self.LMDL[self.S-1]/self.logchoose(self.S*self.NC2,M)
        self.runtime = time.time() - start
        
        return remap_keys(self.C),remap_keys(self.A),self.L 
    
    def evaluate_partition(self,partition,contiguous=False):
        """
        evaluate description length of partition. contiguous option removes cluster label entropy term from description length
        """
        for s,k in enumerate(partition):
            if k in self.C: 
                self.C[k].add(s)
            else: 
                self.C[k] = set()
                self.C[k].add(s)
                
        K = len(self.C)
        for k in range(K):
            self.E[k] = self.generate_Ek(self.C[k])
            self.A[k] = self.update_mode(self.E[k],len(self.C[k]))
            
        self.L = sum([self.Lk(self.A[k],self.E[k],len(self.C[k])) for k in self.C])
        
        if contiguous: 
            self.L -= sum([len(self.C[k])*np.log(self.S/len(self.C[k])) for k in self.C])
            
        M = sum([len(D) for D in self.edgesets])
        self.L /= self.logchoose(self.S*self.NC2,M)
        
        return self.L
