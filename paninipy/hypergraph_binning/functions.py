import numpy as np
from math import lgamma


def logchoose(n, k):
    """
    logarithm of binomial coefficient
    """
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

def logmult(counts):
    """
    logarithm of multinomial coefficient over count data 'counts'
    """
    total = sum(counts)
    if total == 0:
        return 0.
    result = lgamma(total + 1)
    for count in counts:
        result -= lgamma(count + 1)
    return result

def logOmega(rs, cs, swap=True):
    
    """
    logarithm of number of non-negative integer matrices with row sums 'rs' and column sums 'cs'
    'swap' swaps the definition of rows and columns in the effective columns approximation of Jerdee et al for a minor accuracy improvement
    """
    
    rs = np.array([r for r in rs if r > 0])
    cs = np.array([c for c in cs if c > 0])
    if swap and len(cs) < len(rs): # Swap definitions so that (# rows) <= (# columns) to improve performance
        rs = np.array(cs)
        cs = np.array(rs)
    m = len(rs)
    N = np.sum(rs)
    
    if N == 0:
        return 0.
    if len(rs)*len(cs) == 0:
        return 0
    elif (max(rs) == 1) and (max(cs) == 1):
        return 0.
    elif (max(rs) == 1):
        return logmult(cs)
    elif (max(cs) == 1):
        return logmult(rs)
    
    alphaC = ((1-1/N)+(1-np.sum((cs/N)**2))/(m+1e-100))/(np.sum((cs/N)**2)-1/N+1e-100)
    result = -logchoose(N + m*alphaC - 1, m*alphaC - 1)
    for r in rs:
        result += logchoose(r + alphaC - 1, alphaC-1)
    for c in cs:
        result += logchoose(c + m - 1, m - 1)
        
    return result

def MDL_hypergraph_binning(X,dt,exact=True):
    """
    inputs:
        -X is list with entries of form x_i=[s_i,d_i,w_i,t_i] 
            -w_i is 'weight' of event i, counting number of interactions at a single t_i. set to 1's for unweighted event data
        -dt = (t_N-t_1)/T is time step width
    outputs:
        -best_MDL/L0: compression ratio eta for MDL-optimal temporally contiguous partition of event data X
        -labels: partition of the event data into event clusters
        -number of time steps T corresponding to width dt
    """
    
    """initialize S,D,N,T and fill time steps with corresponding event data"""
    N = int(sum([x[2] for x in X]))
    ts = [x[-1] for x in X]
    tmin,tmax = min(ts),max(ts)+1e-10
    S = len(set([x[0] for x in X]))
    D = len(set([x[1] for x in X]))
    
    cellxs,celldata = {},{}
    for i,x in enumerate(X):
        s,d,w,t = x
        cell = int(np.floor((t-tmin)/dt)) + 1
        if not(cell in celldata):
            celldata[cell] = {}
        if not((s,d) in celldata[cell]):
            celldata[cell][(s,d)] = 0
        celldata[cell][(s,d)] += w
        if not(cell in cellxs):
            cellxs[cell] = []
        cellxs[cell].append(i)
        
    T = max(list(celldata.keys()))
    
    if T == 1:
        return 1.,[0]*N,1

    for l in range(1,T+1):
        if not(l in cellxs):
            cellxs[l] = []
        if not(l in celldata):
            celldata[l] = []
    
    past_dls,past_nks = {},{} #saves past computed values for the cluster description lengths and number of events in each cluster
    
    def DL(i,j):
        """
        inputs interval indices i,j that define boundary of cluster
        returns cluster-level description length, accounting for the empty time intervals
        """
        tauk = j - i + 1
        
        if (i,j) in past_dls: 
            return past_dls[(i,j)]
        
        elif ((i-1,j) in past_dls) and (len(celldata[i-1]) == 0): 
            nk = past_nks[(i-1,j)]
            dl = past_dls[(i-1,j)] - logchoose(nk + (tauk+1) - 1, (tauk+1) - 1) + logchoose(nk + tauk - 1, tauk - 1)
            past_dls[(i,j)] = dl
            past_nks[(i,j)] = nk
            return dl
        
        elif ((i,j-1) in past_dls) and (len(celldata[j]) == 0): 
            nk = past_nks[(i,j-1)]
            dl = past_dls[(i,j-1)] - logchoose(nk + (tauk-1) - 1, (tauk-1) - 1) + logchoose(nk + tauk - 1, tauk - 1) 
            past_dls[(i,j)] = dl
            past_nks[(i,j)] = nk
            return dl
        
        G,nkts,ss,ds = {},{},{},{}
        for l in range(i,j+1):
            
            data = celldata[l]
            for edge in data:
                
                n = data[edge]
                
                if not(edge in G):
                    G[edge] = 0
                if not (l in nkts):
                    nkts[l] = 0
                if not (s in ss):
                    ss[s] = 0
                if not (d in ds):
                    ds[d] = 0
                
                G[edge] += n
                nkts[l] += n
                ss[s] += n
                ds[d] += n
                
        nk = sum(nkts.values())
        num_sds,num_kts,num_ss,num_ds = list(G.values()),list(nkts.values()),list(ss.values()),list(ds.values())
        dl = np.log((N-1)*(T-1)) + logchoose(nk + tauk - 1, tauk - 1) + logchoose(nk + S - 1, S - 1) + logchoose(nk + D - 1, D - 1) \
                + logOmega(num_ss,num_ds) + logOmega(num_sds,num_kts)
        past_dls[(i,j)] = dl
        past_nks[(i,j)] = nk

        return dl
    
    if exact: 
        """
        exact dynamic programming solution to identify MDL configuration
        """
        LMDL,LMDL_int = {0:0},{}
        for j in range(1,T+1):
            best_MDL = np.inf
            for i in range(1,j+1):
                L = LMDL[i-1] + DL(i,j) 
                if L < best_MDL:
                    best_MDL = L
                    LMDL_int[j] = i
            LMDL[j] = best_MDL

        j = T
        i = np.inf
        MDL_boundaries = []
        while i > 1:
            i = LMDL_int[j]
            MDL_boundaries.append((i,j))
            j = i-1
        MDL_boundaries = sorted(MDL_boundaries,key = lambda I:I[0])
        best_MDL = LMDL[T]
        
    else:
        """
        approximate greedy agglomerative solution to identify MDL configuration
        """
        MDL_boundaries = [(i,i) for i in range(1,T+1)]
        current_boundaries = [(i,i) for i in range(1,T+1)]
        for K in np.arange(T,1,-1).astype('int'):
            
            best_k,best_delta_dl = -100,np.inf
            for k in range(K-1):
                i1,i2 = current_boundaries[k]
                i3,i4 = current_boundaries[k+1]
                delta_dl = DL(i1,i4) - DL(i1,i2) - DL(i3,i4)
                if delta_dl < best_delta_dl:
                    best_delta_dl = delta_dl
                    best_k = k
            
            i1,i2 = current_boundaries[best_k]
            i3,i4 = current_boundaries[best_k+1]
            current_boundaries = current_boundaries[:best_k] + [(i1,i4)] + current_boundaries[best_k+2:]
            if best_delta_dl < 0:
                MDL_boundaries = [b for b in current_boundaries]
        
        best_MDL = sum([DL(b[0],b[1]) for b in MDL_boundaries])
        
        MDLK1 = DL(1,T)       
        if MDLK1 < best_MDL:
            best_MDL = MDLK1
            MDL_boundaries = [(1,T)]
    
    labels = np.zeros(N).astype('int')
    for label,bds in enumerate(MDL_boundaries):
        c = []
        for cell in range(bds[0],bds[1]+1):
            for i in cellxs[cell]:
                labels[i] = label
        
    L0 = DL(1,T) - np.log((N-1)*(T-1)) #subtract off previously ignored constant from description length 
    best_MDL -= np.log((N-1)*(T-1))
        
    return best_MDL/L0,labels,T
