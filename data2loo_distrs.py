#!/usr/bin/env python
import numpy as np

"""
  Support for quickly compute leave-one-out predictive distributions for data vectors 
  in a discrete dataset using a Naive Bayes classifier.

  The basic question is: how should predictive distribution for each left out
  data vector be updated if we add a feature to a model. It is natural to start 
  with the model without any features so the predictive distribution are based on
  the class proportions (for different classes left out). After that, it depends 
  both on vector's class and its value for the added feature. 
  
  So for each added variable V_i and for each value k (out of r_i possible) 
  of that variable, one has a  class distribution update "distribution" that 
  describes how the predictive class distribution of a vector belonging to a 
  class C and having value k for V should be updated. 
  Such a data structure U is a list of length |V| of matrices with dimensions 
  r_i x |C|x|C| (is this true?).
  
  The process to create this data structure contains two logical parts:
        - counting number of different values for each variable
        - turning counts into leave-one-out distributions

        
  Denoting the number of classes by |C|, number of variables by |V|, and
  the number of values for variable V_i by r__i we will describe the class distribution 
  update as follows: let X_j be the variables already added to the model, 
  and let the class of the data vector be C_j. Let us assume that we have 
  a leave-one-out predictive distribution .
  When we now add variable V_i with value k to the model, we want to compute the 
  updated predictive distribution P(C|X_j,V_i=k,D-{D_j}). To do so we multiply 
  the old predictive distributopn P(C|X_j,D-{D_j}) with the leave-one-out distribution
  U[i][k][C_j] (which is a vector of length |C|) and renormalize.
"""

def file2rows(filename):
    for line in open(filename):
        sline = line.strip()
        if len(sline)>0 and not sline.startswith('#'):
            yield [int(field) for field in sline.split()]

def file2valcounts(filename):
    for line in open(filename):
        sline = line.strip()
        if len(sline)>0 and not sline.startswith('#'):
            yield sline.count('\t')

def init_counts(valcounts, class_ix):
    nof_classes = valcounts[class_ix]
    return [list(map(np.zeros, valcounts)) 
            for c in range(nof_classes)]

def rows2counts(rows, counts, class_ix):
    for row in rows:
        c = row[class_ix]
        yield c
        for col,v in enumerate(row):
            counts[c][col][v] += 1

def data2loo_distrs(data, counter, counts2loo_distrs):
    counts = counter(data)
    return counts2loo_distrs(counts)

def counts2loo_counts(counts):

    def loo_counts(freqs):
        for freq in freqs:
            nof_vals = len(freq)
            loocount = np.repeat([freq], nof_vals, axis=0)
            loocount -= np.diag(freq>0) # take one out if you can 
            yield loocount

    return [list(loo_counts(c)) for c in counts]

def get_dstr_fun(name, param):
    if name == 'sNML':
        e = lambda x: 1.0 if x==0 else (((x+1.0)/x)**x)*(x+1)
    elif name == 'Dir':
        e = lambda x: x + param
    else:
        raise Exception('Unknown dstr function: %s' % name)
    return np.frompyfunc(e, 1, 1)

def loo_matrix2loo_dstrs(mx, dstr_fun):
    dstrs = dstr_fun(mx)
    return np.diag(dstrs / dstrs.sum(axis=1))

def counts2loo_probs(counts4classes, dstr_fun):
    return [[np.diag(loo_matrix2loo_dstrs(loo_matrix, dstr_fun)) 
             for loo_matrix in counts] for counts in counts4classes
           ]

def main(values_filename, data_filename, class_ix, dstr_fun_name, param=1.0):
    # read format of the data
    valcounts = np.fromiter(file2valcounts(values_filename), dtype=int)

    # count frequences in each class
    counts4classes = init_counts(valcounts, class_ix)
    rows = file2rows(data_filename)        # iterator
    classes = list(rows2counts(rows, counts4classes, class_ix))  # does two things, sorry
  
    # turn counts to distributions
    dstr_fun = get_dstr_fun(dstr_fun_name, param)
    all_u_distrs = [list(map(dstr_fun, counts)) for counts in counts4classes]
    nzer = lambda xs: xs/xs.sum()
    u_distrs = (map(dstr_fun, counts) for counts in counts4classes)
    distrs = [list(map(nzer, u_distrs)) for u_distrs in all_u_distrs]

    # turn counts  into leave-one-out counts
    loocounts = counts2loo_counts(counts4classes)
    nof_classes = valcounts[class_ix]
    class_loocounts = [loocounts[c][class_ix] for c in range(nof_classes)]
    cloocounts = np.sum(class_loocounts, axis=0)

    # turn leave-one-out counts to probabilities in predictive distributions 
    cloodstrs = loo_matrix2loo_dstrs(cloocounts, dstr_fun)
    looprobs = counts2loo_probs(loocounts, dstr_fun)

    # for each variable for each value have predicted class update probs 
    # for each true class

    # TO BE DONE - pair distrs and loo distrs and "where" the correct numbers 
    print(sum(cloodstrs))
    return(cloodstrs, looprobs)

main('iris.vd', 'iris.idt', 4, 'Dir')
