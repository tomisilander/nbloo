# NBLOO

## Naive Bayes model
NBLOO provides support for a fast leave-one-out (LOO) variable selection in NaiveBayes model.
In NaiveBayes (NB) model with a class variable C and predictor variable (sub)set S:
$$
P(C|S) \propto P(C,S) = P(C) \prod_{i \in S} P(X_i|C).
$$
The distributions $P(C)$ and $P(X_i|C)$ are in general computed based on data $D$ using sufficient statistics. Here we consider discrete data with $N$ i.i.d data vectors 
$D=(D_1,\ldots,D_N)$ each data vector $D_j$ consisting of class indicator $C_j$ and predictor values $(D_{j1},\ldots D_{jm})$, where $D_{ji} \in \{1,\ldots |X_i| \}$.

For discete data the distributions $P(C;D)$ and $P(X_i|C;D)$ are usually some smoothed versions of (relative) frequencies found in data.

### Bayesian smoothing
To turn frequencies into probability distribution, Bayesian smoothing adds pseudo-count $\epsilon$ to original frequencies and then normalizes. More specfically, frequencies 
$($N_1, \ldots, N_K))$ are turned into probabilties by
$$
P(k;\epsilon) \propto N_k+\epsilon.
$$ 

### SNML smoothing
To turn frequencies into probability distribution, SNML smoothing defines  
$$
P(k) = 
\begin{cases}
1 \text{, if }N_k=0, \\
(N_k+1)\frac{N_k+1}{N_k}^{N_k} \text{, otherwise.}
\end{cases}
$$ 

## Leave-one-out distributions

When doing leave-one-out cross-validation the frequencies of the values change (a little) depending on the data item that is left out. We denote the data set $D-\{D_j\}$ by $D^{-j}$ leading to distributions like 
$P(C;D^{-j})$ and $P(X_i|C;D^{-j})$. In the future we wrote $P^D$ and $P^{-j}$ for these data based distributions (with and without leaving the data item $D_j$ out respectively).

We aim at maintaining the predictive class distribution for each of the left out data items while 
we add or remove predictor variables.

If we have a predictive distribution $P^{-j}(C|D_S)$ and we add a new variable $X_i \notin S$, we have $P^{-j}(C|D_{S+\{X_i\}}) \propto P^{-j}(C|D_S) P^{-j}(D_{ji}|C)$, where
$$
P^{-j}(D_{ji}|C) = \begin{cases} 
 P^D(D_{ji}|C) \text{, if } C \neq C_j,  \\
 P^{D^{-j}}(D_{ji}|C)\text{, if } C = C_j.
\end{cases}
$$
The $P^j(C)$ is based on smoothed frequencies $\bar N_C^D = (N_{C1}, \ldots, N_{C|C|})$ of classes in $D$ with the exception that in $\bar N_C^{D^{-j}}$ $N_{C_j}$ is decreased by one.  

Also the $P^{D^{-j}}(D_{ji}|C_j)$ can be characterized as a small change to 
$P^D(D_{ji}|C_j)$. If the frequency of value $D_{ji}$ in class $C_j$ was $N_{D_{ji}|C_j}$ in data $D$, in data $D^{-j} the frequency is one less.

## Leave-one-out cross-validation 

The leave-one-out cross-validation evaluates the for a set of predictor variables as follows:

$$
LOO(S) = \frac{1}{N}\sum_{j=1}^N P^{-j}(C_j | S_j) 
       = \frac{1}{N}\sum_{j=1}^N 
             \frac {P^{-j}(C=C_j) \prod_{i \in S} P^{-j}(X_i=D_{ji}| C=C_j)}
             {\sum_{c=1}^{|C|} P^{-j}(C=c) \prod_{i \in S} P^{-j}(X_i=D_{ji}| C=c)}.
$$