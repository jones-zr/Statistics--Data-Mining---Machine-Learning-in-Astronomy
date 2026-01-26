### Multidimensional tree structures ([[Chapter 2#2.5. Case Studies Speedup Strategies in Practice|As in 2.5]])
##### $sklearn.neighbors.BallTree$
- A fast ball-tree implementation
---
### Kernel Density Estimation in D dimensions ([[Chapter 6#***KDE with Measurement Errors***|As in 6.1]])
##### $sklearn.neighbours.KernelDensity$
- KDE in $D$ dimensions using different kernels
### Gaussian mixture model in D dimensions ([[Chapter 6#***Gaussian Mixture Model***|As in 6.3]])
##### $sklearn.mixture.GaussianMixure$
- A GMM implementation
### Clustering algorithms ([[Chapter 6#6.4. Finding Clusters in Data|As in 6.4]])
##### $sklearn.cluster.KMeans$
- Implementation of $K$-means using expectation maximisation
##### $sklearn.cluster.MeanShift$
- Implementation of mean shift algorithm
---
### Principle Component Analysis ([[Chapter 7#7.3. Principal Component Analysis|As in 7.3]])
##### $sklearn.decomposition.PCA$
- Implementation of PCA; for large, higher-dimensional problems, use $RandomizedPCA$
### Nonnegative Matrix Factorisation ([[Chapter 7#7.4. Nonnegative Matrix Factorisation|As in 7.4]])
##### $sklearn.decomposition.NMF$
- Implementation of NMF
---
### Manifold Learning ([[Chapter 7#7.5. Manifold Learning|As in 7.5]])
##### $sklearn.manifold.LocallyLinearEmbedding$
- Implementation of LLE with fast tree for neighbour searching, and ARPACK for fast global optimisation
##### $sklearn.manifold.Isomap$
- Implementation of the IsoMap algorithm with fast tree for neighbour searching, and ARPACK for final eigenanalysis
---
### ICA ([[Chapter 7#7.6. Independent Component Analysis and Projection Pursuit|As in 7.6]])
##### $sklearn.decomposition.FastICA$
- Implementation of ICA based on the FastICA algorithm
