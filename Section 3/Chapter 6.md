*Searching for Structure in Point Data*
- **Exploratory data analysis (EDA):** a method for exploring and quantifying structure in multivariate distribution of points
- Three classes of problems frequently encountered: density estimation, cluster finding and statistical description of observed structure
- **Density estimation:** inferring the pdf from a sample of data; also, "data smoothing"
- **Clustering:** concentrations of multivariate points, which correspond to *overdensities* when a density estimate is available
	- Clusters can be distinct objects (eg. gravitationally bound clusters of galaxies) or loose groups of sources with common properties (eg. quasars based on colour properties)
	- **Unsupervised** clustering: no prior information about the number and properties of clusters in data
	- Clusters can have specific physical meaning (eg. hot stars, quasars in multidimensional colour spaces) or carry only statistical information (eg. large-scale clustering of galaxies)
### 6.1. Nonparametric Density Estimation
- **Nonparametrically:** without specifying a functional model for the underlying density of the data; more realistic and more complicated than parametrically
- Go-to method: modelling the underlying distribution with **kernel density estimation (KDE)**
##### ***Kernel Density Estimation***
- Problem with a standard histogram is that the exact location of the bins can make significant differences to distribution, and it's not clear how to choose where the bins should go in advance; alternate method: *allow each point to have it's own bin* rather than a regular grid, and *allow bins to overlap*
	- Doesn't require a choice of bin boundaries; data drives bin positioning
	- Example of a kernel with a **top-hat distribution** centred on each point
- A rectangular kernel can lead to non-smooth distribution and suspicious spikes, so other kernels (eg. Gaussian) are often used instead
- **Narrow** kernels lead to noisy distributions, **wide** kernels lead to excessive smoothing and loss of information; a **well-tuned** kernel leads to accurate estimation of underlying distribution
- For set of measurements $\{x_i\}$, the **kernel density estimator** (estimator of underlying pdf) at an arbitrary $x$ is: $$ \hat f_N (x) = \frac{1}{Nh^D} \sum_{i=1}^{N} K \left( \frac{d(x, x_i)}{h} \right) $$where $K(u)$ is the *kernel function*, $h$ is the *bandwidth* (defines the size of the kernel)
- The local density is estimated as the *weighted mean* of all points, where the weights defined by $K(u)$ and typically decrease with distance $d(x, x_i)$
- **Kernel function** $K(u)$: a smooth function, positive at all points, normalises to unity, has a mean of 0 and variance greater than 0
	- Popular kernels include the Gaussian kernel, the top-hat (box) kernel, and the exponential kernel
- Both histograms and kernels have a parameter: kernel/bin width
- Optimal KDE bandwidth decreases as $\mathcal{O}(N^{-1/5})$ in 1D problems, and the error using the optimal bandwidth converges as $\mathcal{O} (N^{-4/5})$; histograms converge as $\mathcal{O}(N^{-2/3})$, therefore KDE is theoretically superior to the histogram as an estimator of the density
- **Epanechnikov kernel:** the optimal kernel function in turns of minimum variance is $$ K(x) = \frac{3}{4} (1-x^2) $$for $|x| \leq 1$, and $0$ otherwise
- To obtain the height of the density estimate at a singe point $x$, must sum over $N$ kernel functions
##### ***KDE with Measurement Errors***
- To get underlying density $h(x)$, obtain an estimate $f(x)$ from noisy data, then *deconvolve* the noise pdf
- A *convolution* in real space corresponds to a *product* in Fourier space, therefore deconvolution KDE can be computed with the following steps:
	- Find the kernel density estimate of observed data $f(x)$, and compute Fourier transform $F(k)$
	- Compute Fourier transform $G(k)$ of the noise distribution $g(x)$
	- The Fourier transform of the true distribution $h(x)$ is given by $H(k) = F(k)/G(k)$; the underlying noise-free pdf $h(x)$ can be computed by inverse Fourier transform
- For some kernels and noise distributions, deconvolution can be done analytically and the result becomes another modified kernel called the *deconvolved kernel*
##### ***Extensions and Related Methods***
- ...

### 6.2. Nearest-Neighbour Density Estimation
- The implied point density at arbitrary $x$ is: $$ \hat f_K (x) = \frac{N}{V_D (d_K)} $$where volume $V_D$ is evaluated according to dimensionality $D$
	- Assume that the underlying density filed is locally constant
	- *Error* in $\hat f_K (x)$ is $\sigma_f = K^{1/2} / V_D (d_K)$; *fractional error* is $\sigma_f / \hat f = 1/K^{1/2}$; therefore fractional accuracy increases with $K$ at the expense of spatial resolution
- For small samples, KDE and nearest-neighbour methods are noisier than Bayesian blocks method; for larger samples, all three are similar

### 6.3. Parametric Density Estimation
- Nonparametric KDE estimates density by affixing a kernel to each data point
- **Mixture model:** specifies underlying density model for data, uses fewer kernels, fits for kernel locations as well as widths
##### ***Gaussian Mixture Model***
- A **GMM** models the underlying density/pdf of points as a sum of Gaussians; the 1D density of a set of points is: $$ \rho(\mathbf{x}) = N p(\mathbf{x}) = N \sum_{j=1}^{M} \alpha_j \mathcal{N} (\mu_j, \Sigma_j)) $$where there are $M$ Gaussians in the model with locations $\mu_j$ and covariance $\Sigma_j$, and the weight of each Gaussian $\alpha_j$
- The likelihood can be found and optimised; more difficult at higher dimensions but the *expectation maximisation methods* from previously can be applied
- *Common misunderstanding of GMM*: the fact that the information criteria (eg. BIC/AIC) prefer an $N$-component peak doesn't necessarily mean there are $N$ components; if clusters are not near Gaussian or background is strong then number of Gaussian components in mixture may not correspond to number of clusters in data
	- Mixture models are often **more appropriate as a density estimator** instead of cluster identification
	- BIC is good to find how many statistically significant clusters are supported by the data; any number of mixture components can be used when density estimation is the only goal of analysis
- With sufficiently large number of components, mixture models approach the flexibility of nonparametric density estimation methods
- **Determining number of components** $M$:
	- Most MMs require $M$ as input to method; determining $M$ is same as any other *model selection problem* performed by *cross-validation* or using *BIC/AIC* (see [[Chapter 5#5.4. Bayesian Model Selection#***Information Criteria***|5.4.3]])
	- In reality, it is rare to find distinct, isolated and Gaussian clusters of data in astronomical distributions; almost all distributions are *continuous*; *sample size* also influences finding $M$
##### ***Cloning Data in D > 1 Dimensions***
- Cloning an arbitrary higher-dimension distribution requires as estimate of the *local density* at each point
- GMMs are good choice: can flexibly model density fields in any number of dimensions and easily generate new points within the model
- Useful idea when simulating large multidimensional data sets based on small observed samples
##### ***GMM with Errors: Extreme Deconvolution***
- **Extreme deconvolution (XD):** Bayesian estimation of multivariate densities modelled as GMMs with data that have measurement errors
- Each data point $\mathbf{x}$ is sampled from one of $M$ Gaussians with given means and variances, with the weight of each Gaussian being $\alpha_j$, therefore the pdf of $\mathbf{x}$ is: $$ p(\mathbf{x}) = \sum_j \alpha_j \mathcal{N}(\mathbf{x}|\mathbf{\mu}_j, \mathbf{\Sigma}_j) $$
- XD assumes that the noisy observations $\mathbf{x}_i$ and the true values $\mathbf{v}_i$ are related through $$ \mathbf{x}_i = \mathbf{R}_i \mathbf{v}_i + \boldsymbol{\epsilon}_i $$where $\mathbf{R}_i$ is the projection matrix, and noise $\boldsymbol{\epsilon}_i$ is assumed to be drawn from a Gaussian with zero mean and variance $\mathbf{S}_i$
- **XD aims:** Given matrices $\mathbf{R}_i$ and $\mathbf{S}_i$, XD aims to find parameters $\boldsymbol{\mu}_i$, $\boldsymbol{\Sigma}_i$ of the underlying Gaussians, and weights $\alpha_i$ in a way that maximises the likelihood of the observed data
- **Expectation Maximised (EM)** approach (see [[Chapter 4#***The Basics of the Expectation Maximisation Algorithm***|4.4.3]]) results in an iterative procedure that converges to a *local maximum of the likelihood*

### 6.4. Finding Clusters in Data
##### ***General Aspects of Clustering and Unsupervised Learning***
- **Clustering:** structure in multivariate point data, concentration of points; *overdensities* when density estimate is available; partitioning data into smaller parts according to some criteria
- **Unsupervised:** no prior information about number and properties of clusters
- Objective criteria for clustering is more vague than for prediction tasks
##### ***Clustering by Sum-of-Squares Minimisation: K-Means***
- **K-means:** partitioning of points into $K$ disjoint subsets $C_k$ with each subset containint $N_k$ points, so $$ \sum_{k=i}^{k} \sum_{i \in C_k} || x_i - \mu_k ||^2 $$is minimised, where $\mu_k$ is the mean of the points in set $C_k$
- **Procedure**:
	- Choose the centroid, $\mu_k$, of each of the $K$ clusters
	- Assign each point to the cluster it is closest to (according to $C(x_i) = \text{arg min}_k ||x_i - \mu_k ||$)
	- Update centroids by recomputing $\mu_k$ to include new points
	- Continue until no new assignments to clusters/no points left
- Process doesn't guarantee a globally optimal minimum, but it never increases the sum-of-squares error
- Process is run multiple times with different starting centroid values, and result with lowest sum-of-squares error is used
##### ***Clustering by Max-Radius Minimisation: The Gonzales Algorithm***
- **Max-radius minimisation:** minimise the maximum radius of a cluster, $$ \min_k \max_{x_i \in C_k} ||x_i - \mu_k|| $$where $\mu_k$ is the assigned centre of each cluster
- **Gonzales Algorithm:** algorithm that starts with no clusters and progressively adds one cluster at a time (arbitrarily selects cluster centres from data)
	- Then find point $x_i$ which maximises the distance from the centres of existing clusters and set that as next cluster centre
	- Repeat until achieve $K$ clusters
	- Each point is then assigned to it's nearest cluster centre
##### ***Clustering by Nonparametric Density Estimation: Mean Shift***
- Define clusters in terms of modes or peaks of the nonparametric density estimate
- **Mean shift algorithm:** technique to find local modes in a kernel density estimate
	- Concept: move data points int he direction of the log of the gradient of the density of the data, until they converge at the peaks
	- Number of modes $K$ is found implicitly by the method
	- Convergence of procedure is defined by the bandwidth $h$ of the kernel and the parameterisation of $a$
	- eg. for the Epanechnikov kernel (see [[Chapter 6#***Kernel Density Estimation***|6.6.1]]) and the value $$ a = \frac{h^2}{D+2} $$the update rule reduces to the form $$ x_{i}^{m+1} = \text{mean position of points } x_i^m \text{ within distance } h \text{ of } x_i^m $$
##### ***Clustering Procedurally: Hierarchical Clustering***
- **Procedural method:** a method not formally related to some function of the underlying density
- **Hierarchical clustering:** relaxes need to specify $K$ by finding all clusters at all scales
	- When two points are in the same cluster at level $m$, and remain together at all subsequent levels; visualised with a tree diagram/dendrogram
- Procedure:
	- Partition data into $N$ clusters, one per data point
	- Merge the nearest pair of clusters based on some definition of distance
	- Repeat until the $N$th partition contains one cluster
- Can be top-down (**divisive**) or bottom-up (**agglomerative**) procedure
- **Friends-of-friends** clustering: single-linkage hierarchical clustering; often used in cluster analysis for $N$-body simulations
- Most distance algorithms are too slow for large data sets, eg. a minimum spanning tree is $\mathcal{O}(N^3)$ to compute

### 6.5. Correlation Functions
- Can characterise how far, and on what scales, a distribution of points differs from a random distribution; can be used for **testing models of structure formation and evolution** directly against data
- Defined by noting that the probability of finding a point in a volume element, $dV$, is proportional to the density of points, $\rho$
	- The **probability of finding a pair of points in two volume elements separated by a distance** is: $$ dP_{12} = \rho^2 dV_1 dV_2 (1+\xi(r)) $$where $\xi(x)$ is the **two-point correlation function**
- $\xi(r)$ describes the **excess probability** of finding a pair of points, as a function of separation, compared to a random distribution
	- Positive, negative, zero amplitudes in $\xi(r)$ == correlated, anticorrelated, random distributions
- $\xi(r)$ relates directly to the power spectrum through the Fourier transform, $$ \xi(r) = \frac{1}{2 \pi^2} \int k^2 P(k) \frac{\sin(kr)}{kr} dk $$where the scale/wavelength of a fluctuation ($\lambda$) is related to the wave number $k$ by $k = 2\pi / \lambda$
	- Therefore, $\xi(r)$ can be used to describe the density fluctuations of sources
- In galaxy distribution studies, $\xi(r)$ is often **parameterised in terms of a power law**, $$ \xi(r) = \left( \frac{r}{r_0} \right)^{-\gamma} $$where $r_0$ is the clustering scale length and $\gamma$ is the power law exponent
- The **angular correlation function** of apparent positions of objects on the sky is: $$ w(\theta) = \left( \frac{\theta}{\theta_0} \right)^{\delta} $$where $\delta = 1 - \gamma$
- Correlation functions can be **extended to higher dimensions**/orders; these can be expressed in terms of the probability of finding a given configuration of points, eg. the three-point correlation function: $$ dP_{123} = \rho^3 dV_1 dV_2 dV_3 (1 + \xi(r_{12}) + \xi(r_{23}) + \xi(r_{13}) + \zeta(r_{12}, r_{23}, r_{13})) $$where $\zeta$ is the **reduced** or **connected three-point correlation function**, and doesn't depend on the lower-order correlation functions
##### ***Computing the n-Point Correlation Function***
- Random distribution is generated with the same selection function as the data
- Computational cost of estimating the correlation function is dominated by the size of the random data set
- Estimator of $\xi(r)$: $$ \hat \xi(r) = \frac{DD(r)}{RR(r)} - 1 $$where $DD(r)$ is the number of pairs of data points, and $RR(r)$ is the number of pairs of random points
	- Other estimators include the **Landy-Szaley estimator**, which can extended to higher-order correlation functions
- For clustering on small scales, a **ball-tree** based implementation offers significant improvement in computation time for the two-point correlation function over a brute-force method
	- Naive computation of $n$-point correlation function is $\mathcal{O}(N^n)$; space-partitioning trees can reduce this computation to $\mathcal{O}(N^{\log n})$

### 6.6. Which Density Estimation and Clustering Algorithms Should I Use?
- Four measures of "goodness" of methods: accuracy, interpretability, simplicity, speed
- In general:
	- **Highest accuracies** require nonparametric methods
	- Parametric models are **more interpretable** as the meanings of each part of the model are usually clear
- **Table 6.1** summarises authors' assessments of all methods in this chapter


Next chapter: [[Chapter 7]]
New terminology: