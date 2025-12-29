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
- For set of measurements $\{x_i\}$, the **kernel density estimator** (estimator of underlying pdf) at an arbitrary $x$ is: $$ \hat f_N (x) = \frac{1}{Nh^D} \sum_{i=1}^{N} K \left( \frac{d(x, x_i)}{h} \right) $$ where $K(u)$ is the *kernel function*, $h$ is the *bandwidth* (defines the size of the kernel)
- The local density is estimated as the *weighted mean* of all points, where the weights defined by $K(u)$ and typically decrease with distance $d(x, x_i)$
- **Kernel function** $K(u)$: a smooth function, positive at all points, normalises to unity, has a mean of 0 and variance greater than 0
	- Popular kernels include the Gaussian kernel, the top-hat (box) kernel, and the exponential kernel
- Both histograms and kernels have a parameter: kernel/bin width
- Optimal KDE bandwidth decreases as $\mathcal{O}(N^{-1/5})$ in 1D problems, and the error using the optimal bandwidth converges as $\mathcal{O} (N^{-4/5})$; histograms converge as $\mathcal{O}(N^{-2/3})$, therefore KDE is theoretically superior to the histogram as an estimator of the density
- **Epanechnikov kernel:** the optimal kernel function in turns of minimum variance is $$ K(x) = \frac{3}{4} (1-x^2) $$ for $|x| \leq 1$, and $0$ otherwise
- To obtain the height of the density estimate at a singe point $x$, must sum over $N$ kernel functions
##### ***KDE with Measurement Errors***
- To get underlying density $h(x)$, obtain an estimate $f(x)$ from noisy data, then *deconvolve* the noise pdf
- A *convolution* in real space corresponds to a *product* in Fourier space, therefore deconvolution KDE can be computed with the following steps:
	- Find the kernel density estimate of observed data $f(x)$, and compute Fourier transform $F(k)$
	- Compute Fourier transform $G(k)$ of the noise distribution $g(x)$
	- The Fourier transform of the true distribution $h(x)$ is given by $H(k) = F(k)/G(k)$; the underlying noise-free pdf $h(x)$ can be computed by inverse Fourier transform
- For some kernels and noise distributions, deconvolution can be done analytically and the result becomes another modified kernel called the *deconvolved kernel*
##### ***Extensions and Related Methods***

### 6.2. Nearest-Neighbour Density Estimation
- The implied point density at arbitrary $x$ is: $$ \hat f_K (x) = \frac{N}{V_D (d_K)} $$ where volume $V_D$ is evaluated according to dimensionality $D$
	- Assume that the underlying density filed is locally constant
	- *Error* in $\hat f_K (x)$ is $\sigma_f = K^{1/2} / V_D (d_K)$; *fractional error* is $\sigma_f / \hat f = 1/K^{1/2}$; therefore fractional accuracy increases with $K$ at the expense of spatial resolution
- For small samples, KDE and nearest-neighbour methods are noisier than Bayesian blocks method; for larger samples, all three are similar

### 6.3. Parametric Density Estimation
- 264
##### ***Gaussian Mixture Model***

##### ***Cloning Data in D > 1 Dimensions***

##### ***GMM with Errors: Extreme Deconvolution***

### 6.4. Finding Clusters in Data
- 274

##### ***General Aspects of Clustering and Unsupervised Learning***

##### ***Clustering by Sum-of-Squares Minimisation: K-Means***

##### ***Clustering by Max-Radius Minimisation: The Gonzales Algorithm***

##### ***Clustering by Nonparametric Density Estimation: Mean Shift***

##### ***Clustering Procedurally: Hierarchical Clustering***

### 6.5. Correlation Functions
- 280

##### ***Computing the n-Point Correlation Function***

### 6.6. Which Density Estimation and Clustering Algorithms Should I Use?
- 284



Next chapter: [[Chapter 7]]
New terminology: