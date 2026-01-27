*Dimensionality and Its Reduction*
### 7.1. The Curse of Dimensionality
- The more selection conditions (ie. dimensions) you adopt, the tinier the chance of finding the ideal choice; curse of dimensionality impacts the size of the data required to constrain the model, the complexity of the model itself, and the search time required to optimise the model
	- eg. the fraction of points within a search radius $r$ (relative to the full space) will tend to zero as the dimensionality grows => the number of points in a data set required to evenly sample this hypervolume will grow exponentially with dimension

### 7.2. The Data Sets Used in This Chapter
- ...

### 7.3. Principal Component Analysis
- Eg. for a distribution of points that are strongly correlated but the correlation does not align with the initial choice of axes (Figure 7.2 in book)
	- Should rotate the axes to align with the correlation; choose rotation to maximise the ability to discriminate between data points => rotation that **maximises variance** along the resulting axes
	- Define the first axes (**principal component**) in the direction of maximal variance, and second principal component to be orthogonal to the first to maximise residual variance, etc.
	- Mathematically equivalent to a regression than **minimises the square of orthogonal distances** from the points in the principal axes
- **PCA** (or Karhunen-Loève or Hotelling transform) is a **linear transform** applied to multivariate data that defines a set of uncorrelated axes **ordered by the variance** captured by each new axis
	- New axes are aligned with the direction of maximum variance within the data
##### ***The Derivation of Principle Component Analysis***
- For set of data $\{x_i\}$ with $N$ observations made up of $K$ measured features each, $\{x_i\}$ is centred by subtracting the mean of each feature then writing the matrix $N \times K = X$
- Covariance of $X$ is given by $C_X$: $$ C_X = \frac{1}{N-1}X^T X $$
	- Non-zero off-diagonal terms are because there exist correlations between the measured features
- PCA identifies a projection of $\{x_i\}$, $R$, that is aligned with the directions of maximal variance; projection written as $Y = XR$ with covariance: $$ C_Y = R^T X^T XR = R^T C_X R $$
- First principal component of $R$, $r_1$, is defined as the projection with the maximal variance; columns of $R$ are **eigenvectors**/principal components, diagonal values of $C_Y$ define the amount of variance of each component with: $$C_X = R C_Y R^T$$
	- Ordering the eigenvectors by their eigenvalue defines the set of principal components for $X$
- Efficient computation of principal components: **singular value decomposition (SVD)** (figure 7.3)
	- SVD factorises $X$ ($N \times K$ matrix) into $U \Sigma V^T$ where: $$ U \Sigma V^T = \frac{1}{\sqrt{N-1}} X $$where the columns of $U$ are the *left-singular vectors*, the columns of $V$ are the *right-singular vectors*, and $\Sigma$ is always a square matrix of singular values and size $[R \times R]$ where $R = \text{min}(N,K)$
	- The diagonal matrix of eigenvalues $C_Y$ is equivalent to the square of the singular values ($\Sigma^2 = C_Y$) => the principal components can be computed from the SVD of $X$ without constructing $C_X$
	- SVD can also find correlation matrix $M_X$: $$ M_X = U \Sigma^2 U^T $$
	- Therefore, three equivalent ways of computing the principal components $R$ and eigenvalues $C_X$: the SVD of $X$; the eigenvalue decomposition of $C_X$; or the eigenvalue decomposition of $M_X$
		- Optimal procedure depends on the data size $N$ and dimensionality $K$
		- For $N \gg K$: eigenvalue decomposition of $K \times K$ covariance matrix $C_X$
		- For $K \gg N$: eigenvalue decomposition of $N \times N$ correlation matrix $M_X$
		- For intermediate case: computation of the SVD of $X$
##### ***The Application of PCA***
- Before creating matrix $X$ (by subtracting mean of each dimension), data are processed to be maximally informative, eg.:
	- For heterogeneous data (eg. galaxy flux): columns are often divided by variance, to make variance of each feature is comparable
	- For spectra or images: rows are normalised so integrated flux of each object is unity, to remove uninteresting correlations based on brightness
- For galaxy spectra, the principal directions found in high-dimensional data are called **eigenspectra** => a spectrum can be represented by the sum of it's eigenspectra; ordered by eigenvalues that reflect amount of variance in each eigenspectra
- Sum of eigenvalues equals total variance/"information" of entire data set; most variance/information is held in the first handful of eigenvectors; this is how PCA allows for dimensionality reduction
	- Truncation of eigencomponents at $r < R$ will exclude components with small eigenvalues, which predominantly reflect noise within data set
- Eigenvectors with large eigenvalues are predominantly **low-order components** (continuum); **higher-order components** with smaller eigenvalues are predominantly sharp features (emission lines); remaining eigenvectors are noise; combination of these eigenvectors can describe any input spectra
- **Notable aspects of PCA:**
	- Statistically orthogonal components correlate strongly with specific physical properties of galaxies (eg. star formation)
	- Astrophysically interesting components within the spectra (eg. spectral lines or transients) may not be in the largest PCA components; must be careful with truncation
	- Sums of linear components may not always efficiently reconstruct data features (eg. broad-line quasars need 30 components to reproduce spectra, star-forming galaxies need 10)
- **Choosing level of truncation in an expansion:**
	- $r$ often picked on empirical basis
	- Common criterion for picking $r$: $$ \frac{\sum_{i}^{i=r} \sigma_i}{\sum_{i}^{i=R} \sigma_i} < \alpha $$where $\alpha$ is the specified fraction of variance we wish to capture, and $\sigma_i$ are the eigenvalues/diagonals of the matrix $\Sigma$
		- Typical values of $\alpha$ range from 0.70 to 0.95, though is sensitive to the shape of the **scree plot** (eigenvalue number vs normalised/cumulative eigenvalues; eg. figure 7.4 in book)
##### ***PCA with Missing Data***
- Truncation of the expansion provides a signal-to-noise filtering of the data; **PCA bases should be able to correct for missing elements** within the data (eg. detector glitches, variable noise, masking effects)
- Complication: eigenspectra are only defined to be orthogonal over the spectral range on which they are constructed; if data vector does not fully cover that space, then projecting onto the eigenbases results in biased expansion coefficients
##### Scaling to Large Data Sets
- Limitations of PCA: **computational and memory requirements** of the SVD which scale as $\mathcal{O}(D^3)$ and $\mathcal{O}(2 \times D \times D)$ respectively; computational requirements of SVD are set by rank of $X$
- **Eigenvalue decompositions** (EVD) are often more efficient; even then, memory is challenging

### 7.4. Nonnegative Matrix Factorisation
- In PCA, principal components can be positive or negative
- Nonnegative matrix factorisation (NMF) assumes any data matrix can be factored into two matrices, $W$ and $Y$, such that: $$ X = WY $$where both $W$ and $Y$ are **nonnegative**
- Nonnegative bases can be derived using a simple update rule; this does not guarantee nonlocal minima
- Components derived by NMF are broadly consistent with PCA but with different ordering of the basis functions

### 7.5. Manifold Learning
- Real data often contains nonlinear features which are hard to capture with a linear basis like PCA or NMF (eg. emission lines)
- **S-curve** data set test case (figure 7.8 in book): data is drawn from a 2D manifold and embedded in a 3D space
	- PCA cannot capture the intrinsic information; no linear projection in which distant parts of the manifold do not overlap
	- Manifold learning techniques can unwrap/unfold the surface so underlying structure becomes clear
	- More powerful with eg. galaxy or quasar spectra, up to 4000 dimensions
##### ***Locally Linear Embedding***
- **LLE:** unsupervised learning algorithm; embeds high-dimensional data in a lower-dimensional space while preserving geometry of local neighbourhoods of points (relation of points with $k$ nearest neighbours)
- Steps:
	1. For each point, derive a set of weights which best reconstruct the point from its $k$ nearest neighbours; find weight matrix $W$ that encodes the global geometric properties of the data set, or how each point relates to all others
	2. New lower-dimensional data set, $Y$, is found which maintains the neighbourhood relationships of the original $X$ data set
- Point 1 requires a nearest-neighbour search, then a least-squares solution to the corresponding row of $W$
- Point 2 requires an eigenvalue decomposition of the matrix $C_W \equiv (I - W)^T (I - W)$
##### ***IsoMap***
- **Isometric mapping:** multidimensional scaling (MDS) framework; reconstructs data set from a matrix of pairwise distances
- For data set with $N \times K$ matrix $X$, can compute an $N \times N$ distance matrix $D_X$ such that $[D_X]_{ij}$ contains the distance between points $i$ and $j$
- Given $D_X$, MDS finds $Y$ which minimises the error: $$ \mathcal{E}_{XY} = | \tau (D_X) - \tau (D_Y)|^2 $$where $\tau$ is an operator: $$ \tau(D) = \frac{HSH}{2} $$where $S$ is the matrix of square distances $S_{ij} = D_{ij}^{2}$, and $H$ is the centring matrix $H_{ij} = \delta_{ij} - 1/N$
- Recovers nonlinear structure by approximating geodesic curves which lie within the embedded manifold, and computes the distances between each point in the dataset along these geodesic curves
##### ***Weakness of Manifold Learning***
- **Noise or gaps in data:** rely on relations between points; missing points/strong noise disrupt relations
- **Tuning parameters:** currently no solid recommendation for choosing the optimal set of $k$ neighbours for a given embedding
- **Dimensionality:** no guarantee that the embedded manifold is unidimensional
- **Sensitivity to outliers:** outliers between different regions of the manifold can act to "short-circuit" the manifold; algorithm cannot find correct embedding
- **Reconstruction from the manifold:** any mapping from the embedded space to the higher-dimensional input space must be through a reconstruction based on the location of the nearest-neighbours; projection from manifold learning therefore cannot compress data like PCA can

### 7.6. Independent Component Analysis and Projection Pursuit
- **ICA:** uses statistical independence to isolate individual signals (ala the *cocktail party problem*); eg. each galaxy spectrum is a "microphone" picking up a linear combination of input signals from individual stars and HII regions
	- Example can be written in matrix form as: $$ X = AS $$where $X$ and $S$ are matrices for input spectra and stellar spectra, and $A$ is a set of appropriate mixing amplitudes
	- Main principle: the input signals should be **statistically independent**
- **Projection pursuit:** subset of ICA
##### ***The Application of ICA to Astronomical Data***
- Important: prepossessing of input data; mean vector is removed to centre the data, then eigenvalue decomposition of covariance matrix
- FastICA can evaluate each independent component simultaneously
- Size of the mixing matrix increases computational complexity, makes it impractical to calculate the weight matrix directly; reduction in complexity of input signals through PCA is often applied to ICA problems

### 7.7. Which Dimensionality Reduction Technique Should I Use?
- **Accuracy:**
	- PCA gives best square reconstruction error for any given $K$
	- LLE minimises square reconstruction error, but in a nonlinear fashion
	- NMF minimises a notion of reconstruction error, under nonnegativity constraints
	- IsoMap minimises the difference between each pairwise distance in the original space and its counterpart in the reconstructed space
	- NMF-type approaches have a performance advantage over PCA, ICA, and manifold learning for low S/N data
- **Interpretability:**
	- NMF yields more sensible components than PCA; constraints ensure that reconstructions are valid spectra or images
	- PCA advantage is ability to estimate the importance of each principal component
- **Scalability:**
	- Linear methods for dimensionality reductions are usually tractable even for high $N$
	- Nonlinear methods like LLE and IsoMap, first step is most expensive (nearest-neighbour computation)
- **Simplicity:**
	- PCA is "convex" (usable out of the box)
	- ICA and NMF are nonconvex
	- LLE and IsoMap are convex but require careful evaluation of $k$
	- PCA, NMF, ICA can be applied to out-of-sample data; LLE, IsoMap require that the test data be added to existing training set and the whole model retrained
- **Extra:**
	- PCA as a maximum-likelihood-based model is sensitive to outliers; PCA is simplest and most useful technique
	- No clear way to handle missing values with manifold methods
	- NMF maps to many astronomical problems


Next chapter: [[Chapter 8]]
New terminology: