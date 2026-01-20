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
- 300
##### ***PCA with Missing Data***
##### Scaling to Large Data Sets

### 7.4. Nonnegative Matrix Factorisation
- 306

### 7.5. Manifold Learning
- 308
##### ***Locally Linear Embedding***
##### ***IsoMap***
##### ***Weakness of Manifold Learning***

### 7.6. Independent Component Analysis and Projection Pursuit
- 315
##### ***The Application of ICA to Astronomical Data***

### 7.7. Which Dimensionality Reduction Technique Should I Use?
- 317
- 320


Next chapter: [[Chapter 8]]
New terminology: