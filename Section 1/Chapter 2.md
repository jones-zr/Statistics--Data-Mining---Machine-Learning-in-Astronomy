*Fast Computation on Massive Data Sets*

### 2.1. Data Types and Data Management Systems

##### ***Data Types***
- **Continuous:** Variables; real numbers; eg. the results of quantitative measurements and often have an associated measurement error
	- **Circular variables:** Variables where the lowest and highest values are next to each other; eg. angles on the sky, time on a clock
- **Ordinal:** Ranked variables; discrete values that can be ordered; eg. stellar spectral classification
- **Nominal:** Categorical data or attributes; descriptive and unordered data; often non-numeric; eg. types of galaxies
##### ***Data Management Systems***
- **Relational databases** (relational database management systems; RDBMSs): systems designed to serve SQL queries quickly; supports queries with concatenated, unidimensional constraints; not suitable for multidimensional queries
- **"NoSQL" systems:** data is represented in key-value pairs, which works best for text data but can be used of other data types; inefficient for tabular/array-based data

### 2.2. Analysis of Algorithmic Efficiency
- **"Big O" notation:** a simple way to describe the growth of an algorithm's runtime as a function of variables of interest
	- "Big O" == order of growth of runtime
	- $\mathcal{O}(N)$: linear runtime growth with number of data points
	- $\mathcal{O}(\log N)$logarithmic "..."
	- $\mathcal{O}(N \log N)$: common algorithm runtime, only slightly worse than purely linear
	- Any other factors that can affect runtime, eg. coding language, are considered as constants in front of the function of *N*, and become unimportant when *N* becomes large enough

### 2.3. Seven Types of Computational Problem
1. **Basic problems:**
	- Simple statistics-like computations; eg. means, variances, covariance matrices, etc.
	- Typically $\mathcal{O}(N)$ or $\mathcal{O}(N \log N)$ at worst
2. **Generalised *N*-body problems:**
	- Any problems involving distances or similar between pairs (all or many) of points (or higher-order *n*-tuples); eg. nearest-neighbour searches, correlation functions, kernel density estimates
	- Typically $\mathcal{O}(N^{2})$ or $\mathcal{O}(N^{3})$ if done straightforwardly
3. **Linear algebraic problems:**
	- Standard computational linear algebra; eg. linear systems, eigenvalue problems, inverses
	- Generally $\mathcal{O}(N)$ but can be $\mathcal{O}(N^{3})$ if the matrix of interest is $N \times N$
4. **Optimisation problems:**
	- **Optimisation:** finding the minimum or maximum on a function
	- Includes unconstrained, constrained, convex, non-convex, etc. computations
	- $\mathcal{O}(N)$ for unconstrained optimisations, and up to $\mathcal{O}(N^{3})$ for constrained optimisations
5. **Integration problems:**
	- Heavily used in the estimation of Bayesian models and typically involves high-dimensional functions; eg. Markov chain Monte Carlo (**MCMC**) algorithm ([[Chapter 5#5.8. Numerical Methods for Complex Problems (MCMC)]])
6. **Graph-theoretic problems:**
	- Involves traversal of graphs; eg. Friend-of-Friend algorithms
	- $\mathcal{O}(N)$ for the most difficult computations with discrete variables, but can be exponential in bad cases
7. **Alignment problems:**
	- Involves computations for matching two of more data objects or data sets (**cross-matching** in astronomy); closely related to *N*-point problems
	- Worst-case cost is exponential

### 2.4. Eight Strategies for Speeding Things up
- Best general algorithmic strategies and concepts for acceleration computations
1. **Trees:**
	- Divide all space at all scales, then prove that some parts can be ignored or approximated during the computation; useful for eg. nearest-neighbour searches, blind astrometric calibration, asteroid trajectory tracking
	- Difficulty when **intrinsic dimension** (the dimension of the manifold upon which the points actually lie) of the data is high; usually the intrinsic dimension of real data is much lower than its **extrinsic dimension** (the actual number of columns)
	- Can bring $\mathcal{O}(N^{2})$ problems down to $\mathcal{O}(N)$ problems
2. **Subproblem reuse:**
	- Using **dynamic programming** (bottom-up approach) and **memorisation** (top-down/recursive approach) to avoid repeating word on subproblems that have already been performed, including storing solutions and recalling them when needed
	- Memory usage can be problematic in more difficult problems
3. **Locality:**
	- Accounting for the large differences in latencies at the various levels of the memory hierarchy of a real computer system, ie. network, disk, RAM, caches
	- Keep work localised in the fastest parts of the stack and avoid jumping around unnecessarily
	- Often requires deep knowledge and exploitation of the characteristics of a specific computer system, which can change very quickly with new technological improvements
4. **Streaming:**
	- Rearranging computation so that operations can be decomposed and performed on one data point at a time; most appropriate when data actually arrives in a stream, eg. new instrument observations each day; model is updated to account for new data point rather than recalculated again and again; proper usage can yield very good approximations of statistics of entire data set
	- **Online learning** or **stochastic programming**: using streaming to approximate the optimisation in a machine learning method
	- Clear and accurate convergence can be an issue due to stochastic nature of method; streaming algorithms can just as long as a normal batch solution
5. **Function transforms:**
	- Transforming the function into another space, or otherwise decomposing it into simpler functions, eg. the Taylor series approximations and Fourier transforms
	- Typically limited to low/moderate dimensional data
6. **Sampling:**
	- Reducing the number *N* of data points on which a computation will be performed; ranges from random sampling to selecting a subset that is effectively representative of the whole data set; relatively insensitive to the dimension of the data
	- Relaxes strict approximation guarantees to probabilistic guarantees, and potentially adds nondeterminism to the results; selection process can be computationally expensive, so goal is often to reduce human time rather than computational time
	- **Active learning**: a guided special case of sampling, where points are chosen in order to maximally reduce error
7. **Parallelism:**
	-  Breaking computation into parts which can be performed simultaneously by different processors
	- Disadvantages in pragmatic relative difficulty of programming and debugging parallel codes; speed ups are at best linear in the number of machines
	- Ideal is to parallelise fast algorithms rather than brute-force algorithms
8. **Problem transformation:**
	- Change the problem itself

### 2.5. Case Studies: Speedup Strategies in Practice
- **B-trees:** one-dimensional tree data structure; employs the idea of binary searching when dealing with *range search* queries
- **Hash table:** an array with input data mapped to array indices using a hash function
- **Nearest-neighbour searches:**
	- For $N \times D$ matrix $X$ ($N$ vectors in $D$ dimensions), the $i$th point is specified as vector $x_i$, with $i = 1, ..., N$, and each $x_i$ has $D$ components, $x_{i,d}$, $d = 1, ..., D$
	- Given a query point $x$, the closest point in $X$ can be found using a distance metric
		- eg. **Euclidean metric:** $$ D(x,x_i)=\sqrt{\sum^{D}_{d=1}(x_d - x_{i,d})^{2}} $$
		- Goal is finding $x^{*}=\arg \min D(x,x_i)$
	- **Monochromatic case:** the query set (multiple query $x$'s) is the same as the reference set (all $x_i$ in $X$), eg. an all-nearest-neighbour search $$ \forall_{i}, x_{i}^{*} = \arg \min D(x_i,x_j) $$
	- **Bichromatic case:** the query and reference sets are different, eg. cross-matching objects between two catalogs
	- Can **vectorise** computation with matrixes, however:
		- Computation still scales as $\mathcal{O}(N^2)$, which is very slow
		- Memory use also scales as $\mathcal{O}(N^2)$
		- Weak to round-off errors due to the floating-point precision of the computer, can give incorrect results
	- Using **multidimensional tree structures** can increase efficiency of searches by eliminating whole regions of the parameter space
		- **Quad-trees** for 2D, **oct-trees** for 3D; split parameter space in to 4/8 equal-volume, rectilinear regions for each node
			- Decreases the cost of single-query nearest-neighbour searches from $\mathcal{O}(N^2)$ to $\mathcal{O}(\log N$), good for large $N$; building tree initially only $\mathcal{O}(N \log N)$ built time
		- **$k$d-trees:** $k$-dimensional generalisation of a quad- or oct-tree; generally implemented as binary trees, ie. each node has two children
			- Nodes are split into rectilinear regions that better represent the data than quad- or oct-trees; suitable for higher-dimensional data; available in [[SciPy Functions|SciPy]]
			- With increased $N$, computational time increases as $\mathcal{O}(N \log N)$, still way faster than brute-force/vectorisation
			- Loses efficiency when $D \gg log_2 N$; $k$d-trees split along a single dimension in each level, must go through $D$ levels before each dimension has been split
		- **Ball-trees:** Uses the triangle inequality: $$ D(x_1,x_2) + D(x_2,x_3) \geq D(x_1,x_3) $$
			- ie. if $x_1$ is far from $x_2$ and $x_2$ is near $x_3$, then $x_1$ is also far from $x_3$
			- Construct hyperspherical nodes instead of rectilinear nodes; each node is defined by a centroid $\mathbf{c}_i$ and a radius $r_i$, such that the distance for every point $\mathbf{y}$ within the node is $D(\mathbf{y}, \mathbf{c}_i) \leq r_i$
			- More efficient that $k$d-trees at high dimensions in some cases; available in [[Scikit-learn Functions|Scikit-learn]]
		- Others: **cover trees**; **maximum margin trees**; **cosine trees**; **cone trees**
- **Intrinsic dimensionality**: the "true" number of degrees of freedom of a data set; **extrinsic dimensionality**: the literal number of columns
	- The real performance of multidimensional trees is subject to intrinsic dimensionality, which is not known in advance; intrinsic dimensionality is often much smaller than extrinsic dimensionality as dimensions are generally correlated


Next chapter: [[Chapter 3]]
Extra terminology: [[Manifolds]], [[NumPy Functions]], [[SciPy Functions]], [[Scikit-learn Functions]]