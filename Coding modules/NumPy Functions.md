### Searching ([[Chapter 2#2.5. Case Studies Speedup Strategies in Practice|As in 2.5]])
##### $numpy.searchsorted$
- Array-based searching that scales as $O(N \log N)$
##### $numpy.histogram$
- Basic hashing
- See also: $numpy.histogram2d$ and $numpy.histogramdd$
### Sorting ([[Chapter 2#2.5. Case Studies Speedup Strategies in Practice|As in 2.5]])
##### $numpy.sort$
- Sorts array smallest to largest; can also sort each column independently
- Array sorting function; a *quicksort* algorithm which scales as $O(N \log N)$
##### $numpy.argsort$
- Sorts by a particular column
---
### Descriptive statistics ([[Chapter 3#3.2. Descriptive Statistics|As in 3.2]])
##### $numpy.mean$
##### $numpy.median$
##### $numpy.var$
- Variance
##### $numpy.percentile$
##### $numpy.std$
- Standard deviation
### Distribution Functions ([[Chapter 3#3.3. Common Univariate Distribution Functions|As in 3.3]])
##### $numpy.random.multinomial$
- A multinomial (generalised binomial) distribution
##### $numpy.random.multivariate\_normal$
- Implements random samples from a multivariate Gaussian
---
