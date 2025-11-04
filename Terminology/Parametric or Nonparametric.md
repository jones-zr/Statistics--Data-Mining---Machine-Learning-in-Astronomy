From [Difference between Parametric and Non-Parametric Methods](https://www.geeksforgeeks.org/difference-between-parametric-and-non-parametric-methods/):
> Two prominent approaches in statistical analysis are **Parametric and Non-Parametric Methods**. While both aim to draw inferences from data, they differ in their assumptions and underlying principles.

##### ***Parametric Models***
> Parametric methods are statistical techniques that rely on specific assumptions about the underlying distribution of the population being studied. These methods typically assume that the data follows a known Probability distribution, such as the normal distribution, and estimate the parameters of this distribution using the available data.

> The basic idea behind the Parametric method is that there is a set of fixed parameters that are used to determine a probability model that is used in Machine Learning as well. Parametric methods are those methods for which we priory know that the population is normal, or if not then we can easily approximate it using a **Normal Distribution** which is possible by invoking the Central Limit Theorem. Parameters for using the normal distribution are: **Mean** and **Standard Deviation**.

**Assumptions about the data**:
- **Normality:** The data follows a normal (Gaussian) distribution.
- **Homogeneity of variance:** The variance of the population is the same across all groups.
- **Independence:** Observations are independent of each other.

**Examples of statistical tests**: t-test, ANOVA, F-test, $\chi^{2}$ test, Correlation analysis
**Examples of Machine Learning models**: Linear regression, Logistic regression, Naive Bayes, Hidden Markov Models

**Advantages**:
- **More powerful:** When the assumptions are met, parametric tests are generally more powerful than non-parametric tests, meaning they are more likely to detect a real effect when it exists.
- **More efficient:** Parametric tests require smaller sample sizes than non-parametric tests to achieve the same level of power.
- **Provide estimates of population parameters:** Parametric methods provide estimates of the population mean, variance, and other parameters, which can be used for further analysis.

**Disadvantages**:
- **Sensitive to assumptions:** If the assumptions of normality, homogeneity of variance, and independence are not met, parametric tests can be invalid and produce misleading results.
- **Limited flexibility:** Parametric methods are limited to the specific probability distribution they are based on.
- **May not capture complex relationships:** Parametric methods are not well-suited for capturing complex non-linear relationships between variables.

##### ***Nonparametric Models***
> Non-parametric methods are statistical techniques that do not rely on specific assumptions about the underlying distribution of the population being studied. These methods are often referred to as “distribution-free” methods because they make no assumptions about the shape of the distribution.

> The basic idea behind the nonparametric method is that there is no need to make any assumption of parameters for the given population; the methods don’t depend on the population. There is no fixed set of parameters available, and also there is no distribution (normal distribution, etc.) of any kind.

**Assumptions about the data**:
- **Independence:** Data points are independent and not influenced by others.
- **Random Sampling:** Data represents a random sample from the population.
- **Homogeneity of Measurement:** Measurements are consistent across all data points.

**Examples of statistical tests:** Mann-Whitney U test; Kruskal-Wallis test; Spearman’s rank correlation; Wilcoxon signed-rank test
**Examples of Machine Learning models:** K-Nearest Neighbours (KNN); Decision Trees; Support Vector Machines (SVM); Neural networks

**Advantages**:
- **Robust to outliers:** Non-parametric methods are not affected by outliers in the data, making them more reliable in situations where the data is noisy.
- **Widely applicable:** Non-parametric methods can be used with a variety of data types, including ordinal, nominal, and continuous data.
- **Easy to implement:** Non-parametric methods are often computationally simple and easy to implement, making them suitable for a wide range of users.

**Disadvantages**:
- **Less powerful:** When the assumptions of parametric methods are met, non-parametric tests are generally less powerful, meaning they are less likely to detect a real effect when it exists.
- **May require larger sample sizes:** Non-parametric tests may require larger sample sizes than parametric tests to achieve the same level of power.
- **Less information about the population:** Non-parametric methods provide less information about the population parameters than parametric methods.