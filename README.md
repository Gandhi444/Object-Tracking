# Object Tracking
## Used method
Tracking based on Biparite graph optimised with linear_sum_assignment() function from scipy.
Graph is build by calculating weighted average of following image similarity metrics:
1. Structural Similarity Index Measure (SSIM)
2. Template Matching
3. Ratio of Bounding Box Area
4. Histogram Comparison
5. Intersection Over Union
