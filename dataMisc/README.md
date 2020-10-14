# Data for various analyses

## Files

`PolyDataGen.nb` - generates polynomial dataset. Values with ground truth
 from a y = m x + b formula, with input dependent noise with 
 variance(x) = 10 + 5 x + 3.2 x^2.

`poly_xt_vals_200.csv` - dataset from `PolyDataGen.nb`

`sin_xt_vals_300.csv` - dataset duplicating sin function data in BISHOP paper. 
 generated in `code/paper_quad/bishop_quad_gbf.py`. Probably with numpy.random
 .seed(42) 
