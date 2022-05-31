# Regression w/ input dependent noise.

Experiments on input-dependent noise regression using [Bishop and Qazaz (1997): "Regression
with Input-dependent Noise: A Bayesian Treatment"](https://papers.nips.cc/paper/1996/file/b20bb95ab626d93fd976af958fbc61ba-Paper.pdf) 


## Dirs

`dataMisc/` - data for various analyses

`code/` - python code

`refs/` - references


## Files

`bishop_quad_gbf.ipynb` - Reproduction of Bishop and Qazaz paper on same type of dataset they use (Sinusoidal $y(x)$, with $x^2$ noise variance. Fitting both $y(x)$ and the log variance with 4 Gaussian basis functions.) (NB: duplicate of `code/paper_quad/bishop_quad_gbf.ipynb`)

