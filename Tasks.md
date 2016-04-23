## Things to do

# Project Goals
- [x] build A matrix in parallel
    * Used a method that does not store any coefficients since A is very regular.
- [x] build P matrix in parallel
    * Used a strided reduction. This could possibly be improved since it breaks global access coalescing. (Maybe.)
- [x] build E matrix in parallel
- [ ] CUDA Library to calculate eigenvalues from a large (sparse) Matrix?
- [x] Perform efficient matrix/vector multiplications
- [ ] Perform some type of timing analysis to quantify our speed up and show the utility of parallelization.

# Reach Goals

- [ ] figure out how to hook gnuPlot (or similar) into our program to automatically generate XY plots of matrix spectra
	- http://stackoverflow.com/questions/3521209/making-c-code-plot-a-graph-automatically
