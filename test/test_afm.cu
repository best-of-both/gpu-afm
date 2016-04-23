#include <iostream>
#include "typedefs.h"
#include "helmholtz.cuh"
#include "mapping.cuh"
#include "hermite.cuh"
#include "parsers.cuh"

__global__ void
gpu_main(unsigned char* is_forcing, data_type* alpha,
		point_type* bdy_a, point_type* bdy_b, point_type* bdy_c,
		point_type* eta_a, point_type* eta_b, point_type* eta_c,
		point_type* fld_a, point_type* fld_b, point_type* fld_c,
		point_type* forcing)
{
	unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x;
	dt::mapping P(64 * 64, is_forcing);
	dt::helmholtz A(64, 64, 1, 0.25);

	size_type N_F = P.rows();

	if (thread == 0)
		dt::e_cols[0] = (index_type *) malloc(N_F * sizeof(index_type));
	if (thread < N_F)
		dt::hermite(N_F, 64 * 64, 0, 5.0, alpha, 0.1,
				bdy_a, bdy_b, bdy_c, fld_a, fld_b, fld_c,
				eta_a, eta_b, eta_c, forcing);
}

int
main(void)
{
	helpers::uc_parser is_forcing("data/forcing", 4096);
	helpers::coefficient_parser alpha("data/e_coeff_a", 4096);
	helpers::three_point_parser boundary("data/e_boundary", 68);
	helpers::three_point_parser normal("data/e_boundary_normal", 68);
	helpers::three_point_parser fluid("data/e_fluid", 68);
	helpers::one_point_parser forcing("data/e_forcing", 68);

	unsigned char* is_forcing_vals = is_forcing.get_ptr();
	data_type* alpha_vals = alpha.get_ptr();
	point_type* boundary_pts_a = boundary.get_ptr_a();
	point_type* boundary_pts_b = boundary.get_ptr_b();
	point_type* boundary_pts_c = boundary.get_ptr_c();
	point_type* normal_a = normal.get_ptr_a();
	point_type* normal_b = normal.get_ptr_b();
	point_type* normal_c = normal.get_ptr_c();
	point_type* fluid_pts_a = fluid.get_ptr_a();
	point_type* fluid_pts_b = fluid.get_ptr_b();
	point_type* fluid_pts_c = fluid.get_ptr_c();
	point_type* forcing_pts = forcing.get_ptr_a();

	unsigned char* d_is_forcing;
	data_type* d_alpha;
	point_type *d_bdy_a, *d_bdy_b, *d_bdy_c,
			   *d_fld_a, *d_fld_b, *d_fld_c,
			   *d_eta_a, *d_eta_b, *d_eta_c,
			   *d_force;

	cudaMalloc((void **) &d_is_forcing, is_forcing.size() * sizeof(unsigned char));
	cudaMalloc((void **) &d_alpha, alpha.size() * sizeof(data_type));
	cudaMalloc((void **) &d_bdy_a, boundary.size() * sizeof(point_type));
	cudaMalloc((void **) &d_bdy_b, boundary.size() * sizeof(point_type));
	cudaMalloc((void **) &d_bdy_c, boundary.size() * sizeof(point_type));
	cudaMalloc((void **) &d_eta_a, normal.size() * sizeof(point_type));
	cudaMalloc((void **) &d_eta_b, normal.size() * sizeof(point_type));
	cudaMalloc((void **) &d_eta_c, normal.size() * sizeof(point_type));
	cudaMalloc((void **) &d_fld_a, fluid.size() * sizeof(point_type));
	cudaMalloc((void **) &d_fld_b, fluid.size() * sizeof(point_type));
	cudaMalloc((void **) &d_fld_c, fluid.size() * sizeof(point_type));
	cudaMalloc((void **) &d_force, forcing.size() * sizeof(point_type));

	cudaMemcpy(d_is_forcing, is_forcing_vals, is_forcing.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_alpha, alpha_vals, alpha.size() * sizeof(data_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bdy_a, boundary_pts_a, boundary.size() * sizeof(point_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bdy_b, boundary_pts_b, boundary.size() * sizeof(point_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bdy_c, boundary_pts_c, boundary.size() * sizeof(point_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_fld_a, fluid_pts_a, fluid.size() * sizeof(point_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_fld_b, fluid_pts_b, fluid.size() * sizeof(point_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_fld_c, fluid_pts_c, fluid.size() * sizeof(point_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_eta_a, normal_a, normal.size() * sizeof(point_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_eta_b, normal_b, normal.size() * sizeof(point_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_eta_c, normal_c, normal.size() * sizeof(point_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_force, forcing_pts, forcing.size() * sizeof(point_type), cudaMemcpyHostToDevice);

	gpu_main<<<4, 1024>>>(is_forcing_vals, alpha_vals,
			boundary_pts_a, boundary_pts_b, boundary_pts_c,
			normal_a, normal_b, normal_c,
			fluid_pts_a, fluid_pts_b, fluid_pts_c, forcing_pts);
}
