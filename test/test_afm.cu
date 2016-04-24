#include <iostream>
#include "typedefs.h"
#include "helmholtz.cuh"
#include "mapping.cuh"
#include "hermite.cuh"
#include "parsers.cuh"
#include "debug.cuh"

__global__ void
gpu_main(unsigned int* is_forcing, data_type* alpha,
		point_type* bdy_a, point_type* bdy_b, point_type* bdy_c,
		point_type* eta_a, point_type* eta_b, point_type* eta_c,
		point_type* fld_a, point_type* fld_b, point_type* fld_c,
		point_type* forcing)
{
	unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x;
	bool is_forcing_pt = thread < 64 * 64 ? is_forcing[thread] != 0 : false;
	index_type forcing_index = dt::compute_index(64 * 64, is_forcing);
	unsigned int num_forcing = is_forcing[64 * 64 - 1];

	dt::helmholtz A(64, 64, 1, 0.25);
	dt::mapping P(64 * 64, num_forcing, is_forcing_pt, forcing_index);
	dt::hermite(num_forcing, 64 * 64, 0, 5.0, alpha, 0.1,
			bdy_a, bdy_b, bdy_c, fld_a, fld_b, fld_c,
			eta_a, eta_b, eta_c, forcing);
}

int
main(void)
{
	helpers::ui_parser is_forcing("data/forcing", 4096);
	helpers::coefficient_parser alpha("data/e_coeff_a", 4096);
	helpers::three_point_parser boundary("data/e_boundary", 68);
	helpers::three_point_parser normal("data/e_boundary_normal", 68);
	helpers::three_point_parser fluid("data/e_fluid", 68);
	helpers::one_point_parser forcing("data/e_forcing", 68);


	unsigned int* is_forcing_vals = is_forcing.get_ptr();
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

	unsigned int* d_is_forcing;
	data_type* d_alpha;
	point_type *d_bdy_a, *d_bdy_b, *d_bdy_c,
			   *d_fld_a, *d_fld_b, *d_fld_c,
			   *d_eta_a, *d_eta_b, *d_eta_c,
			   *d_force;

	CHECK(cudaMalloc((void **) &d_is_forcing, is_forcing.size() * sizeof(unsigned int)));
	CHECK(cudaMalloc((void **) &d_alpha, alpha.size() * sizeof(data_type)));
	CHECK(cudaMalloc((void **) &d_bdy_a, boundary.size() * sizeof(point_type)));
	CHECK(cudaMalloc((void **) &d_bdy_b, boundary.size() * sizeof(point_type)));
	CHECK(cudaMalloc((void **) &d_bdy_c, boundary.size() * sizeof(point_type)));
	CHECK(cudaMalloc((void **) &d_eta_a, normal.size() * sizeof(point_type)));
	CHECK(cudaMalloc((void **) &d_eta_b, normal.size() * sizeof(point_type)));
	CHECK(cudaMalloc((void **) &d_eta_c, normal.size() * sizeof(point_type)));
	CHECK(cudaMalloc((void **) &d_fld_a, fluid.size() * sizeof(point_type)));
	CHECK(cudaMalloc((void **) &d_fld_b, fluid.size() * sizeof(point_type)));
	CHECK(cudaMalloc((void **) &d_fld_c, fluid.size() * sizeof(point_type)));
	CHECK(cudaMalloc((void **) &d_force, forcing.size() * sizeof(point_type)));

	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start, 0));

	CHECK(cudaMemcpy(d_is_forcing, is_forcing_vals, is_forcing.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_alpha, alpha_vals, alpha.size() * sizeof(data_type), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_bdy_a, boundary_pts_a, boundary.size() * sizeof(point_type), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_bdy_b, boundary_pts_b, boundary.size() * sizeof(point_type), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_bdy_c, boundary_pts_c, boundary.size() * sizeof(point_type), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_fld_a, fluid_pts_a, fluid.size() * sizeof(point_type), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_fld_b, fluid_pts_b, fluid.size() * sizeof(point_type), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_fld_c, fluid_pts_c, fluid.size() * sizeof(point_type), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_eta_a, normal_a, normal.size() * sizeof(point_type), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_eta_b, normal_b, normal.size() * sizeof(point_type), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_eta_c, normal_c, normal.size() * sizeof(point_type), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_force, forcing_pts, forcing.size() * sizeof(point_type), cudaMemcpyHostToDevice));

	gpu_main<<<4, 1024>>>(
			d_is_forcing, d_alpha,
			d_bdy_a, d_bdy_b, d_bdy_c,
			d_eta_a, d_eta_b, d_eta_c,
			d_fld_a, d_fld_b, d_fld_c,
			d_force);
	CHECK(cudaGetLastError());

	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchronize(stop));
	float diff;
	CHECK(cudaEventElapsedTime(&diff, start, stop));

	std::cout << "elapsed time: " << diff << std::endl;

	CHECK(cudaFree(d_is_forcing));
	CHECK(cudaFree(d_alpha));
	CHECK(cudaFree(d_bdy_a));
	CHECK(cudaFree(d_bdy_b));
	CHECK(cudaFree(d_bdy_c));
	CHECK(cudaFree(d_fld_a));
	CHECK(cudaFree(d_fld_b));
	CHECK(cudaFree(d_fld_c));
	CHECK(cudaFree(d_eta_a));
	CHECK(cudaFree(d_eta_b));
	CHECK(cudaFree(d_eta_c));
	CHECK(cudaFree(d_force));
}
