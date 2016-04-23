#include "typedefs.h"
#include "hermite.cuh"
#include "l2.cuh"

namespace dt {

	__device__ data_type* e_entries[NSPECIES];
	__device__ index_type* e_cols[NSPECIES];

	__device__ void
	invert_small_matrix(data_type m[9])
	{
		data_type a0 = m[0], a1 = m[1], a2 = m[2],
				  a3 = m[3], a4 = m[4], a5 = m[5],
				  a6 = m[6], a7 = m[7], a8 = m[8];
		data_type det = a0 * (a4 * a8 - a5 * a7)
		              + a1 * (a5 * a6 - a3 * a8)
		              + a2 * (a3 * a7 - a4 * a6);

		m[0] = (a4 * a8 - a5 * a7) / det;
		m[1] = (a7 * a2 - a8 * a1) / det;
		m[2] = (a3 * a7 - a4 * a6) / det;
		m[3] = (a5 * a6 - a3 * a8) / det;
		m[4] = (a8 * a0 - a6 * a2) / det;
		m[5] = (a6 * a1 - a7 * a0) / det;
		m[6] = (a3 * a7 - a4 * a6) / det;
		m[7] = (a2 * a3 - a0 * a5) / det;
		m[8] = (a0 * a4 - a1 * a3) / det;
	}

	__device__ void
	multiply_small_matrix(data_type l[9], data_type r[9])
	{
		data_type a0 = l[0], a1 = l[1], a2 = l[2],
				  a3 = l[3], a4 = l[4], a5 = l[5],
				  a6 = l[6], a7 = l[7], a8 = l[8];

		l[0] = a0 * r[0] + a1 * r[3] + a2 * r[6];
		l[1] = a0 * r[1] + a1 * r[4] + a2 * r[7];
		l[2] = a0 * r[2] + a2 * r[5] + a2 * r[8];
		l[3] = a3 * r[0] + a4 * r[3] + a5 * r[6];
		l[4] = a3 * r[1] + a4 * r[4] + a5 * r[7];
		l[5] = a3 * r[2] + a4 * r[5] + a5 * r[8];
		l[6] = a6 * r[0] + a7 * r[3] + a8 * r[6];
		l[7] = a6 * r[1] + a7 * r[4] + a8 * r[7];
		l[8] = a6 * r[2] + a7 * r[5] + a8 * r[8];
	}

	__device__ void
	subtract_small_matrix(data_type l[9], data_type r[9])
	{
		l[0] -= r[0];
		l[1] -= r[1];
		l[2] -= r[2];
		l[3] -= r[3];
		l[4] -= r[4];
		l[5] -= r[5];
		l[6] -= r[6];
		l[7] -= r[7];
		l[8] -= r[8];
	}

	__device__ data_type
	hermite::boundary_op(point_type& bpt, point_type& fpt, normal_type& n,
			data_type alpha, data_type beta)
	{
		data_type dx = bpt.x - fpt.x,
		          dy = bpt.y - fpt.y,
		          r = sqrt(dx * dx + dy * dy);
		return alpha * phi(r) - (n.x * dx + n.y * dy) * diff(phi(r));
	}

	__device__ data_type
	hermite::boundary_opop(point_type& pta, point_type& ptb, normal_type& na, normal_type& nb,
			data_type aa, data_type ab, data_type ba, data_type bb)
	{
		data_type dx = pta.x - ptb.x,
		          dy = pta.y - ptb.y,
		          r = sqrt(dx * dx + dy * dy);
		data_type nad = na.x * dx + na.y * dy,
				  nbd = nb.x * dx + nb.y * dy,
				  nab = na.x * nb.x + na.y * nb.y;
		return aa * ab * phi(r)
			+ (ab * ba * nad - aa * bb * nbd - ba * bb * nab) * diff(phi(r))
			- nad * nbd * diff(diff(phi(r)));
	}

	__device__
	hermite::hermite(size_type rows, size_type cols, index_type species,
			data_type eps, data_type* alpha, data_type beta,
			point_type* bdy_a, point_type* bdy_b, point_type* bdy_c,
			point_type* flu_a, point_type* flu_b, point_type* flu_c,
			normal_type* eta_a, normal_type* eta_b, normal_type* eta_c,
			point_type* forcing) :
		matrix(rows, cols), species(species), phi(eps)
	{
		index_type thread = blockDim.x * blockIdx.x + threadIdx.x;

		data_type A[9];
		data_type L[9];
		data_type C[9];
		data_type E[9];

		point_type &ba = bdy_a[thread];
		point_type &bb = bdy_b[thread];
		point_type &bc = bdy_c[thread];

		data_type aa = alpha[ba.index];
		data_type ab = alpha[bb.index];
		data_type ac = alpha[bc.index];

		point_type &fa = flu_a[thread];
		point_type &fb = flu_b[thread];
		point_type &fc = flu_c[thread];

		point_type &f = forcing[thread];

		normal_type& na = eta_a[thread];
		normal_type& nb = eta_b[thread];
		normal_type& nc = eta_c[thread];

		/* elements of symmetric A */
		data_type a0 = phi(0),
				  a1 = phi(norm(fa, fb)),
				  a2 = phi(norm(fa, fc)),
				  a3 = phi(norm(fb, fc));

		/* right hand side */
		data_type r0 = phi(norm(f, fa)),
				  r1 = phi(norm(f, fb)),
				  r2 = phi(norm(f, fc)),
				  r3 = boundary_op(f, ba, na, aa, beta),
				  r4 = boundary_op(f, bb, nb, ab, beta),
				  r5 = boundary_op(f, bc, nc, ac, beta);
		data_type t0, t1, t2;              // temporaries

		C[0] = L[0] = boundary_op(ba, fa, na, aa, beta);
		C[3] = L[1] = boundary_op(ba, fb, na, aa, beta);
		C[6] = L[2] = boundary_op(ba, fc, na, aa, beta);
		C[1] = L[3] = boundary_op(bb, fa, nb, ab, beta);
		C[4] = L[4] = boundary_op(bb, fb, nb, ab, beta);
		C[7] = L[5] = boundary_op(bb, fc, nb, ab, beta);
		C[2] = L[6] = boundary_op(bc, fa, nc, ac, beta);
		C[5] = L[7] = boundary_op(bc, fb, nc, ac, beta);
		C[8] = L[8] = boundary_op(bc, fc, nc, ac, beta);

		A[0] = a0; A[1] = a1; A[2] = a2;
		A[3] = a1; A[4] = a0; A[5] = a3;
		A[6] = a2; A[7] = a3; A[8] = a0;

		E[0] = aa * aa * a0 * a0;
		E[1] = boundary_opop(ba, bb, na, nb, aa, ab, beta, beta);
		E[2] = boundary_opop(ba, bc, na, nc, aa, ac, beta, beta);
		E[3] = boundary_opop(bb, ba, nb, na, ab, aa, beta, beta);
		E[4] = ab * ab * a0 * a0;
		E[5] = boundary_opop(bb, bc, nb, nc, ab, ac, beta, beta);
		E[6] = boundary_opop(bc, ba, nc, na, ac, aa, beta, beta);
		E[7] = boundary_opop(bc, bb, nc, nb, ac, ab, beta, beta);
		E[8] = ac * ac * a0 * a0;

		invert_small_matrix(A);
		multiply_small_matrix(C, A);

		r3 -= C[0] * r0 + C[1] * r1 + C[2] * r2;
		r4 -= C[3] * r0 + C[4] * r1 + C[5] * r2;
		r5 -= C[6] * r0 + C[7] * r1 + C[8] * r2;

		multiply_small_matrix(C, L);
		subtract_small_matrix(E, C);
		invert_small_matrix(E);

		// backsolve
		// 22 block
		t0 = r3; t1 = r4; t2 = r5;
		r3 = E[0] * t0 + E[1] * t1 + E[2] * t2;
		r4 = E[3] * t0 + E[4] * t1 + E[5] * t2;
		r5 = E[6] * t0 + E[7] * t1 + E[8] * t2;

		// 12 block
		r0 -= L[0] * r3 + L[1] * r4 + L[2] * r5;
		r1 -= L[3] * r3 + L[4] * r4 + L[5] * r5;
		r2 -= L[6] * r3 + L[7] * r4 + L[8] * r5;

		// 11 block
		t0 = r0; t1 = r1; t2 = r2;
		r0 = A[0] * t0 + A[1] * t1 + A[2] * t2;
		r1 = A[3] * t0 + A[4] * t1 + A[5] * t2;
		r2 = A[6] * t0 + A[7] * t1 + A[8] * t2;
	}

	__device__ data_type
	hermite::operator*(vector&)
	{
		return 0;
	}

}
