#ifndef DT_MATRIX_H
#define DT_MATRIX_H

#include "vector.cuh"

namespace dt {

	class matrix {
		private:
			const size_type m_rows, m_cols;
		public:
			__device__ matrix(size_type rows, size_type cols) :
				m_rows(rows), m_cols(cols) {}
			__device__ virtual data_type operator*(vector&) = 0;
			__device__ size_type rows() { return m_rows; }
			__device__ size_type cols() { return m_cols; }
	};

}

#endif
