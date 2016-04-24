#ifndef HELPERS_PARSERS_CUH
#define HELPERS_PARSERS_CUH

#include "typedefs.h"

namespace helpers {

	class ui_parser {
		private:
			unsigned int* nums;
			size_type length;
		public:
			__host__ ui_parser(const char*, size_type);
			__host__ size_type size() { return length; }
			__host__ unsigned int* get_ptr() { return nums; }
			__host__ ~ui_parser() { delete[] nums; }
	};

	class coefficient_parser {
		private:
			data_type* coeffs;
			size_type length;
		public:
			__host__ coefficient_parser(const char*, size_type);
			__host__ size_type size() { return length; }
			__host__ data_type* get_ptr() { return coeffs; }
			__host__ ~coefficient_parser() { delete[] coeffs; }
	};

	class one_point_parser {
		private:
			point_type* first;
			size_type length;
		public:
			__host__ one_point_parser(const char*, size_type);
			__host__ size_type size() { return length; }
			__host__ point_type* get_ptr_a() { return first; }
			__host__ ~one_point_parser() { delete[] first; }
	};

	class three_point_parser {
		private:
			point_type* first;
			point_type* second;
			point_type* third;
			size_type length;
		public:
			__host__ three_point_parser(const char*, size_type);
			__host__ size_type size() { return length; }
			__host__ point_type* get_ptr_a() { return first; }
			__host__ point_type* get_ptr_b() { return second; }
			__host__ point_type* get_ptr_c() { return third; }
			__host__ ~three_point_parser()
			{
				delete[] first;
				delete[] second;
				delete[] third;
			}
	};

	class two_index_parser {
		private:
			index_type* first;
			index_type* second;
			size_type length;
		public:
			__host__ two_index_parser(const char*, size_type);
			__host__ size_type size() { return length; }
			__host__ index_type* get_ptr_a() { return first; }
			__host__ index_type* get_ptr_b() { return second; }
			__host__ ~two_index_parser()
			{
				delete[] first;
				delete[] second;
			}
	};

	class three_index_parser {
		private:
			index_type* first;
			index_type* second;
			index_type* third;
			size_type length;
		public:
			__host__ three_index_parser(const char*, size_type);
			__host__ size_type size() { return length; }
			__host__ index_type* get_ptr_a() { return first; }
			__host__ index_type* get_ptr_b() { return second; }
			__host__ index_type* get_ptr_c() { return third; }
			__host__ ~three_index_parser()
			{
				delete[] first;
				delete[] second;
				delete[] third;
			}
	};

}

#endif
