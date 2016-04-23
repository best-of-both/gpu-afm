#include <fstream>
#include "parsers.cuh"

namespace helpers {

	uc_parser::uc_parser(const char* filename, size_type length) :
		length(length)
	{
		nums = new unsigned char[length];
		std::ifstream fd;
		fd.open(filename);

		for (unsigned int i = 0; i < length; ++i) {
			unsigned int datum;
			fd >> datum;
			nums[i] = (unsigned char) datum;
		}

		fd.close();
	}

	coefficient_parser::coefficient_parser(const char* filename, size_type length) :
		length(length)
	{
		coeffs = new data_type[length];
		std::ifstream fd;
		fd.open(filename);
		
		for (unsigned int i = 0; i < length; ++i)
			fd >> coeffs[i];
		
		fd.close();
	}

	one_point_parser::one_point_parser(const char* filename, size_type length) :
		length(length)
	{
		first = new point_type[length];
		std::ifstream fd;
		fd.open(filename);

		for (unsigned int i = 0; i < length; ++i) {
			point_type &p = first[i];
			fd >> p.index; fd >> p.x; fd >> p.y;
		}

		fd.close();
	}

	three_point_parser::three_point_parser(const char* filename, size_type length) :
		length(length)
	{
		first = new point_type[length];
		second = new point_type[length];
		third = new point_type[length];
		std::ifstream fd;
		fd.open(filename);

		for (unsigned int i = 0; i < length; ++i) {
			point_type &p = first[i], &q = second[i], &r = third[i];
			fd >> p.index; fd >> p.x; fd >> p.y;
			fd >> q.index; fd >> q.x; fd >> q.y;
			fd >> r.index; fd >> r.x; fd >> r.y;
		}

		fd.close();
	}

	two_index_parser::two_index_parser(const char* filename, size_type length) :
		length(length)
	{
		first = new index_type[length];
		second = new index_type[length];
		std::ifstream fd;
		fd.open(filename);

		for (unsigned int i = 0; i < length; ++i) {
			index_type x, y;
			fd >> x; fd >> y;
			first[i] = x;
			second[i] = y;
		}

		fd.close();
	}

	three_index_parser::three_index_parser(const char* filename, size_type length) :
		length(length)
	{
		first = new index_type[length];
		second = new index_type[length];
		third = new index_type[length];
		std::ifstream fd;
		fd.open(filename);

		for (unsigned int i = 0; i < length; ++i) {
			index_type x, y, z;
			fd >> x; fd >> y; fd >> z;
			first[i] = x;
			second[i] = y;
			third[i] = z;
		}

		fd.close();
	}

}
