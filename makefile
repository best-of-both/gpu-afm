NVCC=/usr/local/cuda/bin/nvcc
CC=cc

INCLUDE=-I/usr/local/cuda/include \
        -I/usr/local/cuda/samples/common/inc \
        -Iinclude

LIBDIR=-L/usr/local/cuda/lib64
LIBS=-lcublas_static -lculibos

SOURCE=main.cu
EXECUTABLE=AFM_project

$(EXECUTABLE): $(SOURCE)
	$(NVCC) -ccbin=$(CC) -g $(INCLUDE) $(LIBDIR) $< -o $@ $(LIBS)
	
objs/forcing.o: src/forcing.cu
	$(NVCC) -ccbin=$(CC) $(INCLUDE) $(LIBS) $< -c -o $@
	
objs/mapping_matrix.o: src/mapping_matrix.cu
	$(NVCC) -ccbin=$(CC) $(INCLUDE) $(LIBS) $< -c -o $@
	
test_fpm: test/test_forcing_point_map.cu objs/forcing.o objs/mapping_matrix.o
	$(NVCC) -ccbin=$(CC) $(INCLUDE) $(LIBS) $^ -o $@

test: test_fpm

clean:

