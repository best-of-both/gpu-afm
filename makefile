CC=/usr/local/cuda/bin/nvcc
INCLUDE=-I/usr/local/cuda/include \
        -I/usr/local/cuda/samples/common/inc

LIBDIR=-L/usr/local/cuda/lib64
LIBS=-lcublas_static -lculibos

SOURCE=main.cu
EXECUTABLE=AFM_project

$(EXECUTABLE): $(SOURCE)
	$(CC) -g $(INCLUDE) $(LIBDIR) $< -o $@ $(LIBS)

clean:

