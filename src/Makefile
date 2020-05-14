INC="inc"
NVCCFLAGS=-I$(INC)
NVCC=nvcc

all: MatMul Trans VecAdd

MatMul: Matrix_mul.cu
	$(NVCC) $(NVCCFLAGS) Matrix_mul.cu -o MatMul $(LFLAGS)

Trans:	Transpose.cu
	$(NVCC)	$(NVCCFLAGS) Transpose.cu -o Trans $(LFLAGS)

VecAdd:	Vector_add.cu
	$(NVCC)	$(NVCCFLAGS) Vector_add.cu -o VecAdd $(LFLAGS)

clean:
	rm MatMul Trans VecAdd
