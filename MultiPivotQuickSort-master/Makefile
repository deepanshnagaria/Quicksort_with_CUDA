LIBS=--relocatable-device-code=true
ARCH=-arch=sm_20
OPTIONS=-O2

hello: main.o multiquicksortgpu.o
	nvcc $(OPTIONS) $(ARCH) $(LIBS) main.o multiquicksortgpu.o -o hello

main.o: main.cpp
	nvcc $(OPTIONS) $(ARCH) $(LIBS) -c main.cpp -o main.o
	
multiquicksortgpu.o: multiquicksortgpu.cu
	nvcc $(OPTIONS) $(ARCH) $(LIBS) -c multiquicksortgpu.cu -o multiquicksortgpu.o
		
clean:
	rm -rf *.o hello.*
