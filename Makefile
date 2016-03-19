GCC=gcc-4.9
GCPP=g++-4.9

all: cir_d

cir_d: ransac_cuda.o main.cpp circle_detect.o  ransac_link.o
	$(GCPP) -g -pg -c main.cpp -I./ -I/usr/local/cuda-7.5/targets/x86_64-linux/include
	$(GCPP) -g -pg -o  cir_d main.o ransac_link.o ransac_cuda.o circle_detect.o  -I./ -I/usr/local/cuda-7.5/targets/x86_64-linux/include -L/usr/local/cuda-7.5/targets/x86_64-linux/lib  -L/usr/local/cuda/lib64 -lcuda -lcudart -lcudadevrt `pkg-config opencv --cflags --libs`

circle_detect.o : circle_detect.cpp
	$(GCPP) -pg -g -I./ -I/usr/local/cuda-7.5/targets/x86_64-linux/include -c circle_detect.cpp 

ransac_link.o : ransac_cuda.o
	nvcc -arch=sm_20 -dlink -o ransac_link.o ransac_cuda.o -lcudadevrt -lcudart -I./ -I/usr/local/cuda-7.5/targets/x86_64-linux/include

ransac_cuda.o : ransac_cuda.cu
	nvcc -arch=sm_20 -rdc=true -c ransac_cuda.cu   -I./ -I/usr/local/cuda-7.5/targets/x86_64-linux/include

clean : 
	rm *.o
	echo  "-lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_highgui  -lopencv_objdetect -lopencv_imgcodecs "
