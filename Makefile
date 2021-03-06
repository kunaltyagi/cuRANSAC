GCC  = gcc-4.9
GCPP = g++-4.9

CXXFLAGS = -g -pg
CUDA_LIBS = -lcuda -lcudart -lcudadevrt
OCV_COMMAND = `pkg-config opencv --cflags --libs`
ifeq (${HOME}, /home/kunal)
    ocv3 = ${HOME}/local/opencv3
    OPENCV_LIBS = ${OCV_COMMAND} | sed 's/ /\n/g' | grep ^\-l | xargs
    OPENCV_INC = -I${ocv3}/include
    LIBRARY_PATH = ${ocv3}lib
    OPENCV_LINK = -L${LIBRARY_PATH}
    LD_LIBRARY_PATH = ${LIBRARY_PATH}
    OPENCV_LIB_INC = ${OPENCV_INC} ${OPENCV_LINK} ${OPENCV_LIBS}
else
    OPENCV_LIB_INC = `pkg-config opencv --cflags --libs`
endif
INCLUDE_LOCATION = -I./ -I/usr/local/cuda/include
LINK_LOCATION    = -L/usr/local/cuda/lib -L/usr/local/cuda/lib64

all: cir_d

cir_d: ransac_cuda.o main.cpp circle_detect.o  ransac_link.o setup.sh
	$(GCPP) ${CXXFLAGS} -c main.cpp ${INCLUDE_LOCATION}
	$(GCPP) ${CXXFLAGS} -o  cir_d main.o ransac_link.o ransac_cuda.o circle_detect.o ${INCLUDE_LOCATION} ${LINK_LOCATION} -lcuda -lcudart ${OPENCV_LIB_INC}

circle_detect.o : circle_detect.cpp
	$(GCPP) ${CXXFLAGS} -c circle_detect.cpp ${INCLUDE_LOCATION}

ransac_link.o : ransac_cuda.o
	nvcc -arch=sm_50 -dlink -o ransac_link.o ransac_cuda.o ${INCLUDE_LOCATION} ${CUDA_LIBS}

ransac_cuda.o : ransac_cuda.cu
	nvcc -arch=sm_50 -rdc=true -c ransac_cuda.cu ${INCLUDE_LOCATION}

.phony: clean

clean :
	rm *.o
	rm cir_d
