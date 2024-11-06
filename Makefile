ROCM_PATH ?= /opt/rocm-6.2.1
GPU_ARCH ?= gfx90a
CC = ${ROCM_PATH}/bin/hipcc # define the C/C++ compiler to use

HDF5_PATH=${HDF5_ROOT}

SRC_DIR=./src
INCLUDES = -I./src -I${HDF5_PATH}/include -I${ROCM_PATH}/include/hipfft
LIBS = -L${HDF5_PATH}/lib -lhdf5 -L${ROCM_PATH}/lib -lhipfft

CFLAGS = -O3 -std=c++11 --offload-arch=${GPU_ARCH}
CFLAGS += -munsafe-fp-atomics
CFLAGS += ${INCLUDES}

SRCS = $(shell echo $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:.c=.o)

TARGET = particle_mesh_${GPU_ARCH}

.PHONY: clean
    
all:    $(TARGET)
	@echo  Successfully compiled ${TARGET}.

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LFLAGS) $(OBJS) -o $(TARGET) $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


	
clean:
	$(RM) *.o ${TARGET}