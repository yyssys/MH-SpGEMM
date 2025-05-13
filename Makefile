CXX = g++
NVCC = nvcc

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH = /usr/local/cuda-11.8

#includes
INCLUDES = -I$(CUDA_INSTALL_PATH)/include
INCLUDES += -I./inc

SRC = ./src
OBJ = ./obj
INC = ./inc

NVCCFLAGS = -arch=compute_89 -code=sm_89 -O3 -w -Xcompiler -fopenmp -lcusparse

CPPFLAGS = -O3 -std=c++14 -w

COMMON_DEP = $(INC)/common.h $(INC)/*.cuh

OBJ_LIB = $(OBJ)/main.o $(OBJ)/CSR.o $(OBJ)/utils.o $(OBJ)/Timing.o $(OBJ)/Tool.o

spgemm: $(OBJ_LIB)
	$(NVCC) $(NVCCFLAGS) $^ -o $@


$(OBJ)/%.o: $(SRC)/%.cu $(COMMON_DEP)
	$(NVCC) -c $(NVCCFLAGS) $< -o $@ $(INCLUDES)

$(OBJ)/%.o: $(SRC)/%.cu $(INC)/%.h $(COMMON_DEP)
	$(NVCC) -c $(NVCCFLAGS) $< -o $@ $(INCLUDES)

$(OBJ)/%.o: $(SRC)/%.cpp $(INC)/%.h $(COMMON_DEP)
	$(CXX) -c $(CPPFLAGS) $< -o $@ $(INCLUDES)

clean:
	rm -rf $(OBJ)/*