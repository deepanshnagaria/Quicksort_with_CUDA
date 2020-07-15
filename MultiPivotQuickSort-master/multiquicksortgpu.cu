#include "multiquicksortgpu.h"
#include <cmath>
#include <float.h>
#include <set>
#include <algorithm>

#define EMPTY_DIRECTION -1
#define END_DIRECTION -2
#define SIDE_LEFT 0
#define SIDE_RIGHT 1
#define SIDE_END 2

MultiQuickSortGPU::MultiQuickSortGPU(){
    pivotsNumber = 15;
    numberOfDivisionBlocks = pivotsNumber + 1;
}

MultiQuickSortGPU::~MultiQuickSortGPU(){

}

void MultiQuickSortGPU::setDataFile(std::string nameOfFile){
    this->inputFile = nameOfFile;
}

void MultiQuickSortGPU::setResultsFile(std::string nameOfFile){
    this->outputFile = nameOfFile;
}

bool MultiQuickSortGPU::initialize(int numberOfEntities, int device){
    this->numberOfEntities = numberOfEntities;
    this->device = device;

    bool error = false;

    error |= cuCall(cudaSetDevice(device));
    error |= cuCall(cudaDeviceReset());

    error |= cuCall(cudaStreamCreate(&executionStreams));
    error |= cuCall(cudaEventCreate(&startEvents));
    error |= cuCall(cudaEventCreate(&stopEvents));

    error |= cuCall(cudaHostAlloc((void**)&dataTable_host, numberOfEntities*sizeof(float), cudaHostAllocPortable));
    error |= cuCall(cudaMalloc((void**)&dataTable_device, numberOfEntities*sizeof(float)));

    error |= loadData();

    error |= cuCall(cudaMemcpyAsync((void*)dataTable_device, (void*)dataTable_host, numberOfEntities*sizeof(float), cudaMemcpyHostToDevice, executionStreams));

    error |= cuCall(cudaMalloc((void**)&nodeParent, numberOfDivisionBlocks*numberOfEntities*sizeof(int)));
    error |= cuCall(cudaMalloc((void**)&numberOfElemsInNode, numberOfDivisionBlocks*numberOfEntities*sizeof(int)));

    error |= cuCall(cudaHostAlloc((void**)&childrens_HostForDevice, numberOfDivisionBlocks*sizeof(int*), cudaHostAllocPortable));
    error |= cuCall(cudaMalloc((void**)&childrens, numberOfDivisionBlocks*sizeof(int*)));
    for(int i=0 ; i<numberOfDivisionBlocks ; i++){
        error |= cuCall(cudaMalloc((void**)&childrens_HostForDevice[i], numberOfDivisionBlocks*numberOfEntities*sizeof(int)));
    }
    error |= cuCall(cudaMemcpyAsync((void*)childrens, (void*)childrens_HostForDevice, numberOfDivisionBlocks*sizeof(int*), cudaMemcpyHostToDevice, executionStreams));

    error |= cuCall(cudaHostAlloc((void**)&numberOfElems_HostForDevice, numberOfDivisionBlocks*sizeof(int*), cudaHostAllocPortable));
    error |= cuCall(cudaMalloc((void**)&numberOfElems, numberOfDivisionBlocks*sizeof(int*)));
    for(int i=0 ; i<numberOfDivisionBlocks ; i++){
        error |= cuCall(cudaMalloc((void**)&numberOfElems_HostForDevice[i], numberOfDivisionBlocks*numberOfEntities*sizeof(int)));
    }
    error |= cuCall(cudaMemcpyAsync((void*)numberOfElems, (void*)numberOfElems_HostForDevice, numberOfDivisionBlocks*sizeof(int*), cudaMemcpyHostToDevice, executionStreams));

    error |= cuCall(cudaHostAlloc((void**)&biasOfElems_HostForDevice, numberOfDivisionBlocks*sizeof(int*), cudaHostAllocPortable));
    error |= cuCall(cudaMalloc((void**)&biasOfElems, numberOfDivisionBlocks*sizeof(int*)));
    for(int i=0 ; i<numberOfDivisionBlocks ; i++){
        error |= cuCall(cudaMalloc((void**)&biasOfElems_HostForDevice[i], numberOfDivisionBlocks*numberOfEntities*sizeof(int)));
    }
    error |= cuCall(cudaMemcpyAsync((void*)biasOfElems, (void*)biasOfElems_HostForDevice, numberOfDivisionBlocks*sizeof(int*), cudaMemcpyHostToDevice, executionStreams));

    error |= cuCall(cudaHostAlloc((void**)&elems_host, numberOfEntities*sizeof(float), cudaMemcpyHostToDevice));
    error |= cuCall(cudaMalloc((void**)&elems_device1, numberOfEntities*sizeof(float)));
    error |= cuCall(cudaMalloc((void**)&elems_device2, numberOfEntities*sizeof(float)));

    error |= cuCall(cudaHostAlloc((void**)&parent_host, numberOfEntities*sizeof(int), cudaMemcpyHostToDevice));
    error |= cuCall(cudaMalloc((void**)&parent_device1, numberOfEntities*sizeof(int)));
    error |= cuCall(cudaMalloc((void**)&parent_device2, numberOfEntities*sizeof(int)));
    error |= cuCall(cudaMalloc((void**)&parent_device3, numberOfEntities*sizeof(int)));

    error |= cuCall(cudaMalloc((void**)&treeNodeSizeDevice, sizeof(int)));
    error |= cuCall(cudaMalloc((void**)&thereWasDividing, sizeof(int)));

    error |= cuCall(cudaStreamSynchronize(executionStreams));

    return error;
}

bool MultiQuickSortGPU::deinitialize(){
    bool error = false;

    error |= cuCall(cudaSetDevice(device));
    error |= cuCall(cudaDeviceSynchronize());

    error |= cuCall(cudaStreamDestroy(executionStreams));
    error |= cuCall(cudaEventDestroy(startEvents));
    error |= cuCall(cudaEventDestroy(stopEvents));

    error |= cuCall(cudaFreeHost((void*)dataTable_host));
    error |= cuCall(cudaFree((void*)dataTable_device));

    error |= cuCall(cudaFree((void*)nodeParent));
    error |= cuCall(cudaFree((void*)numberOfElemsInNode));

    for(int i=0 ; i<numberOfDivisionBlocks ; i++){
        error |= cuCall(cudaFree((void*)childrens_HostForDevice[i]));
    }
    error |= cuCall(cudaFree((void*)childrens));
    error |= cuCall(cudaFreeHost((void*)childrens_HostForDevice));

    for(int i=0 ; i<numberOfDivisionBlocks ; i++){
        error |= cuCall(cudaFree((void*)numberOfElems_HostForDevice[i]));
    }
    error |= cuCall(cudaFree((void*)numberOfElems));
    error |= cuCall(cudaFreeHost((void*)numberOfElems_HostForDevice));

    for(int i=0 ; i<numberOfDivisionBlocks ; i++){
        error |= cuCall(cudaFree((void*)biasOfElems_HostForDevice[i]));
    }
    error |= cuCall(cudaFree((void*)biasOfElems));
    error |= cuCall(cudaFreeHost((void*)biasOfElems_HostForDevice));

    error |= cuCall(cudaFreeHost((void*)elems_host));
    error |= cuCall(cudaFree((void*)elems_device1));
    error |= cuCall(cudaFree((void*)elems_device2));

    error |= cuCall(cudaFreeHost((void*)parent_host));
    error |= cuCall(cudaFree((void*)parent_device1));
    error |= cuCall(cudaFree((void*)parent_device2));
    error |= cuCall(cudaFree((void*)parent_device3));

    error |= cuCall(cudaFree((void*)treeNodeSizeDevice));
    error |= cuCall(cudaFree((void*)thereWasDividing));

    error |= cuCall(cudaDeviceReset());

    return error;
}

bool MultiQuickSortGPU::saveResultToResultFile(){
    bool error = false;
    error |= cuCall(cudaSetDevice(device));
    error |= cuCall(cudaMemcpyAsync((void*)dataTable_host, (void*)dataTable_device, numberOfEntities*sizeof(float), cudaMemcpyDeviceToHost, executionStreams));
    error |= cuCall(cudaStreamSynchronize(executionStreams));

    for(int i=0 ; i<numberOfEntities-1 ; ++i){
        if(dataTable_host[i+1] < dataTable_host[i]){
            std::cout<<"Cos poszlo nie tak\n";
        }
    }

    //Zapisanie rezultatu do pliku
    std::ofstream ofs;
    ofs.open(outputFile.c_str(), std::ofstream::trunc);
    if(ofs.is_open()){
        ofs<<numberOfEntities<<"\n";
        //zapisywanie punktow
        for(int lp=0 ; lp<numberOfEntities ; ++lp){
            ofs<<dataTable_host[lp];
            if(lp < numberOfEntities-1){
                ofs<<",";
            }
        }
        ofs.close();
    }else{
        std::cout <<"Blad otwarcia pliku z rezultatem\n";
        error |= true;
    }
    return error;
}

std::string trim(std::string const& str){
    if(str.empty())
        return str;

    std::size_t firstScan = str.find_first_not_of(' ');
    std::size_t first     = firstScan == std::string::npos ? str.length() : firstScan;
    std::size_t last      = str.find_last_not_of(' ');
    return str.substr(first, last-first+1);
}

bool MultiQuickSortGPU::loadData(){
    std::ifstream myfile;
    myfile.open(this->inputFile.c_str());
    if (myfile.is_open()){
        std::cout<<"Plik z danymi otwarto\n";
    }else{
        std::cout<<"Blad otwarcia pliku\n";
        return true;
    }
    std::string line;

    std::getline(myfile, line);
    std::getline(myfile, line);

    std::vector<std::string> cuttedString;
    char* lineChar = new char[line.length() + 1];
    std::strcpy(lineChar, line.c_str());
    std::string str;
    char* pch = strtok(lineChar,",");
    while (pch != NULL){
        str = std::string(pch);
        str = trim(str);
        cuttedString.push_back(str);
        pch = strtok (NULL, ",");
    }
    delete [] lineChar;
    for(int i=0 ; i<cuttedString.size() ; ++i){
        this->dataTable_host[i] = atof(cuttedString[i].c_str());
    }
    return false;
}

__global__
void makePartitionOfLeaf0(int* nodeParent, int* numberOfElemsInNode, int** childrens, int** numberOfElems, int** biasOfElems,
                          int* treeNodeSizeDevice, int numberOfDivisionBlocks, int numberOfEntities){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid == 0){
        nodeParent[0] = EMPTY_DIRECTION;
        numberOfElemsInNode[0] = numberOfEntities;
        for(int i=0 ; i<numberOfDivisionBlocks ; ++i){
            childrens[i][0] = EMPTY_DIRECTION;
            numberOfElems[i][0] = 0;
            biasOfElems[i][0] = 0;
        }
        *treeNodeSizeDevice = 1;
    }
}

__global__
void makePartitionOfLeaf1(int* parent_device1, int numberOfEntities){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        parent_device1[tid] = 0;
    }
}

__global__
void makePartitionOfLeafSort(float* elems_device1, int* parent_device1, float* elems_device2, int* parent_device2,
                             int* nodeParent, int* numberOfElemsInNode, int** childrens, int** numberOfElems, int** biasOfElems, int numberOfEntities,
                             int pivotsNumber, int numberOfDivisionBlocks){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        int parent = parent_device1[tid];
        if(parent == EMPTY_DIRECTION){
            return;
        }
        int bias = biasOfElems[0][parent];
        if(tid != bias){
            return;
        }
        int numberToSort = min(numberOfElemsInNode[parent], pivotsNumber);
        //Sortujemy pierwsze pivotsNumber pivotow
        do{
            for(int i=0 ; i<numberToSort-1 ; ++i){
                if(elems_device1[bias+i] > elems_device1[bias+i+1]){
                    float tmp = elems_device1[bias+i];
                    elems_device1[bias+i] = elems_device1[bias+i+1];
                    elems_device1[bias+i+1] = tmp;
                }
            }
            numberToSort -= 1;
        }while(numberToSort > 1);
    }
}

__global__
void makePartitionOfLeaf2(float* elems_device1, int* parent_device1, float* elems_device2, int* parent_device2,
                          int* nodeParent, int** childrens, int** numberOfElems, int** biasOfElems, int numberOfEntities, int pivotsNumber, int numberOfDivisionBlocks){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        //Jezeli mamy pojedynczy element w tablicy to juz nie sortujemy
        int parent = parent_device1[tid];
        if(parent == EMPTY_DIRECTION){
            return;
        }
        int bias = biasOfElems[0][parent];
        //Jezeli watek chce sortowac pivota to nie powinien sortowac.
        //W nastepnym etapie trzeba bedzie pivota przepisac juz we wlasciwe miejsce w posortowanej tablicy
        if((tid - bias) < pivotsNumber){
            parent_device2[tid] = EMPTY_DIRECTION;
            return;
        }
        //Wpisujemy jak podziela sie liczby
        float value = elems_device1[tid];

        int logPos = 0;
        for(int i=0 ; i<pivotsNumber ; ++i){
            if(value > elems_device1[bias+i]){
                logPos += 1;
            }
        }
        int newPos = atomicAdd(&numberOfElems[logPos][parent], 1);
        parent_device2[tid] = newPos;
    }
}

__global__
void makePartitionOfLeaf3(float* elems_device1, int* parent_device1, float* elems_device2, int* parent_device2,
                          int* nodeParent, int** childrens, int** numberOfElems, int** biasOfElems, int numberOfEntities,
                          int pivotsNumber, int numberOfDivisionBlocks){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        int parent = parent_device1[tid];
        if(parent == EMPTY_DIRECTION){
            return;
        }
        int bias = biasOfElems[0][parent];
        if(tid != bias){
            return;
        }
        for(int i=0 ; i<numberOfDivisionBlocks ; ++i){
            if(childrens[i][parent] != EMPTY_DIRECTION){
                return;
            }
        }
        for(int i=1 ; i<numberOfDivisionBlocks ; ++i){
            biasOfElems[i][parent] = biasOfElems[i-1][parent];
            biasOfElems[i][parent] += numberOfElems[i-1][parent]+1;
        }
    }
}

__global__
void makePartitionOfLeaf4(float* elems_device1, int* parent_device1, float* elems_device2, int* parent_device2,
                          int* nodeParent, int** childrens, int** numberOfElems, int** biasOfElems, int numberOfEntities, float* dataTable_device,
                          int pivotsNumber, int numberOfDivisionBlocks){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        //Jezeli mamy pojedynczy element w tablicy to juz nie sortujemy
        int parent = parent_device1[tid];
        if(parent == EMPTY_DIRECTION){
            return;
        }
        int bias = biasOfElems[0][parent];
        //Jezeli watek chce sortowac pivota to wyznaczamy gdzie on trafi do posortowanej tablicy.
        //Dodatkowo zabazpieczamy, gdy bedziemy brac skrajne skrajne liczby
        if((tid - bias) < pivotsNumber){
            float valueAndPivot = elems_device1[tid];
            int biasMain = biasOfElems[tid-bias][parent];
            int elems = numberOfElems[tid-bias][parent];
            dataTable_device[biasMain+elems] = valueAndPivot;
        }
    }
}

__global__
void makePartitionOfLeaf5(float* elems_device1, int* parent_device1, float* elems_device2, int* parent_device2,
                          int* nodeParent, int** childrens, int** numberOfElems, int** biasOfElems, int numberOfEntities,
                          int pivotsNumber, int numberOfDivisionBlocks){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        int parent = parent_device1[tid];
        if(parent == EMPTY_DIRECTION){
            return;
        }
        int bias = biasOfElems[0][parent];
        if((tid - bias) < pivotsNumber){
            return;
        }
        float value = elems_device1[tid];
        int newPos = parent_device2[tid];

        int logPos = 0;
        for(int i=0 ; i<pivotsNumber ; ++i){
            if(value > elems_device1[bias+i]){
                logPos += 1;
            }
        }
        int mainBias = biasOfElems[logPos][parent];

        elems_device2[mainBias+newPos] = value;
    }
}

__global__
void makePartitionOfLeaf6(float* elems_device1, int* parent_device1, float* elems_device2, int* parent_device2,
                          int* nodeParent, int* numberOfElemsInNode, int** childrens, int** numberOfElems, int** biasOfElems,
                          int numberOfEntities, int* thereWasDividing, int* treeNodeSizeDevice,
                          int pivotsNumber, int numberOfDivisionBlocks){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        int parent = parent_device1[tid];
        if(parent == EMPTY_DIRECTION){
            return;
        }
        if(tid != biasOfElems[0][parent]){
            return;
        }
        if(numberOfElemsInNode[parent] <= pivotsNumber){
            return;
        }
        if(*thereWasDividing == 0){
            atomicCAS(thereWasDividing, 0, 1);
        }
        for(int i=0 ; i<numberOfDivisionBlocks ; ++i){
            int idOfChildren = (int)atomicInc((unsigned int*)treeNodeSizeDevice, INT_MAX);
            childrens[i][parent] = idOfChildren;
            nodeParent[idOfChildren] = parent;
            int bb = biasOfElems[i][parent];

            int newDir = END_DIRECTION;
            if(numberOfElems[i][parent] > 0){
                newDir = EMPTY_DIRECTION;
            }

            numberOfElemsInNode[idOfChildren] = numberOfElems[i][parent];

            for(int k=0 ; k<numberOfDivisionBlocks ; ++k){
                childrens[k][idOfChildren] = newDir;
                numberOfElems[k][idOfChildren] = 0;
                biasOfElems[k][idOfChildren] = bb;
            }
        }
    }
}

__global__
void makePartitionOfLeaf7(float* elems_device1, int* parent_device1, float* elems_device2, int* parent_device2, int* parent_device3,
                          int* nodeParent, int** childrens, int** numberOfElems, int** biasOfElems, int numberOfEntities,
                          int pivotsNumber, int numberOfDivisionBlocks){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        int parent = parent_device1[tid];
        if(parent == EMPTY_DIRECTION){
            parent_device3[tid] = EMPTY_DIRECTION;
            return;
        }
        int bias = biasOfElems[0][parent];
        //Zajmujemy sie elementami nie bedacymi pivotami
        float value = elems_device1[tid];
        if((tid - bias) >= pivotsNumber){
            int newPos = parent_device2[tid];

            int logPos = 0;
            for(int i=0 ; i<pivotsNumber ; ++i){
                if(value > elems_device1[bias+i]){
                    logPos += 1;
                }
            }
            int mainBias = biasOfElems[logPos][parent];
            parent_device3[mainBias+newPos] = childrens[logPos][parent];
        }else{
            int mainBias = biasOfElems[tid-bias][parent];
            int elems = numberOfElems[tid-bias][parent];
            parent_device3[mainBias+elems] = EMPTY_DIRECTION;
        }
    }
}

bool MultiQuickSortGPU::sort(){
    bool error = false;
    error |= cuCall(cudaSetDevice(device));

    error |= cuCall(cudaMemcpyAsync((void*)elems_device1, (void*)dataTable_device, numberOfEntities*sizeof(float), cudaMemcpyDeviceToDevice, executionStreams));
    error |= cuCall(cudaMemsetAsync((void*)dataTable_device, 0, numberOfEntities*sizeof(float), executionStreams));

    error |= cuCall(cudaEventRecord(startEvents, executionStreams));

    makePartitionOfLeaf0<<<1, 1, 0, executionStreams>>>(nodeParent, numberOfElemsInNode, childrens, numberOfElems, biasOfElems,
                                                        treeNodeSizeDevice, numberOfDivisionBlocks, numberOfEntities);

    dim3 grid1(ceil(float(numberOfEntities)/256.0), 1);
    dim3 block1(256, 1);
    makePartitionOfLeaf1<<<grid1, block1, 0, executionStreams>>>(parent_device1, numberOfEntities);

    bool treeeIsGoingToBeEdit = true;
    while(treeeIsGoingToBeEdit == true){
        treeeIsGoingToBeEdit = false;
        cuCall(cudaMemsetAsync((void*)thereWasDividing, 0, sizeof(int), executionStreams));

        dim3 gridSort(ceil(float(numberOfEntities)/256.0), 1);
        dim3 blockSort(256, 1);
        makePartitionOfLeafSort<<<gridSort, blockSort, 0, executionStreams>>>(elems_device1, parent_device1, elems_device2, parent_device2,
                                                                              nodeParent, numberOfElemsInNode, childrens, numberOfElems, biasOfElems,
                                                                              numberOfEntities, pivotsNumber, numberOfDivisionBlocks);

        dim3 grid2(ceil(float(numberOfEntities)/256.0), 1);
        dim3 block2(256, 1);
        makePartitionOfLeaf2<<<grid2, block2, 0, executionStreams>>>(elems_device1, parent_device1, elems_device2, parent_device2,
                                                                     nodeParent, childrens, numberOfElems, biasOfElems,
                                                                     numberOfEntities, pivotsNumber, numberOfDivisionBlocks);

        dim3 grid3(ceil(float(numberOfEntities)/256.0), 1);
        dim3 block3(256, 1);
        makePartitionOfLeaf3<<<grid3, block3, 0, executionStreams>>>(elems_device1, parent_device1, elems_device2, parent_device2,
                                                                     nodeParent, childrens, numberOfElems, biasOfElems,
                                                                     numberOfEntities, pivotsNumber, numberOfDivisionBlocks);

        dim3 grid4(ceil(float(numberOfEntities)/256.0), 1);
        dim3 block4(256, 1);
        makePartitionOfLeaf4<<<grid4, block4, 0, executionStreams>>>(elems_device1, parent_device1, elems_device2, parent_device2,
                                                                     nodeParent, childrens, numberOfElems, biasOfElems,
                                                                     numberOfEntities, dataTable_device, pivotsNumber, numberOfDivisionBlocks);

        dim3 grid5(ceil(float(numberOfEntities)/256.0), 1);
        dim3 block5(256, 1);
        makePartitionOfLeaf5<<<grid5, block5, 0, executionStreams>>>(elems_device1, parent_device1, elems_device2, parent_device2,
                                                                     nodeParent, childrens, numberOfElems, biasOfElems,
                                                                     numberOfEntities, pivotsNumber, numberOfDivisionBlocks);

        dim3 grid6(ceil(float(numberOfEntities)/256.0), 1);
        dim3 block6(256, 1);
        makePartitionOfLeaf6<<<grid6, block6, 0, executionStreams>>>(elems_device1, parent_device1, elems_device2, parent_device2,
                                                                     nodeParent, numberOfElemsInNode, childrens, numberOfElems, biasOfElems,
                                                                     numberOfEntities, thereWasDividing, treeNodeSizeDevice,
                                                                     pivotsNumber, numberOfDivisionBlocks);

        dim3 grid7(ceil(float(numberOfEntities)/256.0), 1);
        dim3 block7(256, 1);
        makePartitionOfLeaf7<<<grid7, block7, 0, executionStreams>>>(elems_device1, parent_device1, elems_device2, parent_device2, parent_device3,
                                                                     nodeParent, childrens, numberOfElems, biasOfElems,
                                                                     numberOfEntities, pivotsNumber, numberOfDivisionBlocks);

        int thereWasDividingHost;
        cuCall(cudaMemcpyAsync((void*)&thereWasDividingHost, (void*)thereWasDividing, sizeof(int), cudaMemcpyDeviceToHost, executionStreams));
        cuCall(cudaStreamSynchronize(executionStreams));

        if(thereWasDividingHost != 0){
            treeeIsGoingToBeEdit = true;
        }

        float* tmp1 = elems_device1;
        elems_device1 = elems_device2;
        elems_device2 = tmp1;

        int* tmp2 = parent_device1;
        parent_device1 = parent_device3;
        parent_device3 = tmp2;
    }

    error |= cuCall(cudaEventRecord(stopEvents, executionStreams));
    error |= cuCall(cudaEventSynchronize(stopEvents));
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvents, stopEvents);
    std::cout<<"Urzadzenie "<<device<<": zakonczylo obliczenia w czasie: "<<milliseconds<<" ms\n";

    return error;
}
