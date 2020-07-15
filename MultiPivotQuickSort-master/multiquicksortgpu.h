#ifndef MULTIQUICKSORTGPU_H
#define MULTIQUICKSORTGPU_H

#include <iostream>
#include "cucall.h"
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <map>

class MultiQuickSortGPU{
    public:
        MultiQuickSortGPU();
        ~MultiQuickSortGPU();

        void setDataFile(std::string nameOfFile);
        void setResultsFile(std::string nameOfFile);
        bool initialize(int numberOfEntities, int device);
        bool deinitialize();
        bool saveResultToResultFile();
        bool sort();

    private:
        bool loadData();

        int pivotsNumber;
        int numberOfDivisionBlocks;

        //Path do pliku z danymi
        std::string inputFile;

        //Path do pliku z danymi wynikowymi
        std::string outputFile;

        //Liczba punktow do sortowania
        int numberOfEntities;

        //Urzadzenie do obliczania
        int device;

        //Liczby znajdujace sie na CPU
        float* dataTable_host;

        //Liczby znajdujace sie na GPU
        float* dataTable_device;

        //Stream, na ktorym wszystkie obliczenia sa uruchamiane
        cudaStream_t executionStreams;

        //Event startowy
        cudaEvent_t startEvents;

        //Event koncowy
        cudaEvent_t stopEvents;

        //Elementy pomocne przy sortowaniu
		int* nodeParent;
        int* numberOfElemsInNode;
        int** childrens_HostForDevice;
		int** childrens;
		int** numberOfElems_HostForDevice;
        int** numberOfElems;
		int** biasOfElems_HostForDevice;
        int** biasOfElems;

        float* elems_host;
        float* elems_device1;
        float* elems_device2;

        int* parent_host;
        int* parent_device1;
        int* parent_device2;
        int* parent_device3;

        int* treeNodeSizeDevice;
        int* thereWasDividing;
};

#endif // MULTIQUICKSORTGPU_H
