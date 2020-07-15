#include <iostream>
#include "multiquicksortgpu.h"
#include <vector>
#include <float.h>

using namespace std;

int main(int argc, char** argv){

    if(argc != 4){
        std::cout<<"Wrong number of arguments!\n";
        std::cout<<"Correct launch: hello.exe <initial_file> <output_file> <gpu_id>\n";
        std::cout<<"Sample launch: hello.exe input.csv output.csv 0\n";
        return 1;
    }

    std::string inputFile = std::string(argv[1]);
    std::string outputFile = std::string(argv[2]);
    int device = atoi(argv[3]);

    cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop, device) == cudaSuccess){
        if(prop.major == 1){
            std::cout<<"Device with id "<<device<<" cannot be used. The code requires CC at least 2.0, this GPU has "<<prop.major<<"."<<prop.minor<<"\n";
            return 2;
        }
        if(prop.major == 9999){
            std::cout<<"The computer is not equipped with graphics cards\n";
            return 3;
        }
    }else{
        std::cout<<"Device with id "<<device<<" is not located in this computer\n";
        return 4;
    }

    int numberOfPoints = 0;

    std::ifstream myfile;
    myfile.open(inputFile.c_str());
    if (myfile.is_open()){
    }else{
        std::cout<<"Error while opening the data file\n";
        return 5;
    }
    std::string line;

    std::getline(myfile, line);
    numberOfPoints = atoi(line.c_str());
    myfile.close();

    std::cout<<"Numbers to be sorted: "<<numberOfPoints<<"\n";

    MultiQuickSortGPU mqs;
    mqs.setDataFile(inputFile);
    mqs.setResultsFile(outputFile);
    bool error = mqs.initialize(numberOfPoints, device);
    if(error){
        std::cout<<"Initialization failed\n";
        return 6;
    }
    std::cout<<"I start the analysis\n";
    mqs.sort();
    std::cout<<"Saves the solution to the output file\n";
    mqs.saveResultToResultFile();
    error = mqs.deinitialize();
    if(error){
        std::cout<<"Deinitiation failed\n";
        return 7;
    }

    cout<<"End\n";
    return 0;
}
