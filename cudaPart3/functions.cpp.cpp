#include "header.h"

unsigned char* readImage(int rows,int columns){
    unsigned char *image;
    FILE* imagePointer;
    int elementCount;

    image = malloc(rows*columns*sizeof(unsigned char)); //create image 1-D array
    if(image == NULL){
        printf("Error:Not available memory\n");
        MPI_Finalize();
        exit(1);
    }
    imagePointer = fopen("waterfall_grey_1920_2520.raw","rb"); //open imageFile
    if(imagePointer == NULL){
        printf("Error:Image can not be loaded\n");
        MPI_Finalize();
        exit(1);
    }
    elementCount = fread(image,sizeof(unsigned char),rows*columns,imagePointer); //read and store image's elements
    if(elementCount != columns*rows){
        printf("Error:Image can not be read\n");
        MPI_Finalize();
        exit(1);
    }
    fclose(imagePointer);
    return image;
}

__device__ void findNeighbors(int row,int column,int columns,int rows,unsigned char* dImage,unsigned char* me,unsigned char* upper,unsigned char* lower,
                              unsigned char* left,unsigned char* right,unsigned char* upperLeft,unsigned char* upperRight,
                              unsigned char* lowerLeft,unsigned char* lowerRight)
{
    if( (row-1) == -1 )
        *upper = dImage[row*columns + column];
    else
        *upper = dImage[(row-1)*columns + column];
    if( (row+1) == rows )
        *lower = dImage[row*columns + column];
    else
        *lower = dImage[(row+1)*columns + column];
    if( (column-1) == -1 )
        *left = dImage[row*columns + column];
    else
        *left = dImage[row*columns + (column-1)];
    if( (column+1) == columns )
        *right = dImage[row*columns + column];
    else
        *right = dImage[row*columns + (column+1)];
    //---------------------------------------------
    if( (row-1 == -1)  || (column-1 == -1) )
        *upperLeft = dImage[row*columns + column];
    else
        *upperLeft = dImage[(row-1)*columns + (column-1)];
    if( (row-1 == -1) || (column+1) == columns )
        *upperRight = dImage[row*columns + column];
    else
        *upperRight = dImage[(row-1)*columns + (column+1)];
    if ( (row+1 == rows) || (column-1 == -1) )
        *lowerLeft = dImage[row*columns + column];
    else
        *lowerLeft = dImage[(row+1)*columns + (column-1)];
    if ( (row+1 == rows) || (column+1 == columns) )
        *lowerRight = dImage[row*columns + column];
    else
        *lowerRight = dImage[(row+1)*columns + (column+1)];
    *me = dImage[row*columns + column];
    return;
}

__global__ void filter(unsigned char* dImageA,unsigned char* dImageB,int rows,int columns,double* dFilterArray,int loops){

    unsigned char me,upper,lower,left,right,upperLeft,upperRight,lowerLeft,lowerRight;

    const unsigned int row = blockIdx.x * blockDim.y + threadIdx.x;
    const unsigned int column = blockIdx.y * blockDim.y + threadIdx.y;

    if(row<rows && column<columns){
        if(loops%2 != 0){
            findNeighbors(row,column,columns,rows,dImageA,&me,&upper,&lower,&left,&right,&upperLeft,&upperRight,&lowerLeft,&lowerRight);
            dImageB[row*columns + column] = (unsigned char)(round(upperLeft*dFilterArray[0] + upper*dFilterArray[1] + upperRight*dFilterArray[2] +
                                        left*dFilterArray[3]+ me*dFilterArray[4] + right*dFilterArray[5] + lowerLeft*dFilterArray[6] +
                                        lower*dFilterArray[7] + lowerRight*dFilterArray[8]));
        }
        else{
            findNeighbors(row,column,columns,rows,dImageB,&me,&upper,&lower,&left,&right,&upperLeft,&upperRight,&lowerLeft,&lowerRight);
            dImageA[row*columns + column] = (unsigned char)(round(upperLeft*dFilterArray[0] + upper*dFilterArray[1] + upperRight*dFilterArray[2] +
                                        left*dFilterArray[3]+ me*dFilterArray[4] + right*dFilterArray[5] + lowerLeft*dFilterArray[6] +
                                        lower*dFilterArray[7] + lowerRight*dFilterArray[8]));
        }
    }
    return;
}

extern "C" float filterImage(int rows,int columns,int size,unsigned char* imageA,unsigned char* imageB,double* filterArray){
    int loops = 0,repeatFilter = 1;
    //Allocate vector in device memory
    unsigned char* dImageA;
    CUDA_SAFE_CALL( cudaMalloc((void**)&dImageA,size) );
    unsigned char* dImageB;
    CUDA_SAFE_CALL( cudaMalloc((void**)&dImageB,size) );
    double* dFilterArray;
    CUDA_SAFE_CALL( cudaMalloc((void**)&dFilterArray,9*sizeof(double)) );

    //Copy vectors from host memory to device memory
    CUDA_SAFE_CALL( cudaMemcpy(dImageA,imageA,size,cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(dImageB,imageB,size,cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(dFilterArray,filterArray,9*sizeof(double),cudaMemcpyHostToDevice) );

    timestamp t_start;
    t_start = getTimeStamp();

    //Invoke kernel
    dim3 threadsPerBlock(32,32); //1024 threads which is the max size
    dim3 numBlocks( (rows+threadsPerBlock.x - 1)/threadsPerBlock.x, (columns+threadsPerBlock.y - 1)/threadsPerBlock.y);
    while(repeatFilter){
        loops++;
        filter<<<numBlocks,threadsPerBlock>>>(dImageA,dImageB,rows,columns,dFilterArray,loops);
        if (loops%STEPS == 0 ){
            CUDA_SAFE_CALL( cudaMemcpy(imageA,dImageA,size,cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaMemcpy(imageB,dImageB,size,cudaMemcpyDeviceToHost) );
            repeatFilter = 0;
            for(i=0;i<rows*columns;i++){
                if (imageA[i] != imageB[i]){
                    repeatFilter = 1;
                    break;
                }
            }
        }
    }
    CUDA_SAFE_CALL( cudaGetLastError() );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    float msecs = getElapsedtime(t_start);

    //free device memory
    CUDA_SAFE_CALL( cudaFree(dImageA) );
    CUDA_SAFE_CALL( cudaFree(dImageB) );
    CUDA_SAFE_CALL( cudaFree(dFilterArray) );

    return msecs;
}

