#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#define STEPS 25

unsigned char* readImage(int,int);
unsigned char* scatterImageData(int,int,unsigned char*,MPI_Comm*,int,int,int,int);
void filterImage(int,unsigned char*,int,int,MPI_Comm*);
void createCartCoordinates(int,MPI_Comm*);
void prepareCommunication(int,unsigned char*,int,int,MPI_Comm*,MPI_Request*,MPI_Request*);
double* chooseFilter();
void filterInnerElements(unsigned char*,unsigned char*,int,int,double*);
void filterOuterElements(unsigned char*,unsigned char*,int,int,double*);
void findPixelNeighbors(int,int,unsigned char*,int,int,unsigned char*,unsigned char*,unsigned char*,unsigned char*,unsigned char*,unsigned char*,unsigned char*,unsigned char*,unsigned char*);
void print(int,int,int,int,int,int,unsigned char*,unsigned char*,MPI_Comm*);
void filterLocalImage(int*,int,unsigned char*,unsigned char*,int,int,MPI_Comm*,MPI_Request*,MPI_Request*,double*,MPI_Status*,MPI_Status*,int);
