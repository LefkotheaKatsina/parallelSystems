#include "header.h"

int main(int argc,char* argv[])
{
    unsigned char *image,*localImage;
    int i,myRank,commSize,rows = 1920,columns = 2520;

    MPI_Comm newComm;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&commSize);
    MPI_Comm_rank(MPI_COMM_WORLD,&myRank);

    int npRows = sqrt((double)commSize);
    int npColumns = sqrt((double)commSize);
    int blockRows = rows/npRows; //number of rows in each block
    int blockColumns = columns/npColumns; //number of columns in each block

    //get number of threads from command line
    int thread_count = strtol(argv[1],NULL,10);
    //create a 2-D Cartesian Coordinate structure on the processes
    createCartCoordinates(commSize,&newComm);
    //read and store image's elements
    if (myRank == 0){
        FILE* fp;
        fp = fopen("picture.txt","r");
        if(fp == NULL){
            printf("Error\n");
            exit(1);
        }
        image = malloc(rows*columns*sizeof(unsigned char)); //create image 1-D array
        if(image == NULL){
            printf("Error:Not available memory\n");
            MPI_Finalize();
            exit(1);
        }
        for (i=0;i<rows*columns;i++){
            if(fscanf(fp,"%c",&image[i]) == EOF){
                rewind(fp);
            }
        }
        //image = readImage(rows,columns);
        fclose(fp);
    }
    //scatter image data to processes
    localImage = scatterImageData(blockColumns,blockRows,image,&newComm,rows,columns,commSize,myRank);
    //print(myRank,commSize,rows,columns,blockRows+2,blockColumns+2,image,localImage,&newComm);
    //apply the filter to the image until no changes occur
    filterImage(myRank,localImage,blockColumns+2,blockRows+2,&newComm,thread_count);
    if (myRank == 0)
        free(image);
    MPI_Finalize();
    return 0;
}
