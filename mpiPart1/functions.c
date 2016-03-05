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

unsigned char* scatterImageData(int blockColumns,int blockRows,unsigned char* image,MPI_Comm* newComm,int rows,int columns,int commSize,int myRank){
    int i,j,k,*displs,*sendCounts;
    unsigned char *localImage;
    MPI_Datatype matrixType;
    MPI_Datatype matrixType1;
    MPI_Datatype matrixType2;
    int npRows = sqrt((double)commSize);
    int npColumns = sqrt((double)commSize);

    localImage = malloc((blockColumns+2)*(blockRows+2)*sizeof(unsigned char)); //create localImage 1-D array
    if(localImage == NULL){
        printf("Error:Not available memory\n");
        MPI_Finalize();
        exit(1);
    }
    //for (i=0; i<(blockRows+2)*(blockColumns+2); i++) localImage[i] = 0;

    displs = malloc(npRows*npColumns*sizeof(int)); //create displacements 1-D array
    if(displs == NULL){
        printf("Error:Not available memory\n");
        MPI_Finalize();
        exit(1);
    }
    sendCounts = malloc(npRows*npColumns*sizeof(int)); //create sendCounts 1-D array
    if(sendCounts == NULL){
        printf("Error:Not available memory\n");
        MPI_Finalize();
        exit(1);
    }
    //find displacements and sendCounts that will be used in Scatter-v
    for(i=0;i<npRows;i++){
        for(j=0;j<npColumns;j++){
            displs[i*npColumns+j] = i*columns*blockRows+j*blockColumns;
            sendCounts[i*npColumns+j] = 1;
        }
    }
    //define the matrixType block that will be send
    MPI_Type_vector(blockRows,blockColumns,columns,MPI_UNSIGNED_CHAR,&matrixType);
    MPI_Type_create_resized(matrixType, 0, sizeof(unsigned char), &matrixType1);
    MPI_Type_commit(&matrixType1);
    //define the received matrixType block
    MPI_Type_vector(blockRows, blockColumns,blockColumns+2,MPI_UNSIGNED_CHAR,&matrixType);
    MPI_Type_create_resized(matrixType, 0, sizeof(unsigned char), &matrixType2);
    MPI_Type_commit(&matrixType2);
    //scatter the image elements to processes,each process gets a matrixType block
    //the block is placed in localImage in a way that an external grid is formed
    MPI_Scatterv(image,sendCounts,displs,matrixType1,&localImage[blockColumns+3],1,matrixType2,0,*newComm);
    free(displs);
    free(sendCounts);
    MPI_Type_free(&matrixType1);
    MPI_Type_free(&matrixType2);
    return localImage;
}

void filterImage(int myRank,unsigned char* localImage,int localColumns,int localRows,MPI_Comm* newComm){

    double filterArray[] = {0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625}; //define the filter
    int i,j,repeatFilter = 1,loops = 0,goOn = 0;
    unsigned char* localImageB;
    MPI_Request reqSend[8];
    MPI_Request reqReceive[8];
    MPI_Request reqSendB[8];
    MPI_Request reqReceiveB[8];
    MPI_Status statusReceive[8];
    MPI_Status statusSend[8];

    double localStart,localFinish,localElapsed,elapsed;

    //establish a persistent communication with neighbors
    //localImage and localImageB will be used in turns
    prepareCommunication(myRank,localImage,localColumns,localRows,newComm,reqReceive,reqSend);
    localImageB = malloc(localColumns*localRows*sizeof(unsigned char));
    prepareCommunication(myRank,localImageB,localColumns,localRows,newComm,reqReceiveB,reqSendB);

    MPI_Barrier(*newComm);
    localStart = MPI_Wtime();
    while(repeatFilter == 1){
        loops++;
        if(loops%2 != 0)
            filterLocalImage(&repeatFilter,loops,localImage,localImageB,localColumns,localRows,newComm,reqSend,reqReceive,filterArray,statusSend,statusReceive,myRank);
        else
            filterLocalImage(&repeatFilter,loops,localImageB,localImage,localColumns,localRows,newComm,reqSendB,reqReceiveB,filterArray,statusSend,statusReceive,myRank);
    }
    localFinish = MPI_Wtime();
    localElapsed = localFinish-localStart;
    MPI_Reduce(&localElapsed,&elapsed,1,MPI_DOUBLE,MPI_MAX,0,*newComm);

    if(myRank == 0){
        printf("%d",loops);
        printf("Elapsed time = %f seconds\n",elapsed);
        fflush(stdout);
    }
    MPI_Request_free(reqSend);
    MPI_Request_free(reqReceive);
    free(localImage);
    free(localImageB);
    return;
}

void filterLocalImage(int* repeatFilter,int loops,unsigned char* localImagePrev,unsigned char* localImageNew,int localColumns,
                      int localRows,MPI_Comm* newComm,MPI_Request* reqSend,MPI_Request* reqReceive,double* filterArray,
                      MPI_Status* statusSend,MPI_Status* statusReceive,int myRank){
    int i,j,goOn = 0;
        MPI_Startall(8,reqSend); //send elements to neighbor processes
        MPI_Startall(8,reqReceive); //receive elements from neighbor processes
        //while waiting,filter innerElements
        filterInnerElements(localImagePrev,localImageNew,localRows,localColumns,filterArray);
        MPI_Waitall(8,reqReceive,statusReceive);
        //when elements from neighbor processes are received,filter outerElements
        filterOuterElements(localImagePrev,localImageNew,localRows,localColumns,filterArray);
        if(loops%STEPS == 0){
            for(i=1;i<localRows-1;i++){
                for(j=1;j<localColumns-1;j++){
                    if (localImagePrev[i*localColumns+j] != localImageNew[i*localColumns+j]){
                        goOn = 1;
                        break;
                    }
                }
                if (goOn == 1)
                    break;
            }
            MPI_Reduce(&goOn,repeatFilter,1,MPI_INT,MPI_MAX,0,*newComm);
            MPI_Bcast(repeatFilter,1,MPI_INT,0,*newComm);
        }
        MPI_Waitall(8,reqSend,statusSend);
        return;
}

void prepareCommunication(int myRank,unsigned char* localImage,int localColumns,int localRows,MPI_Comm *newComm,MPI_Request* reqReceive,MPI_Request* reqSend){
    int index;
    int displ = 1;
    int source;
    int dest;
    MPI_Datatype blockTypeHoriz,blockTypeHoriz1,blockTypeVert,blockTypeVert1;

    MPI_Type_vector(1,localColumns-2,0,MPI_UNSIGNED_CHAR,&blockTypeHoriz1); //define a HorizontalType block
    MPI_Type_create_resized(blockTypeHoriz1,0,sizeof(unsigned char),&blockTypeHoriz);
    MPI_Type_commit(&blockTypeHoriz);

    MPI_Type_vector(localRows-2,1,localColumns,MPI_UNSIGNED_CHAR,&blockTypeVert1); //define a VerticalType block
    MPI_Type_create_resized(blockTypeVert1,0,sizeof(unsigned char),&blockTypeVert);
    MPI_Type_commit(&blockTypeVert);

    index = 0; //find upper and lower neighbor processes
    MPI_Cart_shift(*newComm,index,displ,&source,&dest);
    //establish a persistent communication with upper and lower neighbor processes
    MPI_Send_init(&localImage[localColumns+1],1,blockTypeHoriz,source,0,*newComm,&reqSend[0]);
    MPI_Recv_init(&localImage[1],1,blockTypeHoriz,source,1,*newComm,&reqReceive[0]);
    MPI_Send_init(&localImage[(localRows-2)*localColumns+1],1,blockTypeHoriz,dest,1,*newComm,&reqSend[1]);
    MPI_Recv_init(&localImage[(localRows-1)*localColumns+1],1,blockTypeHoriz,dest,0,*newComm,&reqReceive[1]);

    index = 1; //find left and right neighbor processes
    MPI_Cart_shift(*newComm,index,displ,&source,&dest);
    //establish a persistent communication with left and right neighbor processes
    MPI_Send_init(&localImage[localColumns+1],1,blockTypeVert,source,2,*newComm,&reqSend[2]);
    MPI_Recv_init(&localImage[localColumns],1,blockTypeVert,source,3,*newComm,&reqReceive[2]);
    MPI_Send_init(&localImage[2*localColumns-2],1,blockTypeVert,dest,3,*newComm,&reqSend[3]);
    MPI_Recv_init(&localImage[2*localColumns-1],1,blockTypeVert,dest,2,*newComm,&reqReceive[3]);

    int myCoords[2]; //coordinates of process running
    int coords[2]; //coordinates of process neighbor
    MPI_Cart_coords(*newComm,myRank,2,myCoords); //find my own coordinates
    int row,column,rank_id;

    row = myCoords[0]-1;
    column = myCoords[1]-1; //find upper left neighbor process
    coords[0] = row;
    coords[1] = column;
    MPI_Cart_rank(*newComm, coords, &rank_id);
    //establish a persistent communication with upper left neighbor process
    MPI_Send_init(&localImage[localColumns+1],1,MPI_UNSIGNED_CHAR,rank_id,4,*newComm,&reqSend[4]);
    MPI_Recv_init(&localImage[0],1,MPI_UNSIGNED_CHAR,rank_id,7,*newComm,&reqReceive[4]);

    row = myCoords[0]-1;
    column = myCoords[1]+1; //find upper right neighbor process
    coords[0] = row;
    coords[1] = column;
    MPI_Cart_rank(*newComm, coords, &rank_id);
    //establish a persistent communication with upper right neighbor process
    MPI_Send_init(&localImage[2*localColumns-2],1,MPI_UNSIGNED_CHAR,rank_id,5,*newComm,&reqSend[5]);
    MPI_Recv_init(&localImage[localColumns-1],1,MPI_UNSIGNED_CHAR,rank_id,6,*newComm,&reqReceive[5]);

    row = myCoords[0]+1;
    column = myCoords[1]-1; //find lower left neighbor process
    coords[0] = row;
    coords[1] = column;
    MPI_Cart_rank(*newComm, coords, &rank_id);
    //establish a persistent communication with lower left neighbor process
    MPI_Send_init(&localImage[(localRows-2)*localColumns+1],1,MPI_UNSIGNED_CHAR,rank_id,6,*newComm,&reqSend[6]);
    MPI_Recv_init(&localImage[(localRows-1)*localColumns],1,MPI_UNSIGNED_CHAR,rank_id,5,*newComm,&reqReceive[6]);

    row = myCoords[0]+1;
    column = myCoords[1]+1; //find lower right neighbor process
    coords[0] = row;
    coords[1] = column;
    MPI_Cart_rank(*newComm, coords, &rank_id);
    //establish a persistent communication with lower right neighbor process
    MPI_Send_init(&localImage[(localRows-2)*localColumns+localColumns-2],1,MPI_UNSIGNED_CHAR,rank_id,7,*newComm,&reqSend[7]);
    MPI_Recv_init(&localImage[(localRows-1)*localColumns+localColumns-1],1,MPI_UNSIGNED_CHAR,rank_id,4,*newComm,&reqReceive[7]);
    return;
}

void createCartCoordinates(int commSize,MPI_Comm* newComm){
    int ndims, reorder, periods[2], dimSize[2];
    ndims = 2; // 2D matrix/grid
    dimSize[0] = sqrt((double)commSize); // rows
    dimSize[1] = sqrt((double)commSize); // columns
    periods[0] = 1; // rows periodic
    periods[1] = 1; // columns periodic
    reorder = 1; // allows processes reordered for efficiency
    MPI_Cart_create(MPI_COMM_WORLD,ndims,dimSize,periods,reorder,newComm);
    return;
}

void filterInnerElements(unsigned char* localImagePrev,unsigned char* localImageNew,int localRows,int localColumns,double* filterArray){
    unsigned char me,upper,lower,left,right,upperLeft,upperRight,lowerLeft,lowerRight;
    int row,column;

    for(row=2;row<localRows-2;row++){ //for every pixelElement
        for(column=2;column<localColumns-2;column++){
            findPixelNeighbors(localRows,localColumns,localImagePrev,row,column,&me,&upper,&lower,&left,&right,&upperLeft,&upperRight,&lowerLeft,&lowerRight);
            // filter pixelElement
            localImageNew[row*localColumns + column] = (unsigned char)(round(upperLeft*filterArray[0] + upper*filterArray[1] + upperRight*filterArray[2] + left*filterArray[3]+
                me*filterArray[4] + right*filterArray[5] + lowerLeft*filterArray[6] + lower*filterArray[7] + lowerRight*filterArray[8]));
        }
    }
    return;
}

void filterOuterElements(unsigned char* localImagePrev,unsigned char* localImageNew,int localRows,int localColumns,double* filterArray){
    unsigned char me,upper,lower,left,right,upperLeft,upperRight,lowerLeft,lowerRight;
    int row,column;

    row=1; //first row
    for(column=1;column<=localColumns-2;column++){
        findPixelNeighbors(localRows,localColumns,localImagePrev,row,column,&me,&upper,&lower,&left,&right,&upperLeft,&upperRight,&lowerLeft,&lowerRight);
        // filter pixelElement
        localImageNew[row*localColumns + column] = (unsigned char)(round(upperLeft*filterArray[0] + upper*filterArray[1] + upperRight*filterArray[2] + left*filterArray[3]+
                me*filterArray[4] + right*filterArray[5] + lowerLeft*filterArray[6] + lower*filterArray[7] + lowerRight*filterArray[8]));
    }
    for(row=2;row<=localRows-3;row++){
        for(column=1;column<=localColumns-2;column+=localColumns-3){
            findPixelNeighbors(localRows,localColumns,localImagePrev,row,column,&me,&upper,&lower,&left,&right,&upperLeft,&upperRight,&lowerLeft,&lowerRight);
            // filter pixelElement
            localImageNew[row*localColumns + column] = (unsigned char)(round(upperLeft*filterArray[0] + upper*filterArray[1] + upperRight*filterArray[2] + left*filterArray[3]+
                me*filterArray[4] + right*filterArray[5] + lowerLeft*filterArray[6] + lower*filterArray[7] + lowerRight*filterArray[8]));
        }
    }
    row=localRows-2; //last row
    for(column=1;column<=localColumns-2;column++){
        findPixelNeighbors(localRows,localColumns,localImagePrev,row,column,&me,&upper,&lower,&left,&right,&upperLeft,&upperRight,&lowerLeft,&lowerRight);
        // filter pixelElement
        localImageNew[row*localColumns + column] = (unsigned char)(round(upperLeft*filterArray[0] + upper*filterArray[1] + upperRight*filterArray[2] + left*filterArray[3]+
                me*filterArray[4] + right*filterArray[5] + lowerLeft*filterArray[6] + lower*filterArray[7] + lowerRight*filterArray[8]));
    }
    return;
}

void findPixelNeighbors(int localRows,int localColumns,unsigned char* localImage,int row,int column,unsigned char* me,unsigned char* upper,unsigned char* lower,unsigned char* left,unsigned char* right,unsigned char* upperLeft,unsigned char* upperRight,unsigned char* lowerLeft,unsigned char* lowerRight){
    //find all 8 pixel neighbors
    *upper = localImage[(row-1)*localColumns + column];
    *lower = localImage[(row+1)*localColumns + column];
    *left = localImage[row*localColumns + (column-1)];
    *right = localImage[row*localColumns + (column+1)];
    *upperLeft = localImage[(row-1)*localColumns + (column-1)];
    *upperRight = localImage[(row-1)*localColumns + (column+1)];
    *lowerLeft = localImage[(row+1)*localColumns + (column-1)];
    *lowerRight = localImage[(row+1)*localColumns + (column+1)];
    *me = localImage[row*localColumns + column];
    return;
}

//helper print function
void print(int rank,int p,int rows,int columns,int localRows,int localColumns,unsigned char* a,unsigned char* b,MPI_Comm* newComm){
    int ii,jj,proc,coords[2];
    for (proc=0; proc<p; proc++) {
        if (proc == rank) {
            MPI_Cart_coords(*newComm,proc,2,coords);
            printf("Rank = %d Coords = (%d,%d)\n", rank,coords[0],coords[1]);
            if (rank == 0 && a!=NULL ) {
                printf("Global matrix: \n");
                for (ii=0; ii<rows; ii++) {
                    for (jj=0; jj<columns; jj++) {
                        printf("%3d ",(int)a[ii*columns+jj]);
                    }
                    printf("\n");
                }
            }
            printf("Local Matrix:\n");
            for (ii=0; ii<localRows; ii++) {
                for (jj=0; jj<localColumns; jj++) {
                    printf("%3d ",(int)b[ii*localColumns+jj]);
                }
                printf("\n");
            }
            printf("\n");
        }
        MPI_Barrier(*newComm);
    }
}
