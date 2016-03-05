#include "header.h"

extern "C" float filterImage(int,int,int,unsigned char*,unsigned char*,double*);

int main(void)
{
    unsigned char *imageA,*localImage;
    int i,myRank,rows = 1920,columns = 2520;
    float msecs;

    //read and store image's elements
    imageA = malloc(rows*columns*sizeof(unsigned char)); //create image 1-D array
    if(imageA == NULL){
        printf("Error:Not available memory\n");
        exit(1);
    }
    FILE* fp;
    fp = fopen("picture.txt","r");
    if(fp == NULL){
        printf("Error\n");
        exit(1);
    }
    for (i=0;i<rows*columns;i++){
        if(fscanf(fp,"%c",&image[i]) == EOF){
            rewind(fp);
        }
    }
    fclose(fp);
    //image = readImage(rows,columns);

    double filterArray[] = {0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625}; //define the filter
    size_t size = rows * columns * sizeof(unsigned char);
    unsigned char* imageB = malloc(size);
    //apply the filter to the image until no changes occur
    msecs = filterImage(rows,columns,size,imageA,imageB,filterArray);
    printf("Execution time %.2f m secs",msecs);

    //free host memory
    free(imageB);
    free(image);
    return 0;
}
