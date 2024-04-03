#include <stdio.h>

__global__ void hello_from_gpu(void) {
    printf("Hello from GPU!\n"); //这会在GPU的所有线程中打印
}

int main(void) {
    hello_from_gpu<<<2, 4>>>(); //前面的1代表启动一个block，后面的10代表block中有10个线程
    cudaDeviceSynchronize();    //这个函数会等待GPU上的所有线程都执行完
    return 0;
}
