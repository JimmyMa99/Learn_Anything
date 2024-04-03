#include <stdio.h>

__global__ void hello_from_gpu(void) {
    const int bid = blockIdx.x; //这个函数会返回当前block的ID
    const int tid = threadIdx.x; //这个函数会返回当前线程的ID
    const int id = threadIdx.x + blockIdx.x * blockDim.x; //这个函数会返回当前线程的全局ID

    printf("Hello from block %d, thread %d, global thread %d!\n", bid, tid, id); //这会在GPU的所有线程中打印
    // printf("Hello from GPU!\n"); //这会在GPU的所有线程中打印
}

int main(void) {
    hello_from_gpu<<<2, 4>>>(); //grid的大小是2，block的大小是4
    cudaDeviceSynchronize();    //这个函数会等待GPU上的所有线程都执行完
    return 0;
}
