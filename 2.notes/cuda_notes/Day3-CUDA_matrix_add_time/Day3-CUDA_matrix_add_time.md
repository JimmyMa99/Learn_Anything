# CUDA程序基本框架
```cpp
#include <头文件>

  

__global__ void 函数名(参数列表) {

    // 代码

}

  

int main(void){

    设置GPU设备;

    分配主机和设备内存;

    初始化主机中的数据;

    数据从主机复制到设备;

    调用核函数<<<blocks, threads>>>(参数列表);

    将计算得到的数据从设备传到主机;

    释放主机与设备内存;

}

```

## 设置GPU设备
1. 获取GPU设备数量
	```cpp
	int iDeviceCount = 0;
	cudaGetDeviceCount(&iDeviceCount);
	
	```
2. 设置GPU执行时使用的设备
	```
	int iDev = 0;
	cudaSetDevice(iDev)
	```
## 内存管理
- CUDA通过内存分配、数据传递、内存初始化、内存释放进行内存管理
- 标准C语言内存管理函数->CUDA内存管理函数

|       | C FUNC | CUDA C FUNC |
| :---: | :----: | :---------: |
| 内存分配  | malloc | cudaMalloc  |
| 数据传递  | memcpy | cudaMemcpy  |
| 内存初始化 | memset | cudaMemset  |
| 内存释放  |  free  |  cudaFree   |
### 内存分配
- 主机分配内存： `extern void *malloc(unsigned int num_bytes);`
- 代码：
```cpp
float *fpHost_A
fpHost_A = (float *)malloc(nBytes);
```
- 设备分配内存
- 代码：
```cpp
float*fpDeviceA;
cudaMalloe(float**)&fpDeviceA,nBytes);
```

## 数据的拷贝
- 主机数据拷贝：`void *memcpy(void *dest, const void *src,size_t n);`
		代码：`memcpy((void*)d, (void*)s, nBytes)
- 设备数据拷贝：
		代码：`cudaMemcpy(Device_A, Host_A, nBytes, cudaMemcpyHostToHost)

kind为最后一个参数，可选为

|           kind           | 解释    |
| :----------------------: | ----- |
|   cudaMemcpyHostToHost   | 主机→主机 |
|  cudaMemcpyHostToDevice  | 主机→设备 |
|  cudaMemcpyDeviceToHost  | 设备→主机 |
| cudaMemcpyDeviceToDevice | 设备→设备 |
cudaMemcpyDefault默认默认方式只允许在支持统一虚拟寻址的系统上使用

## 内存初始化
- 主机内存初始化：`void *memset（void *str,intc ,size_tn);`
	代码：
	```cpp
	memset(fpHostA,O,nBytes);
	
	```
- 设备内存初始化
	代码：
	```cpp
	cudaMemset(fpDevice_A,0,nBytes);
	```
## 内存释放
- 释放主机内存：
	代码：`free(pHost_A)`
- 释放设备内存：
	代码：`cudaFree(pDevice_A)`
	
## 自定义设备函数
1. 设备函数（device function）
	（1）定义只能执行在GPU设备上的函数为设备函数
	（2）设备函数只能被核函数或其他设备函数调用
	（3）设备函数用`__device__`修饰
2. 核函数（kernel function）
	（1）用`__global__`修饰的函数称为核函数，一般由主机调用，在设备中执行
	（2）`__global__` 修饰符既不能和`__host__`同时使用，也不可与`__device__`同时使用
3. 主机函数（host function）
	（1）主机端的普通C++函数可用`__host__`修饰
	（2）对于主机端的函数，`__host__`修饰符可省略
	（3）可以用`__host__`和`__device__`同时修饰一个函数减少余代码。编译器会针对主机和设备分别编译该函数。
# CUDA错误检查
## 运行时API错误代码

1. CUDA运行时API大多支持返回错误代码，返回值类型：cudaError_t
2. 运行时API成功执行，返回值为cudaSuccess
3. 运行时API返回的执行状态值是枚举变量
## 错误检查函数
1. 获取错误代码对应名称：cudaGetErrorName
2. 获取错误代码描述信息：cudaGetErrorString
1、在调用CUDA运行时API时，调用ErrorCheck函数进行包装
2、参数filename一般使用FILE；参数lineNumber一般使用__LINE__
3、错误函数返回运行时API调用的错误代码
## 检查核函数
错误检测函数问题：不能捕捉调用核函数的相关错误
捕捉调用核函数可能发生错误的方法：
	```
	ErrorCheck(cudaGetLastError(),__FILE__,__LINE__);
	ErrorCheck(cudaDeviceSynchronize(),__FILE__,__LINE__;
	```
	
核函数定义：`__global__ void kernel function(argument arg);`

# CUDA记时
## 事件记时
1. 程序执行时间记时：是CUDA程序执行性能的重要表现
2. 使用CUDA事件（event）记时方式
3. CUDA事件记时可为主机代码、设备代码记时
一般不记录第一次核函数计算时间，因为第一次往往会花费更多
取后面10次的平均
## nvprof性能剖析
1. nvprof是一个可执行文件
2. 执行命令：nvprof./exename
使用nvprof并不需要使用计时函数进行计时

由于4060不支持直接使用，会报
```bash
======== Warning: This version of nvprof doesn't support the underlying device, GPU profiling skipped
======== Warning: nvprof is not supported on devices with compute capability 8.0 and higher.
                  Use NVIDIA Nsight Systems for GPU tracing and CPU sampling and NVIDIA Nsight Compute for GPU profiling.
                  Refer https://developer.nvidia.com/tools-overview for more details.
```
解决方案：
[Nsight Systems - Get Started | NVIDIA Developer](https://developer.nvidia.com/nsight-systems/get-started)
安装 nsight-systems
```bash
wget -c https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_2/nsightsystems-linux-public-2024.2.1.106-3403790.run

bash nsightsystems-linux-public-2024.2.1.106-3403790.run

export PATH="/home/mzm/nsight-systems-2024.2.1/bin:$PATH"
```

```bash
#修改.bashrc
cd ~
ls -a
vim .bashrc
#在最后加入 export PATH="/home/mzm/nsight-systems-2024.2.1/bin:$PATH"
source ~/.bashrc
```

```bash
nsys nvprof ./nvprofAnalysis
WARNING: nvprofAnalysis and any of its children processes will be profiled.

The count of GPUs is 1.
set GPU 0 for computing.
Generating '/tmp/nsys-report-6045.qdstrm'
[1/7] [========================100%] report1.nsys-rep
[2/7] [========================100%] report1.sqlite
[3/7] Executing 'nvtx_sum' stats report
SKIPPED: /home/mzm/Code/cudacode/CUDA-code/3.3lesson/report1.sqlite does not contain NV Tools Extension (NVTX) data.
[4/7] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)        Name      
 --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ----------------
     82.4        118838977          3  39612992.3      3604.0      2802  118832571   68606167.6  cudaMalloc      
     15.8         22809271          1  22809271.0  22809271.0  22809271   22809271          0.0  cudaDeviceReset 
      1.7          2379878          3    793292.7    446518.0      5094    1928266    1007391.3  cudaFree        
      0.1            97619          4     24404.8     19888.5     16672      41170      11279.2  cudaMemcpy      
      0.0            47481          3     15827.0      8384.0      8243      30854      13014.0  cudaMemset      
      0.0            12245          1     12245.0     12245.0     12245      12245          0.0  cudaLaunchKernel
      0.0             2945          1      2945.0      2945.0      2945       2945          0.0  cuCtxSynchronize

[5/7] Executing 'cuda_gpu_kern_sum' stats report
SKIPPED: /home/mzm/Code/cudacode/CUDA-code/3.3lesson/report1.sqlite does not contain CUDA kernel data.
[6/7] Executing 'cuda_gpu_mem_time_sum' stats report
SKIPPED: /home/mzm/Code/cudacode/CUDA-code/3.3lesson/report1.sqlite does not contain GPU memory data.
[7/7] Executing 'cuda_gpu_mem_size_sum' stats report
SKIPPED: /home/mzm/Code/cudacode/CUDA-code/3.3lesson/report1.sqlite does not contain GPU memory data.
Generated:
    /home/mzm/Code/cudacode/CUDA-code/3.3lesson/report1.nsys-rep
    /home/mzm/Code/cudacode/CUDA-code/3.3lesson/report1.sqlite
```