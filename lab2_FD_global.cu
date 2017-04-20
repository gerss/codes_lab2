/*** Heat equation with FD in global memory ***/
#include <iostream>
#include <fstream>
#include <cmath>
#include <sys/time.h>

void checkErrors(char *label)
{
// we need to synchronise first to catch errors due to
// asynchroneous operations that would otherwise
// potentially go unnoticed
cudaError_t err;
err = cudaThreadSynchronize();
if (err != cudaSuccess)
{
char *e = (char*) cudaGetErrorString(err);
fprintf(stderr, "CUDA Error: %s (at %s)\n", e, label);
}
err = cudaGetLastError();
if (err != cudaSuccess)
{
char *e = (char*) cudaGetErrorString(err);
fprintf(stderr, "CUDA Error: %s (at %s)\n", e, label);
}
}

double get_time() 
{  struct timeval tim;
  cudaThreadSynchronize();
  gettimeofday(&tim, NULL);
  return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}

// GPU kernels
__global__ void copy_array (float *u, float *u_prev, int N, int BSZ)
{	/******* write your kernel here! ***/

	int i = threadIdx.x + BSZ*blockIdx.x;
	int j = threadIdx.y + BSZ*blockIdx.y;
	int I;

	if(j<N)
	{	if(i<N)
		{	I=j*N+i;
			u_prev[I] = u[I];
		}
	}

}

	__syncthreads()

__global__ void update (float *u, float *u_prev, int N, float h, float dt, float alpha, int BSZ)
{	/***** write your kernel here! ***/
	
	int i = threadIdx.x + BSZ*blockIdx.x;
	int j = threadIdx.y + BSZ*blockIdx.y;
	int I;
	
	if(j<N)
	{
		if(i<N)
		{	I=j*N+i;
			u[I] = u_prev[I] + alpha*dt/(h*h) * (u_prev[I+1] + u_prev[I-1] + u_prev[I+N] + u_prev[I-N] - 4*u_prev[I]);
		}
	}
}
	__syncthreads()

int main()
{
printf("----------------------------------------\n");
printf(" ALLOCATE AND INITIALIZE DATA ON CPU\n");
printf("----------------------------------------\n");

	// Allocate in CPU
	int N = 128;
	std::cout<<"N° of threads : "<<N*N<<std::endl;
	int BLOCKSIZE = 16;
	int block_grid   = int((N-0.5)/BLOCKSIZE)+1;
	std::cout<<"N° of block : "<<block_grid<<std::endl;

	float xmin 	= 0.0f;
	float xmax 	= 3.5f;
	float ymin 	= 0.0f;
	//float ymax 	= 2.0f;
	float h   	= (xmax-xmin)/(N-1);
	float dt	= 0.00001f;	
	float alpha	= 0.645f;
	float time 	= 0.4f;

	int steps = ceil(time/dt);
	int I;

	float *x  	= new float[N*N]; 
	float *y  	= new float[N*N]; 
	float *u  	= new float[N*N];
	float *u_prev  	= new float[N*N];

	printf("\n DATOS INICIALES");
	std::cout<<"Distancia eje x : "<<xmax-xmin<<std::endl;
	std::cout<<"dx : "<<h<<std::endl;
	std::cout<<"Time : "<<time<<std::endl;
	std::cout<<"dt : "<<dt<<std::endl;
	std::cout<<"Pasos de tiempo : "<<steps<<std::endl;

	// Generate mesh and intial condition
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = N*j + i;
			x[I] = xmin + h*i;
			y[I] = ymin + h*j;
			u[I] = 0.0f;
			if ( (i==0) || (j==0)) 
				{u[I] = 200.0f;}
		}
	}
printf("\n Generate mesh\n");
	
printf("----------------------------------------\n");
printf(" ALLOCATE DATA ON GPU\n");
printf("----------------------------------------\n");
	// Allocate in GPU
	float *u_d, *u_prev_d;

	cudaMalloc((void**) &u_d, N*N*sizeof(float));
	cudaMalloc((void**) &u_prev_d, N*N*sizeof(float));

	// Copy to GPU
printf(" TRANSFER DATA FROM CPU TO GPU\n");
printf("----------------------------------------\n");

	cudaMemcpy(u_d, u, N*N*sizeof(float), cudaMemcpyHostToDevice);

	// Loop 
printf(" RUN KERNEL");

	
	dim3 dimGrid(block_grid, block_grid); // number of blocks?
	dim3 dimBlock(BLOCKSIZE,BLOCKSIZE); // threads per block?

	double start = get_time(); //Initial time
	for (int t=0; t<steps; t++)
	{	copy_array <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N, BLOCKSIZE);
		update <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N, h, dt, alpha, BLOCKSIZE);
	}
	double finish = get_time(); //Final time
	double diff = finish - start;
	std::cout<<"  time ="<<diff<<" [s]"<<std::endl;
	
	// Copy result back to host
printf("----------------------------------------\n");
printf(" TRANSFER DATA FROM GPU TO CPU\n");
printf("----------------------------------------\n");

	cudaMemcpy(u, u_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);

	std::ofstream temperature("temperature_global.txt");
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = N*j + i;
			temperature<<x[I]<<"\t"<<y[I]<<"\t"<<u[I]<<std::endl;
		}
		temperature<<"\n";
	}

	temperature.close();


	// Free device
	cudaFree(u_d);
	cudaFree(u_prev_d);

printf(" FREE MEMORY\n");
printf("----------------------------------------\n");

}
