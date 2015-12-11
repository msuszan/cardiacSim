/* 
 * Solves the Panfilov model using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory 
 * and reimplementation by Scott B. Baden, UCSD
 * 
 * Modified and  restructured by Didem Unat, Koc University
 *
 */
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "cuPrintf.cu"
using namespace std;


// Utilities
// 

// Timer
// Make successive calls and take a difference to get the elapsed time.
static const double kMicro = 1.0e-6;
double getTime()
{
  struct timeval TV;
  struct timezone TZ;

  const int RC = gettimeofday(&TV, &TZ);
  if(RC == -1) {
    cerr << "ERROR: Bad call to gettimeofday" << endl;
    return(-1);
  }

  return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );

}  // end getTime()

// Allocate a 2D array
double **alloc2D(int m,int n){
  double **E;
  int nx=n, ny=m;
  E = (double**)malloc(sizeof(double*)*ny + sizeof(double)*nx*ny);
  assert(E);
  int j;
  for(j=0;j<ny;j++) 
    E[j] = (double*)(E+ny) + j*nx;
  return(E);
}

// Reports statistics about the computation
// These values should not vary (except to within roundoff)
// when we use different numbers of  processes to solve the problem
double stats(double **E, int m, int n, double *_mx){
  double mx = -1;
  double l2norm = 0;
  int i, j;
  for (j=1; j<=m; j++)
    for (i=1; i<=n; i++) {
      l2norm += E[j][i]*E[j][i];
      if (E[j][i] > mx)
        mx = E[j][i];
    }
  *_mx = mx;
  l2norm /= (double) ((m)*(n));
  l2norm = sqrt(l2norm);
  return l2norm;
}

// External functions
extern "C" {
  void splot(double **E, double T, int niter, int m, int n);
}
void cmdLine(int argc, char *argv[], double& T, int& n, int& px, int& py, int& plot_freq, int& no_comm, int&num_threads, int& block_size);

__global__ 
void KernelMethod(double* E, double *E_prev, double* R, const int N, 
                  double alpha, double dt, double kk, double a, double b, 
                  double epsilon, double M1, double M2) {

  int row = blockIdx.y*blockDim.y + threadIdx.y+1;
  int col = blockIdx.x*blockDim.x + threadIdx.x+1;
  
  if ((row < N+1) && (col < N+1)) {
    // Left ghost region
    E_prev[row*(N+2)]  = E_prev[row*(N+2) + 2];
    // Right ghost region
    E_prev[row*(N+2)+(N+1)] = E_prev[row*(N+2)+(N-1)];
    // Top ghost region
    E_prev[col] = E_prev[2*(N+2)+col];
    // Bottom ghost region
    E_prev[(N+1)*(N+2)+col] = E_prev[(N-1)*(N+2)+col];
  
    int j = row*(N+2), i = col;

    E[row*(N+2)+col] = E_prev[row*(N+2)+col]+alpha*(E_prev[row*(N+2)+col+1]+E_prev[row*(N+2)+col-1]-4*E_prev[row*(N+2)+col]+E_prev[(row-1)*(N+2)+col]+E_prev[(row+1)*(N+2)+col]);
    
    double e = E[j+i]; 
    double r = R[j+i];

    e = e -dt*(kk* e *(e-a)*(e-1)+ e *r);
    r = r + dt*(epsilon+M1* r/( e+M2))*(-r-kk* e*(e-b-1));
     
    E[i+j] = e;
    R[i+j] = r;
   
    }

}


void simulate (const double alpha, const int n, const int m, const double kk,
    const double dt, const double a, const double epsilon,
    const double M1,const double  M2, const double b, 
    double* E_cuda, double* E_prev_cuda, double* R_cuda, const int block_size)
{
  /*int i, j; */
  /* 
   * Copy data from boundary of the computational box 
   * to the padding region, set up for differencing
   * on the boundary of the computational box
   * Using mirror boundaries
   */

  dim3 tblock(block_size, block_size, 1);
  dim3 grid(ceil(m*1.0/tblock.x), ceil(m*1.0/tblock.y));

  KernelMethod<<<grid, tblock>>>(E_cuda, E_prev_cuda, R_cuda, m, alpha, dt, kk, a, b, epsilon, M1, M2);
  cudaThreadSynchronize();
}

void unFlattenArray(double * flat, double ** ary, int N) {
  int j, i;
  for (j=1; j<N+2; j++) {
    for (i=1; i<N+2; i++) {
      ary[j][i] = flat[j*(N+2)+i];
    }
  }
}

// Main program
int main (int argc, char** argv)
{
  /*
   *  Solution arrays
   *   E is the "Excitation" variable, a voltage
   *   R is the "Recovery" variable
   *   E_prev is the Excitation variable for the previous timestep,
   *      and is used in time integration
   */
  double **E, **R, **E_prev;

  // Various constants - these definitions shouldn't change
  const double a=0.1, b=0.1, kk=8.0, M1= 0.07, M2=0.3, epsilon=0.01, d=5e-5;

  double T=1000.0;
  int m=200,n=200;
  int plot_freq = 0;
  int px = 1, py = 1;
  int no_comm = 0;
  int num_threads=1; 
  int block_size = 16;

  cmdLine( argc, argv, T, n,px, py, plot_freq, no_comm, num_threads, block_size);
  m = n;  
  // Allocate contiguous memory for solution arrays
  // The computational box is defined on [1:m+1,1:n+1]
  // We pad the arrays in order to facilitate differencing on the 
  // boundaries of the computation box
  E = alloc2D(m+2,n+2);
  E_prev = alloc2D(m+2,n+2);
  R = alloc2D(m+2,n+2);

  printf("Block size: %d\n", block_size);


  int i,j;
  // Initialization
  for (j=1; j<=m; j++)
    for (i=1; i<=n; i++) 
      E_prev[j][i] = R[j][i] = 0;

  for (j=1; j<=m; j++)
    for (i=n/2+1; i<=n; i++)
      E_prev[j][i] = 1.0;

  for (j=m/2+1; j<=m; j++) 
    for (i=1; i<=n; i++) 
      R[j][i] = 1.0;

  double * E_cuda; 
  double * R_cuda;
  double * E_prev_cuda;

  double * E_flat = (double *) malloc(sizeof(double) * (m+2) * (n+2));
  double * R_flat = (double *) malloc(sizeof(double) * (m+2) * (n+2));
  double * E_prev_flat = (double *) malloc(sizeof(double) * (m+2) * (n+2));

  for (j=1; j<=m; j++) {
    for (i=1; i<=n;i++) {
      E_prev_flat[(j*(n+2))+i] = E_prev[j][i];
      R_flat[(j*(n+2))+i] = R[j][i];
    }
  }

  cudaMalloc((void**) &E_cuda, (m+2)*(m+2)*sizeof(double));
  cudaMalloc((void**) &R_cuda, (m+2)*(m+2)*sizeof(double));
  cudaMalloc((void**) &E_prev_cuda, (m+2)*(m+2)*sizeof(double));

  cudaMemcpy(E_cuda, E_flat, (m+2)*(n+2)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(R_cuda, R_flat, (m+2)*(n+2)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(E_prev_cuda, E_prev_flat, (m+2)*(n+2)*sizeof(double), cudaMemcpyHostToDevice);

  double dx = 1.0/n;
  // For time integration, these values shouldn't change 
  double rp= kk*(b+1)*(b+1)/4;
  double dte=(dx*dx)/(d*4+((dx*dx))*(rp+kk));
  double dtr=1/(epsilon+((M1/M2)*rp));
  double dt = (dte<dtr) ? 0.95*dte : 0.95*dtr;
  double alpha = d*dt/(dx*dx);

  cout << "Grid Size       : " << n << endl; 
  cout << "Duration of Sim : " << T << endl; 
  cout << "Time step dt    : " << dt << endl; 
  cout << "Process geometry: " << px << " x " << py << endl;
  if (no_comm)
    cout << "Communication   : DISABLED" << endl;

  cout << endl;

  // Start the timer
  double t0 = getTime();

  // Simulated time is different from the integer timestep number
  // Simulated time
  double t = 0.0;
  // Integer timestep number
  int niter=0;


  while (t<T) {

    t += dt;
    niter++;

    simulate(alpha, n, m, kk, dt, a, epsilon, M1, M2, b, E_cuda, E_prev_cuda, R_cuda, block_size); 

    //swap current E with previous E
    double *tmp = E_cuda; E_cuda = E_prev_cuda; E_prev_cuda = tmp;

    if (plot_freq){
      int k = (int)(t/plot_freq);
      if ((t - k * plot_freq) < dt){
        splot(E,t,niter,m+2,n+2);
      }
    }
  }//end of while loop

  cudaMemcpy(E_flat, E_cuda, (m+2)*(n+2)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(R_flat, R_cuda, (m+2)*(n+2)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(E_prev_flat, E_prev_cuda, (m+2)*(n+2)*sizeof(double), cudaMemcpyDeviceToHost);

  double time_elapsed = getTime() - t0;

  unFlattenArray(E_flat, E, m);
  unFlattenArray(E_prev_flat, E_prev, m);
  unFlattenArray(R_flat, R, m);

  double Gflops = (double)(niter * (1E-9 * n * n ) * 28.0) / time_elapsed ;
  double BW = (double)(niter * 1E-9 * (n * n * sizeof(double) * 4.0  ))/time_elapsed;

  cout << "Number of Iterations        : " << niter << endl;
  cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
  cout << "Sustained Gflops Rate       : " << Gflops << endl; 
  cout << "Sustained Bandwidth (GB/sec): " << BW << endl << endl; 

  double mx;
  double l2norm = stats(E_prev,m,n,&mx);
  cout << "Max: " << mx <<  " L2norm: "<< l2norm << endl;

  if (plot_freq){
    cout << "\n\nEnter any input to close the program and the plot..." << endl;
    getchar();
  }

  free (E);
  free (E_prev);
  free (R);

  free (E_flat);
  free (E_prev_flat);
  free (R_flat);

  cudaFree(E_cuda);
  cudaFree(E_prev_cuda);
  cudaFree(R_cuda);
  return 0;
}
