#include "fft2.cuh"


void create_spectral(cufftComplex *output, float lambda, float distance, int size_x, int size_y, int pasu, int pasv)
{
  float *u;
  float *v;
  cudaMalloc(&u, size_x * sizeof(float));
  cudaMalloc(&v, size_y * sizeof(float));

  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size_x + threads - 1) / threads;

  // Hardware limit !!
  if (blocks > get_max_blocks())
    blocks = get_max_blocks() - 1;

  fft2_make_u_v<<<blocks, threads>>>(pasu, pasv, u, v, size_x, size_y);

  int mesh_size_u = size_x * size_x;
  int mesh_size_v = size_y * size_y;
  float* u_mesh;
  float* v_mesh;
  cudaMalloc(&u_mesh, mesh_size_u * sizeof(float));
  cudaMalloc(&v_mesh, mesh_size_v * sizeof(float));

  blocks = ((size_x * size_y) + threads - 1) / threads;
  // Hardware limit !!
  if (blocks > get_max_blocks())
    blocks = get_max_blocks() - 1;

  meshgrind_square<<<blocks,threads>>>(u, v, u_mesh, v_mesh, size_x, size_y);
  cudaFree(u);
  cudaFree(v);

  spectral <<<blocks,threads>>>(u_mesh, v_mesh, output, size_x * size_y, lambda, distance);
  cudaFree(u_mesh);
  cudaFree(v_mesh);
}