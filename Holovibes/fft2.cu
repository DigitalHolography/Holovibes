#include "fft2.cuh"


cufftComplex *create_spectral(float lambda, float distance, int size_x, int size_y, float pasu, float pasv)
{
  cufftComplex *output;
  cudaMalloc(&output, size_x * size_y * sizeof(cufftComplex));
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
  return output;
}

void fft_2(int nbimages, holovibes::Queue *q, cufftComplex *lens, float *sqrt_vect, unsigned short *result_buffer, cufftHandle plan)
{
  // Sizes
  unsigned int pixel_size = q->get_frame_desc().width * q->get_frame_desc().height * nbimages;
  unsigned int complex_size = pixel_size * sizeof(cufftComplex);
  unsigned int short_size = pixel_size * sizeof(unsigned short);

  // Loaded images --> complex
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (pixel_size + threads - 1) / threads;

  // Hardware limit !!
  if (blocks > get_max_blocks())
    blocks = get_max_blocks() - 1;
  blocks = 65535;


  cufftComplex* complex_input = make_contigous_complex(q, nbimages, sqrt_vect);
  cufftExecC2C(plan, complex_input, complex_input, CUFFT_FORWARD);
  apply_quadratic_lens <<<blocks, threads >> >(complex_input, pixel_size, lens, q->get_pixels());
  cufftExecC2C(plan, complex_input, complex_input, CUFFT_FORWARD);
  
  //back to real
  complex_2_module << <blocks, threads >> >(complex_input, result_buffer, pixel_size);

  // Free all
  cudaFree(complex_input);

}