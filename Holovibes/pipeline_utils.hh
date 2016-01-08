#pragma once

# include <functional>
# include <vector>
# include <deque>
# include <cuda_runtime.h>
# include <cufft.h>

namespace holovibes
{
  /*! \brief Vector of procedures type */
  using FnType = std::function<void()>;
  using FnVector = std::vector<FnType>;
  using FnDeque = std::deque<FnType>;
}

template <class X, class Res, class Y>
Res cudaDestroy(X* addr_buf, Res(*f)(Y))
{
  Res res = static_cast<Res>(0);

  if (*addr_buf)
    res = (*f)(*addr_buf);
  *addr_buf = 0;
  return (res);
}

template <class Res>
Res cudaDestroy(cufftComplex** addr_buf)
{
  return (cudaDestroy<cufftComplex*, Res, void*>(addr_buf, &cudaFree));
}

template <class Res>
Res cudaDestroy(float** addr_buf)
{
  return (cudaDestroy<float*, Res, void*>(addr_buf, &cudaFree));
}

template <class Res>
Res cudaDestroy(cufftHandle* addr_buf)
{
  return (cudaDestroy<cufftHandle, Res, cufftHandle>(addr_buf, &cufftDestroy));
}