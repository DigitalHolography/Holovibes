/*! \file
 *
 * \brief Declaration of the CufftHandle class
 */
#pragma once

#include <memory>

#include <cufft.h>
#include <cufftXt.h>

namespace holovibes::cuda_tools
{
/*! \class CufftHandle
 *
 * \brief RAII wrapper for cufftHandle
 */
class CufftHandle
{
  public:
    /*! \brief Default constructor */
    CufftHandle();

    static void set_stream(const cudaStream_t& stream);

    /*! \brief Constructor calling plan2d */
    CufftHandle(const int x, const int y, const cufftType type);

    /*! \brief Destroy the created plan (if any) */
    ~CufftHandle();

    /*! \brief Destroy the created plan (if any) */
    void reset();

    /*! \brief Calls plan2d
     *
     * Could be overloaded for plan1d and plan3d
     */
    void plan(const int x, const int y, const cufftType type);

    /*! \brief Calls planMany */
    void planMany(int rank,
                  int* n,
                  int* inembed,
                  int istride,
                  int idist,
                  int* onembed,
                  int ostride,
                  int odist,
                  cufftType type,
                  int batch);

    /*! \brief Calls XtplanMany
     *
     * Warning ! Allocating a plan has a VRAM memory cost :
     * You can check the amount of memory used by checking the ws (working size)
     * variable
     *
     * The memory taken by the plan is used by cufft to make the FFT
     * It's equal in byte to : frame_width * frame_height * sizeof(T) (in our
     * case cuComplex) * batch_size
     */
    void XtplanMany(int rank,
                    long long* n,
                    long long* inembed,
                    long long istride,
                    long long idist,
                    cudaDataType inputtype,
                    long long* onembed,
                    long long ostride,
                    long long odist,
                    cudaDataType outputtype,
                    long long batch,
                    cudaDataType executiontype);

    /*! \brief Get a reference to the underlying cufftHandle */
    cufftHandle& get();

    /*! \brief Implicit cast to the underlying cufftHandle */
    operator cufftHandle&();

    /*! \brief Implicit cast to a ptr to the underlying cufftHandle */
    operator cufftHandle*();

  private:
    /*! \brief The cufftHandle
     *
     * We chose a unique_ptr to represent a possibly uninitialized one
     */
    std::unique_ptr<cufftHandle> val_;

    static cudaStream_t stream_;
};
} // namespace holovibes::cuda_tools
