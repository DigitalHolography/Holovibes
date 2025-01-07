/*! \file cufft_handle.hh
 *
 *  \brief Declaration of the CufftHandle class. Used to wrap the `cufftHandle` class of CUDA.
 */
#pragma once

#include <memory>

#include <cufft.h>
#include <cufftXt.h>

namespace holovibes::cuda_tools
{
/*! \class CufftHandle
 *
 *  \brief RAII wrapper for CUDA cufftHandle.
 */
class CufftHandle
{
  public:
    /*! \brief Default constructor */
    CufftHandle();

    /*! \brief Setter for the current stream used for cufft API.
     *  \param[in] stream The new stream.
     */
    static void set_stream(const cudaStream_t& stream);

    /*! \brief Constructor calling plan2d function.
     *
     *  Be careful, the x and y are inversed in cuda since the cufft API uses col-major.
     *  Hence use the height for the x param and width for the y param.
     *  For more information go check the following cufft documentation (nx and ny params):
     *  https://docs.nvidia.com/cuda/cufft/#cufftplan2d
     *
     *  \param[in] x The height of the plan.
     *  \param[in] y The width of the plan.
     *  \param[in] type The type of the cufft plan.
     */
    CufftHandle(const int x, const int y, const cufftType type);

    /*! \brief Destroy the created plan (if any) */
    ~CufftHandle();

    /*! \brief Destroy the created plan (if any) */
    void reset();

    /*! \brief Calls cufftPlan2d to get a 2d plan.
     *
     *  Could be overloaded for plan1d and plan3d.
     *  Be careful, the x and y are inversed in cuda since the cufft API uses col-major.
     *  Hence use the height for the x param and width for the y param.
     *  For more information go check the following cufft documentation (nx and ny params):
     *  https://docs.nvidia.com/cuda/cufft/#cufftplan2d
     *
     *  \param[in] x The height of the plan.
     *  \param[in] y The width of the plan.
     *  \param[in] type The transform data type.
     */
    void plan(const int x, const int y, const cufftType type);

    /*! \brief Calls cufftPlanMany.
     *
     *  \param[in] rank Dimensionality of the transform.
     *  \param[in] n Array of size `rank`, describing the size of each dimension, `n[0]` being the size of the outermost
     *  and `n[rank-1]` innermost (contiguous) dimension of a transform.
     *  \param[in] inembed Pointer of size `rank` that indicates the storage dimensions of the input data in memory.
     *  If set to NULL all other advanced data layout parameters are ignored.
     *  \param[in] istride Indicates the distance between two successive input elements in the least significant
     *  (i.e., innermost) dimension.
     *  \param[in] idist Indicates the distance between the first element of two consecutive signals in a batch of the
     *  input data.
     *  \param[in] onembed Pointer of size `rank` that indicates the storage dimensions of the output data in memory. If
     *  set to NULL all other advanced data layout parameters are ignored.
     *  \param[in] ostride Indicates the distance between two successive output elements in the output array in the
     *  least significant (i.e., innermost) dimension.
     *  \param[in] odist Indicates the distance between the first element of two consecutive signals in a batch of the
     *  output data.
     *  \param[in] type The transform data type.
     *  \param[in] batch Batch size for this transform.
     */
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

    /*! \brief Calls cufftXtMakePlanMany.
     *
     *  Warning: Allocating a plan has a VRAM memory cost.
     *  You can check the amount of memory used by checking the ws (working size) variable.
     *
     *  The memory taken by the plan is used by cufft to make the FFT.
     *  It's equal in byte to : frame_width * frame_height * sizeof(T) (in our
     *  case cuComplex) * batch_size.
     *
     *  \param[in] rank Dimensionality of the transform.
     *  \param[in] n Array of size `rank`, describing the size of each dimension, `n[0]` being the size of the outermost
     *  and `n[rank-1]` innermost (contiguous) dimension of a transform.
     *  \param[in] inembed Pointer of size `rank` that indicates the storage dimensions of the input data in memory.
     *  If set to NULL all other advanced data layout parameters are ignored.
     *  \param[in] istride Indicates the distance between two successive input elements in the least significant
     *  (i.e., innermost) dimension.
     *  \param[in] idist Indicates the distance between the first element of two consecutive signals in a batch of the
     *  input data.
     *  \param inputtype Type of input data.
     *  \param[in] onembed Pointer of size `rank` that indicates the storage dimensions of the output data in memory. If
     *  set to NULL all other advanced data layout parameters are ignored.
     *  \param[in] ostride Indicates the distance between two successive output elements in the output array in the
     *  least significant (i.e., innermost) dimension.
     *  \param[in] odist Indicates the distance between the first element of two consecutive signals in a batch of the
     *  output data.
     *  \param[in] outputtype Type of output data.
     *  \param[in] batch Batch size for this transform.
     *  \param[in] executiontype Type of data to be used for computations.
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

    /*! \brief Get a reference to the underlying cufftHandle.
     *  \return cufftHandle& A reference to the cufftHandle instance.
     */
    cufftHandle& get();

    /*! \brief Implicit cast to the underlying cufftHandle. */
    operator cufftHandle&();

    /*! \brief Implicit cast to a ptr to the underlying cufftHandle. */
    operator cufftHandle*();

  private:
    /*! \brief The cufftHandle.
     *  We chose a unique_ptr to represent a possibly uninitialized one.
     */
    std::unique_ptr<cufftHandle> val_;

    /*! \brief The CUDA stream used for the cufft API. */
    static cudaStream_t stream_;
};
} // namespace holovibes::cuda_tools
