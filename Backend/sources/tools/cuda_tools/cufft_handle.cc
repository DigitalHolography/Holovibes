#include "cufft_handle.hh"

#include "common.cuh"

using holovibes::cuda_tools::CufftHandle;

cudaStream_t CufftHandle::stream_;

CufftHandle::CufftHandle() {}

CufftHandle::CufftHandle(const int x, const int y, const cufftType type) { plan(x, y, type); }

CufftHandle::~CufftHandle() { reset(); }

void CufftHandle::reset()
{
    if (val_)
        cufftSafeCall(cufftDestroy(*val_.get()));
    val_.reset();
}

void CufftHandle::plan(const int x, const int y, const cufftType type)
{
    reset();
    val_.reset(new cufftHandle);
    cufftSafeCall(cufftPlan2d(val_.get(), x, y, type));
    cufftSafeCall(cufftSetStream(*val_.get(), stream_));
}

void CufftHandle::planMany(int rank,
                           int* n,
                           int* inembed,
                           int istride,
                           int idist,
                           int* onembed,
                           int ostride,
                           int odist,
                           cufftType type,
                           int batch)
{
    reset();
    val_.reset(new cufftHandle);
    cufftSafeCall(cufftPlanMany(val_.get(), rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch));
    cufftSafeCall(cufftSetStream(*val_.get(), stream_));
}

void CufftHandle::XtplanMany(int rank,
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
                             cudaDataType executionType)
{
    reset();
    val_.reset(new cufftHandle);

    cufftSafeCall(cufftCreate(val_.get()));

    size_t ws;
    cufftSafeCall(cufftXtGetSizeMany(*val_.get(),
                                     rank,
                                     n,
                                     inembed,
                                     istride,
                                     idist,
                                     inputtype,
                                     onembed,
                                     ostride,
                                     odist,
                                     outputtype,
                                     batch,
                                     &ws,
                                     executionType));
    cufftSafeCall(cufftXtMakePlanMany(*val_.get(),
                                      rank,
                                      n,
                                      inembed,
                                      istride,
                                      idist,
                                      inputtype,
                                      onembed,
                                      ostride,
                                      odist,
                                      outputtype,
                                      batch,
                                      &ws,
                                      executionType));
    cufftSafeCall(cufftSetStream(*val_.get(), stream_));
}

cufftHandle& CufftHandle::get() { return *val_; }

CufftHandle::operator cufftHandle&() { return get(); }

CufftHandle::operator cufftHandle*() { return val_.get(); }

void CufftHandle::set_stream(const cudaStream_t& stream) { stream_ = stream; }
