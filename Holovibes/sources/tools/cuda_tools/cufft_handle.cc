/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#include "cufft_handle.hh"

#include "Common.cuh"

using holovibes::cuda_tools::CufftHandle;

CufftHandle::CufftHandle() : ws_(nullptr)
{}

CufftHandle::CufftHandle(int x, int y, cufftType type) : ws_(nullptr)
{
	plan(x, y, type);
}

CufftHandle::~CufftHandle()
{
	reset();
}

void CufftHandle::reset()
{
	if (val_)
		cufftSafeCall(cufftDestroy(*val_.get()));
    if (ws_ != nullptr)
    {
        delete ws_;
        ws_ = nullptr;
    }
	val_.reset();
}

void CufftHandle::plan(int x, int y, cufftType type)
{
	reset();
	val_.reset(new cufftHandle);
	cufftSafeCall(cufftPlan2d(val_.get(), x, y, type));
}

void CufftHandle::planMany(int rank,
	int *n,
	int *inembed, int istride, int idist,
	int *onembed, int ostride, int odist,
	cufftType type,
	int batch)
{
	reset();
	val_.reset(new cufftHandle);
    cufftSafeCall(cufftPlanMany(val_.get(), rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch));
}

void CufftHandle::XtplanMany(int rank,
    long long *n,
    long long *inembed, long long istride, long long idist, cudaDataType inputtype,
    long long *onembed, long long ostride, long long odist, cudaDataType outputtype,
    long long batch,
    cudaDataType executionType)
{
    reset();
	val_.reset(new cufftHandle);

	cufftSafeCall(cufftCreate(val_.get()));

    // Init the work size only once to help cufft
    if (ws_ == nullptr)
    {
        ws_ = new size_t;
        cufftSafeCall(cufftXtGetSizeMany(*val_.get(), rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist, outputtype, batch, ws_, executionType));
    }
    cufftSafeCall(cufftXtMakePlanMany(*val_.get(), rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist, outputtype, batch, ws_, executionType));
}

cufftHandle &CufftHandle::get()
{
	return *val_;
}

CufftHandle::operator cufftHandle&()
{
	return get();
}

CufftHandle::operator cufftHandle*()
{
	return val_.get();
}
