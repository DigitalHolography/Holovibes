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

using holovibes::cuda_tools::CufftHandle;


CufftHandle::CufftHandle(int x, int y, cufftType type)
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
		cufftDestroy(*val_);
	val_.reset();
}

cufftResult CufftHandle::plan(int x, int y, cufftType type)
{
	reset();
	val_.reset(new cufftHandle);
	return cufftPlan2d(val_.get(), x, y, type);
}

cufftResult CufftHandle::planMany(int rank,
	int *n,
	int *inembed, int istride, int idist,
	int *onembed, int ostride, int odist,
	cufftType type,
	int batch)
{
	reset();
	val_.reset(new cufftHandle);
	return cufftPlanMany(val_.get(), rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
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
