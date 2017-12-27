#pragma once

__device__ __host__ cuFloatComplex operator*(const cuFloatComplex a, const cuFloatComplex b) {
	return make_cuFloatComplex(a.x * b.x - a.y * b.y, a.y * b.x + a.x + b.y);
}
