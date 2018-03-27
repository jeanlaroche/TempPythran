#ifndef PYTHONIC_NUMPY_VAR_HPP
#define PYTHONIC_NUMPY_VAR_HPP

#include "pythonic/include/numpy/fft/rfft.hpp"
//#include "pythonic/include/numpy/fft/fftpack.h"

#include "pythonic/utils/functor.hpp"
#include "pythonic/include/utils/array_helper.hpp"
#include <array>
#include "pythonic/types/ndarray.hpp"
#include "pythonic/__builtin__/None.hpp"
#include "pythonic/__builtin__/ValueError.hpp"
#include "pythonic/numpy/reshape.hpp"
#include "pythonic/numpy/empty_like.hpp"

#include <algorithm>
#include <stdio.h>

#include "pythonic/numpy/fft/fftpack.c"


PYTHONIC_NS_BEGIN

namespace numpy
{
// 	template <class T>
// 	types::numpy_texpr<types::ndarray<T, 2>> rfft(types::ndarray<T, 2> const &arr)
// 	{
// 		return {arr};
// 	}
	namespace fft
	{
      template<class T, size_t N> 
	  types::ndarray<std::complex<typename std::common_type<T, double>::type>, N> rfft(types::ndarray<T, N> const &a, long M)
	  {
	  	  auto shape = a.shape();
	  	  int i;
	  	  printf("Shape %d N = %d -- Flat size: %d\n",shape,N,a.flat_size());
	  	  
	  	  // Create output array.
	      types::ndarray<std::complex<typename std::common_type<T, double>::type>, N> out_array(shape,__builtin__::None);
	      
	      // Create the twiddle factors
	      // This is from fftpack_litemodule.c
	      types::array<long, 1> twiddle_shape = {(long)(2*M+15)};
	      types::ndarray<T, 1> wsave(twiddle_shape,__builtin__::None);
	      // Call initialization of the factors
		  npy_rffti(M, (double *)wsave.buffer);
		  
	      // Call fft (fftpack.py) r = work_function(a, wsave)
// 			nrepeats = PyArray_SIZE(data)/npts;
// 			rptr = (double *)PyArray_DATA(ret);
// 			dptr = (double *)PyArray_DATA(data);
// 
// 			Py_BEGIN_ALLOW_THREADS;
// 			NPY_SIGINT_ON;
// 			for (i = 0; i < nrepeats; i++) {
// 				memcpy((char *)(rptr+1), dptr, npts*sizeof(double));
// 				npy_rfftf(npts, rptr+1, wsave);
// 				rptr[0] = rptr[1];
// 				rptr[1] = 0.0;
// 				rptr += rstep;
// 				dptr += npts;
// 			}

			double* rptr = (double*)out_array.buffer;
			double* dptr = (double*)a.buffer;
			
			long nrepeats = a.flat_size()/shape[0];
			long out_size = M/2+1;
			for (i = 0; i < nrepeats; i++) {
				memcpy((char *)(rptr+1), dptr, M*sizeof(double));
				npy_rfftf(M, rptr+1, (double*)wsave.buffer);
				rptr[0] = rptr[1];
				rptr[1] = 0.0;
				rptr[M+1] = 0.0;
				rptr += out_size;
				dptr += M;
			}
			out_array._shape[0] = out_size;
	      	
	      return out_array;
	  }

	  template <class T, size_t N>
	  types::ndarray<std::complex<typename std::common_type<T, double>::type>, N> rfft(types::ndarray<T, N> const &a)
	  {
	  	auto shape = a.shape();
	  	long M = shape[0];
		return rfft(a,M);
	  }


	  DEFINE_FUNCTOR(pythonic::numpy::fft, rfft);
  }
}
PYTHONIC_NS_END

#endif
