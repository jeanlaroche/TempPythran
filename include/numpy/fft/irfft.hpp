#ifndef PYTHONIC_INCLUDE_NUMPY_FFT_IRFFT_HPP
#define PYTHONIC_INCLUDE_NUMPY_FFT_IRFFT_HPP

#include "pythonic/include/utils/functor.hpp"
#include "pythonic/include/types/ndarray.hpp"

PYTHONIC_NS_BEGIN

namespace numpy
{
	namespace fft
	{	
	
        template<class T, size_t N>
        types::ndarray<double, N> irfft(types::ndarray<T, N> const &, long, long, bool);

        template<class T, size_t N>
        types::ndarray<double, N> irfft(types::ndarray<T, N> const &, long, long);

        template<class T, size_t N>
        types::ndarray<double, N> irfft(types::ndarray<T, N> const &, long);

        template <class T, size_t N>
        types::ndarray<double, N> irfft(types::ndarray<T, N> const &);
        
        NUMPY_EXPR_TO_NDARRAY0_DECL(irfft);
		DECLARE_FUNCTOR(pythonic::numpy::fft, irfft);
	}
}
PYTHONIC_NS_END

#endif
