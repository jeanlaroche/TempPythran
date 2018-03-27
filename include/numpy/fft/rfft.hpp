#ifndef PYTHONIC_INCLUDE_NUMPY_VAR_HPP
#define PYTHONIC_INCLUDE_NUMPY_VAR_HPP

#include "pythonic/include/utils/functor.hpp"
#include "pythonic/include/types/ndarray.hpp"
#include "pythonic/include/__builtin__/None.hpp"
#include "pythonic/include/numpy/add.hpp"
#include "pythonic/include/numpy/mean.hpp"
#include "pythonic/include/numpy/reshape.hpp"
#include "pythonic/include/numpy/sum.hpp"

#include <algorithm>

PYTHONIC_NS_BEGIN

namespace numpy
{
	namespace fft
	{		
// 		template <class T>
// 		types::numpy_texpr<types::ndarray<T, 2>> rfft(types::ndarray<T, 2> const &arr);

		template<class T, size_t N> 
		types::ndarray<std::complex<typename std::common_type<T, double>::type>, N> rfft(types::ndarray<T, N> const &, long);

		template <class T, size_t N>
		types::ndarray<std::complex<typename std::common_type<T, double>::type>, N> rfft(types::ndarray<T, N> const &);

		// JEAN: This is taken from transpose, but I don't understand what it does / why it's needed
// 		NUMPY_EXPR_TO_NDARRAY0_DECL(rfft);
		DECLARE_FUNCTOR(pythonic::numpy::fft, rfft);
	}
}
PYTHONIC_NS_END

#endif
