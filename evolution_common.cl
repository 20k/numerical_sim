#ifndef EVOLUTION_COMMON_CL
#define EVOLUTION_COMMON_CL

///common include for bssn and hydrodynamics

#define IS_DEGENERATE(x) (isnan(x) || !isfinite(x))

#define NANCHECK_IMPL(w) if(IS_DEGENERATE(w[index])){printf("NAN " #w " %i %i %i %f\n", ix, iy, iz, w[index]);}
#define LNANCHECK_IMPL(w)  if(IS_DEGENERATE(w)){printf("NAN " #w " %i %i %i %f\n", ix, iy, iz, w);}
#define NNANCHECK_IMPL(w, name) if(IS_DEGENERATE(w)){printf("NAN " name " %i %i %i %f\n", ix, iy, iz, w);}

//#define DEBUGGING
#ifdef DEBUGGING
#define NANCHECK(w) NANCHECK_IMPL(w)
#define LNANCHECK(w) LNANCHECK_IMPL(w)
#define NNANCHECK(w, name) NNANCHECK_IMPL(w, name)
#else
#define NANCHECK(w)
#define LNANCHECK(w)
#define NNANCHECK(w, name)
#endif

/*#define GET_ARGLIST(a, p) a p##cY0, a p##cY1, a p##cY2, a p##cY3, a p##cY4, a p##cY5, \
                a p##cA0, a p##cA1, a p##cA2, a p##cA3, a p##cA4, a p##cA5, \
                a p##cGi0, a p##cGi1, a p##cGi2, a p##K, a p##X, a p##gA, a p##gB0, a p##gB1, a p##gB2, \
                a p##gBB0, a p##gBB1, a p##gBB2, \
                a p##Dp_star, a p##De_star, a p##DcS0, a p##DcS1, a p##DcS2, \
                a p##dRed, a p##dGreen, a p##dBlue*/

///opencl has finally beaten me. Contains only the macro GET_ARGLIST
#include "generated_arglist.cl"

#define GET_DERIVLIST(a, p) a p##dcYij0, a p##dcYij1, a p##dcYij2, a p##dcYij3, a p##dcYij4, a p##dcYij5, a p##dcYij6, a p##dcYij7, a p##dcYij8, a p##dcYij9, a p##dcYij10, a p##dcYij11, a p##dcYij12, a p##dcYij13, a p##dcYij14, a p##dcYij15, a p##dcYij16, a p##dcYij17, \
                    a p##digA0, a p##digA1, a p##digA2, \
                    a p##digB0, a p##digB1, a p##digB2, a p##digB3, a p##digB4, a p##digB5, a p##digB6, a p##digB7, a p##digB8, \
                    a p##dX0, a p##dX1, a p##dX2

#define STANDARD_ARGS(p) GET_ARGLIST(__global float*, p)
#define STANDARD_DERIVS(p) GET_DERIVLIST(__global DERIV_PRECISION*, p)

#define ALL_ARGS(p) GET_ARGLIST(, p), GET_DERIVLIST(, p)
#define GET_STANDARD_ARGS(p) GET_ARGLIST(, p)

#endif // EVOLUTION_COMMON_CL
