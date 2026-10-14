// Own transcendental functions (exp, log, exp2, tanh, pow and their float
// forms), so that the output does not depend on the system maths library.
// Integer-exact range reduction and fixed-order polynomials; this unit is
// compiled with -ffp-model=precise -ffp-contract=off.  The split of ln 2 into
// kLn2Hi/kLn2Lo and the decimal constants below are those of fdlibm (Sun
// Microsystems).
#include <stdint.h>
#include <string.h>

namespace {

const double kLn2Hi   = 6.93147180369123816490e-01;   /* 0x3FE62E42FEE00000 */
const double kLn2Lo   = 1.90821492927058770002e-10;
const double kLog2e   = 1.44269504088896338700e+00;
const double kLn2     = 6.93147180559945286227e-01;

inline double Scale2(double x, int k) {           /* x * 2^k, |k| < 1000 */
  if (k > 1023) { x *= 8.98846567431158e307; k -= 1023; if (k > 1023) return x * 8.98846567431158e307; }
  if (k < -1022) { x *= 2.2250738585072014e-308; k += 1022;
                   if (k < -1022) { x *= 2.2250738585072014e-308; k += 1022; if (k < -1022) k = -1022; } }
  uint64_t bits = (uint64_t)(k + 1023) << 52;
  double p; memcpy(&p, &bits, 8);
  return x * p;
}

/* e^x for |x| < 700, error < 1 ulp */
double ExpCore(double x) {
  if (x != x) return x;                              /* NaN */
  if (x > 709.78) return 1.7976931348623157e308 * 1.7976931348623157e308;
  if (x < -745.2) return 0.0;
  double t = x * kLog2e;
  double k = (t >= 0.0) ? (double)(int64_t)(t + 0.5) : (double)(int64_t)(t - 0.5);
  double r = (x - k * kLn2Hi) - k * kLn2Lo;          /* |r| <= 0.3466 */
  /* Taylor, Horner, fixed order; r^13/13! < 1e-17 */
  double p = 1.0 / 6227020800.0;
  p = p * r + 1.0 / 479001600.0;
  p = p * r + 1.0 / 39916800.0;
  p = p * r + 1.0 / 3628800.0;
  p = p * r + 1.0 / 362880.0;
  p = p * r + 1.0 / 40320.0;
  p = p * r + 1.0 / 5040.0;
  p = p * r + 1.0 / 720.0;
  p = p * r + 1.0 / 120.0;
  p = p * r + 1.0 / 24.0;
  p = p * r + 1.0 / 6.0;
  p = p * r + 0.5;
  p = p * r + 1.0;
  p = p * r + 1.0;
  return Scale2(p, (int)k);
}

/* ln(x), error < 1 ulp */
double LogCore(double x) {
  if (x != x) return x;
  if (x < 0.0) return (x - x) / 0.0;                 /* NaN */
  if (x == 0.0) return -1.0 / 0.0;
  uint64_t b; memcpy(&b, &x, 8);
  int e = (int)((b >> 52) & 0x7FF);
  if (e == 0) {                                      /* subnormal */
    x *= 9007199254740992.0; memcpy(&b, &x, 8);
    e = (int)((b >> 52) & 0x7FF) - 53;
  } else if (e == 0x7FF) {
    return x;                                        /* inf */
  }
  e -= 1023;
  b = (b & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
  double m; memcpy(&m, &b, 8);                       /* m in [1,2) */
  if (m > 1.4142135623730951) { m = m * 0.5; e += 1; }
  double s = (m - 1.0) / (m + 1.0);                  /* |s| <= 0.1716 */
  double z = s * s;
  double p = 1.0 / 19.0;
  p = p * z + 1.0 / 17.0;
  p = p * z + 1.0 / 15.0;
  p = p * z + 1.0 / 13.0;
  p = p * z + 1.0 / 11.0;
  p = p * z + 1.0 / 9.0;
  p = p * z + 1.0 / 7.0;
  p = p * z + 1.0 / 5.0;
  p = p * z + 1.0 / 3.0;
  p = p * z + 1.0;
  double lm = 2.0 * s * p;
  double de = (double)e;
  return (de * kLn2Hi + lm) + de * kLn2Lo;
}

double Exp2Core(double x) {
  if (x != x) return x;
  if (x > 1024.0) return 1.7976931348623157e308 * 1.7976931348623157e308;
  if (x < -1080.0) return 0.0;
  double k = (x >= 0.0) ? (double)(int64_t)(x + 0.5) : (double)(int64_t)(x - 0.5);
  double f = x - k;                                  /* |f| <= 0.5 */
  double r = f * kLn2;                               /* |r| <= 0.347 */
  double p = 1.0 / 6227020800.0;
  p = p * r + 1.0 / 479001600.0;
  p = p * r + 1.0 / 39916800.0;
  p = p * r + 1.0 / 3628800.0;
  p = p * r + 1.0 / 362880.0;
  p = p * r + 1.0 / 40320.0;
  p = p * r + 1.0 / 5040.0;
  p = p * r + 1.0 / 720.0;
  p = p * r + 1.0 / 120.0;
  p = p * r + 1.0 / 24.0;
  p = p * r + 1.0 / 6.0;
  p = p * r + 0.5;
  p = p * r + 1.0;
  p = p * r + 1.0;
  return Scale2(p, (int)k);
}

/* ---------------------------------------------------------------------------
 * Fast path for the three functions of the adaptive path (expf/exp2f/tanhf).
 * 2^x = T[j] * poly(r) * 2^m  with  n = round(32*x), m = n>>5, j = n&31,
 * r = x - n/32 (|r| <= 1/64).  T[] are exact double constants (2^(j/32)).
 * Order and rounding are thereby fully determined; no FMA, no
 * reassociation (the TU is compiled with -ffp-model=precise
 * -ffp-contract=off).  Error < 1 ulp in the float result.
 * ------------------------------------------------------------------------ */
const double kT32[32] = {
  1.00000000000000000e+00, 1.02189714865411663e+00, 1.04427378242741375e+00, 1.06714040067682370e+00,
  1.09050773266525769e+00, 1.11438674259589243e+00, 1.13878863475669156e+00, 1.16372485877757748e+00,
  1.18920711500272103e+00, 1.21524735998046896e+00, 1.24185781207348400e+00, 1.26905095719173322e+00,
  1.29683955465100964e+00, 1.32523664315974132e+00, 1.35425554693689265e+00, 1.38390988196383202e+00,
  1.41421356237309515e+00, 1.44518080697704665e+00, 1.47682614593949935e+00, 1.50916442759342284e+00,
  1.54221082540794074e+00, 1.57598084510788650e+00, 1.61049033194925428e+00, 1.64575547815396495e+00,
  1.68179283050742900e+00, 1.71861929812247793e+00, 1.75625216037329945e+00, 1.79470907500310717e+00,
  1.83400808640934243e+00, 1.87416763411029996e+00, 1.91520656139714740e+00, 1.95714412417540018e+00
};

inline double Exp2Fast(double x) {                 /* |x| < 1000 */
  double xn = x * 32.0;
  double nd = (xn >= 0.0) ? (double)(int32_t)(xn + 0.5) : (double)(int32_t)(xn - 0.5);
  int32_t n = (int32_t)nd;
  double r = (x - nd * 0.03125) * kLn2;            /* |r| <= ln2/64 */
  double p = 1.0 / 24.0;
  p = p * r + 1.0 / 6.0;
  p = p * r + 0.5;
  p = p * r + 1.0;
  p = p * r + 1.0;
  double t = kT32[n & 31] * p;
  return Scale2(t, n >> 5);
}

double TanhFast(double x) {
  double ax = x < 0.0 ? -x : x;
  if (ax < 1.0e-4) return x - (x * x) * x / 3.0;
  if (ax > 20.0) return x < 0.0 ? -1.0 : 1.0;
  double t = Exp2Fast((-2.0 * ax) * kLog2e);
  double r = (1.0 - t) / (1.0 + t);
  return x < 0.0 ? -r : r;
}

double TanhCore(double x) {
  double ax = x < 0.0 ? -x : x;
  if (ax < 1.0e-4) return x - (x * x) * x / 3.0;
  if (ax > 20.0) return x < 0.0 ? -1.0 : 1.0;
  double t = ExpCore(-2.0 * ax);
  double r = (1.0 - t) / (1.0 + t);
  return x < 0.0 ? -r : r;
}

}  /* namespace */

extern "C" {
double exp(double x)   { return ExpCore(x); }
double log(double x)   { return LogCore(x); }
float  expf(float x)   { double d=(double)x;
                         if (d > 88.8) return 3.4028235e38f*3.4028235e38f;
                         if (d < -104.0) return 0.0f;
                         return (float)Exp2Fast(d * kLog2e); }
float  logf(float x)   { return (float)LogCore((double)x); }
float  exp2f(float x)  { double d=(double)x;
                         if (d > 128.0) return 3.4028235e38f*3.4028235e38f;
                         if (d < -150.0) return 0.0f;
                         return (float)Exp2Fast(d); }
float  tanhf(float x)  { return (float)TanhFast((double)x); }
/* Double variants, in case the compiler emits calls to them */
double tanh(double x)  { return TanhCore(x); }
double exp2(double x)  { return Exp2Core(x); }
double pow(double b, double e) {                     /* last resort only */
  if (e == 0.0) return 1.0;
  if (b == 0.0) return 0.0;
  return ExpCore(e * LogCore(b));
}
float powf(float b, float e) { return (float)pow((double)b, (double)e); }
}
