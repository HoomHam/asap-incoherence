/* nufft.c — gridding NUFFT + PSF pipeline for the incoherence explorer.
 *
 * Faithful port of workspace/helpers/gridding_proto.py (validated against
 * finufft eps=1e-9 to 4e-4 max-rel on the golden PSFs):
 *   oversampling sigma = 2, Kaiser-Bessel width W = 4, Beatty beta
 *   adjoint  f(x) = sum_j c_j exp(-i w_j.x)   (isign -1, x centered)
 *   forward  c_j  = sum_x f(x) exp(+i w_j.x)  (isign +1)
 *   DCF: dens = |A(A^H(1))|, w = 1/clip(dens, max*1e-4), w /= mean
 *   PSF: A^H( DCF * A(delta) ), complex-normalized by center value
 *
 * Grids are float32 interleaved complex; kernel/deapod tables double.
 * FFT: iterative radix-2, power-of-two sizes only => image n in {32,64,128}.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#define SIGMA 2
#define KW 4          /* kernel width, oversampled grid units */
#define KTAB 4096     /* kernel lookup entries over [0, KW/2] */

static double kb_beta(void)
{
    double w = KW, s = SIGMA;
    return M_PI * sqrt((w / s) * (w / s) * (s - 0.5) * (s - 0.5) - 0.8);
}

static double bessi0(double x)
{
    /* series: converges fast for |x| <= ~15 (beta = 8.9962 here) */
    double sum = 1.0, term = 1.0, hx = 0.5 * x;
    for (int k = 1; k < 64; k++) {
        term *= (hx / k) * (hx / k);
        sum += term;
        if (term < 1e-18 * sum) break;
    }
    return sum;
}

static double *ktab = NULL;   /* KB kernel over u in [0, KW/2] */
static double BETA, I0BETA;

static void kb_init(void)
{
    if (ktab) return;
    BETA = kb_beta();
    I0BETA = bessi0(BETA);
    ktab = malloc((KTAB + 1) * sizeof(double));
    for (int i = 0; i <= KTAB; i++) {
        double u = (double)i / KTAB * (KW / 2.0);
        double x = 1.0 - (2.0 * u / KW) * (2.0 * u / KW);
        ktab[i] = x > 0 ? bessi0(BETA * sqrt(x)) / I0BETA : 0.0;
    }
}

static inline double kb(double u)
{
    double t = fabs(u) / (KW / 2.0) * KTAB;
    int i = (int)t;
    if (i >= KTAB) return 0.0;
    double f = t - i;
    return ktab[i] * (1.0 - f) + ktab[i + 1] * f;
}

/* analytic FT of the KB kernel at image coord x (units of oversampled FOV) */
static double kb_ft(double x)
{
    double t = (M_PI * KW * x) * (M_PI * KW * x) - BETA * BETA;
    double v;
    if (t < 0) {
        double sq = sqrt(-t);
        v = sinh(sq) / sq;
    } else {
        double sq = sqrt(t);
        v = (sq < 1e-12) ? 1.0 : sin(sq) / sq;
    }
    return v / sinh(BETA) * BETA;
}

/* ------------------------------------------------------------------ FFT
 * In-place iterative radix-2 complex-float FFT along strided lines. */

static void fft1d(float *re, float *im, int n, size_t stride, int isign)
{
    /* bit-reversal permutation */
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            size_t a = i * stride, b = j * stride;
            float tr = re[a]; re[a] = re[b]; re[b] = tr;
            float ti = im[a]; im[a] = im[b]; im[b] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = isign * 2.0 * M_PI / len;
        double wr = cos(ang), wi = sin(ang);
        for (int i = 0; i < n; i += len) {
            double cwr = 1.0, cwi = 0.0;
            for (int k = 0; k < len / 2; k++) {
                size_t a = (i + k) * stride, b = (i + k + len / 2) * stride;
                double ur = re[a], ui = im[a];
                double vr = re[b] * cwr - im[b] * cwi;
                double vi = re[b] * cwi + im[b] * cwr;
                re[a] = (float)(ur + vr); im[a] = (float)(ui + vi);
                re[b] = (float)(ur - vr); im[b] = (float)(ui - vi);
                double nwr = cwr * wr - cwi * wi;
                cwi = cwr * wi + cwi * wr;
                cwr = nwr;
            }
        }
    }
}

/* 3D FFT on split re/im arrays of size G^3, C order (z fastest = index x?):
 * we use linear index l = (ix*G + iy)*G + iz. Transform each dim. */
static void fft3d(float *re, float *im, int G, int isign)
{
    size_t G2 = (size_t)G * G;
    /* dim 2 (iz, stride 1): G^2 contiguous lines */
    for (size_t l = 0; l < G2 * G; l += G)
        fft1d(re + l, im + l, G, 1, isign);
    /* dim 1 (iy, stride G) */
    for (int ix = 0; ix < G; ix++)
        for (int iz = 0; iz < G; iz++)
            fft1d(re + (size_t)ix * G2 + iz, im + (size_t)ix * G2 + iz, G, G, isign);
    /* dim 0 (ix, stride G^2) */
    for (int iy = 0; iy < G; iy++)
        for (int iz = 0; iz < G; iz++)
            fft1d(re + (size_t)iy * G + iz, im + (size_t)iy * G + iz, G, G2, isign);
}

/* ------------------------------------------------------------- spreading */

/* Spread M complex samples at radian coords (wx,wy,wz) onto G^3 grid. */
static void spread(const double *wx, const double *wy, const double *wz,
                   const float *cre, const float *cim, int M,
                   float *gre, float *gim, int G)
{
    double scale = G / (2.0 * M_PI);
    memset(gre, 0, (size_t)G * G * G * sizeof(float));
    memset(gim, 0, (size_t)G * G * G * sizeof(float));
    size_t G2 = (size_t)G * G;

    for (int j = 0; j < M; j++) {
        double gx = wx[j] * scale, gy = wy[j] * scale, gz = wz[j] * scale;
        int bx = (int)floor(gx), by = (int)floor(gy), bz = (int)floor(gz);
        double kx[KW], ky[KW], kz[KW];
        int ixs[KW], iys[KW], izs[KW];
        for (int d = 0; d < KW; d++) {
            int o = d - KW / 2 + 1;
            kx[d] = kb(bx + o - gx);
            ky[d] = kb(by + o - gy);
            kz[d] = kb(bz + o - gz);
            ixs[d] = ((bx + o) % G + G) % G;
            iys[d] = ((by + o) % G + G) % G;
            izs[d] = ((bz + o) % G + G) % G;
        }
        double re = cre[j], im = cim[j];
        for (int a = 0; a < KW; a++) {
            double wxa = kx[a];
            size_t offx = (size_t)ixs[a] * G2;
            for (int b = 0; b < KW; b++) {
                double wab = wxa * ky[b];
                size_t offxy = offx + (size_t)iys[b] * G;
                for (int c = 0; c < KW; c++) {
                    double w = wab * kz[c];
                    size_t l = offxy + izs[c];
                    gre[l] += (float)(re * w);
                    gim[l] += (float)(im * w);
                }
            }
        }
    }
}

/* Interpolate G^3 grid at M radian coords -> complex samples. */
static void interp(const float *gre, const float *gim, int G,
                   const double *wx, const double *wy, const double *wz, int M,
                   float *cre, float *cim)
{
    double scale = G / (2.0 * M_PI);
    size_t G2 = (size_t)G * G;
    for (int j = 0; j < M; j++) {
        double gx = wx[j] * scale, gy = wy[j] * scale, gz = wz[j] * scale;
        int bx = (int)floor(gx), by = (int)floor(gy), bz = (int)floor(gz);
        double kx[KW], ky[KW], kz[KW];
        int ixs[KW], iys[KW], izs[KW];
        for (int d = 0; d < KW; d++) {
            int o = d - KW / 2 + 1;
            kx[d] = kb(bx + o - gx);
            ky[d] = kb(by + o - gy);
            kz[d] = kb(bz + o - gz);
            ixs[d] = ((bx + o) % G + G) % G;
            iys[d] = ((by + o) % G + G) % G;
            izs[d] = ((bz + o) % G + G) % G;
        }
        double sre = 0, sim = 0;
        for (int a = 0; a < KW; a++) {
            size_t offx = (size_t)ixs[a] * G2;
            for (int b = 0; b < KW; b++) {
                double wab = kx[a] * ky[b];
                size_t offxy = offx + (size_t)iys[b] * G;
                for (int c = 0; c < KW; c++) {
                    double w = wab * kz[c];
                    size_t l = offxy + izs[c];
                    sre += gre[l] * w;
                    sim += gim[l] * w;
                }
            }
        }
        cre[j] = (float)sre;
        cim[j] = (float)sim;
    }
}

/* --------------------------------------------------------- adjoint/forward */

static double *deapod = NULL;
static int deapod_n = 0;

static void deapod_init(int n, int G)
{
    if (deapod_n == n) return;
    free(deapod);
    deapod = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        deapod[i] = kb_ft((double)(i - n / 2) / G);
    deapod_n = n;
}

/* A^H: samples -> centered n^3 image (split re/im). Grid scratch provided. */
static void adjoint_op(const double *wx, const double *wy, const double *wz,
                       const float *cre, const float *cim, int M, int n,
                       float *ire, float *iim, float *gre, float *gim)
{
    int G = SIGMA * n;
    size_t G2 = (size_t)G * G;
    kb_init();
    deapod_init(n, G);
    spread(wx, wy, wz, cre, cim, M, gre, gim, G);
    fft3d(gre, gim, G, -1);           /* isign -1: numpy forward fftn */
    for (int x = 0; x < n; x++) {
        int fx = ((x - n / 2) % G + G) % G;
        for (int y = 0; y < n; y++) {
            int fy = ((y - n / 2) % G + G) % G;
            for (int z = 0; z < n; z++) {
                int fz = ((z - n / 2) % G + G) % G;
                size_t src = (size_t)fx * G2 + (size_t)fy * G + fz;
                size_t dst = ((size_t)x * n + y) * n + z;
                double dp = deapod[x] * deapod[y] * deapod[z];
                ire[dst] = (float)(gre[src] / dp);
                iim[dst] = (float)(gim[src] / dp);
            }
        }
    }
}

/* A: centered n^3 image -> samples. */
static void forward_op(const float *ire, const float *iim, int n,
                       const double *wx, const double *wy, const double *wz, int M,
                       float *cre, float *cim, float *gre, float *gim)
{
    int G = SIGMA * n;
    size_t G2 = (size_t)G * G;
    kb_init();
    deapod_init(n, G);
    memset(gre, 0, (size_t)G * G * G * sizeof(float));
    memset(gim, 0, (size_t)G * G * G * sizeof(float));
    for (int x = 0; x < n; x++) {
        int fx = ((x - n / 2) % G + G) % G;
        for (int y = 0; y < n; y++) {
            int fy = ((y - n / 2) % G + G) % G;
            for (int z = 0; z < n; z++) {
                int fz = ((z - n / 2) % G + G) % G;
                size_t dst = (size_t)fx * G2 + (size_t)fy * G + fz;
                size_t src = ((size_t)x * n + y) * n + z;
                double dp = deapod[x] * deapod[y] * deapod[z];
                gre[dst] = (float)(ire[src] / dp);
                gim[dst] = (float)(iim[src] / dp);
            }
        }
    }
    fft3d(gre, gim, G, +1);           /* isign +1: unnormalized inverse */
    interp(gre, gim, G, wx, wy, wz, M, cre, cim);
}

/* ------------------------------------------------------------ public API */

static float *psf_re = NULL, *psf_im = NULL;   /* last computed PSF, n^3 */
static float *dcf_w = NULL;                    /* last computed DCF */
static int last_n = 0, last_M = 0;

EMSCRIPTEN_KEEPALIVE float *nufft_get_psf_re(void) { return psf_re; }
EMSCRIPTEN_KEEPALIVE float *nufft_get_psf_im(void) { return psf_im; }
EMSCRIPTEN_KEEPALIVE float *nufft_get_dcf(void)    { return dcf_w; }

/* Compute PSF of the sample set given radian coords (wx,wy,wz), image size n.
 * Full reference pipeline: DCF then A^H(DCF * A(delta)), center-normalized.
 * Returns 1 on success, 0 on alloc failure / bad n. */
EMSCRIPTEN_KEEPALIVE
int nufft_psf(const double *wx, const double *wy, const double *wz,
              int M, int n)
{
    if (n < 8 || (n & (n - 1)) != 0) return 0;   /* power of two only */
    int G = SIGMA * n;
    size_t Nv = (size_t)n * n * n, Gv = (size_t)G * G * G;

    free(psf_re); free(psf_im); free(dcf_w);
    psf_re = malloc(Nv * sizeof(float));
    psf_im = malloc(Nv * sizeof(float));
    dcf_w  = malloc((size_t)M * sizeof(float));
    float *gre = malloc(Gv * sizeof(float));
    float *gim = malloc(Gv * sizeof(float));
    float *cre = malloc((size_t)M * sizeof(float));
    float *cim = malloc((size_t)M * sizeof(float));
    float *ire = malloc(Nv * sizeof(float));
    float *iim = malloc(Nv * sizeof(float));
    if (!psf_re || !psf_im || !dcf_w || !gre || !gim || !cre || !cim || !ire || !iim)
        goto fail;

    /* DCF: dens = |A(A^H(ones))| */
    for (int j = 0; j < M; j++) { cre[j] = 1.0f; cim[j] = 0.0f; }
    adjoint_op(wx, wy, wz, cre, cim, M, n, ire, iim, gre, gim);
    forward_op(ire, iim, n, wx, wy, wz, M, cre, cim, gre, gim);
    double dmax = 0;
    for (int j = 0; j < M; j++) {
        double d = sqrt((double)cre[j] * cre[j] + (double)cim[j] * cim[j]);
        dcf_w[j] = (float)d;
        if (d > dmax) dmax = d;
    }
    double lo = dmax * 1e-4, wsum = 0;
    for (int j = 0; j < M; j++) {
        double d = dcf_w[j] < lo ? lo : dcf_w[j];
        dcf_w[j] = (float)(1.0 / d);
        wsum += dcf_w[j];
    }
    double wmean = wsum / M;
    for (int j = 0; j < M; j++) dcf_w[j] = (float)(dcf_w[j] / wmean);

    /* PSF: A(delta) -> * w -> A^H */
    memset(ire, 0, Nv * sizeof(float));
    memset(iim, 0, Nv * sizeof(float));
    ire[(((size_t)n / 2) * n + n / 2) * n + n / 2] = 1.0f;
    forward_op(ire, iim, n, wx, wy, wz, M, cre, cim, gre, gim);
    for (int j = 0; j < M; j++) { cre[j] *= dcf_w[j]; cim[j] *= dcf_w[j]; }
    adjoint_op(wx, wy, wz, cre, cim, M, n, psf_re, psf_im, gre, gim);

    /* complex normalize by center value */
    size_t cidx = (((size_t)n / 2) * n + n / 2) * n + n / 2;
    double cr = psf_re[cidx], ci = psf_im[cidx];
    double c2 = cr * cr + ci * ci;
    for (size_t l = 0; l < Nv; l++) {
        double r = psf_re[l], i = psf_im[l];
        psf_re[l] = (float)((r * cr + i * ci) / c2);
        psf_im[l] = (float)((i * cr - r * ci) / c2);
    }

    last_n = n; last_M = M;
    free(gre); free(gim); free(cre); free(cim); free(ire); free(iim);
    return 1;
fail:
    free(gre); free(gim); free(cre); free(cim); free(ire); free(iim);
    return 0;
}

/* malloc/free helpers for the JS side */
EMSCRIPTEN_KEEPALIVE void  *wasm_malloc(int nbytes) { return malloc((size_t)nbytes); }
EMSCRIPTEN_KEEPALIVE void   wasm_free(void *p)      { free(p); }
