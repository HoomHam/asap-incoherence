/* test_native.c — native validation of nufft.c against the golden trajectory.
 * Reads workspace/helpers/golden_rad.f64 (M*3 doubles: wx[],wy[],wz[] blocks),
 * computes full-set and block-0 PSFs at n=64, writes split-complex raw floats.
 * Compared against finufft reference by workspace/helpers/compare_native.py. */

#include <stdio.h>
#include <stdlib.h>

extern int nufft_psf(const double *wx, const double *wy, const double *wz,
                     int M, int n);
extern float *nufft_get_psf_re(void);
extern float *nufft_get_psf_im(void);

static void dump(const char *path, int n)
{
    FILE *f = fopen(path, "wb");
    size_t Nv = (size_t)n * n * n;
    fwrite(nufft_get_psf_re(), sizeof(float), Nv, f);
    fwrite(nufft_get_psf_im(), sizeof(float), Nv, f);
    fclose(f);
}

int main(int argc, char **argv)
{
    const char *in = argc > 1 ? argv[1] : "golden_rad.f64";
    FILE *f = fopen(in, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", in); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int M = (int)(sz / sizeof(double) / 3);
    double *wx = malloc(M * sizeof(double));
    double *wy = malloc(M * sizeof(double));
    double *wz = malloc(M * sizeof(double));
    fread(wx, sizeof(double), M, f);
    fread(wy, sizeof(double), M, f);
    fread(wz, sizeof(double), M, f);
    fclose(f);
    printf("M = %d samples\n", M);

    int n = argc > 2 ? atoi(argv[2]) : 64;
    if (!nufft_psf(wx, wy, wz, M, n)) { fprintf(stderr, "psf full failed\n"); return 2; }
    dump("psf_full_native.raw", n);
    printf("full-set PSF done\n");

    int Mb = 26 * (M / 832);           /* block 0 = first 26 interleaves */
    if (!nufft_psf(wx, wy, wz, Mb, n)) { fprintf(stderr, "psf b0 failed\n"); return 3; }
    dump("psf_b0_native.raw", n);
    printf("block-0 PSF done (%d samples)\n", Mb);
    return 0;
}
