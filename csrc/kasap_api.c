/* kasap_api.c — parameterized WASM entry for the ASAP trajectory generator.
 *
 * Includes pristine kasap.c (byte-identical, main() renamed away) and reuses
 * its helpers (rotarb, kr/krdot/krddot, calchedgehog, norm, cpyvec, dot).
 * makegrads() itself is re-implemented here as makegrads_p() because three
 * things the UI must control are hardcoded inside it: the per-repetition
 * rotation angle (M_PI/2), the slew/gradient limits (MAXS/MAXG), and the
 * source of the per-repetition rotation axis (Thomson hedgehog). The math is
 * otherwise a line-for-line copy.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#define main kasap_reference_main_unused
#include "kasap.c"
#undef main
#undef idx

#define IDX(ilv,point,em) ((ilv)*(size_t)NPTS*3 + (point)*3 + (em))

/* Line-for-line port of makegrads() with parameterized rotangle, maxs, maxg,
 * and rep-rotation axis. No static caching — caller manages basis lifetime. */
static void makegrads_p(float gx[], float gy[], float gz[],
                        float kx[], float ky[], float kz[],
                        float at, float fov, float t0, float ms, float gam,
                        float dt, float n, int NI, int NPTS, int irep,
                        float rotangle, float maxs, float maxg,
                        const float *repaxis, /* unit axis for this irep */
                        const float *basis,   /* NI x 3 interleave basis */
                        float *k, float *g)   /* NI*NPTS*3 scratch */
{
    float rotvec[3] = { 0.0f, 0.0f, 1.0f };
    float xhat[3]   = { 1.0f, 0.0f, 0.0f };

    for (int ir = 0, firstime = 1; ir < NPTS; ir++)
    {
        float t = ir * dt;
        float thisk = kr(t, t0, maxs * gam, ms, fov, at, n);

        if (thisk < ASMALLNUMBER)
        {
            for (int j = 0; j < NI; j++)
                for (int m = 0; m < 3; m++)
                    { g[IDX(j,ir,m)] = 0.0f; k[IDX(j,ir,m)] = 0.0f; }
            continue;
        }

        float thiskdot  = krdot (t, t0, maxs * gam, ms, fov, at, n);
        float thiskddot = krddot(t, t0, maxs * gam, ms, fov, at, n);

        float wG = sqrtf(SQ(maxg * gam) - SQ(thiskdot)) / thisk;

        float tmpA = 2.0f * SQ(maxs * gam * thisk)
                   + SQ(SQ(thiskdot))
                   - 2.0f * thisk * SQ(thiskdot) * thiskddot
                   - SQ(thisk * thiskdot);
        float wS = sqrtf(sqrtf(tmpA) + thisk * thiskddot - SQ(thiskdot)) / 1.41421356237f / thisk;

        float w = (wS < wG) ? wS : wG;

        rotarb(rotvec, w * dt, xhat);
        float rotangle_step = w * dt;

        for (int j = 0; j < NI; j++)
        {
            if (firstime) cpyvec(&k[IDX(j,ir,0)], &basis[3*j]);
            else          cpyvec(&k[IDX(j,ir,0)], &k[IDX(j,ir-1,0)]);

            norm(&k[IDX(j,ir,0)], thisk);
            rotarb(&k[IDX(j,ir,0)], rotangle_step, rotvec);

            for (int m = 0; m < 3; m++)
                g[IDX(j,ir,m)] = (ir == 0) ? 0.0f : (k[IDX(j,ir,m)] - k[IDX(j,ir-1,m)]) / dt / gam;
        }
        firstime = 0;
    }

    /* ramp down to g=0 */
    int ir = NPTS - 2;
    for (int j = 0; j < NI; j++)
        for (; sqrtf(dot(&g[IDX(j,ir,0)], &g[IDX(j,ir,0)])) > (NPTS - 1 - ir) * dt * maxs; ir--) { }

    for (int irp = ir + 1; irp < NPTS; irp++)
        for (int j = 0; j < NI; j++)
            for (int m = 0; m < 3; m++)
                g[IDX(j,irp,m)] = g[IDX(j,ir,m)] * (float)(NPTS - 1 - irp) / (float)(NPTS - 1 - ir);

    /* rotate by rotangle around the repetition axis (kasap: M_PI/2, hedgehog) */
    float ax[3] = { repaxis[0], repaxis[1], repaxis[2] };
    for (int ir2 = 0; ir2 < NPTS; ir2++)
        for (int j = 0; j < NI; j++)
        {
            rotarb(&g[IDX(j,ir2,0)], rotangle, ax);
            rotarb(&k[IDX(j,ir2,0)], rotangle, ax);
        }

    for (int ir3 = 0; ir3 < NPTS; ir3++)
        for (int j = 0; j < NI; j++)
        {
            size_t lin = ir3 + (size_t)j * NPTS;
            gx[lin] = g[IDX(j,ir3,0)];
            gy[lin] = g[IDX(j,ir3,1)];
            gz[lin] = g[IDX(j,ir3,2)];
            kx[lin] = k[IDX(j,ir3,0)];
            ky[lin] = k[IDX(j,ir3,1)];
            kz[lin] = k[IDX(j,ir3,2)];
        }
}

/* ----------------------------------------------------------------- state */

static float *out_kx, *out_ky, *out_kz, *out_gx, *out_gy, *out_gz;
static float *cur_basis, *cur_reprot;
static int cur_NI = 0, cur_NPTS = 0, cur_NREPS = 0;

EMSCRIPTEN_KEEPALIVE float *kasap_get_kx(void) { return out_kx; }
EMSCRIPTEN_KEEPALIVE float *kasap_get_ky(void) { return out_ky; }
EMSCRIPTEN_KEEPALIVE float *kasap_get_kz(void) { return out_kz; }
EMSCRIPTEN_KEEPALIVE float *kasap_get_gx(void) { return out_gx; }
EMSCRIPTEN_KEEPALIVE float *kasap_get_gy(void) { return out_gy; }
EMSCRIPTEN_KEEPALIVE float *kasap_get_gz(void) { return out_gz; }
EMSCRIPTEN_KEEPALIVE float *kasap_get_basis(void) { return cur_basis; }
EMSCRIPTEN_KEEPALIVE float *kasap_get_reprot(void) { return cur_reprot; }
EMSCRIPTEN_KEEPALIVE int    kasap_get_total(void) { return cur_NI * cur_NPTS * cur_NREPS; }

/* Generate the full trajectory set.
 *   NI, NPTS, NREPS : counts
 *   n, at, fov, ms  : radial shape power, readout dur (s), FOV (m), matrix
 *   optimize        : 1 = Thomson-optimize both bases, 0 = raw Fibonacci
 *   rotangle        : per-repetition rotation angle (rad; kasap default pi/2)
 *   fixedaxis       : 0 = hedgehog axis per rep (kasap), 1 = fixed axis (ax,ay,az)
 *   t0, dt          : gradient start delay, dwell (s); kasap: 4e-5, 1e-5
 *   maxs, maxg      : slew (T/m/s) and gradient (T/m) limits; kasap: 150, 0.04
 * Output layout: sample lin = ir + ilv*NPTS + irep*NI*NPTS (kasap dump order).
 * k in physical cycles/m. Returns total sample count, 0 on alloc failure. */
EMSCRIPTEN_KEEPALIVE
int kasap_generate(int NI, int NPTS, int NREPS,
                   float n, float at, float fov, float ms,
                   int optimize, float rotangle,
                   int fixedaxis, float axx, float axy, float axz,
                   float t0, float dt, float maxs, float maxg)
{
    const float gam = 42.57638507e6f;

    free(out_kx); free(out_ky); free(out_kz);
    free(out_gx); free(out_gy); free(out_gz);
    free(cur_basis); free(cur_reprot);

    size_t total = (size_t)NPTS * NI * NREPS;
    out_kx = malloc(total * sizeof(float));
    out_ky = malloc(total * sizeof(float));
    out_kz = malloc(total * sizeof(float));
    out_gx = malloc(total * sizeof(float));
    out_gy = malloc(total * sizeof(float));
    out_gz = malloc(total * sizeof(float));
    cur_basis  = malloc((size_t)NI * 3 * sizeof(float));
    cur_reprot = malloc((size_t)NREPS * 3 * sizeof(float));
    float *k = malloc((size_t)NI * NPTS * 3 * sizeof(float));
    float *g = malloc((size_t)NI * NPTS * 3 * sizeof(float));
    if (!out_kx || !out_ky || !out_kz || !out_gx || !out_gy || !out_gz ||
        !cur_basis || !cur_reprot || !k || !g) { free(k); free(g); return 0; }

    calchedgehog(NI, optimize, cur_basis);
    calchedgehog(NREPS, optimize, cur_reprot);

    float fixax[3] = { axx, axy, axz };
    float nrm = sqrtf(fixax[0]*fixax[0] + fixax[1]*fixax[1] + fixax[2]*fixax[2]);
    if (nrm < 1e-6f) { fixax[0] = 0; fixax[1] = 0; fixax[2] = 1; nrm = 1; }
    fixax[0] /= nrm; fixax[1] /= nrm; fixax[2] /= nrm;

    for (int irep = 0; irep < NREPS; irep++)
    {
        size_t off = (size_t)NI * NPTS * irep;
        const float *axis = fixedaxis ? fixax : &cur_reprot[3 * (irep % NREPS)];
        makegrads_p(out_gx + off, out_gy + off, out_gz + off,
                    out_kx + off, out_ky + off, out_kz + off,
                    at, fov, t0, ms, gam, dt, n, NI, NPTS, irep,
                    rotangle, maxs, maxg, axis, cur_basis, k, g);
    }

    free(k); free(g);
    cur_NI = NI; cur_NPTS = NPTS; cur_NREPS = NREPS;
    return (int)total;
}
