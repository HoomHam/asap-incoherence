#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif
#define ASMALLNUMBER 1E-6
#define AREALLYSMALLNUMBER 1E-11

// idx uses the function-scope static localNPTS declared inside makegrads()
#define idx(ilv,point,em) ((ilv)*localNPTS*3 + (point)*3 + (em))

static inline float SQ(float x) { return x * x; }
static inline float SIGN(float x) { return (x >= 0.0f) ? 1.0f : -1.0f; }

void matmul(float a00, float a01, float a02,
            float a10, float a11, float a12,
            float a20, float a21, float a22,
            float x[3])
{
    // matrix multiply a (3x3) by x (3x1)
    float xp0 = a00 * x[0] + a01 * x[1] + a02 * x[2];
    float xp1 = a10 * x[0] + a11 * x[1] + a12 * x[2];
    x[2]      = a20 * x[0] + a21 * x[1] + a22 * x[2];
    x[0] = xp0;
    x[1] = xp1;
}

void rotarb(float *v, float th, float *u)
{
    // rotate vector v about axis u by angle th (Rodrigues / matrix form)
    float c = cosf(th), s = sinf(th), cm = 1.0f - c;
    matmul(c + SQ(u[0]) * cm,        u[0] * u[1] * cm - u[2] * s, u[0] * u[2] * cm + u[1] * s,
           u[1] * u[0] * cm + u[2] * s, c + SQ(u[1]) * cm,        u[1] * u[2] * cm - u[0] * s,
           u[2] * u[0] * cm - u[1] * s, u[2] * u[1] * cm + u[0] * s, c + SQ(u[2]) * cm, v);
}

float dot(float *x, float *y) { return x[0]*y[0] + x[1]*y[1] + x[2]*y[2]; }

void norm(float *x, float r)
{
    float n = sqrtf(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) / r;
    if (n < ASMALLNUMBER) n = 1.0f;
    x[0] /= n; x[1] /= n; x[2] /= n;
}

void cpyvec(float *x, const float *y) { x[0]=y[0]; x[1]=y[1]; x[2]=y[2]; }

void cross(float *x, float *y, float *result)
{
    result[0] = x[1] * y[2] - y[1] * x[2];
    result[1] = y[0] * x[2] - x[0] * y[2];
    result[2] = x[0] * y[1] - y[0] * x[1];
    norm(result, 1.0f);
}

float calct1(float atp, float kmax, float kddmax, float n)
{
    if ((n >= 2.0f - ASMALLNUMBER) && (n <= 2.0f + ASMALLNUMBER))
        return atp;
    return atp / (powf(2.0f * kmax / kddmax / atp / atp, 1.0f / (n - 2.0f)) - 1.0f);
}

float krp(float tp, float kmax, float t1, float atp, float n)
{
    float A = kmax / atp / atp / powf(1.0f + atp / t1, n - 2.0f);
    return A * tp * tp * powf(1.0f + tp / t1, n - 2.0f);
}

float kr(float t, float t0, float kddmax, float ms, float fov, float AT, float n)
{
    if (t < t0) return 0.0f;
    float kmax = ms / (2.0f * fov);
    float t1   = calct1(AT - t0, kmax, kddmax, n);
    return krp(t - t0, kmax, t1, AT - t0, n);
}

float krdotp(float tp, float kmax, float t1, float atp, float n)
{
    if (tp < ASMALLNUMBER) return 0.0f;
    float A = kmax / atp / atp / powf(1.0f + atp / t1, n - 2.0f);
    return A * tp * powf(1.0f + tp / t1, n-3.0f) * (2.0f + n * tp / t1);
}

float krdot(float t, float t0, float kddmax, float ms, float fov, float at, float n)
{
    if (t < t0) return 0.0f;
    float kmax = ms / (2.0f * fov);
    float t1   = calct1(at - t0, kmax, kddmax, n);
    return krdotp(t - t0, kmax, t1, at - t0, n);
}

float krddot(float t, float t0, float kddmax, float ms, float fov, float at, float n)
{
    float tp = t - t0;
    if (tp < ASMALLNUMBER) return 0.0f;
    float kmax = ms / (2.0f * fov);
    float atp  = at - t0;
    float t1   = calct1(at - t0, ms / (2.0f * fov), kddmax, n);
    float A    = kmax / atp / atp / powf(1.0f + atp / t1, n - 2.0f);
    float newway = A * powf(1.0f + tp / t1, n - 4.0f) * (2.0f + (n - 1.0f) * tp / t1 * (4.0f + n * tp / t1));
    return newway;
}

void optimizehedgehog(int n, float *v)
{
    float *vel = (float *)malloc(3 * n * sizeof(float));
    float dv[3];
    float sqn = sqrtf((float)n);

    for (int i = 0; i < 3*n; i++) vel[i] = 0.0f;

    for (int it = 0; it < 10000; it++)
    {
        float maxnorm = 0.0f;
        for (int j = 0; j < n; j++)
        {
            int jidx = j * 3;
            for (int k = j + 1; k < n; k++)
            {
                int kidx = k * 3;
                dv[0] = v[jidx]   - v[kidx];
                dv[1] = v[jidx+1] - v[kidx+1];
                dv[2] = v[jidx+2] - v[kidx+2];
                float nrm   = dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2];
                float denom = nrm * sqn * 3.0f + 1e-12f;

                float jdot = dv[0]*v[jidx] + dv[1]*v[jidx+1] + dv[2]*v[jidx+2];
                vel[jidx]   += (dv[0] - jdot * v[jidx])     / denom;
                vel[jidx+1] += (dv[1] - jdot * v[jidx+1])   / denom;
                vel[jidx+2] += (dv[2] - jdot * v[jidx+2])   / denom;

                float kdot = dv[0]*v[kidx] + dv[1]*v[kidx+1] + dv[2]*v[kidx+2];
                vel[kidx]   -= (dv[0] - kdot * v[kidx])     / denom;
                vel[kidx+1] -= (dv[1] - kdot * v[kidx+1])   / denom;
                vel[kidx+2] -= (dv[2] - kdot * v[kidx+2])   / denom;
            }
        }

        for (int j = 0; j < n; j++)
        {
            int jidx = j * 3;
            v[jidx]   += vel[jidx];
            v[jidx+1] += vel[jidx+1];
            v[jidx+2] += vel[jidx+2];

            float nrm = sqrtf(v[jidx]*v[jidx] + v[jidx+1]*v[jidx+1] + v[jidx+2]*v[jidx+2]);
            if (nrm < ASMALLNUMBER) nrm = 1.0f;
            v[jidx]   /= nrm;
            v[jidx+1] /= nrm;
            v[jidx+2] /= nrm;

            vel[jidx]   *= 0.9f;
            vel[jidx+1] *= 0.9f;
            vel[jidx+2] *= 0.9f;

            float v2 = vel[jidx]*vel[jidx] + vel[jidx+1]*vel[jidx+1] + vel[jidx+2]*vel[jidx+2];
            if (v2 > maxnorm) maxnorm = v2;
        }
        if (maxnorm < AREALLYSMALLNUMBER) break;
    }

    free(vel);
}

void calchedgehog(int n, int optimize, float *target)
{
    float phi = (float)M_PI * (sqrtf(5.0f) - 1.0f);
    for (int i = 0; i < n; i++)
    {
        int idx3 = 3 * i;
        float theta = phi * (float)i;
        target[idx3] = 1.0f - ((float)i / (float)(n - 1)) * 2.0f;
        float r = sqrtf(1.0f - target[idx3] * target[idx3]);
        target[idx3 + 1] = cosf(theta) * r;
        target[idx3 + 2] = sinf(theta) * r;
        norm(&target[idx3], 1.0f);
    }
    if (optimize) optimizehedgehog(n, target);
}

// Global basis vectors
float *initv  = (float *)0;
float *reprot = (float *)0;

// Compute gradients and k, pack both out
void makegrads(float gx[], float gy[], float gz[],
               float kx[], float ky[], float kz[],
               float at, float fov, float t0, float ms, float gam, float dt, float n,
               int NI, int NPTS, int NREPS, int optimize, int irep)
{
    static int localNI = -1, localNPTS = -1, localNREPS = -1, localoptimize = -1;
    static float *k = (float *)0, *g = (float *)0;

    if ((NI != localNI) || (NPTS != localNPTS) || (NREPS != localNREPS) || (optimize != localoptimize))
    {
        if (k) free(k);
        if (g) free(g);
        if ((initv)  && (NI    != localNI))   { free(initv);  initv  = 0; }
        if ((reprot) && (NREPS != localNREPS)){ free(reprot); reprot = 0; }

        k = (float *)malloc(NI * NPTS * 3 * sizeof(float));
        g = (float *)malloc(NI * NPTS * 3 * sizeof(float));

        if ((NI != localNI) || (optimize != localoptimize))
        {
            initv = (float *)malloc(NI * 3 * sizeof(float));
            calchedgehog(NI, optimize, initv);
        }
        if ((NREPS != localNREPS) || (optimize != localoptimize))
        {
            reprot = (float *)malloc(NREPS * 3 * sizeof(float));
            calchedgehog(NREPS, optimize, reprot);
        }

        localNI = NI; localNPTS = NPTS; localNREPS = NREPS; localoptimize = optimize;
    }

    const float MAXS = 150.0f; // T/m/s
    const float MAXG = 0.04f;  // T/m

    float rotvec[3] = { 0.0f, 0.0f, 1.0f };
    float xhat[3]   = { 1.0f, 0.0f, 0.0f };

    for (int ir = 0, firstime = 1; ir < NPTS; ir++)
    {
        float t = ir * dt;
        float thisk = kr(t, t0, MAXS * gam, ms, fov, at, n);

        if (thisk < ASMALLNUMBER)
        {
            for (int j = 0; j < NI; j++)
                for (int m = 0; m < 3; m++)
                    g[idx(j,ir,m)] = 0.0f;
            continue;
        }

        float thiskdot  = krdot (t, t0, MAXS * gam, ms, fov, at, n);
        float thiskddot = krddot(t, t0, MAXS * gam, ms, fov, at, n);

        float wG = sqrtf(SQ(MAXG * gam) - SQ(thiskdot)) / thisk;

        float tmpA = 2.0f * SQ(MAXS * gam * thisk)
                   + SQ(SQ(thiskdot))
                   - 2.0f * thisk * SQ(thiskdot) * thiskddot
                   - SQ(thisk * thiskdot);
        float wS = sqrtf(sqrtf(tmpA) + thisk * thiskddot - SQ(thiskdot)) / 1.41421356237f / thisk;

        float w = (wS < wG) ? wS : wG;

        // rotate rotvec around xhat at optimal rate
        rotarb(rotvec, w * dt, xhat);
        float rotangle = w * dt;

        for (int j = 0; j < NI; j++)
        {
            if (firstime) cpyvec(&k[idx(j,ir,0)], &initv[3*j]);
            else          cpyvec(&k[idx(j,ir,0)], &k[idx(j,ir-1,0)]);

            norm(&k[idx(j,ir,0)], thisk);
            rotarb(&k[idx(j,ir,0)], rotangle, rotvec);

            for (int m = 0; m < 3; m++)
                g[idx(j,ir,m)] = (ir == 0) ? 0.0f : (k[idx(j,ir,m)] - k[idx(j,ir-1,m)]) / dt / gam;
        }
        firstime = 0;
    }

    // ramp down to g=0
    int ir = NPTS - 2;
    for (int j = 0; j < NI; j++)
        for (; sqrtf(dot(&g[idx(j,ir,0)], &g[idx(j,ir,0)])) > (NPTS - 1 - ir) * dt * MAXS; ir--) { /* shrink window */ }

    for (int irp = ir + 1; irp < NPTS; irp++)
        for (int j = 0; j < NI; j++)
            for (int m = 0; m < 3; m++)
                g[idx(j,irp,m)] = g[idx(j,ir,m)] * (float)(NPTS - 1 - irp) / (float)(NPTS - 1 - ir);

    // rotate by 90 deg around reprot ray for this repetition
    for (int ir2 = 0; ir2 < NPTS; ir2++)
        for (int j = 0; j < NI; j++)
        {
            rotarb(&g[idx(j,ir2,0)], (float)M_PI / 2.0f, &reprot[3 * (irep % NREPS)]);
            rotarb(&k[idx(j,ir2,0)], (float)M_PI / 2.0f, &reprot[3 * (irep % NREPS)]);
        }

    // pack gradient AND k into caller-provided linear arrays
    for (int ir3 = 0; ir3 < NPTS; ir3++)
        for (int j = 0; j < NI; j++)
        {
            int lin = ir3 + j * localNPTS; // caller passes an offset per irep
            gx[lin] = g[idx(j,ir3,0)];
            gy[lin] = g[idx(j,ir3,1)];
            gz[lin] = g[idx(j,ir3,2)];

            kx[lin] = k[idx(j,ir3,0)];
            ky[lin] = k[idx(j,ir3,1)];
            kz[lin] = k[idx(j,ir3,2)];
        }
}

int main(void)  
{
    const int   NI   = 26;
    const int   NPTS = 512;
    const int   NREPS= 32;
    const float n    = 2.0f;
    const float at   = NPTS * 1.0e-5f;
    const float gamma= 42.57638507e6f;
    const float fov  = 350.0f / 1000.0f;
    const float ms   = 160.0f;
    // const int intro = 4;
    // const int outro = 35 - (int)(fov * 15.0f);

    size_t total = (size_t)NPTS * NI * NREPS;

    float *gx = (float *)malloc(total * sizeof(float));
    float *gy = (float *)malloc(total * sizeof(float));
    float *gz = (float *)malloc(total * sizeof(float));
    float *kx = (float *)malloc(total * sizeof(float));
    float *ky = (float *)malloc(total * sizeof(float));
    float *kz = (float *)malloc(total * sizeof(float));

    if (!gx || !gy || !gz || !kx || !ky || !kz) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    // unoptimized basis (Fibonacci)
    for (int irep = 0; irep < NREPS; irep++)
    {
        size_t off = (size_t)NI * NPTS * irep;
        makegrads(gx+off, gy+off, gz+off,
                  kx+off, ky+off, kz+off,
                  at, fov, 4.0e-5f, ms, gamma, 1.0e-5f, n,
                  NI, NPTS, NREPS, 0, irep);
    }

    // dump unoptimized interleave basis
    FILE *f = fopen("ilvbasis_preopt.txt", "w");
    if (!f) { fprintf(stderr, "Cannot open ilvbasis_preopt.txt\n"); return 2; }
    for (int j = 0; j < NI; j++)
        fprintf(f, "%f %f %f\n", initv[3*j], initv[3*j+1], initv[3*j+2]);
    fclose(f);

    // remake with optimized (Thompson) basis
    for (int irep = 0; irep < NREPS; irep++)
    {
        size_t off = (size_t)NI * NPTS * irep;
        makegrads(gx+off, gy+off, gz+off,
                  kx+off, ky+off, kz+off,
                  at, fov, 4.0e-5f, ms, gamma, 1.0e-5f, n,
                  NI, NPTS, NREPS, 1, irep);
    }

    // dump optimized interleave basis
    f = fopen("ilvbasis_postopt.txt", "w");
    if (!f) { fprintf(stderr, "Cannot open ilvbasis_postopt.txt\n"); return 3; }
    for (int j = 0; j < NI; j++)
        fprintf(f, "%f %f %f\n", initv[3*j], initv[3*j+1], initv[3*j+2]);
    fclose(f);

    // dump gradient trajectory (G)
    f = fopen("fancytraj.txt", "w");
    if (!f) { fprintf(stderr, "Cannot open fancytraj.txt\n"); return 4; }
    fprintf(f, "%d %d %d\n", NI, NREPS, NPTS);
    for (size_t i = 0; i < total; i++)
        fprintf(f, "%f %f %f\n", gx[i], gy[i], gz[i]);
    fclose(f);

    // dump k-space trajectory (K)
    f = fopen("kspacetraj.txt", "w");
    if (!f) { fprintf(stderr, "Cannot open kspacetraj.txt\n"); return 5; }
    fprintf(f, "%d %d %d\n", NI, NREPS, NPTS);
    for (size_t i = 0; i < total; i++)
        fprintf(f, "%f %f %f\n", kx[i], ky[i], kz[i]);
    fclose(f);

    free(gx); free(gy); free(gz);
    free(kx); free(ky); free(kz);
    if (initv)  free(initv);
    if (reprot) free(reprot);

    return 0;
}