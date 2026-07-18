#!/bin/bash
# Build kasap + nufft WASM module -> web/src/wasm/kasap.js/.wasm
set -e
cd "$(dirname "$0")/.."
emcc csrc/kasap_api.c csrc/nufft.c -O3 -msimd128 \
  -s MODULARIZE=1 -s EXPORT_ES6=1 -s ENVIRONMENT=web,worker,node \
  -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=2GB \
  -s EXPORTED_FUNCTIONS='["_kasap_generate","_kasap_get_kx","_kasap_get_ky","_kasap_get_kz","_kasap_get_gx","_kasap_get_gy","_kasap_get_gz","_kasap_get_basis","_kasap_get_reprot","_kasap_get_total","_nufft_psf","_nufft_get_psf_re","_nufft_get_psf_im","_nufft_get_dcf","_wasm_malloc","_wasm_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["HEAPF32","HEAPF64","HEAP32"]' \
  -o web/src/wasm/kasap.js
echo "built web/src/wasm/kasap.js"
