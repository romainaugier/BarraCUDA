#ifndef BARRACUDA_AMDGPU_ENCODE_H
#define BARRACUDA_AMDGPU_ENCODE_H

#include "amdgpu.h"

/* Binary-encode a single machine function into A->code[].
   Called by emit.c during ELF generation. */
void encode_function(amd_module_t *A, uint32_t mf_idx);

#endif /* BARRACUDA_AMDGPU_ENCODE_H */
