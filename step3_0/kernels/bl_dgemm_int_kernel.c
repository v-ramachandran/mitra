/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2016, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <stdio.h>
#include <immintrin.h> // AVX

#include <bl_dgemm_kernel.h>
#include <avx_types.h>

#define inc_t unsigned long long 

//micro-panel a is stored in column major, lda=DGEMM_MR.
#define a(i,j) a[ (j)*DGEMM_MR + (i) ]
#define acol(j) a[ (j)*DGEMM_MR ]
//micro-panel b is stored in row major, ldb=DGEMM_NR.
#define b(i,j) b[ (i)*DGEMM_NR + (j) ]
//result      c is stored in column major.
#define c(i,j) c[ (j)*ldc + (i) ]
#define ccol(j) c[ (j)*ldc ]

void bl_daxpy_asm_4x1(
        double *alpha,
        double *x,
        double *y
        )
{
    __asm__ volatile
    (
	"                                            \n\t"
	"movq                %1, %%rax               \n\t" // load address of x.              ( v )
	"movq                %2, %%rbx               \n\t" // load address of y.              ( v )
	"movq                %0, %%rcx               \n\t" // load address of alpha.          ( s )
	"                                            \n\t"
    "vxorpd    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to 0                   ( v )
    "vxorpd    %%ymm1,  %%ymm1,  %%ymm1          \n\t" // set ymm1 to 0                   ( v )
    "vxorpd    %%ymm2,  %%ymm2,  %%ymm2          \n\t" // set ymm2 to 0                   ( v )
	"                                            \n\t"
    "vmovapd   0 * 32(%%rax), %%ymm0             \n\t" // load x
    "vmovapd   0 * 32(%%rbx), %%ymm1             \n\t" // load y
	"                                            \n\t"
	"vbroadcastsd       0 *  8(%%rcx), %%ymm2    \n\t" // load alpha, broacast to ymm2
	"vfmadd231pd       %%ymm2, %%ymm0, %%ymm1    \n\t" // y := alpha * x + y (fma)
	"vmovaps           %%ymm1, 0 * 32(%%rbx)     \n\t" // store back y
	"                                            \n\t"
	".DDONE:                                     \n\t"
	"                                            \n\t"
	: // output operands (none)
	: // input operands
	  "m" (alpha),        // 0
	  "m" (x),            // 1
	  "m" (y)             // 2
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	);
}

void bl_daxpy(
        double *alpha,
        double *x,
        double *y,
        int n
        )
{
    int i;
    for ( i = 0; i < n; i += 4 ) {
        bl_daxpy_asm_4x1( alpha, x, y );
        //bl_daxpy_int_4x1( alpha, x, y );
        x += 4;
        y += 4;
    }
}

void bl_daxpy_ref(
        double *alpha,
        double *x,
        double *y,
        int n
        )
{
    int i;
    for ( i = 0; i < n; i ++ ) {
        y[ i ] = *alpha * x[ i ] + y[ i ];
    }
}

void bl_dgemm_int_kernel(
                        int      k,
                        double*  a,
                        double*  b,
                        double*  c,
                        unsigned long long ldc,
                        aux_t*         data
                      )
{
  dim_t i;
  
  for ( i = 0; i < k; ++i ) {                 
    double* ai = &acol(i);
    double* ci;
    // First column
    ci = &ccol(0);
    bl_daxpy(&b(i,0), ai, ci, 8);

    // Second column
    ci = &ccol(1);
    bl_daxpy(&b(i,1), ai, ci, 8);
    
    // Third column
    ci = &ccol(2);
    bl_daxpy(&b(i,2), ai, ci, 8);
    
    // Fourth column
    ci = &ccol(3);
    bl_daxpy(&b(i,3), ai, ci, 8);
  }
}


