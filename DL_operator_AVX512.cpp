#include <iostream>
#include <omp.h>
#include <immintrin.h>
//compile option: g++ operator_AVX512.cpp -march=native  -o operator_AVX512


// src: (rows,cols), img size: (N,C,H,W)->rows = N, cols = CxHxW
// y = x/sqrt(sum(xi^2)/N + eps)*gamma
void RMSnorm_fp32(float* src, float* gamma, float* dst, float ln_eps, int rows, int cols)
{
    auto one = _mm512_set1_ps(1.0F);
    auto eps = _mm512_set1_ps(ln_eps);
    auto len = (float)(cols);
    for(int i = 0; i < rows; i++)
    {
        auto sum_x_square = _mm512_setzero_ps(); // float32 array[16] with element 0
        for(int j = 0; j < cols; j+=16)
        {
            auto dat = _mm512_loadu_ps(src + i*cols + j); // load data with address src+i*cols+j to src+i*cols+j+511 bit
            // a*b+c, fma->xi^2+x2^2+...
            sum_x_square = _mm512_fmadd_ps(dat, dat, sum_x_square);
        }
        
        // sum(xi^2)/N
        float rms_val = _mm512_reduce_add_ps(sum_x_square) / len;

        // 1/sqrt(sum(xi^2)/N + eps)
        auto rms = _mm512_div_ps(one, _mm512_sqrt_ps(_mm512_add_ps(eps, _mm512_set1_ps(rms_val))));

        for(int j = 0; j < cols; j+=16)
        {
            // gamma * 1/sqrt(sum(xi^2)/N + eps)
            auto amp = _mm512_mul_ps(rms, _mm512_loadu_ps(gamma + j));
            auto dat = _mm512_loadu_ps(src + i*cols + j);
            // gamma * 1/sqrt(sum(xi^2)/N + eps) * x
            auto dst_val = _mm512_mul_ps(amp, dat);
            
            // store result
            _mm512_storeu_ps(dst + i*cols + j, dst_val);
        }
    }
}

// src: (rows,cols), img size: (N,C,H,W)->rows = N, cols = CxHxW
// y = (x - mu)/sqrt(var + eps)*gamma + beta
void layernorm_avx(float *src, float *gamma, float *beta, float *dst, float &ln_eps, int rows, int cols)
{
    auto one = _mm512_set1_ps(1.0F);
    auto eps = _mm512_set1_ps(ln_eps);
    auto len = (float)(cols);
    for(int i = 0; i < rows; i++)
    {
        auto sum_mean = _mm512_setzero_ps(); // float32 array[16] with element 0
        auto sum_var = _mm512_setzero_ps(); // float32 array[16]
        for(int j = 0; j < cols; j+=16)
        {
            // dat = _m512
            auto dat = _mm512_loadu_ps(src + i * cols + j); // load data with address src+i*cols+j to src+i*cols+j+511 bit
            sum_mean = _mm512_add_ps(dat, sum_mean); // dat + sum_mean
            // a*b+c, fma=fused multiply and add; x1^2+x2^2...
            sum_var = _mm512_fmadd_ps(dat, dat, sum_var);
        }

        // add instruction
        float mean_val = _mm512_reduce_add_ps(sum_mean) / len; // E(x)

        // var(x) = E(x^2)-E(x)^2
        float var_val = _mm512_reduce_add_ps(sum_var) / len - mean_val * mean_val;

        // 1/sqrt(var+eps), _mm512_set1_ps: broadcast
        auto var = _mm512_div_ps(one, _mm512_sqrt_ps(_mm512_add_ps(eps, _mm512_set1_ps(var_val))));
        auto mean = _mm512_set1_ps(mean_val);

        for(int j = 0; j < cols; j+=16)
        {
            // gamma / sqrt(var+eps)
            auto amp = _mm512_mul_ps(var, _mm512_loadu_ps(gamma + j)); // read each 16 element from gamme, then multiply var

            // x - E(x)
            auto dat = _mm512_loadu_ps(src + i * cols + j);
            auto x_sub_mean = _mm512_sub_ps(dat, mean);

            // a*b+c, fma, (x - mu) / sqrt(var+eps)*gamma + beta
            auto dst_val = _mm512_fmadd_ps(x_sub_mean, amp, _mm512_loadu_ps(beta + j));

            // consider in-place to save allocate ds buffer->src
            _mm512_storeu_ps(dst + i * cols + j, dst_val);
        }
    }
}

// load data using mask = 1, pad 0 to data using mask = 0
__m512 _maskz_loadu(const float *data_base, __mmask16 mask)
{
    return (__m512)(_mm512_maskz_loadu_ps(mask, (__m512*)(data_base)));
}

// store result to data using mask = 1
void _mask_store(float *data_base, __m512 res, __mmask16 mask)
{
    _mm512_maskz_storeu_ps((__m512*)data_base, mask, res);
}

void BiasAdd(float *src, float *bias, int rows, int cols, int stride)
{
    for(int i = 0; i < rows; i++)
    {
        // cols = 256-271
        int j = 0;
        for(; j <= cols - 16; j+=16)
        {
            auto dat = _mm512_loadu_ps(src + i*cols + j);
            auto bias_dat = _mm512_loadu_ps(bias + j);
            auto vec_out = _mm512_add_ps(dat, bias_dat);
            _mm512_storeu_ps(src + i*cols + j, vec_out);
        }
        
        if(j < cols) // cols=260, this part handle (260 - 256 + 1) tail data
        {
            __mmask16 mask = (1 << cols - j) - 1; // 10000 - 1 =>01111
            auto dat = _maskz_loadu(src + i*cols + j, mask);
            auto bias_dat = _maskz_loadu(bias + j, mask);
            auto vec_out = _mm512_add_ps(dat, bias_dat);
            _mask_store(src + i*cols + j, vec_out, mask);
        }
    }
}

void LayNorm_main()
{
    int rows = 100;
    int cols = 2048;
    float* src = (float*)malloc(rows * cols * sizeof(float));
    float* gamma = (float*)malloc(cols * sizeof(float));
    float* beta = (float*)malloc(cols * sizeof(float));
    float* dst = (float*)malloc(rows * cols * sizeof(float));
    float ln_eps = 1e-5;

    // intialize
    for(int i = 0; i < rows * cols; i++)
    {
        src[i] = (float)(i % 4 + 1); // 1 2 3 4 1 2 3 4
        if(i < cols)
        {
            gamma[i] = (float)((i % 4 + 1) * 0.5);
            beta[i] = (float)((i % 4 + 1) * 0.5);
        }
    }

    // call kernel
    layernorm_avx(src, gamma, beta, dst, ln_eps, rows, cols);
    std::cout << "layernorm output: " << dst[0] << std::endl; // 0.17082
    free(src);
    free(dst);
    free(gamma);
    free(beta);
    free(dst);
}

void BiasAdd_main()
{
    int rows = 100;
    int cols = 260;
    float* src = (float*)malloc(rows * cols * sizeof(float));
    float* bias = (float*)malloc(rows * sizeof(float));

    // initialize
    for(int i = 0; i < rows * cols; i++)
    {
        src[i] = (float)(i % 4 + 1);
        if(i < cols)
        {
            bias[i] = 1.0F;
        }
    }
    
    // call kernel
    BiasAdd(src, bias, rows, cols, 2);
    std::cout << "biasAdd output: " << src[259] << std::endl; // 4+1=5

    free(src);
    free(bias);
}

void RMSNorm_main()
{
    int rows = 100;
    int cols = 2048;
    float* src = (float*)malloc(rows * cols * sizeof(float));
    float* gamma = (float*)malloc(cols * sizeof(float));
    float* dst = (float*)malloc(rows * cols * sizeof(float));
    float ln_eps = 1e-5;

    // intialize
    for(int i = 0; i < rows * cols; i++)
    {
        src[i] = (float)(i % 4 + 1); // 1 2 3 4 1 2 3 4
        if(i < cols)
        {
            gamma[i] = (float)((i % 4 + 1) * 0.5);
        }
    }

    // call kernel
    RMSnorm_fp32(src, gamma, dst, ln_eps, rows, cols);
    std::cout << "layernorm output: " << dst[0] << std::endl; // -0.17082
    free(src);
    free(dst);
    free(gamma);
    free(dst);
}

//./lesson8 1 to run layernorm
//./lesson8 to run biasadd
int main(int argc, char *argv[]) {
  if(argv[1])
  {
    LayNorm_main();
  }
  else if(argv[2]){
    BiasAdd_main();
  }
  else
  {
    RMSNorm_main();
  }
}