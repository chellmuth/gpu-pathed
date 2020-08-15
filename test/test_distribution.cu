#include <vector>

#include <catch2/catch.hpp>

#include <core/sample.h>
#include <macro_helper.h>
#include <math/distribution.h>
#include <renderers/random.h>

#define checkCUDA(result) { gpuAssert((result), __FILE__, __LINE__); }

using namespace rays;

// TEST DISTRIBUTION

__global__ static void pmfKernel(Distribution distribution, float *results)
{
    for (int i = 0; i < distribution.getSize(); i++) {
        results[i] = distribution.pmf(i);
    }
}

__global__ static void totalMassKernel(Distribution distribution, float *results)
{
    results[0] = distribution.getTotalMass();
}

__global__ static void sampleKernel(
    Distribution distribution,
    float *xis,
    int count,
    int *results1,
    float *results2
) {
    for (int i = 0; i < count; i++) {
        float pmf;
        results1[i] = distribution.sample(&pmf, xis[i]);
        results2[i] = pmf;
    }
}

TEST_CASE("pmf method", "[distribution]") {
    const std::vector<float> values = {
        1.f,
        1.f,
        0.f,
        3.f
    };

    float *d_results;
    const size_t resultSize = values.size() * sizeof(float);
    checkCUDA(cudaMalloc((void **)&d_results, resultSize));

    DistributionParams params(values);
    auto distribution = params.createDistribution();
    pmfKernel<<<1, 1>>>(distribution, d_results);
    checkCUDA(cudaDeviceSynchronize());

    float *results = (float *)malloc(resultSize);
    checkCUDA(cudaMemcpy(
        results,
        d_results,
        resultSize,
        cudaMemcpyDeviceToHost
    ));

    REQUIRE(results[0] == Approx(0.2f));
    REQUIRE(results[1] == Approx(0.2f));
    REQUIRE(results[2] == Approx(0.f));
    REQUIRE(results[3] == Approx(0.6f));
}

TEST_CASE("pmf handles empty distributions", "[distribution]") {
    const std::vector<float> values = { 0.f, 0.f };

    float *d_results;
    const size_t resultSize = values.size() * sizeof(float);
    checkCUDA(cudaMalloc((void **)&d_results, resultSize));

    DistributionParams params(values);
    auto distribution = params.createDistribution();
    pmfKernel<<<1, 1>>>(distribution, d_results);
    checkCUDA(cudaDeviceSynchronize());

    float *results = (float *)malloc(resultSize);
    checkCUDA(cudaMemcpy(
        results,
        d_results,
        resultSize,
        cudaMemcpyDeviceToHost
    ));

    REQUIRE(results[0] == 0.f);
    REQUIRE(results[1] == 0.f);
}

TEST_CASE("Distribution knows its total mass", "[distribution]") {
    const std::vector<float> values = {
        1.f,
        1.f,
        0.f,
        3.f
    };

    float *d_results;
    const size_t resultSize = sizeof(float);
    checkCUDA(cudaMalloc((void **)&d_results, resultSize));

    DistributionParams params(values);
    auto distribution = params.createDistribution();
    totalMassKernel<<<1, 1>>>(distribution, d_results);
    checkCUDA(cudaDeviceSynchronize());

    float *results = (float *)malloc(resultSize);
    checkCUDA(cudaMemcpy(
        results,
        d_results,
        resultSize,
        cudaMemcpyDeviceToHost
    ));

    REQUIRE(results[0] == Approx(5.f));
}

TEST_CASE("Distribution can be sampled", "[distribution]") {
    const std::vector<float> values = {
        1.f,
        1.f,
        0.f,
        3.f,
        0.f
    };

    const float eta = 1e-5;
    const std::vector<float> xis = {
        0.f,
        0.1f - eta,
        0.2f - eta,
        0.3f - eta,
        0.4f - eta,
        0.5f - eta,
        0.6f - eta,
        1.f,
    };

    const std::vector<int> expectedIndices = {
        0, 0, 0, 1, 1, 3, 3, 3,
    };

    const std::vector<float> expectedPmfs = {
        0.2, 0.2f, 0.2f, 0.2f, 0.2f, 0.6f, 0.6f, 0.6f
    };

    int *d_results1;
    const size_t result1Size = values.size() * sizeof(int);
    checkCUDA(cudaMalloc((void **)&d_results1, result1Size));

    float *d_results2;
    const size_t result2Size = values.size() * sizeof(float);
    checkCUDA(cudaMalloc((void **)&d_results2, result1Size));

    float *d_xis;
    const size_t xiSize = xis.size() * sizeof(float);
    checkCUDA(cudaMalloc((void **)&d_xis, xiSize));
    checkCUDA(cudaMemcpy(
        d_xis,
        xis.data(),
        xiSize,
        cudaMemcpyHostToDevice
    ));

    DistributionParams params(values);
    auto distribution = params.createDistribution();
    sampleKernel<<<1, 1>>>(distribution, d_xis, values.size(), d_results1, d_results2);
    checkCUDA(cudaDeviceSynchronize());

    int *results1 = (int *)malloc(result1Size);
    checkCUDA(cudaMemcpy(
        results1,
        d_results1,
        result1Size,
        cudaMemcpyDeviceToHost
    ));

    float *results2 = (float *)malloc(result2Size);
    checkCUDA(cudaMemcpy(
        results2,
        d_results2,
        result2Size,
        cudaMemcpyDeviceToHost
    ));

    for (int i = 0; i < values.size(); i++) {
        REQUIRE(results1[i] == expectedIndices[i]);
        REQUIRE(results2[i] == Approx(expectedPmfs[i]));
    }
}

// Skip; Kernel asserts destroy the CUDA context in an unrecoverable way
//
// TEST_CASE("Cannot sample empty distribution", "[distribution]") {
//     const std::vector<float> values = { 0.f };

//     const std::vector<float> xis = { 0.f };

//     int *d_results1;
//     const size_t result1Size = values.size() * sizeof(int);
//     checkCUDA(cudaMalloc((void **)&d_results1, result1Size));

//     float *d_results2;
//     const size_t result2Size = values.size() * sizeof(float);
//     checkCUDA(cudaMalloc((void **)&d_results2, result1Size));

//     float *d_xis;
//     const size_t xiSize = xis.size() * sizeof(float);
//     checkCUDA(cudaMalloc((void **)&d_xis, xiSize));
//     checkCUDA(cudaMemcpy(
//         d_xis,
//         xis.data(),
//         xiSize,
//         cudaMemcpyHostToDevice
//     ));

//     DistributionParams params(values);
//     auto distribution = params.createDistribution();
//     sampleKernel<<<1, 1>>>(distribution, d_xis, values.size(), d_results1, d_results2);

//     cudaError_t result = cudaDeviceSynchronize();
//     REQUIRE(result == cudaErrorAssert);
// }


// TEST DISTRIBUTION2D

__global__ static void pmfKernel(Distribution2D distribution2D, float *results)
{
    const int height = distribution2D.getHeight();
    const int width = distribution2D.getWidth();

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int index = y * width + x;
            results[index] = distribution2D.pmf(x, y);
        }
    }
}

__global__ static void sampleKernel(
    const Distribution2D distribution2D,
    const float *xis,
    const int count,
    Distribution2D::Sample *results1,
    float *results2
) {
    for (int i = 0; i < count; i++) {
        float pmf;
        float2 sample = {xis[i*2], xis[i*2 + 1]};
        results1[i] = distribution2D.sample(&pmf, sample);
        results2[i] = pmf;
    }
}

TEST_CASE("2d pmf method", "[distribution]") {
    constexpr int width = 2;
    constexpr int height = 4;
    constexpr int size = width * height;
    const float values[size] = {
        1.f, 4.f,
        1.f, 0.f,
        0.f, 0.f,
        3.f, 1.f,
    };

    const std::vector<float> expected = {
        0.1f, 0.4f,
        0.1f, 0.f,
        0.f, 0.f,
        0.3f, 0.1f
    };

    float *d_results;
    const size_t resultSize = size * sizeof(float);
    checkCUDA(cudaMalloc((void **)&d_results, resultSize));

    Distribution2DBuilder builder(values, width, height);
    auto distribution = builder.buildDistribution2D();
    pmfKernel<<<1, 1>>>(distribution, d_results);
    checkCUDA(cudaDeviceSynchronize());

    float *results = (float *)malloc(resultSize);
    checkCUDA(cudaMemcpy(
        results,
        d_results,
        resultSize,
        cudaMemcpyDeviceToHost
    ));

    for (int i = 0; i < size; i++) {
        REQUIRE(results[i] == Approx(expected[i]));
    }
}

TEST_CASE("2d Distribution can be sampled", "[distribution]") {
    constexpr int width = 2;
    constexpr int height = 4;
    constexpr int size = width * height;
    const float values[size] = {
        1.f, 4.f,
        1.f, 0.f,
        0.f, 0.f,
        3.f, 1.f,
    };

    const float eta = 1e-5;
    const std::vector<float> xis = {
        0.f, 0.2f - eta,
        0.1f - eta, 0.3f - eta,
        0.51f, 1.f,
        0.7f - eta, 0.75f - eta,
        0.7f - eta, 0.76f,
    };

    const std::vector<Distribution2D::Sample> expectedIndices = {
        { 0, 0 },
        { 1, 0 },
        { 0, 1 },
        { 0, 3 },
        { 1, 3 },
    };

    const std::vector<float> expectedPmfs = {
        0.1f, 0.4f, 0.1f, 0.3f, 0.1f
    };

    int asserts = expectedIndices.size();

    Distribution2D::Sample *d_results1;
    const size_t result1Size = asserts * sizeof(Distribution2D::Sample);
    checkCUDA(cudaMalloc((void **)&d_results1, result1Size));

    float *d_results2;
    const size_t result2Size = asserts * sizeof(float);
    checkCUDA(cudaMalloc((void **)&d_results2, result1Size));

    float *d_xis;
    const size_t xiSize = xis.size() * sizeof(float);
    checkCUDA(cudaMalloc((void **)&d_xis, xiSize));
    checkCUDA(cudaMemcpy(
        d_xis,
        xis.data(),
        xiSize,
        cudaMemcpyHostToDevice
    ));

    Distribution2DBuilder builder(values, width, height);
    auto distribution2D = builder.buildDistribution2D();
    sampleKernel<<<1, 1>>>(distribution2D, d_xis, asserts, d_results1, d_results2);
    checkCUDA(cudaDeviceSynchronize());

    Distribution2D::Sample *results1 = (Distribution2D::Sample *)malloc(result1Size);
    checkCUDA(cudaMemcpy(
        results1,
        d_results1,
        result1Size,
        cudaMemcpyDeviceToHost
    ));

    float *results2 = (float *)malloc(result2Size);
    checkCUDA(cudaMemcpy(
        results2,
        d_results2,
        result2Size,
        cudaMemcpyDeviceToHost
    ));

    for (int i = 0; i < asserts; i++) {
        REQUIRE(results1[i] == expectedIndices[i]);
        REQUIRE(results2[i] == Approx(expectedPmfs[i]));
    }
}


// TEST PHI-THETA DISTRIBUTION

__global__ static void pdfCanonicalKernel(PhiThetaDistribution distribution, float *results, int width, int height)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const float phi = (x + 0.5f) / width;
            const float theta = (y + 0.5f) / height;

            const int index = y * width + x;
            results[index] = distribution.pdf(phi, theta);
        }
    }
}

__global__ static void pdfPhiThetaKernel(PhiThetaDistribution distribution, float *results)
{
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int x = threadIdx.x + blockIdx.x * blockDim.x;

    const int width = gridDim.x * blockDim.x;

    const int index = y * width + x;
    unsigned int seed = tea<4>(index, 0);

    const float xi1 = rnd(seed);
    const float xi2 = rnd(seed);

    Vec3 wi = Sample::uniformSphere(xi1, xi2);
    results[index] = distribution.pdf(wi);
}

TEST_CASE("phi-theta pdf canonical", "[distribution]") {
    constexpr int width = 2;
    constexpr int height = 4;
    constexpr int size = width * height;

    const float values[size] = {
        1.f, 4.f,
        1.f, 0.f,
        0.f, 0.f,
        3.f, 1.f,
    };

    constexpr int multiplier = 8;
    const int multiplier2 = multiplier * multiplier;
    const int cells = width * height * multiplier2;

    float *d_results;
    const size_t resultSize = cells * sizeof(float);
    checkCUDA(cudaMalloc((void **)&d_results, resultSize));

    PhiThetaDistributionBuilder builder(values, width, height);
    auto distribution = builder.buildPhiThetaDistribution();
    pdfCanonicalKernel<<<1, 1>>>(distribution, d_results, width * multiplier, height * multiplier);
    checkCUDA(cudaDeviceSynchronize());

    float *results = (float *)malloc(resultSize);
    checkCUDA(cudaMemcpy(
        results,
        d_results,
        resultSize,
        cudaMemcpyDeviceToHost
    ));

    float sum = 0.f;
    for (int j = 0; j < cells; j++) {
        sum += results[j] / cells;
    }

    REQUIRE(sum == Approx(1.f));
}

TEST_CASE("phi-theta pdf spherical", "[distribution]") {
    constexpr int width = 2;
    constexpr int height = 4;
    constexpr int size = width * height;

    const float values[size] = {
        1.f, 4.f,
        1.f, 0.f,
        0.f, 0.f,
        3.f, 1.f,
    };

    constexpr int dim = 1000;
    const int cells = dim * dim;

    float *d_results;
    const size_t resultSize = cells * sizeof(float);
    checkCUDA(cudaMalloc((void **)&d_results, resultSize));

    PhiThetaDistributionBuilder builder(values, width, height);
    auto distribution = builder.buildPhiThetaDistribution();
    pdfPhiThetaKernel<<<dim3(dim, dim), 1>>>(distribution, d_results);
    checkCUDA(cudaDeviceSynchronize());

    float *results = (float *)malloc(resultSize);
    checkCUDA(cudaMemcpy(
        results,
        d_results,
        resultSize,
        cudaMemcpyDeviceToHost
    ));

    float sum = 0.f;
    for (int j = 0; j < cells; j++) {
        sum += results[j] * M_PI * 4.f;
    }

    REQUIRE(sum / cells == Approx(1.f).epsilon(1e-2));
}
