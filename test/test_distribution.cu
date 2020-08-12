#include <vector>

#include <catch2/catch.hpp>

#include <math/distribution.h>

using namespace rays;

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
    cudaMalloc((void **)&d_results, resultSize);

    DistributionParams params(values);
    auto distribution = params.createDistribution();
    pmfKernel<<<1, 1>>>(distribution, d_results);
    cudaDeviceSynchronize();

    float *results = (float *)malloc(resultSize);
    cudaMemcpy(
        results,
        d_results,
        resultSize,
        cudaMemcpyDeviceToHost
    );

    REQUIRE(results[0] == Approx(0.2f));
    REQUIRE(results[1] == Approx(0.2f));
    REQUIRE(results[2] == Approx(0.f));
    REQUIRE(results[3] == Approx(0.6f));
}

TEST_CASE("pmf handles empty distributions", "[distribution]") {
    const std::vector<float> values = { 0.f, 0.f };

    float *d_results;
    const size_t resultSize = values.size() * sizeof(float);
    cudaMalloc((void **)&d_results, resultSize);

    DistributionParams params(values);
    auto distribution = params.createDistribution();
    pmfKernel<<<1, 1>>>(distribution, d_results);
    cudaDeviceSynchronize();

    float *results = (float *)malloc(resultSize);
    cudaMemcpy(
        results,
        d_results,
        resultSize,
        cudaMemcpyDeviceToHost
    );

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
    cudaMalloc((void **)&d_results, resultSize);

    DistributionParams params(values);
    auto distribution = params.createDistribution();
    totalMassKernel<<<1, 1>>>(distribution, d_results);
    cudaDeviceSynchronize();

    float *results = (float *)malloc(resultSize);
    cudaMemcpy(
        results,
        d_results,
        resultSize,
        cudaMemcpyDeviceToHost
    );

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

    const std::vector<float> xis = {
        0.f,
        0.1f,
        0.2f,
        0.3f,
        0.4f,
        0.5f,
        0.6f,
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
    cudaMalloc((void **)&d_results1, result1Size);

    float *d_results2;
    const size_t result2Size = values.size() * sizeof(float);
    cudaMalloc((void **)&d_results2, result1Size);

    float *d_xis;
    const size_t xiSize = xis.size() * sizeof(float);
    cudaMalloc((void **)&d_xis, xiSize);
    cudaMemcpy(
        d_xis,
        xis.data(),
        xiSize,
        cudaMemcpyHostToDevice
    );

    DistributionParams params(values);
    auto distribution = params.createDistribution();
    sampleKernel<<<1, 1>>>(distribution, d_xis, values.size(), d_results1, d_results2);
    cudaDeviceSynchronize();

    int *results1 = (int *)malloc(result1Size);
    cudaMemcpy(
        results1,
        d_results1,
        result1Size,
        cudaMemcpyDeviceToHost
    );

    float *results2 = (float *)malloc(result2Size);
    cudaMemcpy(
        results2,
        d_results2,
        result2Size,
        cudaMemcpyDeviceToHost
    );

    for (int i = 0; i < values.size(); i++) {
        REQUIRE(results1[i] == expectedIndices[i]);
        REQUIRE(results2[i] == Approx(expectedPmfs[i]));
    }
}

TEST_CASE("Cannot sample empty distribution", "[distribution]") {
    const std::vector<float> values = { 0.f };

    const std::vector<float> xis = { 0.f };

    int *d_results1;
    const size_t result1Size = values.size() * sizeof(int);
    cudaMalloc((void **)&d_results1, result1Size);

    float *d_results2;
    const size_t result2Size = values.size() * sizeof(float);
    cudaMalloc((void **)&d_results2, result1Size);

    float *d_xis;
    const size_t xiSize = xis.size() * sizeof(float);
    cudaMalloc((void **)&d_xis, xiSize);
    cudaMemcpy(
        d_xis,
        xis.data(),
        xiSize,
        cudaMemcpyHostToDevice
    );

    DistributionParams params(values);
    auto distribution = params.createDistribution();
    sampleKernel<<<1, 1>>>(distribution, d_xis, values.size(), d_results1, d_results2);
    cudaError_t result = cudaDeviceSynchronize();
    REQUIRE(result == cudaErrorAssert);
}
