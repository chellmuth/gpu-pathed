#include "renderers/optix.h"

#include <assert.h>
#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include "framebuffer.h"
#include "macro_helper.h"
#include "scene_data.h"
#include "optix_kernel.h"
#include "triangle.h"

#define checkCUDA(result) { gpuAssert((result), __FILE__, __LINE__); }
#define checkOptix(result) { optixAssert((result), __FILE__, __LINE__); }

namespace rays {

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

static void contextLogCallback(
    unsigned int level,
    const char *tag,
    const char *message,
    void * /*cbdata */
)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << std::endl;
}

static void initContext(OptixDeviceContext &context)
{
    // Initialize CUDA
    checkCUDA(cudaFree(0));

    checkOptix(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &contextLogCallback;
    options.logCallbackLevel = 4;

    CUcontext cuContext = 0; // zero means take the current context
    checkOptix(optixDeviceContextCreate(cuContext, &options, &context));
}

static void initAcceleration(
    OptixDeviceContext &context,
    OptixTraversableHandle &gasHandle,
    CUdeviceptr &d_gasOutputBuffer,
    const SceneData &sceneData
) {
    // Use default options for simplicity
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    std::vector<float3> vertices;
    std::vector<int> materialIndices;
    for (const auto &triangle : sceneData.triangles) {
        const Vec3 p0 = triangle.p0();
        const Vec3 p1 = triangle.p1();
        const Vec3 p2 = triangle.p2();

        vertices.push_back({ p0.x(), p0.y(), p0.z() });
        vertices.push_back({ p1.x(), p1.y(), p1.z() });
        vertices.push_back({ p2.x(), p2.y(), p2.z() });

        materialIndices.push_back(triangle.materialIndex());
    }

    const size_t verticesSize = sizeof(float3) * vertices.size();
    CUdeviceptr d_vertices = 0;
    checkCUDA(cudaMalloc(reinterpret_cast<void **>(&d_vertices), verticesSize));
    checkCUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_vertices),
        vertices.data(),
        verticesSize,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_materialIndices = 0;
    const size_t materialIndicesSizeInBytes = materialIndices.size() * sizeof(int);
    checkCUDA(cudaMalloc(reinterpret_cast<void **>(&d_materialIndices), materialIndicesSizeInBytes));
    checkCUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_materialIndices),
        materialIndices.data(),
        materialIndicesSizeInBytes,
        cudaMemcpyHostToDevice
    ));

    const std::vector<Material> &materials = sceneData.materials;
    std::vector<uint32_t> triangleInputFlags;
    triangleInputFlags.reserve(vertices.size());
    for (const auto &material : materials) {
        triangleInputFlags.push_back(OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
    }

    // Build input is list of non-indexed triangle vertices
    OptixBuildInput triangleInput = {};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
    triangleInput.triangleArray.vertexBuffers = &d_vertices;
    triangleInput.triangleArray.flags = triangleInputFlags.data();
    triangleInput.triangleArray.numSbtRecords = materials.size();
    triangleInput.triangleArray.sbtIndexOffsetBuffer = d_materialIndices;
    triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(int);
    triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    OptixAccelBufferSizes gasBufferSizes;
    checkOptix(optixAccelComputeMemoryUsage(
        context,
        &accelOptions,
        &triangleInput,
        1, // Number of build inputs
        &gasBufferSizes
    ));

    CUdeviceptr d_tempBufferGas;
    checkCUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_tempBufferGas),
        gasBufferSizes.tempSizeInBytes
    ));
    checkCUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_gasOutputBuffer),
        gasBufferSizes.outputSizeInBytes
    ));

    checkOptix(optixAccelBuild(
        context,
        0,                  // CUDA stream
        &accelOptions,
        &triangleInput,
        1,                  // num build inputs
        d_tempBufferGas,
        gasBufferSizes.tempSizeInBytes,
        d_gasOutputBuffer,
        gasBufferSizes.outputSizeInBytes,
        &gasHandle,
        nullptr,            // emitted property list
        0                   // num emitted properties
    ));

    checkCUDA(cudaFree(reinterpret_cast<void *>(d_tempBufferGas)));
    // TODO: Might want these later
    checkCUDA(cudaFree(reinterpret_cast<void *>(d_vertices)));
}

static void createModule(
    OptixDeviceContext &context,
    OptixModule &module,
    OptixPipelineCompileOptions &pipelineCompileOptions,
    char *log
) {
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 3;
    pipelineCompileOptions.numAttributeValues = 3;
#ifdef DEBUG
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    std::string ptx(ptxSource);

    size_t sizeofLog = sizeof(log);

    checkOptix(optixModuleCreateFromPTX(
        context,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        ptx.c_str(),
        ptx.size(),
        log,
        &sizeofLog,
        &module
    ));
}

static void createProgramGroup(
    OptixDeviceContext &context,
    OptixProgramGroup &raygenProgramGroup,
    OptixProgramGroup &missProgramGroup,
    OptixProgramGroup &hitgroupProgramGroup,
    OptixModule &module,
    char *log
) {
    OptixProgramGroupOptions programGroupOptions = {}; // Initialize to zeros

    OptixProgramGroupDesc raygenProgramGroupDesc = {};
    raygenProgramGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenProgramGroupDesc.raygen.module = module;
    raygenProgramGroupDesc.raygen.entryFunctionName = "__raygen__rg";
    size_t sizeofLog = sizeof(log);
    checkOptix(optixProgramGroupCreate(
        context,
        &raygenProgramGroupDesc,
        1,   // num program groups
        &programGroupOptions,
        log,
        &sizeofLog,
        &raygenProgramGroup
    ));

    OptixProgramGroupDesc missProgramGroupDesc = {};
    missProgramGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missProgramGroupDesc.miss.module = module;
    missProgramGroupDesc.miss.entryFunctionName = "__miss__ms";
    sizeofLog = sizeof(log);
    checkOptix(optixProgramGroupCreate(
        context,
        &missProgramGroupDesc,
        1,   // num program groups
        &programGroupOptions,
        log,
        &sizeofLog,
        &missProgramGroup
    ));

    OptixProgramGroupDesc hitgroupProgramGroupDesc = {};
    hitgroupProgramGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupProgramGroupDesc.hitgroup.moduleCH = module;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    sizeofLog = sizeof(log);
    checkOptix(optixProgramGroupCreate(
        context,
        &hitgroupProgramGroupDesc,
        1,   // num program groups
        &programGroupOptions,
        log,
        &sizeofLog,
        &hitgroupProgramGroup
    ));
}

static void linkPipeline(
    OptixDeviceContext &context,
    OptixPipeline &pipeline,
    OptixPipelineCompileOptions &pipelineCompileOptions,
    OptixProgramGroup &raygenProgramGroup,
    OptixProgramGroup &missProgramGroup,
    OptixProgramGroup &hitgroupProgramGroup,
    char *log
) {
    const uint32_t maxTraceDepth  = 1;
    OptixProgramGroup programGroups[] = {
        raygenProgramGroup,
        missProgramGroup,
        hitgroupProgramGroup
    };

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = maxTraceDepth;
    pipelineLinkOptions.debugLevel  = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    size_t sizeofLog = sizeof(log);
    checkOptix(optixPipelineCreate(
        context,
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups,
        sizeof(programGroups) / sizeof(programGroups[0]),
        log,
        &sizeofLog,
        &pipeline
    ));

    OptixStackSizes stackSizes = {};
    for(auto& prog_group : programGroups) {
        checkOptix(optixUtilAccumulateStackSizes(prog_group, &stackSizes));
    }

    uint32_t directCallableStackSizeFromTraversal;
    uint32_t directCallableStackSizeFromState;
    uint32_t continuationStackSize;
    checkOptix(optixUtilComputeStackSizes(
        &stackSizes,
        maxTraceDepth,
        0,  // maxCCDepth
        0,  // maxDCDEpth
        &directCallableStackSizeFromTraversal,
        &directCallableStackSizeFromState, &continuationStackSize
    ));
    checkOptix(optixPipelineSetStackSize(
        pipeline,
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState,
        continuationStackSize,
        1  // maxTraversableDepth
    ));
}

static void setupShaderBindingTable(
    OptixShaderBindingTable &sbt,
    OptixProgramGroup &raygenProgramGroup,
    OptixProgramGroup &missProgramGroup,
    OptixProgramGroup &hitgroupProgramGroup,
    const SceneData &sceneData
) {
    CUdeviceptr raygenRecord;
    const size_t raygenRecordSize = sizeof(RayGenSbtRecord);
    checkCUDA(cudaMalloc(reinterpret_cast<void **>(&raygenRecord), raygenRecordSize));

    RayGenSbtRecord raygenSbt;
    checkOptix(optixSbtRecordPackHeader(raygenProgramGroup, &raygenSbt));
    checkCUDA(cudaMemcpy(
        reinterpret_cast<void *>(raygenRecord),
        &raygenSbt,
        raygenRecordSize,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr missRecord;
    size_t missRecordSize = sizeof(MissSbtRecord);
    checkCUDA(cudaMalloc(reinterpret_cast<void **>(&missRecord), missRecordSize));

    MissSbtRecord missSbt;
    checkOptix(optixSbtRecordPackHeader(missProgramGroup, &missSbt));
    checkCUDA(cudaMemcpy(
        reinterpret_cast<void *>(missRecord),
        &missSbt,
        missRecordSize,
        cudaMemcpyHostToDevice
    ));

    const std::vector<Material> &materials = sceneData.materials;
    std::vector<HitGroupSbtRecord> hitgroupRecords;
    hitgroupRecords.reserve(materials.size());
    int i = 0;
    for (const auto &material : materials) {
        HitGroupSbtRecord record;
        checkOptix(optixSbtRecordPackHeader(
            hitgroupProgramGroup,
            &record
        ));

        record.data.materialIndex = i++;

        hitgroupRecords.push_back(record);
    }

    CUdeviceptr d_hitgroupRecords;
    size_t hitgroupRecordSize = sizeof(HitGroupSbtRecord);
    checkCUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_hitgroupRecords),
        hitgroupRecordSize * materials.size()
    ));

    checkCUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_hitgroupRecords),
        hitgroupRecords.data(),
        hitgroupRecordSize * materials.size(),
        cudaMemcpyHostToDevice
    ));

    sbt.raygenRecord = raygenRecord;
    sbt.missRecordBase = missRecord;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = d_hitgroupRecords;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount = materials.size();
}

void Optix::updateMaterials(const Scene &scene)
{
    const Material *materials = scene.getMaterialsData();
    const size_t materialsSizeInBytes = scene.getMaterialsSize();
    checkCUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_materials),
        materials,
        materialsSizeInBytes,
        cudaMemcpyHostToDevice
    ));
}

void Optix::init(int width, int height, const Scene &scene)
{
    char log[2048];

    OptixDeviceContext context = nullptr;
    initContext(context);

    OptixTraversableHandle gasHandle;
    CUdeviceptr d_gasOutputBuffer;
    initAcceleration(context, gasHandle, d_gasOutputBuffer, scene.getSceneData());

    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    createModule(context, module, pipelineCompileOptions, log);

    OptixProgramGroup raygenProgramGroup = nullptr;
    OptixProgramGroup missProgramGroup = nullptr;
    OptixProgramGroup hitgroupProgramGroup = nullptr;
    createProgramGroup(
        context,
        raygenProgramGroup,
        missProgramGroup,
        hitgroupProgramGroup,
        module,
        log
    );

    OptixPipeline pipeline = nullptr;
    linkPipeline(
        context,
        pipeline,
        pipelineCompileOptions,
        raygenProgramGroup,
        missProgramGroup,
        hitgroupProgramGroup,
        log
    );

    OptixShaderBindingTable sbt = {};
    setupShaderBindingTable(
        sbt,
        raygenProgramGroup,
        missProgramGroup,
        hitgroupProgramGroup,
        scene.getSceneData()
    );

    // TODO: Refactor
    checkCUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_passRadiances),
        width * height * sizeof(Vec3)
    ));

    checkCUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_radiances),
        width * height * sizeof(Vec3)
    ));

    checkCUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_image),
        width * height * sizeof(uchar4)
    ));

   const size_t materialsSizeInBytes = scene.getMaterialsSize();
    checkCUDA(cudaMalloc(reinterpret_cast<void **>(&d_materials), materialsSizeInBytes));
    updateMaterials(scene);

    const std::vector<Triangle> &triangles = scene.getSceneData().triangles;
    Triangle *d_triangles = 0;
    const size_t trianglesSizeInBytes = triangles.size() * sizeof(Triangle);
    checkCUDA(cudaMalloc(reinterpret_cast<void **>(&d_triangles), trianglesSizeInBytes));
    checkCUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_triangles),
        triangles.data(),
        trianglesSizeInBytes,
        cudaMemcpyHostToDevice
    ));

    m_params.passRadiances = d_passRadiances;
    m_params.launchCount = 0;
    m_params.width = width;
    m_params.height = height;
    m_params.camera = scene.getCamera();
    m_params.materials = d_materials;
    m_params.primitives = d_triangles;
    m_params.handle = gasHandle;

    checkCUDA(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params)));

    m_width = width;
    m_height = height;
    m_gasHandle = gasHandle;
    m_pipeline = pipeline;
    m_sbt = sbt;
}

static uchar4 *launchOptix(
    Params &params,
    int width, int height,
    int currentSamples,
    OptixTraversableHandle &gasHandle,
    OptixPipeline &pipeline,
    OptixShaderBindingTable &sbt,
    uchar4 *d_image,
    Vec3 *d_passRadiances,
    Vec3 *d_radiances,
    CUdeviceptr d_param
) {
    CUstream stream;
    checkCUDA(cudaStreamCreate(&stream));

    checkCUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_param),
        &params,
        sizeof(params),
        cudaMemcpyHostToDevice
    ));

    checkOptix(optixLaunch(
        pipeline,
        stream,
        d_param,
        sizeof(Params),
        &sbt,
        width,
        height,
        /*depth=*/1
    ));

    updateFramebuffer(
        d_image,
        d_passRadiances,
        d_radiances,
        1,
        currentSamples,
        width,
        height,
        stream
    );

    checkCUDA(cudaDeviceSynchronize());

    uchar4 *image = (uchar4 *)malloc(
        width * height * sizeof(uchar4)
    );
    checkCUDA(cudaMemcpy(
        reinterpret_cast<void *>(image),
        d_image,
        width * height * sizeof(uchar4),
        cudaMemcpyDeviceToHost
    ));

    params.launchCount += 1;

    return image;
}

uchar4 *Optix::launch(int currentSamples)
{
    uchar4 *image = launchOptix(
        m_params,
        m_width,
        m_height,
        currentSamples,
        m_gasHandle,
        m_pipeline,
        m_sbt,
        d_image,
        d_passRadiances,
        d_radiances,
        d_param
    );

    return image;
}

}
