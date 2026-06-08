#include "training_session.h"

#include "config.h"
#include "perf_timer.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

std::map<std::string, std::string> TrainingSession::create_spirv_path_map(const std::string& spirv_dir) {
    std::vector<std::string> spirv_paths = {
        "projection_forward",
        "generate_keys",
        "compute_tile_ranges",
        "rasterize_forward",
        "rasterize_backward_0",
        "rasterize_backward_1",
        "rasterize_backward_2",
        "rasterize_backward_3",
        "rasterize_backward_4",
        "cumsum_single_pass",
        "cumsum_block_scan",
        "cumsum_scan_block_sums",
        "cumsum_add_block_offsets",
        "radix_sort/upsweep",
        "radix_sort/spine",
        "radix_sort/downsweep",
        "ssim_forward",
        "ssim_backward",
        "fused_projection_backward_optimizer",
        "sum",
        "where",
        "default_update_state",
        "default_compute_grow_mask",
        "default_duplicate",
        "default_split",
        "default_compute_prune_mask",
        "default_prune",
        "default_prune_mean",
        "default_prune_sh",
        "default_reset_opa",
        "mcmc_inject_noise",
        "mcmc_compute_probs",
        "mcmc_compute_relocation_index_map",
        "mcmc_compute_relocation",
        "mcmc_update_relocation",
        "mcmc_compute_add_index_map",
        "mcmc_compute_add",
        "mcmc_update_add",
        "morton_sort_compute_stats",
        "morton_sort_generate_keys",
        "morton_sort_apply_indices",
        "morton_sort_apply_indices_sh",
        "morton_sort_update_buffer",
        "morton_sort_update_buffer_sh",
    };

    std::map<std::string, std::string> spirv_paths_dict;
    for (const std::string& name : spirv_paths) {
        spirv_paths_dict[name] =
            (name.find('/') == std::string::npos) ?
            spirv_dir + "generated/" + name + ".spv" :
            spirv_dir + name + ".spv";
    }
    return spirv_paths_dict;
}

void TrainingSession::initialize(const std::string& spirv_dir, int device_id) {
    buffers.num_splats = 0;
    trainer.initialize(create_spirv_path_map(spirv_dir), device_id);
    PerfTimer::reset();
}

void TrainingSession::set_train_config(const TrainerConfig& train_config) {
    config = train_config;
    trainer.load_colmap_dataset(config, buffers);
    trainer.get_train_camera(0, uniforms);
    uniforms.active_sh = 3;
}

size_t TrainingSession::num_train() const {
    return trainer.num_train();
}

size_t TrainingSession::num_val() const {
    return trainer.num_val();
}

VulkanGSTrainer::DatasetImage& TrainingSession::get_train_image(size_t idx) {
    return trainer.get_train_image(idx);
}

VulkanGSTrainer::DatasetImage& TrainingSession::get_val_image(size_t idx) {
    return trainer.get_val_image(idx);
}

std::map<std::string, std::string> TrainingSession::get_train_image_path(size_t idx) {
    auto image = trainer.get_train_image(idx);
    std::map<std::string, std::string> result;
    result["image_path"] = image.image_path;
    if (!image.mask_path.empty())
        result["mask_path"] = image.mask_path;
    return result;
}

std::map<std::string, std::string> TrainingSession::get_val_image_path(size_t idx) {
    auto image = trainer.get_val_image(idx);
    std::map<std::string, std::string> result;
    result["image_path"] = image.image_path;
    if (!image.mask_path.empty())
        result["mask_path"] = image.mask_path;
    return result;
}

glm::mat4 TrainingSession::get_dataparser_transform() const {
    return trainer.get_dataparser_transform();
}

std::map<std::string, std::variant<uint32_t, std::vector<uint32_t>, bool, std::string>> TrainingSession::get_device_info() const {
    return trainer.get_device_info();
}

void TrainingSession::set_uniforms(
    uint32_t active_sh,
    const std::array<float, 16>& row_major_world_view_transform,
    uint32_t image_height,
    uint32_t image_width,
    float fx,
    float fy,
    float cx,
    float cy,
    bool is_fisheye
) {
    uniforms.active_sh = active_sh;
    uniforms.image_height = image_height;
    uniforms.image_width = image_width;
    uniforms.camera_model = is_fisheye ? 2 : 0;
    uniforms.fx = fx;
    uniforms.fy = fy;
    uniforms.cx = cx;
    uniforms.cy = cy;
    uniforms.grid_height = _CEIL_DIV(image_height, TILE_HEIGHT);
    uniforms.grid_width = _CEIL_DIV(image_width, TILE_WIDTH);

    // Convert Python/viewer row-major matrices to the renderer's column-major layout.
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            uniforms.world_view_transform[4*i+j] = row_major_world_view_transform[4*j+i];

    uniforms.step = 0;
}

void TrainingSession::set_train_image(size_t train_idx) {
    trainer.get_train_camera(train_idx, uniforms);
}

void TrainingSession::projection_forward() {
    uniforms.num_splats = static_cast<uint32_t>(buffers.num_splats);
    try {
        size_t alloc_reserve = config.strategy == TrainerConfig::Strategy::MCMC ?
            config.cap_max : 0;
        trainer.executeProjectionForward(uniforms, buffers, alloc_reserve);
    } catch (const std::runtime_error& err) {
        _THROW_ERROR(std::string(err.what()) + ". trainer.executeProjectionForward failed");
    }
}

void TrainingSession::process_tiles() {
    auto deviceGuard = DeviceGuard(&trainer);

    try {
        trainer.executeCalculateIndexBufferOffset(buffers);
    } catch (const std::runtime_error& err) {
        _THROW_ERROR(std::string(err.what()) + ". trainer.executeCalculateIndexBufferOffset failed");
    }
    if (buffers.num_indices == 0)
        return;

    try {
        trainer.executeGenerateKeys(uniforms, buffers);
    } catch (const std::runtime_error& err) {
        _THROW_ERROR(std::string(err.what()) + ". trainer.executeGenerateKeys failed");
    }

    try {
        trainer.executeSort(uniforms, buffers, -1);
    } catch (const std::runtime_error& err) {
        _THROW_ERROR(std::string(err.what()) + ". trainer.executeSort failed");
    }

    try {
        trainer.executeComputeTileRanges(uniforms, buffers);
    } catch (const std::runtime_error& err) {
        _THROW_ERROR(std::string(err.what()) + ". trainer.executeComputeTileRanges failed");
    }
}

void TrainingSession::rasterize_forward() {
    try {
        trainer.executeRasterizeForward(uniforms, buffers);
    } catch (const std::runtime_error& err) {
        _THROW_ERROR(std::string(err.what()) + ". trainer.executeRasterizeForward failed");
    }
}

void TrainingSession::rasterize_backward() {
    try {
        trainer.executeRasterizeBackward(uniforms, buffers);
    } catch (const std::runtime_error& err) {
        _THROW_ERROR(std::string(err.what()) + ". trainer.executeRasterizeBackward failed");
    }
}

void TrainingSession::forward() {
    auto deviceGuard = DeviceGuard(&trainer);
    projection_forward();
    process_tiles();
    rasterize_forward();
}

void TrainingSession::backward_optimize(int step) {
    auto deviceGuard = DeviceGuard(&trainer);

    rasterize_backward();

    try {
        trainer.executeFusedProjectionBackwardOptimizerStep(config, uniforms, buffers, step+1);
    } catch (const std::runtime_error& err) {
        _THROW_ERROR(std::string(err.what()) + ". trainer.executeFusedProjectionBackwardOptimizerStep failed");
    }
}

void TrainingSession::compute_pixel_loss_grad(size_t train_idx) {
    if (buffers.num_indices == 0)
        return;
    try {
        trainer.executeComputeSSIMGradient(config, uniforms, buffers, train_idx);
    } catch (const std::runtime_error& err) {
        _THROW_ERROR(std::string(err.what()) + ". trainer.executeComputeSSIMGradient failed");
    }
}

void TrainingSession::post_backward_step(int step) {
    if (config.strategy == TrainerConfig::Strategy::Default) {
        try {
            trainer.executeDefaultPostBackward(config, uniforms, buffers, step);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeDefaultPostBackward failed");
        }
    }
    else if (config.strategy == TrainerConfig::Strategy::MCMC) {
        try {
            trainer.executeMCMCPostBackward(config, uniforms, buffers, step);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeMCMCPostBackward failed");
        }
    }
}

void TrainingSession::train_step(size_t train_idx, int step) {
    set_train_image(train_idx);
    const int kShDegreeInterval = 1000;
    uniforms.active_sh = std::min(step / kShDegreeInterval, 3);
    uniforms.step = step;

    auto deviceGuard = DeviceGuard(&trainer);

    forward();
    compute_pixel_loss_grad(train_idx);
    backward_optimize(step);
    post_backward_step(step);
}

void TrainingSession::render_train(size_t idx) {
    trainer.get_train_camera(idx, uniforms);
    forward();
}

void TrainingSession::render_val(size_t idx) {
    trainer.get_val_camera(idx, uniforms);
    forward();
}

size_t TrainingSession::get_vram_usage() const {
    return buffers.getTotalAllocSize();
}

size_t TrainingSession::get_peak_vram_usage() const {
    return trainer.getPeakAllocSize();
}

std::map<std::string, std::tuple<size_t, double>> TrainingSession::get_timing_breakdown() const {
    return PerfTimer::get_summary();
}

std::map<std::string, size_t> TrainingSession::get_vram_breakdown() const {
    return buffers.getVramBreakdown();
}

size_t TrainingSession::num_splats() const {
    return buffers.num_splats;
}

void TrainingSession::write_ply(const std::string& filename) {
    trainer.writePLY(filename, buffers);
}

void TrainingSession::cleanup() {
    trainer.cleanupBuffers(buffers);
    trainer.cleanup();
}
