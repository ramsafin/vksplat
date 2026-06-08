#include "training_session.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

// Hardcoded development defaults. Adjust these constants before running locally.
constexpr int kTrainDeviceId = -1;
constexpr int kTrainSteps = 30000;
constexpr int kProgressInterval = 100;

const std::filesystem::path kProjectDir = ".";
const std::filesystem::path kDatasetDir = R"(D:\vksplat\data\flowers)";
const std::filesystem::path kImageDir = "images_4";
const std::filesystem::path kSparseDir = "sparse/0";
const std::filesystem::path kMaskDir = "";
const std::filesystem::path kOutputDir = R"(D:\vksplat\output\cpp_train)";
const std::string kOutputPlyFilename = "splat.ply";

std::string with_trailing_separator(const std::filesystem::path& path) {
    if (path.empty())
        return "";

    std::string value = path.string();
    if (!value.empty() && value.back() != '/' && value.back() != '\\')
        value += std::filesystem::path::preferred_separator;
    return value;
}

std::string bytes_to_mib(size_t bytes) {
    return std::to_string(static_cast<double>(bytes) / (1024.0 * 1024.0)) + " MiB";
}

TrainerConfig make_default_config() {
    TrainerConfig config;

    config.output_dir = with_trailing_separator(kOutputDir);
    config.output_ply = (kOutputDir / kOutputPlyFilename).string();

    config.dataset_dir = with_trailing_separator(kDatasetDir);
    config.image_dir = with_trailing_separator(kDatasetDir / kImageDir);
    config.mask_dir = kMaskDir.empty() ? "" : with_trailing_separator(kDatasetDir / kMaskDir);
    config.sparse_dir = with_trailing_separator(kDatasetDir / kSparseDir);
    config.eval_interval = 8;

    config.image_cache_device = TrainerConfig::CacheImage::CPU;

    config.global_scale = 1.0f;
    config.init_scale = 1.0f;
    config.init_opacity = 0.1f;
    config.strategy = TrainerConfig::Strategy::Default;

    config.max_steps = kTrainSteps;
    config.ssim_lambda = 0.2f;
    config.means_lr = 1.6e-4f;
    config.means_lr_final = 1.6e-6f;
    config.features_dc_lr = 0.0025f;
    config.features_rest_lr = 0.0025f / 20.0f;
    config.opacities_lr = 0.05f;
    config.scales_lr = 0.005f;
    config.quats_lr = 0.001f;
    config.scale_reg = 0.0f;
    config.opacity_reg = 0.0f;

    config.refine_start_iter = 500;
    config.refine_stop_iter = 15000;
    config.refine_every = 100;

    config.prune_opa = 0.005f;
    config.grow_grad2d = 0.0002f;
    config.grow_scale3d = 0.01f;
    config.grow_scale2d = 0.05f;
    config.prune_scale3d = 0.1f;
    config.prune_scale2d = 0.15f;
    config.refine_scale2d_stop_iter = 0;
    config.reset_every = 3000;
    config.stop_reset_at = -1;
    config.pause_refine_after_reset = 0;

    config.noise_lr = 5e5f;
    config.min_opacity = 0.005f;
    config.grow_factor = 1.05f;
    config.cap_max = 1000000;

    return config;
}

} // namespace

int main() {
    TrainingSession session;
    bool initialized = false;

    try {
        const TrainerConfig config = make_default_config();
        const std::string spirv_dir = with_trailing_separator(kProjectDir / "shader");

        std::filesystem::create_directories(config.output_dir);

        std::cout << "Initializing VkSplat training session\n";
        std::cout << "Dataset: " << config.dataset_dir << '\n';
        std::cout << "Images:  " << config.image_dir << '\n';
        std::cout << "Sparse:  " << config.sparse_dir << '\n';
        std::cout << "Output:  " << config.output_dir << '\n';

        session.initialize(spirv_dir, kTrainDeviceId);
        initialized = true;
        session.set_train_config(config);

        const size_t num_train = session.num_train();
        if (num_train == 0)
            throw std::runtime_error("No training images were loaded");

        auto start = std::chrono::steady_clock::now();
        for (int step = 0; step < kTrainSteps; ++step) {
            session.train_step(static_cast<size_t>(step) % num_train, step);
            if (step == 0 || (step + 1) % kProgressInterval == 0 || step + 1 == kTrainSteps) {
                std::cout << "Step " << (step + 1) << '/' << kTrainSteps
                          << " | splats=" << session.num_splats()
                          << " | vram=" << bytes_to_mib(session.get_vram_usage())
                          << " | peak=" << bytes_to_mib(session.get_peak_vram_usage())
                          << '\n';
            }
        }
        auto stop = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(stop - start).count();

        std::cout << "Training complete\n";
        std::cout << "Elapsed time: " << elapsed << " seconds\n";
        std::cout << "Num splats: " << session.num_splats() << '\n';
        std::cout << "Current VRAM usage: " << bytes_to_mib(session.get_vram_usage()) << '\n';
        std::cout << "Peak VRAM usage: " << bytes_to_mib(session.get_peak_vram_usage()) << '\n';
        std::cout << "Writing PLY: " << config.output_ply << '\n';
        session.write_ply(config.output_ply);

        session.cleanup();
        initialized = false;
        return 0;
    } catch (const std::exception& err) {
        std::cerr << "vksplat_train failed: " << err.what() << '\n';
        if (initialized) {
            try {
                session.cleanup();
            } catch (const std::exception& cleanup_err) {
                std::cerr << "cleanup failed: " << cleanup_err.what() << '\n';
            }
        }
        return 1;
    }
}
