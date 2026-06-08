#include "training_session.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kProgressInterval = 100;

struct AppOptions {
    int train_device_id = -1;
    int train_steps = 30000;
    std::filesystem::path shader_dir = "shader";
    std::filesystem::path dataset_dir = R"(D:\vksplat\data\flowers)";
    std::filesystem::path image_dir = "images_4";
    std::filesystem::path sparse_dir = "sparse/0";
    std::filesystem::path mask_dir = "";
    std::filesystem::path output_dir = R"(D:\vksplat\output\cpp_train)";
    std::string output_ply_filename = "splat.ply";
};

void print_usage(const char* exe) {
    std::cout
        << "Usage: " << exe << " [options]\n"
        << "  --shader-dir PATH      Directory containing shader/ generated SPIR-V files (default: shader)\n"
        << "  --dataset-dir PATH     COLMAP dataset root\n"
        << "  --image-dir PATH       Image subdirectory under dataset root (default: images_4)\n"
        << "  --sparse-dir PATH      Sparse COLMAP subdirectory under dataset root (default: sparse/0)\n"
        << "  --mask-dir PATH        Optional mask subdirectory under dataset root\n"
        << "  --output-dir PATH      Output directory\n"
        << "  --output-ply NAME      Output PLY file name (default: splat.ply)\n"
        << "  --steps N              Number of training steps (default: 30000)\n"
        << "  --device ID            Vulkan device index (default: -1 auto)\n"
        << "  --help                 Show this help\n";
}

AppOptions parse_options(int argc, char** argv) {
    AppOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("Missing value for ") + name);
            return argv[++i];
        };

        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--shader-dir") {
            options.shader_dir = need_value("--shader-dir");
        } else if (arg == "--dataset-dir") {
            options.dataset_dir = need_value("--dataset-dir");
        } else if (arg == "--image-dir") {
            options.image_dir = need_value("--image-dir");
        } else if (arg == "--sparse-dir") {
            options.sparse_dir = need_value("--sparse-dir");
        } else if (arg == "--mask-dir") {
            options.mask_dir = need_value("--mask-dir");
        } else if (arg == "--output-dir") {
            options.output_dir = need_value("--output-dir");
        } else if (arg == "--output-ply") {
            options.output_ply_filename = need_value("--output-ply");
        } else if (arg == "--steps") {
            options.train_steps = std::stoi(need_value("--steps"));
        } else if (arg == "--device") {
            options.train_device_id = std::stoi(need_value("--device"));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return options;
}

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

TrainerConfig make_default_config(const AppOptions& options) {
    TrainerConfig config;

    config.output_dir = with_trailing_separator(options.output_dir);
    config.output_ply = (options.output_dir / options.output_ply_filename).string();

    config.dataset_dir = with_trailing_separator(options.dataset_dir);
    config.image_dir = with_trailing_separator(options.dataset_dir / options.image_dir);
    config.mask_dir = options.mask_dir.empty() ? "" : with_trailing_separator(options.dataset_dir / options.mask_dir);
    config.sparse_dir = with_trailing_separator(options.dataset_dir / options.sparse_dir);
    config.eval_interval = 8;

    config.image_cache_device = TrainerConfig::CacheImage::CPU;

    config.global_scale = 1.0f;
    config.init_scale = 1.0f;
    config.init_opacity = 0.1f;
    config.strategy = TrainerConfig::Strategy::Default;

    config.max_steps = options.train_steps;
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

int main(int argc, char** argv) {
    TrainingSession session;
    bool initialized = false;

    try {
        const AppOptions options = parse_options(argc, argv);
        const TrainerConfig config = make_default_config(options);
        const std::string spirv_dir = with_trailing_separator(options.shader_dir);

        std::filesystem::create_directories(config.output_dir);

        std::cout << "Initializing VkSplat training session\n";
        std::cout << "Dataset: " << config.dataset_dir << '\n';
        std::cout << "Images:  " << config.image_dir << '\n';
        std::cout << "Sparse:  " << config.sparse_dir << '\n';
        std::cout << "Output:  " << config.output_dir << '\n';

        session.initialize(spirv_dir, options.train_device_id);
        initialized = true;
        session.set_train_config(config);

        const size_t num_train = session.num_train();
        if (num_train == 0)
            throw std::runtime_error("No training images were loaded");

        auto start = std::chrono::steady_clock::now();
        for (int step = 0; step < options.train_steps; ++step) {
            session.train_step(static_cast<size_t>(step) % num_train, step);
            if (step == 0 || (step + 1) % kProgressInterval == 0 || step + 1 == options.train_steps) {
                std::cout << "Step " << (step + 1) << '/' << options.train_steps
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
