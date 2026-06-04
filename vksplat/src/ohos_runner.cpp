#include "gs_trainer.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <map>
#include <sstream>
#include <string>
#include <variant>

namespace {

std::string join_dir(std::string path) {
    if (!path.empty() && path.back() != '/' && path.back() != '\\')
        path.push_back('/');
    return path;
}

std::map<std::string, std::string> make_spirv_paths(const std::string& shader_dir, bool training) {
    const std::string root = join_dir(shader_dir);
    std::vector<std::string> names = {
        "projection_forward", "generate_keys", "compute_tile_ranges",
        "rasterize_forward", "rasterize_backward_0", "rasterize_backward_1",
        "rasterize_backward_2", "rasterize_backward_3", "rasterize_backward_4",
        "cumsum_single_pass", "cumsum_block_scan", "cumsum_scan_block_sums",
        "cumsum_add_block_offsets", "radix_sort/upsweep", "radix_sort/spine",
        "radix_sort/downsweep", "sum", "where",
    };
    if (training) {
        const std::vector<std::string> train_names = {
            "ssim_forward", "ssim_backward", "fused_projection_backward_optimizer",
            "default_update_state", "default_compute_grow_mask", "default_duplicate",
            "default_split", "default_compute_prune_mask", "default_prune",
            "default_prune_mean", "default_prune_sh", "default_reset_opa",
            "mcmc_inject_noise", "mcmc_compute_probs", "mcmc_compute_relocation_index_map",
            "mcmc_compute_relocation", "mcmc_update_relocation", "mcmc_compute_add_index_map",
            "mcmc_compute_add", "mcmc_update_add", "morton_sort_compute_stats",
            "morton_sort_generate_keys", "morton_sort_apply_indices", "morton_sort_apply_indices_sh",
            "morton_sort_update_buffer", "morton_sort_update_buffer_sh",
        };
        names.insert(names.end(), train_names.begin(), train_names.end());
    }

    std::map<std::string, std::string> paths;
    for (const std::string& name : names) {
        paths[name] = name.find('/') == std::string::npos ?
            root + "generated/" + name + ".spv" : root + name + ".spv";
    }
    return paths;
}

class ProbePipeline : public VulkanGSPipeline {
protected:
    DeviceRequirement getDeviceRequirement() override {
        return DeviceRequirement{{1, 1, 1}, {1, 1, 1}, 0};
    }
};

std::string json_value(const std::variant<uint32_t, std::vector<uint32_t>, bool, std::string>& value) {
    std::ostringstream os;
    std::visit([&](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, uint32_t>) {
            os << v;
        } else if constexpr (std::is_same_v<T, bool>) {
            os << (v ? "true" : "false");
        } else if constexpr (std::is_same_v<T, std::string>) {
            os << '"' << v << '"';
        } else {
            os << '[';
            for (size_t i = 0; i < v.size(); ++i) {
                if (i) os << ',';
                os << v[i];
            }
            os << ']';
        }
    }, value);
    return os.str();
}

void write_device_info(const std::string& filename, const std::map<std::string, std::variant<uint32_t, std::vector<uint32_t>, bool, std::string>>& info) {
    std::ofstream os(filename);
    os << "{\n";
    bool first = true;
    for (const auto& [key, value] : info) {
        if (!first) os << ",\n";
        first = false;
        os << "  \"" << key << "\": " << json_value(value);
    }
    os << "\n}\n";
}

std::string select_shader_dir(const std::string& shader_root,
                              const std::map<std::string, std::variant<uint32_t, std::vector<uint32_t>, bool, std::string>>& info) {
    const auto get_u32 = [&](const std::string& key) {
        return std::get<uint32_t>(info.at(key));
    };
    const auto get_bool = [&](const std::string& key) {
        return std::get<bool>(info.at(key));
    };

    const std::string root = join_dir(shader_root);
    if (get_u32("subgroup_size") == 64 &&
        get_u32("max_compute_work_group_invocations") >= 512 &&
        !get_bool("has_int64") &&
        !get_bool("has_shader_atomic_float_extension")) {
        const std::string maleoon = root + "ohos_maleoon_sg64_wg512_i64emul_f32atomicemul";
        if (std::filesystem::exists(maleoon))
            return maleoon;
    }
    return shader_root;
}

void run_synthetic_forward(const std::string& shader_dir, int device_id, const std::string& output_dir) {
    VulkanGSRenderer renderer;
    VulkanGSPipelineBuffers buffers;
    VulkanGSRendererUniforms uniforms{};

    renderer.initialize(make_spirv_paths(shader_dir, false), device_id);

    buffers.num_splats = 1;
    buffers.xyz_ws.assign({0.0f, 0.0f, 2.0f});
    buffers.rotations.assign({1.0f, 0.0f, 0.0f, 0.0f});
    buffers.scales_opacs.assign({0.1f, 0.1f, 0.1f, 0.8f});
    buffers.sh_coeffs.assign(16 * 3, 0.0f);
    buffers.sh_coeffs[0] = 1.0f;
    buffers.sh_coeffs[1] = 1.0f;
    buffers.sh_coeffs[2] = 1.0f;
    buffers.reorderSH(buffers.sh_coeffs);
    renderer.copyToDevice(buffers.xyz_ws);
    renderer.copyToDevice(buffers.rotations);
    renderer.copyToDevice(buffers.scales_opacs);
    renderer.copyToDevice(buffers.sh_coeffs);

    uniforms.image_height = 128;
    uniforms.image_width = 128;
    uniforms.grid_height = _CEIL_DIV(uniforms.image_height, TILE_HEIGHT);
    uniforms.grid_width = _CEIL_DIV(uniforms.image_width, TILE_WIDTH);
    uniforms.num_splats = (uint32_t)buffers.num_splats;
    uniforms.active_sh = 0;
    uniforms.camera_model = VulkanGSTrainer::Camera::PINHOLE;
    uniforms.fx = 80.0f;
    uniforms.fy = 80.0f;
    uniforms.cx = 64.0f;
    uniforms.cy = 64.0f;
    for (int i = 0; i < 16; ++i)
        uniforms.world_view_transform[i] = (i % 5 == 0) ? 1.0f : 0.0f;

    {
        auto guard = DeviceGuard(&renderer);
        renderer.executeProjectionForward(uniforms, buffers);
        renderer.executeCalculateIndexBufferOffset(buffers);
        if (buffers.num_indices != 0) {
            renderer.executeGenerateKeys(uniforms, buffers);
            renderer.executeSort(uniforms, buffers, -1);
            renderer.executeComputeTileRanges(uniforms, buffers);
            renderer.executeRasterizeForward(uniforms, buffers);
        }
    }

    renderer.copyFromDevice(buffers.pixel_state);
    std::filesystem::create_directories(output_dir);
    std::ofstream os(join_dir(output_dir) + "synthetic_pixel_state.f32", std::ios::binary);
    os.write(reinterpret_cast<const char*>(buffers.pixel_state.data()), (std::streamsize)buffers.pixel_state.byteLength());
    renderer.cleanupBuffers(buffers);
    renderer.cleanup();
}

TrainerConfig default_config(const std::string& dataset_dir, const std::string& output_dir, int steps) {
    TrainerConfig config{};
    config.output_dir = join_dir(output_dir);
    config.output_ply = join_dir(output_dir) + "splat.ply";
    config.dataset_dir = join_dir(dataset_dir);
    config.image_dir = join_dir(dataset_dir) + "images/";
    config.mask_dir = "";
    config.sparse_dir = join_dir(dataset_dir) + "sparse/0/";
    if (!std::filesystem::exists(config.sparse_dir))
        config.sparse_dir = join_dir(dataset_dir) + "sparse/";
    config.eval_interval = 8;
    config.image_cache_device = TrainerConfig::CacheImage::CPU;
    config.global_scale = 1.0f;
    config.init_scale = 1.0f;
    config.init_opacity = 0.1f;
    config.strategy = TrainerConfig::Strategy::Default;
    config.max_steps = steps;
    config.ssim_lambda = 0.2f;
    config.means_lr = 1.6e-4f;
    config.means_lr_final = 1.6e-6f;
    config.features_dc_lr = 0.0025f;
    config.features_rest_lr = 0.000125f;
    config.opacities_lr = 0.05f;
    config.scales_lr = 0.005f;
    config.quats_lr = 0.001f;
    config.scale_reg = 0.01f;
    config.opacity_reg = 0.01f;
    config.refine_start_iter = 500;
    config.refine_stop_iter = 25000;
    config.refine_every = 100;
    config.prune_opa = 0.005f;
    config.grow_grad2d = 0.0002f;
    config.grow_scale3d = 0.01f;
    config.grow_scale2d = 0.05f;
    config.prune_scale3d = 0.1f;
    config.prune_scale2d = 0.15f;
    config.refine_scale2d_stop_iter = 0;
    config.reset_every = 3000;
    config.stop_reset_at = 15000;
    config.pause_refine_after_reset = 0;
    config.noise_lr = 5e5f;
    config.min_opacity = 0.005f;
    config.grow_factor = 1.05f;
    config.cap_max = 1000000;
    return config;
}

void run_training(const std::string& shader_dir, int device_id, const std::string& dataset_dir, const std::string& output_dir, int steps) {
    VulkanGSTrainer trainer;
    VulkanGSPipelineBuffers buffers;
    VulkanGSRendererUniforms uniforms{};
    TrainerConfig config = default_config(dataset_dir, output_dir, steps);

    std::filesystem::create_directories(output_dir);
    trainer.initialize(make_spirv_paths(shader_dir, true), device_id);
    trainer.load_colmap_dataset(config, buffers);
    trainer.get_train_camera(0, uniforms);
    uniforms.active_sh = 0;

    for (int step = 0; step < steps; ++step) {
        trainer.get_train_camera((size_t)(step % trainer.num_train()), uniforms);
        uniforms.active_sh = std::min(step / 1000, 3);
        uniforms.step = step;
        auto guard = DeviceGuard(&trainer);
        trainer.executeProjectionForward(uniforms, buffers);
        trainer.executeCalculateIndexBufferOffset(buffers);
        if (buffers.num_indices != 0) {
            trainer.executeGenerateKeys(uniforms, buffers);
            trainer.executeSort(uniforms, buffers, -1);
            trainer.executeComputeTileRanges(uniforms, buffers);
            trainer.executeRasterizeForward(uniforms, buffers);
            trainer.executeComputeSSIMGradient(config, uniforms, buffers, (size_t)(step % trainer.num_train()));
            trainer.executeRasterizeBackward(uniforms, buffers);
            trainer.executeFusedProjectionBackwardOptimizerStep(config, uniforms, buffers, step + 1);
        }
        trainer.executeDefaultPostBackward(config, uniforms, buffers, step);
    }

    trainer.writePLY(config.output_ply, buffers);
    trainer.cleanupBuffers(buffers);
    trainer.cleanup();
}

} // namespace

int main(int argc, char** argv) {
    std::string mode = "probe";
    std::string shader_root = "/data/local/tmp/vksplat/shaders";
    std::string dataset_dir = "/data/local/tmp/vksplat/dataset";
    std::string output_dir = "/data/local/tmp/vksplat/output";
    int device_id = -1;
    int steps = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("Missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--mode") mode = need_value("--mode");
        else if (arg == "--shader-root") shader_root = need_value("--shader-root");
        else if (arg == "--dataset-dir") dataset_dir = need_value("--dataset-dir");
        else if (arg == "--output-dir") output_dir = need_value("--output-dir");
        else if (arg == "--device") device_id = std::stoi(need_value("--device"));
        else if (arg == "--steps") steps = std::stoi(need_value("--steps"));
        else throw std::runtime_error("Unknown argument: " + arg);
    }

    try {
        std::filesystem::create_directories(output_dir);
        ProbePipeline probe;
        probe.initialize(device_id);
        auto info = probe.get_device_info();
        write_device_info(join_dir(output_dir) + "device_info.json", info);
        std::string shader_dir = select_shader_dir(shader_root, info);
        std::cout << "Selected shader dir: " << shader_dir << std::endl;
        probe.cleanup();

        if (mode == "probe") return 0;
        if (mode == "forward") {
            run_synthetic_forward(shader_dir, device_id, output_dir);
            return 0;
        }
        if (mode == "train") {
            run_training(shader_dir, device_id, dataset_dir, output_dir, steps);
            return 0;
        }
        throw std::runtime_error("Unknown mode: " + mode);
    } catch (const std::exception& e) {
        std::cerr << "vksplat_ohos_runner failed: " << e.what() << std::endl;
        return 1;
    }
}
