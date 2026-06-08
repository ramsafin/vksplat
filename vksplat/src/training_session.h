#pragma once

#include "gs_trainer.h"

#include <array>
#include <map>
#include <string>
#include <tuple>
#include <variant>

class TrainingSession {
public:
    VulkanGSTrainer trainer;
    VulkanGSRendererUniforms uniforms;
    VulkanGSPipelineBuffers buffers;
    TrainerConfig config;

    void initialize(const std::string& spirv_dir, int device_id);
    void set_train_config(const TrainerConfig& train_config);

    size_t num_train() const;
    size_t num_val() const;

    VulkanGSTrainer::DatasetImage& get_train_image(size_t idx);
    VulkanGSTrainer::DatasetImage& get_val_image(size_t idx);
    std::map<std::string, std::string> get_train_image_path(size_t idx);
    std::map<std::string, std::string> get_val_image_path(size_t idx);

    glm::mat4 get_dataparser_transform() const;
    std::map<std::string, std::variant<uint32_t, std::vector<uint32_t>, bool, std::string>> get_device_info() const;

    void set_uniforms(
        uint32_t active_sh,
        const std::array<float, 16>& row_major_world_view_transform,
        uint32_t image_height,
        uint32_t image_width,
        float fx,
        float fy,
        float cx,
        float cy,
        bool is_fisheye
    );

    void set_train_image(size_t train_idx);
    void projection_forward();
    void process_tiles();
    void rasterize_forward();
    void rasterize_backward();
    void forward();
    void backward_optimize(int step);
    void compute_pixel_loss_grad(size_t train_idx);
    void post_backward_step(int step);
    void train_step(size_t train_idx, int step);
    void render_train(size_t idx);
    void render_val(size_t idx);

    size_t get_vram_usage() const;
    size_t get_peak_vram_usage() const;
    std::map<std::string, std::tuple<size_t, double>> get_timing_breakdown() const;
    std::map<std::string, size_t> get_vram_breakdown() const;
    size_t num_splats() const;

    void write_ply(const std::string& filename);
    void cleanup();

private:
    static std::map<std::string, std::string> create_spirv_path_map(const std::string& spirv_dir);
};
