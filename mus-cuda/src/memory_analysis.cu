#include "mus_cuda.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

struct MemoryAnalysis {
    std::string name;
    size_t bytes;
    float gb;
    std::string category;
};

void analyze_memory_usage(const MUSConfig& cfg, const char* model_name) {
    printf("\n=== Memory Analysis for %s ===\n", model_name);
    printf("Model config: D=%d, L=%d, H=%d, FF=%d\n", 
           cfg.get_D(), cfg.get_L(), cfg.get_H(), cfg.get_D_ff());
    printf("Vision: %s, Audio: %s, Cross-attention: %s\n",
           cfg.enable_vision ? "enabled" : "disabled",
           cfg.enable_audio ? "enabled" : "disabled", 
           cfg.enable_cross_attention ? "enabled" : "disabled");
    
    std::vector<MemoryAnalysis> analysis;
    
    // Model weights
    int total_params = mus_total_params(cfg);
    size_t weights_bytes = (size_t)total_params * sizeof(half);
    size_t gradients_bytes = (size_t)total_params * sizeof(half);
    size_t adam_m_bytes = (size_t)total_params * sizeof(float);  // FP32 for stability
    size_t adam_v_bytes = (size_t)total_params * sizeof(float);
    
    analysis.push_back({"Weights", weights_bytes, weights_bytes / (1024.0f * 1024.0f * 1024.0f), "Model"});
    analysis.push_back({"Gradients", gradients_bytes, gradients_bytes / (1024.0f * 1024.0f * 1024.0f), "Model"});
    analysis.push_back({"Adam Momentum", adam_m_bytes, adam_m_bytes / (1024.0f * 1024.0f * 1024.0f), "Optimizer"});
    analysis.push_back({"Adam Variance", adam_v_bytes, adam_v_bytes / (1024.0f * 1024.0f * 1024.0f), "Optimizer"});
    
    // RMSNorm (FP32) - minimal overhead
    int D = cfg.get_D();
    int L = cfg.get_L();
    size_t norm_bytes = (size_t)L * D * 2 * sizeof(float) + D * sizeof(float);
    analysis.push_back({"RMSNorm Weights", norm_bytes, norm_bytes / (1024.0f * 1024.0f * 1024.0f), "Model"});
    
    // Workspace
    size_t workspace_bytes = 512 * 1024 * 1024; // 512MB
    analysis.push_back({"Workspace", workspace_bytes, workspace_bytes / (1024.0f * 1024.0f * 1024.0f), "Runtime"});
    
    // Multimodal features
    if (cfg.enable_vision) {
        int vision_vocab = 1000;
        size_t vision_bytes = vision_vocab * D * sizeof(half);  // Embeddings
        vision_bytes += D * cfg.vision_feature_dim * sizeof(half);  // Projections
        vision_bytes += L * D * 4 * D * sizeof(half);  // Cross-attention
        analysis.push_back({"Vision Features", vision_bytes, vision_bytes / (1024.0f * 1024.0f * 1024.0f), "Multimodal"});
    }
    
    if (cfg.enable_audio) {
        int audio_vocab = 500;
        size_t audio_bytes = audio_vocab * D * sizeof(half);  // Embeddings
        audio_bytes += D * cfg.audio_feature_dim * sizeof(half);  // Projections
        audio_bytes += L * D * 4 * D * sizeof(half);  // Cross-attention
        analysis.push_back({"Audio Features", audio_bytes, audio_bytes / (1024.0f * 1024.0f * 1024.0f), "Multimodal"});
    }
    
    if (cfg.enable_cross_attention && cfg.cross_attention_layers > 0) {
        size_t cross_bytes = cfg.cross_attention_layers * 4 * D * D * sizeof(half);
        analysis.push_back({"Cross-Attention", cross_bytes, cross_bytes / (1024.0f * 1024.0f * 1024.0f), "Multimodal"});
    }
    
    // Activations (B=1, S=256)
    int B = 1, S = 256;
    size_t activation_bytes = (size_t)B * S * D * sizeof(half);
    analysis.push_back({"Activations", activation_bytes, activation_bytes / (1024.0f * 1024.0f * 1024.0f), "Runtime"});
    
    // Calculate totals
    size_t total_model = 0, total_optimizer = 0, total_runtime = 0, total_multimodal = 0;
    
    for (const auto& item : analysis) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  " << std::left << std::setw(20) << item.name 
                  << " " << std::setw(8) << item.gb << "GB "
                  << "[" << item.category << "]" << std::endl;
        
        if (item.category == "Model") total_model += item.bytes;
        else if (item.category == "Optimizer") total_optimizer += item.bytes;
        else if (item.category == "Runtime") total_runtime += item.bytes;
        else if (item.category == "Multimodal") total_multimodal += item.bytes;
    }
    
    size_t total = total_model + total_optimizer + total_runtime + total_multimodal;
    
    std::cout << "\n" << std::string(50, '-') << std::endl;
    std::cout << "  " << std::left << std::setw(20) << "Model Total" << " " << std::setw(8) << (total_model / (1024.0f * 1024.0f * 1024.0f)) << "GB" << std::endl;
    std::cout << "  " << std::left << std::setw(20) << "Optimizer Total" << " " << std::setw(8) << (total_optimizer / (1024.0f * 1024.0f * 1024.0f)) << "GB" << std::endl;
    std::cout << "  " << std::left << std::setw(20) << "Runtime Total" << " " << std::setw(8) << (total_runtime / (1024.0f * 1024.0f * 1024.0f)) << "GB" << std::endl;
    std::cout << "  " << std::left << std::setw(20) << "Multimodal Total" << " " << std::setw(8) << (total_multimodal / (1024.0f * 1024.0f * 1024.0f)) << "GB" << std::endl;
    std::cout << "  " << std::left << std::setw(20) << "GRAND TOTAL" << " " << std::setw(8) << (total / (1024.0f * 1024.0f * 1024.0f)) << "GB" << std::endl;
    
    // Optimization recommendations
    printf("\n=== Optimization Recommendations ===\n");
    size_t optimized_usage = mus_vram_usage_optimized(cfg);
    printf("Optimized usage (FP32 Adam, reduced workspace): %.1fGB\n", 
           optimized_usage / (1024.0f * 1024.0f * 1024.0f));
    
    if (optimized_usage > 8 * 1024 * 1024 * 1024) {
        printf("⚠️  Still over 8GB limit. Consider:\n");
        printf("  - Disable audio support (-1.0GB)\n");
        printf("  - Disable cross-attention (-0.5GB)\n");
        printf("  - Use gradient checkpointing (-0.3GB)\n");
        printf("  - Reduce vision feature dim (-0.2GB)\n");
    }
    
    // Test different configurations
    printf("\n=== Alternative Configurations ===\n");
    
    MUSConfig test_cfg = get_1b_optimized_config();
    printf("1B Optimized: %.1fGB\n", mus_vram_usage_optimized(test_cfg) / (1024.0f * 1024.0f * 1024.0f));
    
    test_cfg = get_800m_config();
    printf("800M Optimized: %.1fGB\n", mus_vram_usage_optimized(test_cfg) / (1024.0f * 1024.0f * 1024.0f));
    
    test_cfg = get_500m_config();
    printf("500M Optimized: %.1fGB\n", mus_vram_usage_optimized(test_cfg) / (1024.0f * 1024.0f * 1024.0f));

    test_cfg = get_400m_4gb_config();
    printf("400M 4GB: %.1fGB ", mus_vram_usage_optimized(test_cfg) / (1024.0f * 1024.0f * 1024.0f));
    printf("(params=%.0fM)\n", mus_total_params(test_cfg) / 1e6f);

    test_cfg = get_700m_6gb_config();
    printf("700M 6GB: %.1fGB ", mus_vram_usage_optimized(test_cfg) / (1024.0f * 1024.0f * 1024.0f));
    printf("(params=%.0fM)\n", mus_total_params(test_cfg) / 1e6f);
}

int main(int argc, char** argv) {
    printf("MUS Memory Analysis Tool\n");
    printf("Analyzing VRAM usage for different model configurations\n");
    
    if (argc > 1) {
        std::string config = argv[1];
        
        if (config == "1b") {
            MUSConfig cfg = get_1b_config();
            analyze_memory_usage(cfg, "1B Full");
        } else if (config == "1b_opt") {
            MUSConfig cfg = get_1b_optimized_config();
            analyze_memory_usage(cfg, "1B Optimized");
        } else if (config == "800m") {
            MUSConfig cfg = get_800m_config();
            analyze_memory_usage(cfg, "800M Optimized");
        } else if (config == "500m") {
            MUSConfig cfg = get_500m_config();
            analyze_memory_usage(cfg, "500M Optimized");
        } else if (config == "400m") {
            MUSConfig cfg = get_400m_4gb_config();
            analyze_memory_usage(cfg, "400M 4GB");
        } else if (config == "700m") {
            MUSConfig cfg = get_700m_6gb_config();
            analyze_memory_usage(cfg, "700M 6GB");
        } else {
            printf("Unknown config: %s\n", config.c_str());
            printf("Available configs: 1b, 1b_opt, 800m, 500m, 400m, 700m\n");
            return 1;
        }
    } else {
        printf("Analyzing all configurations:\n\n");
        
        analyze_memory_usage(get_1b_config(), "1B Full");
        printf("\n");
        analyze_memory_usage(get_1b_optimized_config(), "1B Optimized");
        printf("\n");
        analyze_memory_usage(get_800m_config(), "800M Optimized");
        printf("\n");
        analyze_memory_usage(get_500m_config(), "500M Optimized");
        printf("\n");
        analyze_memory_usage(get_400m_4gb_config(), "400M 4GB");
        printf("\n");
        analyze_memory_usage(get_700m_6gb_config(), "700M 6GB");
    }
    
    return 0;
}