/**
 * Multi-Device Parallel Inference Runner
 * 
 * This tool allows running multiple models simultaneously on different devices
 * (CPU, GPU, IGPU) without interference.
 * 
 * Usage:
 *   multi-device-runner -c config.json
 *   multi-device-runner -m model.gguf -d cpu,gpu0,gpu1 -p "Hello"
 * 
 * Build:
 *   cd hardware-acceleration/tools
 *   g++ -O3 -std=c++17 -I../../ -I../../src -I../../ggml/include \
 *       multi-device-runner.cpp -o multi-device-runner \
 *       -L../../build/src -lllama -L../../build/ggml/src -lggml \
 *       -L../../build/ggml/src/ggml-cpu -lggml-cpu \
 *       -L../../build/ggml/src/ggml-metal -lggml-metal \
 *       -lpthread -ldl
 */

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <memory>
#include <functional>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "llama.h"
#include "common/common.h"
#include "common/sampling.h"

struct device_info {
    int index;
    std::string name;
    std::string type;
    size_t memory_total;
    size_t memory_free;
    ggml_backend_dev_t dev;
};

struct model_config {
    std::string name;
    std::string path;
    std::string device;
    std::string prompt;
    int n_predict;
    int n_ctx;
    int n_batch;
    int n_threads;
    float temperature;
    int n_gpu_layers;
};

struct inference_result {
    std::string model_name;
    std::string device_name;
    std::string output;
    double tokens_per_second;
    double total_time_ms;
    int tokens_generated;
    bool success;
    std::string error;
};

std::mutex print_mutex;
std::atomic<int> active_threads{0};

void print_thread_safe(const std::string& msg) {
    std::lock_guard<std::mutex> lock(print_mutex);
    std::cout << msg << std::flush;
}

std::vector<device_info> detect_devices() {
    std::vector<device_info> devices;
    
    int dev_count = ggml_backend_dev_count();
    for (int i = 0; i < dev_count; ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        
        device_info info;
        info.index = i;
        info.dev = dev;
        info.name = ggml_backend_dev_name(dev);
        
        auto dev_type = ggml_backend_dev_type(dev);
        switch (dev_type) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:
                info.type = "CPU";
                break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:
                info.type = "GPU";
                break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU:
                info.type = "IGPU";
                break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                info.type = "ACCEL";
                break;
            default:
                info.type = "UNKNOWN";
        }
        
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        info.memory_total = props.memory_total;
        info.memory_free = props.memory_free;
        
        devices.push_back(info);
    }
    
    return devices;
}

void print_devices(const std::vector<device_info>& devices) {
    printf("\n=== Detected Devices ===\n");
    printf("%-4s %-20s %-8s %12s %12s\n", "ID", "Name", "Type", "Total Mem", "Free Mem");
    printf("%-4s %-20s %-8s %12s %12s\n", "---", "----", "----", "---------", "--------");
    
    for (const auto& d : devices) {
        std::string mem_total = d.memory_total > 0 ? 
            std::to_string(d.memory_total / (1024*1024)) + " MB" : "N/A";
        std::string mem_free = d.memory_free > 0 ? 
            std::to_string(d.memory_free / (1024*1024)) + " MB" : "N/A";
        printf("%-4d %-20s %-8s %12s %12s\n", 
               d.index, d.name.c_str(), d.type.c_str(), 
               mem_total.c_str(), mem_free.c_str());
    }
    printf("\n");
}

ggml_backend_dev_t find_device(const std::vector<device_info>& devices, 
                                const std::string& device_spec) {
    if (device_spec == "cpu" || device_spec == "CPU") {
        for (const auto& d : devices) {
            if (d.type == "CPU") return d.dev;
        }
    }
    
    if (device_spec.substr(0, 3) == "gpu" || device_spec.substr(0, 3) == "GPU") {
        int gpu_idx = 0;
        if (device_spec.length() > 3) {
            gpu_idx = std::stoi(device_spec.substr(3));
        }
        int gpu_count = 0;
        for (const auto& d : devices) {
            if (d.type == "GPU" || d.type == "IGPU") {
                if (gpu_count == gpu_idx) return d.dev;
                gpu_count++;
            }
        }
    }
    
    for (const auto& d : devices) {
        if (d.name.find(device_spec) != std::string::npos) {
            return d.dev;
        }
    }
    
    return nullptr;
}

inference_result run_inference(const model_config& config, 
                               const std::vector<device_info>& devices) {
    inference_result result;
    result.model_name = config.name;
    result.success = false;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ggml_backend_dev_t target_dev = find_device(devices, config.device);
    if (!target_dev) {
        result.error = "Device not found: " + config.device;
        return result;
    }
    result.device_name = ggml_backend_dev_name(target_dev);
    
    llama_model_params mparams = llama_model_default_params();
    mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
    mparams.devices = &target_dev;
    mparams.n_gpu_layers = config.n_gpu_layers;
    
    llama_model* model = llama_model_load_from_file(config.path.c_str(), mparams);
    if (!model) {
        result.error = "Failed to load model: " + config.path;
        return result;
    }
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = config.n_ctx;
    cparams.n_batch = config.n_batch;
    cparams.n_threads = config.n_threads;
    cparams.n_seq_max = 1;
    
    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        result.error = "Failed to create context";
        llama_model_free(model);
        return result;
    }
    
    common_params_sampling sparams;
    sparams.temp = config.temperature;
    
    common_sampler* sampler = common_sampler_init(model, sparams);
    if (!sampler) {
        result.error = "Failed to create sampler";
        llama_free(ctx);
        llama_model_free(model);
        return result;
    }
    
    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, config.prompt, true, true);
    if (prompt_tokens.empty()) {
        result.error = "Failed to tokenize prompt";
        common_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        return result;
    }
    
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        result.error = "Failed to decode prompt";
        common_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        return result;
    }
    
    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::string output = config.prompt;
    int tokens_generated = 0;
    
    for (int i = 0; i < config.n_predict; ++i) {
        llama_token token = common_sampler_sample(sampler, ctx, batch.n_tokens - 1);
        common_sampler_accept(sampler, token, true);
        
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        
        output += common_token_to_piece(ctx, token);
        tokens_generated++;
        
        batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch) != 0) {
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    result.output = output;
    result.tokens_generated = tokens_generated;
    result.total_time_ms = total_time;
    result.tokens_per_second = tokens_generated > 0 ? 
        (tokens_generated * 1000.0 / total_time) : 0;
    result.success = true;
    
    common_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    
    return result;
}

void run_parallel_inference(const std::vector<model_config>& configs,
                            const std::vector<device_info>& devices,
                            std::vector<inference_result>& results) {
    std::vector<std::thread> threads;
    results.resize(configs.size());
    
    active_threads = (int)configs.size();
    
    for (size_t i = 0; i < configs.size(); ++i) {
        threads.emplace_back([&, i]() {
            print_thread_safe("[Thread " + std::to_string(i) + "] Starting model: " + 
                             configs[i].name + " on device: " + configs[i].device + "\n");
            
            results[i] = run_inference(configs[i], devices);
            
            std::string status = results[i].success ? 
                "Completed: " + std::to_string(results[i].tokens_per_second) + " tokens/sec" :
                "Failed: " + results[i].error;
            print_thread_safe("[Thread " + std::to_string(i) + "] " + status + "\n");
            
            active_threads--;
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
}

void print_results(const std::vector<inference_result>& results) {
    printf("\n=== Inference Results ===\n\n");
    
    for (const auto& r : results) {
        printf("Model: %s\n", r.model_name.c_str());
        printf("Device: %s\n", r.device_name.c_str());
        printf("Status: %s\n", r.success ? "SUCCESS" : "FAILED");
        
        if (r.success) {
            printf("Tokens Generated: %d\n", r.tokens_generated);
            printf("Total Time: %.2f ms\n", r.total_time_ms);
            printf("Speed: %.2f tokens/sec\n", r.tokens_per_second);
            printf("Output Preview: %.100s%s\n", 
                   r.output.c_str(), 
                   r.output.length() > 100 ? "..." : "");
        } else {
            printf("Error: %s\n", r.error.c_str());
        }
        printf("\n");
    }
    
    double total_tps = 0;
    int success_count = 0;
    for (const auto& r : results) {
        if (r.success) {
            total_tps += r.tokens_per_second;
            success_count++;
        }
    }
    printf("Total Throughput: %.2f tokens/sec (%d/%zu models successful)\n", 
           total_tps, success_count, results.size());
}

void print_usage(const char* prog) {
    printf("Multi-Device Parallel Inference Runner\n\n");
    printf("Usage:\n");
    printf("  %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -h, --help              Show this help\n");
    printf("  -l, --list-devices      List available devices\n");
    printf("  -m, --model FILE        Model path (can be specified multiple times)\n");
    printf("  -d, --device DEVICE     Target device: cpu, gpu0, gpu1, etc.\n");
    printf("  -p, --prompt TEXT       Prompt text\n");
    printf("  -n, --n-predict N       Number of tokens to predict (default: 128)\n");
    printf("  -t, --threads N         Number of threads (default: auto)\n");
    printf("  --temp FLOAT            Temperature (default: 0.8)\n");
    printf("  --n-gpu-layers N        GPU layers to offload (default: 99)\n");
    printf("  -b, --batch N           Batch size (default: 512)\n");
    printf("  --ctx N                 Context size (default: 4096)\n\n");
    printf("Examples:\n");
    printf("  %s -l                              # List devices\n", prog);
    printf("  %s -m model.gguf -d cpu -p \"Hi\"   # Run on CPU\n", prog);
    printf("  %s -m a.gguf -d cpu -m b.gguf -d gpu0  # Parallel on CPU and GPU\n", prog);
}

int main(int argc, char** argv) {
    std::vector<std::string> model_paths;
    std::vector<std::string> devices;
    std::string prompt = "Hello, how are you?";
    int n_predict = 128;
    int n_threads = 0;
    int n_batch = 512;
    int n_ctx = 4096;
    int n_gpu_layers = 99;
    float temperature = 0.8f;
    bool list_devices_only = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "-l" || arg == "--list-devices") {
            list_devices_only = true;
        }
        else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_paths.push_back(argv[++i]);
        }
        else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            devices.push_back(argv[++i]);
        }
        else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        }
        else if ((arg == "-n" || arg == "--n-predict") && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        }
        else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            n_threads = std::stoi(argv[++i]);
        }
        else if (arg == "--temp" && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        }
        else if (arg == "--n-gpu-layers" && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        }
        else if ((arg == "-b" || arg == "--batch") && i + 1 < argc) {
            n_batch = std::stoi(argv[++i]);
        }
        else if (arg == "--ctx" && i + 1 < argc) {
            n_ctx = std::stoi(argv[++i]);
        }
    }
    
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    
    auto detected_devices = detect_devices();
    
    if (list_devices_only) {
        print_devices(detected_devices);
        llama_backend_free();
        return 0;
    }
    
    std::vector<model_config> configs;
    
    if (model_paths.empty()) {
        fprintf(stderr, "No models specified. Use -m option.\n");
        print_usage(argv[0]);
        return 1;
    }
    
    if (n_threads == 0) {
#ifdef _WIN32
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        n_threads = (int)sysinfo.dwNumberOfProcessors;
#else
        n_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    }
    
    for (size_t i = 0; i < model_paths.size(); ++i) {
        model_config cfg;
        cfg.name = "model_" + std::to_string(i);
        cfg.path = model_paths[i];
        cfg.device = (i < devices.size()) ? devices[i] : "cpu";
        cfg.prompt = prompt;
        cfg.n_predict = n_predict;
        cfg.n_ctx = n_ctx;
        cfg.n_batch = n_batch;
        cfg.n_threads = n_threads;
        cfg.temperature = temperature;
        cfg.n_gpu_layers = n_gpu_layers;
        configs.push_back(cfg);
    }
    
    printf("=== Multi-Device Runner ===\n");
    printf("Models: %zu\n", configs.size());
    printf("Threads per model: %d\n\n", n_threads);
    print_devices(detected_devices);
    
    std::vector<inference_result> results;
    run_parallel_inference(configs, detected_devices, results);
    
    print_results(results);
    
    llama_backend_free();
    return 0;
}