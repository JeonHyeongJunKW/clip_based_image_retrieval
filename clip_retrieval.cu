#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ============================================================
// TensorRT Logger
// ============================================================
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << std::endl;
    }
};

static Logger gLogger;

// ============================================================
// Model type
// ============================================================
enum class ModelType { CLIP, DINOV2 };

struct NormParams {
    float mean[3];
    float std_val[3];
};

NormParams getNormParams(ModelType model) {
    if (model == ModelType::DINOV2) {
        // ImageNet normalization
        return {{0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f}};
    }
    // CLIP normalization
    return {{0.48145466f, 0.4578275f, 0.40821073f}, {0.26862954f, 0.26130258f, 0.27577711f}};
}

// ============================================================
// Fisheye (equidistant) undistortion for v4l2 images
// ============================================================
struct FisheyeUndistorter {
    cv::Mat K;
    cv::Mat D;
    cv::Mat map1, map2;
    cv::Size cached_size;
    bool enabled = false;

    void init() {
        K = (cv::Mat_<double>(3, 3) <<
             338.518619, 0.0,        317.804486,
             0.0,        337.361647, 176.025956,
             0.0,        0.0,        1.0);
        D = (cv::Mat_<double>(4, 1) <<
             -0.16211838, 0.18883822, -0.21422656, 0.10004908);
        enabled = true;
    }

    cv::Mat undistort(const cv::Mat& img) {
        if (img.size() != cached_size) {
            cv::fisheye::initUndistortRectifyMap(
                K, D, cv::Mat::eye(3, 3, CV_64F), K,
                img.size(), CV_16SC2, map1, map2);
            cached_size = img.size();
        }
        cv::Mat out;
        cv::remap(img, out, map1, map2, cv::INTER_LINEAR);
        return out;
    }
};

static FisheyeUndistorter gFisheye;

bool shouldUndistort(const std::string& path) {
    return gFisheye.enabled && path.find("v4l2") != std::string::npos;
}

cv::Mat loadImage(const std::string& path) {
    cv::Mat img = cv::imread(path);
    if (img.empty()) return img;
    if (shouldUndistort(path)) {
        return gFisheye.undistort(img);
    }
    return img;
}

// ============================================================
// Data structures
// ============================================================
struct ImageEntry {
    std::string path;
    std::vector<float> embedding;
};

struct MatchResult {
    int index;
    float similarity;
};

// ============================================================
// TensorRT engine loading
// ============================================================
nvinfer1::ICudaEngine* loadEngine(const std::string& path, nvinfer1::IRuntime* runtime) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "엔진 파일 열기 실패: " << path << std::endl;
        return nullptr;
    }
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    f.read(buf.data(), size);
    return runtime->deserializeCudaEngine(buf.data(), size);
}

// ============================================================
// Image preprocessing: BGR -> RGB, resize+center crop, normalize, NCHW fp16
// ============================================================
void preprocessImage(const cv::Mat& bgr, __half* output, int target_size, const NormParams& norm) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // Resize shortest edge to target_size, then center crop
    int h = rgb.rows, w = rgb.cols;
    int short_edge = std::min(h, w);
    double scale = static_cast<double>(target_size) / short_edge;
    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);
    cv::Mat scaled;
    cv::resize(rgb, scaled, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);

    int crop_x = (new_w - target_size) / 2;
    int crop_y = (new_h - target_size) / 2;
    cv::Mat resized = scaled(cv::Rect(crop_x, crop_y, target_size, target_size)).clone();

    int plane = target_size * target_size;
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < target_size; y++) {
            for (int x = 0; x < target_size; x++) {
                float pixel = resized.at<cv::Vec3b>(y, x)[c] / 255.0f;
                float normalized = (pixel - norm.mean[c]) / norm.std_val[c];
                output[c * plane + y * target_size + x] = __float2half(normalized);
            }
        }
    }
}

// ============================================================
// L2 normalization
// ============================================================
void l2Normalize(float* vec, int dim) {
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) norm += vec[i] * vec[i];
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (int i = 0; i < dim; i++) vec[i] /= norm;
    }
}

// ============================================================
// Dot product (cosine similarity for L2-normalized vectors)
// ============================================================
float dotProduct(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += a[i] * b[i];
    return sum;
}

// ============================================================
// Collect image paths from a folder
// ============================================================
std::vector<std::string> collectImagePaths(const std::string& folder) {
    std::vector<std::string> all_paths;
    const std::vector<std::string> extensions = {
        "/*.jpg", "/*.jpeg", "/*.png", "/*.bmp",
        "/*.JPG", "/*.JPEG", "/*.PNG", "/*.BMP"
    };

    for (const auto& ext : extensions) {
        std::vector<cv::String> files;
        cv::glob(folder + ext, files, false);
        for (const auto& f : files) {
            all_paths.push_back(f);
        }
    }

    std::sort(all_paths.begin(), all_paths.end());
    all_paths.erase(std::unique(all_paths.begin(), all_paths.end()), all_paths.end());

    return all_paths;
}

// ============================================================
// Embed a single image
// ============================================================
void embedImage(const cv::Mat& bgr,
                nvinfer1::IExecutionContext* context,
                void* d_input, void* d_output,
                int inputIdx, int outputIdx,
                int inputSizePx, int outputDim,
                std::vector<__half>& h_input,
                std::vector<__half>& h_output_half,
                float* out_embedding,
                const NormParams& norm) {
    size_t inputSize = 1 * 3 * inputSizePx * inputSizePx * sizeof(__half);
    size_t outputSize = 1 * outputDim * sizeof(__half);

    preprocessImage(bgr, h_input.data(), inputSizePx, norm);
    cudaMemcpy(d_input, h_input.data(), inputSize, cudaMemcpyHostToDevice);

    void* bindings[2];
    bindings[inputIdx] = d_input;
    bindings[outputIdx] = d_output;

    context->executeV2(bindings);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_half.data(), d_output, outputSize, cudaMemcpyDeviceToHost);
    for (int i = 0; i < outputDim; i++) {
        out_embedding[i] = __half2float(h_output_half[i]);
    }

    l2Normalize(out_embedding, outputDim);
}

// ============================================================
// Build database: embed all images in folder A
// ============================================================
std::vector<ImageEntry> buildDatabase(
    const std::vector<std::string>& paths,
    nvinfer1::IExecutionContext* context,
    void* d_input, void* d_output,
    int inputIdx, int outputIdx,
    int inputSizePx, int outputDim,
    const NormParams& norm) {

    std::vector<ImageEntry> database;
    database.reserve(paths.size());

    std::vector<__half> h_input(1 * 3 * inputSizePx * inputSizePx);
    std::vector<__half> h_output_half(outputDim);
    std::vector<float> embedding(outputDim);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < paths.size(); i++) {
        cv::Mat img = loadImage(paths[i]);
        if (img.empty()) {
            std::cerr << "이미지 로드 실패 (건너뜀): " << paths[i] << std::endl;
            continue;
        }

        embedImage(img, context, d_input, d_output,
                   inputIdx, outputIdx, inputSizePx, outputDim,
                   h_input, h_output_half, embedding.data(), norm);

        ImageEntry entry;
        entry.path = paths[i];
        entry.embedding = embedding;
        database.push_back(std::move(entry));

        if ((i + 1) % 50 == 0 || i == paths.size() - 1) {
            printf("  [%zu/%zu] 임베딩 완료\n", i + 1, paths.size());
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    printf("데이터베이스 구축 완료: %zu장, %.2f초\n", database.size(), elapsed);

    return database;
}

// ============================================================
// Find top-K matches
// ============================================================
std::vector<MatchResult> findTopK(
    const float* query_embedding,
    const std::vector<ImageEntry>& database,
    int dim, int k, float threshold) {

    std::vector<MatchResult> candidates;
    candidates.reserve(database.size());

    for (size_t i = 0; i < database.size(); i++) {
        float sim = dotProduct(query_embedding, database[i].embedding.data(), dim);
        if (sim >= threshold) {
            candidates.push_back({static_cast<int>(i), sim});
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const MatchResult& a, const MatchResult& b) {
                  return a.similarity > b.similarity;
              });

    if (static_cast<int>(candidates.size()) > k) {
        candidates.resize(k);
    }

    return candidates;
}

// ============================================================
// Resize image preserving aspect ratio to a target height
// ============================================================
cv::Mat resizeToHeight(const cv::Mat& img, int target_height) {
    double scale = static_cast<double>(target_height) / img.rows;
    int new_width = static_cast<int>(img.cols * scale);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_width, target_height), 0, 0, cv::INTER_LINEAR);
    return resized;
}

// ============================================================
// Extract filename from path
// ============================================================
std::string getFilename(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

// ============================================================
// Display query image + top-K results
// ============================================================
void displayResults(const cv::Mat& query_img,
                    const std::string& query_path,
                    const std::vector<MatchResult>& matches,
                    const std::vector<ImageEntry>& database,
                    float threshold) {
    const int DISPLAY_HEIGHT = 300;
    const int GAP = 10;
    const int TEXT_BAR_HEIGHT = 40;

    cv::Mat query_resized = resizeToHeight(query_img, DISPLAY_HEIGHT);

    std::vector<cv::Mat> match_images;
    for (const auto& m : matches) {
        cv::Mat img = loadImage(database[m.index].path);
        if (!img.empty()) {
            match_images.push_back(resizeToHeight(img, DISPLAY_HEIGHT));
        }
    }

    int total_width = query_resized.cols + GAP;
    for (const auto& m : match_images) {
        total_width += m.cols + GAP;
    }
    if (match_images.empty()) {
        total_width += 300;
    }
    int total_height = DISPLAY_HEIGHT + TEXT_BAR_HEIGHT;

    cv::Mat canvas(total_height, total_width, CV_8UC3, cv::Scalar(40, 40, 40));

    query_resized.copyTo(canvas(cv::Rect(0, 0, query_resized.cols, query_resized.rows)));

    cv::rectangle(canvas,
                  cv::Point(0, DISPLAY_HEIGHT),
                  cv::Point(query_resized.cols, total_height),
                  cv::Scalar(0, 0, 0), -1);
    std::string query_label = "QUERY: " + getFilename(query_path);
    cv::putText(canvas, query_label,
                cv::Point(5, DISPLAY_HEIGHT + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 255), 1);

    cv::rectangle(canvas,
                  cv::Point(0, 0),
                  cv::Point(query_resized.cols - 1, DISPLAY_HEIGHT - 1),
                  cv::Scalar(255, 255, 0), 2);

    int x_offset = query_resized.cols + GAP;

    if (matches.empty()) {
        cv::putText(canvas, "No matches above threshold",
                    cv::Point(x_offset + 10, DISPLAY_HEIGHT / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 0, 255), 2);
    } else {
        for (size_t i = 0; i < match_images.size(); i++) {
            const auto& mimg = match_images[i];
            const auto& m = matches[i];

            mimg.copyTo(canvas(cv::Rect(x_offset, 0, mimg.cols, mimg.rows)));

            cv::Scalar border_color = cv::Scalar(0, 255, 0);
            if (threshold > 0.0f && m.similarity < threshold) {
                border_color = cv::Scalar(0, 0, 255);
            }
            cv::rectangle(canvas,
                          cv::Point(x_offset, 0),
                          cv::Point(x_offset + mimg.cols - 1, DISPLAY_HEIGHT - 1),
                          border_color, 2);

            cv::rectangle(canvas,
                          cv::Point(x_offset, DISPLAY_HEIGHT),
                          cv::Point(x_offset + mimg.cols, total_height),
                          cv::Scalar(0, 0, 0), -1);

            char label_buf[128];
            snprintf(label_buf, sizeof(label_buf), "#%zu sim=%.4f", i + 1, m.similarity);
            cv::putText(canvas, label_buf,
                        cv::Point(x_offset + 5, DISPLAY_HEIGHT + 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(255, 255, 255), 1);

            std::string fname = getFilename(database[m.index].path);
            cv::putText(canvas, fname,
                        cv::Point(x_offset + 5, DISPLAY_HEIGHT + 33),
                        cv::FONT_HERSHEY_SIMPLEX, 0.35,
                        cv::Scalar(200, 200, 200), 1);

            x_offset += mimg.cols + GAP;
        }
    }

    cv::imshow("Image Retrieval", canvas);
}

// ============================================================
// Save result as image file (for headless environments)
// ============================================================
void saveResult(const cv::Mat& query_img,
                const std::string& query_path,
                const std::vector<MatchResult>& matches,
                const std::vector<ImageEntry>& database,
                float threshold,
                const std::string& save_dir,
                size_t query_index) {
    const int DISPLAY_HEIGHT = 300;
    const int GAP = 10;
    const int TEXT_BAR_HEIGHT = 40;

    cv::Mat query_resized = resizeToHeight(query_img, DISPLAY_HEIGHT);

    std::vector<cv::Mat> match_images;
    for (const auto& m : matches) {
        cv::Mat img = loadImage(database[m.index].path);
        if (!img.empty()) {
            match_images.push_back(resizeToHeight(img, DISPLAY_HEIGHT));
        }
    }

    int total_width = query_resized.cols + GAP;
    for (const auto& m : match_images) {
        total_width += m.cols + GAP;
    }
    if (match_images.empty()) {
        total_width += 300;
    }
    int total_height = DISPLAY_HEIGHT + TEXT_BAR_HEIGHT;

    cv::Mat canvas(total_height, total_width, CV_8UC3, cv::Scalar(40, 40, 40));

    query_resized.copyTo(canvas(cv::Rect(0, 0, query_resized.cols, query_resized.rows)));

    cv::rectangle(canvas,
                  cv::Point(0, DISPLAY_HEIGHT),
                  cv::Point(query_resized.cols, total_height),
                  cv::Scalar(0, 0, 0), -1);
    std::string query_label = "QUERY: " + getFilename(query_path);
    cv::putText(canvas, query_label,
                cv::Point(5, DISPLAY_HEIGHT + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 255), 1);

    cv::rectangle(canvas,
                  cv::Point(0, 0),
                  cv::Point(query_resized.cols - 1, DISPLAY_HEIGHT - 1),
                  cv::Scalar(255, 255, 0), 2);

    int x_offset = query_resized.cols + GAP;

    if (matches.empty()) {
        cv::putText(canvas, "No matches above threshold",
                    cv::Point(x_offset + 10, DISPLAY_HEIGHT / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 0, 255), 2);
    } else {
        for (size_t i = 0; i < match_images.size(); i++) {
            const auto& mimg = match_images[i];
            const auto& m = matches[i];

            mimg.copyTo(canvas(cv::Rect(x_offset, 0, mimg.cols, mimg.rows)));

            cv::Scalar border_color = cv::Scalar(0, 255, 0);
            if (threshold > 0.0f && m.similarity < threshold) {
                border_color = cv::Scalar(0, 0, 255);
            }
            cv::rectangle(canvas,
                          cv::Point(x_offset, 0),
                          cv::Point(x_offset + mimg.cols - 1, DISPLAY_HEIGHT - 1),
                          border_color, 2);

            cv::rectangle(canvas,
                          cv::Point(x_offset, DISPLAY_HEIGHT),
                          cv::Point(x_offset + mimg.cols, total_height),
                          cv::Scalar(0, 0, 0), -1);

            char label_buf[128];
            snprintf(label_buf, sizeof(label_buf), "#%zu sim=%.4f", i + 1, m.similarity);
            cv::putText(canvas, label_buf,
                        cv::Point(x_offset + 5, DISPLAY_HEIGHT + 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(255, 255, 255), 1);

            std::string fname = getFilename(database[m.index].path);
            cv::putText(canvas, fname,
                        cv::Point(x_offset + 5, DISPLAY_HEIGHT + 33),
                        cv::FONT_HERSHEY_SIMPLEX, 0.35,
                        cv::Scalar(200, 200, 200), 1);

            x_offset += mimg.cols + GAP;
        }
    }

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/result_%04zu.jpg", save_dir.c_str(), query_index);
    cv::imwrite(filename, canvas);
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "사용법: " << argv[0]
                  << " <engine_path> <folder_A> <folder_B> [옵션]"
                  << std::endl;
        std::cerr << "  engine_path : TensorRT 엔진 파일 경로" << std::endl;
        std::cerr << "  folder_A    : 데이터베이스 이미지 폴더 (검색 대상)" << std::endl;
        std::cerr << "  folder_B    : 쿼리 이미지 폴더" << std::endl;
        std::cerr << "\n옵션:" << std::endl;
        std::cerr << "  --model <dinov2|clip>  : 전처리 정규화 (기본: dinov2 = ImageNet norm)" << std::endl;
        std::cerr << "  --threshold <float>    : 코사인 유사도 최소값 (기본: 0.3)" << std::endl;
        std::cerr << "  --save-dir <path>      : 결과 이미지 저장 폴더 (imshow 대신 파일 저장)" << std::endl;
        std::cerr << "  --top-k <int>          : 검색 결과 개수 (기본: 3)" << std::endl;
        std::cerr << "  --undistort            : 경로에 'v4l2' 포함 이미지를 fisheye undistortion" << std::endl;
        return 1;
    }

    std::string engine_path = argv[1];
    std::string folder_a = argv[2];
    std::string folder_b = argv[3];
    float threshold = 0.3f;
    ModelType model_type = ModelType::DINOV2;
    std::string save_dir;
    int top_k = 3;

    for (int i = 4; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--threshold" && i + 1 < argc) {
            threshold = std::stof(argv[++i]);
        } else if (arg == "--model" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "dinov2" || m == "vpr" || m == "eigenplaces" || m == "cosplace") {
                model_type = ModelType::DINOV2; // ImageNet normalization
            } else if (m == "clip") {
                model_type = ModelType::CLIP;
            } else {
                std::cerr << "알 수 없는 모델: " << m << " (clip/dinov2/vpr/eigenplaces/cosplace)" << std::endl;
                return 1;
            }
        } else if (arg == "--save-dir" && i + 1 < argc) {
            save_dir = argv[++i];
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        } else if (arg == "--undistort") {
            gFisheye.init();
        }
    }

    NormParams norm = getNormParams(model_type);
    const char* model_name = (model_type == ModelType::DINOV2) ? "DINOv2" : "CLIP";

    printf("=== Image Retrieval ===\n");
    printf("모델: %s\n", model_name);
    printf("Fisheye undistortion (v4l2 경로): %s\n", gFisheye.enabled ? "ON" : "OFF");
    printf("엔진: %s\n", engine_path.c_str());
    printf("데이터베이스 폴더 (A): %s\n", folder_a.c_str());
    printf("쿼리 폴더 (B): %s\n", folder_b.c_str());
    printf("유사도 임계값: %.4f\n", threshold);
    printf("Top-K: %d\n", top_k);
    if (!save_dir.empty()) printf("결과 저장 폴더: %s\n", save_dir.c_str());
    printf("========================\n\n");

    // Create save directory if needed
    if (!save_dir.empty()) {
        std::string mkdir_cmd = "mkdir -p " + save_dir;
        system(mkdir_cmd.c_str());
    }

    // Collect image paths
    auto db_paths = collectImagePaths(folder_a);
    auto query_paths = collectImagePaths(folder_b);

    if (db_paths.empty()) {
        std::cerr << "데이터베이스 폴더에 이미지가 없습니다: " << folder_a << std::endl;
        return 1;
    }
    if (query_paths.empty()) {
        std::cerr << "쿼리 폴더에 이미지가 없습니다: " << folder_b << std::endl;
        return 1;
    }

    printf("데이터베이스 이미지: %zu장\n", db_paths.size());
    printf("쿼리 이미지: %zu장\n\n", query_paths.size());

    // Load TensorRT engine
    printf("TensorRT 엔진 로딩...\n");
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = loadEngine(engine_path, runtime);
    if (!engine) {
        std::cerr << "엔진 로드 실패" << std::endl;
        return 1;
    }

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // Auto-detect bindings: find input and output indices, input size, output dim
    int inputIdx = -1, outputIdx = -1;
    int outputDim = 0;
    int inputSizePx = 224;

    int nbBindings = engine->getNbBindings();
    for (int i = 0; i < nbBindings; i++) {
        auto dims = engine->getBindingDimensions(i);
        if (engine->bindingIsInput(i)) {
            inputIdx = i;
            // Expected NCHW: [N, 3, H, W]
            if (dims.nbDims >= 4) {
                inputSizePx = dims.d[2]; // H (assume square)
            }
            printf("입력 바인딩[%d]: %s, %dx%d\n",
                   i, engine->getBindingName(i), inputSizePx, inputSizePx);
        } else {
            outputIdx = i;
            outputDim = dims.d[dims.nbDims - 1];
            printf("출력 바인딩[%d]: %s, dim=%d\n",
                   i, engine->getBindingName(i), outputDim);
        }
    }

    if (inputIdx < 0 || outputIdx < 0 || outputDim <= 0) {
        std::cerr << "엔진 바인딩 감지 실패" << std::endl;
        return 1;
    }

    printf("입력 크기: %dx%d, 출력 차원: %d\n\n", inputSizePx, inputSizePx, outputDim);

    // Allocate GPU memory
    size_t inputSize = 1 * 3 * inputSizePx * inputSizePx * sizeof(__half);
    size_t outputSize = 1 * outputDim * sizeof(__half);

    void* d_input = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    // Warmup
    {
        std::vector<__half> h_input(1 * 3 * inputSizePx * inputSizePx);
        cv::Mat dummy_img = loadImage(db_paths[0]);
        if (!dummy_img.empty()) {
            preprocessImage(dummy_img, h_input.data(), inputSizePx, norm);
            cudaMemcpy(d_input, h_input.data(), inputSize, cudaMemcpyHostToDevice);
            void* bindings[2];
            bindings[inputIdx] = d_input;
            bindings[outputIdx] = d_output;
            context->executeV2(bindings);
            cudaDeviceSynchronize();
            printf("워밍업 완료\n\n");
        }
    }

    // Build database
    printf("데이터베이스 임베딩 중...\n");
    auto database = buildDatabase(db_paths, context, d_input, d_output,
                                  inputIdx, outputIdx, inputSizePx, outputDim, norm);

    // Debug: pairwise similarity statistics
    {
        int sample = std::min(static_cast<int>(database.size()), 20);
        float min_sim = 1.0f, max_sim = -1.0f, sum_sim = 0.0f;
        int count = 0;
        for (int i = 0; i < sample; i++) {
            for (int j = i + 1; j < sample; j++) {
                float sim = dotProduct(database[i].embedding.data(),
                                       database[j].embedding.data(), outputDim);
                min_sim = std::min(min_sim, sim);
                max_sim = std::max(max_sim, sim);
                sum_sim += sim;
                count++;
            }
        }
        if (count > 0) {
            printf("[디버그] DB 이미지 간 유사도 (샘플 %d장):\n", sample);
            printf("  min=%.4f, max=%.4f, avg=%.4f\n",
                   min_sim, max_sim, sum_sim / count);
        }
    }
    printf("\n");

    // Process queries
    std::vector<__half> h_input(1 * 3 * inputSizePx * inputSizePx);
    std::vector<__half> h_output_half(outputDim);
    std::vector<float> query_embedding(outputDim);

    if (save_dir.empty()) {
        printf("쿼리 검색 시작 (아무 키나 누르면 다음 쿼리, ESC/q로 종료)\n\n");
    } else {
        printf("쿼리 검색 시작 (결과 -> %s)\n\n", save_dir.c_str());
    }

    for (size_t qi = 0; qi < query_paths.size(); qi++) {
        cv::Mat query_img = loadImage(query_paths[qi]);
        if (query_img.empty()) {
            std::cerr << "쿼리 이미지 로드 실패: " << query_paths[qi] << std::endl;
            continue;
        }

        embedImage(query_img, context, d_input, d_output,
                   inputIdx, outputIdx, inputSizePx, outputDim,
                   h_input, h_output_half, query_embedding.data(), norm);

        auto matches = findTopK(query_embedding.data(), database, outputDim, top_k, threshold);

        printf("Query [%zu/%zu]: %s\n", qi + 1, query_paths.size(),
               getFilename(query_paths[qi]).c_str());
        if (matches.empty()) {
            printf("  매칭 결과 없음 (threshold=%.4f)\n", threshold);
        } else {
            for (size_t i = 0; i < matches.size(); i++) {
                printf("  %zu. %s (similarity: %.4f)\n",
                       i + 1,
                       getFilename(database[matches[i].index].path).c_str(),
                       matches[i].similarity);
            }
        }
        printf("\n");

        if (save_dir.empty()) {
            displayResults(query_img, query_paths[qi], matches, database, threshold);
            int key = cv::waitKey(0) & 0xFF;
            if (key == 27 || key == 'q') {
                printf("사용자 종료\n");
                break;
            }
        } else {
            saveResult(query_img, query_paths[qi], matches, database,
                       threshold, save_dir, qi);
        }
    }

    if (save_dir.empty()) {
        cv::destroyAllWindows();
    }

    cudaFree(d_input);
    cudaFree(d_output);
    delete context;
    delete engine;
    delete runtime;

    printf("완료\n");
    return 0;
}
