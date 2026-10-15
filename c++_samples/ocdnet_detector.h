// ocdnet_detector.h
//
// Header-only OCDNet (DBNet) text detector for the RADIO-CLIP STR sample.
//
// It deliberately does NOT link libnvocdr: the RADIO-CLIP export is a 4-engine
// graph that the nvOCDR C API cannot drive. Everything here is plain
// TensorRT 10 + OpenCV + CUDA, so it drops straight into
// simple_inference_det_crop_recognize_radioclip.cpp. The library's internal
// src/OCDNetEngine.h is not usable from a g++-built sample because it includes
// src/kernel.h, which contains __host__ __device__ code and requires nvcc.
//
// The preprocessing reproduces nhwc2nchwWithMeanScaleResize() from
// src/kernel.cu (BGR means 104.00698793 / 116.66876762 / 122.67891434,
// scale 0.00392156) and the post-processing reproduces
// OCDNetEngine::postprocess / findCoutourWarp / unclip from
// src/OCDNetEngine.cpp plus RectEngine::formatPoints from src/RectEngine.cpp.
//
// Two detection modes:
//
//   tile_overlap == 0 (default)  single pass. The whole image is resized to the
//                                engine's input resolution. Equivalent to
//                                nvOCDR_inference().
//   tile_overlap  > 0            overlapped tiling. Reproduces the detection
//                                half of nvOCDR_high_resolution_inference(),
//                                i.e. nvOCDR::wrapInferPatch() in
//                                src/nvOCDR.cpp. Pass 0.5 for the same
//                                overlap_rate the crop_based_inference.cpp
//                                sample uses. Much better on dense pages,
//                                because small text is never downscaled to the
//                                engine's input resolution.
//
// Usage:
//     ocd::Detector::Options o;
//     o.engine_path = "/path/to/ocdnet.fp16.engine";
//     o.tile_overlap = 0.5f;              // omit or 0.f for single pass
//     ocd::Detector det(o);
//     std::vector<ocd::Box> boxes = det.detect(bgr_image);
//     cv::Mat crop = ocd::warp_crop(bgr_image, boxes[i], o.expand_ratio);

#ifndef NVOCDR_SAMPLE_OCDNET_DETECTOR_H
#define NVOCDR_SAMPLE_OCDNET_DETECTOR_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"

namespace ocd {

// Quadrilateral in ORIGINAL image coordinates, ordered
// 0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left.
struct Box {
    cv::Point2f pts[4];
    float score = 0.f;
};

namespace detail {

[[noreturn]] inline void fail(const std::string& msg)
{
    std::cerr << "[ocdnet] " << msg << std::endl;
    std::exit(1);
}

inline void cuda_ok(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        fail(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING) {
            std::cout << "[ocdnet] " << msg << std::endl;
        }
    }
};

inline Logger& logger()
{
    static Logger instance;
    return instance;
}

inline size_t elem_size(nvinfer1::DataType dtype)
{
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kBOOL: return 1;
    case nvinfer1::DataType::kUINT8: return 1;
    default: break;
    }
    fail("Unsupported OCDNet tensor dtype");
}

inline int64_t volume(const nvinfer1::Dims& dims)
{
    int64_t n = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        n *= dims.d[i] < 0 ? 1 : dims.d[i];
    }
    return n;
}

// --- BEGIN tiling math (self-contained: ints + <cmath> + <vector> only) -----
//
// Ported from nvOCDR::wrapInferPatch (src/nvOCDR.cpp:707) for the canvas size
// and tile grid, and from nvOCDR::patchMaskMergeCUDA (src/nvOCDR.cpp:610) for
// the per-tile keep-region that stitches the probability maps back together.
// Each tile contributes only its centre; half the overlap is trimmed off every
// inner edge so neighbouring tiles meet without a seam.
//
// Two deliberate deviations from the library, both bug fixes:
//
//  1. The library reads ocdnet_infer_input_shape[1] for BOTH patch height and
//     patch width (src/nvOCDR.cpp:711-712), so a non-square engine tiles with
//     square patches. Here tile_w and tile_h are passed in independently from
//     the engine's real input dims.
//  2. When the grid is a single column (or row) the library still trims
//     half the overlap off that tile's right (or bottom) edge, leaving a strip
//     of the merged mask never written — and that buffer is not zeroed
//     (src/nvOCDR.cpp:478-479 allocate without memset). Here the first and
//     last index keep their outer edge, so the keep-regions cover the canvas
//     exactly.

struct TileRoi {
    int tile_x = 0;   // tile origin inside the canvas
    int tile_y = 0;
    int keep_x = 0;   // keep-region origin inside the TILE
    int keep_y = 0;
    int dst_x = 0;    // keep-region origin inside the CANVAS
    int dst_y = 0;
    int keep_w = 0;
    int keep_h = 0;
};

struct TilePlan {
    int canvas_w = 0;
    int canvas_h = 0;
    int tile_w = 0;
    int tile_h = 0;
    int num_col = 0;
    int num_row = 0;
    std::vector<TileRoi> tiles;
};

// Returns an empty plan (num_col == 0) when the parameters are unusable.
inline TilePlan plan_tiles(int orig_w, int orig_h, int tile_w, int tile_h,
                           float overlap)
{
    TilePlan plan;
    if (orig_w <= 0 || orig_h <= 0 || tile_w <= 1 || tile_h <= 1) {
        return plan;
    }
    if (!(overlap > 0.f) || overlap >= 1.f) {
        return plan;
    }

    const int overlap_w = static_cast<int>(overlap * static_cast<float>(tile_w));
    const int overlap_h = static_cast<int>(overlap * static_cast<float>(tile_h));
    const int step_w = tile_w - overlap_w;
    const int step_h = tile_h - overlap_h;
    if (step_w <= 0 || step_h <= 0) {
        return plan;
    }

    // Grow the canvas to a whole number of steps so the last tile lands flush
    // against the right / bottom edge.
    const int extra_col = orig_w > tile_w
        ? static_cast<int>(std::ceil(static_cast<float>(orig_w - tile_w) /
                                     static_cast<float>(step_w)))
        : 0;
    const int extra_row = orig_h > tile_h
        ? static_cast<int>(std::ceil(static_cast<float>(orig_h - tile_h) /
                                     static_cast<float>(step_h)))
        : 0;

    plan.tile_w = tile_w;
    plan.tile_h = tile_h;
    plan.canvas_w = extra_col * step_w + tile_w;
    plan.canvas_h = extra_row * step_h + tile_h;
    plan.num_col = extra_col + 1;
    plan.num_row = extra_row + 1;

    // Left/top neighbour gives up half_*, right/bottom gives up the remainder.
    // Splitting it this way (rather than half_* on both sides, as the library
    // does) keeps the seam exact when overlap_* is ODD: tile n's keep-region
    // ends on precisely the pixel where tile n+1's begins.
    const int half_w = overlap_w / 2;
    const int half_h = overlap_h / 2;
    const int rest_w = overlap_w - half_w;
    const int rest_h = overlap_h - half_h;

    plan.tiles.reserve(static_cast<size_t>(plan.num_col) * plan.num_row);
    for (int row = 0; row < plan.num_row; ++row) {
        for (int col = 0; col < plan.num_col; ++col) {
            TileRoi roi;
            roi.tile_x = col * step_w;
            roi.tile_y = row * step_h;
            roi.keep_x = col == 0 ? 0 : half_w;
            roi.keep_y = row == 0 ? 0 : half_h;
            const int keep_right = col == plan.num_col - 1 ? tile_w : tile_w - rest_w;
            const int keep_bottom = row == plan.num_row - 1 ? tile_h : tile_h - rest_h;
            roi.keep_w = keep_right - roi.keep_x;
            roi.keep_h = keep_bottom - roi.keep_y;
            roi.dst_x = roi.tile_x + roi.keep_x;
            roi.dst_y = roi.tile_y + roi.keep_y;
            if (roi.keep_w > 0 && roi.keep_h > 0) {
                plan.tiles.push_back(roi);
            }
        }
    }
    return plan;
}

// Aspect-preserving fit of orig into canvas, anchored top-left. Mirrors
// KeepAspectRatioResize in src/kernel.cu:486 (including its int truncation).
// Returns the factor that maps canvas coordinates back to original ones.
inline float fit_keep_aspect(int orig_w, int orig_h, int canvas_w, int canvas_h,
                             int* new_w, int* new_h)
{
    const float ar_in = static_cast<float>(orig_w) / static_cast<float>(orig_h);
    const float ar_out = static_cast<float>(canvas_w) / static_cast<float>(canvas_h);
    if (ar_in >= ar_out) {
        *new_w = canvas_w;
        *new_h = static_cast<int>(static_cast<float>(*new_w) / ar_in);
        return static_cast<float>(orig_w) / static_cast<float>(*new_w);
    }
    *new_h = canvas_h;
    *new_w = static_cast<int>(static_cast<float>(*new_h) * ar_in);
    return static_cast<float>(orig_h) / static_cast<float>(*new_h);
}
// --- END tiling math -------------------------------------------------------

// --- ported verbatim from OCDNetEngine::contourScore ------------------------
inline float contour_score(const cv::Mat& binary, const std::vector<cv::Point>& contour)
{
    cv::Rect rect = cv::boundingRect(contour);
    const int xmin = std::max(rect.x, 0);
    const int xmax = std::min(rect.x + rect.width, binary.cols - 1);
    const int ymin = std::max(rect.y, 0);
    const int ymax = std::min(rect.y + rect.height, binary.rows - 1);
    if (xmax < xmin || ymax < ymin) {
        return 0.f;
    }

    cv::Mat roi = binary(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
    cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
    std::vector<cv::Point> shifted;
    shifted.reserve(contour.size());
    for (const cv::Point& p : contour) {
        shifted.emplace_back(p.x - xmin, p.y - ymin);
    }
    std::vector<std::vector<cv::Point>> polys = {shifted};
    cv::fillPoly(mask, polys, cv::Scalar(1));
    return static_cast<float>(cv::mean(roi, mask).val[0]);
}

// --- ported verbatim from OCDNetEngine::unclip ------------------------------
inline void unclip(const std::vector<cv::Point2f>& in_poly, float ratio,
                   std::vector<cv::Point2f>& out_poly)
{
    const float area = static_cast<float>(cv::contourArea(in_poly));
    const float length = static_cast<float>(cv::arcLength(in_poly, true));
    if (length <= 0.f) {
        out_poly = in_poly;
        return;
    }
    const float distance = area * ratio / length;

    const size_t num_points = in_poly.size();
    std::vector<std::vector<cv::Point2f>> new_lines;
    for (size_t i = 0; i < num_points; ++i) {
        const cv::Point2f pt1 = in_poly[i];
        const cv::Point2f pt2 = in_poly[(i + num_points - 1) % num_points];
        const cv::Point2f vec = pt1 - pt2;
        const double n = cv::norm(vec);
        if (n < 1e-6) {
            out_poly = in_poly;
            return;
        }
        const float unclip_dis = static_cast<float>(distance / n);
        const cv::Point2f rotate_vec(vec.y * unclip_dis, -vec.x * unclip_dis);
        new_lines.push_back({pt1 + rotate_vec, pt2 + rotate_vec});
    }

    const size_t num_lines = new_lines.size();
    for (size_t i = 0; i < num_lines; ++i) {
        const cv::Point2f a = new_lines[i][0];
        const cv::Point2f b = new_lines[i][1];
        const cv::Point2f c = new_lines[(i + 1) % num_lines][0];
        const cv::Point2f d = new_lines[(i + 1) % num_lines][1];
        const cv::Point2f v1 = b - a;
        const cv::Point2f v2 = d - c;
        const double denom_norm = cv::norm(v1) * cv::norm(v2);
        const float cos_angle = denom_norm > 0.0
            ? static_cast<float>((v1.x * v2.x + v1.y * v2.y) / denom_norm)
            : 1.f;

        cv::Point2f pt;
        if (std::fabs(cos_angle) > 0.7f) {
            pt.x = (b.x + c.x) * 0.5f;
            pt.y = (b.y + c.y) * 0.5f;
        } else {
            const float denom = a.x * (d.y - c.y) + b.x * (c.y - d.y) +
                                d.x * (b.y - a.y) + c.x * (a.y - b.y);
            if (std::fabs(denom) < 1e-6f) {
                pt.x = (b.x + c.x) * 0.5f;
                pt.y = (b.y + c.y) * 0.5f;
            } else {
                const float num = a.x * (d.y - c.y) + c.x * (a.y - d.y) + d.x * (c.y - a.y);
                const float s = num / denom;
                pt.x = a.x + s * (b.x - a.x);
                pt.y = a.y + s * (b.y - a.y);
            }
        }
        out_poly.push_back(pt);
    }
}

// --- ported from RectEngine::formatPoints -----------------------------------
// 0 left-top, 1 right-top, 2 right-bottom, 3 left-bottom.
inline void format_points(const std::vector<cv::Point2f>& poly, float rot_thresh,
                          cv::Point2f out[4])
{
    std::vector<cv::Point2f> points(poly.begin(), poly.begin() + 4);
    std::sort(points.begin(), points.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });

    if (points[0].y <= points[1].y) {
        out[0] = points[0];
        out[3] = points[1];
    } else {
        out[0] = points[1];
        out[3] = points[0];
    }
    if (points[2].y <= points[3].y) {
        out[1] = points[2];
        out[2] = points[3];
    } else {
        out[1] = points[3];
        out[2] = points[2];
    }

    const double top = cv::norm(out[1] - out[0]);
    const double left = cv::norm(out[3] - out[0]);
    if (left > 1e-6 && (top / left) < rot_thresh) {
        const cv::Point2f tmp = out[0];
        out[0] = out[3];
        out[3] = out[2];
        out[2] = out[1];
        out[1] = tmp;
    }
}

}  // namespace detail

// Expand a quad outward from its centroid, then perspective-warp it to an
// upright crop. Returns BGR CV_8UC3 suitable for the sample's preprocess().
inline cv::Mat warp_crop(const cv::Mat& image, const Box& box, float expand_ratio = 0.f)
{
    cv::Point2f src[4];
    cv::Point2f centre(0.f, 0.f);
    for (int i = 0; i < 4; ++i) {
        centre += box.pts[i];
    }
    centre *= 0.25f;
    for (int i = 0; i < 4; ++i) {
        src[i] = centre + (box.pts[i] - centre) * (1.f + expand_ratio);
    }

    const double top = cv::norm(src[1] - src[0]);
    const double bottom = cv::norm(src[2] - src[3]);
    const double left = cv::norm(src[3] - src[0]);
    const double right = cv::norm(src[2] - src[1]);
    const int w = static_cast<int>(std::lround(std::max(top, bottom)));
    const int h = static_cast<int>(std::lround(std::max(left, right)));
    if (w < 2 || h < 2) {
        return cv::Mat();
    }

    const cv::Point2f dst[4] = {
        cv::Point2f(0.f, 0.f),
        cv::Point2f(static_cast<float>(w), 0.f),
        cv::Point2f(static_cast<float>(w), static_cast<float>(h)),
        cv::Point2f(0.f, static_cast<float>(h)),
    };
    const cv::Mat m = cv::getPerspectiveTransform(src, dst);
    cv::Mat crop;
    cv::warpPerspective(image, crop, m, cv::Size(w, h), cv::INTER_LINEAR,
                        cv::BORDER_REPLICATE);
    return crop;
}

inline void draw_box(cv::Mat& canvas, const Box& box, const std::string& label)
{
    std::vector<cv::Point> poly;
    poly.reserve(4);
    for (int i = 0; i < 4; ++i) {
        poly.emplace_back(cv::Point(static_cast<int>(std::lround(box.pts[i].x)),
                                    static_cast<int>(std::lround(box.pts[i].y))));
    }
    cv::polylines(canvas, poly, true, cv::Scalar(0, 255, 0), 2);
    if (!label.empty()) {
        const cv::Point anchor(poly[0].x, std::max(14, poly[0].y - 4));
        cv::putText(canvas, label, anchor, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 0, 0), 3);
        cv::putText(canvas, label, anchor, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 255), 1);
    }
}

class Detector {
public:
    struct Options {
        std::string engine_path;
        // Mirrors nvOCDRParam. Defaults match the scene-text sample.
        float binarize_threshold = 0.1f;
        float polygon_threshold = 0.3f;
        float unclip_ratio = 1.5f;
        // Matches nvOCDRParam::ocdnet_max_candidate in the nvOCDR samples.
        // Keep it high: tiled detection contours a full-resolution mask, which
        // yields far more candidates than a single 320x320 map, and a cap that
        // binds silently drops real text.
        int max_candidates = 5000;
        // width/height ratio below which the quad is rotated 90 degrees.
        // nvOCDR defaults to 0 (never rotate); set ~1.0 for vertical text.
        float rotation_threshold = 0.f;
        // Enlarge each quad before cropping; STR models like a little margin.
        float expand_ratio = 0.05f;
        // Drop quads whose short side is below this many pixels.
        int min_side = 4;
        // Override the engine's input resolution (0 = use the engine's own).
        int infer_height = 0;
        int infer_width = 0;
        // 0  = single pass over the whole image (nvOCDR_inference).
        // >0 = overlapped tiling at the engine's input resolution, the
        //      detection half of nvOCDR_high_resolution_inference. 0.5 is the
        //      overlap_rate used by crop_based_inference.cpp. Must be < 1.
        float tile_overlap = 0.f;
        // Print the tile grid once per detect() call.
        bool verbose_tiling = true;
    };

    explicit Detector(const Options& options)
        : opts_(options)
    {
        if (opts_.tile_overlap < 0.f || opts_.tile_overlap >= 1.f) {
            detail::fail("tile_overlap must be in [0, 1); got " +
                         std::to_string(opts_.tile_overlap));
        }

        std::ifstream file(opts_.engine_path, std::ios::binary);
        if (!file) {
            detail::fail("Failed to open OCDNet engine " + opts_.engine_path +
                         "\nBuild one with:\n"
                         "  trtexec --onnx=ocdnet.onnx \\\n"
                         "          --minShapes=input:1x3x736x1280 \\\n"
                         "          --optShapes=input:1x3x736x1280 \\\n"
                         "          --maxShapes=input:1x3x736x1280 \\\n"
                         "          --fp16 --saveEngine=ocdnet.fp16.engine");
        }
        file.seekg(0, std::ios::end);
        const std::streamoff size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> blob(static_cast<size_t>(size));
        file.read(blob.data(), size);
        if (!file) {
            detail::fail("Failed to read OCDNet engine " + opts_.engine_path);
        }

        runtime_ = nvinfer1::createInferRuntime(detail::logger());
        if (runtime_ == nullptr) {
            detail::fail("createInferRuntime failed");
        }
        engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
        if (engine_ == nullptr) {
            detail::fail("deserializeCudaEngine failed for " + opts_.engine_path);
        }
        context_ = engine_->createExecutionContext();
        if (context_ == nullptr) {
            detail::fail("createExecutionContext failed");
        }

        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                if (!input_.empty()) {
                    detail::fail("OCDNet engine has more than one input");
                }
                input_ = name;
            } else {
                if (output_.empty()) {
                    output_ = name;
                }
            }
        }
        if (input_.empty() || output_.empty()) {
            detail::fail("OCDNet engine is missing an input or output tensor");
        }

        resolve_input_shape();

        if (!context_->setInputShape(input_.c_str(), in_dims_)) {
            detail::fail("setInputShape failed for OCDNet input");
        }
        out_dims_ = context_->getTensorShape(output_.c_str());

        in_dtype_ = engine_->getTensorDataType(input_.c_str());
        out_dtype_ = engine_->getTensorDataType(output_.c_str());
        in_count_ = static_cast<size_t>(detail::volume(in_dims_));
        out_count_ = static_cast<size_t>(detail::volume(out_dims_));

        detail::cuda_ok(cudaMalloc(&d_in_, in_count_ * detail::elem_size(in_dtype_)),
                        "cudaMalloc ocdnet input");
        detail::cuda_ok(cudaMalloc(&d_out_, out_count_ * detail::elem_size(out_dtype_)),
                        "cudaMalloc ocdnet output");
        detail::cuda_ok(cudaStreamCreate(&stream_), "cudaStreamCreate");

        std::cout << "[ocdnet] loaded " << opts_.engine_path << std::endl;
        std::cout << "[ocdnet]   " << input_ << " [" << in_dims_.d[0] << "," << in_dims_.d[1]
                  << "," << in_dims_.d[2] << "," << in_dims_.d[3] << "]" << std::endl;
        std::cout << "[ocdnet]   " << output_ << " [";
        for (int i = 0; i < out_dims_.nbDims; ++i) {
            std::cout << (i ? "," : "") << out_dims_.d[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "[ocdnet]   mode "
                  << (opts_.tile_overlap > 0.f ? "tiled" : "single-pass");
        if (opts_.tile_overlap > 0.f) {
            std::cout << " overlap " << opts_.tile_overlap;
        }
        std::cout << std::endl;
    }

    ~Detector()
    {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
        if (d_in_ != nullptr) {
            cudaFree(d_in_);
        }
        if (d_out_ != nullptr) {
            cudaFree(d_out_);
        }
        delete context_;
        delete engine_;
        delete runtime_;
    }

    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    int infer_height() const { return in_dims_.d[2]; }
    int infer_width() const { return in_dims_.d[3]; }

    std::vector<Box> detect(const cv::Mat& bgr)
    {
        if (bgr.empty() || bgr.type() != CV_8UC3) {
            detail::fail("detect() expects a non-empty CV_8UC3 BGR image");
        }
        std::vector<Box> boxes = opts_.tile_overlap > 0.f ? detect_tiled(bgr)
                                                          : detect_single(bgr);
        // Reading order: top-to-bottom, then left-to-right.
        std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
            const float ay = (a.pts[0].y + a.pts[1].y) * 0.5f;
            const float by = (b.pts[0].y + b.pts[1].y) * 0.5f;
            if (std::fabs(ay - by) > 12.f) {
                return ay < by;
            }
            return a.pts[0].x < b.pts[0].x;
        });
        return boxes;
    }

private:
    // One pass: resize the whole frame to the engine input, as nvOCDR_inference
    // does. Small text on a large page gets downscaled away.
    std::vector<Box> detect_single(const cv::Mat& bgr)
    {
        const cv::Mat prob = infer_map(bgr);
        dump_mask(prob, "single");
        const float scale_w = static_cast<float>(bgr.cols) / static_cast<float>(prob.cols);
        const float scale_h = static_cast<float>(bgr.rows) / static_cast<float>(prob.rows);
        return boxes_from_map(prob, scale_w, scale_h, bgr.cols, bgr.rows);
    }

    // Write the probability map as a PNG when NVOCDR_DUMP_MASK is set, so a
    // bad mask can be told apart from bad coordinate mapping by eye.
    void dump_mask(const cv::Mat& prob, const char* tag) const
    {
        const char* dir = std::getenv("NVOCDR_DUMP_MASK");
        if (dir == nullptr) {
            return;
        }
        double lo = 0.0;
        double hi = 0.0;
        cv::minMaxLoc(prob, &lo, &hi);
        cv::Mat vis;
        prob.convertTo(vis, CV_8U, 255.0);
        const std::string path = std::string(dir) + "/mask_" + tag + ".png";
        cv::imwrite(path, vis);
        std::cout << "[ocdnet] mask " << prob.cols << "x" << prob.rows
                  << " range [" << lo << ", " << hi << "] -> " << path << std::endl;
    }

    // Overlapped tiling: the detection half of wrapInferPatch. Every tile is
    // inferred at the engine's native resolution, the probability maps are
    // stitched into one full-canvas mask, and contours are found ONCE on that
    // mask — so a text line crossing a tile seam stays a single box.
    std::vector<Box> detect_tiled(const cv::Mat& bgr)
    {
        const int tile_w = in_dims_.d[3];
        const int tile_h = in_dims_.d[2];

        const detail::TilePlan plan = detail::plan_tiles(bgr.cols, bgr.rows, tile_w,
                                                         tile_h, opts_.tile_overlap);
        if (plan.num_col == 0) {
            detail::fail("Could not build a tile plan; check tile_overlap and the "
                         "engine input resolution");
        }

        int fit_w = 0;
        int fit_h = 0;
        const float rescale = detail::fit_keep_aspect(bgr.cols, bgr.rows, plan.canvas_w,
                                                      plan.canvas_h, &fit_w, &fit_h);
        if (fit_w <= 0 || fit_h <= 0) {
            detail::fail("Aspect-preserving fit produced an empty canvas");
        }

        if (opts_.verbose_tiling) {
            std::cout << "[ocdnet] tiling " << bgr.cols << "x" << bgr.rows << " -> canvas "
                      << plan.canvas_w << "x" << plan.canvas_h << " (content " << fit_w
                      << "x" << fit_h << "), " << plan.num_col << "x" << plan.num_row
                      << " tiles of " << tile_w << "x" << tile_h << ", rescale "
                      << rescale << std::endl;
        }

        // Unlike the library, the pad region is zeroed. wrapInferPatch leaves
        // it as whatever cudaMalloc returned, which can invent detections.
        cv::Mat canvas = cv::Mat::zeros(plan.canvas_h, plan.canvas_w, CV_8UC3);
        cv::Mat fitted;
        cv::resize(bgr, fitted, cv::Size(fit_w, fit_h), 0, 0, cv::INTER_LINEAR);
        fitted.copyTo(canvas(cv::Rect(0, 0, fit_w, fit_h)));

        cv::Mat merged = cv::Mat::zeros(plan.canvas_h, plan.canvas_w, CV_32F);
        int inferred = 0;
        for (const detail::TileRoi& roi : plan.tiles) {
            // A tile that starts past the fitted content is pure padding.
            if (roi.tile_x >= fit_w || roi.tile_y >= fit_h) {
                continue;
            }
            const cv::Mat tile = canvas(cv::Rect(roi.tile_x, roi.tile_y, tile_w, tile_h));
            cv::Mat prob = infer_map(tile);
            ++inferred;
            // DBNet normally emits a map at the input resolution; resize if the
            // engine disagrees so the keep-region maths stays in tile pixels.
            if (prob.cols != tile_w || prob.rows != tile_h) {
                cv::resize(prob, prob, cv::Size(tile_w, tile_h), 0, 0, cv::INTER_LINEAR);
            }
            prob(cv::Rect(roi.keep_x, roi.keep_y, roi.keep_w, roi.keep_h))
                .copyTo(merged(cv::Rect(roi.dst_x, roi.dst_y, roi.keep_w, roi.keep_h)));
        }
        if (opts_.verbose_tiling) {
            std::cout << "[ocdnet] inferred " << inferred << " of " << plan.tiles.size()
                      << " tiles (" << (plan.tiles.size() - inferred)
                      << " skipped as padding)" << std::endl;
        }

        // Contour ONLY the fitted content, never the zero padding.
        //
        // The canvas is larger than the fitted image (it is rounded up to a
        // whole number of tile steps), so merged[] has a black band along the
        // bottom and right. That band is zeroed and cannot contour on its own,
        // but the content/pad boundary is a maximum-contrast edge and DBNet
        // activates strongly along it. Those activations used to be contoured,
        // scaled, then clamped to clip_h-1 / clip_w-1 -- which pinned a row of
        // bogus boxes to the bottom (and right) edge of the output image.
        const cv::Rect content(0, 0, std::min(fit_w, plan.canvas_w),
                               std::min(fit_h, plan.canvas_h));
        const cv::Mat content_mask = merged(content);

        dump_mask(content_mask, "tiled");

        // Canvas -> original is one uniform factor on both axes.
        return boxes_from_map(content_mask, rescale, rescale, bgr.cols, bgr.rows);
    }

    // Preprocess + infer one BGR image, returning its probability map as an
    // owned CV_32F. Mirrors nhwc2nchwWithMeanScaleResize in src/kernel.cu.
    cv::Mat infer_map(const cv::Mat& bgr)
    {
        const int net_h = in_dims_.d[2];
        const int net_w = in_dims_.d[3];

        cv::Mat resized;
        if (bgr.cols == net_w && bgr.rows == net_h) {
            resized = bgr;
        } else {
            cv::resize(bgr, resized, cv::Size(net_w, net_h), 0, 0, cv::INTER_LINEAR);
        }
        cv::Mat f32;
        resized.convertTo(f32, CV_32FC3);
        // OpenCV channel order is BGR, matching the kernel's dst_c 0/1/2 means.
        const float mean[3] = {104.00698793f, 116.66876762f, 122.67891434f};
        const float scale = 0.00392156f;
        std::vector<float> nchw(static_cast<size_t>(3) * net_h * net_w);
        for (int y = 0; y < net_h; ++y) {
            const cv::Vec3f* row = f32.ptr<cv::Vec3f>(y);
            for (int x = 0; x < net_w; ++x) {
                for (int c = 0; c < 3; ++c) {
                    nchw[(static_cast<size_t>(c) * net_h + y) * net_w + x] =
                        (row[x][c] - mean[c]) * scale;
                }
            }
        }

        upload(nchw);
        if (!context_->setTensorAddress(input_.c_str(), d_in_) ||
            !context_->setTensorAddress(output_.c_str(), d_out_)) {
            detail::fail("setTensorAddress failed");
        }
        if (!context_->enqueueV3(stream_)) {
            detail::fail("enqueueV3 failed");
        }
        detail::cuda_ok(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

        std::vector<float> prob(out_count_);
        download(prob);

        // DBNet's probability map is the LAST two dims. Guard the assumption:
        // a multi-head export (pred + thresh + thresh_binary) or an NHWC-style
        // [1,H,W,1] output would silently produce a wrong-shaped Mat, which
        // then yields a wrong scale factor and boxes clamped to one edge.
        const int map_h = out_dims_.d[out_dims_.nbDims - 2];
        const int map_w = out_dims_.d[out_dims_.nbDims - 1];
        if (map_h <= 1 || map_w <= 1) {
            detail::fail("OCDNet output " + output_ + " has trailing dims " +
                         std::to_string(map_h) + "x" + std::to_string(map_w) +
                         ", which is not a probability map. Re-export so the "
                         "last two dimensions are height x width.");
        }
        const size_t map_count = static_cast<size_t>(map_h) * map_w;
        if (out_count_ % map_count != 0) {
            detail::fail("OCDNet output element count " + std::to_string(out_count_) +
                         " is not a multiple of " + std::to_string(map_h) + "x" +
                         std::to_string(map_w));
        }
        if (out_count_ != map_count && !warned_multi_head_) {
            warned_multi_head_ = true;
            std::cout << "[ocdnet] note: output holds " << (out_count_ / map_count)
                      << " maps; using the first (DBNet 'pred')" << std::endl;
        }
        // Deep copy: prob goes out of scope with this frame.
        return cv::Mat(map_h, map_w, CV_32F, prob.data()).clone();
    }

    // --- post-process: OCDNetEngine::postprocess + findCoutourWarp ----------
    // scale_x / scale_y map map-space coordinates onto the ORIGINAL image.
    std::vector<Box> boxes_from_map(const cv::Mat& prob, float scale_x, float scale_y,
                                    int clip_w, int clip_h) const
    {
        cv::Mat bitmap;
        cv::threshold(prob, bitmap, opts_.binarize_threshold, 255, cv::THRESH_BINARY);
        bitmap.convertTo(bitmap, CV_8U);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

        // Score EVERY contour first, then keep the best max_candidates.
        //
        // Truncating the raw findContours() list instead (as this code used to
        // do, and as OCDNetEngine::postprocess does) keeps whichever contours
        // the scanline happened to reach first, so once the cap binds the
        // survivors are all from one end of the image. That looks exactly like
        // a coordinate bug: boxes bunched along one edge, the rest of the page
        // empty. Ranking by score makes the cap degrade by confidence instead
        // of by position.
        std::vector<std::pair<float, size_t>> ranked;
        ranked.reserve(contours.size());
        size_t too_small = 0;
        for (size_t i = 0; i < contours.size(); ++i) {
            if (contours[i].size() < 4) {
                ++too_small;
                continue;
            }
            const float score = detail::contour_score(prob, contours[i]);
            if (score < opts_.polygon_threshold) {
                continue;
            }
            ranked.emplace_back(score, i);
        }

        const size_t above_threshold = ranked.size();
        std::sort(ranked.begin(), ranked.end(),
                  [](const std::pair<float, size_t>& a,
                     const std::pair<float, size_t>& b) { return a.first > b.first; });
        if (opts_.max_candidates > 0 &&
            ranked.size() > static_cast<size_t>(opts_.max_candidates)) {
            ranked.resize(static_cast<size_t>(opts_.max_candidates));
        }

        if (opts_.verbose_tiling) {
            std::printf("[ocdnet] contours: %zu found, %zu scored >= %.2f, %zu kept"
                        " (cap %d)%s\n",
                        contours.size(), above_threshold, opts_.polygon_threshold,
                        ranked.size(), opts_.max_candidates,
                        above_threshold > ranked.size()
                            ? "  <-- CAP HIT, raise max_candidates" : "");
            if (too_small != 0) {
                std::printf("[ocdnet]   (%zu contours had < 4 points)\n", too_small);
            }
        }

        std::vector<Box> boxes;
        boxes.reserve(ranked.size());
        for (const std::pair<float, size_t>& entry : ranked) {
            const float score = entry.first;
            const std::vector<cv::Point>& contour = contours[entry.second];

            std::vector<cv::Point> scaled;
            scaled.reserve(contour.size());
            for (const cv::Point& p : contour) {
                scaled.emplace_back(static_cast<int>(p.x * scale_x),
                                    static_cast<int>(p.y * scale_y));
            }

            cv::RotatedRect rect = cv::minAreaRect(scaled);
            if (std::min(rect.size.width, rect.size.height) < 1.f) {
                continue;
            }

            cv::Point2f vertex[4];
            rect.points(vertex);
            const std::vector<cv::Point2f> approx(vertex, vertex + 4);
            std::vector<cv::Point2f> poly;
            detail::unclip(approx, opts_.unclip_ratio, poly);
            if (poly.size() < 4) {
                continue;
            }

            Box box;
            box.score = score;
            detail::format_points(poly, opts_.rotation_threshold, box.pts);

            const double top = cv::norm(box.pts[1] - box.pts[0]);
            const double left = cv::norm(box.pts[3] - box.pts[0]);
            if (std::min(top, left) < static_cast<double>(opts_.min_side)) {
                continue;
            }
            for (int k = 0; k < 4; ++k) {
                box.pts[k].x = std::min(std::max(box.pts[k].x, 0.f),
                                        static_cast<float>(clip_w - 1));
                box.pts[k].y = std::min(std::max(box.pts[k].y, 0.f),
                                        static_cast<float>(clip_h - 1));
            }
            boxes.push_back(box);
        }
        return boxes;
    }

    void resolve_input_shape()
    {
        in_dims_ = engine_->getTensorShape(input_.c_str());
        if (in_dims_.nbDims != 4) {
            detail::fail("OCDNet input must be rank 4 (NCHW)");
        }
        nvinfer1::Dims opt = in_dims_;
        bool dynamic = false;
        for (int i = 0; i < in_dims_.nbDims; ++i) {
            if (in_dims_.d[i] < 0) {
                dynamic = true;
            }
        }
        if (dynamic) {
            opt = engine_->getProfileShape(input_.c_str(), 0,
                                           nvinfer1::OptProfileSelector::kOPT);
        }
        in_dims_.d[0] = 1;
        in_dims_.d[1] = 3;
        in_dims_.d[2] = opts_.infer_height > 0 ? opts_.infer_height : opt.d[2];
        in_dims_.d[3] = opts_.infer_width > 0 ? opts_.infer_width : opt.d[3];
        if (in_dims_.d[2] <= 0 || in_dims_.d[3] <= 0) {
            detail::fail("Could not resolve the OCDNet input resolution; set "
                         "Options::infer_height / infer_width");
        }
    }

    void upload(const std::vector<float>& host)
    {
        if (host.size() != in_count_) {
            detail::fail("OCDNet input element count mismatch");
        }
        if (in_dtype_ == nvinfer1::DataType::kFLOAT) {
            detail::cuda_ok(cudaMemcpyAsync(d_in_, host.data(), host.size() * sizeof(float),
                                            cudaMemcpyHostToDevice, stream_),
                            "upload ocdnet fp32");
            return;
        }
        if (in_dtype_ == nvinfer1::DataType::kHALF) {
            std::vector<__half> half(host.size());
            for (size_t i = 0; i < host.size(); ++i) {
                half[i] = __float2half(host[i]);
            }
            detail::cuda_ok(cudaMemcpyAsync(d_in_, half.data(), half.size() * sizeof(__half),
                                            cudaMemcpyHostToDevice, stream_),
                            "upload ocdnet fp16");
            detail::cuda_ok(cudaStreamSynchronize(stream_), "sync after fp16 upload");
            return;
        }
        detail::fail("Unsupported OCDNet input dtype");
    }

    void download(std::vector<float>& host)
    {
        if (out_dtype_ == nvinfer1::DataType::kFLOAT) {
            detail::cuda_ok(cudaMemcpy(host.data(), d_out_, host.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost),
                            "download ocdnet fp32");
            return;
        }
        if (out_dtype_ == nvinfer1::DataType::kHALF) {
            std::vector<__half> half(host.size());
            detail::cuda_ok(cudaMemcpy(half.data(), d_out_, half.size() * sizeof(__half),
                                       cudaMemcpyDeviceToHost),
                            "download ocdnet fp16");
            for (size_t i = 0; i < host.size(); ++i) {
                host[i] = __half2float(half[i]);
            }
            return;
        }
        detail::fail("Unsupported OCDNet output dtype");
    }

    Options opts_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    std::string input_;
    std::string output_;
    nvinfer1::Dims in_dims_{};
    nvinfer1::Dims out_dims_{};
    nvinfer1::DataType in_dtype_ = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType out_dtype_ = nvinfer1::DataType::kFLOAT;
    size_t in_count_ = 0;
    size_t out_count_ = 0;
    mutable bool warned_multi_head_ = false;
    void* d_in_ = nullptr;
    void* d_out_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

}  // namespace ocd

#endif  // NVOCDR_SAMPLE_OCDNET_DETECTOR_H
