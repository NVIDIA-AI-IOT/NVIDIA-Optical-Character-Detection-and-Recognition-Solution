// RADIO-CLIP VL4STR recognition-only sample.
//
// Drives the four static-batch TensorRT engines:
//   {stem}.visual_encode.fp16.engine   imgs -> img_feature
//   {stem}.visual_decode.fp16.engine   26-step host autoregressive decode
//   {stem}.text_encode.fp16.engine     SigLIP2 tokens + img_feature -> cross_memory
//   {stem}.text_decode.fp16.engine     26-step host autoregressive decode
//
// The nvOCDR C API takes a single OCRNet engine path and cannot express a
// 4-engine graph with a host-side decode loop, so this sample calls TensorRT
// directly and does not link libnvocdr.
//
// There is NO detection stage here: the whole input image is resized to
// 224x224 and recognized as one crop. Feed it a tight crop of a single word or
// short text line. A full scene or document page decodes to a nearly empty
// string, because the 26-step decoder emits at most 26 characters for one text
// instance and the resize destroys every word's aspect ratio. Use
// simple_inference_det_crop_recognize_radioclip.cpp for full images.
//
// pos_queries are not in the engines. They are read from the Lightning checkpoint
// (visual_decoder.pos_queries and cross_decoder.pos_queries).
//
// SigLIP2 BPE lives in c++_samples/siglip2_bpe.bin (run from this directory, or
// from the repo root). Build with TensorRT 10:
//
//   g++ -std=c++17 ./simple_inference_recognize_radioclip.cpp -I../include -I/usr/include/opencv4/ -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvinfer -lnvinfer_plugin -lcudart -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -o simple_inference_recognize_radioclip
//   ./simple_inference_recognize_radioclip [image_path]
//
// A GPU-free tokenizer and pos_queries self-test is built into this file:
//
//   g++ -std=c++17 -DRADIOCLIP_TOKENIZER_TEST ./simple_inference_recognize_radioclip.cpp -I../include -o tok_test
//   ./tok_test

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int kImage = 224;
constexpr int kSteps = 26;
constexpr int kClasses = 95;
constexpr int kTokens = 196;
constexpr int kWidth = 1536;
constexpr int kTextLen = 64;
constexpr int kCrossLen = 260;
constexpr int kBosId = 95;
constexpr int kPadId = 96;
constexpr int kEosId = 0;
constexpr int kSiglipEos = 1;
constexpr int kSiglipPad = 0;
constexpr int kSiglipUnk = 3;
constexpr int kPosQueryFloats = kSteps * kWidth;

const char* kEngineStem =
    "/mnt/data/models/maglev_models/ocrnet09052/deploy/onnx_fp32/vl4str_radioclip_09052_epoch4";
const char* kCheckpoint =
    "/mnt/data/models/maglev_models/ocrnet09052/vl4str_radioclip_epoch4.ckpt";
const char kCharset[] =
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
const char kPunct[] = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
const char kSiglipBar[] = "\xE2\x96\x81";

static_assert(sizeof(kCharset) - 1 == 94, "STR charset must be 94 characters");

[[noreturn]] void die(const std::string& message)
{
    std::cerr << message << std::endl;
    std::exit(1);
}

uint16_t read_u16(const uint8_t* p)
{
    uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

uint32_t read_u32(const uint8_t* p)
{
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

uint64_t read_u64(const uint8_t* p)
{
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

std::string find_existing(const std::vector<std::string>& paths)
{
    for (const std::string& path : paths) {
        std::ifstream in(path, std::ios::binary);
        if (in.good()) {
            return path;
        }
    }
    std::ostringstream oss;
    oss << "Could not open any of:";
    for (const std::string& path : paths) {
        oss << "\n  " << path;
    }
    die(oss.str());
}

// Uncompressed member of a zip/zip64 archive. The checkpoint is several GB and
// the tensor storages sit past the 4 GiB mark, so the central directory is zip64.
std::vector<uint8_t> read_zip_stored(const std::string& zip_path, const std::string& member)
{
    std::ifstream in(zip_path, std::ios::binary);
    if (!in) {
        die("Failed to open checkpoint " + zip_path);
    }
    in.seekg(0, std::ios::end);
    const std::streamoff file_size = in.tellg();
    if (file_size < 22) {
        die("Checkpoint is not a zip file: " + zip_path);
    }

    const std::streamoff tail_len = std::min<std::streamoff>(file_size, 1024 * 1024);
    std::vector<uint8_t> tail(static_cast<size_t>(tail_len));
    in.seekg(file_size - tail_len);
    in.read(reinterpret_cast<char*>(tail.data()), tail_len);
    if (!in) {
        die("Failed to read the end of " + zip_path);
    }

    int eocd_at = -1;
    for (int i = static_cast<int>(tail.size()) - 22; i >= 0; --i) {
        if (read_u32(tail.data() + i) == 0x06054b50u) {
            eocd_at = i;
            break;
        }
    }
    if (eocd_at < 0) {
        die("Zip end-of-central-directory not found in " + zip_path);
    }

    uint64_t cd_offset = read_u32(tail.data() + eocd_at + 16);
    uint64_t cd_size = read_u32(tail.data() + eocd_at + 12);
    if (cd_offset == 0xffffffffu || cd_size == 0xffffffffu) {
        if (eocd_at < 20) {
            die("Zip64 locator is missing from " + zip_path);
        }
        const uint8_t* loc = tail.data() + eocd_at - 20;
        if (read_u32(loc) != 0x07064b50u) {
            die("Zip64 end locator not found in " + zip_path);
        }
        const uint64_t z64_offset = read_u64(loc + 8);
        uint8_t z64[56];
        in.seekg(static_cast<std::streamoff>(z64_offset));
        in.read(reinterpret_cast<char*>(z64), sizeof(z64));
        if (!in || read_u32(z64) != 0x06064b50u) {
            die("Zip64 end-of-central-directory is invalid in " + zip_path);
        }
        cd_size = read_u64(z64 + 40);
        cd_offset = read_u64(z64 + 48);
    }

    std::vector<uint8_t> central(cd_size);
    in.seekg(static_cast<std::streamoff>(cd_offset));
    in.read(reinterpret_cast<char*>(central.data()), static_cast<std::streamsize>(cd_size));
    if (!in) {
        die("Failed to read the zip central directory of " + zip_path);
    }

    uint64_t local_offset = 0;
    uint64_t uncompressed = 0;
    bool found = false;
    size_t cursor = 0;
    while (cursor + 46 <= central.size()) {
        if (read_u32(central.data() + cursor) != 0x02014b50u) {
            die("Corrupt zip central directory in " + zip_path);
        }
        const uint32_t comp32 = read_u32(central.data() + cursor + 20);
        const uint32_t uncomp32 = read_u32(central.data() + cursor + 24);
        const uint16_t method = read_u16(central.data() + cursor + 10);
        const uint16_t name_len = read_u16(central.data() + cursor + 28);
        const uint16_t extra_len = read_u16(central.data() + cursor + 30);
        const uint16_t comment_len = read_u16(central.data() + cursor + 32);
        const uint32_t local32 = read_u32(central.data() + cursor + 42);
        if (cursor + 46 + name_len + extra_len + comment_len > central.size()) {
            die("Zip central directory entry overruns the directory");
        }
        const std::string name(reinterpret_cast<const char*>(central.data() + cursor + 46), name_len);
        uint64_t uncomp = uncomp32;
        uint64_t local = local32;
        const uint8_t* extra = central.data() + cursor + 46 + name_len;
        const uint8_t* extra_end = extra + extra_len;
        const uint8_t* field = extra;
        while (field + 4 <= extra_end) {
            const uint16_t id = read_u16(field);
            const uint16_t size = read_u16(field + 2);
            field += 4;
            if (field + size > extra_end) {
                break;
            }
            if (id == 0x0001) {
                const uint8_t* q = field;
                if (uncomp32 == 0xffffffffu) {
                    uncomp = read_u64(q);
                    q += 8;
                }
                if (comp32 == 0xffffffffu) {
                    q += 8;
                }
                if (local32 == 0xffffffffu && q + 8 <= field + size) {
                    local = read_u64(q);
                }
            }
            field += size;
        }
        if (name == member) {
            if (method != 0) {
                die("Checkpoint member " + member + " is compressed; expected a stored tensor");
            }
            local_offset = local;
            uncompressed = uncomp;
            found = true;
            break;
        }
        cursor += 46 + name_len + extra_len + comment_len;
    }
    if (!found) {
        die("Checkpoint is missing " + member);
    }

    uint8_t local_hdr[30];
    in.seekg(static_cast<std::streamoff>(local_offset));
    in.read(reinterpret_cast<char*>(local_hdr), sizeof(local_hdr));
    if (!in || read_u32(local_hdr) != 0x04034b50u) {
        die("Local zip header missing for " + member);
    }
    const uint16_t local_name = read_u16(local_hdr + 26);
    const uint16_t local_extra = read_u16(local_hdr + 28);
    in.seekg(static_cast<std::streamoff>(local_offset + 30 + local_name + local_extra));
    std::vector<uint8_t> payload(uncompressed);
    in.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(uncompressed));
    if (!in) {
        die("Failed to read checkpoint member " + member);
    }
    return payload;
}

std::string storage_key_for(const std::vector<uint8_t>& pickle, const std::string& tensor_name)
{
    const std::string bytes(pickle.begin(), pickle.end());
    const size_t at = bytes.find(tensor_name);
    if (at == std::string::npos) {
        die("Checkpoint pickle has no tensor named " + tensor_name);
    }
    const size_t window = std::min(bytes.size(), at + tensor_name.size() + 96);
    for (size_t i = at + tensor_name.size(); i + 5 < window; ++i) {
        if (bytes[i] != 'X') {
            continue;
        }
        const uint32_t len = read_u32(reinterpret_cast<const uint8_t*>(bytes.data() + i + 1));
        if (len == 0 || len > 16 || i + 5 + len > bytes.size()) {
            continue;
        }
        const std::string key = bytes.substr(i + 5, len);
        if (std::all_of(key.begin(), key.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
            return key;
        }
    }
    die("Could not find the storage id for " + tensor_name);
}

std::vector<float> load_pos_queries(const std::string& ckpt, const std::string& tensor_name)
{
    const std::vector<uint8_t> pickle = read_zip_stored(ckpt, "archive/data.pkl");
    const std::string key = storage_key_for(pickle, tensor_name);
    const std::vector<uint8_t> raw = read_zip_stored(ckpt, "archive/data/" + key);
    if (raw.size() != kPosQueryFloats * sizeof(float)) {
        die(tensor_name + " storage " + key + " has " + std::to_string(raw.size()) +
            " bytes, expected " + std::to_string(kPosQueryFloats * sizeof(float)));
    }
    std::vector<float> values(kPosQueryFloats);
    std::memcpy(values.data(), raw.data(), raw.size());
    std::cout << tensor_name << " <- archive/data/" << key << std::endl;
    return values;
}

struct MergeKey {
    std::string left;
    std::string right;
    bool operator==(const MergeKey& other) const
    {
        return left == other.left && right == other.right;
    }
};

struct MergeHash {
    size_t operator()(const MergeKey& key) const
    {
        const size_t h1 = std::hash<std::string>{}(key.left);
        const size_t h2 = std::hash<std::string>{}(key.right);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

class SiglipBpe {
public:
    static SiglipBpe load(const std::string& path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            die("Failed to open SigLIP2 BPE table " + path);
        }
        char magic[4];
        in.read(magic, 4);
        if (!in || std::string(magic, 4) != "SBP1") {
            die("SigLIP2 BPE table has a bad header: " + path);
        }
        uint32_t n_vocab = 0;
        uint32_t n_merges = 0;
        in.read(reinterpret_cast<char*>(&n_vocab), 4);
        in.read(reinterpret_cast<char*>(&n_merges), 4);
        SiglipBpe bpe;
        bpe.token_to_id_.reserve(n_vocab);
        bpe.merges_.reserve(n_merges);
        for (uint32_t id = 0; id < n_vocab; ++id) {
            const std::string token = read_string(in);
            bpe.token_to_id_.emplace(token, static_cast<int32_t>(id));
        }
        for (uint32_t rank = 0; rank < n_merges; ++rank) {
            MergeKey key{read_string(in), read_string(in)};
            bpe.merges_.emplace(std::move(key), rank);
        }
        if (!in) {
            die("Truncated SigLIP2 BPE table " + path);
        }
        std::cout << "SigLIP2 BPE " << path << " vocab " << n_vocab << " merges " << n_merges << std::endl;
        return bpe;
    }

    // Canonicalize, Gemma byte-level BPE, append eos, right-pad to 64 with pad id 0.
    std::vector<int64_t> tokenize(const std::string& text) const
    {
        const std::string canonical = canonicalize(text);
        std::string marked;
        marked.reserve(canonical.size() * 3);
        for (unsigned char ch : canonical) {
            if (ch == ' ') {
                marked += kSiglipBar;
            } else {
                marked.push_back(static_cast<char>(ch));
            }
        }

        std::vector<std::string> symbols = initial_symbols(marked);
        while (symbols.size() > 1) {
            bool have_rank = false;
            uint32_t best_rank = 0;
            std::string best_left;
            std::string best_right;
            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                const auto it = merges_.find(MergeKey{symbols[i], symbols[i + 1]});
                if (it == merges_.end()) {
                    continue;
                }
                if (!have_rank || it->second < best_rank) {
                    have_rank = true;
                    best_rank = it->second;
                    best_left = symbols[i];
                    best_right = symbols[i + 1];
                }
            }
            if (!have_rank) {
                break;
            }
            std::vector<std::string> merged;
            merged.reserve(symbols.size());
            for (size_t i = 0; i < symbols.size();) {
                if (i + 1 < symbols.size() && symbols[i] == best_left && symbols[i + 1] == best_right) {
                    merged.push_back(best_left + best_right);
                    i += 2;
                } else {
                    merged.push_back(symbols[i]);
                    ++i;
                }
            }
            symbols.swap(merged);
        }

        std::vector<int64_t> ids;
        ids.reserve(kTextLen);
        for (const std::string& symbol : symbols) {
            const auto it = token_to_id_.find(symbol);
            ids.push_back(it == token_to_id_.end() ? kSiglipUnk : it->second);
        }
        ids.push_back(kSiglipEos);
        if (static_cast<int>(ids.size()) > kTextLen) {
            ids.resize(kTextLen);
            ids.back() = kSiglipEos;
        }
        ids.resize(kTextLen, kSiglipPad);
        return ids;
    }

private:
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<MergeKey, uint32_t, MergeHash> merges_;

    static std::string read_string(std::ifstream& in)
    {
        uint16_t len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string text(len, '\0');
        if (len > 0) {
            in.read(text.data(), len);
        }
        return text;
    }

    static std::string canonicalize(const std::string& text)
    {
        std::string stripped;
        stripped.reserve(text.size());
        for (unsigned char ch : text) {
            if (ch == '_') {
                ch = ' ';
            }
            if (std::strchr(kPunct, ch) != nullptr) {
                continue;
            }
            stripped.push_back(static_cast<char>(std::tolower(ch)));
        }
        std::string collapsed;
        bool pending_space = false;
        for (unsigned char ch : stripped) {
            if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
                pending_space = !collapsed.empty();
                continue;
            }
            if (pending_space) {
                collapsed.push_back(' ');
                pending_space = false;
            }
            collapsed.push_back(static_cast<char>(ch));
        }
        return collapsed;
    }

    std::vector<std::string> initial_symbols(const std::string& text) const
    {
        std::vector<std::string> symbols;
        for (size_t i = 0; i < text.size();) {
            const size_t len = utf8_len(static_cast<unsigned char>(text[i]), text.size() - i);
            const std::string ch = text.substr(i, len);
            i += len;
            if (token_to_id_.find(ch) != token_to_id_.end()) {
                symbols.push_back(ch);
                continue;
            }
            for (unsigned char byte : ch) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>", byte);
                const std::string token(buf);
                if (token_to_id_.find(token) == token_to_id_.end()) {
                    symbols.emplace_back("<unk>");
                } else {
                    symbols.push_back(token);
                }
            }
        }
        return symbols;
    }

    static size_t utf8_len(unsigned char lead, size_t remain)
    {
        size_t len = 1;
        if ((lead & 0x80u) == 0) {
            len = 1;
        } else if ((lead & 0xe0u) == 0xc0u) {
            len = 2;
        } else if ((lead & 0xf0u) == 0xe0u) {
            len = 3;
        } else if ((lead & 0xf8u) == 0xf0u) {
            len = 4;
        }
        return std::min(len, remain);
    }
};

std::string charset_adapt(const std::string& text)
{
    std::string out;
    for (unsigned char ch : text) {
        const unsigned char lower = static_cast<unsigned char>(std::tolower(ch));
        if (std::isalnum(lower)) {
            out.push_back(static_cast<char>(lower));
        }
    }
    return out;
}

struct Decoded {
    std::string text;
    double confidence;
};

Decoded decode_logits(const std::vector<float>& logits)
{
    Decoded decoded{"", 1.0};
    for (int step = 0; step < kSteps; ++step) {
        const float* row = logits.data() + static_cast<size_t>(step) * kClasses;
        float max_logit = row[0];
        for (int cls = 1; cls < kClasses; ++cls) {
            max_logit = std::max(max_logit, row[cls]);
        }
        float sum = 0.f;
        int arg = 0;
        float best = -1.f;
        for (int cls = 0; cls < kClasses; ++cls) {
            const float e = std::exp(row[cls] - max_logit);
            sum += e;
            if (e > best) {
                best = e;
                arg = cls;
            }
        }
        decoded.confidence *= static_cast<double>(best / sum);
        if (arg == kEosId) {
            break;
        }
        if (arg >= 1 && arg <= 94) {
            decoded.text.push_back(kCharset[arg - 1]);
        }
    }
    return decoded;
}

void expect_tokens(const SiglipBpe& bpe, const std::string& text, const std::vector<int64_t>& prefix)
{
    const std::vector<int64_t> ids = bpe.tokenize(text);
    if (static_cast<int>(ids.size()) != kTextLen) {
        die("tokenize(" + text + ") length is not 64");
    }
    for (size_t i = 0; i < prefix.size(); ++i) {
        if (ids[i] != prefix[i]) {
            die("tokenize(" + text + ") mismatch at " + std::to_string(i) +
                " got " + std::to_string(ids[i]) + " expected " + std::to_string(prefix[i]));
        }
    }
    for (size_t i = prefix.size(); i < ids.size(); ++i) {
        if (ids[i] != kSiglipPad) {
            die("tokenize(" + text + ") padding mismatch at " + std::to_string(i));
        }
    }
}

}  // namespace

#ifdef RADIOCLIP_TOKENIZER_TEST

int main()
{
    const std::string bpe_path = find_existing({
        "siglip2_bpe.bin",
        "c++_samples/siglip2_bpe.bin",
    });
    const SiglipBpe bpe = SiglipBpe::load(bpe_path);
    expect_tokens(bpe, "hello", {17534, kSiglipEos});
    expect_tokens(bpe, "Hello", {17534, kSiglipEos});
    expect_tokens(bpe, "hello world", {17534, 2134, kSiglipEos});
    expect_tokens(bpe, "Hello World!", {17534, 2134, kSiglipEos});
    expect_tokens(bpe, "statefarm", {2626, 36752, kSiglipEos});
    expect_tokens(bpe, "the quick brown fox", {1175, 4320, 8426, 25341, kSiglipEos});
    expect_tokens(bpe, "", {kSiglipEos});
    std::cout << "tokenizer ok" << std::endl;

    const std::vector<float> visual = load_pos_queries(kCheckpoint, "visual_decoder.pos_queries");
    const std::vector<float> cross = load_pos_queries(kCheckpoint, "cross_decoder.pos_queries");
    if (std::fabs(visual[0] - 0.7406468987464905f) > 1e-5f) {
        die("visual pos_queries[0] mismatch");
    }
    if (std::fabs(cross[0] - -0.20794221758842468f) > 1e-5f) {
        die("cross pos_queries[0] mismatch");
    }
    std::cout << "pos_queries ok" << std::endl;
    return 0;
}

#else

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvInferPlugin.h"

namespace {

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

Logger g_logger;

void check_cuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        die(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

size_t dtype_size(nvinfer1::DataType dtype)
{
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT64:
        return 8;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kUINT8:
        return 1;
    default:
        die("Unsupported TensorRT dtype");
    }
}

const char* dtype_name(nvinfer1::DataType dtype)
{
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
        return "float";
    case nvinfer1::DataType::kHALF:
        return "half";
    case nvinfer1::DataType::kINT32:
        return "int32";
    case nvinfer1::DataType::kINT64:
        return "int64";
    case nvinfer1::DataType::kBOOL:
        return "bool";
    default:
        return "other";
    }
}

uint16_t float_to_half(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t src_exp = (bits >> 23) & 0xffu;
    const uint32_t mant = bits & 0x7fffffu;
    if ((bits & 0x7fffffffu) == 0) {
        return static_cast<uint16_t>(sign);
    }
    if (src_exp == 255u) {
        return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
    }
    const int32_t exp = static_cast<int32_t>(src_exp) - 127 + 15;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        const uint32_t full = mant | 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half_mant = full >> shift;
        if ((full >> (shift - 1)) & 1u) {
            half_mant += 1u;
        }
        return static_cast<uint16_t>(sign | half_mant);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    if (mant & 0x1000u) {
        half = static_cast<uint16_t>(half + 1);
    }
    return half;
}

float half_to_float(uint16_t half)
{
    const uint32_t sign = (static_cast<uint32_t>(half) & 0x8000u) << 16;
    uint32_t exp = (half >> 10) & 0x1fu;
    uint32_t mant = half & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

struct GpuMem {
    void* ptr = nullptr;
    explicit GpuMem(size_t bytes)
    {
        check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
    }
    ~GpuMem()
    {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
    GpuMem(const GpuMem&) = delete;
    GpuMem& operator=(const GpuMem&) = delete;
};

class TrtEngine {
public:
    explicit TrtEngine(const std::string& path)
    {
        static bool plugins_ready = false;
        if (!plugins_ready) {
            initLibNvInferPlugins(&g_logger, "");
            plugins_ready = true;
        }
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            die("Failed to open engine " + path);
        }
        file.seekg(0, std::ios::end);
        const std::streamoff size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> blob(static_cast<size_t>(size));
        file.read(blob.data(), size);
        if (!file) {
            die("Failed to read engine " + path);
        }
        runtime_ = nvinfer1::createInferRuntime(g_logger);
        if (runtime_ == nullptr) {
            die("createInferRuntime failed for " + path);
        }
        engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
        if (engine_ == nullptr) {
            die("deserializeCudaEngine failed for " + path);
        }
        context_ = engine_->createExecutionContext();
        if (context_ == nullptr) {
            die("createExecutionContext failed for " + path);
        }
        std::cout << "loaded " << path << std::endl;
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            const nvinfer1::Dims dims = engine_->getTensorShape(name);
            std::cout << "  " << name << " " << dtype_name(engine_->getTensorDataType(name)) << " [";
            for (int d = 0; d < dims.nbDims; ++d) {
                if (d > 0) {
                    std::cout << ",";
                }
                std::cout << dims.d[d];
            }
            std::cout << "]" << std::endl;
        }
    }

    ~TrtEngine()
    {
        delete context_;
        delete engine_;
        delete runtime_;
    }

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    nvinfer1::DataType dtype(const char* name) const
    {
        return engine_->getTensorDataType(name);
    }

    nvinfer1::Dims shape(const char* name) const
    {
        return engine_->getTensorShape(name);
    }

    int64_t elements(const char* name) const
    {
        const nvinfer1::Dims dims = shape(name);
        int64_t count = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] < 0) {
                die(std::string("Dynamic tensor is not supported: ") + name);
            }
            count *= dims.d[i];
        }
        return count;
    }

    void bind(const char* name, void* device_ptr)
    {
        if (!context_->setTensorAddress(name, device_ptr)) {
            die(std::string("setTensorAddress failed for ") + name);
        }
    }

    void enqueue()
    {
        if (!context_->enqueueV3(0)) {
            die("enqueueV3 failed");
        }
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

private:
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
};

void expect_shape(const TrtEngine& engine, const char* name, const std::vector<int>& expected)
{
    const nvinfer1::Dims dims = engine.shape(name);
    if (static_cast<int>(expected.size()) != dims.nbDims) {
        die(std::string("Rank mismatch for ") + name);
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (dims.d[i] != expected[i]) {
            die(std::string("Shape mismatch for ") + name);
        }
    }
}

void upload_floats(void* device, nvinfer1::DataType dtype, const float* host, size_t count)
{
    if (dtype == nvinfer1::DataType::kFLOAT) {
        check_cuda(cudaMemcpy(device, host, count * sizeof(float), cudaMemcpyHostToDevice), "upload float");
        return;
    }
    if (dtype == nvinfer1::DataType::kHALF) {
        std::vector<uint16_t> half(count);
        for (size_t i = 0; i < count; ++i) {
            half[i] = float_to_half(host[i]);
        }
        check_cuda(cudaMemcpy(device, half.data(), count * sizeof(uint16_t), cudaMemcpyHostToDevice), "upload half");
        return;
    }
    die(std::string("Cannot upload floats into TensorRT dtype ") + dtype_name(dtype));
}

void upload_int64(void* device, nvinfer1::DataType dtype, const int64_t* host, size_t count)
{
    if (dtype == nvinfer1::DataType::kINT64) {
        check_cuda(cudaMemcpy(device, host, count * sizeof(int64_t), cudaMemcpyHostToDevice), "upload int64");
        return;
    }
    if (dtype == nvinfer1::DataType::kINT32) {
        std::vector<int32_t> narrow(count);
        for (size_t i = 0; i < count; ++i) {
            narrow[i] = static_cast<int32_t>(host[i]);
        }
        check_cuda(cudaMemcpy(device, narrow.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice), "upload int32");
        return;
    }
    die(std::string("Cannot upload tokens into TensorRT dtype ") + dtype_name(dtype));
}

void upload_bool(void* device, nvinfer1::DataType dtype, const uint8_t* host, size_t count)
{
    if (dtype != nvinfer1::DataType::kBOOL && dtype != nvinfer1::DataType::kUINT8 &&
        dtype != nvinfer1::DataType::kINT8) {
        die(std::string("Cannot upload a mask into TensorRT dtype ") + dtype_name(dtype));
    }
    check_cuda(cudaMemcpy(device, host, count, cudaMemcpyHostToDevice), "upload bool");
}

void download_floats(float* host, nvinfer1::DataType dtype, const void* device, size_t count)
{
    if (dtype == nvinfer1::DataType::kFLOAT) {
        check_cuda(cudaMemcpy(host, device, count * sizeof(float), cudaMemcpyDeviceToHost), "download float");
        return;
    }
    if (dtype == nvinfer1::DataType::kHALF) {
        std::vector<uint16_t> half(count);
        check_cuda(cudaMemcpy(half.data(), device, count * sizeof(uint16_t), cudaMemcpyDeviceToHost), "download half");
        for (size_t i = 0; i < count; ++i) {
            host[i] = half_to_float(half[i]);
        }
        return;
    }
    die(std::string("Cannot download TensorRT dtype ") + dtype_name(dtype) + " as float");
}

std::vector<float> preprocess(const cv::Mat& bgr)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(kImage, kImage), 0, 0, cv::INTER_CUBIC);
    cv::Mat f32;
    resized.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    const float stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    std::vector<float> nchw(3 * kImage * kImage);
    for (int y = 0; y < kImage; ++y) {
        const cv::Vec3f* row = f32.ptr<cv::Vec3f>(y);
        for (int x = 0; x < kImage; ++x) {
            for (int c = 0; c < 3; ++c) {
                nchw[(c * kImage + y) * kImage + x] = (row[x][c] - mean[c]) / stdv[c];
            }
        }
    }
    return nchw;
}

std::vector<float> greedy_decode(
    TrtEngine& decoder,
    const char* memory_name,
    GpuMem& memory,
    const std::vector<float>& pos_queries)
{
    expect_shape(decoder, "tgt_in", {1, kSteps});
    expect_shape(decoder, "pos_query", {1, 1, kWidth});
    expect_shape(decoder, "query_mask", {1, kSteps});
    expect_shape(decoder, "padding_mask", {1, kSteps});
    expect_shape(decoder, "logits", {1, 1, kClasses});

    const nvinfer1::DataType tgt_dtype = decoder.dtype("tgt_in");
    const nvinfer1::DataType pos_dtype = decoder.dtype("pos_query");
    const nvinfer1::DataType query_dtype = decoder.dtype("query_mask");
    const nvinfer1::DataType pad_dtype = decoder.dtype("padding_mask");
    const nvinfer1::DataType logit_dtype = decoder.dtype("logits");

    GpuMem tgt(static_cast<size_t>(decoder.elements("tgt_in")) * dtype_size(tgt_dtype));
    GpuMem pos(static_cast<size_t>(decoder.elements("pos_query")) * dtype_size(pos_dtype));
    GpuMem query(static_cast<size_t>(decoder.elements("query_mask")) * dtype_size(query_dtype));
    GpuMem padding(static_cast<size_t>(decoder.elements("padding_mask")) * dtype_size(pad_dtype));
    GpuMem logits(static_cast<size_t>(decoder.elements("logits")) * dtype_size(logit_dtype));

    decoder.bind("tgt_in", tgt.ptr);
    decoder.bind(memory_name, memory.ptr);
    decoder.bind("pos_query", pos.ptr);
    decoder.bind("query_mask", query.ptr);
    decoder.bind("padding_mask", padding.ptr);
    decoder.bind("logits", logits.ptr);

    std::vector<int64_t> target(kSteps, kPadId);
    target[0] = kBosId;
    std::vector<float> all_logits(static_cast<size_t>(kSteps) * kClasses);
    const float neg_inf = -std::numeric_limits<float>::infinity();

    for (int step = 0; step < kSteps; ++step) {
        std::vector<float> query_row(kSteps, 0.f);
        std::vector<uint8_t> pad_row(kSteps, 0);
        for (int col = step + 1; col < kSteps; ++col) {
            query_row[col] = neg_inf;
            pad_row[col] = 1;
        }
        upload_int64(tgt.ptr, tgt_dtype, target.data(), target.size());
        upload_floats(pos.ptr, pos_dtype, pos_queries.data() + static_cast<size_t>(step) * kWidth, kWidth);
        upload_floats(query.ptr, query_dtype, query_row.data(), query_row.size());
        upload_bool(padding.ptr, pad_dtype, pad_row.data(), pad_row.size());
        decoder.enqueue();
        download_floats(all_logits.data() + static_cast<size_t>(step) * kClasses, logit_dtype, logits.ptr, kClasses);
        if (step + 1 < kSteps) {
            const float* row = all_logits.data() + static_cast<size_t>(step) * kClasses;
            int arg = 0;
            float best = row[0];
            for (int cls = 1; cls < kClasses; ++cls) {
                if (row[cls] > best) {
                    best = row[cls];
                    arg = cls;
                }
            }
            target[step + 1] = arg;
        }
    }
    return all_logits;
}

}  // namespace

int main(int argc, char** argv)
{
    const std::string image_path = argc > 1 ? argv[1] : find_existing({
        "test_img/1_1.jpg",
        "c++_samples/test_img/1_1.jpg",
    });
    const std::string bpe_path = find_existing({
        "siglip2_bpe.bin",
        "c++_samples/siglip2_bpe.bin",
    });

    const SiglipBpe bpe = SiglipBpe::load(bpe_path);
    expect_tokens(bpe, "hello", {17534, kSiglipEos});

    const std::vector<float> visual_pos = load_pos_queries(kCheckpoint, "visual_decoder.pos_queries");
    const std::vector<float> cross_pos = load_pos_queries(kCheckpoint, "cross_decoder.pos_queries");

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        die("Failed to read image " + image_path);
    }
    const std::vector<float> input = preprocess(image);

    TrtEngine visual_encode(std::string(kEngineStem) + ".visual_encode.fp16.engine");
    TrtEngine visual_decode(std::string(kEngineStem) + ".visual_decode.fp16.engine");
    TrtEngine text_encode(std::string(kEngineStem) + ".text_encode.fp16.engine");
    TrtEngine text_decode(std::string(kEngineStem) + ".text_decode.fp16.engine");

    expect_shape(visual_encode, "imgs", {1, 3, kImage, kImage});
    expect_shape(visual_encode, "img_feature", {1, kTokens, kWidth});
    expect_shape(visual_decode, "img_feature", {1, kTokens, kWidth});
    expect_shape(text_encode, "text_token", {1, kTextLen});
    expect_shape(text_encode, "img_feature", {1, kTokens, kWidth});
    expect_shape(text_encode, "cross_memory", {1, kCrossLen, kWidth});
    expect_shape(text_decode, "cross_memory", {1, kCrossLen, kWidth});

    const nvinfer1::DataType image_dtype = visual_encode.dtype("imgs");
    const nvinfer1::DataType feature_dtype = visual_encode.dtype("img_feature");
    if (visual_decode.dtype("img_feature") != feature_dtype || text_encode.dtype("img_feature") != feature_dtype) {
        die("img_feature dtypes differ between visual_encode, visual_decode, and text_encode");
    }
    const nvinfer1::DataType memory_dtype = text_encode.dtype("cross_memory");
    if (text_decode.dtype("cross_memory") != memory_dtype) {
        die("cross_memory dtypes differ between text_encode and text_decode");
    }

    const int64_t feature_count = visual_encode.elements("img_feature");
    const int64_t memory_count = text_encode.elements("cross_memory");
    GpuMem images(static_cast<size_t>(visual_encode.elements("imgs")) * dtype_size(image_dtype));
    GpuMem features(static_cast<size_t>(feature_count) * dtype_size(feature_dtype));
    GpuMem tokens(static_cast<size_t>(text_encode.elements("text_token")) * dtype_size(text_encode.dtype("text_token")));
    GpuMem memory(static_cast<size_t>(memory_count) * dtype_size(memory_dtype));

    upload_floats(images.ptr, image_dtype, input.data(), input.size());
    visual_encode.bind("imgs", images.ptr);
    visual_encode.bind("img_feature", features.ptr);
    visual_encode.enqueue();

    const std::vector<float> visual_logits = greedy_decode(visual_decode, "img_feature", features, visual_pos);
    const Decoded visual = decode_logits(visual_logits);
    const std::string caption = charset_adapt(visual.text);
    // text_token is a static [1, 64] input. The checkpoint records context_length 16,
    // but this export was traced at the SigLIP2 length, so the ids stay padded to 64.
    const std::vector<int64_t> text_ids = bpe.tokenize(caption);
    std::cout << "visual caption: " << caption << std::endl;

    upload_int64(tokens.ptr, text_encode.dtype("text_token"), text_ids.data(), text_ids.size());
    text_encode.bind("text_token", tokens.ptr);
    text_encode.bind("img_feature", features.ptr);
    text_encode.bind("cross_memory", memory.ptr);
    text_encode.enqueue();

    const std::vector<float> text_logits = greedy_decode(text_decode, "cross_memory", memory, cross_pos);
    const Decoded text = decode_logits(text_logits);
    const std::string adapted = charset_adapt(text.text);

    std::cout << "text: " << text.text << std::endl;
    std::cout << "adapted: " << adapted << std::endl;
    std::cout << "confidence: " << text.confidence << std::endl;

    cv::putText(image, text.text, cv::Point(8, std::max(24, image.rows - 12)),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    const std::string out_path = image_path + "_v.jpg";
    cv::imwrite(out_path, image);
    std::cout << "saved " << out_path << std::endl;
    return 0;
}

#endif
