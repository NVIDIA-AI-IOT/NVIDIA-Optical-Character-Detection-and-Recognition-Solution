#ifndef __NVOCDR_OCRN_HEADER__
#define __NVOCDR_OCRN_HEADER__
#include <string>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <algorithm> 
#include <codecvt>
#include "MemManager.h"
#include "TRTEngine.h"

#define OCRNET_OUTPUT_ID "output_id"
#define OCRNET_OUTPUT_PROB "output_prob"

using namespace nvinfer1;
namespace nvocdr
{

enum DecodeMode
{
    CTC,
    Attention,
    Transformer
};


class SimpleTokenizer 
{
    private:
        // Member variables
        std::map<int, std::wstring> byte_encoder; // Byte-to-Unicode encoder
        std::unordered_map<std::wstring, int> byte_decoder; // Unicode-to-Byte decoder
        std::vector<std::pair<std::string, std::string>> merges; // BPE merge rules
        std::map<std::pair<std::string, std::string>, int> bpe_ranks; // Merge ranks
        std::unordered_map<std::wstring, int> encoder; // Token-to-ID mapping
        std::unordered_map<int, std::wstring> decoder; // ID-to-Token mapping
        std::unordered_map<std::string, std::string> cache; // Cache for BPE results
        std::regex pat = std::regex(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]+|[^\s\w]+)"); // Regular expression for tokenization
        std::wstring_convert<std::codecvt_utf8<wchar_t>> mConverter;
        int mStartTextToken;
        int mEndTextToken;

    public:
        // Constructor: Initialize tokenizer with BPE rules and vocabulary size
        SimpleTokenizer(){};

        SimpleTokenizer(const std::string& bpe_path, int vocab_size = 32000){
            initTokenizer(bpe_path, vocab_size);
        }
        
        void initTokenizer(const std::string& bpe_path, int vocab_size)
        {
            // Initialize byte encoder/decoder
            std::vector<std::pair<int, std::wstring>> b2u = bytes_to_unicode();
            for (int i = 0; i < b2u.size(); ++i) {
                byte_encoder[b2u[i].first] = b2u[i].second;
                encoder[b2u[i].second] = i;
                encoder[b2u[i].second + L"</w>"] = i + b2u.size();
            }

            for (const auto& [k, v] : byte_encoder) {
                byte_decoder[v] = k;
            }

            // Load BPE merges from file
            std::ifstream file(bpe_path, std::ios::in);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open BPE file: " + bpe_path);
            }

            // Read the BPE merges file and process it
            std::string line;
            std::getline(file, line);
            // Calculate max merges based on encoder size
            int max_merges = vocab_size - 256 - 256 - 2 ; 
            while (std::getline(file, line) && max_merges-- > 0) {
                std::istringstream iss(line);
                std::string first, second;
                if (iss >> first >> second) {
                    merges.emplace_back(first, second);
                }
            }
            file.close();

            // Build BPE ranks
            for (size_t i = 0; i < merges.size(); ++i) {
                bpe_ranks[merges[i]] = static_cast<int>(i);
            }

            for (const auto& merge : merges) {
                std::string key_merge = merge.first + merge.second;
                std::wstring key_merge_w = mConverter.from_bytes(key_merge);
                encoder[key_merge_w] = encoder.size();

            }
            encoder[L"<|startoftext|>"] = encoder.size();
            encoder[L"<|endoftext|>"] = encoder.size();
            for (const auto& [key, value] : encoder) {
                decoder[value] = key;
            }

            cache["<|startoftext|>"] = "<|startoftext|>";
            cache["<|endoftext|>"] = "<|endoftext|>";
            mStartTextToken = encoder[L"<|startoftext|>"];
            mEndTextToken = encoder[L"<|endoftext|>"];
        };

        std::string bpe(const std::string& token) {

            if (cache.find(token) != cache.end()) {
                return cache[token];
            }

            std::vector<std::string> word;
            for (size_t i = 0; i < token.size(); ++i) {
                word.push_back(std::string(1, token[i]));
            }
            word.back() += "</w>";

            auto pairs = get_pairs(word);

            if (pairs.empty()) {
                return token + "</w>";
            }

            while (true) {
                auto bigram = *std::min_element(
                    pairs.begin(), pairs.end(),
                    [this](const std::pair<std::string, std::string>& a, const std::pair<std::string, std::string>& b) {
                        return bpe_ranks.count(a) ? (bpe_ranks.count(b) ? bpe_ranks.at(a) < bpe_ranks.at(b) : true) : false;
                    });

                if (bpe_ranks.find(bigram) == bpe_ranks.end()) {
                    break;
                }

                std::vector<std::string> new_word;
                size_t i = 0;
                while (i < word.size()) {
                    auto it = std::find(word.begin() + i, word.end(), bigram.first);
                    if (it != word.end() && it + 1 != word.end() && *(it + 1) == bigram.second) {
                        new_word.insert(new_word.end(), word.begin() + i, it);
                        new_word.push_back(bigram.first + bigram.second);
                        i = it - word.begin() + 2;
                    } else {
                        new_word.push_back(word[i]);
                        ++i;
                    }
                }

                word = new_word;

                if (word.size() == 1) {
                    break;
                }

                pairs = get_pairs(word);
            }

            std::ostringstream result;
            for (const auto& w : word) {
                result << w << " ";
            }
            std::string final_result = result.str();
            final_result.pop_back(); 

            cache[token] = final_result;
            return final_result;
        };

        // Encode function
        std::vector<int> encode(const std::string& text) {
            std::vector<int> bpe_tokens;
            
            // Clean and lowercase the input text
            std::string cleaned_text = whitespace_clean(basic_clean(text));
            std::transform(cleaned_text.begin(), cleaned_text.end(), cleaned_text.begin(), ::tolower);
            // Tokenize the cleaned text using regex
            std::sregex_iterator it(cleaned_text.begin(), cleaned_text.end(), pat);
            std::sregex_iterator end;

            while (it != end) {
                if (it->str().empty()) 
                {
                    continue;
                }
                std::string token = it->str();
                
                // Convert token to its byte representation
                std::string byte_representation;
                for (char b : token) {
                    
                    byte_representation += mConverter.to_bytes(byte_encoder[static_cast<char>(b)]);
                }
                // Apply BPE to the byte representation
                std::string bpe_result = bpe(byte_representation);

                // Encode each BPE token to its corresponding ID
                std::istringstream iss(bpe_result);
                std::string bep_str;
                while (std::getline(iss, bep_str, ' ')) {
                    std::wstring bpe_token = mConverter.from_bytes(bep_str);
                    bpe_tokens.push_back(encoder[bpe_token]);
                }
                
                ++it;
            }

            return bpe_tokens;
        };


        // Decode function
        std::string decode(const std::vector<int>& tokens) {
            // std::ostringstream decoded_text;
            std::string result;
            for (int token : tokens) {
                if (decoder.find(token) != decoder.end()) {
                    result = mConverter.to_bytes(decoder[token]);
                }
            }

            // Convert the decoded string to a byte array and replace </w> with space
            result = std::regex_replace(result, std::regex("</w>"), " "); // Replace </w> with space

            return result;
        };

        std::vector<std::pair<int, std::wstring>> bytes_to_unicode() {
            std::vector<int> bs;
            for (int i = static_cast<int>(u'!'); i <= static_cast<int>(u'~'); ++i) bs.push_back(i);
            for (int i = static_cast<int>(u'¡'); i <= static_cast<int>(u'¬'); ++i) bs.push_back(i);
            for (int i = static_cast<int>(u'®'); i <= static_cast<int>(u'ÿ'); ++i) bs.push_back(i);
            
            std::vector<int> cs = bs;
            int n = 0;
            for (int b = 0; b < 256; b++) {
                if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                    bs.push_back(b);
                    cs.push_back(256 + n);
                    n++;
                }
            }
            
            std::vector<std::pair<int, std::wstring>> result;
            for (size_t i = 0; i < bs.size(); i++) {
                result.push_back(std::make_pair(bs[i], std::wstring(1, static_cast<wchar_t>(cs[i]))));
            }
            
            return result;
        }

        std::set<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word) {
            std::set<std::pair<std::string, std::string>> pairs;
            for (size_t i = 0; i < word.size() - 1; ++i) {
                pairs.emplace(word[i], word[i + 1]);
            }
            return pairs;
        };

        std::string whitespace_clean(const std::string& text) {
            std::regex whitespace_regex("\\s+");
            std::string result = std::regex_replace(text, whitespace_regex, " ");

            size_t start = result.find_first_not_of(" ");
            size_t end = result.find_last_not_of(" ");
            
            if (start == std::string::npos) {
                return "";
            }
            
            return result.substr(start, end - start + 1);
        };

        // Function to decode HTML entities
        std::string html_unescape(const std::string& input) {
            static const std::unordered_map<std::string, std::string> html_entities = {
                {"&quot;", "\""}, {"&amp;", "&"}, {"&lt;", "<"},
                {"&gt;", ">"},   {"&nbsp;", " "}, {"&#39;", "'"}
            };

            std::string output = input;
            for (const auto& [entity, character] : html_entities) {
                size_t pos = 0;
                while ((pos = output.find(entity, pos)) != std::string::npos) {
                    output.replace(pos, entity.length(), character);
                    pos += character.length();
                }
            }
            return output;
        };

        // Function to trim leading and trailing whitespace
        std::string trim(const std::string& str) {
            size_t start = str.find_first_not_of(" \t\n\r");
            size_t end = str.find_last_not_of(" \t\n\r");
            return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
        };

        // Main `basic_clean` function
        std::string basic_clean(const std::string& text) {
            std::string cleaned_text = text;

            // Decode HTML entities twice
            cleaned_text = html_unescape(cleaned_text);
            cleaned_text = html_unescape(cleaned_text);

            // Trim whitespace
            cleaned_text = trim(cleaned_text);

            return cleaned_text;
        };

        int getStartTextToken() {return mStartTextToken;};  
        int getEndTextToken() {return mEndTextToken;};

};


class OCRNetEngine
{
    public:
        OCRNetEngine(const std::string& engine_path, const std::string& dict_path,
                     const bool upside_down, const DecodeMode decode_mode, const std::string& text_engine_path="", bool only_alnum=true, bool only_lowercase=true, const std::string& vocab_file="", const int vocab_size=32000);
        ~OCRNetEngine();

        bool initTRTBuffer(BufferManager& buffer_mgr);
        int32_t getMaxBatchSize() {return mEngine->getMaxBatchSize(); };
        bool setInputShape(const Dims& input_shape);
        bool setInputDeviceBuffer(DeviceBuffer& device_buffer, const int index);
        bool setOutputDeviceBuffer(DeviceBuffer& device_buffer, const int index);
        bool infer(BufferManager& buffer_mgr, std::vector<std::pair<std::string, float>>& de_texts,
                   const cudaStream_t& stream = 0);
        std::pair<std::string, float> clip4strDecode( const std::vector<float>& output_prob, const int batch_idx, const int context_len, const int charset_len);
        int mTRTInputBufferIndex;
        int mTRTOutputBufferIndex;
        // CLIP4STR buffer
        int mVisTRTOutputDecodeProbsBufferIndex;
        int mVisTRTOutputImgFeatureBufferIndex;
        int mVisTRTOutputContextBufferIndex;

        int mTextTRTInputTextTokenBufferIndex;
        int mTextTRTOutputBufferIndex;
    private:
        std::unique_ptr<TRTEngine> mEngine;
        std::vector<std::string> mDict;
        bool mUDFlag;
        DecodeMode mDecodeMode;

        // CLIP4STR
        std::unique_ptr<TRTEngine> mTextEngine;
        std::string mVisualOutDecodeProbName = "visual_decode_probs";
        std::string mVisualOutContextName = "tgt_in";
        std::string mVisualOutImgFeaturebName = "img_feature";
        std::string mTextInTokenName = "text_token";
        std::string mTextOutLogitName = "logits";
        int mVocabSize = 32000;
        int mMaxContextLen = 16;
        // only output alpha and digit
        bool mOnlyAlNum = true;
        // only output lower case alpha
        bool mOnlyLowerCase = true;
        SimpleTokenizer mTokenizer;
};
}
#endif