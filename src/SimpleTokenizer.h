#ifndef __NVOCDR_SIMPLETOKENIZER_HEADER__
#define __NVOCDR_SIMPLETOKENIZER_HEADER__

#include <string>
#include <regex>
#include <set>
#include <algorithm> 
#include <codecvt>
#include <vector>
#include <unordered_map>

namespace nvocdr
{

class SimpleTokenizer 
{
    private:
        // Member variables
        std::map<int, std::wstring> mByteEncoder; // Byte-to-Unicode encoder
        std::unordered_map<std::wstring, int> mByteDecoder; // Unicode-to-Byte decoder
        std::vector<std::pair<std::string, std::string>> mMerges; // BPE merge rules
        std::map<std::pair<std::string, std::string>, int> mBpeRanks; // Merge ranks
        std::unordered_map<std::wstring, int> mEncoder; // Token-to-ID mapping
        std::unordered_map<int, std::wstring> mDecoder; // ID-to-Token mapping
        std::unordered_map<std::string, std::string> mCache; // Cache for BPE results
        std::regex mPat = std::regex(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]+|[^\s\w]+)"); // Regular expression for tokenization
        std::wstring_convert<std::codecvt_utf8<wchar_t>> mConverter;
        int mStartTextToken;
        int mEndTextToken;

    public:
        // Constructor: Initialize tokenizer with BPE rules and vocabulary size
        // SimpleTokenizer();
        

        SimpleTokenizer(const std::string& bpe_path, int vocab_size = 32000);
        ~SimpleTokenizer(){};
        
        void initTokenizer(const std::string& bpe_path, int vocab_size);

        std::string bpe(const std::string& token);

        // Encode function
        std::vector<int> encode(const std::string& text);


        // Decode function
        std::string decode(const std::vector<int>& tokens);

        std::vector<std::pair<int, std::wstring>> bytes_to_unicode();

        std::set<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word);

        std::string whitespace_clean(const std::string& text);

        // Function to decode HTML entities
        std::string html_unescape(const std::string& input);

        // Function to trim leading and trailing whitespace
        std::string trim(const std::string& str);

        // Main `basic_clean` function
        std::string basic_clean(const std::string& text);

        int getStartTextToken() {return mStartTextToken;};  
        int getEndTextToken() {return mEndTextToken;};

};

}


#endif