#pragma once

#include <codecvt>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

namespace nvocdr {
struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& p) const {
    // auto h1 = std::hash<T1>{}(p.first);
    // auto h2 = std::hash<T2>{}(p.second);

    // // Mainly for demonstration purposes, i.e. works but is overly simple
    // // In the real world, use sth. like boost.hash_combine
    return std::hash<T1>{}(p.first + p.second);
  }
};

using Merge = std::pair<std::string, std::string>;
using MergeRule = std::unordered_map<Merge, int, pair_hash>;

class BPETokenizer {
 public:
  BPETokenizer(const std::string& vocab_file, size_t cnt);
  void encode(const std::string& input, size_t max_len, int* dst);
  void decode(const std::vector<int>& embedding);

 private:
  void initVocab();
  std::unordered_map<std::string, size_t> mEncoder;
  std::unordered_map<size_t, std::string> mDecoder;
  MergeRule mMergeRule;
  std::wstring_convert<std::codecvt_utf8<wchar_t>> mConverter;
  std::regex mPat = std::regex(
      R"(<|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+)");
};

// todo simpliy
class TokenPreProcess {
 public:
  static std::string process(const std::string& input) {
    return whitespace_clean(trim(html_unescape(html_unescape(input))));
  }

  static std::string html_unescape(const std::string& input) {
    static const std::unordered_map<std::string, std::string> html_entities = {
        {"&quot;", "\""}, {"&amp;", "&"},  {"&lt;", "<"},
        {"&gt;", ">"},    {"&nbsp;", " "}, {"&#39;", "'"}};

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
  static std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    size_t end = str.find_last_not_of(" \t\n\r");
    return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
  };

  static std::string whitespace_clean(const std::string& text) {
    std::regex whitespace_regex("\\s+");
    std::string result = std::regex_replace(text, whitespace_regex, " ");

    size_t start = result.find_first_not_of(" ");
    size_t end = result.find_last_not_of(" ");

    if (start == std::string::npos) {
      return "";
    }

    return result.substr(start, end - start + 1);
  };
};
}  // namespace nvocdr
