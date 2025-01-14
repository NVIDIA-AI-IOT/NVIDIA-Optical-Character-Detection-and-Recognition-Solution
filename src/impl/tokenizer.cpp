#include "tokenizer.h"
#include <glog/logging.h>
#include <algorithm>
#include <fstream>
#include <iostream>
namespace nvocdr {

// todo so tricky vocab init !!
void BPETokenizer::initVocab() {
  std::vector<std::string> tmp;
  for (size_t i = static_cast<size_t>(u'!'); i <= static_cast<size_t>(u'~'); ++i) {
    std::string k = mConverter.to_bytes(static_cast<wchar_t>(i));

    mEncoder.emplace(k, mEncoder.size());
    tmp.push_back(k);
    // LOG(INFO) << k << " " << v;
  }
  for (size_t i = static_cast<size_t>(u'¡'); i <= static_cast<size_t>(u'¬'); ++i) {
    std::string k = mConverter.to_bytes(static_cast<wchar_t>(i));
    //   std::string(1, static_cast<char>(i));
    //   size_t v = i - static_cast<size_t>(u'!');

    mEncoder.emplace(k, mEncoder.size());
    tmp.push_back(k);
  }
  for (size_t i = static_cast<size_t>(u'®'); i <= static_cast<size_t>(u'ÿ'); ++i) {
    std::string k = mConverter.to_bytes(static_cast<wchar_t>(i));
    //   std::string(1, static_cast<char>(i));
    //   size_t v = i - static_cast<size_t>(u'!');

    mEncoder.emplace(k, mEncoder.size());
    tmp.push_back(k);
  }

  for (int b = 0; b < 256; b++) {
    std::string k = mConverter.to_bytes(static_cast<wchar_t>(b));
    //   std::string(1, static_cast<char>(i));
    //   size_t v = i - static_cast<size_t>(u'!');

    if (mEncoder.count(k) == 0) {
      std::string c = mConverter.to_bytes(static_cast<wchar_t>(b + 256));
      mEncoder[c] = mEncoder.size();
      // LOG(INFO) << c << " " << mEncoder.size();
      tmp.push_back(c);
    }
  }

  for (auto const& k : tmp) {
    mEncoder.emplace(k + "</w>", mEncoder.size());
    // LOG(INFO) << k + "</w>" << " " << mEncoder.size();
  }
  // for (size_t i = static_cast<size_t>(u'!'); i <= static_cast<size_t>(u'~'); ++i) {
  // //   std::string k = mConverter.to_bytes(static_cast<wchar_t>(i));
  //   std::string k = mConverter.to_bytes(static_cast<wchar_t>(i)) + "</w>";
  // //   std::string(1, static_cast<char>(i));
  // //   size_t v = i - static_cast<size_t>(u'!');

  //   mEncoder.emplace(k, mEncoder.size());
  // }
  // for (size_t i = static_cast<size_t>(u'¡'); i <= static_cast<size_t>(u'¬'); ++i) {
  //   std::string k = mConverter.to_bytes(static_cast<wchar_t>(i))+ "</w>";
  // //   std::string(1, static_cast<char>(i));
  // //   size_t v = i - static_cast<size_t>(u'!');

  //   mEncoder.emplace(k, mEncoder.size());

  // }
  // for (size_t i = static_cast<size_t>(u'®'); i <= static_cast<size_t>(u'ÿ'); ++i) {
  //   std::string k = mConverter.to_bytes(static_cast<wchar_t>(i))+ "</w>";
  // //   std::string(1, static_cast<char>(i));
  // //   size_t v = i - static_cast<size_t>(u'!');

  //   mEncoder.emplace(k, mEncoder.size());
  //   LOG(INFO) << k << " " << mEncoder.size();

  // }

  // for (int i = 128; i <= 140; ++i) {
  //   std::string k = mConverter.to_bytes(static_cast<wchar_t>(i));
  // //   mEncoder.emplace(k, i);
  //   LOG(INFO) << k << " " << i << " " << k.size() ;
  // }
  // LOG(INFO) << "a" << mEncoder["a"];
}

BPETokenizer::BPETokenizer(const std::string& vocab_file, size_t cnt) {

  initVocab();
  // std::wifstream wif(vocab_file);
  // wif.imbue(std::locale(std::locale(), new std::codecvt_utf8<wchar_t>));

  // std::wstring wline;
  // std::getline(wif, wline);
  // size_t wcnt = cnt;
  // while (std::getline(wif, wline) && wcnt > 0) {
  //     std::wstringstream wiss(wline);
  //     std::wstring first, second;
  //     if (wiss >> first >> second) {
  //         // mMergeRule.emplace(std::make_pair(first, second), cnt);
  //         std::wcout << first << " " << second << "\n";
  //         wcnt--;
  //     }
  // }

  // std::wstringstream wss;

  // wss << wif.rdbuf();
  // return wss.str();

  std::ifstream file(vocab_file, std::ios::in);
  std::string line;
  std::getline(file, line);
  // Calculate max merges based on encoder size
  while (std::getline(file, line) && cnt > 0) {
    std::istringstream iss(line);
    std::string first, second;
    if (iss >> first >> second) {
      mMergeRule.emplace(std::make_pair(first, second), cnt);
      mEncoder[first + second] = mEncoder.size();
      // if (first == "fo" && second == "x</w>") {
      //     LOG(ERROR) << mEncoder.size() << " " << cnt;
      // }
      // if (first+second == "fox</w>") {
      //     LOG(ERROR) << mEncoder.size() << " " << cnt;
      // }
      // LOG(INFO) << first+second <<" "<< mEncoder.size();
      cnt--;
    }
  }
  file.close();

  LOG(ERROR) << "load vocab size: " << mEncoder.size();
  // Merge m = std::make_pair("ðŁĲ", "į</w>");
  // // auto &merge =
  // LOG(INFO) << mMergeRule[m];
  // for(auto const& [merge, rank]: mMergeRule) {
  //   LOG(INFO) << merge.first << " " << merge.second << ", rank " << rank;
  // }
}

void BPETokenizer::encode(const std::string& input, size_t max_len, int* dst) {
  std::string input_text = TokenPreProcess::process(input);
  std::transform(input_text.begin(), input_text.end(), input_text.begin(), ::tolower);
  // LOG(INFO) << text.size();
  // LOG(INFO) << "encode for " << input_text;

  std::sregex_iterator it(input_text.begin(), input_text.end(), mPat);
  std::sregex_iterator end;
  size_t out_cnt = 0;

  while (it != end) {
    std::string text = it->str();
    if (text.empty()) {
      it++;
      continue;
    }
    size_t n = text.size();
    text += "</w>";
    // LOG(INFO) << "encode for " << text;

    std::vector<size_t> pairs(n + 1);
    size_t num_pairs = n;
    for (size_t i = 0; i < n; ++i) {
      pairs[i] = i;
    }
    pairs[n] = n + 4;

    while (true) {
      int64_t mx = 0;
      size_t idx = 0;
      // std::string first;
      // std::string second;
      for (size_t i = 1; i <= num_pairs; ++i) {
        std::string first = text.substr(pairs[i - 1], pairs[i] - pairs[i - 1]);
        std::string second = text.substr(pairs[i], pairs[i + 1] - pairs[i]);
        if (mMergeRule.count({first, second}) && mMergeRule[{first, second}] > mx) {
          mx = mMergeRule[{first, second}];
          idx = i;

          // LOG(INFO) << first + second << " "<<mMergeRule[{first, second}];
        }
      }
      if (idx > 0) {
        for (size_t i = idx; i < pairs.size() - 1; ++i) {
          pairs[i] = pairs[i + 1];
        }
        num_pairs -= 1;
      } else {
        break;
      }
    }
    // LOG(INFO) << num_pairs;
    for (size_t i = 1; i <= num_pairs && out_cnt < max_len; ++i) {
      auto const s = text.substr(pairs[i - 1], pairs[i] - pairs[i - 1]);
      if (mEncoder.count(s) != 0) {
        *(dst + out_cnt) = static_cast<int>(mEncoder[s]);
        out_cnt += 1;
      };
    }
    ++it;
  }
  for (size_t i = out_cnt; i < max_len; ++i) {
    *(dst + i) = 0;
  }

  // for(size_t i = 0; i < text.size(); ++i) {
  //   LOG(INFO) << text[i];
  // }
}

}  // namespace nvocdr
