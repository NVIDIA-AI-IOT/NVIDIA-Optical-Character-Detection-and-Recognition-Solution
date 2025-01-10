#include <memory>
#include <string>
#include <gtest/gtest.h>
#include "tokenizer.h"

TEST(TEST_BPE_TOKENIZER, BASIC) {
    std::unique_ptr<nvocdr::BPETokenizer> tokenizer = std::make_unique<nvocdr::BPETokenizer>("/home/csh/nvocdr/onnx_models/bpe_simple_vocab_16e6.txt", 32000);

    // tokenizer->encode("ðŁĲ į");
    static constexpr size_t MAX_LEN = 16;
    std::vector<int> result(MAX_LEN);
    // tokenizer->encode("hello", MAX_LEN, result.data());
    // ASSERT_EQ(result[0], 3306);
    // tokenizer->encode("world", MAX_LEN, result.data());
    // ASSERT_EQ(result[0], 1002);


    tokenizer->encode("FOX", MAX_LEN, result.data());
    ASSERT_EQ(result[0], 3240);

    tokenizer->encode("1947", MAX_LEN, result.data());

    tokenizer->encode("Since", MAX_LEN, result.data());
    ASSERT_EQ(result[0], 1855);

    tokenizer->encode("WASH", MAX_LEN, result.data());
    ASSERT_EQ(result[0], 7810);

    

    // for(size_t i = 0; i < MAX_LEN; ++i) {
    //     std::cout<< " " << result[i] << " ";
    // }


    // tokenizer->encode("I'm AI");
    // tokenizer->encode("FloydHub is the fastest way to build, train and deploy deep learning models. Build deep learning models in the cloud. Train deep learning models.");
    // tokenizer->encode("Since");
}