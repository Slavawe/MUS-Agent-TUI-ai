#pragma once

#include "mus_model.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

// ══════════════════════════════════════════════════════════════════════
//  Uragan 1.0 — C++ Source Code Tokenizer
//
//  Токенизирует C++ код в ID словаря Uragan.
//  Диапазоны токенов:
//    [2201, 48000]  BPE-словарь — текстовые токены
//
//  Категории токенов C++:
//    - Keywords       (auto, const, int, class, template ...)
//    - Identifiers    (имена переменных, функций, типов)
//    - Literals       (числа, строки, char, boolean, nullptr)
//    - Operators      (+, -, ->, ::, <<, && ...)
//    - Punctuation    ({, }, (, ), ;, ,)
//    - Preprocessor   (#include, #define, #ifdef ...)
//    - Comments       (// и /* */)
// ══════════════════════════════════════════════════════════════════════

enum class CPPTokenType {
    Keyword,
    Identifier,
    NumericLiteral,
    StringLiteral,
    CharLiteral,
    BoolLiteral,
    NullptrLiteral,
    Operator,
    Punctuation,
    Preprocessor,
    Comment,
    Whitespace,
    Unknown,
    EndOfFile
};

struct CPPToken {
    CPPTokenType type;
    std::string text;
    int64_t token_id;
    int line;
    int column;

    std::string type_name() const;
};

class CPPTokenizer {
public:
    CPPTokenizer();

    explicit CPPTokenizer(int base_id);

    std::vector<CPPToken> tokenize(const std::string& source) const;

    std::vector<int64_t> tokenize_to_ids(const std::string& source) const;

    std::string detokenize(const std::vector<int64_t>& ids) const;

    std::vector<CPPToken> tokenize_file(const std::string& path) const;

    std::vector<int64_t> tokenize_file_to_ids(const std::string& path) const;

    int base_id() const { return base_id_; }
    int vocab_size() const { return vocab_size_; }

    static std::string token_type_name(CPPTokenType t);

private:
    int base_id_;
    int vocab_size_;

    std::unordered_map<std::string, int> keyword_ids_;
    std::unordered_map<std::string, int> operator_ids_;
    std::unordered_map<std::string, int> prep_ids_;

    void build_keywords();
    void build_operators();
    void build_preprocessor();

    int assign_id(const std::string& text, std::unordered_map<std::string, int>& map) const;

    static bool is_cpp_keyword(const std::string& word);
    static bool is_cpp_type(const std::string& word);

    bool try_consume_preprocessor(const std::string& src, size_t& pos, CPPToken& tok) const;
    bool try_consume_comment(const std::string& src, size_t& pos, CPPToken& tok) const;
    bool try_consume_string(const std::string& src, size_t& pos, CPPToken& tok) const;
    bool try_consume_char(const std::string& src, size_t& pos, CPPToken& tok) const;
    bool try_consume_number(const std::string& src, size_t& pos, CPPToken& tok) const;
    bool try_consume_identifier(const std::string& src, size_t& pos, CPPToken& tok) const;
    bool try_consume_operator(const std::string& src, size_t& pos, CPPToken& tok) const;
    bool try_consume_punctuation(const std::string& src, size_t& pos, CPPToken& tok) const;
};
