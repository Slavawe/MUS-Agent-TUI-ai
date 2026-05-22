#include "mus_tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <set>

static const std::set<std::string> CPP_KEYWORDS = {
    "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor",
    "bool", "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t",
    "class", "concept", "const", "consteval", "constexpr", "constinit",
    "continue", "co_await", "co_return", "co_yield", "decltype", "default",
    "delete", "do", "double", "dynamic_cast", "else", "enum", "explicit",
    "export", "extern", "false", "float", "for", "friend", "goto", "if",
    "inline", "int", "long", "mutable", "namespace", "new", "noexcept",
    "not", "not_eq", "nullptr", "operator", "or", "or_eq", "override",
    "private", "protected", "public", "register", "reinterpret_cast",
    "requires", "return", "short", "signed", "sizeof", "static",
    "static_assert", "static_cast", "struct", "switch", "template",
    "this", "thread_local", "throw", "true", "try", "typedef", "typeid",
    "typename", "union", "unsigned", "using", "virtual", "void",
    "volatile", "wchar_t", "while", "xor", "xor_eq"
};

static const std::set<std::string> CPP_TYPES = {
    "int", "long", "short", "char", "wchar_t", "char8_t", "char16_t", "char32_t",
    "float", "double", "bool", "void", "size_t", "ptrdiff_t", "int8_t",
    "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t",
    "uint64_t", "string", "vector", "map", "set", "unordered_map",
    "unordered_set", "shared_ptr", "unique_ptr", "weak_ptr", "optional",
    "variant", "any", "span", "string_view", "array", "tuple", "pair"
};

CPPTokenizer::CPPTokenizer() : base_id_(2201), vocab_size_(48000) {
    build_keywords();
    build_operators();
    build_preprocessor();
}

CPPTokenizer::CPPTokenizer(int base_id) : base_id_(base_id), vocab_size_(48000) {
    build_keywords();
    build_operators();
    build_preprocessor();
}

void CPPTokenizer::build_keywords() {
    int id = base_id_;
    for (const auto& kw : CPP_KEYWORDS)
        keyword_ids_[kw] = id++;
}

void CPPTokenizer::build_operators() {
    static const char* ops[] = {
        "::", "->", "=>", "++", "--", "<<", ">>", "<=>", "<=", ">=", "==",
        "!=", "&&", "||", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
        "<<=", ">>=", "->*", ".*", "##", "..", "...", ".*", "+", "-",
        "*", "/", "%", "^", "&", "|", "~", "!", "=", "<", ">",
    };
    int id = base_id_ + 200;
    for (const char* op : ops)
        operator_ids_[op] = id++;
}

void CPPTokenizer::build_preprocessor() {
    static const char* dirs[] = {
        "#include", "#define", "#undef", "#ifdef", "#ifndef", "#if",
        "#else", "#elif", "#endif", "#pragma", "#error", "#line", "#import",
        "#include_next"
    };
    int id = base_id_ + 300;
    for (const char* d : dirs)
        prep_ids_[d] = id++;
}

int CPPTokenizer::assign_id(const std::string& text, std::unordered_map<std::string, int>& map) const {
    auto it = map.find(text);
    if (it != map.end()) return it->second;
    int id = base_id_ + 400 + (int)map.size();
    map[text] = id;
    return id;
}

bool CPPTokenizer::is_cpp_keyword(const std::string& word) {
    return CPP_KEYWORDS.count(word) > 0;
}

bool CPPTokenizer::is_cpp_type(const std::string& word) {
    return CPP_TYPES.count(word) > 0;
}

std::string CPPTokenizer::token_type_name(CPPTokenType t) {
    switch (t) {
        case CPPTokenType::Keyword:      return "keyword";
        case CPPTokenType::Identifier:   return "identifier";
        case CPPTokenType::NumericLiteral: return "number";
        case CPPTokenType::StringLiteral: return "string";
        case CPPTokenType::CharLiteral:  return "char";
        case CPPTokenType::BoolLiteral:  return "bool";
        case CPPTokenType::NullptrLiteral: return "nullptr";
        case CPPTokenType::Operator:     return "operator";
        case CPPTokenType::Punctuation:  return "punctuation";
        case CPPTokenType::Preprocessor: return "preprocessor";
        case CPPTokenType::Comment:      return "comment";
        case CPPTokenType::Whitespace:   return "whitespace";
        case CPPTokenType::Unknown:      return "unknown";
        case CPPTokenType::EndOfFile:    return "eof";
    }
    return "?";
}

std::string CPPToken::type_name() const {
    return CPPTokenizer::token_type_name(type);
}

bool CPPTokenizer::try_consume_preprocessor(const std::string& src, size_t& pos, CPPToken& tok) const {
    if (pos >= src.size() || src[pos] != '#') return false;
    size_t start = pos;
    while (pos < src.size() && (src[pos] != '\n' && src[pos] != '\r'))
        pos++;
    std::string text = src.substr(start, pos - start);
    std::string directive;
    size_t d_end = text.find_first_of(" \t");
    if (d_end != std::string::npos)
        directive = text.substr(0, d_end);
    else
        directive = text;
    int id = assign_id(text, const_cast<std::unordered_map<std::string, int>&>(prep_ids_));
    auto it = prep_ids_.find(directive);
    if (it != prep_ids_.end()) id = it->second;
    tok = {CPPTokenType::Preprocessor, text, id, 0, (int)start};
    return true;
}

bool CPPTokenizer::try_consume_comment(const std::string& src, size_t& pos, CPPToken& tok) const {
    if (pos >= src.size()) return false;
    size_t start = pos;
    if (pos + 1 < src.size() && src[pos] == '/' && src[pos + 1] == '/') {
        pos += 2;
        while (pos < src.size() && src[pos] != '\n') pos++;
        std::string text = src.substr(start, pos - start);
        tok = {CPPTokenType::Comment, text, base_id_ + 500, 0, (int)start};
        return true;
    }
    if (pos + 1 < src.size() && src[pos] == '/' && src[pos + 1] == '*') {
        pos += 2;
        while (pos + 1 < src.size() && !(src[pos] == '*' && src[pos + 1] == '/')) pos++;
        if (pos + 1 < src.size()) pos += 2;
        std::string text = src.substr(start, pos - start);
        tok = {CPPTokenType::Comment, text, base_id_ + 501, 0, (int)start};
        return true;
    }
    return false;
}

bool CPPTokenizer::try_consume_string(const std::string& src, size_t& pos, CPPToken& tok) const {
    if (pos >= src.size() || src[pos] != '"') return false;
    size_t start = pos;
    pos++;
    while (pos < src.size()) {
        if (src[pos] == '\\') { pos += 2; continue; }
        if (src[pos] == '"') { pos++; break; }
        pos++;
    }
    std::string text = src.substr(start, pos - start);
    tok = {CPPTokenType::StringLiteral, text, base_id_ + 510, 0, (int)start};
    return true;
}

bool CPPTokenizer::try_consume_char(const std::string& src, size_t& pos, CPPToken& tok) const {
    if (pos >= src.size() || src[pos] != '\'') return false;
    size_t start = pos;
    pos++;
    if (pos < src.size() && src[pos] == '\\') pos += 2;
    else pos++;
    if (pos < src.size() && src[pos] == '\'') pos++;
    std::string text = src.substr(start, pos - start);
    tok = {CPPTokenType::CharLiteral, text, base_id_ + 520, 0, (int)start};
    return true;
}

bool CPPTokenizer::try_consume_number(const std::string& src, size_t& pos, CPPToken& tok) const {
    if (pos >= src.size()) return false;
    char c = src[pos];
    if (!std::isdigit(c) && c != '.') return false;
    size_t start = pos;
    bool is_hex = false;
    bool is_float = false;
    if (c == '0' && pos + 1 < src.size() && (src[pos + 1] == 'x' || src[pos + 1] == 'X')) {
        pos += 2; is_hex = true;
        while (pos < src.size() && std::isxdigit(src[pos])) pos++;
    } else if (c == '0' && pos + 1 < src.size() && (src[pos + 1] == 'b' || src[pos + 1] == 'B')) {
        pos += 2;
        while (pos < src.size() && (src[pos] == '0' || src[pos] == '1')) pos++;
    } else {
        while (pos < src.size() && std::isdigit(src[pos])) pos++;
        if (pos < src.size() && src[pos] == '.') { is_float = true; pos++;
            while (pos < src.size() && std::isdigit(src[pos])) pos++; }
        if (pos < src.size() && (src[pos] == 'e' || src[pos] == 'E')) { pos++;
            if (pos < src.size() && (src[pos] == '+' || src[pos] == '-')) pos++;
            while (pos < src.size() && std::isdigit(src[pos])) pos++;
            is_float = true; }
    }
    if (pos < src.size() && (src[pos] == 'f' || src[pos] == 'F' || src[pos] == 'u' ||
                             src[pos] == 'U' || src[pos] == 'l' || src[pos] == 'L')) pos++;
    std::string text = src.substr(start, pos - start);
    tok = {CPPTokenType::NumericLiteral, text, base_id_ + 530, 0, (int)start};
    return true;
}

bool CPPTokenizer::try_consume_identifier(const std::string& src, size_t& pos, CPPToken& tok) const {
    if (pos >= src.size()) return false;
    char c = src[pos];
    if (!std::isalpha(c) && c != '_') return false;
    size_t start = pos;
    while (pos < src.size() && (std::isalnum(src[pos]) || src[pos] == '_'))
        pos++;
    std::string word = src.substr(start, pos - start);
    CPPTokenType type = CPPTokenType::Identifier;
    int id;
    if (word == "true" || word == "false") {
        type = CPPTokenType::BoolLiteral;
        id = base_id_ + 540;
    } else if (word == "nullptr") {
        type = CPPTokenType::NullptrLiteral;
        id = base_id_ + 541;
    } else if (is_cpp_keyword(word)) {
        type = CPPTokenType::Keyword;
        auto it = keyword_ids_.find(word);
        id = (it != keyword_ids_.end()) ? it->second : assign_id(word, const_cast<std::unordered_map<std::string, int>&>(keyword_ids_));
    } else {
        id = assign_id(word, const_cast<std::unordered_map<std::string, int>&>(keyword_ids_));
    }
    tok = {type, word, id, 0, (int)start};
    return true;
}

bool CPPTokenizer::try_consume_operator(const std::string& src, size_t& pos, CPPToken& tok) const {
    if (pos >= src.size()) return false;
    std::string best;
    for (int len = 3; len >= 1; len--) {
        if (pos + len <= src.size()) {
            std::string sub = src.substr(pos, len);
            if (operator_ids_.count(sub)) { best = sub; break; }
        }
    }
    if (best.empty()) return false;
    pos += best.size();
    tok = {CPPTokenType::Operator, best, operator_ids_.at(best), 0, (int)(pos - best.size())};
    return true;
}

bool CPPTokenizer::try_consume_punctuation(const std::string& src, size_t& pos, CPPToken& tok) const {
    static const std::string punct = "{}()[],.;:?";
    if (pos >= src.size()) return false;
    char c = src[pos];
    if (punct.find(c) == std::string::npos) return false;
    pos++;
    int id = base_id_ + 600 + (int)(punct.find(c));
    tok = {CPPTokenType::Punctuation, std::string(1, c), id, 0, (int)(pos - 1)};
    return true;
}

std::vector<CPPToken> CPPTokenizer::tokenize(const std::string& source) const {
    std::vector<CPPToken> tokens;
    size_t pos = 0;
    while (pos < source.size()) {
        CPPToken tok;
        if (source[pos] == ' ' || source[pos] == '\t' || source[pos] == '\n' || source[pos] == '\r') {
            size_t start = pos;
            while (pos < source.size() && (source[pos] == ' ' || source[pos] == '\t' ||
                                           source[pos] == '\n' || source[pos] == '\r')) pos++;
            tokens.push_back({CPPTokenType::Whitespace, source.substr(start, pos - start),
                              base_id_ + 700, 0, (int)start});
            continue;
        }
        if (try_consume_comment(source, pos, tok)) { tokens.push_back(tok); continue; }
        if (try_consume_preprocessor(source, pos, tok)) { tokens.push_back(tok); continue; }
        if (try_consume_string(source, pos, tok)) { tokens.push_back(tok); continue; }
        if (try_consume_char(source, pos, tok)) { tokens.push_back(tok); continue; }
        if (try_consume_number(source, pos, tok)) { tokens.push_back(tok); continue; }
        if (try_consume_identifier(source, pos, tok)) { tokens.push_back(tok); continue; }
        if (try_consume_operator(source, pos, tok)) { tokens.push_back(tok); continue; }
        if (try_consume_punctuation(source, pos, tok)) { tokens.push_back(tok); continue; }
        tok = {CPPTokenType::Unknown, std::string(1, source[pos]), base_id_ + 999, 0, (int)pos};
        tokens.push_back(tok);
        pos++;
    }
    tokens.push_back({CPPTokenType::EndOfFile, "", base_id_ + 1000, 0, (int)pos});
    return tokens;
}

std::vector<int64_t> CPPTokenizer::tokenize_to_ids(const std::string& source) const {
    auto tokens = tokenize(source);
    std::vector<int64_t> ids;
    ids.reserve(tokens.size());
    for (const auto& t : tokens)
        ids.push_back(t.token_id);
    return ids;
}

std::string CPPTokenizer::detokenize(const std::vector<int64_t>& ids) const {
    std::string result;
    for (int64_t id : ids) {
        bool found = false;
        for (const auto& [k, v] : keyword_ids_) {
            if (v == id) { result += k; found = true; break; }
        }
        if (found) continue;
        for (const auto& [k, v] : operator_ids_) {
            if (v == id) { result += k; found = true; break; }
        }
        if (found) continue;
        if (!result.empty()) result += ' ';
        result += "<ID:" + std::to_string(id) + ">";
    }
    return result;
}

std::vector<CPPToken> CPPTokenizer::tokenize_file(const std::string& path) const {
    std::ifstream f(path);
    if (!f) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return tokenize(ss.str());
}

std::vector<int64_t> CPPTokenizer::tokenize_file_to_ids(const std::string& path) const {
    auto tokens = tokenize_file(path);
    std::vector<int64_t> ids;
    ids.reserve(tokens.size());
    for (const auto& t : tokens)
        ids.push_back(t.token_id);
    return ids;
}
