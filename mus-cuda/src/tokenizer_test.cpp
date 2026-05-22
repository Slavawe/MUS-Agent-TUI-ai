#include "mus_tokenizer.h"
#include <iostream>
#include <iomanip>

int main() {
    CPPTokenizer tok;

    const char* code = R"cpp(
#include <iostream>
#include <vector>

// Fibonacci sequence
int fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

int main() {
    const int N = 10;
    for (int i = 0; i < N; i++) {
        std::cout << "fib(" << i << ") = " << fib(i) << std::endl;
    }
    return 0;
}
)cpp";

    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Uragan C++ Tokenizer — демонстрация                       ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    auto tokens = tok.tokenize(code);
    std::cout << "Всего токенов: " << tokens.size() << "\n\n";

    std::cout << "┌──────┬─────────────────┬─────────────┬──────────┐\n";
    std::cout << "│  ID  │ Текст           │ Тип          │ Token ID │\n";
    std::cout << "├──────┼─────────────────┼─────────────┼──────────┤\n";

    for (const auto& t : tokens) {
        if (t.type == CPPTokenType::Whitespace || t.type == CPPTokenType::EndOfFile)
            continue;
        std::cout << "│ " << std::setw(4) << (&t - &tokens[0]) << " │ "
                  << std::left << std::setw(15) << t.text.substr(0, 15) << " │ "
                  << std::setw(11) << t.type_name() << " │ "
                  << std::right << std::setw(8) << t.token_id << " │\n";
    }

    std::cout << "└──────┴─────────────────┴─────────────┴──────────┘\n\n";

    auto ids = tok.tokenize_to_ids(code);
    std::cout << "ID sequence (" << ids.size() << " tokens): ";
    for (size_t i = 0; i < std::min(ids.size(), size_t(40)); i++)
        std::cout << ids[i] << " ";
    std::cout << (ids.size() > 40 ? "..." : "") << "\n";

    // Tokenize a file
    auto file_tokens = tok.tokenize_file("src/tokenizer_test.cpp");
    std::cout << "\nФайл tokenizer_test.cpp: " << file_tokens.size() << " токенов\n";

    return 0;
}
