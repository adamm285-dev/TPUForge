#include "tpuforge/openai_server.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <locale>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace tpuforge {

namespace {

constexpr float kMaskBlockValue = -1e9f;
constexpr size_t kMaxHttpHeaderBytes = 16384;
constexpr size_t kMaxHttpBodyBytes = 1048576; // 1 MiB
constexpr size_t kMaxJsonDepth = 32;
constexpr size_t kMaxTokensPerRequest = 4096;

uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

uint64_t draw_entropy_u64() {
    std::random_device rd;
    uint64_t a = static_cast<uint64_t>(rd());
    uint64_t b = static_cast<uint64_t>(rd());
    uint64_t tick = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()
    );
    return splitmix64((a << 32) ^ b ^ tick);
}

std::string to_lower_ascii(std::string s) {
    for (char& ch : s) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    return s;
}

std::string trim_ows(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\t')) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && (s[end - 1] == ' ' || s[end - 1] == '\t')) {
        --end;
    }
    return s.substr(start, end - start);
}

std::string json_escape(const std::string& raw) {
    std::string out;
    out.reserve(raw.size() + 8);
    for (unsigned char c : raw) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    static constexpr char kHex[] = "0123456789abcdef";
                    out += "\\u00";
                    out += kHex[(c >> 4) & 0xF];
                    out += kHex[c & 0xF];
                } else {
                    out += static_cast<char>(c);
                }
                break;
        }
    }
    return out;
}

void append_utf8_codepoint(uint32_t cp, std::string& out) {
    if (cp <= 0x7Fu) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFu) {
        out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
        out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}

struct JsonValue {
    enum class Kind { Null, Bool, Number, String, Array, Object };

    Kind kind{Kind::Null};
    bool bool_val{false};
    double num_val{0.0};
    int64_t int_val{0};
    bool is_integer{false};
    std::string str_val;
    std::vector<JsonValue> arr_val;
    std::map<std::string, JsonValue> obj_val;

    const JsonValue* find(const std::string& key) const {
        if (kind != Kind::Object) {
            return nullptr;
        }
        auto it = obj_val.find(key);
        return (it == obj_val.end()) ? nullptr : &it->second;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& input) : src_(input) {}

    JsonValue parse() {
        skip_ws();
        JsonValue v = parse_value(0);
        skip_ws();
        if (pos_ != src_.size()) {
            throw std::invalid_argument("Unexpected trailing characters after JSON value");
        }
        return v;
    }

private:
    void skip_ws() {
        while (pos_ < src_.size()) {
            char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    char peek() const {
        if (pos_ >= src_.size()) {
            throw std::invalid_argument("Unexpected end of JSON input");
        }
        return src_[pos_];
    }

    char consume() {
        char c = peek();
        ++pos_;
        return c;
    }

    uint32_t parse_hex4() {
        if (pos_ + 4 > src_.size()) {
            throw std::invalid_argument("Truncated \\uXXXX escape in JSON string");
        }
        uint32_t val = 0;
        for (int i = 0; i < 4; ++i) {
            char c = src_[pos_++];
            val <<= 4;
            if (c >= '0' && c <= '9') val |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') val |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') val |= static_cast<uint32_t>(c - 'A' + 10);
            else throw std::invalid_argument("Invalid hex digit in \\uXXXX escape");
        }
        return val;
    }

    std::string parse_string() {
        if (consume() != '"') {
            throw std::invalid_argument("Expected '\"' starting JSON string");
        }
        std::string out;
        while (pos_ < src_.size()) {
            unsigned char ch = static_cast<unsigned char>(src_[pos_++]);
            if (ch == '"') {
                return out;
            }
            if (ch < 0x20u) {
                throw std::invalid_argument("Unescaped control character in JSON string");
            }
            if (ch == '\\') {
                if (pos_ >= src_.size()) {
                    throw std::invalid_argument("Trailing backslash in JSON string");
                }
                char esc = src_[pos_++];
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        uint32_t hi = parse_hex4();
                        if (hi >= 0xDC00u && hi <= 0xDFFFu) {
                            throw std::invalid_argument("Unpaired low UTF-16 surrogate in JSON string");
                        }
                        if (hi >= 0xD800u && hi <= 0xDBFFu) {
                            if (pos_ + 2 > src_.size() || src_[pos_] != '\\' || src_[pos_ + 1] != 'u') {
                                throw std::invalid_argument("Unpaired high UTF-16 surrogate in JSON string");
                            }
                            pos_ += 2;
                            uint32_t lo = parse_hex4();
                            if (lo < 0xDC00u || lo > 0xDFFFu) {
                                throw std::invalid_argument("Invalid low UTF-16 surrogate pair in JSON string");
                            }
                            uint32_t cp = 0x10000u + (((hi - 0xD800u) << 10) | (lo - 0xDC00u));
                            append_utf8_codepoint(cp, out);
                        } else {
                            append_utf8_codepoint(hi, out);
                        }
                        break;
                    }
                    default:
                        throw std::invalid_argument("Invalid escape sequence in JSON string");
                }
            } else {
                out.push_back(static_cast<char>(ch));
            }
        }
        throw std::invalid_argument("Unterminated JSON string");
    }

    JsonValue parse_number() {
        const size_t start = pos_;
        if (src_[pos_] == '-') {
            ++pos_;
        }
        if (pos_ >= src_.size()) {
            throw std::invalid_argument("Invalid JSON number after '-'");
        }
        if (src_[pos_] == '0') {
            ++pos_;
            if (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') {
                throw std::invalid_argument("Leading zero is invalid in JSON number");
            }
        } else if (src_[pos_] >= '1' && src_[pos_] <= '9') {
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') {
                ++pos_;
            }
        } else {
            throw std::invalid_argument("Invalid JSON number mantissa");
        }

        bool has_fraction_or_exp = false;
        if (pos_ < src_.size() && src_[pos_] == '.') {
            has_fraction_or_exp = true;
            ++pos_;
            if (pos_ >= src_.size() || src_[pos_] < '0' || src_[pos_] > '9') {
                throw std::invalid_argument("Invalid JSON fractional digits");
            }
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') {
                ++pos_;
            }
        }
        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            has_fraction_or_exp = true;
            ++pos_;
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) {
                ++pos_;
            }
            if (pos_ >= src_.size() || src_[pos_] < '0' || src_[pos_] > '9') {
                throw std::invalid_argument("Invalid JSON exponent digits");
            }
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') {
                ++pos_;
            }
        }

        const char* tok_begin = src_.data() + start;
        const char* tok_end = src_.data() + pos_;
        JsonValue val;
        val.kind = JsonValue::Kind::Number;

        auto [ptr, ec] = std::from_chars(tok_begin, tok_end, val.num_val);
        if (ec != std::errc{} || ptr != tok_end || !std::isfinite(val.num_val)) {
            throw std::invalid_argument("JSON number out of finite floating-point range");
        }

        if (!has_fraction_or_exp) {
            int64_t parsed_int = 0;
            auto [iptr, iec] = std::from_chars(tok_begin, tok_end, parsed_int);
            if (iec == std::errc{} && iptr == tok_end &&
                !(src_[start] == '-' && parsed_int == 0)) {
                val.int_val = parsed_int;
                val.is_integer = true;
            }
        }
        return val;
    }

    JsonValue parse_value(size_t depth) {
        if (depth >= kMaxJsonDepth) {
            throw std::invalid_argument("JSON nesting depth exceeds maximum allowed limit (32)");
        }
        skip_ws();
        char c = peek();
        if (c == '"') {
            JsonValue v;
            v.kind = JsonValue::Kind::String;
            v.str_val = parse_string();
            return v;
        }
        if (c == '{') {
            return parse_object(depth + 1);
        }
        if (c == '[') {
            return parse_array(depth + 1);
        }
        if (c == 't') {
            if (src_.compare(pos_, 4, "true") == 0) {
                pos_ += 4;
                JsonValue v;
                v.kind = JsonValue::Kind::Bool;
                v.bool_val = true;
                return v;
            }
            throw std::invalid_argument("Invalid JSON literal starting with 't'");
        }
        if (c == 'f') {
            if (src_.compare(pos_, 5, "false") == 0) {
                pos_ += 5;
                JsonValue v;
                v.kind = JsonValue::Kind::Bool;
                v.bool_val = false;
                return v;
            }
            throw std::invalid_argument("Invalid JSON literal starting with 'f'");
        }
        if (c == 'n') {
            if (src_.compare(pos_, 4, "null") == 0) {
                pos_ += 4;
                return JsonValue{};
            }
            throw std::invalid_argument("Invalid JSON literal starting with 'n'");
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            return parse_number();
        }
        throw std::invalid_argument("Unexpected character while parsing JSON value");
    }

    JsonValue parse_array(size_t depth) {
        if (depth >= kMaxJsonDepth) {
            throw std::invalid_argument("JSON nesting depth exceeds maximum allowed limit (32)");
        }
        if (consume() != '[') {
            throw std::invalid_argument("Expected '['");
        }
        JsonValue arr;
        arr.kind = JsonValue::Kind::Array;
        skip_ws();
        if (peek() == ']') {
            consume();
            return arr;
        }
        while (true) {
            arr.arr_val.push_back(parse_value(depth));
            skip_ws();
            char sep = consume();
            if (sep == ']') {
                return arr;
            }
            if (sep != ',') {
                throw std::invalid_argument("Expected ',' or ']' in JSON array");
            }
        }
    }

    JsonValue parse_object(size_t depth) {
        if (depth >= kMaxJsonDepth) {
            throw std::invalid_argument("JSON nesting depth exceeds maximum allowed limit (32)");
        }
        if (consume() != '{') {
            throw std::invalid_argument("Expected '{'");
        }
        JsonValue obj;
        obj.kind = JsonValue::Kind::Object;
        skip_ws();
        if (peek() == '}') {
            consume();
            return obj;
        }
        while (true) {
            skip_ws();
            if (peek() != '"') {
                throw std::invalid_argument("Expected string key in JSON object");
            }
            std::string key = parse_string();
            skip_ws();
            if (consume() != ':') {
                throw std::invalid_argument("Expected ':' after JSON object key");
            }
            JsonValue val = parse_value(depth);
            auto [it, inserted] = obj.obj_val.emplace(std::move(key), std::move(val));
            if (!inserted) {
                throw std::invalid_argument("Duplicate key in JSON object");
            }
            skip_ws();
            char sep = consume();
            if (sep == '}') {
                return obj;
            }
            if (sep != ',') {
                throw std::invalid_argument("Expected ',' or '}' in JSON object");
            }
        }
    }

    const std::string& src_;
    size_t pos_{0};
};

// Strictly validates RFC 3629 UTF-8 strings using explicit decoder error tracking.
bool is_valid_utf8(const std::string& s) {
    Utf8StreamDecoder dec;
    for (unsigned char b : s) {
        (void)dec.push_byte(b);
        if (dec.saw_error()) {
            return false;
        }
    }
    (void)dec.flush();
    return !dec.saw_error();
}

#if defined(_WIN32)
using NativeSocket = SOCKET;
constexpr NativeSocket kInvalidSocket = INVALID_SOCKET;

void ensure_winsock_initialized() {
    static std::once_flag once;
    static int init_rc = 0;
    std::call_once(once, []() {
        WSADATA wsa{};
        init_rc = WSAStartup(MAKEWORD(2, 2), &wsa);
    });
    if (init_rc != 0) {
        throw std::runtime_error("WSAStartup failed with code " + std::to_string(init_rc));
    }
}

void close_native_socket(NativeSocket s) noexcept {
    if (s != kInvalidSocket) {
        closesocket(s);
    }
}

void graceful_shutdown_send(NativeSocket s) noexcept {
    if (s != kInvalidSocket) {
        (void)shutdown(s, SD_SEND);
    }
}

bool set_socket_recv_timeout(NativeSocket s, uint32_t timeout_ms) noexcept {
    DWORD tv = static_cast<DWORD>(std::max<uint32_t>(1u, timeout_ms));
    return setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof(tv)) == 0;
}

bool set_socket_timeouts(NativeSocket s, uint32_t timeout_ms) noexcept {
    DWORD tv = static_cast<DWORD>(std::max<uint32_t>(1u, timeout_ms));
    if (setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof(tv)) != 0) {
        return false;
    }
    if (setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&tv), sizeof(tv)) != 0) {
        return false;
    }
    return true;
}

bool wait_socket_writable(NativeSocket s, std::chrono::steady_clock::duration rem) noexcept {
    if (rem <= std::chrono::steady_clock::duration::zero()) {
        return false;
    }
    const auto rem_ms = std::chrono::duration_cast<std::chrono::milliseconds>(rem).count();
    const uint32_t clamped_ms = static_cast<uint32_t>(std::max<int64_t>(1, rem_ms));
    fd_set wfds;
    FD_ZERO(&wfds);
    FD_SET(s, &wfds);
    struct timeval tv{};
    tv.tv_sec = static_cast<long>(clamped_ms / 1000u);
    tv.tv_usec = static_cast<long>((clamped_ms % 1000u) * 1000u);
    int rc = select(0, nullptr, &wfds, nullptr, &tv);
    return rc > 0;
}

bool set_socket_nonblocking(NativeSocket s, bool nonblocking) noexcept {
    u_long mode = nonblocking ? 1UL : 0UL;
    return ioctlsocket(s, FIONBIO, &mode) == 0;
}

bool send_all_with_budget(
    NativeSocket s,
    const char* data,
    size_t len,
    std::chrono::steady_clock::duration* budget
) noexcept {
    size_t sent = 0;
    while (sent < len) {
        if (*budget <= std::chrono::steady_clock::duration::zero()) {
            return false;
        }
        const auto step_begin = std::chrono::steady_clock::now();
        if (!wait_socket_writable(s, *budget)) {
            *budget = std::chrono::steady_clock::duration::zero();
            return false;
        }
        const size_t chunk_sz = std::min<size_t>(len - sent, static_cast<size_t>(65536));
        const int rc = send(s, data + sent, static_cast<int>(chunk_sz), 0);
        const auto elapsed = std::chrono::steady_clock::now() - step_begin;
        *budget = (elapsed >= *budget)
            ? std::chrono::steady_clock::duration::zero()
            : (*budget - elapsed);
        if (rc <= 0) {
            return false;
        }
        sent += static_cast<size_t>(rc);
    }
    return true;
}

bool send_all_bytes(NativeSocket s, const char* data, size_t len, uint32_t timeout_ms = 5000u) noexcept {
    std::chrono::steady_clock::duration budget =
        std::chrono::milliseconds(std::max<uint32_t>(1u, timeout_ms));
    return send_all_with_budget(s, data, len, &budget);
}

int recv_bytes(NativeSocket s, char* buf, size_t cap) noexcept {
    return recv(s, buf, static_cast<int>(std::min<size_t>(cap, 65536)), 0);
}
#else
using NativeSocket = int;
constexpr NativeSocket kInvalidSocket = -1;

void ensure_winsock_initialized() {}

void close_native_socket(NativeSocket s) noexcept {
    if (s != kInvalidSocket) {
        close(s);
    }
}

void graceful_shutdown_send(NativeSocket s) noexcept {
    if (s != kInvalidSocket) {
        (void)shutdown(s, SHUT_WR);
    }
}

bool set_socket_recv_timeout(NativeSocket s, uint32_t timeout_ms) noexcept {
    uint32_t clamped = std::max<uint32_t>(1u, timeout_ms);
    struct timeval tv{};
    tv.tv_sec = static_cast<time_t>(clamped / 1000u);
    tv.tv_usec = static_cast<suseconds_t>((clamped % 1000u) * 1000u);
    return setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0;
}

bool set_socket_timeouts(NativeSocket s, uint32_t timeout_ms) noexcept {
    uint32_t clamped = std::max<uint32_t>(1u, timeout_ms);
    struct timeval tv{};
    tv.tv_sec = static_cast<time_t>(clamped / 1000u);
    tv.tv_usec = static_cast<suseconds_t>((clamped % 1000u) * 1000u);
    if (setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) {
        return false;
    }
    if (setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0) {
        return false;
    }
#ifdef SO_NOSIGPIPE
    int yes = 1;
    setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif
    return true;
}

bool set_socket_nonblocking(NativeSocket s, bool nonblocking) noexcept {
    int fl = fcntl(s, F_GETFL, 0);
    if (fl < 0) {
        return false;
    }
    int target = nonblocking ? (fl | O_NONBLOCK) : (fl & ~O_NONBLOCK);
    if (target == fl) {
        return true;
    }
    return fcntl(s, F_SETFL, target) == 0;
}

bool wait_socket_writable(NativeSocket s, std::chrono::steady_clock::duration rem) noexcept {
    if (rem <= std::chrono::steady_clock::duration::zero()) {
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + rem;
    while (true) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            return false;
        }
        const auto rem_ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        const int poll_ms = static_cast<int>(std::min<int64_t>(
            std::max<int64_t>(1, rem_ms),
            static_cast<int64_t>(std::numeric_limits<int>::max())
        ));
        struct pollfd pfd{};
        pfd.fd = s;
        pfd.events = POLLOUT;
        int rc = poll(&pfd, 1, poll_ms);
        if (rc < 0 && errno == EINTR) {
            continue;
        }
        return rc > 0;
    }
}

bool send_all_with_budget(
    NativeSocket s,
    const char* data,
    size_t len,
    std::chrono::steady_clock::duration* budget
) noexcept {
    size_t sent = 0;
    while (sent < len) {
        if (*budget <= std::chrono::steady_clock::duration::zero()) {
            return false;
        }
        const auto step_begin = std::chrono::steady_clock::now();
        if (!wait_socket_writable(s, *budget)) {
            *budget = std::chrono::steady_clock::duration::zero();
            return false;
        }
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags = MSG_NOSIGNAL;
#endif
        const ssize_t rc = send(s, data + sent, len - sent, flags);
        if (rc < 0 && errno == EINTR) {
            const auto elapsed = std::chrono::steady_clock::now() - step_begin;
            *budget = (elapsed >= *budget)
                ? std::chrono::steady_clock::duration::zero()
                : (*budget - elapsed);
            continue;
        }
        const auto elapsed = std::chrono::steady_clock::now() - step_begin;
        *budget = (elapsed >= *budget)
            ? std::chrono::steady_clock::duration::zero()
            : (*budget - elapsed);
        if (rc <= 0) {
            return false;
        }
        sent += static_cast<size_t>(rc);
    }
    return true;
}

bool send_all_bytes(NativeSocket s, const char* data, size_t len, uint32_t timeout_ms = 5000u) noexcept {
    std::chrono::steady_clock::duration budget =
        std::chrono::milliseconds(std::max<uint32_t>(1u, timeout_ms));
    return send_all_with_budget(s, data, len, &budget);
}

int recv_bytes(NativeSocket s, char* buf, size_t cap) noexcept {
    while (true) {
        ssize_t rc = recv(s, buf, std::min<size_t>(cap, 65536), 0);
        if (rc < 0 && errno == EINTR) {
            continue;
        }
        return static_cast<int>(rc);
    }
}
#endif

struct SocketCloser {
    NativeSocket sock{kInvalidSocket};
    ~SocketCloser() { close_native_socket(sock); }
    SocketCloser(const SocketCloser&) = delete;
    SocketCloser& operator=(const SocketCloser&) = delete;
    explicit SocketCloser(NativeSocket s) noexcept : sock(s) {}
};

const char* http_reason_phrase(int status) noexcept {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 413: return "Payload Too Large";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        default: return "Error";
    }
}

bool send_json_response(
    NativeSocket sock,
    int status,
    const std::string& json_body,
    const std::string& extra_headers = "",
    uint32_t timeout_ms = 5000u
) {
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << "HTTP/1.1 " << status << " " << http_reason_phrase(status) << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << json_body.size() << "\r\n"
        << "Connection: close\r\n"
        << extra_headers
        << "\r\n"
        << json_body;
    const std::string wire = oss.str();
    bool ok = send_all_bytes(sock, wire.data(), wire.size(), timeout_ms);
    graceful_shutdown_send(sock);
    return ok;
}

bool send_json_error(
    NativeSocket sock,
    int status,
    const std::string& message,
    const std::string& error_type = "invalid_request_error",
    const std::string& extra_headers = "",
    uint32_t timeout_ms = 5000u
) {
    std::ostringstream body;
    body.imbue(std::locale::classic());
    body << "{\"error\":{\"message\":\"" << json_escape(message)
         << "\",\"type\":\"" << json_escape(error_type)
         << "\",\"code\":" << status << "}}";
    return send_json_response(sock, status, body.str(), extra_headers, timeout_ms);
}

std::string build_sse_chunk(
    const std::string& completion_id,
    int64_t created_ts,
    const std::string& model_id,
    const std::string& delta_fields_json,
    const std::string& finish_reason_json,
    const std::string& trailing_fields_json = ""
) {
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << "data: {\"id\":\"" << json_escape(completion_id) << "\","
        << "\"object\":\"chat.completion.chunk\","
        << "\"created\":" << created_ts << ","
        << "\"model\":\"" << json_escape(model_id) << "\","
        << "\"choices\":[{\"index\":0,\"delta\":" << delta_fields_json
        << ",\"finish_reason\":" << finish_reason_json << "}]"
        << trailing_fields_json << "}\n\n";
    return oss.str();
}

std::string build_telemetry_json(const GenerationResult& res) {
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << "\"usage\":{\"prompt_tokens\":" << res.prompt_tokens_count
        << ",\"completion_tokens\":" << res.completion_tokens_count
        << ",\"total_tokens\":" << res.total_tokens_count << "},"
        << "\"tpuforge_telemetry\":{\"target_forward_passes\":" << res.target_forward_passes
        << ",\"decode_forward_passes\":" << res.decode_forward_passes
        << ",\"draft_tokens_accepted\":" << res.draft_tokens_accepted
        << ",\"mean_tokens_per_decode_pass\":" << res.mean_tokens_per_decode_pass
        << ",\"compile_count\":" << res.compile_count << "}";
    return oss.str();
}

} // namespace

// --- Utf8StreamDecoder ---

std::string Utf8StreamDecoder::push_byte(uint8_t b) {
    std::string out;

    if (pending_len_ > 0) {
        bool valid_cont = false;
        if (pending_len_ == 1) {
            uint8_t b0 = pending_[0];
            if (b0 == 0xE0u) {
                valid_cont = (b >= 0xA0u && b <= 0xBFu);
            } else if (b0 == 0xEDu) {
                valid_cont = (b >= 0x80u && b <= 0x9Fu);
            } else if (b0 == 0xF0u) {
                valid_cont = (b >= 0x90u && b <= 0xBFu);
            } else if (b0 == 0xF4u) {
                valid_cont = (b >= 0x80u && b <= 0x8Fu);
            } else {
                valid_cont = (b >= 0x80u && b <= 0xBFu);
            }
        } else {
            valid_cont = (b >= 0x80u && b <= 0xBFu);
        }

        if (valid_cont) {
            pending_[pending_len_++] = b;
            if (pending_len_ == expected_len_) {
                out.assign(reinterpret_cast<const char*>(pending_.data()), pending_len_);
                pending_len_ = 0;
                expected_len_ = 0;
            }
            return out;
        }

        saw_error_ = true;
        out += "\xEF\xBF\xBD";
        pending_len_ = 0;
        expected_len_ = 0;
    }

    if (b <= 0x7Fu) {
        out.push_back(static_cast<char>(b));
        return out;
    }
    if (b >= 0xC2u && b <= 0xDFu) {
        pending_[0] = b;
        pending_len_ = 1;
        expected_len_ = 2;
        return out;
    }
    if (b >= 0xE0u && b <= 0xEFu) {
        pending_[0] = b;
        pending_len_ = 1;
        expected_len_ = 3;
        return out;
    }
    if (b >= 0xF0u && b <= 0xF4u) {
        pending_[0] = b;
        pending_len_ = 1;
        expected_len_ = 4;
        return out;
    }

    saw_error_ = true;
    out += "\xEF\xBF\xBD";
    return out;
}

std::string Utf8StreamDecoder::push_token(int32_t token_id, size_t vocab_size, bool skip_special) {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size) {
        throw std::out_of_range("Token ID out of tokenizer vocab range");
    }
    if (token_id >= ByteTokenizer::kByteOffset &&
        token_id < ByteTokenizer::kByteOffset + 256) {
        return push_byte(static_cast<uint8_t>(token_id - ByteTokenizer::kByteOffset));
    }

    std::string out = flush();
    if (!skip_special) {
        switch (token_id) {
            case ByteTokenizer::kPadToken: out += "<pad>"; break;
            case ByteTokenizer::kBosToken: out += "<bos>"; break;
            case ByteTokenizer::kEosToken: out += "<eos>"; break;
            case ByteTokenizer::kSepToken: out += "<sep>"; break;
            default: out += "<t:" + std::to_string(token_id) + ">"; break;
        }
    }
    return out;
}

std::string Utf8StreamDecoder::flush() {
    if (pending_len_ > 0) {
        pending_len_ = 0;
        expected_len_ = 0;
        saw_error_ = true;
        return "\xEF\xBF\xBD";
    }
    return {};
}

void Utf8StreamDecoder::reset() noexcept {
    pending_len_ = 0;
    expected_len_ = 0;
    saw_error_ = false;
}

// --- ByteTokenizer ---

ByteTokenizer::ByteTokenizer(size_t vocab_size) : vocab_size_(vocab_size) {
    if (vocab_size_ < kMinVocabSize ||
        vocab_size_ > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument(
            "ByteTokenizer requires vocab_size in [260, INT32_MAX]"
        );
    }
}

std::vector<int32_t> ByteTokenizer::encode(
    const std::string& text,
    bool add_bos,
    bool add_eos
) const {
    std::vector<int32_t> tokens;
    tokens.reserve(text.size() + (add_bos ? 1 : 0) + (add_eos ? 1 : 0));
    if (add_bos) {
        tokens.push_back(kBosToken);
    }
    for (unsigned char b : text) {
        tokens.push_back(kByteOffset + static_cast<int32_t>(b));
    }
    if (add_eos) {
        tokens.push_back(kEosToken);
    }
    return tokens;
}

std::vector<int32_t> ByteTokenizer::encode_chat(const std::vector<ChatMessage>& messages) const {
    if (messages.empty()) {
        throw std::invalid_argument("Chat messages array cannot be empty");
    }
    std::vector<int32_t> tokens;
    tokens.push_back(kBosToken);
    for (const auto& msg : messages) {
        if (msg.role.empty()) {
            throw std::invalid_argument("Each chat message must have a non-empty role");
        }
        const std::string role_prefix = msg.role + ":";
        for (unsigned char b : role_prefix) {
            tokens.push_back(kByteOffset + static_cast<int32_t>(b));
        }
        for (unsigned char b : msg.content) {
            tokens.push_back(kByteOffset + static_cast<int32_t>(b));
        }
        tokens.push_back(kSepToken);
    }
    static constexpr char kAssistantPrompt[] = "assistant:";
    for (size_t i = 0; i < sizeof(kAssistantPrompt) - 1; ++i) {
        tokens.push_back(kByteOffset + static_cast<int32_t>(static_cast<unsigned char>(kAssistantPrompt[i])));
    }
    return tokens;
}

std::string ByteTokenizer::decode(const std::vector<int32_t>& tokens, bool skip_special) const {
    Utf8StreamDecoder dec;
    std::string out;
    out.reserve(tokens.size());
    for (int32_t tok : tokens) {
        out += dec.push_token(tok, vocab_size_, skip_special);
    }
    out += dec.flush();
    return out;
}

// --- ForgeEngine ---

ForgeEngine::ForgeEngine(
    PjrtDriver& driver,
    const StaticArena::Config& config,
    const std::string& program_code,
    const std::string& program_format,
    std::string model_id
)
    : config_(config),
      model_id_(std::move(model_id)),
      tokenizer_(config.vocab_size),
      arena_(config),
      executor_(std::make_unique<ZeroRecompileExecutor>(driver)) {
    if (config_.vocab_size < ByteTokenizer::kMinVocabSize) {
        throw std::invalid_argument("ForgeEngine requires vocab_size >= 260");
    }
    if (config_.max_seq_len < 2 * SystolicConstants::kSystolicTileSize) {
        throw std::invalid_argument("ForgeEngine requires max_seq_len >= 256");
    }
    if (program_code.empty()) {
        throw std::invalid_argument(
            "ForgeEngine PJRT constructor requires an explicit non-empty program_code"
        );
    }
    if (model_id_.empty()) {
        model_id_ = "tpuforge-systolic-128";
    }
    std::string err;
    if (!executor_->compile_once(config_, program_code, program_format, &err)) {
        throw std::runtime_error("ForgeEngine PJRT compile_once failed: " + err);
    }
    if (executor_->num_outputs() < 3) {
        throw std::invalid_argument(
            "Compiled PJRT executable must expose 3 outputs (logits, step_k, step_v) for stateful chat serving"
        );
    }
}

ForgeEngine::ForgeEngine(
    const StaticArena::Config& config,
    TargetStepFn target_step_fn,
    CommitKvFn commit_kv_fn,
    std::string model_id
)
    : config_(config),
      model_id_(std::move(model_id)),
      tokenizer_(config.vocab_size),
      arena_(config),
      target_step_fn_(std::move(target_step_fn)),
      commit_kv_fn_(std::move(commit_kv_fn)) {
    if (config_.vocab_size < ByteTokenizer::kMinVocabSize) {
        throw std::invalid_argument("ForgeEngine requires vocab_size >= 260");
    }
    if (config_.max_seq_len < 2 * SystolicConstants::kSystolicTileSize) {
        throw std::invalid_argument("ForgeEngine requires max_seq_len >= 256");
    }
    if (!target_step_fn_) {
        throw std::invalid_argument("ForgeEngine custom backend requires a non-null target_step_fn");
    }
    if (model_id_.empty()) {
        model_id_ = "tpuforge-custom-128";
    }
}

size_t ForgeEngine::compile_count() const {
    return executor_ ? executor_->compile_count() : 0;
}

std::vector<int32_t> ForgeEngine::validate_request(const GenerationRequest& req) const {
    if (!std::isfinite(req.temperature) || req.temperature < 0.0f || req.temperature > 2.0f) {
        throw std::invalid_argument("temperature must be a finite float in [0.0, 2.0]");
    }
    if (req.max_tokens > kMaxTokensPerRequest) {
        throw std::invalid_argument("max_tokens exceeds maximum allowed per request (4096)");
    }

    std::vector<int32_t> prompt = !req.prompt_tokens.empty()
        ? req.prompt_tokens
        : tokenizer_.encode_chat(req.messages);

    if (prompt.empty()) {
        throw std::invalid_argument("Prompt token sequence cannot be empty");
    }
    const size_t max_prefix = config_.max_seq_len - SystolicConstants::kSystolicTileSize;
    if (prompt.size() > max_prefix) {
        throw std::invalid_argument(
            "Prompt length (" + std::to_string(prompt.size()) +
            ") exceeds arena capacity (" + std::to_string(max_prefix) + ")"
        );
    }
    for (int32_t tok : prompt) {
        if (tok < 0 || static_cast<size_t>(tok) >= config_.vocab_size) {
            throw std::out_of_range("Prompt token ID out of [0, vocab_size) bounds");
        }
    }
    return prompt;
}

void ForgeEngine::execute_tile_step() {
    constexpr size_t T = SystolicConstants::kSystolicTileSize;
    if (executor_) {
        std::string err;
        if (!executor_->execute_step_with_kv(arena_, &pjrt_step_k_, &pjrt_step_v_, T, &err)) {
            throw std::runtime_error("PJRT execute_step_with_kv failed: " + err);
        }
    } else {
        target_step_fn_(
            arena_.valid_prefix_len(),
            arena_.token_tile_128(),
            arena_.position_tile_128(),
            arena_.causal_mask_128x128(),
            arena_.logits_tile_128()
        );
    }
}

GenerationResult ForgeEngine::generate(
    const GenerationRequest& req,
    StreamTokenCallback stream_cb
) {
    std::vector<int32_t> prompt = validate_request(req);
    const uint64_t request_base_seed = req.seed.has_value() ? *req.seed : draw_entropy_u64();

    std::scoped_lock lk(engine_mu_);

    constexpr size_t T = SystolicConstants::kSystolicTileSize;
    const size_t max_prefix = config_.max_seq_len - T;

    arena_.set_valid_prefix_len(0);

    GenerationResult result;
    result.prompt_tokens_count = prompt.size();

    // 1. Static 128-row causal prefill over prompt[0 .. N-2].
    // The final prompt token becomes the root of the first 128-node speculative decode tree.
    const size_t prefill_len = prompt.size() - 1;
    std::vector<int32_t> prefill_accepted;
    prefill_accepted.reserve(T);

    for (size_t offset = 0; offset < prefill_len; offset += T) {
        const size_t chunk = std::min(T, prefill_len - offset);
        int32_t* tok_tile = arena_.token_tile_128();
        int32_t* pos_tile = arena_.position_tile_128();
        float* mask_tile = arena_.causal_mask_128x128();

        std::fill_n(tok_tile, T, ByteTokenizer::kPadToken);
        std::fill_n(pos_tile, T, 0);
        std::fill_n(mask_tile, T * T, kMaskBlockValue);

        for (size_t r = 0; r < chunk; ++r) {
            tok_tile[r] = prompt[offset + r];
            pos_tile[r] = static_cast<int32_t>(offset + r);
            for (size_t c = 0; c <= r; ++c) {
                mask_tile[r * T + c] = 0.0f;
            }
        }
        // Keep inactive rows focused on their own diagonal so softmax never receives all -1e9.
        for (size_t r = chunk; r < T; ++r) {
            mask_tile[r * T + r] = 0.0f;
        }

        arena_.set_valid_prefix_len(static_cast<int32_t>(offset));
        execute_tile_step();
        result.target_forward_passes += 1;

        prefill_accepted.clear();
        for (size_t r = 0; r < chunk; ++r) {
            prefill_accepted.push_back(static_cast<int32_t>(r));
        }
        if (executor_) {
            arena_.commit_accepted_kv_rows(pjrt_step_k_.data(), pjrt_step_v_.data(), prefill_accepted);
        } else {
            if (commit_kv_fn_) {
                commit_kv_fn_(static_cast<int32_t>(offset), prefill_accepted);
            }
            arena_.set_valid_prefix_len(static_cast<int32_t>(offset + chunk));
        }
    }

    // 2. 128-node systolic tree speculative decode loop.
    std::vector<int32_t> history = std::move(prompt);
    std::vector<int32_t> generated;
    generated.reserve(req.max_tokens);
    Utf8StreamDecoder utf8_decoder;
    bool client_aborted = false;
    bool hit_eos = false;

    while (generated.size() < req.max_tokens &&
           static_cast<size_t>(arena_.valid_prefix_len()) < max_prefix) {
        const int32_t cur_prefix = arena_.valid_prefix_len();
        const size_t remaining_prefix = max_prefix - static_cast<size_t>(cur_prefix);
        const size_t budget = std::min(req.max_tokens - generated.size(), remaining_prefix);
        const size_t max_depth = std::min<size_t>(16, remaining_prefix);

        SystolicTree tree = SuffixTreeDrafter::build_tree(
            history, SystolicTree::kMaxNodes, std::max<size_t>(1, max_depth)
        );
        PackedSystolicTile packed = tree.pack(cur_prefix);

        std::memcpy(arena_.token_tile_128(), packed.token_ids.data(), T * sizeof(int32_t));
        std::memcpy(arena_.position_tile_128(), packed.position_offsets.data(), T * sizeof(int32_t));
        std::memcpy(arena_.causal_mask_128x128(), packed.additive_mask_f32.data(), T * T * sizeof(float));

        execute_tile_step();
        result.target_forward_passes += 1;
        result.decode_forward_passes += 1;

        TreeVerificationResult verify_res;
        if (req.temperature < 1e-6f) {
            verify_res = SystolicTreeVerifier::verify_greedy(
                tree, arena_.logits_tile_128(), config_.vocab_size
            );
        } else {
            const uint64_t step_seed = splitmix64(
                request_base_seed ^
                (static_cast<uint64_t>(result.decode_forward_passes) * 0x9E3779B97F4A7C15ULL)
            );
            verify_res = SystolicTreeVerifier::verify_stochastic(
                tree, arena_.logits_tile_128(), config_.vocab_size, req.temperature, step_seed
            );
        }

        size_t commit_count = 0;
        const size_t limit = std::min(budget, verify_res.emitted_tokens.size());
        for (size_t i = 0; i < limit; ++i) {
            const int32_t tok = verify_res.emitted_tokens[i];
            if (!req.ignore_eos && tok == ByteTokenizer::kEosToken) {
                hit_eos = true;
                break;
            }
            generated.push_back(tok);
            history.push_back(tok);
            commit_count += 1;

            std::string delta = utf8_decoder.push_token(tok, config_.vocab_size, true);
            if (!delta.empty()) {
                result.text += delta;
                if (stream_cb && !stream_cb(delta, tok)) {
                    client_aborted = true;
                    break;
                }
            }
        }

        if (commit_count > 0) {
            if (commit_count > 1) {
                result.draft_tokens_accepted += (commit_count - 1);
            }
            std::vector<int32_t> committed_indices(
                verify_res.accepted_node_indices.begin(),
                verify_res.accepted_node_indices.begin() +
                    static_cast<std::ptrdiff_t>(std::min(commit_count, verify_res.accepted_node_indices.size()))
            );
            if (executor_) {
                arena_.commit_accepted_kv_rows(pjrt_step_k_.data(), pjrt_step_v_.data(), committed_indices);
            } else {
                if (commit_kv_fn_) {
                    commit_kv_fn_(cur_prefix, committed_indices);
                }
                arena_.set_valid_prefix_len(cur_prefix + static_cast<int32_t>(commit_count));
            }
        }

        if (hit_eos || client_aborted || commit_count == 0) {
            break;
        }
    }

    std::string tail = utf8_decoder.flush();
    if (!tail.empty()) {
        result.text += tail;
        if (stream_cb && !client_aborted) {
            (void)stream_cb(tail, -1);
        }
    }

    result.token_ids = std::move(generated);
    result.completion_tokens_count = result.token_ids.size();
    result.total_tokens_count = result.prompt_tokens_count + result.completion_tokens_count;
    result.finish_reason = hit_eos ? "stop" : "length";
    result.mean_tokens_per_decode_pass = (result.decode_forward_passes > 0)
        ? (static_cast<double>(result.completion_tokens_count) /
           static_cast<double>(result.decode_forward_passes))
        : 0.0;
    result.compile_count = compile_count();
    return result;
}

// --- OpenAIServer ---

struct OpenAIServer::Impl {
    ForgeEngine& engine;
    OpenAIServer::Config config;

    std::mutex state_mu;
    std::condition_variable state_cv;
    bool stopping{false};
    std::atomic<bool> running{false};
    std::atomic<uint16_t> bound_port{0};
    std::atomic<bool> stop_requested{false};
    NativeSocket listen_sock{kInvalidSocket};

    std::thread accept_thread;
    std::vector<std::thread> workers;
    std::mutex queue_mu;
    std::condition_variable queue_cv;
    std::deque<NativeSocket> pending_clients;
    std::atomic<uint64_t> completion_seq{0};

    Impl(ForgeEngine& eng, const OpenAIServer::Config& cfg)
        : engine(eng), config(cfg) {
        if (config.worker_threads == 0) {
            config.worker_threads = 1;
        }
        if (config.max_queued_connections == 0) {
            config.max_queued_connections = 1;
        }
        if (config.socket_timeout_ms == 0) {
            config.socket_timeout_ms = 5000;
        }
    }

    void accept_loop(NativeSocket lsock) {
        while (!stop_requested.load(std::memory_order_acquire)) {
#if defined(_WIN32)
            fd_set read_fds;
            FD_ZERO(&read_fds);
            FD_SET(lsock, &read_fds);
            struct timeval tv{};
            tv.tv_sec = 0;
            tv.tv_usec = 25000;
            int ready = select(0, &read_fds, nullptr, nullptr, &tv);
#else
            struct pollfd pfd{};
            pfd.fd = lsock;
            pfd.events = POLLIN;
            int ready = poll(&pfd, 1, 25);
#endif
            if (ready < 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            if (ready == 0) {
                continue;
            }
            if (stop_requested.load(std::memory_order_acquire)) {
                break;
            }

            sockaddr_in client_addr{};
#if defined(_WIN32)
            int addr_len = sizeof(client_addr);
#else
            socklen_t addr_len = sizeof(client_addr);
#endif
            NativeSocket client_sock = accept(
                lsock, reinterpret_cast<sockaddr*>(&client_addr), &addr_len
            );
            if (client_sock == kInvalidSocket) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            (void)set_socket_nonblocking(client_sock, false);
            (void)set_socket_timeouts(client_sock, config.socket_timeout_ms);

            bool queued = false;
            {
                std::lock_guard qlk(queue_mu);
                if (!stop_requested.load(std::memory_order_acquire) &&
                    pending_clients.size() < config.max_queued_connections) {
                    pending_clients.push_back(client_sock);
                    queued = true;
                }
            }
            if (queued) {
                queue_cv.notify_one();
            } else {
                SocketCloser closer(client_sock);
                (void)send_json_error(
                    client_sock, 503, "Server request queue full", "server_busy", "", config.socket_timeout_ms
                );
            }
        }
    }

    void worker_loop() {
        while (true) {
            NativeSocket client_sock = kInvalidSocket;
            {
                std::unique_lock qlk(queue_mu);
                queue_cv.wait(qlk, [this]() {
                    return stop_requested.load(std::memory_order_acquire) || !pending_clients.empty();
                });
                if (stop_requested.load(std::memory_order_acquire)) {
                    return;
                }
                client_sock = pending_clients.front();
                pending_clients.pop_front();
            }
            SocketCloser closer(client_sock);
            try {
                handle_client(client_sock);
            } catch (...) {
                // Prevent worker exceptions from calling std::terminate on the host process.
            }
        }
    }

    int recv_with_deadline(
        NativeSocket sock,
        char* dst,
        size_t cap,
        std::chrono::steady_clock::time_point deadline,
        bool* timed_out
    ) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            *timed_out = true;
            return -1;
        }
        const auto rem_ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        const uint32_t bounded_ms = static_cast<uint32_t>(
            std::max<int64_t>(1, std::min<int64_t>(rem_ms, static_cast<int64_t>(config.socket_timeout_ms)))
        );
        (void)set_socket_recv_timeout(sock, bounded_ms);
        int n = recv_bytes(sock, dst, cap);
        if (n < 0) {
#if defined(_WIN32)
            const int err_code = WSAGetLastError();
            const bool is_sys_timeout = (err_code == WSAETIMEDOUT);
#else
            const int err_code = errno;
            const bool is_sys_timeout =
                (err_code == EAGAIN || err_code == EWOULDBLOCK || err_code == ETIMEDOUT);
#endif
            if (is_sys_timeout || std::chrono::steady_clock::now() >= deadline) {
                *timed_out = true;
                return -1;
            }
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            *timed_out = true;
            return -1;
        }
        *timed_out = false;
        return n;
    }

    void handle_client(NativeSocket sock) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(config.socket_timeout_ms);

        std::string buffer;
        buffer.reserve(2048);
        size_t header_end = std::string::npos;

        char chunk[4096];
        while (true) {
            header_end = buffer.find("\r\n\r\n");
            if (header_end != std::string::npos) {
                if (header_end > kMaxHttpHeaderBytes) {
                    (void)send_json_error(sock, 431, "HTTP request headers exceed 16384 byte limit");
                    return;
                }
                break;
            }
            if (buffer.size() > kMaxHttpHeaderBytes) {
                (void)send_json_error(sock, 431, "HTTP request headers exceed 16384 byte limit");
                return;
            }
            bool timed_out = false;
            int n = recv_with_deadline(sock, chunk, sizeof(chunk), deadline, &timed_out);
            if (timed_out) {
                (void)send_json_error(sock, 408, "HTTP header read deadline exceeded", "timeout_error");
                return;
            }
            if (n <= 0) {
                if (!buffer.empty()) {
                    (void)send_json_error(sock, 400, "Incomplete HTTP request headers");
                }
                return;
            }
            buffer.append(chunk, static_cast<size_t>(n));
            if (buffer.size() > kMaxHttpHeaderBytes) {
                size_t found = buffer.find("\r\n\r\n");
                if (found == std::string::npos || found > kMaxHttpHeaderBytes) {
                    (void)send_json_error(sock, 431, "HTTP request headers exceed 16384 byte limit");
                    return;
                }
                header_end = found;
                break;
            }
        }

        const std::string header_block = buffer.substr(0, header_end);
        std::string body = buffer.substr(header_end + 4);

        const size_t first_crlf = header_block.find("\r\n");
        const std::string request_line = (first_crlf == std::string::npos)
            ? header_block
            : header_block.substr(0, first_crlf);

        const size_t sp1 = request_line.find(' ');
        const size_t sp2 = (sp1 == std::string::npos) ? std::string::npos : request_line.find(' ', sp1 + 1);
        if (sp1 == std::string::npos || sp2 == std::string::npos || request_line.find(' ', sp2 + 1) != std::string::npos) {
            (void)send_json_error(sock, 400, "Malformed HTTP request line");
            return;
        }

        const std::string method = request_line.substr(0, sp1);
        std::string target = request_line.substr(sp1 + 1, sp2 - sp1 - 1);
        const std::string version = request_line.substr(sp2 + 1);
        if (version != "HTTP/1.1" && version != "HTTP/1.0") {
            (void)send_json_error(sock, 400, "Unsupported HTTP version (expected HTTP/1.1 or HTTP/1.0)");
            return;
        }
        const size_t qpos = target.find('?');
        if (qpos != std::string::npos) {
            target = target.substr(0, qpos);
        }

        bool has_transfer_encoding = false;
        bool has_content_length = false;
        size_t content_length = 0;

        if (first_crlf != std::string::npos) {
            size_t line_pos = first_crlf + 2;
            while (line_pos < header_block.size()) {
                size_t next_crlf = header_block.find("\r\n", line_pos);
                std::string line = (next_crlf == std::string::npos)
                    ? header_block.substr(line_pos)
                    : header_block.substr(line_pos, next_crlf - line_pos);
                if (!line.empty() && (line[0] == ' ' || line[0] == '\t')) {
                    (void)send_json_error(sock, 400, "Obsolete HTTP header line folding (obs-fold) rejected");
                    return;
                }
                size_t colon = line.find(':');
                if (colon == std::string::npos || colon == 0) {
                    (void)send_json_error(sock, 400, "Malformed HTTP header line");
                    return;
                }
                const std::string raw_name = line.substr(0, colon);
                for (unsigned char ch : raw_name) {
                    if (ch == ' ' || ch == '\t' || ch < 0x21u) {
                        (void)send_json_error(sock, 400, "Invalid whitespace in HTTP header name");
                        return;
                    }
                }
                const std::string name = to_lower_ascii(raw_name);
                const std::string value = trim_ows(line.substr(colon + 1));

                if (name == "transfer-encoding") {
                    has_transfer_encoding = true;
                } else if (name == "content-length") {
                    if (value.empty()) {
                        (void)send_json_error(sock, 400, "Empty Content-Length header");
                        return;
                    }
                    size_t parsed_len = 0;
                    for (unsigned char ch : value) {
                        if (ch < '0' || ch > '9') {
                            (void)send_json_error(sock, 400, "Non-numeric Content-Length header");
                            return;
                        }
                        size_t d = static_cast<size_t>(ch - '0');
                        if (parsed_len > (std::numeric_limits<size_t>::max() - d) / 10) {
                            (void)send_json_error(sock, 413, "Content-Length exceeds maximum limit");
                            return;
                        }
                        parsed_len = parsed_len * 10 + d;
                    }
                    if (has_content_length && parsed_len != content_length) {
                        (void)send_json_error(sock, 400, "Conflicting duplicate Content-Length headers");
                        return;
                    }
                    has_content_length = true;
                    content_length = parsed_len;
                }

                if (next_crlf == std::string::npos) {
                    break;
                }
                line_pos = next_crlf + 2;
            }
        }

        if (has_transfer_encoding) {
            (void)send_json_error(
                sock, 501, "Transfer-Encoding is not implemented; use Content-Length", "not_implemented_error"
            );
            return;
        }
        if (content_length > kMaxHttpBodyBytes) {
            (void)send_json_error(sock, 413, "HTTP request body exceeds 1 MiB limit");
            return;
        }

        while (body.size() < content_length) {
            bool timed_out = false;
            int n = recv_with_deadline(sock, chunk, sizeof(chunk), deadline, &timed_out);
            if (timed_out) {
                (void)send_json_error(sock, 408, "HTTP body read deadline exceeded", "timeout_error");
                return;
            }
            if (n <= 0) {
                (void)send_json_error(sock, 400, "Incomplete HTTP request body");
                return;
            }
            body.append(chunk, static_cast<size_t>(n));
            if (body.size() > kMaxHttpBodyBytes) {
                (void)send_json_error(sock, 413, "HTTP request body exceeds 1 MiB limit");
                return;
            }
        }
        if (body.size() > content_length) {
            body.resize(content_length);
        }

        // Restore full send/recv socket timeout before response generation/writing.
        (void)set_socket_timeouts(sock, config.socket_timeout_ms);

        // Route dispatch
        if (target == "/health") {
            if (method != "GET") {
                (void)send_json_error(
                    sock, 405, "Method not allowed on /health", "method_not_allowed", "Allow: GET\r\n"
                );
                return;
            }
            std::ostringstream oss;
            oss.imbue(std::locale::classic());
            oss << "{\"status\":\"ok\",\"engine\":\"TPUForge\",\"model\":\""
                << json_escape(engine.model_id()) << "\",\"ready\":true,\"compile_count\":"
                << engine.compile_count() << "}";
            (void)send_json_response(sock, 200, oss.str());
            return;
        }

        if (target == "/v1/models") {
            if (method != "GET") {
                (void)send_json_error(
                    sock, 405, "Method not allowed on /v1/models", "method_not_allowed", "Allow: GET\r\n"
                );
                return;
            }
            std::ostringstream oss;
            oss.imbue(std::locale::classic());
            oss << "{\"object\":\"list\",\"data\":[{\"id\":\""
                << json_escape(engine.model_id())
                << "\",\"object\":\"model\",\"owned_by\":\"tpuforge\",\"max_seq_len\":"
                << engine.config().max_seq_len << ",\"vocab_size\":"
                << engine.config().vocab_size << "}]}";
            (void)send_json_response(sock, 200, oss.str());
            return;
        }

        if (target == "/v1/chat/completions") {
            if (method != "POST") {
                (void)send_json_error(
                    sock, 405, "Method not allowed on /v1/chat/completions",
                    "method_not_allowed", "Allow: POST\r\n"
                );
                return;
            }
            handle_chat_completions(sock, body);
            return;
        }

        const std::string not_found_msg = is_valid_utf8(target)
            ? ("Endpoint not found: " + target)
            : std::string("Endpoint not found");
        (void)send_json_error(sock, 404, not_found_msg, "not_found_error");
    }

    void handle_chat_completions(NativeSocket sock, const std::string& body) {
        if (body.empty()) {
            (void)send_json_error(sock, 400, "Request body cannot be empty");
            return;
        }

        JsonValue root;
        try {
            JsonParser parser(body);
            root = parser.parse();
        } catch (const std::exception& e) {
            (void)send_json_error(sock, 400, std::string("Invalid JSON payload: ") + e.what());
            return;
        }

        if (root.kind != JsonValue::Kind::Object) {
            (void)send_json_error(sock, 400, "JSON payload must be an object");
            return;
        }

        const JsonValue* msgs_v = root.find("messages");
        if (!msgs_v || msgs_v->kind != JsonValue::Kind::Array || msgs_v->arr_val.empty()) {
            (void)send_json_error(sock, 400, "'messages' must be a non-empty array");
            return;
        }

        GenerationRequest gen_req;
        gen_req.messages.reserve(msgs_v->arr_val.size());
        for (const auto& item : msgs_v->arr_val) {
            if (item.kind != JsonValue::Kind::Object) {
                (void)send_json_error(sock, 400, "Each element in 'messages' must be an object");
                return;
            }
            const JsonValue* role_v = item.find("role");
            const JsonValue* content_v = item.find("content");
            if (!role_v || role_v->kind != JsonValue::Kind::String || role_v->str_val.empty()) {
                (void)send_json_error(sock, 400, "Each message must have a non-empty string 'role'");
                return;
            }
            if (!content_v || content_v->kind != JsonValue::Kind::String) {
                (void)send_json_error(sock, 400, "Each message must have a string 'content'");
                return;
            }
            if (!is_valid_utf8(role_v->str_val) || !is_valid_utf8(content_v->str_val)) {
                (void)send_json_error(sock, 400, "Message role/content must be valid UTF-8");
                return;
            }
            gen_req.messages.push_back(ChatMessage{role_v->str_val, content_v->str_val});
        }

        if (const JsonValue* model_v = root.find("model")) {
            if (model_v->kind != JsonValue::Kind::String || model_v->str_val.empty()) {
                (void)send_json_error(sock, 400, "'model' must be a non-empty string");
                return;
            }
        }

        const JsonValue* max_tok_v = root.find("max_completion_tokens");
        if (!max_tok_v) {
            max_tok_v = root.find("max_tokens");
        }
        if (max_tok_v && max_tok_v->kind != JsonValue::Kind::Null) {
            if (max_tok_v->kind != JsonValue::Kind::Number || !max_tok_v->is_integer ||
                max_tok_v->int_val < 0 ||
                static_cast<uint64_t>(max_tok_v->int_val) > kMaxTokensPerRequest) {
                (void)send_json_error(
                    sock, 400, "'max_tokens' must be a non-negative integer <= 4096"
                );
                return;
            }
            gen_req.max_tokens = static_cast<size_t>(max_tok_v->int_val);
        }

        if (const JsonValue* temp_v = root.find("temperature")) {
            if (temp_v->kind != JsonValue::Kind::Null) {
                if (temp_v->kind != JsonValue::Kind::Number ||
                    !std::isfinite(temp_v->num_val) ||
                    temp_v->num_val < 0.0 || temp_v->num_val > 2.0) {
                    (void)send_json_error(sock, 400, "'temperature' must be a number in [0.0, 2.0]");
                    return;
                }
                gen_req.temperature = static_cast<float>(temp_v->num_val);
            }
        }

        if (const JsonValue* seed_v = root.find("seed")) {
            if (seed_v->kind != JsonValue::Kind::Null) {
                if (seed_v->kind != JsonValue::Kind::Number || !seed_v->is_integer || seed_v->int_val < 0) {
                    (void)send_json_error(sock, 400, "'seed' must be a non-negative integer");
                    return;
                }
                gen_req.seed = static_cast<uint64_t>(seed_v->int_val);
            }
        }

        bool stream = false;
        if (const JsonValue* stream_v = root.find("stream")) {
            if (stream_v->kind != JsonValue::Kind::Null) {
                if (stream_v->kind != JsonValue::Kind::Bool) {
                    (void)send_json_error(sock, 400, "'stream' must be a boolean");
                    return;
                }
                stream = stream_v->bool_val;
            }
        }

        if (const JsonValue* ignore_eos_v = root.find("ignore_eos")) {
            if (ignore_eos_v->kind != JsonValue::Kind::Null) {
                if (ignore_eos_v->kind != JsonValue::Kind::Bool) {
                    (void)send_json_error(sock, 400, "'ignore_eos' must be a boolean");
                    return;
                }
                gen_req.ignore_eos = ignore_eos_v->bool_val;
            }
        }

        try {
            (void)engine.validate_request(gen_req);
        } catch (const std::exception& e) {
            (void)send_json_error(sock, 400, e.what());
            return;
        }

        const uint64_t seq = completion_seq.fetch_add(1, std::memory_order_relaxed) + 1;
        const std::string completion_id = "chatcmpl-tpuforge-" + std::to_string(seq);
        const int64_t created_ts = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        const std::string& model_id = engine.model_id();

        if (!stream) {
            GenerationResult res;
            try {
                res = engine.generate(gen_req);
            } catch (const std::exception& e) {
                (void)send_json_error(
                    sock, 500, e.what(), "engine_error", "", config.socket_timeout_ms
                );
                return;
            }

            std::ostringstream oss;
            oss.imbue(std::locale::classic());
            oss << "{\"id\":\"" << json_escape(completion_id) << "\","
                << "\"object\":\"chat.completion\","
                << "\"created\":" << created_ts << ","
                << "\"model\":\"" << json_escape(model_id) << "\","
                << "\"choices\":[{\"index\":0,"
                << "\"message\":{\"role\":\"assistant\",\"content\":\"" << json_escape(res.text) << "\"},"
                << "\"finish_reason\":\"" << json_escape(res.finish_reason) << "\"}],"
                << build_telemetry_json(res) << "}";
            (void)send_json_response(sock, 200, oss.str(), "", config.socket_timeout_ms);
            return;
        }

        // Incremental SSE streaming decoupled from engine_mu_:
        // a dedicated generation thread pushes validated UTF-8 deltas into a lightweight
        // queue while this worker drains frames to the socket under a cumulative write budget.
        std::chrono::steady_clock::duration write_budget =
            std::chrono::milliseconds(config.socket_timeout_ms);

        const std::string sse_headers =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n\r\n";
        if (!send_all_with_budget(sock, sse_headers.data(), sse_headers.size(), &write_budget)) {
            return;
        }

        const std::string role_chunk = build_sse_chunk(
            completion_id, created_ts, model_id, "{\"role\":\"assistant\",\"content\":\"\"}", "null"
        );
        if (!send_all_with_budget(sock, role_chunk.data(), role_chunk.size(), &write_budget)) {
            return;
        }

        struct SseStreamState {
            std::mutex mu;
            std::condition_variable cv;
            std::deque<std::string> deltas;
            bool finished{false};
            bool failed{false};
            std::string error_msg;
            GenerationResult result;
            std::atomic<bool> aborted{false};
        } stream_state;

        struct GenThreadGuard {
            SseStreamState& st;
            std::thread worker;
            ~GenThreadGuard() {
                st.aborted.store(true, std::memory_order_release);
                if (worker.joinable()) {
                    worker.join();
                }
            }
        } gen_guard{stream_state, {}};

        gen_guard.worker = std::thread([this, &gen_req, &stream_state]() {
            try {
                GenerationResult res = engine.generate(
                    gen_req,
                    [&stream_state](const std::string& delta, int32_t) -> bool {
                        if (stream_state.aborted.load(std::memory_order_acquire)) {
                            return false;
                        }
                        if (!delta.empty()) {
                            std::lock_guard lk(stream_state.mu);
                            if (stream_state.aborted.load(std::memory_order_acquire)) {
                                return false;
                            }
                            stream_state.deltas.push_back(delta);
                        }
                        stream_state.cv.notify_one();
                        return true;
                    }
                );
                std::lock_guard lk(stream_state.mu);
                stream_state.result = std::move(res);
                stream_state.finished = true;
            } catch (const std::exception& e) {
                std::lock_guard lk(stream_state.mu);
                stream_state.failed = true;
                stream_state.error_msg = e.what();
                stream_state.finished = true;
            } catch (...) {
                std::lock_guard lk(stream_state.mu);
                stream_state.failed = true;
                stream_state.error_msg = "Unknown native engine failure";
                stream_state.finished = true;
            }
            stream_state.cv.notify_one();
        });

        bool write_ok = true;
        while (write_ok) {
            std::deque<std::string> batch;
            bool done = false;
            {
                std::unique_lock lk(stream_state.mu);
                stream_state.cv.wait(lk, [&stream_state]() {
                    return !stream_state.deltas.empty() || stream_state.finished;
                });
                batch.swap(stream_state.deltas);
                done = stream_state.finished && batch.empty();
            }

            while (!batch.empty()) {
                const std::string delta = std::move(batch.front());
                batch.pop_front();
                const std::string delta_json = "{\"content\":\"" + json_escape(delta) + "\"}";
                const std::string frame = build_sse_chunk(
                    completion_id, created_ts, model_id, delta_json, "null"
                );
                if (!send_all_with_budget(sock, frame.data(), frame.size(), &write_budget)) {
                    stream_state.aborted.store(true, std::memory_order_release);
                    write_ok = false;
                    break;
                }
            }
            if (done) {
                break;
            }
        }

        if (gen_guard.worker.joinable()) {
            gen_guard.worker.join();
        }
        if (!write_ok) {
            return;
        }

        if (stream_state.failed) {
            std::ostringstream eoss;
            eoss.imbue(std::locale::classic());
            eoss << "data: {\"error\":{\"message\":\"" << json_escape(stream_state.error_msg)
                 << "\",\"type\":\"engine_error\",\"code\":500}}\n\ndata: [DONE]\n\n";
            const std::string err_sse = eoss.str();
            (void)send_all_with_budget(sock, err_sse.data(), err_sse.size(), &write_budget);
            graceful_shutdown_send(sock);
            return;
        }

        std::string final_frame = build_sse_chunk(
            completion_id,
            created_ts,
            model_id,
            "{}",
            "\"" + json_escape(stream_state.result.finish_reason) + "\"",
            "," + build_telemetry_json(stream_state.result)
        );
        final_frame += "data: [DONE]\n\n";
        (void)send_all_with_budget(sock, final_frame.data(), final_frame.size(), &write_budget);
        graceful_shutdown_send(sock);
    }
};

OpenAIServer::OpenAIServer(ForgeEngine& engine, const Config& config)
    : impl_(std::make_unique<Impl>(engine, config)) {}

OpenAIServer::OpenAIServer(ForgeEngine& engine)
    : OpenAIServer(engine, Config{}) {}

OpenAIServer::~OpenAIServer() {
    stop();
}

bool OpenAIServer::start(std::string* error_out) {
    std::unique_lock lk(impl_->state_mu);
    impl_->state_cv.wait(lk, [this]() { return !impl_->stopping; });
    if (impl_->running.load(std::memory_order_acquire)) {
        if (error_out) {
            *error_out = "OpenAIServer is already running";
        }
        return false;
    }

    try {
        ensure_winsock_initialized();
    } catch (const std::exception& e) {
        if (error_out) {
            *error_out = e.what();
        }
        return false;
    }

    std::string host_ip = impl_->config.host;
    if (host_ip.empty() || host_ip == "localhost") {
        host_ip = "127.0.0.1";
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(impl_->config.port);
    if (inet_pton(AF_INET, host_ip.c_str(), &addr.sin_addr) != 1) {
        if (error_out) {
            *error_out = "Invalid IPv4 bind host: " + impl_->config.host;
        }
        return false;
    }

    NativeSocket sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == kInvalidSocket) {
        if (error_out) {
            *error_out = "Failed to create TCP listening socket";
        }
        return false;
    }

#if defined(_WIN32)
    int exclusive = 1;
    (void)setsockopt(
        sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
        reinterpret_cast<const char*>(&exclusive), sizeof(exclusive)
    );
#else
    int reuse = 1;
    (void)setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
#endif
    (void)set_socket_nonblocking(sock, true);

    if (bind(sock, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
        close_native_socket(sock);
        if (error_out) {
            *error_out = "Failed to bind TCP socket on " + host_ip + ":" + std::to_string(impl_->config.port);
        }
        return false;
    }

    if (listen(sock, static_cast<int>(std::max<size_t>(8, impl_->config.max_queued_connections))) != 0) {
        close_native_socket(sock);
        if (error_out) {
            *error_out = "listen() failed on " + host_ip;
        }
        return false;
    }

    sockaddr_in bound_addr{};
#if defined(_WIN32)
    int bound_len = sizeof(bound_addr);
#else
    socklen_t bound_len = sizeof(bound_addr);
#endif
    if (getsockname(sock, reinterpret_cast<sockaddr*>(&bound_addr), &bound_len) != 0) {
        close_native_socket(sock);
        if (error_out) {
            *error_out = "getsockname() failed after bind";
        }
        return false;
    }

    impl_->listen_sock = sock;
    impl_->bound_port.store(ntohs(bound_addr.sin_port), std::memory_order_release);
    impl_->stop_requested.store(false, std::memory_order_release);

    try {
        impl_->workers.reserve(impl_->config.worker_threads);
        for (size_t i = 0; i < impl_->config.worker_threads; ++i) {
            impl_->workers.emplace_back([this]() { impl_->worker_loop(); });
        }
        impl_->accept_thread = std::thread([this, sock]() { impl_->accept_loop(sock); });
    } catch (...) {
        impl_->stopping = true;
        impl_->stop_requested.store(true, std::memory_order_release);
        impl_->queue_cv.notify_all();
        std::thread local_accept = std::move(impl_->accept_thread);
        std::vector<std::thread> local_workers = std::move(impl_->workers);
        impl_->workers.clear();
        impl_->listen_sock = kInvalidSocket;
        impl_->bound_port.store(0, std::memory_order_release);
        lk.unlock();

        if (local_accept.joinable()) {
            local_accept.join();
        }
        for (auto& w : local_workers) {
            if (w.joinable()) {
                w.join();
            }
        }
        close_native_socket(sock);

        lk.lock();
        impl_->stopping = false;
        impl_->state_cv.notify_all();
        if (error_out) {
            *error_out = "Failed to launch HTTP worker/accept threads";
        }
        return false;
    }

    impl_->running.store(true, std::memory_order_release);
    return true;
}

void OpenAIServer::stop() {
    std::thread local_accept;
    std::vector<std::thread> local_workers;
    NativeSocket local_listen = kInvalidSocket;

    {
        std::unique_lock lk(impl_->state_mu);
        if (impl_->stopping) {
            impl_->state_cv.wait(lk, [this]() { return !impl_->stopping; });
            return;
        }
        if (!impl_->running.load(std::memory_order_acquire) &&
            impl_->listen_sock == kInvalidSocket) {
            return;
        }
        impl_->stopping = true;
        impl_->stop_requested.store(true, std::memory_order_release);
        impl_->running.store(false, std::memory_order_release);
        impl_->bound_port.store(0, std::memory_order_release);
        local_accept = std::move(impl_->accept_thread);
        local_workers = std::move(impl_->workers);
        impl_->workers.clear();
        local_listen = impl_->listen_sock;
        impl_->listen_sock = kInvalidSocket;
    }

    impl_->queue_cv.notify_all();

    if (local_accept.joinable()) {
        local_accept.join();
    }
    for (auto& w : local_workers) {
        if (w.joinable()) {
            w.join();
        }
    }

    {
        std::lock_guard qlk(impl_->queue_mu);
        while (!impl_->pending_clients.empty()) {
            close_native_socket(impl_->pending_clients.front());
            impl_->pending_clients.pop_front();
        }
    }

    close_native_socket(local_listen);

    {
        std::lock_guard lk(impl_->state_mu);
        impl_->stopping = false;
    }
    impl_->state_cv.notify_all();
}

bool OpenAIServer::is_running() const noexcept {
    return impl_->running.load(std::memory_order_acquire);
}

uint16_t OpenAIServer::bound_port() const noexcept {
    return impl_->bound_port.load(std::memory_order_acquire);
}

} // namespace tpuforge
