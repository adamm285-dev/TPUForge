#pragma once

#include "tpuforge/pjrt_driver.h"
#include "tpuforge/static_arena.h"
#include "tpuforge/tree_speculator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace tpuforge {

struct ChatMessage {
    std::string role;
    std::string content;
};

// Stateful RFC 3629 UTF-8 byte accumulator so streaming multi-byte sequences
// never split across SSE delta chunks or emit invalid UTF-8.
class Utf8StreamDecoder {
public:
    Utf8StreamDecoder() = default;

    std::string push_byte(uint8_t b);
    std::string push_token(int32_t token_id, size_t vocab_size, bool skip_special = true);
    std::string flush();
    void reset() noexcept;
    bool saw_error() const noexcept { return saw_error_; }

private:
    std::array<uint8_t, 4> pending_{};
    size_t pending_len_{0};
    size_t expected_len_{0};
    bool saw_error_{false};
};

// Byte-level tokenizer with isolated control tokens:
//   0=<pad>, 1=<bos>, 2=<eos>, 3=<sep>, 4..259=raw byte 0x00..0xFF.
// Because raw bytes never map to 0..3, user content cannot forge role boundaries.
class ByteTokenizer {
public:
    static constexpr int32_t kPadToken = 0;
    static constexpr int32_t kBosToken = 1;
    static constexpr int32_t kEosToken = 2;
    static constexpr int32_t kSepToken = 3;
    static constexpr int32_t kByteOffset = 4;
    static constexpr size_t kMinVocabSize = 260;

    explicit ByteTokenizer(size_t vocab_size = kMinVocabSize);

    size_t vocab_size() const noexcept { return vocab_size_; }

    std::vector<int32_t> encode(
        const std::string& text,
        bool add_bos = false,
        bool add_eos = false
    ) const;

    std::vector<int32_t> encode_chat(const std::vector<ChatMessage>& messages) const;

    std::string decode(const std::vector<int32_t>& tokens, bool skip_special = true) const;

private:
    size_t vocab_size_{kMinVocabSize};
};

struct GenerationRequest {
    std::vector<ChatMessage> messages;
    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> eos_token_ids;
    size_t max_tokens{64};
    float temperature{0.0f};
    std::optional<uint64_t> seed;
    bool ignore_eos{false};
};

struct GenerationResult {
    std::string text;
    std::vector<int32_t> token_ids;
    std::string finish_reason{"length"};
    size_t prompt_tokens_count{0};
    size_t completion_tokens_count{0};
    size_t total_tokens_count{0};
    size_t target_forward_passes{0};
    size_t decode_forward_passes{0};
    size_t draft_tokens_accepted{0};
    double mean_tokens_per_decode_pass{0.0};
    size_t compile_count{0};
};

// Thread-safe engine combining static 128-row causal prefill and
// 128-node systolic tree speculative decoding over either a compiled
// PJRT ZeroRecompileExecutor or a custom target-step callback.
class ForgeEngine {
public:
    using TargetStepFn = std::function<void(
        int32_t valid_prefix_len,
        const int32_t* token_tile_128,
        const int32_t* pos_tile_128,
        const float* mask_128x128,
        float* out_logits_128xV
    )>;
    using CommitKvFn = std::function<void(
        int32_t prev_valid_prefix_len,
        const std::vector<int32_t>& accepted_row_indices
    )>;
    using PrefillStepFn = std::function<void(
        int32_t valid_prefix_len,
        const int32_t* token_tile,
        const int32_t* pos_tile,
        const float* mask_TxT
    )>;
    using GreedyStepFn = std::function<void(
        int32_t valid_prefix_len,
        const int32_t* token_tile,
        const int32_t* pos_tile,
        const float* mask_TxT,
        int32_t* out_argmax_tokens
    )>;
    using StreamTokenCallback = std::function<bool(const std::string& utf8_delta, int32_t token_id)>;

    ForgeEngine(
        PjrtDriver& driver,
        const StaticArena::Config& config,
        const std::string& program_code,
        const std::string& program_format = "mlir",
        std::string model_id = "tpuforge-systolic-128"
    );

    ForgeEngine(
        const StaticArena::Config& config,
        TargetStepFn target_step_fn,
        CommitKvFn commit_kv_fn = nullptr,
        std::string model_id = "tpuforge-custom-128",
        PrefillStepFn prefill_step_fn = nullptr,
        GreedyStepFn greedy_step_fn = nullptr
    );

    ~ForgeEngine() = default;

    ForgeEngine(const ForgeEngine&) = delete;
    ForgeEngine& operator=(const ForgeEngine&) = delete;
    ForgeEngine(ForgeEngine&&) = delete;
    ForgeEngine& operator=(ForgeEngine&&) = delete;

    // Validates request bounds and resolves prompt token IDs before locking or streaming headers.
    std::vector<int32_t> validate_request(const GenerationRequest& req) const;

    // Runs prefill + 128-node systolic speculative decoding under internal mutex serialization.
    GenerationResult generate(
        const GenerationRequest& req,
        StreamTokenCallback stream_cb = nullptr
    );

    const std::string& model_id() const noexcept { return model_id_; }
    const StaticArena::Config& config() const noexcept { return config_; }
    const ByteTokenizer& tokenizer() const noexcept { return tokenizer_; }
    size_t compile_count() const;

private:
    void execute_tile_step();

    StaticArena::Config config_;
    std::string model_id_;
    ByteTokenizer tokenizer_;
    StaticArena arena_;
    std::unique_ptr<ZeroRecompileExecutor> executor_;
    TargetStepFn target_step_fn_;
    CommitKvFn commit_kv_fn_;
    PrefillStepFn prefill_step_fn_;
    GreedyStepFn greedy_step_fn_;
    std::vector<float> pjrt_step_k_;
    std::vector<float> pjrt_step_v_;
    mutable std::mutex engine_mu_;
};

// Lightweight thread-pooled HTTP/1.1 OpenAI-compatible server exposing:
//   GET  /health
//   GET  /v1/models
//   POST /v1/chat/completions  (non-streaming JSON and SSE stream=true)
class OpenAIServer {
public:
    struct Config {
        std::string host{"127.0.0.1"};
        uint16_t port{8080};
        size_t worker_threads{4};
        size_t max_queued_connections{64};
        uint32_t socket_timeout_ms{5000};
    };

    OpenAIServer(ForgeEngine& engine, const Config& config);
    explicit OpenAIServer(ForgeEngine& engine);
    ~OpenAIServer();

    OpenAIServer(const OpenAIServer&) = delete;
    OpenAIServer& operator=(const OpenAIServer&) = delete;
    OpenAIServer(OpenAIServer&&) = delete;
    OpenAIServer& operator=(OpenAIServer&&) = delete;

    bool start(std::string* error_out = nullptr);
    void stop();
    bool is_running() const noexcept;
    uint16_t bound_port() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tpuforge
