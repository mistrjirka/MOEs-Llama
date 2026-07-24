#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace moe_cache {

enum class Segment : uint8_t {
    probation,
    protected_segment,
};

enum class SlotState : uint8_t {
    empty,
    loading,
    ready,
};

enum class RequestStatus : uint8_t {
    hit,
    loading,
    miss_no_admission,
    miss_reserved,
    blocked,
};

struct AccessContext {
    bool prefill;
    bool allow_admission;

    AccessContext(bool prefill_ = false, bool allow_admission_ = true)
        : prefill(prefill_), allow_admission(allow_admission_) {}
};

struct Handle {
    int32_t slot;
    uint64_t generation;

    Handle(int32_t slot_ = -1, uint64_t generation_ = 0)
        : slot(slot_), generation(generation_) {}

    bool valid() const { return slot >= 0; }
};

struct RequestResult {
    RequestStatus status;
    Handle handle;
    int32_t evicted_expert;

    RequestResult(
            RequestStatus status_ = RequestStatus::blocked,
            Handle handle_ = Handle(),
            int32_t evicted_expert_ = -1)
        : status(status_), handle(handle_), evicted_expert(evicted_expert_) {}
};

struct SlotSnapshot {
    int32_t slot;
    int32_t expert;
    SlotState state;
    Segment segment;
    uint32_t in_flight;
    uint32_t frequency;
    uint64_t last_use;
    uint64_t generation;

    SlotSnapshot(
            int32_t slot_ = -1,
            int32_t expert_ = -1,
            SlotState state_ = SlotState::empty,
            Segment segment_ = Segment::probation,
            uint32_t in_flight_ = 0,
            uint32_t frequency_ = 0,
            uint64_t last_use_ = 0,
            uint64_t generation_ = 0)
        : slot(slot_),
          expert(expert_),
          state(state_),
          segment(segment_),
          in_flight(in_flight_),
          frequency(frequency_),
          last_use(last_use_),
          generation(generation_) {}
};

class LayerCache {
public:
    LayerCache(
        int32_t layer,
        int32_t capacity,
        double protected_fraction = 0.8,
        bool second_touch_admission = true,
        bool admit_prefill = false);

    RequestResult request(int32_t expert, const AccessContext & context = AccessContext());

    // Transition a reserved slot from loading to ready after all matrices in the
    // expert bundle have completed upload.
    bool mark_ready(const Handle & handle);

    // Acquire an already-ready handle. Successful acquisition increments the
    // in-flight count and protects the slot from eviction.
    bool acquire(const Handle & handle);

    // Release one in-flight reference. Stale handles are rejected by generation.
    bool release(const Handle & handle);

    void decay_frequencies();
    void clear_admission_history();

    int32_t layer() const { return layer_; }
    int32_t capacity() const { return static_cast<int32_t>(slots_.size()); }
    int32_t protected_capacity() const { return protected_capacity_; }
    int32_t probation_capacity() const { return probation_capacity_; }

    bool contains(int32_t expert) const;
    SlotSnapshot snapshot(int32_t slot) const;
    std::vector<SlotSnapshot> snapshots() const;

private:
    struct Slot {
        int32_t expert = -1;
        SlotState state = SlotState::empty;
        Segment segment = Segment::probation;
        uint32_t in_flight = 0;
        uint32_t frequency = 0;
        uint64_t last_use = 0;
        uint64_t generation = 0;
    };

    int32_t layer_;
    int32_t protected_capacity_;
    int32_t probation_capacity_;
    bool second_touch_admission_;
    bool admit_prefill_;
    uint64_t clock_ = 0;
    std::vector<Slot> slots_;
    std::unordered_map<int32_t, int32_t> expert_to_slot_;
    std::unordered_map<int32_t, uint8_t> admission_history_;

    bool handle_matches(const Handle & handle, SlotState required) const;
    int32_t choose_empty_slot() const;
    int32_t choose_probation_victim() const;
    int32_t choose_lru_protected(int32_t except_slot) const;
    int32_t count_segment(Segment segment) const;
    void touch_ready_hit(int32_t slot);
    void promote_to_protected(int32_t slot);
};

} // namespace moe_cache
