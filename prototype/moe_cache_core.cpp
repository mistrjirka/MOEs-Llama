#include "moe_cache_core.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace moe_cache {

LayerCache::LayerCache(
        int32_t layer,
        int32_t capacity,
        double protected_fraction,
        bool second_touch_admission,
        bool admit_prefill)
    : layer_(layer),
      protected_capacity_(0),
      probation_capacity_(0),
      second_touch_admission_(second_touch_admission),
      admit_prefill_(admit_prefill) {
    if (capacity < 0) {
        throw std::invalid_argument("cache capacity cannot be negative");
    }
    if (!(protected_fraction >= 0.0 && protected_fraction <= 1.0)) {
        throw std::invalid_argument("protected fraction must be in [0, 1]");
    }

    slots_.resize(static_cast<size_t>(capacity));
    if (capacity > 0) {
        protected_capacity_ = static_cast<int32_t>(capacity * protected_fraction + 0.5);
        protected_capacity_ = std::max<int32_t>(0, std::min<int32_t>(capacity, protected_capacity_));
        probation_capacity_ = capacity - protected_capacity_;
        if (probation_capacity_ == 0) {
            probation_capacity_ = 1;
            protected_capacity_ = capacity - 1;
        }
    }
}

bool LayerCache::handle_matches(const Handle & handle, SlotState required) const {
    if (!handle.valid() || handle.slot >= capacity()) {
        return false;
    }
    const Slot & slot = slots_[static_cast<size_t>(handle.slot)];
    return slot.generation == handle.generation && slot.state == required;
}

int32_t LayerCache::choose_empty_slot() const {
    for (int32_t index = 0; index < capacity(); ++index) {
        if (slots_[static_cast<size_t>(index)].state == SlotState::empty) {
            return index;
        }
    }
    return -1;
}

int32_t LayerCache::choose_probation_victim() const {
    int32_t victim = -1;
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (int32_t index = 0; index < capacity(); ++index) {
        const Slot & slot = slots_[static_cast<size_t>(index)];
        if (slot.state != SlotState::ready ||
            slot.segment != Segment::probation ||
            slot.in_flight != 0) {
            continue;
        }
        if (slot.last_use < oldest) {
            oldest = slot.last_use;
            victim = index;
        }
    }
    return victim;
}

int32_t LayerCache::choose_lru_protected(int32_t except_slot) const {
    int32_t victim = -1;
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (int32_t index = 0; index < capacity(); ++index) {
        if (index == except_slot) {
            continue;
        }
        const Slot & slot = slots_[static_cast<size_t>(index)];
        if (slot.state != SlotState::ready || slot.segment != Segment::protected_segment) {
            continue;
        }
        if (slot.last_use < oldest) {
            oldest = slot.last_use;
            victim = index;
        }
    }
    return victim;
}

int32_t LayerCache::count_segment(Segment segment) const {
    int32_t count = 0;
    for (const Slot & slot : slots_) {
        if (slot.state != SlotState::empty && slot.segment == segment) {
            ++count;
        }
    }
    return count;
}

void LayerCache::promote_to_protected(int32_t slot_index) {
    if (protected_capacity_ == 0) {
        return;
    }
    Slot & slot = slots_[static_cast<size_t>(slot_index)];
    if (slot.segment == Segment::protected_segment) {
        return;
    }

    if (count_segment(Segment::protected_segment) >= protected_capacity_) {
        const int32_t demote = choose_lru_protected(slot_index);
        if (demote >= 0) {
            slots_[static_cast<size_t>(demote)].segment = Segment::probation;
        }
    }
    slot.segment = Segment::protected_segment;
}

void LayerCache::touch_ready_hit(int32_t slot_index) {
    Slot & slot = slots_[static_cast<size_t>(slot_index)];
    ++clock_;
    slot.last_use = clock_;
    if (slot.frequency != std::numeric_limits<uint32_t>::max()) {
        ++slot.frequency;
    }
    if (slot.segment == Segment::probation) {
        promote_to_protected(slot_index);
    }
}

RequestResult LayerCache::request(int32_t expert, const AccessContext & context) {
    ++clock_;
    const auto existing = expert_to_slot_.find(expert);
    if (existing != expert_to_slot_.end()) {
        const int32_t index = existing->second;
        Slot & slot = slots_[static_cast<size_t>(index)];
        slot.last_use = clock_;
        if (slot.state == SlotState::loading) {
            return {RequestStatus::loading, {index, slot.generation}, -1};
        }
        if (slot.state == SlotState::ready) {
            if (slot.frequency != std::numeric_limits<uint32_t>::max()) {
                ++slot.frequency;
            }
            if (slot.segment == Segment::probation) {
                promote_to_protected(index);
            }
            ++slot.in_flight;
            return {RequestStatus::hit, {index, slot.generation}, -1};
        }
        throw std::logic_error("expert map references an empty slot");
    }

    if (capacity() == 0 || !context.allow_admission || (context.prefill && !admit_prefill_)) {
        return {RequestStatus::miss_no_admission, {}, -1};
    }

    if (second_touch_admission_) {
        auto history = admission_history_.find(expert);
        if (history == admission_history_.end()) {
            admission_history_[expert] = 1;
            return {RequestStatus::miss_no_admission, {}, -1};
        }
        admission_history_.erase(history);
    }

    int32_t index = choose_empty_slot();
    if (index < 0) {
        index = choose_probation_victim();
    }
    if (index < 0) {
        return {RequestStatus::blocked, {}, -1};
    }

    Slot & slot = slots_[static_cast<size_t>(index)];
    const int32_t evicted = slot.state == SlotState::empty ? -1 : slot.expert;
    if (evicted >= 0) {
        expert_to_slot_.erase(evicted);
    }

    ++slot.generation;
    slot.expert = expert;
    slot.state = SlotState::loading;
    slot.segment = Segment::probation;
    slot.in_flight = 0;
    slot.frequency = 1;
    slot.last_use = clock_;
    expert_to_slot_[expert] = index;

    return {RequestStatus::miss_reserved, {index, slot.generation}, evicted};
}

bool LayerCache::mark_ready(const Handle & handle) {
    if (!handle_matches(handle, SlotState::loading)) {
        return false;
    }
    slots_[static_cast<size_t>(handle.slot)].state = SlotState::ready;
    return true;
}

bool LayerCache::acquire(const Handle & handle) {
    if (!handle_matches(handle, SlotState::ready)) {
        return false;
    }
    Slot & slot = slots_[static_cast<size_t>(handle.slot)];
    ++clock_;
    slot.last_use = clock_;
    ++slot.in_flight;
    return true;
}

bool LayerCache::release(const Handle & handle) {
    if (!handle_matches(handle, SlotState::ready)) {
        return false;
    }
    Slot & slot = slots_[static_cast<size_t>(handle.slot)];
    if (slot.in_flight == 0) {
        return false;
    }
    --slot.in_flight;
    return true;
}

void LayerCache::decay_frequencies() {
    for (Slot & slot : slots_) {
        slot.frequency = (slot.frequency + 1) / 2;
    }
}

void LayerCache::clear_admission_history() {
    admission_history_.clear();
}

bool LayerCache::contains(int32_t expert) const {
    return expert_to_slot_.find(expert) != expert_to_slot_.end();
}

SlotSnapshot LayerCache::snapshot(int32_t slot_index) const {
    if (slot_index < 0 || slot_index >= capacity()) {
        throw std::out_of_range("slot index out of range");
    }
    const Slot & slot = slots_[static_cast<size_t>(slot_index)];
    return {
        slot_index,
        slot.expert,
        slot.state,
        slot.segment,
        slot.in_flight,
        slot.frequency,
        slot.last_use,
        slot.generation,
    };
}

std::vector<SlotSnapshot> LayerCache::snapshots() const {
    std::vector<SlotSnapshot> result;
    result.reserve(slots_.size());
    for (int32_t index = 0; index < capacity(); ++index) {
        result.push_back(snapshot(index));
    }
    return result;
}

} // namespace moe_cache
