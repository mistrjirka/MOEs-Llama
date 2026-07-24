#include "../prototype/moe_cache_core.h"

#include <cassert>
#include <iostream>

using namespace moe_cache;

static Handle admit(LayerCache & cache, int expert) {
    RequestResult first = cache.request(expert);
    assert(first.status == RequestStatus::miss_no_admission);
    RequestResult second = cache.request(expert);
    assert(second.status == RequestStatus::miss_reserved);
    assert(cache.mark_ready(second.handle));
    assert(cache.acquire(second.handle));
    assert(cache.release(second.handle));
    return second.handle;
}

static void test_second_touch_and_hit() {
    LayerCache cache(0, 4, 0.75, true, false);
    Handle handle = admit(cache, 7);
    RequestResult hit = cache.request(7);
    assert(hit.status == RequestStatus::hit);
    assert(hit.handle.slot == handle.slot);
    assert(cache.release(hit.handle));
    assert(cache.snapshot(hit.handle.slot).segment == Segment::protected_segment);
}

static void test_prefill_no_admission() {
    LayerCache cache(0, 2, 0.5, false, false);
    AccessContext prefill;
    prefill.prefill = true;
    for (int i = 0; i < 3; ++i) {
        assert(cache.request(4, prefill).status == RequestStatus::miss_no_admission);
    }
    assert(!cache.contains(4));
}

static void test_in_flight_not_evicted() {
    LayerCache cache(0, 2, 0.5, false, false);

    RequestResult a = cache.request(1);
    assert(a.status == RequestStatus::miss_reserved);
    assert(cache.mark_ready(a.handle));
    assert(cache.acquire(a.handle));

    RequestResult b = cache.request(2);
    assert(b.status == RequestStatus::miss_reserved);
    assert(cache.mark_ready(b.handle));

    // Expert 1 is the probation entry and is in flight. Expert 2 cannot evict it
    // after promotion leaves no evictable probation slot.
    RequestResult b_hit = cache.request(2);
    assert(b_hit.status == RequestStatus::hit);
    assert(cache.release(b_hit.handle));

    RequestResult blocked = cache.request(3);
    assert(blocked.status == RequestStatus::blocked);
    assert(cache.release(a.handle));

    RequestResult admitted = cache.request(3);
    assert(admitted.status == RequestStatus::miss_reserved);
}

static void test_stale_generation_rejected() {
    LayerCache cache(0, 1, 0.0, false, false);
    RequestResult first = cache.request(1);
    assert(first.status == RequestStatus::miss_reserved);
    assert(cache.mark_ready(first.handle));

    RequestResult second = cache.request(2);
    assert(second.status == RequestStatus::miss_reserved);
    assert(second.handle.slot == first.handle.slot);
    assert(second.handle.generation != first.handle.generation);
    assert(!cache.mark_ready(first.handle));
    assert(!cache.acquire(first.handle));
}

int main() {
    test_second_touch_and_hit();
    test_prefill_no_admission();
    test_in_flight_not_evicted();
    test_stale_generation_rejected();
    std::cout << "ok\n";
    return 0;
}
