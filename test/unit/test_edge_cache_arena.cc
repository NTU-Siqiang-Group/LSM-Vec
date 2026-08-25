// EdgeLRUCache arena/open-addressing rewrite: differential stress test.
//
// Drives the cache and a reference model (per-shard FIFO-with-put-refresh,
// mirroring the documented semantics: get does not reorder, put-existing
// refreshes to front, eviction takes the shard tail) through randomized
// put/get/erase workloads and checks they agree exactly. The backward-shift
// deletion in the probe table is the fiddliest part of the rewrite, so erase
// weight is high and ids are chosen to collide across probe chains.

#include "doctest.h"

#include "astervec_index.h"

#include <deque>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

namespace {

// Reference shard model with the same semantics, on std containers.
struct RefShards {
    struct Shard {
        std::deque<astervec::node_id_t> order;  // front = newest
        std::unordered_map<astervec::node_id_t,
                           std::vector<astervec::node_id_t>> data;
    };
    size_t per_shard_cap;
    std::vector<Shard> shards;

    explicit RefShards(size_t capacity)
        : per_shard_cap(std::max<size_t>(1, capacity / 64)), shards(64) {}

    static size_t shard_for(astervec::node_id_t id) {
        std::uint64_t h = static_cast<std::uint64_t>(id);
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        return static_cast<size_t>(h % 64);
    }

    bool get(astervec::node_id_t id, std::vector<astervec::node_id_t>* out) {
        Shard& s = shards[shard_for(id)];
        auto it = s.data.find(id);
        if (it == s.data.end()) return false;
        *out = it->second;
        return true;
    }

    void put(astervec::node_id_t id, std::vector<astervec::node_id_t> nbrs) {
        Shard& s = shards[shard_for(id)];
        auto it = s.data.find(id);
        if (it != s.data.end()) {
            it->second = std::move(nbrs);
            for (auto oit = s.order.begin(); oit != s.order.end(); ++oit) {
                if (*oit == id) { s.order.erase(oit); break; }
            }
            s.order.push_front(id);
            return;
        }
        if (s.data.size() >= per_shard_cap) {
            astervec::node_id_t victim = s.order.back();
            s.order.pop_back();
            s.data.erase(victim);
        }
        s.order.push_front(id);
        s.data[id] = std::move(nbrs);
    }

    void erase(astervec::node_id_t id) {
        Shard& s = shards[shard_for(id)];
        if (s.data.erase(id) == 0) return;
        for (auto oit = s.order.begin(); oit != s.order.end(); ++oit) {
            if (*oit == id) { s.order.erase(oit); break; }
        }
    }
};

std::vector<astervec::node_id_t> payload(astervec::node_id_t id, int salt) {
    // Deterministic per-(id, salt) payload so value mismatches are caught.
    std::vector<astervec::node_id_t> v;
    v.reserve(4);
    for (int i = 0; i < 4; ++i) v.push_back(id * 31 + salt * 7 + i);
    return v;
}

}  // namespace

TEST_CASE("EdgeLRUCache arena rewrite matches reference model under churn") {
    constexpr size_t kCapacity = 64 * 13;  // 13 per shard — small, evicts a lot
    astervec::EdgeLRUCache cache(kCapacity);
    RefShards ref(kCapacity);

    std::mt19937_64 rng(20260825);
    // Narrow id range → heavy shard collisions and probe-chain overlap.
    std::uniform_int_distribution<astervec::node_id_t> id_dist(0, 4000);
    std::uniform_int_distribution<int> op_dist(0, 9);

    for (int step = 0; step < 200000; ++step) {
        astervec::node_id_t id = id_dist(rng);
        int op = op_dist(rng);
        if (op < 5) {                       // 50% put
            cache.put(id, payload(id, step & 7));
            ref.put(id, payload(id, step & 7));
        } else if (op < 8) {                // 30% get
            std::vector<astervec::node_id_t> got, want;
            bool ch = cache.get(id, &got);
            bool rh = ref.get(id, &want);
            REQUIRE(ch == rh);
            if (ch) REQUIRE(got == want);
        } else {                            // 20% erase (backward-shift stress)
            cache.erase(id);
            ref.erase(id);
        }
    }

    // Full sweep: every id agrees on presence and value at the end.
    size_t live = 0;
    for (astervec::node_id_t id = 0; id <= 4000; ++id) {
        std::vector<astervec::node_id_t> got, want;
        bool ch = cache.get(id, &got);
        bool rh = ref.get(id, &want);
        REQUIRE(ch == rh);
        if (ch) {
            REQUIRE(got == want);
            ++live;
        }
    }
    CHECK(live > 0);
}

TEST_CASE("EdgeLRUCache tiny capacity edge cases") {
    // capacity < shards → per-shard capacity clamps to 1: every put into a
    // shard evicts its previous occupant; erase on missing ids is a no-op.
    astervec::EdgeLRUCache cache(1);
    std::vector<astervec::node_id_t> out;

    cache.put(1, {10, 11});
    REQUIRE(cache.get(1, &out));
    REQUIRE(out == std::vector<astervec::node_id_t>{10, 11});

    cache.erase(999);           // absent: no-op
    cache.erase(1);
    REQUIRE_FALSE(cache.get(1, &out));

    // Re-put after erase exercises the free list, then same-shard eviction.
    cache.put(1, {12});
    cache.put(1 + 64 * 7, {13});  // may land in another shard; both live
    REQUIRE(cache.get(1, &out));
}

TEST_CASE("EdgeLRUCache round-trips huge and bit-63 ids (varint width)") {
    // LEB128 must round-trip the full uint64 range, including indirect ids
    // with bit 63 set (10-byte encodings) — both as keys and as neighbors.
    astervec::EdgeLRUCache cache(256);
    const astervec::node_id_t kBit63 = (astervec::node_id_t{1} << 63);
    std::vector<astervec::node_id_t> payload = {
        0, 1, 127, 128, 16383, 16384,
        kBit63, kBit63 | 12345,
        std::numeric_limits<astervec::node_id_t>::max() - 1,
    };
    std::vector<astervec::node_id_t> out;

    cache.put(kBit63 | 42, payload);
    REQUIRE(cache.get(kBit63 | 42, &out));
    REQUIRE(out == payload);

    cache.put(7, payload);
    REQUIRE(cache.get(7, &out));
    REQUIRE(out == payload);

    cache.erase(kBit63 | 42);
    REQUIRE_FALSE(cache.get(kBit63 | 42, &out));
    REQUIRE(cache.get(7, &out));
    REQUIRE(out == payload);
}
