// Exercise production tag selection with independently specified cache states.
#include <cstdio>
#include <stdexcept>
#include <string>
#include "gpgpu-sim/gpu-cache.h"

static void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

static constexpr unsigned kSets = 4, kWays = 4, kLineBytes = 128;

class fixture_tags : public tag_array {
 public:
  explicit fixture_tags(cache_config &config) : tag_array(config, 0, 0) {}
  void line(unsigned set, unsigned way, cache_block_state state,
            unsigned long long allocated, unsigned long long accessed,
            bool reserve_other_sector = false) {
    auto *block = m_lines[set * kWays + way];
    const new_addr_type addr = 0x10000 + (way * kSets + set) * kLineBytes;
    block->allocate(m_config.tag(addr), addr, allocated, mem_access_sector_mask_t(1));
    block->set_status(state, mem_access_sector_mask_t(1));
    block->set_last_access_time(accessed, mem_access_sector_mask_t(1));
    if (reserve_other_sector)
      block->set_status(RESERVED, mem_access_sector_mask_t(2));
    if (block->is_modified_line()) ++m_dirty;
  }
};

static void run(char policy, char type) {
  cache_config config;
  char spec[128];
  std::snprintf(spec, sizeof(spec), "%c:4:128:4,%c:T:m:L:L,A:16:4,8:0,32", type, policy);
  config.init(spec, FuncCachePreferNone);
  config.m_wr_percent = 25;
  const auto mask = mem_access_sector_mask_t(1);
  const new_addr_type miss_addr = 0x20000;
  unsigned index = ~0U;
  {
    fixture_tags tags(config);
    // Three modified ways + one reserved: below the global 25% threshold.
    // LRU selects way 1; FIFO selects way 0. A reservation is never a victim.
    tags.line(0, 0, MODIFIED, 10, 30);
    tags.line(0, 1, MODIFIED, 20, 10);
    tags.line(0, 2, MODIFIED, 30, 20);
    tags.line(0, 3, RESERVED, 1, 1);
    require(tags.probe(miss_addr, index, mask, false) == MISS,
            "modified-line protection must not deadlock an allocatable set");
    require(index == (policy == 'L' ? 1U : 0U), "fallback ignored replacement order");
    require(tags.probe(miss_addr, index, mask, true) == MISS, "store miss cannot progress");
    require(tags.probe(0x10400, index, mask, true) == HIT && index == 2,
            "fallback must not hide a later tag hit");
    require(tags.probe(0x10600, index, mask, false) == HIT_RESERVED && index == 3,
            "reserved hit changed");
  }
  {
    fixture_tags tags(config);
    tags.line(0, 0, MODIFIED, 1, 1);
    tags.line(0, 1, MODIFIED, 2, 2);
    tags.line(0, 2, VALID, 100, 100);
    tags.line(0, 3, RESERVED, 0, 0);
    require(tags.probe(miss_addr, index, mask, false) == MISS && index == 2,
            "below-threshold policy must still prefer clean lines");
  }
  {
    fixture_tags tags(config);
    tags.line(0, 0, MODIFIED, 1, 1);
    tags.line(0, 1, MODIFIED, 2, 2);
    tags.line(0, 2, RESERVED, 0, 0);
    require(tags.probe(miss_addr, index, mask, false) == MISS && index == 3,
            "invalid way must precede modified fallback");
  }
  {
    fixture_tags tags(config);
    for (unsigned way = 0; way < 4; ++way) tags.line(0, way, RESERVED, 1, 1);
    require(tags.probe(miss_addr, index, mask, false) == RESERVATION_FAIL,
            "all-reserved set must retain backpressure");
  }
  {
    fixture_tags tags(config);
    tags.line(0, 0, MODIFIED, 1, 1);
    tags.line(0, 1, MODIFIED, 2, 2);
    tags.line(0, 2, VALID, 100, 100);
    tags.line(0, 3, RESERVED, 0, 0);
    tags.line(1, 0, MODIFIED, 1, 1);
    tags.line(1, 1, MODIFIED, 1, 1);
    require(tags.probe(miss_addr, index, mask, false) == MISS && index == 0,
            "at-threshold replacement order changed");
  }
  if (type == 'S') {
    fixture_tags tags(config);
    tags.line(0, 0, MODIFIED, 1, 1, true);
    tags.line(0, 1, MODIFIED, 2, 2);
    tags.line(0, 2, RESERVED, 1, 1);
    tags.line(0, 3, RESERVED, 1, 1);
    require(tags.probe(miss_addr, index, mask, false) == MISS && index == 1,
            "modified line with reserved sector must never be evicted");
    require(tags.probe(0x10000, index, mem_access_sector_mask_t(4), false) == SECTOR_MISS,
            "sector miss changed");
  }
}

int main() {
  try {
    for (char policy : {'L', 'F'}) for (char type : {'S', 'N'}) run(policy, type);
    std::puts("CACHE_REPLACEMENT_PASS: LRU/FIFO, sector/normal, read/store, protected/clean/invalid/reserved/hit");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "CACHE_REPLACEMENT_FAIL: %s\n", error.what());
    return 1;
  }
}
