// Exercise the production sector splitter and LRC trace lifecycle without
// issuing shader instructions or changing cache behavior.
#include <clocale>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpgpu_context.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "gpgpu-sim/l2cache.h"
#include "gpgpu-sim/l2_traffic_trace.h"
#include "gpgpu-sim/trace_file_manager.h"
#include "option_parser.h"
#include "trace_driven.h"

namespace {
void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

class fixture_gpu : public trace_gpgpu_sim {
 public:
  fixture_gpu(const gpgpu_sim_config &config, gpgpu_context *ctx)
      : trace_gpgpu_sim(config, ctx) {}
  using gpgpu_sim::handle_lrc_reply;
  memory_sub_partition *partition(unsigned id) {
    return m_memory_sub_partition.at(id);
  }
};

unsigned json_unsigned(const std::string &line, const char *key) {
  const std::string prefix = std::string("\"") + key + "\":";
  const auto pos = line.find(prefix);
  require(pos != std::string::npos, "missing numeric audit field");
  return std::stoul(line.substr(pos + prefix.size()));
}

unsigned long long json_address(const std::string &line) {
  const std::string prefix = "\"addr\":\"";
  const auto pos = line.find(prefix);
  require(pos != std::string::npos, "missing audit address");
  return std::stoull(line.substr(pos + prefix.size()), nullptr, 16);
}

std::vector<mem_fetch *> children(fixture_gpu *gpu, mem_fetch *parent) {
  TraceFileManager::instance().flush_all();
  std::ifstream audit(std::getenv("L2_TRAFFIC_AUDIT_FILE"));
  std::vector<mem_fetch *> result;
  std::map<unsigned long long, unsigned> service_by_addr;
  for (std::string line; std::getline(audit, line);) {
    if (line.find("\"event\":\"ingress\"") == std::string::npos ||
        json_unsigned(line, "source_uid") != parent->get_request_uid())
      continue;
    const unsigned uid = json_unsigned(line, "transport_uid");
    const auto addr = json_address(line);
    service_by_addr.emplace(addr, uid);
    auto &entry = gpu->partition(parent->get_sub_partition_id())
                      ->get_lrc()->get_entry(addr, service_by_addr.at(addr));
    mem_fetch *child = nullptr;
    for (const auto &member : entry)
      if (member.first->get_request_uid() == uid) child = member.first;
    require(child != nullptr, "audited split child missing from LRC");
    result.push_back(child);
  }
  return result;
}

mem_fetch *make_request(fixture_gpu *gpu, gpgpu_context *ctx,
                        unsigned long long addr, unsigned access_size,
                        unsigned packet_size, unsigned mask,
                        l2_traffic_source source, unsigned ordinal,
                        bool shared_constructor,
                        unsigned byte_sectors = UINT_MAX,
                        unsigned long long source_cycle = 100,
                        bool multicast = false) {
  auto inst = std::make_shared<warp_inst_t>(gpu->getShaderCoreConfig());
  inst->pc = 0x100 + ordinal * 0x10;
  const bool write = source == L2_SOURCE_L1_WRITE ||
      source == L2_SOURCE_L1_BYPASS_STORE || source == L2_SOURCE_TMA_STORE;
  const bool tma = source == L2_SOURCE_TMA_LOAD || source == L2_SOURCE_TMA_STORE;
  inst->op = tma ? TMA_OP : (write ? STORE_OP : LOAD_OP);
  inst->memory_op = write ? memory_store : memory_load;
  inst->space = memory_space_t(global_space);
  inst->cache_op = CACHE_GLOBAL;
  inst->latency = inst->initiation_interval = 1;
  inst->set_kernel_uid(9001);
  inst->set_kernel_name("l2_traffic_runtime");
  inst->set_inst_ordinal(ordinal);
  active_mask_t active;
  active.set();
  inst->issue(active, 0, 100, 7, 0, 0);
  mem_access_byte_mask_t bytes;
  if (byte_sectors == UINT_MAX) byte_sectors = mask;
  for (unsigned i = 0; i < MAX_MEMORY_ACCESS_SIZE; ++i)
    if (byte_sectors & (1U << (i / SECTOR_SIZE))) bytes.set(i);
  const mem_access_t access(write ? GLOBAL_ACC_W : GLOBAL_ACC_R, addr,
                            access_size, write, active, bytes,
                            mem_access_sector_mask_t(mask), ctx, tma, 0,
                            multicast, multicast ? 0xf : 0);
  mem_fetch *parent = shared_constructor
      ? new mem_fetch(access, inst, 0, READ_PACKET_SIZE, 0, 0, 0,
                       gpu->getMemoryConfig(), 100)
      : new mem_fetch(access, inst.get(), 0, READ_PACKET_SIZE, 0, 0, 0,
                       gpu->getMemoryConfig(), 100);
  require(!parent->has_l2_source(), "new parent unexpectedly has provenance");
  require(parent->get_l2_source_request_uid() == UINT_MAX,
          "unclassified parent must have unknown source UID");
  parent->set_data_size(packet_size);
  if (source != L2_SOURCE_UNKNOWN) parent->set_l2_source(source, source_cycle);
  return parent;
}

void check_metadata(const mem_fetch *child, const mem_fetch *parent) {
  require(child->get_l2_source() == parent->get_l2_source(), "source lost");
  require(child->get_l2_source_request_uid() == parent->get_request_uid(),
          "original departure UID lost");
  require(child->get_l2_source_cycle() == parent->get_l2_source_cycle(),
          "departure cycle lost");
  require(child->get_trace_metadata() == parent->get_trace_metadata(),
          "immutable instruction metadata not inherited");
  require(child->get_trace_metadata()->valid, "invalid instruction metadata");
  require(child->get_trace_metadata()->kernel_uid == 9001, "kernel lost");
  require(child->get_trace_metadata()->dynamic_wid == 7, "warp lost");
}

void check_constructors(fixture_gpu *gpu, mem_fetch *parent) {
  const auto &access = parent->get_mem_access();
  mem_fetch raw_child(access, static_cast<const warp_inst_t *>(nullptr),
                      0, READ_PACKET_SIZE, 0, 0, 0,
                      gpu->getMemoryConfig(), 120, parent);
  mem_fetch shared_child(access, std::shared_ptr<warp_inst_t>(),
                         0, READ_PACKET_SIZE, 0, 0, 0,
                         gpu->getMemoryConfig(), 120, parent);
  check_metadata(&raw_child, parent);
  check_metadata(&shared_child, parent);
  const auto progress = mem_fetch::memory_progress();
  raw_child.set_status(IN_ICNT_TO_SHADER, 120);
  require(mem_fetch::memory_progress() == progress + 1,
          "real memory status transition did not advance drain watchdog");
  raw_child.set_status(IN_ICNT_TO_SHADER, 121);
  require(mem_fetch::memory_progress() == progress + 1,
          "blocked status retry incorrectly reset drain watchdog");
  mem_fetch write_allocate(access, static_cast<const warp_inst_t *>(nullptr),
                           0, READ_PACKET_SIZE, 0, 0, 0,
                           gpu->getMemoryConfig(), 120, nullptr, parent);
  require(!write_allocate.has_l2_source(),
          "original-write inheritance must not reuse departure identity");
  write_allocate.inherit_trace_metadata(parent);
  require(!write_allocate.has_l2_source(),
          "instruction metadata inheritance must not copy source identity");
}

void split_case(fixture_gpu *gpu, gpgpu_context *ctx, const char *name,
                 unsigned long long addr, unsigned access_size,
                 unsigned packet_size, unsigned mask,
                 l2_traffic_source source, unsigned ordinal, unsigned count,
                 unsigned byte_sectors = UINT_MAX, bool constant_pair = false) {
  auto *parent = make_request(gpu, ctx, addr, access_size, packet_size, mask,
                              source, ordinal, ordinal % 2 == 0, byte_sectors);
  check_constructors(gpu, parent);
  audit_l2_traffic(parent, "departure", 100, false);
  gpu->partition(parent->get_sub_partition_id())->push(parent, 140);
  const auto sectors = children(gpu, parent);
  require(sectors.size() == count, "wrong number of split children");
  for (unsigned i = 0; i < sectors.size(); ++i) {
    auto *child = sectors[i];
    check_metadata(child, parent);
    const unsigned start_sector = constant_pair && addr % 128 != 0 ? 2 : 0;
    require(child->get_addr() == (constant_pair ? addr : addr + 32 * i),
            "wrong split address");
    require(child->get_access_sector_mask().to_ulong() ==
                (1U << (i + start_sector)),
            "wrong child sector mask");
    require(child->get_data_size() == 32, "child must transport one sector");
    require((child == parent) == (count == 1), "wrong split transport identity");
    auto &entry = gpu->partition(parent->get_sub_partition_id())
                      ->get_lrc()->get_entry(child->get_addr(),
                          constant_pair ? sectors.front()->get_request_uid()
                                        : child->get_request_uid());
    finalize_lrc_trace(entry.trace);
    finalize_lrc_trace(entry.trace);
    require(entry.trace.emitted, "trace not finalized");
    auto *representative = entry.front().first;
    gpu->partition(parent->get_sub_partition_id())->complete_l2_service(
        representative, 200 + i);
    // Repeated completion/disposal hooks must not duplicate transport rows.
    log_l2_to_icnt_timing(child, representative, gpu);
    require(child->l2_timing_logged(), "split child has no completion");
  }
  std::cout << "RUNTIME_CASE " << name << " source_uid=" << parent->get_request_uid()
            << " children=" << sectors.size() << "\n";
}

unsigned reply_budget;
std::vector<mem_fetch *> transferred;
bool fixture_has_buffer(unsigned, unsigned) { return reply_budget != 0; }
void fixture_push(unsigned, unsigned, void *data, unsigned) {
  require(reply_budget != 0, "interconnect fixture exhausted");
  --reply_budget;
  transferred.push_back(static_cast<mem_fetch *>(data));
}

void mixed_case(fixture_gpu *gpu, gpgpu_context *ctx, bool untracked_base) {
  std::vector<mem_fetch *> requests;
  const l2_traffic_source types[] = {
      L2_SOURCE_L1_READ, L2_SOURCE_L1_BYPASS_LOAD, L2_SOURCE_TMA_LOAD};
  const auto addr = untracked_base ? 0xb000 : 0x9000;
  if (untracked_base) {
    auto *mf = make_request(gpu, ctx, addr, 32, 32, 1,
                            L2_SOURCE_UNKNOWN, 20, false);
    mf->set_access_type(CONST_ACC_R);
    gpu->partition(mf->get_sub_partition_id())->push(mf, 149);
    requests.push_back(mf);
  }
  for (unsigned i = 0; i < 3; ++i) {
    auto *mf = make_request(gpu, ctx, addr, 32, 32, 1, types[i], 10 + i, true,
                            UINT_MAX, 100 + i);
    audit_l2_traffic(mf, "departure", 100 + i, false);
    gpu->partition(mf->get_sub_partition_id())->push(mf, 150 + i);
    requests.push_back(mf);
  }
  auto *base = requests.front();
  auto *partition = gpu->partition(base->get_sub_partition_id());
  auto &entry = partition->get_lrc()->get_entry(base->get_addr(), base->get_request_uid());
  require(entry.size() == requests.size(), "mixed request sources did not merge");
  require(entry.trace.contributors.size() == requests.size(), "missing mixed trace member");
  partition->complete_l2_service(base, 250);
  for (auto *mf : requests) {
    require(mf->l2_timing_logged() == mf->has_l2_source(),
            "tracked contributor completeness must not depend on representative");
    log_l2_to_icnt_timing(mf, base, gpu);
  }
  const auto saved_has_buffer = icnt_has_buffer;
  const auto saved_push = icnt_push;
  icnt_has_buffer = fixture_has_buffer;
  icnt_push = fixture_push;
  const unsigned subpartition = base->get_sub_partition_id();
  const unsigned before = partition->get_lrc()->size();
  // Allow one reply on each invocation and retire it before retrying. The
  // representative must stay alive until the final retry releases the entry.
  for (unsigned i = 0; i < requests.size(); ++i) {
    reply_budget = 1;
    transferred.clear();
    unsigned sent = 0;
    gpu->handle_lrc_reply(subpartition, base, sent);
    require(sent == 1 && transferred.size() == 1, "wrong retry fanout");
    require((transferred.front() == base) == (i + 1 == requests.size()),
            "representative transferred before pending contributors");
    delete transferred.front();
  }
  require(partition->get_lrc()->size() + 1 == before, "reply group not retired");
  icnt_has_buffer = saved_has_buffer;
  icnt_push = saved_push;
}

void write_cases(fixture_gpu *gpu, gpgpu_context *ctx) {
  const l2_traffic_source types[] = {
      L2_SOURCE_L1_WRITE, L2_SOURCE_L1_BYPASS_STORE, L2_SOURCE_TMA_STORE};
  for (unsigned i = 0; i < 3; ++i) {
    auto *mf = make_request(gpu, ctx, 0xc000 + 0x1000 * i, 32, 32, 1,
                            types[i], 30 + i, true);
    audit_l2_traffic(mf, "departure", 100, false);
    gpu->partition(mf->get_sub_partition_id())->push(mf, 160);
    gpu->partition(mf->get_sub_partition_id())->complete_l2_service(mf, 300);
    require(mf->l2_timing_logged(), "store has no timing");
    const auto &access = mf->get_mem_access();
    mem_fetch auxiliary(access, static_cast<const warp_inst_t *>(nullptr),
                         0, READ_PACKET_SIZE, 0, 0, 0,
                         gpu->getMemoryConfig(), 120, nullptr, mf);
    auxiliary.set_access_type(L2_WR_ALLOC_R);
    auxiliary.set_l2_fill_complete_cycle(300);
    log_l2_to_icnt_timing(&auxiliary, &auxiliary, gpu);
    require(!auxiliary.l2_timing_logged(), "write allocation gained extra timing");
    mem_fetch forwarded(access, static_cast<const warp_inst_t *>(nullptr),
                         0, READ_PACKET_SIZE, 0, 0, 0,
                         gpu->getMemoryConfig(), 120, mf);
    require(!forwarded.l2_timing_logged(), "child inherited emission state");
    forwarded.set_write_interchip(1);
    forwarded.set_l2_fill_complete_cycle(300);
    log_l2_to_icnt_timing(&forwarded, &forwarded, gpu);
    require(!forwarded.l2_timing_logged(), "auxiliary forward gained timing");
  }
  auto *wb = make_request(gpu, ctx, 0xf000, 32, 32, 1,
                          L2_SOURCE_L1_WRITE, 40, false);
  wb->set_access_type(L1_WRBK_ACC);
  audit_l2_traffic(wb, "departure", 100, false);
  auto *partition = gpu->partition(wb->get_sub_partition_id());
  partition->push(wb, 160);
  const auto previous_cycle = gpu->gpu_sim_cycle;
  gpu->gpu_sim_cycle = 400;
  partition->set_done(wb);
  partition->set_done(wb);
  require(wb->l2_timing_logged(), "writeback retirement has no timing");
  gpu->gpu_sim_cycle = previous_cycle;
}

void multicast_and_retired_parent(fixture_gpu *gpu, gpgpu_context *ctx) {
  const unsigned long long start = (1ULL << 32) + 100;
  auto *parent = make_request(gpu, ctx, 0x11000, 32, 32, 1,
                              L2_SOURCE_TMA_LOAD, 50, true, UINT_MAX, start, true);
  audit_l2_traffic(parent, "departure", start, false);
  auto *child = new mem_fetch(parent->get_mem_access(),
      static_cast<const warp_inst_t *>(nullptr), 0, READ_PACKET_SIZE, 0, 0, 0,
      gpu->getMemoryConfig(), start, parent);
  // Simulate the parent lifetime ending before its transport service.
  delete parent;
  auto *partition = gpu->partition(child->get_sub_partition_id());
  partition->push(child, start + 20);
  partition->complete_l2_service(child, start + 80);
  require(child->l2_timing_logged(), "multicast transport has no timing");
  auto &entry = partition->get_lrc()->get_entry(child->get_addr(), child->get_request_uid());
  finalize_lrc_trace(entry.trace);
  for (unsigned destination = 0; destination < 4; ++destination)
    log_l2_to_icnt_timing(child, child, gpu);
}

}  // namespace

int main(int argc, const char **argv) {
  try {
    require(std::getenv("L2_TRACE_DIR") && std::getenv("L2_TRAFFIC_AUDIT_FILE"),
            "test needs L2_TRACE_DIR and L2_TRAFFIC_AUDIT_FILE");
    auto *ctx = new gpgpu_context();
    auto *config = new gpgpu_sim_config(ctx);
    ctx->the_gpgpusim->g_the_gpu_config = config;
    option_parser_t options = option_parser_create();
    ctx->ptx_reg_options(options);
    ctx->func_sim->ptx_opcocde_latency_options(options);
    icnt_reg_options(options);
    config->reg_options(options);
    option_parser_cmdline(options, argc, argv);
    std::setlocale(LC_NUMERIC, "C");
    config->init();
    auto *gpu = new fixture_gpu(*config, ctx);
    ctx->the_gpgpusim->g_the_gpu = gpu;
    gpu->init();
    require(gpu->getMemoryConfig()->lrc_enabled, "runtime test requires LRC");
    split_case(gpu, ctx, "singleton", 0x1000, 32, 32, 1,
                L2_SOURCE_L1_READ, 1, 1);
    split_case(gpu, ctx, "bypass64", 0x2000, 64, 64, 3,
                L2_SOURCE_L1_BYPASS_LOAD, 2, 2);
    split_case(gpu, ctx, "tma128", 0x3000, 128, 128, 15,
                L2_SOURCE_TMA_LOAD, 3, 4);
    split_case(gpu, ctx, "l1_line_fill128", 0x4000, 32, 128, 1,
                L2_SOURCE_L1_READ, 4, 4);
    // The legacy special64B branch is used by constant-cache packets. Keep
    // its same-address children unchanged while checking both line halves.
    // The source tag here exercises provenance; it does not model a shader.
    split_case(gpu, ctx, "special64_low", 0x5000, 64, 64, 15,
                L2_SOURCE_L1_READ, 5, 2, 3, true);
    split_case(gpu, ctx, "special64_high", 0x6040, 64, 64, 0,
                L2_SOURCE_L1_READ, 6, 2, 12, true);
    mixed_case(gpu, ctx, false);
    mixed_case(gpu, ctx, true);
    write_cases(gpu, ctx);
    multicast_and_retired_parent(gpu, ctx);
    TraceFileManager::instance().close_all();
    std::cout << "RUNTIME_L2_PROVENANCE_PASS\n";
    // Requests intentionally remain in the fixture's ROP queues: this tests
    // splitting, timing hooks and reply backpressure. End-to-end cache/DRAM
    // completion and final draining are checked by synthetic trace tests.
    return 0;
  } catch (const std::exception &exc) {
    std::cerr << "Runtime L2 test failed: " << exc.what() << "\n";
    return 1;
  }
}
