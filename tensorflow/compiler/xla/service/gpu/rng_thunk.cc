#include "tensorflow/compiler/xla/service/gpu/rng_thunk.h"

namespace xla {
namespace gpu {

static int64_t rng_states[64];

Status RngGetAndUpdateStateThunk::Initialize(const GpuExecutable& executable,
                                             se::StreamExecutor* executor) {
  int device_ordinal = executor->device_ordinal();
  rng_states[device_ordinal] = 0;

  return Status::OK();
}

Status RngGetAndUpdateStateThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto dest_addr = params.buffer_allocations->GetDeviceAddress(dest_);

  TF_ASSIGN_OR_RETURN(const GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());
  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID logical_id,
                      params.device_assn->LogicalIdForDevice(global_device_id));

  int device_ordinal = params.stream->parent()->device_ordinal();
  rng_states[device_ordinal] += delta_;
  int64_t seed = rng_states[device_ordinal] +
                 (params.rng_seed ^ logical_id.computation_id);

  // Note: Here we only use the lowest 32 bits of seed for performance concern.
  // Ideally, the seed should be a 128-bit integer.
  params.stream->ThenMemset32(&dest_addr, static_cast<int32_t>(seed),
                              dest_addr.size());
  return Status::OK();
}

Status RngGetStateThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto dest_addr = params.buffer_allocations->GetDeviceAddress(dest_);

  TF_ASSIGN_OR_RETURN(const GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());
  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID logical_id,
                      params.device_assn->LogicalIdForDevice(global_device_id));

  int device_ordinal = params.stream->parent()->device_ordinal();

  // Note: Here we only use the lowest 32 bits of seed for performance concern.
  // Ideally, the seed should be a 128-bit integer.
  void *cpu_state = static_cast<void *>(rng_states + device_ordinal);
  std::cerr << "getting state " << rng_states[device_ordinal] << " with size " << dest_addr.size();
  params.stream->ThenMemcpy(&dest_addr, cpu_state, dest_addr.size());
  return Status::OK();
}

Status RngSetStateThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto src_addr = params.buffer_allocations->GetDeviceAddress(src_);

  TF_ASSIGN_OR_RETURN(const GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());
  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID logical_id,
                      params.device_assn->LogicalIdForDevice(global_device_id));

  int device_ordinal = params.stream->parent()->device_ordinal();

  // Note: Here we only use the lowest 32 bits of seed for performance concern.
  // Ideally, the seed should be a 128-bit integer.
  void *cpu_state = static_cast<void *>(rng_states + device_ordinal);
  params.stream->ThenMemcpy(cpu_state, src_addr, src_addr.size());
  // auto done_event = std::make_unique<se::Event>(params.stream->parent());
  // TF_RET_CHECK(done_event->Init());
  // params.stream->ThenRecordEvent(done_event.get());
  params.stream->BlockHostUntilDone();
  std::cerr << "setting state " << rng_states[device_ordinal] << " with size " << src_addr.size();
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
