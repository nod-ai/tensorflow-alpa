#include "tensorflow/compiler/xla/service/gpu/swap_thunk.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/errors.h"

namespace xla {
namespace gpu {

SwapThunk::SwapThunk(Kind kind, ThunkInfo thunk_info)
    : Thunk(kind, thunk_info) {}

se::Event* SwapThunk::DoneEvent(int device_ordinal) const {
  return done_events_.at(device_ordinal).get();
}

SwapOutThunk::SwapOutThunk(ThunkInfo thunk_info,
                           std::vector<BufferAllocation::Slice> operands,
                           std::vector<int64_t> byte_sizes)
    : SwapThunk(Thunk::kSwapOut, thunk_info),
      operands_(std::move(operands)),
      byte_sizes_(std::move(byte_sizes)) {}

Status SwapOutThunk::Initialize(const GpuExecutable& executable,
                                se::StreamExecutor* executor) {
  return Status::OK();
}

SwapOutThunk::~SwapOutThunk() {
  // deallocate memory for this thunk
  if (!address_list_.empty()) {
    for (auto iter : address_list_) {
      executor_->HostMemoryDeallocate(iter);
    }
  }
}

Status SwapOutThunk::ExecuteOnStream(const ExecuteParams& params) {
  // gpu_stream is CUstream or e.g. the equivalent type in ROCm.
  TF_ASSIGN_OR_RETURN(const GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());
  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID logical_id,
                      params.device_assn->LogicalIdForDevice(global_device_id));
  int PartitionId = logical_id.computation_id;
  int device_ordinal = params.async_comms_stream->parent()->device_ordinal();

  if (address_list_.empty()) {
    // alloc memory for the first time. todo: will this influence profile?
    executor_ = params.async_comms_stream->parent();
    for (int64_t byte_size : byte_sizes_) {
      address_list_.push_back(executor_->HostMemoryAllocate(byte_size));
    }
    // todo: GpuExecutor's HostMemoryAllocate is simply a new char[]. It does
    // not consider NUMA. Allocate it manually and then uses a
    // HostMemoryRegister instead.
  }
  params.async_comms_stream->ThenWaitFor(params.stream);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  CHECK(operands_.size() == address_list_.size());
  for (int32_t i = 0; i < operands_.size(); ++i) {
    const BufferAllocation::Slice& slice = operands_.at(i);
    if (!slice.allocation()) {
      return InternalError("custom call input missing buffer allocation");
    }
    se::DeviceMemoryBase src_data =
        params.buffer_allocations->GetDeviceAddress(slice);

    void* source_address_ = address_list_.at(i);
    params.async_comms_stream->ThenMemcpy(source_address_, src_data, byte_sizes_.at(i));
  }

  auto done_event = std::make_unique<se::Event>(params.async_comms_stream->parent());
  TF_RET_CHECK(done_event->Init());
  params.async_comms_stream->ThenRecordEvent(done_event.get());

  {
    absl::MutexLock lock(&mu_);
    done_events_.insert_or_assign(device_ordinal, std::move(done_event));
  }
#else   //  GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Unavailable(
      "Swap on GPU are not supported in this configuration. Please "
      "build with --config=cuda");
#endif  //   GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Status::OK();
}

SwapInThunk::SwapInThunk(ThunkInfo thunk_info,
                         std::vector<BufferAllocation::Slice> results,
                         std::vector<int64_t> byte_sizes,
                         SwapOutThunk* memory_ref,
                         absl::InlinedVector<const SwapThunk*, 3> waits_for)
    : SwapThunk(Thunk::kSwapIn, thunk_info),
      results_(std::move(results)),
      byte_sizes_(std::move(byte_sizes)),
      memory_ref_(memory_ref),
      waits_for_(waits_for) {}

Status SwapInThunk::Initialize(const GpuExecutable& executable,
                               se::StreamExecutor* executor) {
  return Status::OK();
}

Status SwapInThunk::ExecuteOnStream(const ExecuteParams& params) {
  // gpu_stream is CUstream or e.g. the equivalent type in ROCm.

  TF_ASSIGN_OR_RETURN(const GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());
  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID logical_id,
                      params.device_assn->LogicalIdForDevice(global_device_id));
  int PartitionId = logical_id.computation_id;
  int device_ordinal = params.async_comms_stream->parent()->device_ordinal();

  params.async_comms_stream->ThenWaitFor(memory_ref_->DoneEvent(device_ordinal));
  for (const SwapThunk* thunk : waits_for_) {
    params.async_comms_stream->ThenWaitFor(thunk->DoneEvent(device_ordinal));
  }
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  CHECK(memory_ref_->AddressList().size() == results_.size())
      << memory_ref_->AddressList().size() << " v.s. " << results_.size();
  for (int32_t i = 0; i < results_.size(); ++i) {
    const BufferAllocation::Slice& slice = results_.at(i);
    if (!slice.allocation()) {
      return InternalError("custom call output missing buffer allocation");
    }
    se::DeviceMemoryBase destination_data =
        params.buffer_allocations->GetDeviceAddress(slice);

    void* source_address_ = memory_ref_->AddressList().at(i);
    params.async_comms_stream->ThenMemcpy(&destination_data, source_address_,
                              byte_sizes_.at(i));
  }

  auto done_event = std::make_unique<se::Event>(params.async_comms_stream->parent());
  TF_RET_CHECK(done_event->Init());
  params.async_comms_stream->ThenRecordEvent(done_event.get());

  {
    absl::MutexLock lock(&mu_);
    done_events_.insert_or_assign(device_ordinal, std::move(done_event));
  }
#else   //  GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Unavailable(
      "Swap on GPU are not supported in this configuration. Please "
      "build with --config=cuda");
#endif  //   GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Status::OK();
}

SwapDoneThunk::SwapDoneThunk(ThunkInfo thunk_info, const SwapThunk* start)
    : Thunk(Thunk::kSwapDone, thunk_info), start_(start) {}

Status SwapDoneThunk::ExecuteOnStream(const ExecuteParams& params) {
  int device_ordinal = params.stream->parent()->device_ordinal();

  params.stream->ThenWaitFor(start_->DoneEvent(device_ordinal));
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
