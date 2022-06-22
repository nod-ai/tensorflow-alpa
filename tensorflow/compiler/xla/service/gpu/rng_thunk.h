#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RNG_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RNG_THUNK_H_

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"

namespace xla {
namespace gpu {

class RngGetStateThunk : public Thunk {
 public:
  Status ExecuteOnStream(const ExecuteParams& params) override;

  RngGetStateThunk(ThunkInfo thunk_info, const BufferAllocation::Slice& dest)
      : Thunk(Kind::kRngGetAndUpdateState, thunk_info), dest_(dest) {}

 private:
  const BufferAllocation::Slice dest_;
};

class RngSetStateThunk : public Thunk {
 public:
  Status ExecuteOnStream(const ExecuteParams& params) override;

  RngSetStateThunk(ThunkInfo thunk_info, const BufferAllocation::Slice& src)
      : Thunk(Kind::kRngGetAndUpdateState, thunk_info), src_(src) {}

 private:
  const BufferAllocation::Slice src_;
};

class RngGetAndUpdateStateThunk : public Thunk {
 public:
  Status ExecuteOnStream(const ExecuteParams& params) override;

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;

  RngGetAndUpdateStateThunk(ThunkInfo thunk_info,
                            const BufferAllocation::Slice& dest,
                            int64_t delta)
      : Thunk(Kind::kRngGetAndUpdateState, thunk_info), dest_(dest), delta_(delta) {}

 private:
  const BufferAllocation::Slice dest_;
  int64_t delta_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RNG_THUNK_H_
