#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_NOD_AUTO_SHARDING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_NOD_AUTO_SHARDING_H_

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace spmd {

class NodAutoSharding : public HloModulePass {
 public:
  NodAutoSharding() = default;
  ~NodAutoSharding() override = default;
  absl::string_view name() const override { return "nod_auto_sharding"; }
  StatusOr<bool> Run(HloModule* module) override { return false; }
};

}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_AUTO_SHARDING_H_
