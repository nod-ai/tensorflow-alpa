#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_STATFUL_RNG_DEPENDENCY_ADDER_H
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_STATFUL_RNG_DEPENDENCY_ADDER_H

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// Add control dependency according to opaque in rng get/set state custom call.

class RngDependencyAdder : public HloModulePass {
 public:
  RngDependencyAdder() = default;
  ~RngDependencyAdder() override = default;

  absl::string_view name() const override { return "rng-dependency-adder"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_STATFUL_RNG_DEPENDENCY_ADDER_H