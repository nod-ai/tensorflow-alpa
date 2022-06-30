#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RECORD_HLO_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RECORD_HLO_H_

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace spmd {

class RecordHLO : public HloModulePass {
 public:
  RecordHLO(std::string hlo_module_filepath, std::string device_mesh_filepath):
	  hlo_module_filepath_(hlo_module_filepath), device_mesh_filepath_(device_mesh_filepath) {};
  ~RecordHLO() override = default;
  absl::string_view name() const override { return "record_hlo"; }

  StatusOr<bool> Run(HloModule* module) override;
 private:
  std::string hlo_module_filepath_;
  std::string device_mesh_filepath_;
};

}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RECORD_HLO_H_
