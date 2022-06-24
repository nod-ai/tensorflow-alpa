#include "tensorflow/compiler/xla/service/spmd/stateful_rng_dependency_adder.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {

StatusOr<bool> RngDependencyAdder::Run(HloModule* module) {
  bool changed = false;

  using InstructionVector = absl::InlinedVector<HloInstruction*, 2>;
  using absl::flat_hash_map;
  for (HloComputation* computation : module->computations()) {
    std::cerr << "processing computation:\n" << computation->ToString() << "\n";
    flat_hash_map<std::string, HloInstruction*> op_name_to_inst;
    std::vector<HloInstruction*> replace_insts;
    for (HloInstruction* ins : computation->instructions()) {
      if (ins->IsCustomCall("identity") ||
          (ins->IsCustomCall("pipeline_marker") &&
           ins->metadata().op_type() == "end")) {
        op_name_to_inst[ins->metadata().op_name()] = ins;
      } else if (ins->IsCustomCall("alpa$get-state") ||
                 ins->IsCustomCall("alpa$set-state")) {
        replace_insts.push_back(ins);
      }
    }
    for (HloInstruction* ins : replace_insts) {
      // Split operands to actual and data deps
      InstructionVector dep_insts;
      int64_t dep_var_start = ins->IsCustomCall("alpa$get-state") ? 0 : 1;
      std::vector<HloInstruction*> actual_operands;
      if (ins->IsCustomCall("alpa$set-state")) {
        actual_operands.push_back(ins->mutable_operands()[0]);
      }
      for (int64_t i = dep_var_start; i < ins->operand_count(); ++i) {
        dep_insts.push_back(ins->mutable_operands()[i]);
      }
      // Clone and remove extra operands
      HloInstruction* cloned =
          computation->AddInstruction(HloInstruction::CreateCustomCall(
              ins->shape(), actual_operands, ins->custom_call_target()));
      Cast<HloCustomCallInstruction>(cloned)->set_custom_call_has_side_effect(
          true);
      TF_RETURN_IF_ERROR(ins->ReplaceAllUsesWith(cloned));
      // Add dependency
      HloCustomCallInstruction* custom_call =
          Cast<HloCustomCallInstruction>(ins);
      std::vector<std::string> dep_names =
          absl::StrSplit(custom_call->opaque(), ";");
      for (const std::string& dep : dep_names) {
        if (dep == "") {
          continue;
        }
        cloned->AddControlDependencyTo(op_name_to_inst[dep]);
      }
      for (HloInstruction* suc : dep_insts) {
        suc->AddControlDependencyTo(cloned);
      }
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(ins));
      changed = true;
    }
  }
  return changed;
}
};  // namespace xla