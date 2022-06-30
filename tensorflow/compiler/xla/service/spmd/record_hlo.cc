#include "tensorflow/compiler/xla/service/spmd/record_hlo.h"

#include <fstream>

#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/xla/service/pass_context.h"

namespace xla {
namespace spmd {



StatusOr<bool> RecordHLO::Run(HloModule* module) {
  Array<int64_t> device_mesh(
      pass_context::GetIntVector("auto_sharding::device_mesh_shape"));
  device_mesh.SetValues(
      pass_context::GetIntVector("auto_sharding::device_mesh_ids"));
  std::vector<double> mesh_alpha =
	  pass_context::GetDoubleVector("auto_sharding::device_mesh_alpha");
  std::vector<double> mesh_beta =
	  pass_context::GetDoubleVector("auto_sharding::device_mesh_beta");
  
  std::string serialized_module;
  if (!module->ToProto().SerializeToString(&serialized_module)) {
    return Unknown("Failed to serialize the HloModuleProto.");
  }

  std::ofstream module_file(hlo_module_filepath_);
  if(!module_file) {
    return Unknown("Failed to open file");
  }
  module_file << serialized_module;
  module_file.close();

  std::string serialized_mesh;
  serialized_mesh += "Mesh Shape\n";
  for (int64_t shape : device_mesh.dimensions()) {
    serialized_mesh += std::to_string(shape) + ", ";
  }
  serialized_mesh += "\n";

  serialized_mesh += "Mesh Alpha\n";
  for (double a : mesh_alpha) {
    int64_t value = *reinterpret_cast<int64_t*>(&a);
    serialized_mesh += std::to_string(value) + ", ";
  }
  serialized_mesh += "\n";
  
  serialized_mesh += "Mesh Beta\n";
  for (double b : mesh_beta) {
    int64_t value = *reinterpret_cast<int64_t*>(&b);
    serialized_mesh += std::to_string(value) + ", ";
  }
  serialized_mesh += "\n";
    
  std::ofstream mesh_file(device_mesh_filepath_);
  if(!mesh_file) {
    return Unknown("Failed to open file");
  }
  mesh_file << serialized_mesh;
  mesh_file.close();

  return true;
}

}  // namespace spmd
}  // namespace xla
