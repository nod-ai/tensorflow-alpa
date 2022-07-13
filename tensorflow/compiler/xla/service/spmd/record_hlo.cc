#include "tensorflow/compiler/xla/service/spmd/record_hlo.h"

#include <fstream>

#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/xla/service/pass_context.h"
#include "tensorflow/compiler/xla/service/spmd/mesh.pb.h"

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

  nod::Mesh mesh;
  for (int64_t id : device_mesh.dimensions()) {
    mesh.add_mesh_ids(id);
  }
  for (double a : mesh_alpha) {
    mesh.add_alpha(a);
  }
  for (double b : mesh_beta) {
    mesh.add_beta(b);
  }
  std::ofstream mesh_file(device_mesh_filepath_);
  if(!mesh_file) {
    return Unknown("Failed to open file");
  }
  mesh.SerializeToOstream(&mesh_file);
  mesh_file.close();

  return true;
}

}  // namespace spmd
}  // namespace xla
