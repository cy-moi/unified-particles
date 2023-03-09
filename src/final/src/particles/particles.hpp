#pragma once

#include "cgp/cgp.hpp"
#include <memory>

// Include Eigen
#define EIGEN_NO_DEBUG
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO
#include "../third_party/eigen/Eigen/Eigen"
#include "../third_party/eigen/Eigen/SVD"
#include "../third_party/eigen/Eigen/Sparse"

// global parameters
static float epsilon = 1e-4;
extern float particle_dim;
extern bool gdebug_stop;
extern bool enable_friction;
extern float sfric;  // static friction factor
extern float kfric; // knetic friction factor
// particle

struct particle_bubble {
  cgp::vec3 x = {0, 0, 0};  // position
  cgp::vec3 xg = {0, 0, 0}; // position guess
  cgp::vec3 v = {0, 0, 0};  // speed
  cgp::vec3 fext = {0, 0, 0};

  float invmass = 1.0;  // inverse mass = 1.0/mass
  float tinvmass = 1.0; // temporary inverse mass

  // distance to center of obj in the rest configuration
  // used by RigidShapeMatchingConstraint
  cgp::vec3 r;

  bool is_fixed = false;
  cgp::vec3 fix_x = {0, 0, 0};

  // used to organize particles into group
  int pid = -1;
};

extern std::vector<std::shared_ptr<particle_bubble>> all_particles;

// sdf
struct SDF {
  cgp::vec3 gradiant = {0, 0, 0};
  float distance = 0.0;
};

extern std::vector<std::shared_ptr<SDF>> sdf_list;

// body
enum obj_type : int { RIGID = 0, CLOTH, Granular };

extern int gdobj_count;
struct dobj {
  int dobj_id = -1;
  cgp::mesh mesh;
  cgp::mesh_drawable visual;
  std::vector<std::shared_ptr<particle_bubble>> particles;
  cgp::vec3 center;
  cgp::vec3 r_center; // original center
  cgp::mat3 rotation_matrix;
  cgp::vec3 trans;

  obj_type type = obj_type::RIGID;

  cgp::numarray<cgp::vec3> original_mesh_position;
  dobj() = default;
  dobj(cgp::mesh &mesh);

  void update_rotation();
  void update_mesh();
};
extern std::vector<std::shared_ptr<dobj>> dobj_list;
extern std::vector<std::shared_ptr<dobj>> static_dobj_list;

std::shared_ptr<dobj> create_cube(cgp::vec3 center, float edge_length,
                                  float x_rad = 0.0, float y_rad = 0.0);

std::shared_ptr<dobj> create_sphere(cgp::vec3 center, float radius);
std::shared_ptr<dobj> create_sphere2(cgp::vec3 center, float radius);

std::shared_ptr<dobj>
create_cloth(cgp::vec3 center, float width, float height,
             const std::vector<std::vector<int>> &fixed_idx =
                 std::vector<std::vector<int>>());

std::shared_ptr<dobj> create_cube_stack(cgp::vec3 center, float edge_length);

// Constrains
class Constraint {
public:
  virtual void project() = 0;
  int cardinality;
  std::vector<std::shared_ptr<particle_bubble>> particles;
  Eigen::MatrixXf coefficients;
};

// DistanceConstraint
class DistanceConstraint : public Constraint {
public:
  void project() override;
  float distance;
};

std::shared_ptr<Constraint>
buildDistanceConstraint(std::shared_ptr<particle_bubble> &A,
                        std::shared_ptr<particle_bubble> &B, float distance);

// CollisionConstrains
class CollisionConstraint : public Constraint {
public:
  cgp::vec3 normal;
};

// StaticCollisionConstraint
class StaticCollisionConstraint : public CollisionConstraint {
public:
  void project() override;
  cgp::vec3 position;
};
std::shared_ptr<Constraint>
buildStaticCollisionConstraint(std::shared_ptr<particle_bubble> &p,
                               cgp::vec3 normal, cgp::vec3 position);

// RigidShapeMatchingConstraint
class RigidShapeMatchingConstraint : public Constraint {
public:
  void project() override;
  dobj *obj;
};
std::shared_ptr<Constraint> buildRigidShapeMatchingConstraint(dobj *obj_);

// RigidCollisionConstraint SDF collision
class RigidCollisionConstraint : public Constraint {
public:
  void project() override;
  cgp::vec3 n = {0, 0, 0};
  float d = 0.0;
  int p1, p2;
};
std::shared_ptr<Constraint> buildRigidCollisionConstraint(int p1, int p2);

// global parameters
extern std::vector<std::shared_ptr<Constraint>> g_constraints;
extern std::vector<std::shared_ptr<Constraint>> g_collision_constraints;

//
void init_constraints(dobj &obj);

void generate_collision_constraints();

void sim(float delta_time = 0.01);