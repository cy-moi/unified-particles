#include "particles.hpp"

using namespace cgp;

bool gdebug_stop = false;
bool enable_friction = false;
float particle_dim = 0.2;
float sfric = 0.1;  // static friction factor
float kfric = 0.25; // knetic friction factor

std::vector<std::shared_ptr<particle_bubble>> all_particles;
std::vector<std::shared_ptr<SDF>> sdf_list;

int gdobj_count = 0;
std::vector<std::shared_ptr<dobj>> dobj_list;
std::vector<std::shared_ptr<dobj>> static_dobj_list;

// cube-only currently
dobj::dobj(cgp::mesh &m) {
  dobj_id = gdobj_count++;
  mesh = m;
}

void dobj::update_rotation() {
  /*
  where Q is a rotation matrix given by the polar-decomposition of
  the deformed shape’s covariance matrix A, calculated as:
    A = Sum^n_i(x^*_i − c ) · r^T_i (16)
    n: the number of particles of the obj
    r^T_i: particle_i - obj.center
  */

  // calculate c
  int particles_num = particles.size();
  center = {0, 0, 0};
  for (auto &p : particles) {
    center += p->xg;
  }
  center = center / particles_num;

  // get A
  Eigen::MatrixXf M, N;
  M.resize(3, particles_num);
  N.resize(3, particles_num);
  for (size_t i = 0; i < particles_num; i++) {
    auto a = particles[i]->xg - center;
    M.col(i) << a(0), a(1), a(2);

    auto b = particles[i]->r;

    N.col(i) << b(0), b(1), b(2);
  }
  Eigen::Matrix3f A = M * N.transpose();
  // slove SVD
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU |
                                               Eigen::ComputeFullV);
  const Eigen::Matrix3f &U = svd.matrixU();
  const Eigen::Matrix3f &V = svd.matrixV();
  Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
  I(2, 2) = (U * V.transpose()).determinant();
  // R
  Eigen::Matrix3f R =
      U * I * V.transpose(); // U.transpose() * I * V.transpose()

  rotation_matrix = cgp::mat3{
      R(0, 0), R(0, 1), R(0, 2),

      R(1, 0), R(1, 1), R(1, 2),

      R(2, 0), R(2, 1), R(2, 2),
  };

  // T
  trans = center - (transpose(rotation_matrix) * r_center);
}

void dobj::update_mesh() {
  for (size_t idx = 0; idx < mesh.position.size(); idx++) {
    switch (type) {
    case obj_type::CLOTH: {
      mesh.position[idx] = particles[idx]->xg;
    } break;

    case obj_type::RIGID: {
      cgp::vec3 p = original_mesh_position[idx];
      p = rotation_matrix * (p - r_center) + center;
      mesh.position[idx] = p;
    } break;

    case obj_type::Granular: {
      return;
    } break;

    default: { mesh.position[idx] = particles[idx]->xg; }
    }
  }

  visual.vbo_position.update(mesh.position);
  mesh.normal_update();
  visual.vbo_normal.update(mesh.normal);
}

std::shared_ptr<dobj> create_cube(cgp::vec3 center, float edge_length,
                                  float x_rad, float y_rad) {
  auto mesh = mesh_primitive_cube({0, 0, 0}, edge_length);
  auto dcube = std::make_shared<dobj>(mesh);

  // rotate and transfer
  auto R1 = rotation_transform::from_axis_angle({0, 0, 1}, x_rad);
  auto R2 = rotation_transform::from_axis_angle({1, 0, 0}, y_rad);
  cgp::vec3 trans = center;

  cgp::vec3 max = dcube->mesh.position[0];
  cgp::vec3 min = dcube->mesh.position[0];
  for (auto v : dcube->mesh.position) {
    max.x = max.x > v.x ? max.x : v.x;
    max.y = max.y > v.y ? max.y : v.y;
    max.z = max.z > v.z ? max.z : v.z;
    min.x = min.x > v.x ? v.x : min.x;
    min.y = min.y > v.y ? v.y : min.y;
    min.z = min.z > v.z ? v.z : min.z;
  }
  for (auto &v : dcube->mesh.position) {
    v = R2 * R1 * v + trans;
  }

  // build particles
  int e_count = static_cast<int>((max.x + epsilon - min.x) / particle_dim);

  for (int z_id = 0; z_id < e_count; z_id++) {
    float z = min.z + particle_dim * z_id + particle_dim / 2.0;
    for (int x_id = 0; x_id < e_count; x_id++) {
      float x = min.x + particle_dim * x_id + particle_dim / 2.0;
      for (int y_id = 0; y_id < e_count; y_id++) {
        // create particle for a cube
        float y = min.y + particle_dim * y_id + particle_dim / 2.0;
        auto particle = std::make_shared<particle_bubble>();
        particle->x = R2 * R1 * vec3({x, y, z}) + trans;
        particle->v = {0., 0., 0.};
        particle->pid = dcube->dobj_id;

        dcube->particles.push_back(particle);
        all_particles.push_back(particle);

        // create sdf for a particle
        auto sdf = std::make_shared<SDF>();
        if (x_id != 0 && y_id != 0 && z_id != 0 &&
            (x_id != e_count - 1 && y_id != e_count - 1 &&
             z_id != e_count - 1)) {
          // particles on the medial axis are assigned a
          // gradient direction arbitrarily.
          sdf->gradiant = {0, 0, 1};
          // todo it should be particle_dim * n - particle_dim/2, where n is the
          // distance from the center of particle to the surface of the cube
          sdf->distance = -particle_dim;
        } else {
          if (z_id == 0) {
            sdf->gradiant.z = -1;
          }
          if (z_id == e_count - 1) {
            sdf->gradiant.z = 1;
          }
          if (x_id == 0) {
            sdf->gradiant.x = -1;
          }
          if (x_id == e_count - 1) {
            sdf->gradiant.x = 1;
          }
          if (y_id == 0) {
            sdf->gradiant.y = -1;
          }
          if (y_id == e_count - 1) {
            sdf->gradiant.y = 1;
          }
          sdf->distance = -particle_dim / 2;
        }
        sdf->gradiant = R2 * R1 * normalize(sdf->gradiant);

        sdf_list.push_back(sdf);
      }
    }
  }

  // init center
  for (auto &p : dcube->particles) {
    dcube->center += p->x;
  }
  dcube->center = dcube->center / dcube->particles.size();

  dcube->r_center = center;
  std::cout << "center:" << center << " rcenter:" << dcube->r_center
            << std::endl;
  // update r, distance from particles to center in the rest configuration
  for (auto &p : dcube->particles) {
    p->r = p->x - dcube->center;
  }

  // save mesh original position
  for (cgp::vec3 p : dcube->mesh.position) {
    dcube->original_mesh_position.push_back(p);
  }

  std::cout << "max:" << max << std::endl;
  std::cout << "min:" << min << std::endl;
  std::cout << "particle num:" << dcube->particles.size() << std::endl;
  dcube->visual.initialize_data_on_gpu(dcube->mesh);

  g_constraints.push_back(buildRigidShapeMatchingConstraint(dcube.get()));
  std::cout << "Init constraints for obj_id:" << dcube->dobj_id
            << " constraints size:" << g_constraints.size() << std::endl;
  dobj_list.push_back(dcube);
  return dcube;
}

std::shared_ptr<dobj> create_sphere(cgp::vec3 center, float radius) {
  auto mesh = mesh_primitive_sphere(radius, center);
  auto obj = std::make_shared<dobj>(mesh);

  obj->center = center;
  obj->r_center = center;

  vec3 offset = {radius, radius, radius};
  vec3 max = center + offset;
  vec3 min = center - offset;

  // build particles
  int e_count = static_cast<int>((max.x + epsilon - min.x) / particle_dim);

  for (int z_id = 0; z_id < e_count; z_id++) {
    float z = min.z + particle_dim * z_id + particle_dim / 2.0;
    for (int x_id = 0; x_id < e_count; x_id++) {
      float x = min.x + particle_dim * x_id + particle_dim / 2.0;
      for (int y_id = 0; y_id < e_count; y_id++) {
        // create particle for a cube
        float y = min.y + particle_dim * y_id + particle_dim / 2.0;
        vec3 p = vec3({x, y, z});
        if (norm(p - obj->center) > radius - particle_dim / 2.0 + epsilon) {
          // not in the ball
          continue;
        }
        auto particle = std::make_shared<particle_bubble>();
        particle->x = p;
        particle->v = {0., 0., 0.};
        particle->pid = obj->dobj_id;

        obj->particles.push_back(particle);
        all_particles.push_back(particle);

        // create sdf for a particle
        auto sdf = std::make_shared<SDF>();
        sdf->gradiant = particle->x - obj->center;
        sdf->distance = -radius + norm(sdf->gradiant);
        // std::cout << "x,y,z:" << x_id << "," << y_id << "," << z_id
        //           << ", g:" << sdf->gradiant << std::endl;

        // sdf->distance = norm(sdf->gradiant * particle_dim / 2);
        // sdf->gradiant = normalize(sdf->gradiant);
        sdf->gradiant = normalize(sdf->gradiant);
        sdf_list.push_back(sdf);
      }
    }
  }
  // update r, distance from particles to center in the rest configuration
  for (auto &p : obj->particles) {
    p->r = p->x - obj->center;
  }

  // save mesh original position
  for (cgp::vec3 p : obj->mesh.position) {
    obj->original_mesh_position.push_back(p);
  }

  std::cout << "max:" << max << std::endl;
  std::cout << "min:" << min << std::endl;
  std::cout << "particle num:" << obj->particles.size() << std::endl;
  obj->visual.initialize_data_on_gpu(obj->mesh);

  g_constraints.push_back(buildRigidShapeMatchingConstraint(obj.get()));
  std::cout << "Init constraints for obj_id:" << obj->dobj_id
            << " constraints size:" << g_constraints.size() << std::endl;
  dobj_list.push_back(obj);
  return obj;
}

std::shared_ptr<dobj> create_sphere2(cgp::vec3 center, float radius) {
  auto mesh = mesh_primitive_sphere(radius, center);
  auto obj = std::make_shared<dobj>(mesh);

  obj->center = center;
  obj->r_center = center;

  // build particles
  for (vec3 p : obj->mesh.position) {
    // create particle for a cube
    auto particle = std::make_shared<particle_bubble>();
    particle->x = p;
    particle->v = {0., 0., 0.};
    particle->pid = obj->dobj_id;

    obj->particles.push_back(particle);
    all_particles.push_back(particle);

    // create sdf for a particle
    auto sdf = std::make_shared<SDF>();
    // std::cout << "x,y,z:" << x_id << "," << y_id << "," << z_id
    //           << ", g:" << sdf->gradiant << std::endl;

    // sdf->distance = norm(sdf->gradiant * particle_dim / 2);
    // sdf->gradiant = normalize(sdf->gradiant);

    sdf_list.push_back(sdf);
  }
  // update r, distance from particles to center in the rest configuration
  for (auto &p : obj->particles) {
    p->r = p->x - obj->center;
  }

  // save mesh original position
  for (cgp::vec3 p : obj->mesh.position) {
    obj->original_mesh_position.push_back(p);
  }

  std::cout << "particle num:" << obj->particles.size() << std::endl;
  obj->visual.initialize_data_on_gpu(obj->mesh);

  g_constraints.push_back(buildRigidShapeMatchingConstraint(obj.get()));
  std::cout << "Init constraints for obj_id:" << obj->dobj_id
            << " constraints size:" << g_constraints.size() << std::endl;
  dobj_list.push_back(obj);
  return obj;
}

std::shared_ptr<dobj>
create_cloth(cgp::vec3 center, float height, float width,
             const std::vector<std::vector<int>> &fixed_idx) {
  // mesh_primitive_grid(vec3 const& p00 (-1,-1), vec3 const& p10 (1,-1), vec3
  // const& p11 (1,1), vec3 (-1,1) const& p01, int Nu, int Nv)
  vec3 p00 = {center.x - height / 2.0 + particle_dim / 2.0,
              center.y - width / 2.0 + particle_dim / 2.0, center.z};
  vec3 p10 = {center.x + height / 2.0 - particle_dim / 2.0,
              center.y - width / 2.0 + particle_dim / 2.0, center.z};
  vec3 p11 = {center.x + height / 2.0 - particle_dim / 2.0,
              center.y + width / 2.0 - particle_dim / 2.0, center.z};
  vec3 p01 = {center.x - height / 2.0 + particle_dim / 2.0,
              center.y + width / 2.0 - particle_dim / 2.0, center.z};

  int Nu = (int)(height / particle_dim);
  int Nv = (int)(width / particle_dim);
  auto mesh = mesh_primitive_grid(p00, p10, p11, p01, Nu, Nv);
  auto obj = std::make_shared<dobj>(mesh);

  obj->type = obj_type::CLOTH;
  obj->center = center;
  obj->r_center = center;

  // build particles
  for (int idx = 0; idx < obj->mesh.position.size(); idx++) {
    // create particle for a cube
    auto particle = std::make_shared<particle_bubble>();
    particle->x = obj->mesh.position[idx];
    particle->v = {0., 0., 0.};
    particle->pid = obj->dobj_id;

    obj->particles.push_back(particle);
    all_particles.push_back(particle);

    // create sdf for a particle
    // cloth particle doesn't need it
    auto sdf = std::make_shared<SDF>();
    sdf_list.push_back(sdf);
  }

  // update r, distance from particles to center in the rest configuration
  // cloth doesn't need it
  for (auto &p : obj->particles) {
    p->r = p->x - obj->center;
  }

  // save mesh original position
  for (cgp::vec3 p : obj->mesh.position) {
    obj->original_mesh_position.push_back(p);
  }

  std::cout << "particle num:" << obj->particles.size() << std::endl;
  obj->visual.initialize_data_on_gpu(obj->mesh);

  // update distance constraint
  std::cout << Nu << " " << Nv << std::endl;
  for (int i = 0; i < Nu; i++) {
    for (int j = 0; j < Nv; j++) {
      auto &p1 = obj->particles[i * Nv + j];
      auto gen_constrain = [&p1, &obj, Nv, Nu](int i, int j) {
        if (i < 0 || j < 0 || i >= Nu || j >= Nv)
          return;
        auto &p2 = obj->particles[i * Nv + j];
        g_constraints.push_back(buildDistanceConstraint(p1, p2, particle_dim));
      };
      gen_constrain(i, j - 1);
      gen_constrain(i, j + 1);
      gen_constrain(i - 1, j);
      gen_constrain(i + 1, j);
    }
  }
  for (auto idx_xy : fixed_idx) {
    auto idx = idx_xy[0] * Nv + idx_xy[1];
    if (idx > obj->particles.size()) {
      std::cout << "invalid point: " << idx_xy[0] << "," << idx_xy[1]
                << std::endl;
      continue;
    }

    auto &p = obj->particles[idx];
    p->is_fixed = true;
    p->fix_x = p->x;
  }
  std::cout << "Init constraints for obj_id:" << obj->dobj_id
            << " constraints size:" << g_constraints.size() << std::endl;
  dobj_list.push_back(obj);
  return obj;
}

std::shared_ptr<dobj> create_cube_stack(cgp::vec3 center, float edge_length) {
  auto mesh = mesh_primitive_cube(center, edge_length);
  auto dcube = std::make_shared<dobj>(mesh);
  dcube->type = obj_type::Granular;

  cgp::vec3 max = dcube->mesh.position[0];
  cgp::vec3 min = dcube->mesh.position[0];
  for (auto v : dcube->mesh.position) {
    max.x = max.x > v.x ? max.x : v.x;
    max.y = max.y > v.y ? max.y : v.y;
    max.z = max.z > v.z ? max.z : v.z;
    min.x = min.x > v.x ? v.x : min.x;
    min.y = min.y > v.y ? v.y : min.y;
    min.z = min.z > v.z ? v.z : min.z;
  }

  // build particles
  int e_count = static_cast<int>((max.x + epsilon - min.x) / particle_dim);

  for (int z_id = 0; z_id < e_count; z_id++) {
    float z = min.z + particle_dim * z_id + particle_dim / 2.0;
    for (int x_id = 0; x_id < e_count; x_id++) {
      float x = min.x + particle_dim * x_id + particle_dim / 2.0;
      for (int y_id = 0; y_id < e_count; y_id++) {
        // create particle for a cube
        float y = min.y + particle_dim * y_id + particle_dim / 2.0;
        auto particle = std::make_shared<particle_bubble>();
        particle->x = vec3({x, y, z});
        particle->v = {0., 0., 0.};
        particle->pid = dcube->dobj_id;
        particle->invmass = 5;
        particle->tinvmass = 5;

        dcube->particles.push_back(particle);
        all_particles.push_back(particle);

        // create sdf for a particle, use deafult distance 0.0
        auto sdf = std::make_shared<SDF>();
        sdf_list.push_back(sdf);
      }
    }
  }

  // save mesh original position. This object don't have mesh
  std::cout << "particle num:" << dcube->particles.size() << std::endl;
  // it not necessary to show it
  // dcube->visual.initialize_data_on_gpu(dcube->mesh);

  std::cout << "Init constraints for obj_id:" << dcube->dobj_id
            << " constraints size:" << g_constraints.size() << std::endl;
  dobj_list.push_back(dcube);
  return dcube;
}

// Constraints
std::vector<std::shared_ptr<Constraint>> g_constraints;
std::vector<std::shared_ptr<Constraint>> g_collision_constraints;

void generate_collision_constraints() {
  g_collision_constraints.clear();
  for (size_t i = 0; i < all_particles.size(); i++) {
    auto &pi = all_particles[i];
    // todo a better way to get collision
    float d = pi->xg.z;
    if (d < particle_dim / 2.0 + epsilon) {
      g_collision_constraints.push_back(buildStaticCollisionConstraint(
          pi, {0, 0, 1}, {pi->xg.x, pi->xg.y, particle_dim / 2.0 + epsilon}));
    }
    //
    for (size_t j = i + 1; j < all_particles.size(); j++) {
      auto &pj = all_particles[j];
      if (pi->pid == pj->pid && dobj_list[pi->pid]->type == obj_type::RIGID &&
          dobj_list[pj->pid]->type == obj_type::RIGID) {
        continue; // skip particle in same  rigid obj
      }

      float dist = norm(pi->xg - pj->xg);
      if (dist < particle_dim - epsilon) {
        // ridig collision
        g_collision_constraints.push_back(buildRigidCollisionConstraint(i, j));
      }
    }
  }
}

// DistanceConstraint
std::shared_ptr<Constraint>
buildDistanceConstraint(std::shared_ptr<particle_bubble> &A,
                        std::shared_ptr<particle_bubble> &B, float distance) {
  auto constraint = std::make_shared<DistanceConstraint>();
  constraint->cardinality = 2;
  constraint->particles.push_back(A);
  constraint->particles.push_back(B);
  constraint->distance = distance;
  return constraint;
}

void DistanceConstraint::project() {
  auto &p1 = particles[0];
  auto &p2 = particles[1];
  cgp::vec3 p12 = p1->xg - p2->xg;

  float d = norm(p12);
  // if (d > distance && distance != 0.0f) {
  //   // tear
  //   return;
  // }

  cgp::vec3 delta =
      (d - distance) / (p1->invmass + p2->invmass) * (p12 / d) * 1.0;

  if (p1->is_fixed) {
    // if this particle is a fixed one, we don't change it position anyway.
    p1->xg = p1->fix_x;
  } else {
    p1->xg += -delta * p1->invmass;
  }

  if (p2->is_fixed) {
    p2->xg = p2->fix_x;
  } else {
    p2->xg += delta * p2->invmass;
  }
}

// StaticCollisionConstraint
std::shared_ptr<Constraint>
buildStaticCollisionConstraint(std::shared_ptr<particle_bubble> &p, vec3 normal,
                               vec3 position) {
  auto constraint = std::make_shared<StaticCollisionConstraint>();
  constraint->cardinality = 1;
  constraint->particles.push_back(p);
  constraint->normal = normal;
  constraint->position = position;
  return constraint;
}
void StaticCollisionConstraint::project() {
  auto &p = particles[0];
  vec3 pointToPosition = (p->xg - position);

  if (dot(pointToPosition, normal) >= 0.0f && norm(pointToPosition) >= 0.0f)
    return;

  float a = dot(pointToPosition, normal);
  vec3 b = pointToPosition / (norm(pointToPosition));
  vec3 displacement = a * b;
  if (norm(displacement) < epsilon)
    return;
  p->xg += displacement;

  // friction
  if (enable_friction) {
    vec3 dp = p->xg - p->x;
    vec3 dpt = dp - dot(dp, normal) * normal;
    float ldpt = norm(dpt);
    if (ldpt < epsilon)
      return;

    if (ldpt < sfric * (particle_dim / 2.0)) {
      p->xg -= dpt;
    } else {
      p->xg -= dpt * std::min(kfric * (particle_dim / 2.0) / ldpt, 1.0);
    }
  }
}

// RigidShapeMatchingConstraint
std::shared_ptr<Constraint> buildRigidShapeMatchingConstraint(dobj *obj_) {
  auto constraint = std::make_shared<RigidShapeMatchingConstraint>();
  constraint->obj = obj_;
  constraint->cardinality = obj_->particles.size();
  constraint->particles = obj_->particles;
  return constraint;
}

void RigidShapeMatchingConstraint::project() {
  // paper page 5 section 5
  //∆xi = (Qri + c) − x∗  Eq.(15)
  // get Q

  obj->update_rotation();
  for (auto &p : particles) {
    auto detal_x = (obj->rotation_matrix * p->r + obj->center) - p->xg;
    p->xg += detal_x * 1.0; // in rigid stiffness shoule be 1
  }
}

// RigidCollisionConstraint
std::shared_ptr<Constraint> buildRigidCollisionConstraint(int p1, int p2) {
  auto constraint = std::make_shared<RigidCollisionConstraint>();
  constraint->cardinality = 2;
  constraint->p1 = p1;
  constraint->p2 = p2;
  return constraint;
}

void RigidCollisionConstraint::project() {
  auto &mp1 = all_particles[p1];
  auto &mp2 = all_particles[p2];
  auto &sdf_1 = sdf_list[p1];
  auto &sdf_2 = sdf_list[p2];

  if (norm(mp1->xg - mp2->xg) > particle_dim / 2.0 + epsilon) {
    // just handle it as normal collision
    cgp::vec3 p12 = mp2->xg - mp1->xg;
    float len = cgp::norm(p12);
    d = particle_dim - len;
    if (d < epsilon)
      return;
    n = p12 / len;
  } else {
    // interlock occur, tunneling case mentioned in paper 5.1
    // for inner particles

    if (sdf_1->distance < sdf_2->distance) {
      d = sdf_1->distance;
      n = dobj_list[mp1->pid]->rotation_matrix * sdf_1->gradiant;
    } else {
      d = sdf_2->distance;
      n = -(dobj_list[mp2->pid]->rotation_matrix * sdf_2->gradiant);
    }
    // for surface particles
    // magnitude < r
    if (d < particle_dim / 2.0 + epsilon) {
      auto p12 = mp1->xg - mp2->xg;
      auto len_p12 = norm(p12);
      d = particle_dim - len_p12;
      if (d < epsilon)
        return;
      p12 = p12 / len_p12;
      auto d_p12_n = cgp::dot(p12, n);
      if (d_p12_n < 0) {
        n = p12 - 2 * d_p12_n * n;
      } else {
        n = p12;
      }
      // debug log start
      // std::cout << "surface interlock occured" << std::endl;
      // debug log end
    } else {
      // debug log start
      // std::cout << "internal interlock occured" << std::endl;
      // debug log end
    }
  }

  vec3 delta_1 = -mp1->tinvmass / (mp1->tinvmass + mp2->tinvmass) * d * n;
  vec3 delta_2 = mp2->tinvmass / (mp1->tinvmass + mp2->tinvmass) * d * n;

  mp1->xg += delta_1 / 2.0;
  mp2->xg += delta_2 / 2.0;

  // friction
  if (enable_friction) {
    vec3 nf = normalize(n);
    vec3 delta_pp_x_xg = (mp1->xg - mp1->x) - (mp2->xg - mp2->x);
    vec3 delta_t = delta_pp_x_xg - dot(delta_pp_x_xg, nf) * nf;
    float ldpt = norm(delta_t);
    if (ldpt < epsilon)
      return;

    auto wsum = mp1->tinvmass + mp2->tinvmass;

    if (ldpt < sfric * d) {
      mp1->xg -= delta_t * mp1->tinvmass / wsum;
      mp2->xg += delta_t * mp2->tinvmass / wsum;
    } else {
      vec3 delta = delta_t * std::min<float>(kfric * d / ldpt, 1.0);
      mp1->xg -= delta * mp1->tinvmass / wsum;
      mp2->xg += delta * mp2->tinvmass / wsum;
    }
  }
}