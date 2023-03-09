#include "particles.hpp"

static cgp::vec3 m_gravity({0, 0, -9.8});

void sim(float delta_time) {
  // (1)
  for (auto &p : all_particles) {
    // (2) Apply external forces
    p->v = p->v + delta_time * m_gravity;
    // Dampen velocities TODO better velocity damping
    p->v = p->v * 0.98;
    // (3) Initialise estimate positions
    p->xg = p->x + p->v * delta_time;
    // (4) mass scaling
    p->tinvmass = 1.0 / ((1.0 / p->invmass) * exp(-p->x.z));
  }
  // (5)

  // (6) (7) (8) (9) Generate collision constraints
  generate_collision_constraints();

  // (16) Project constraints iteratively
  for (int iter = 0; iter < 3; iter++) {
    // (17) (18) (19) (20) (21)
    for (auto c : g_collision_constraints) {
      c->project();
    }
    for (auto c : g_constraints) {
      c->project();
    }
    // (21)
  }
  // (22)

  for (auto &obj : dobj_list) {
    obj->update_mesh();
  }

  // (23-28)
  // update positions and velocities
  for (auto &p : all_particles) {
    if (norm(p->xg - p->x) < epsilon) {
      p->v = {0., 0., 0.};
    } else {
      p->v = (p->xg - p->x) / delta_time;
      p->x = p->xg;
    }
  }
}