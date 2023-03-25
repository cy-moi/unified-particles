#include "scene.hpp"

using namespace cgp;

void scene_structure::initialize() {
  camera_control.initialize(inputs,
                            window); // Give access to the inputs and window
                                     // global state to the camera controler
  camera_control.set_rotation_axis_z();
  camera_control.look_at({3.0f, 2.0f, 2.0f}, {0, 0, 0}, {0, 0, 1});
  global_frame.initialize_data_on_gpu(mesh_primitive_frame());

  // ground
  float ground_size = particle_dim * 40;
  ground = mesh_primitive_quadrangle({-1 * ground_size, -1 * ground_size, 0},
                                     {1 * ground_size, -1 * ground_size, 0},
                                     {1 * ground_size, 1 * ground_size, 0},
                                     {-1 * ground_size, 1 * ground_size, 0});
  vi_ground.initialize_data_on_gpu(ground);
  dground.mesh = ground;
  dground.dobj_id = 0xffff;
  static_dobj_list.push_back(std::shared_ptr<dobj>(&dground));

  // particles
  vi_sphere.initialize_data_on_gpu(mesh_primitive_sphere(particle_dim / 2));

  // sdf gradient
  curve_positions.push_back({0, 0, 0});
  curve_positions.push_back({0, 0, 0});
  curve.initialize_data_on_gpu(curve_positions);
  vi_sphere_mini.initialize_data_on_gpu(
      mesh_primitive_sphere(particle_dim / 5));

  // default scene
  init_cube_stick();
}

void scene_structure::display_frame() {
  // Set the light to the current position of the camera
  environment.light = camera_control.camera_model.position();

  if (gui.display_frame)
    draw(global_frame, environment);

  if (gui.display_wireframe) {
    draw_wireframe(vi_ground, environment);
    // for (auto &obj : dobj_list) {
    //   draw_wireframe(obj->visual, environment);
    // }
  }

  if (gui.display_shape) {
    for (auto &obj : dobj_list) {
      draw(obj->visual, environment);
    }
    draw(vi_ground, environment);
  }

  if (gui.display_internal_particles) {
    for (auto &obj : dobj_list) {
      vi_sphere.material.color = {0, 0, 0};
      float color_step = 1.0 / obj->particles.size();
      for (auto &particle : obj->particles) {
        vi_sphere.material.color += {color_step, color_step, color_step};
        vi_sphere.model.translation = particle->x;
        draw(vi_sphere, environment);
      }
    }
  }

  if (gui.display_SDF) {
    for (size_t idx = 0; idx < all_particles.size(); idx++) {
      auto &sdf = sdf_list[idx];
      auto &p = all_particles[idx];
      curve_positions[0] = p->x;

      curve_positions[1] = p->x + dobj_list[p->pid]->rotation_matrix *
                                      sdf->gradiant * sdf->distance;
      curve.vbo_position.update(curve_positions);
      vi_sphere_mini.model.translation = p->x;
      draw(vi_sphere_mini, environment);
      draw(curve, environment);
    }
  }

  if (gui.start_sim) {
    if (gdebug_stop)
      return;
    sim(gui.time_step);
    if (gui.step_run) {
      gdebug_stop = true;
    }
  }
}

void scene_structure::display_gui() {
  ImGui::SliderFloat("Timer scale", &gui.time_step, 0.005f, 0.2f, "%0.005f");
  ImGui::SliderFloat("StaticFriction", &sfric, 0.0f, 1.0f, "%0.05f");
  ImGui::SliderFloat("KneticFriction", &kfric, 0.0f, 1.0f, "%0.05f");
  ImGui::SliderFloat("Timer scale", &gui.time_step, 0.005f, 0.2f, "%0.005f");
  ImGui::Checkbox("Frame", &gui.display_frame);
  ImGui::Checkbox("Wireframe", &gui.display_wireframe);
  ImGui::Checkbox("ShowMesh", &gui.display_shape);
  ImGui::Checkbox("ShowParticles", &gui.display_internal_particles);
  ImGui::Checkbox("Start", &gui.start_sim);
  ImGui::Checkbox("display_SDF", &gui.display_SDF);
  ImGui::Checkbox("Step run", &gui.step_run);
  ImGui::Checkbox("Enable Friction", &enable_friction);
  auto step = ImGui::Button("Step, click it or press space");
  if (step) {
    gdebug_stop = false;
  }

  auto reset = ImGui::Button("Rest");

  static int Obj = 0;
  bool change_obj = ImGui::Combo("Scene", &Obj, "CubeStick\0Cloth\0Granular\0All");
  if (change_obj || reset) {
    std::cout << "selecet scene:" << Obj << std::endl;
    switch (Obj) {
    case 0: {
      init_cube_stick();
    }; break;
    case 1: {
      init_cloth_scene();
    }; break;
    case 2: {
      init_granular_scene();
    } break;
    case 3: {
      init_all_scene();
    } break;
    default: { std::cout << "unknown scene" << std::endl; }
    }
  }
}

void scene_structure::mouse_move_event() {
  if (!inputs.keyboard.shift)
    camera_control.action_mouse_move(environment.camera_view);
  
  // move the cloth by pressing shift
	if (inputs.keyboard.shift)
	{
		// Current position of the mouse
		vec2 const& p = inputs.mouse.position.current;

		// Apply Deformation: press on shift key + left click on the mouse when a vertex is already selected
		if (inputs.mouse.click.left) {
      if(control_particle->pid == -1) {
        for(auto &obj : dobj_list) {
          if (obj->type == obj_type::CLOTH) {
            control_particle = obj->particles[obj->particles.size()/2];
            control_particle->is_fixed = true;
          }
        }
      } else {
        // Current translation in 2D window coordinates
        // vec2 const translation_screen = p - picking.screen_clicked;
        control_particle->x.y -= epsilon;
      }

		}

	}
  // else
	// 	picking.active = false; // Unselect picking when shift is released
}
void scene_structure::mouse_click_event() {
  camera_control.action_mouse_click(environment.camera_view);
  if (inputs.mouse.click.last_action == last_mouse_cursor_action::release_left && control_particle->pid != -1)
	{
		control_particle->is_fixed = false;
    control_particle = std::make_shared<particle_bubble>();

	}
}
void scene_structure::keyboard_event() {
  camera_control.action_keyboard(environment.camera_view);
  if (!inputs.keyboard.last_action.is_released("space")) {
    gdebug_stop = false;
  }
}
void scene_structure::idle_frame() {
  camera_control.idle_frame(environment.camera_view);
}

void scene_structure::rest_all() {
  all_particles.clear();
  sdf_list.clear();
  gdobj_count = 0;
  dobj_list.clear();
  // static_dobj_list.clear();
  g_collision_constraints.clear();
  g_constraints.clear();
}

void scene_structure::init_cube_stick() {
  rest_all();
  for (int i = 0; i < 10; i++) {
    create_cube({-particle_dim * 10, 0, particle_dim * 4 * (i + 1)},
                particle_dim * 3);
  }

  for (int i = 0; i < 10; i++) {
    create_cube({0, 0, particle_dim * 4 * (i + 1)}, particle_dim * 2);
  }

  for (int i = 0; i < 10; i++) {
    if (i < 5) {
      create_cube({particle_dim * 10, 0, particle_dim * 4 * (i + 1)},
                  particle_dim * 3);
    } else {
      create_cube({particle_dim * 10, 0, particle_dim * 4 * (i + 1)},
                  particle_dim * 2);
    }
  }

  for (int i = 0; i < 3; i++) {
    create_cube(
        {particle_dim * 15, particle_dim * 15, particle_dim * 8 * (i + 1)},
        particle_dim * 4, Pi / 4.0, Pi / 4.0);
  }

  for (int i = 0; i < 2; i++) {
    if (i < 1) {
      create_cube({particle_dim * 20, 0, particle_dim * 4 * (i + 1)},
                  particle_dim * 3);
    } else {
      create_cube({particle_dim * 20, 0, particle_dim * 4 * (i + 1)},
                  particle_dim * 2);
    }
  }

  create_cube({0, particle_dim * 10, particle_dim * 10}, particle_dim * 5,
              Pi / 4.0, Pi / 4.0);
  create_sphere({particle_dim * 10, -particle_dim * 10, particle_dim * 5},
                particle_dim * 4);
};

void scene_structure::init_cloth_scene() {
  rest_all();
  create_cloth({-10 * particle_dim, 0, particle_dim * 15}, particle_dim * 15,
               particle_dim * 15, {{0, 0}, {0, 14}});

  create_cube({-10 * particle_dim, 0, particle_dim * 5}, particle_dim * 5);

  create_cube({5 * particle_dim, 0.0, particle_dim * 25}, particle_dim * 3,
              Pi / 4.0, Pi / 4.0);
  create_cube({5 * particle_dim, 0.0, particle_dim * 20}, particle_dim * 2);

  int cloth_size = 15;
  create_cloth({5 * particle_dim, 0, particle_dim * 15},
               particle_dim * cloth_size, particle_dim * cloth_size,
               {{0, 0},
                {0, cloth_size - 1},
                {cloth_size - 1, 0},
                {cloth_size - 1, cloth_size - 1}});
}

void scene_structure::init_granular_scene() {
  rest_all();
  int size = 15;
  create_cube({0.0, 0.0, particle_dim * 20}, particle_dim * 4, Pi / 4.0,
              Pi / 4.0);

  create_cube_stack({0, 0, particle_dim * size / 2.0 + epsilon},
                    particle_dim * size);
}

void scene_structure::init_all_scene() {
  rest_all();
  create_cube({0.0, 0.0, particle_dim * 25}, particle_dim * 4, Pi / 4.0,
            Pi / 4.0);

  int cloth_size = 15;
  create_cloth({5 * particle_dim, 0, particle_dim * 10},
               particle_dim * cloth_size, particle_dim * cloth_size,
               {{0, 0},
                {0, cloth_size - 1},
                {cloth_size - 1, 0},
                {cloth_size - 1, cloth_size - 1}});
  create_cube_stack({0, 0, particle_dim * 20 + epsilon},
                    particle_dim * 5);
}