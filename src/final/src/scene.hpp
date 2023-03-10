#pragma once

#include "cgp/cgp.hpp"
#include "environment.hpp"
#include "particles/particles.hpp"

// The element of the GUI that are not already stored in other structures
struct gui_parameters {
  bool display_frame = false;
  bool display_wireframe = true;
  bool display_shape = false;
  bool display_internal_particles = true;
  bool start_sim = false;
  bool display_SDF = false;
  bool step_run = true;
  float time_step = 0.01;
};

// The structure of the custom scene
struct scene_structure : cgp::scene_inputs_generic {

  // ****************************** //
  // Elements and shapes of the scene
  // ****************************** //
  camera_controller_orbit_euler camera_control;
  camera_projection_perspective camera_projection;
  window_structure window;

  mesh_drawable global_frame;        // The standard global frame
  environment_structure environment; // Standard environment controler
  input_devices
      inputs; // Storage for inputs status (mouse, keyboard, window dimension)
  gui_parameters gui; // Standard GUI element storage

  // ****************************** //
  // Elements and shapes of the scene
  // ****************************** //

  // Surface data
  cgp::mesh ground;
  cgp::mesh cube;

  dobj dground;
  std::shared_ptr<dobj> dcube;
  enum obj_type : int { RIGID = 0, CLOTH, Granular };

  std::shared_ptr<particle_bubble> control_particle = std::make_shared<particle_bubble>();

  // Least-square data
  // linear_system_structure linear_system;

  // Visual helper
  cgp::mesh_drawable vi_ground;
  cgp::mesh_drawable vi_sphere;      // to show particles
  cgp::curve_drawable curve;         // to show SDF gradient
  cgp::mesh_drawable vi_sphere_mini; // to show SDF gradient
  numarray<vec3> curve_positions;

  // ****************************** //
  // Functions
  // ****************************** //

  void initialize(); // Standard initialization to be called before the
                     // animation loop
  void
  display_frame(); // The frame display to be called within the animation loop
  void display_gui(); // The display of the GUI, also called within the
                      // animation loop

  void display_constraints();
  void display_selection_rectangle();

  void mouse_move_event();
  void mouse_click_event();
  void keyboard_event();
  void idle_frame();

  void rest_all();
  void init_cube_stick();
  void init_cloth_scene();
  void init_granular_scene();
  void init_all_scene();
};
