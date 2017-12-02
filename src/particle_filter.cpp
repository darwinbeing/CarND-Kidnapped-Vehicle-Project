/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include "particle_filter.h"
#include <math.h>
#include <limits>
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <iterator>
#include <vector>

using std::string;
using std::vector;
using std::stringstream;
using std::ostream_iterator;
using std::default_random_engine;
using std::normal_distribution;
using std::discrete_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  default_random_engine gen;
  Particle particle;

  // Creates a normal (Gaussian) distribution for x.
  normal_distribution<double> dist_x(x, std[0]);

  // Create normal distributions for y and theta.
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  num_particles = kNumParticles;
  for (int i = 0; i < num_particles; ++i) {
    particle.id = -1;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;
    weights.push_back(1);
    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;

  for (int i = 0; i < num_particles; ++i) {
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    double x_p;
    double y_p;
    double theta_p;

    // avoid division by zero
    if (fabs(yaw_rate) > 0.001) {
      x_p = x + (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
      y_p = y + (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
      theta_p = theta + yaw_rate * delta_t;
    } else {
      x_p = x + velocity * cos(theta) * delta_t;
      y_p = y + velocity * sin(theta) * delta_t;
      theta_p = theta + yaw_rate * delta_t;
    }

    // Creates a normal (Gaussian) distribution for x.
    normal_distribution<double> dist_x(x_p, std_pos[0]);

    // Create normal distributions for y and theta.
    normal_distribution<double> dist_y(y_p, std_pos[1]);
    normal_distribution<double> dist_theta(theta_p, std_pos[2]);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  for (int i = 0; i < observations.size(); i++) {
    int minLandmarkIndex = 0;

    double minDistance = std::numeric_limits<double>::max();
    for (int j = 0; j < predicted.size(); j++) {
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (distance < minDistance) {
        minDistance = distance;
        minLandmarkIndex = j;
      }
    }
    observations[i].id = minLandmarkIndex;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  double gauss_norm = (1.0/(2 * M_PI * sig_x * sig_y));

  for (int i = 0; i < particles.size(); i++) {
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    vector<LandmarkObs> observations_map = vector<LandmarkObs>(observations.size());

    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs& observation_map = observations_map[j];

      double x_map = x + (cos(theta) * observations[j].x) - (sin(theta) * observations[j].y);
      double y_map = y + (cos(theta) * observations[j].y) + (sin(theta) * observations[j].x);

      observation_map.id = j;
      observation_map.x = x_map;
      observation_map.y = y_map;
    }

    vector<LandmarkObs> predictions;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      double distance = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);

      int id = map_landmarks.landmark_list[j].id_i;
      float x_f = map_landmarks.landmark_list[j].x_f;
      float y_f = map_landmarks.landmark_list[j].y_f;

      LandmarkObs landmark = {id, x_f, y_f};

      if (distance <= sensor_range) {
        predictions.push_back(landmark);
      }
    }

    dataAssociation(predictions, observations_map);

    double weight = 1.0;
    for (int j = 0; j < observations_map.size(); j++) {
      LandmarkObs& obs_map = observations_map.at(j);
      LandmarkObs& pred = predictions.at(obs_map.id);

      // double distance = dist(pred.x, pred.y, x, y);
      // if (distance > sensor_range)
      //   continue;

      double mu_x = pred.x;
      double mu_y = pred.y;
      double exponent = pow(obs_map.x - mu_x, 2)/(2 * pow(sig_x, 2)) + pow(obs_map.y - mu_y, 2)/(2 * pow(sig_y, 2));

      weight *= gauss_norm * exp(-exponent);
    }

    particles[i].weight = weight;
    weights[i] = weight;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  discrete_distribution<int> dd(weights.begin(), weights.end());

  vector<Particle> particles_resample;

  for (int i = 0; i < num_particles; i++) {
    particles_resample.push_back(particles.at(dd(gen)));
  }

  particles = particles_resample;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y) {
  // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
