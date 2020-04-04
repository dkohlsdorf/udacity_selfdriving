/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  // Code from Implementing Particle Filter 5.5
  num_particles = 100;
  normal_distribution<double> normal_x(0.0, std[0]);
  normal_distribution<double> normal_y(0.0, std[1]);
  normal_distribution<double> normal_theta(0.0, std[2]);
  for (int i = 0; i < num_particles; i++)
  {
    Particle p;
    p.id = i;
    p.x = x + normal_x(gen);
    p.y = y + normal_y(gen);
    p.theta = theta + normal_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  // Equations from Motion Models 3.4 Yaw rate
  normal_distribution<double> normal_x(0.0, std_pos[0]);
  normal_distribution<double> normal_y(0.0, std_pos[1]);
  normal_distribution<double> normal_theta(0.0, std_pos[2]);
  for (int i = 0; i < num_particles; i++)
  {
    double next_theta = particles[i].theta + yaw_rate * delta_t;
    if (yaw_rate == 0.0)
    {
      particles[i].x += delta_t * velocity * cos(particles[i].theta) + normal_x(gen);
      particles[i].y += delta_t * velocity * sin(particles[i].theta) + normal_y(gen);
      particles[i].theta = next_theta + normal_theta(gen);
    }
    else
    {
      particles[i].x += velocity / yaw_rate * (sin(next_theta) - sin(particles[i].theta)) + normal_x(gen);
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(next_theta)) + normal_y(gen);
      particles[i].theta = next_theta + normal_theta(gen);
    }
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  // Code from Implementing Particle Filter quizes 5.14 - 5.18
  int n = observations.size();
  int m = predicted.size();
  for (int i = 0; i < n; i++)
  {
    double mindist = numeric_limits<double>::max();
    LandmarkObs min_prediction;
    for (int j = 0; j < m; j++)
    {
      double distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
      if (distance < mindist)
      {
        mindist = distance;
        min_prediction = predicted[j];
      }
    }
    observations[i].id = min_prediction.id;
  }
}

void transform(Particle p, const vector<LandmarkObs> &observations, vector<LandmarkObs> &result)
{
  // Code from Implementing Particle Filter quizes 5.16 and http://planning.cs.uiuc.edu/node99.html
  int n = observations.size();
  double sin_theta = sin(p.theta);
  double cos_theta = cos(p.theta);
  for (int i = 0; i < n; i++)
  {
    double x = cos_theta * observations[i].x - sin_theta * observations[i].y + p.x;
    double y = sin_theta * observations[i].x + cos_theta * observations[i].y + p.y;
    LandmarkObs l;
    l.x = x;
    l.y = y;
    l.id = observations[i].id;
    result.push_back(l);
  }
}

void filter(Particle p, const Map &map_landmarks, double range, vector<LandmarkObs> &result)
{
  // Code from Implementing Particle Filter explanation 5.21
  int n = map_landmarks.landmark_list.size();
  for (int i = 0; i < n; i++)
  {
    bool in_range = dist(p.x, p.y, map_landmarks.landmark_list[i].x_f, map_landmarks.landmark_list[i].y_f);

    if (in_range)
    {
      LandmarkObs l;
      l.x = map_landmarks.landmark_list[i].x_f;
      l.y = map_landmarks.landmark_list[i].y_f;
      l.id = map_landmarks.landmark_list[i].id_i;
      result.push_back(l);
    }
  }
}

double normal_scaler_2d(double std_landmark[])
{
  double std = std_landmark[0] * std_landmark[1];
  return 1.0 / (std * sqrt(2 * M_PI));
}

double normal_2d(LandmarkObs x, LandmarkObs mu, double std_landmark[], double scaler)
{
  double log_unscaled = -1 * (pow(x.x - mu.x, 2) / (2 * pow(std_landmark[0], 2)) +
                              pow(x.y - mu.y, 2) / (2 * pow(std_landmark[1], 2)));
  double unscaled = exp(log_unscaled);
  return scaler * unscaled;
}

void compute_weight(Particle &particle, vector<LandmarkObs> predicted, vector<LandmarkObs> observations, double std_landmark[])
{
  int n = observations.size();
  double prob = 1.0;
  for (int i = 0; i < n; i++)
  {
    for(int j = 0; j < predicted.size(); ++j)
    {
      if (observations[i].id == predicted[j].id) {
        LandmarkObs o = observations[i];
        LandmarkObs p = predicted[j];
        prob *= normal_2d(o, p, std_landmark, normal_scaler_2d(std_landmark));
      }
    }
  }
  particle.weight = prob;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  double scaler = 0.0;
  for (int i = 0; i < num_particles; i++)
  {
    vector<LandmarkObs> transformed, filtered;
    transform(particles[i], observations, transformed);
    filter(particles[i], map_landmarks, sensor_range, filtered);
    dataAssociation(filtered, transformed);
    compute_weight(particles[i], filtered, transformed, std_landmark);
    scaler += particles[i].weight;
  }
  for (int i = 0; i < num_particles; i++)
  {
    particles[i].weight /= scaler;
  }
}

void ParticleFilter::resample()
{
  vector<double> weights(num_particles);
  for(int i = 0; i < num_particles; i++) {
    weights[i] = particles[i].weight;
  }
  discrete_distribution<int> sampler(weights.begin(), weights.end());
  vector<Particle> resampled(num_particles);
  for (int i = 0; i < num_particles; ++i) {
    resampled[i] = particles[sampler(gen)];
  }
  particles = resampled;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}