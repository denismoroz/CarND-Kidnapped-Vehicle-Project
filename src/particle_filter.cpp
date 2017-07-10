/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = 100;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  default_random_engine gen;

  for (int i=0; i < num_particles; i++) {
      Particle particle;
      particle.id = i;
      particle.x = dist_x(gen);
      particle.y = dist_y(gen);
      particle.theta = dist_theta(gen);
      particle.weight = 1;
      particles.push_back(particle);
  }

  weights.resize(num_particles);
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  std::random_device gen;
  std::normal_distribution<double> dist_x   (0, std_pos[0]*delta_t);
  std::normal_distribution<double> dist_y   (0, std_pos[1]*delta_t);
  std::normal_distribution<double> dist_yaw (0, std_pos[2]*delta_t);

  for (auto&& particle: particles)
  {
      if (yaw_rate == 0)
      {
         particle.x = particle.x + velocity*delta_t*cos(particle.theta);
         particle.y = particle.y + velocity*delta_t*sin(particle.theta);
         particle.theta = particle.theta;
      }
      else
      {
          particle.x = particle.x + velocity/yaw_rate*(sin(particle.theta+yaw_rate*delta_t)-sin(particle.theta)) + dist_x(gen);
          particle.y = particle.y + velocity/yaw_rate*(cos(particle.theta)-cos(particle.theta+yaw_rate*delta_t)) + dist_y(gen);
          particle.theta = particle.theta + yaw_rate*delta_t + dist_yaw(gen);
      }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

  for (auto&& o: observations) {

      double min_dist = numeric_limits<double>::max();
      int map_id = -1;

      for (auto p: predicted) {
        // get distance between current/predicted landmarks
        double cur_dist = dist(o.x, o.y, p.x, p.y);

        // find nearest predicted landmark
        if (cur_dist < min_dist) {
            min_dist = cur_dist;
            map_id = p.id;
        }
      }
      // set the observation's id to the nearest predicted landmark's id
      o.id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

  std::random_device gen;

  for (auto&& particle: particles) {
     vector<LandmarkObs> predictions;

     for (auto landmark: map_landmarks.landmark_list) {
       // only consider landmarks within sensor range of the particle (rather than using the "dist" method considering a circular
       // region around the particle, this considers a rectangular region but is computationally faster)
       if (fabs(landmark.x_f - particle.x) <= sensor_range && fabs(landmark.y_f - particle.y) <= sensor_range) {
         predictions.push_back(LandmarkObs{ landmark.id_i, landmark.x_f, landmark.y_f });
       }
     }

     // create and populate a copy of the list of observations transformed from vehicle coordinates to map coordinates
     vector<LandmarkObs> transformed_observations;
     for (auto observation: observations) {
       double t_x = cos(particle.theta)*observation.x - sin(particle.theta)*observation.y + particle.x;
       double t_y = sin(particle.theta)*observation.x + cos(particle.theta)*observation.y + particle.y;
       transformed_observations.push_back(LandmarkObs{ observation.id, t_x, t_y });
     }

     dataAssociation(predictions, transformed_observations);

     particle.weight = 1.0;

     for (auto t_os: transformed_observations) {
       double pr_x, pr_y;
       int associated_prediction = t_os.id;

       // get the x,y coordinates of the prediction
       for (auto p: predictions) {
         if (p.id == associated_prediction) {
           pr_x = p.x;
           pr_y = p.y;
         }
       }

       // recalculate weight with multivariate Gaussian
       double s_x = std_landmark[0];
       double s_y = std_landmark[1];
       double obs_w = ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(pr_x-t_os.x,2)/(2*pow(s_x, 2)) + (pow(pr_y-t_os.y,2)/(2*pow(s_y, 2))) ) );

       particle.weight *= obs_w;
     }
  }

}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::random_device gen;
  vector<Particle> new_particles;
  // get all of the current weights
  vector<double> weights;
  for (auto p: particles) {
      weights.push_back(p.weight);
  }

  // generate random starting index for resampling wheel
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);

  double max_weight = *max_element(weights.begin(), weights.end());
  uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;

  // Resampling Wheel !
  for (int i = 0; i < num_particles; i++) {
      beta += unirealdist(gen) * 2.0;
      while (beta > weights[index]) {
          beta -= weights[index];
          index = (index + 1) % num_particles;
     }
     new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
