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
#include <random>
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// set the number of best_particles
	num_particles = 100;

	default_random_engine gen;
	// sample initial state of all particles
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);
	for(int i=0; i<num_particles; i++){
		Particle p;
		// initialize the single particle with random sampling
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
		// initialize the corresponding weight of the particle to 1
		weights.push_back(1);
	}

	// finish initilization
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// set noise generator
	default_random_engine gen;
	// cout << "velocity is: " << velocity << " yaw angle is: " << yaw_rate << endl;
	for(int i=0; i<num_particles; ++i){
		std::normal_distribution<double> dist_x(0, std_pos[0]);
		std::normal_distribution<double> dist_y(0, std_pos[1]);
		std::normal_distribution<double> dist_theta(0, std_pos[2]);
	
		if(fabs(yaw_rate) > 0.00001){
			particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta ));
			particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			particles[i].theta += yaw_rate*delta_t;
		}
		else{
			particles[i].x += velocity*cos(particles[i].theta );
			particles[i].y += velocity*sin(particles[i].theta );
		}

		// add noise here
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for(int i=0; i<observations.size(); i++){
		// use a very large value to represent minimum
		double closest_dist = numeric_limits<double>::max();
		int temp_id = -1;

		for(int j=0; j<predicted.size(); j++){
			double x_dist = predicted[j].x -  observations[i].x;
			double y_dist = predicted[j].y -  observations[i].y;
			double dist = x_dist * x_dist + y_dist * y_dist;

			if(dist < closest_dist){
				closest_dist = dist;
				temp_id = predicted[j].id;
			}
		}
		observations[i].id = temp_id;
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

	std::vector<LandmarkObs> ob_map;
	ob_map = observations;
	double gauss_norm = 1.0/(2* M_PI * std_landmark[0] * std_landmark[1]);
	double weights_sum = 0;
	weight_max = 0;
	std::vector<LandmarkObs> transformed_obs;
	for(int p_idx=0; p_idx < num_particles; p_idx++){

		double p_x = particles[p_idx].x;
		double p_y = particles[p_idx].y;
		double p_theta = particles[p_idx].theta;

		//first, transform the observation into map coordinates 
		transformed_obs.resize(ob_map.size());
		for(int ob_idx=0; ob_idx<observations.size(); ob_idx++){
				LandmarkObs ts_ob;
				ts_ob.x = p_x + cos(p_theta)*observations[ob_idx].x - sin(p_theta)*observations[ob_idx].y;
				ts_ob.y = p_y + sin(p_theta)*observations[ob_idx].x + cos(p_theta)*observations[ob_idx].y;
				transformed_obs[ob_idx] = ts_ob;
		}
		//second, do data association (find the landmark closest to each observation and within sensor range)
			// a. find all landmarks in the sensor range make it as predicted observation
		std::vector<LandmarkObs> predicted;
		for(int lm = 0; lm < map_landmarks.landmark_list.size(); lm++){
			float lm_x = map_landmarks.landmark_list[lm].x_f;
			float lm_y = map_landmarks.landmark_list[lm].y_f;
			int lm_id =  map_landmarks.landmark_list[lm].id_i; 
			// make sure each measurement of x and y is in sensor range
			if(fabs(lm_x - particles[p_idx].x) <= sensor_range && fabs(lm_y - particles[p_idx].y) <= sensor_range){
				predicted.push_back(LandmarkObs{lm_id, lm_x, lm_y});
			}
		} 

		if(predicted.size() == 0){
			cout << "No landmarks" << endl;
			return;
		}
			// b. do data association with the real transformed observation data
		dataAssociation(predicted, transformed_obs);
		//third, update the particle with by doing normal distribution
		double w = 1.0;
		for(const auto &ob : transformed_obs){
			float lm_x, lm_y;
			// we have to iterate all landmarks to find the right one
			for(const auto &lm : map_landmarks.landmark_list){
				if(lm.id_i == ob.id){
					lm_x = lm.x_f;
					lm_y = lm.y_f;
					break;
				}
			}
			double exponent = pow((ob.x - lm_x)/std_landmark[0], 2)/2.0\
			       				+ pow((ob.y - lm_y)/std_landmark[1], 2)/2.0;

			double temp_w = gauss_norm * exp(-exponent);
			if(temp_w == 0){
				temp_w = 0.0001;
			}
			w *= temp_w;
		}
		particles[p_idx].weight = w;
		weights_sum += w;
	}

	// weights normalization
	for(int i = 0; i < particles.size(); i++){
		particles[i].weight /= weights_sum;
		if(particles[i].weight > weight_max)
			weight_max = particles[i].weight;
		weights[i] = particles[i].weight;
	}
	// std::cout << "Current highest weight is: " << weight_max << std::endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	double beta = 0;
	std::vector<Particle> updated_p;
	updated_p.resize(num_particles);
	uniform_real_distribution<double> realDist(0.0, weight_max);
	uniform_int_distribution<int> intDist(0, num_particles - 1);
	int index = intDist(gen);

	for(int i = 0; i < num_particles; i++){
		beta += realDist(gen) * 2;
		while(beta > weights[index]){
			beta -= weights[index++];
			index %= num_particles;
		}
		updated_p[i] = particles[index];
	}
	particles = updated_p;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

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
